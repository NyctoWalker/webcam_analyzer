"""
FastAPI app for webcam analyzer.

Endpoints list
GET  /              Single-page UI (live + analytics tabs)
GET  /api/status    Live counters + analyzer running flag
POST /api/start     Start analyzer thread
POST /api/stop      Stop analyzer thread (flushes partial batch)
GET  /api/stats     Aggregated stats - ?range=24h&bucket=auto or ?start=...&end=...&bucket=1 hour
GET  /video/feed    MJPEG stream of latest annotated frame

Run: python main.py
or: uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import db
from analyzer import WebcamAnalyzer


BASE_DIR = Path(__file__).parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# app state. created in lifespan so analyzer+pool share running loop
class AppState:
    pool = None
    analyzer: WebcamAnalyzer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    try:
        AppState.pool = await db.init_pool()
    except Exception as e:
        print(f"[startup] DB init failed - analyzer will keep retrying on flush: {e}")
        AppState.pool = None
    AppState.analyzer = WebcamAnalyzer(loop=loop, pool=AppState.pool)
    yield

    # shutdown
    if AppState.analyzer:
        AppState.analyzer.stop()
    if AppState.pool:
        await db.close_pool(AppState.pool)


app = FastAPI(title="Webcam Analyzer", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# pages
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return TEMPLATES.TemplateResponse("index.html", {"request": request})


# control+status API
@app.post("/api/start")
async def api_start():
    if AppState.analyzer is None:
        raise HTTPException(503, "Analyzer not initialized")
    if not AppState.pool:
        # try to (re)create pool lazily so UI works even if DB was down at boot
        try:
            AppState.pool = await db.init_pool()
            AppState.analyzer._pool = AppState.pool
        except Exception as e:
            raise HTTPException(503, f"Database unavailable: {e}")
    started = AppState.analyzer.start()
    return {"started": started, "running": AppState.analyzer.is_running()}


@app.post("/api/stop")
async def api_stop():
    if AppState.analyzer:
        AppState.analyzer.stop()
        return {"stopped": True, "running": AppState.analyzer.is_running()}
    return {"stopped": False, "running": False}


@app.get("/api/status")
async def api_status():
    if AppState.analyzer is None:
        return {"running": False, "db_ready": bool(AppState.pool)}
    state = AppState.analyzer.get_live_state()
    state["db_ready"] = bool(AppState.pool)
    return state


# stats / analytics API
RANGE_PRESETS = {
    "1h":  3600,
    "24h": 24 * 3600,
    "7d":  7 * 86400,
    "30d": 30 * 86400,
}

BUCKET_PRESETS = {
    "1m": "1 minute",
    "5m": "5 minutes",
    "30m": "30 minutes",
    "1h": "1 hour",
    "6h": "6 hours",
    "1d": "1 day",
}


@app.get("/api/stats")
async def api_stats(
    range: str | None = Query(None, description="Preset: 1h | 24h | 7d | 30d"),
    start: datetime | None = Query(None, description="ISO start (overrides range)"),
    end: datetime | None = Query(None, description="ISO end (overrides range)"),
    bucket: str = Query("auto", description="auto | 1m | 5m | 30m | 1h | 6h | 1d"),
):
    # resolve bucket
    if bucket == "auto":
        now_for_bucket = datetime.now(tz=timezone.utc)
        if start and end:
            s_tmp = start.astimezone(timezone.utc) if start.tzinfo else start.replace(tzinfo=timezone.utc)
            e_tmp = end.astimezone(timezone.utc) if end.tzinfo else end.replace(tzinfo=timezone.utc)
        elif range and range in RANGE_PRESETS:
            e_tmp = now_for_bucket
            s_tmp = now_for_bucket - timedelta(seconds=RANGE_PRESETS[range])
        else:
            e_tmp = now_for_bucket
            s_tmp = now_for_bucket - timedelta(seconds=RANGE_PRESETS["24h"])
        bucket_interval = db.auto_bucket((e_tmp - s_tmp).total_seconds())
    elif bucket in BUCKET_PRESETS:
        bucket_interval = BUCKET_PRESETS[bucket]
    else:
        raise HTTPException(
            400,
            f"Invalid bucket={bucket!r}. Use 'auto' or one of: "
            f"{', '.join(sorted(BUCKET_PRESETS.keys()))}",
        )

    # сonvert to timedelta
    try:
        bucket_td = db.bucket_to_timedelta(bucket_interval)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if AppState.pool is None:
        raise HTTPException(503, "Database unavailable")

    # resolve window
    now = datetime.now(tz=timezone.utc)
    if start and end:
        start = start.astimezone(timezone.utc) if start.tzinfo else start.replace(tzinfo=timezone.utc)
        end = end.astimezone(timezone.utc) if end.tzinfo else end.replace(tzinfo=timezone.utc)
    elif range and range in RANGE_PRESETS:
        end = now
        start = now - timedelta(seconds=RANGE_PRESETS[range])
    else:
        # default to last 24h
        end = now
        start = now - timedelta(seconds=RANGE_PRESETS["24h"])

    span_seconds = (end - start).total_seconds()

    ts = await db.get_timeseries(
        AppState.pool, start=start, end=end, bucket=bucket_td
    )
    summary = await db.get_summary(AppState.pool, start=start, end=end)

    return JSONResponse({
        "start": start.isoformat(),
        "end": end.isoformat(),
        "bucket": bucket_interval,
        "bucket_label": bucket if bucket != "auto" else f"auto ({bucket_interval})",
        "summary": summary,
        "timeseries": ts,
    })


# Settings (HUD overlay + video display toggles)
from pydantic import BaseModel


class SettingsUpdate(BaseModel):
    overlay: bool | None = None
    display: bool | None = None


@app.get("/api/settings")
async def api_get_settings():
    if AppState.analyzer is None:
        return {"overlay": True, "display": True}
    return {
        "overlay": AppState.analyzer.overlay_enabled,
        "display": AppState.analyzer.display_enabled,
    }


@app.post("/api/settings")
async def api_set_settings(s: SettingsUpdate):
    if AppState.analyzer is None:
        raise HTTPException(503, "Analyzer not initialized")
    if s.overlay is not None:
        AppState.analyzer.overlay_enabled = s.overlay
    if s.display is not None:
        AppState.analyzer.display_enabled = s.display
    return {
        "overlay": AppState.analyzer.overlay_enabled,
        "display": AppState.analyzer.display_enabled,
    }


# MJPEG video feed
@app.get("/video/feed")
async def video_feed():
    boundary = "frame"

    async def generate():
        # send placeholder frame when analyzer is off
        last_placeholder_at = 0.0
        last_placeholder_text = ""
        while True:
            if AppState.analyzer is None:
                await asyncio.sleep(0.5)
                continue

            # decide what placeholder text to show when there's no JPEG.
            if not AppState.analyzer.display_enabled:
                placeholder_text = "Video display is off"
                placeholder_hint = "Stats still collecting - enable Display to see feed"
            elif not AppState.analyzer.is_running():
                placeholder_text = "Analyzer is stopped"
                placeholder_hint = "Click Start to begin"
            else:
                placeholder_text = ""
                placeholder_hint = ""

            jpeg = AppState.analyzer.get_jpeg_frame()
            if jpeg is None:
                # throttle placeholder frames to ~1 fps and only re-encode if text changed.
                now = time.time()
                if (
                    placeholder_text
                    and (now - last_placeholder_at >= 1.0
                    or last_placeholder_text != placeholder_text)
                ):
                    placeholder = _placeholder_frame(placeholder_text, placeholder_hint)
                    if placeholder is not None:
                        yield (
                            b"--" + boundary.encode()
                            + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(placeholder)).encode()
                            + b"\r\n\r\n" + placeholder + b"\r\n"
                        )
                    last_placeholder_at = now
                    last_placeholder_text = placeholder_text
                await asyncio.sleep(0.5)
                continue

            yield (
                b"--" + boundary.encode()
                + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(jpeg)).encode()
                + b"\r\n\r\n" + jpeg + b"\r\n"
            )
            # ~20 fps cap to keep CPU / bandwidth reasonable
            await asyncio.sleep(0.05)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=" + boundary,
    )


def _placeholder_frame(text: str, hint: str = "") -> bytes | None:
    """generate small JPEG with given text"""
    try:
        import cv2
        import numpy as np
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, text, (80, 230), cv2.FONT_HERSHEY_DUPLEX, 1.2, (210, 210, 210), 2)
        if hint:
            cv2.putText(img, hint, (90, 280), cv2.FONT_HERSHEY_DUPLEX, 0.85, (130, 130, 130), 1)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        return buf.tobytes() if ok else None
    except Exception:
        return None


# CLI launch
if __name__ == "__main__":
    import uvicorn
    print("http://localhost:8000")
    uvicorn.run("main:app", port=8000, reload=False)
