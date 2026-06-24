from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from datetime import datetime, timezone

import cv2
import numpy as np
import sounddevice as sd
from scipy.spatial import distance as dist

import mediapipe as mp

from db import insert_batch


# config
BATCH_INTERVAL = 11   # seconds between DB flushes

SAMPLE_RATE = 16000
BLOCK_SIZE = 8000   # 500ms (SAMPLE_RATE/2) per audio callback

BLINK_THRESH = 0.21
BLINK_FRAMES = 2

SMILE_THRESH = 1.0
SMILE_FRAMES = 5
NON_SMILE_FRAMES = 5

FONT = cv2.FONT_HERSHEY_DUPLEX
STAT_TEXT_COLOR = (0, 200, 0)
METRICS_TEXT_COLOR = (0, 150, 255)
HINT_TEXT_COLOR = (200, 200, 200)

# landmarks, do not touch
L_EYE = [33, 160, 158, 133, 153, 144]
R_EYE = [362, 385, 387, 263, 373, 380]
L_IRIS, R_IRIS = 468, 473

DRAW_LANDMARKS = True
SHOW_METRICS = True
JPEG_QUALITY = 80


# audio state
class AudioState:
    def __init__(self) -> None:
        self.current_dBFS: float = 0.0
        self.batch_ms_sum: float = 0.0
        self.batch_ms_count: int = 0
        self.batch_max_dBFS: float = 0.0

    def reset_batch(self) -> None:
        self.batch_ms_sum = 0.0
        self.batch_ms_count = 0
        self.batch_max_dBFS = 0.0


def calculate_EAR(eye) -> float:
    v1 = dist.euclidean(eye[1], eye[5])
    v2 = dist.euclidean(eye[2], eye[4])
    h = dist.euclidean(eye[0], eye[3])
    return (v1 + v2) / (2.0 * h) if h else 0.0


# main class
class WebcamAnalyzer:
    def __init__(self, loop: asyncio.AbstractEventLoop, pool) -> None:
        self._loop = loop
        self._pool = pool
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # frame > mjpeg
        self._frame_lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._latest_frame_ts: float = 0.0

        # counters/states
        self._state_lock = threading.Lock()
        self._blink_count = 0
        self._smile_count = 0
        self._total_smile_time = 0.0
        self._is_smiling = False
        self._smile_start_time: float | None = None
        self._batch_start_time = 0.0
        self._frames_processed = 0

        # last error captured by analyzer thread (like camera busy)
        self._last_error: str | None = None

        self.overlay_enabled: bool = True
        self.display_enabled: bool = True

    # lifecycle
    def start(self) -> bool:
        if self.is_running():
            return False
        self._stop_event.clear()
        self._batch_start_time = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True, name="analyzer")
        self._thread.start()
        return True

    def stop(self) -> None:
        if not self.is_running():
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=6)
        self._thread = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # read API
    def get_jpeg_frame(self) -> bytes | None:
        with self._frame_lock:
            return self._latest_jpeg

    def get_live_state(self) -> dict:
        now = time.time()
        with self._state_lock:
            live_smile_time = self._total_smile_time
            if self._is_smiling and self._smile_start_time is not None:
                live_smile_time += now - self._smile_start_time
            return {
                "running": self.is_running(),
                "blink_count": self._blink_count,
                "smile_count": self._smile_count,
                "smile_time_s": round(live_smile_time, 2),
                "is_smiling": self._is_smiling,
                "frames_processed": self._frames_processed,
                "batch_remaining_s": max(0, BATCH_INTERVAL - (now - self._batch_start_time)),
                "last_error": self._last_error,
                "overlay_enabled": self.overlay_enabled,
                "display_enabled": self.display_enabled,
            }

    # analyzer thread
    def _run(self) -> None:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            self._last_error = "Could not open webcam (busy or not connected)."
            print(f"[analyzer] {self._last_error}")
            return
        self._last_error = None

        audio_state = AudioState()
        audio_lock = threading.Lock()
        audio_stream = None

        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"Audio status: {status}")
            ms = float(np.mean(indata ** 2))
            rms = np.sqrt(ms)
            dBFS = (20 * np.log10(rms) + 50) if rms > 0 else 0.0
            with audio_lock:
                audio_state.current_dBFS = dBFS
                audio_state.batch_ms_sum += ms
                audio_state.batch_ms_count += 1
                if dBFS > audio_state.batch_max_dBFS:
                    audio_state.batch_max_dBFS = dBFS

        try:
            audio_stream = sd.InputStream(
                callback=audio_callback,
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
            )
            audio_stream.start()
            mic_available = True
        except Exception as e:
            print(f"[analyzer] Could not start microphone: {e}")
            mic_available = False

        # per-thread working state
        blink_counter = 0
        smile_frame_counter = 0
        non_smile_counter = 0

        # local counter update
        with self._state_lock:
            local_blink_count = self._blink_count
            local_smile_count = self._smile_count
            local_total_smile_time = self._total_smile_time
            local_is_smiling = self._is_smiling
            local_smile_start_time = self._smile_start_time

        try:
            while not self._stop_event.is_set():
                ret, frame = cam.read()
                if not ret:
                    self._last_error = "Frame read failed."
                    break

                frame = cv2.resize(frame, (640, 480))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                smile_metric = 0.0
                avg_EAR = 0.0

                if results.multi_face_landmarks:
                    for fl in results.multi_face_landmarks:
                        h, w = frame.shape[:2]

                        def pt(i):
                            return (fl.landmark[i].x * w, fl.landmark[i].y * h)

                        left_eye = [pt(i) for i in L_EYE]
                        right_eye = [pt(i) for i in R_EYE]
                        avg_EAR = (calculate_EAR(left_eye) + calculate_EAR(right_eye)) / 2.0

                        if avg_EAR < BLINK_THRESH:
                            blink_counter += 1
                        else:
                            if blink_counter >= BLINK_FRAMES:
                                local_blink_count += 1
                            blink_counter = 0

                        l_corner = pt(61)
                        r_corner = pt(291)
                        upper_lip = pt(13)
                        lower_lip = pt(14)
                        l_iris = pt(L_IRIS)
                        r_iris = pt(R_IRIS)

                        pupil_dist = dist.euclidean(l_iris, r_iris)
                        mouth_width = dist.euclidean(l_corner, r_corner)
                        width_ratio = mouth_width / pupil_dist if pupil_dist else 0.0

                        center_y = (upper_lip[1] + lower_lip[1]) / 2.0
                        elev_l = center_y - l_corner[1]
                        elev_r = center_y - r_corner[1]
                        avg_elevation = (elev_l + elev_r) / 2.0
                        elevation_ratio = avg_elevation / pupil_dist if pupil_dist else 0.0

                        smile_metric = width_ratio + (elevation_ratio * 2.0)

                        if DRAW_LANDMARKS and self.overlay_enabled:
                            for idx in L_EYE + R_EYE + [61, 291, 13, 14]:
                                px = int(fl.landmark[idx].x * w)
                                py = int(fl.landmark[idx].y * h)
                                cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)
                            for idx in (L_IRIS, R_IRIS):
                                px = int(fl.landmark[idx].x * w)
                                py = int(fl.landmark[idx].y * h)
                                cv2.circle(frame, (px, py), 2, (255, 200, 0), -1)

                # smile state machine
                if smile_metric > SMILE_THRESH:
                    smile_frame_counter += 1
                    non_smile_counter = 0
                else:
                    smile_frame_counter = 0
                    non_smile_counter += 1

                if not local_is_smiling and smile_frame_counter >= SMILE_FRAMES:
                    local_is_smiling = True
                    local_smile_count += 1
                    local_smile_start_time = time.time()

                if local_is_smiling and non_smile_counter >= NON_SMILE_FRAMES:
                    local_is_smiling = False
                    if local_smile_start_time is not None:
                        local_total_smile_time += time.time() - local_smile_start_time
                        local_smile_start_time = None

                live_smile_time = local_total_smile_time
                if local_is_smiling and local_smile_start_time is not None:
                    live_smile_time += time.time() - local_smile_start_time

                # local counters shared to HTTP
                with self._state_lock:
                    self._blink_count = local_blink_count
                    self._smile_count = local_smile_count
                    self._total_smile_time = local_total_smile_time
                    self._is_smiling = local_is_smiling
                    self._smile_start_time = local_smile_start_time
                    self._frames_processed += 1

                # batch flush
                if time.time() - self._batch_start_time >= BATCH_INTERVAL:
                    self._flush_batch(
                        audio_state, audio_lock, mic_available,
                        local_blink_count, local_smile_count, local_total_smile_time,
                        local_is_smiling, local_smile_start_time,
                    )
                    # reset counters belonging to flushed batch
                    local_blink_count = 0
                    local_smile_count = 0
                    local_total_smile_time = 0.0
                    # on ongoing smile restart smile timer for new batch
                    if local_is_smiling:
                        local_smile_start_time = time.time()
                    with self._state_lock:
                        self._blink_count = local_blink_count
                        self._smile_count = local_smile_count
                        self._total_smile_time = local_total_smile_time
                        self._smile_start_time = local_smile_start_time
                        self._batch_start_time = time.time()

                # HUD overlay
                if self.overlay_enabled:
                    cv2.putText(frame, f'Blink count: {local_blink_count}', (30, 30), FONT, 1, STAT_TEXT_COLOR, 2)
                    cv2.putText(frame, f'Smile count: {local_smile_count}', (30, 60), FONT, 1, STAT_TEXT_COLOR, 2)
                    cv2.putText(frame, f'Smile time: {live_smile_time:.2f} s', (30, 90), FONT, 1, STAT_TEXT_COLOR, 2)

                    if SHOW_METRICS and results.multi_face_landmarks:
                        cv2.putText(frame, f'EAR: {avg_EAR:.2f}', (30, 120), FONT, 0.75, METRICS_TEXT_COLOR, 2)
                        cv2.putText(frame, f'Smile metric: {smile_metric:.2f}', (30, 145), FONT, 0.75, METRICS_TEXT_COLOR, 2)

                    if mic_available:
                        with audio_lock:
                            ms_sum = audio_state.batch_ms_sum
                            ms_cnt = audio_state.batch_ms_count
                            disp_max_dB = audio_state.batch_max_dBFS
                        if ms_cnt > 0 and ms_sum > 0:
                            avg_rms = np.sqrt(ms_sum / ms_cnt)
                            disp_avg_dB = (20 * np.log10(avg_rms) + 50) if avg_rms > 0 else 0.0
                        else:
                            disp_avg_dB = 0.0
                        cv2.putText(frame, f'Max Loudness: {disp_max_dB:.1f} dBFS', (30, 200), FONT, 1, STAT_TEXT_COLOR, 2)
                        cv2.putText(frame, f'Avg Loudness ({BATCH_INTERVAL}s): {disp_avg_dB:.1f} dBFS', (30, 230), FONT, 1, STAT_TEXT_COLOR, 2)

                    remaining = max(0, BATCH_INTERVAL - (time.time() - self._batch_start_time))
                    cv2.putText(frame, f'Next batch: {remaining:.0f}s', (30, 260), FONT, 0.7, HINT_TEXT_COLOR, 1)

                # Encode + publish for MJPEG
                if self.display_enabled:
                    ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                    if ok:
                        with self._frame_lock:
                            self._latest_jpeg = buf.tobytes()
                            self._latest_frame_ts = time.time()
                else:
                    with self._frame_lock:
                        if self._latest_jpeg is not None:
                            self._latest_jpeg = None

        except Exception as e:
            self._last_error = f"Analyzer crashed: {e}"
            print(f"[analyzer] crash: {e}")
        finally:
            # final flush of the partial batch (on end)
            try:
                self._flush_batch(
                    audio_state, audio_lock, mic_available,
                    local_blink_count, local_smile_count, local_total_smile_time,
                    local_is_smiling, local_smile_start_time,
                )
            except Exception as e:
                print(f"[analyzer] final flush failed: {e}")

            if audio_stream is not None:
                try:
                    audio_stream.stop()
                    audio_stream.close()
                except Exception:
                    pass
            cam.release()
            face_mesh.close()
            with self._frame_lock:
                self._latest_jpeg = None
            print("[analyzer] stopped.")

    # batch flush - snapshot counters, schedule DB insert on FastAPI loop
    def _flush_batch(
        self,
        audio_state: AudioState,
        audio_lock: threading.Lock,
        mic_available: bool,
        blinks: int,
        smiles: int,
        smile_time: float,
        is_smiling: bool,
        smile_start_time: float | None,
    ) -> None:
        now = time.time()
        duration = now - self._batch_start_time

        # snapshot smile time into current batch
        batch_smile_time = smile_time
        if is_smiling and smile_start_time is not None:
            batch_smile_time += now - smile_start_time

        if mic_available:
            with audio_lock:
                ms_sum = audio_state.batch_ms_sum
                ms_cnt = audio_state.batch_ms_count
                mx_dB = audio_state.batch_max_dBFS
                audio_state.reset_batch()
            if ms_cnt > 0 and ms_sum > 0:
                avg_rms = np.sqrt(ms_sum / ms_cnt)
                avg_dB = (20 * np.log10(avg_rms) + 50) if avg_rms > 0 else 0.0
            elif ms_cnt > 0:
                avg_dB = 0.0
            else:
                avg_dB = None
                mx_dB = None
        else:
            avg_dB = None
            mx_dB = None

        print(
            f"[Batch] {duration:.1f}s | blinks={blinks} | smiles={smiles} | "
            f"smile_t={batch_smile_time:.2f}s | avg_dB={avg_dB} | max_dB={mx_dB}"
        )

        coro = insert_batch(
            pool=self._pool,
            batch_start=datetime.fromtimestamp(self._batch_start_time, tz=timezone.utc),
            batch_end=datetime.fromtimestamp(now, tz=timezone.utc),
            duration=duration,
            blinks=blinks,
            smiles=smiles,
            smile_time=batch_smile_time,
            avg_loud=avg_dB,
            max_loud=mx_dB,
        )
        try:
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            future.result(timeout=10)
        except Exception as e:
            print(f"[analyzer] DB insert error: {e}")
