import cv2
import mediapipe as mp
import time
import threading
from scipy.spatial import distance as dist
import sounddevice as sd
import numpy as np
from collections import deque
import math
import os
import asyncio
import asyncpg
from datetime import datetime, timezone


# Config
BATCH_INTERVAL = 11  # DB write interval, sec
DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://usr:secret@localhost:1984/webcam_stats",
)

cam = cv2.VideoCapture(0)
FONT = cv2.FONT_HERSHEY_DUPLEX
STAT_TEXT_COLOR = (0, 200, 0)
METRICS_TEXT_COLOR = (0, 150, 255)

# DEBUG
DRAW_LANDMARKS = True
SHOW_METRICS = True

# audio
SAMPLE_RATE = 16000
BLOCK_SIZE = 8000  # 500ms of audio/callback

class AudioState:
    def __init__(self):
        self.current_dBFS = 0.0  # instant value
        self.batch_ms_sum = 0.0  # MS sum for avg
        self.batch_ms_count = 0  # callbacks number
        self.batch_max_dBFS = 0.0 # peak dBFS in current batch

    def reset_batch(self):
        self.batch_ms_sum = 0.0
        self.batch_ms_count = 0
        self.batch_max_dBFS = 0.0

audio_state = AudioState()
audio_lock = threading.Lock()


# microphone input thread
def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio status: {status}")

    ms = np.mean(indata ** 2)
    rms = np.sqrt(ms)
    dBFS = (20 * np.log10(rms) + 50) if rms > 0 else 0

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
    print(f"Could not start microphone: {e}")
    mic_available = False


# face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


# EAR for blinking
def calculate_EAR(eye):
    v1 = dist.euclidean(eye[1], eye[5])
    v2 = dist.euclidean(eye[2], eye[4])
    h = dist.euclidean(eye[0], eye[3])
    return (v1 + v2) / (2.0 * h) if h else 0.0

# eye landmarks
L_EYE = [33, 160, 158, 133, 153, 144]
R_EYE = [362, 385, 387, 263, 373, 380]
L_IRIS, R_IRIS = 468, 473

# blink state
BLINK_THRESH = 0.21
BLINK_FRAMES = 2
blink_counter = 0
blink_count = 0

# smile state
SMILE_THRESH = 1
SMILE_FRAMES = 5
NON_SMILE_FRAMES = 5
smile_frame_counter = 0
non_smile_counter = 0
is_smiling = False
smile_start_time = None
smile_count = 0
total_smile_time = 0.0


# DB writing using asyncpg and separate event loop thread. in some way my practice of asyncpg instead of psycopg.
db_loop = asyncio.new_event_loop()
db_thread = threading.Thread(target=lambda: db_loop.run_forever(), daemon=True)
db_thread.start()

async_pool = None
db_available = False


async def _init_pool():
    global async_pool, db_available
    async_pool = await asyncpg.create_pool(DB_URL, min_size=1, max_size=3)
    async with async_pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS stats_batch (
                id              SERIAL PRIMARY KEY,
                batch_start     TIMESTAMPTZ NOT NULL,
                batch_end       TIMESTAMPTZ NOT NULL,
                duration_s      DOUBLE PRECISION NOT NULL,
                blink_count     INTEGER NOT NULL DEFAULT 0,
                smile_count     INTEGER NOT NULL DEFAULT 0,
                smile_time_s    DOUBLE PRECISION NOT NULL DEFAULT 0,
                avg_loudness    DOUBLE PRECISION,
                max_loudness    DOUBLE PRECISION
            );
            
            CREATE INDEX IF NOT EXISTS idx_stats_batch_start ON stats_batch (batch_start);
        """)
    db_available = True
    print("Database ready.")


def _run_async(coro):
    """Submit a coroutine to the background loop and block until it returns."""
    future = asyncio.run_coroutine_threadsafe(coro, db_loop)
    return future.result(timeout=10)


# Initialise pool (retry on each batch flush if it fails)
try:
    _run_async(_init_pool())
except Exception as e:
    print(f"DB pool init failed (will retry on first flush): {e}")


async def _ensure_pool():
    """Reconnect if the pool was never created or is closed."""
    global async_pool, db_available
    if async_pool is not None and not async_pool._closed:
        return
    await _init_pool()


async def _insert_batch(batch_start, batch_end, duration, blinks, smiles, smile_time, avg_loud, max_loud):
    await _ensure_pool()
    async with async_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO stats_batch
                (batch_start, batch_end, duration_s,
                 blink_count, smile_count, smile_time_s,
                 avg_loudness, max_loudness)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            batch_start, batch_end, duration,
            blinks, smiles, smile_time, avg_loud, max_loud,
        )


def insert_batch_sync(batch_start, batch_end, duration, blinks, smiles, smile_time, avg_loud, max_loud):
    if not db_available and async_pool is None:
        try:
            _run_async(_ensure_pool())
        except Exception as e:
            print(f"DB still unreachable: {e}")
            return
    try:
        _run_async(_insert_batch(
            batch_start, batch_end, duration, blinks,
            smiles, smile_time, avg_loud, max_loud,
        ))
    except Exception as e:
        print(f"DB insert error: {e}")


# working with batches
batch_start_time = time.time()


def flush_batch():
    """Snapshot counters > DB > reset for next batch. """
    global blink_count, smile_count, total_smile_time
    global smile_start_time, batch_start_time

    now = time.time()
    duration = now - batch_start_time

    # blink
    batch_blinks = blink_count
    blink_count = 0

    # smile
    batch_smile_time = total_smile_time
    if is_smiling and smile_start_time is not None:
        batch_smile_time += now - smile_start_time
        smile_start_time = now

    batch_smiles = smile_count
    smile_count = 0
    total_smile_time = 0.0

    # audio
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
            avg_dB = 0.0  # silence
        else:
            avg_dB = None  # no callbacks
            mx_dB = None
    else:
        avg_dB = None
        mx_dB = None

    # writing
    insert_batch_sync(
        batch_start=datetime.fromtimestamp(batch_start_time, tz=timezone.utc),
        batch_end=datetime.fromtimestamp(now, tz=timezone.utc),
        duration=duration,
        blinks=batch_blinks,
        smiles=batch_smiles,
        smile_time=batch_smile_time,
        avg_loud=avg_dB,
        max_loud=mx_dB,
    )

    print(
        f"[Batch] {duration:.1f}s | blinks={batch_blinks} | "
        f"smiles={batch_smiles} | smile_t={batch_smile_time:.2f}s | "
        f"avg_dB={avg_dB} | max_dB={mx_dB}"
    )

    batch_start_time = now

while True:
    ret, frame = cam.read()
    if not ret:
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

            # blink
            left_eye = [pt(i) for i in L_EYE]
            right_eye = [pt(i) for i in R_EYE]
            avg_EAR = (calculate_EAR(left_eye) + calculate_EAR(right_eye)) / 2.0

            if avg_EAR < BLINK_THRESH:
                blink_counter += 1
            else:
                if blink_counter >= BLINK_FRAMES:
                    blink_count += 1
                blink_counter = 0

            # smile
            l_corner = pt(61)
            r_corner = pt(291)
            upper_lip = pt(13)
            lower_lip = pt(14)
            l_iris = pt(L_IRIS)
            r_iris = pt(R_IRIS)

            pupil_dist = dist.euclidean(l_iris, r_iris)
            mouth_width = dist.euclidean(l_corner, r_corner)

            # width
            width_ratio = mouth_width / pupil_dist if pupil_dist else 0.0

            # corner elevation
            center_y = (upper_lip[1] + lower_lip[1]) / 2.0
            elev_l = center_y - l_corner[1]
            elev_r = center_y - r_corner[1]
            avg_elevation = (elev_l + elev_r) / 2.0
            elevation_ratio = avg_elevation / pupil_dist if pupil_dist else 0.0

            # "magic" combined metric
            smile_metric = width_ratio + (elevation_ratio * 2.0)

            # DEBUG
            if DRAW_LANDMARKS:
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

    if not is_smiling and smile_frame_counter >= SMILE_FRAMES:
        is_smiling = True
        smile_count += 1
        smile_start_time = time.time()

    if is_smiling and non_smile_counter >= NON_SMILE_FRAMES:
        is_smiling = False
        if smile_start_time is not None:
            total_smile_time += time.time() - smile_start_time
            smile_start_time = None

    # smile time for the current batch
    live_smile_time = total_smile_time
    if is_smiling and smile_start_time is not None:
        live_smile_time += time.time() - smile_start_time

    # flushing batch
    if time.time() - batch_start_time >= BATCH_INTERVAL:
        flush_batch()

    # audio stats display
    disp_avg_dB = 0.0
    disp_max_dB = 0.0
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

    # text
    cv2.putText(frame, f'Blink count: {blink_count}', (30, 30), FONT, 1, STAT_TEXT_COLOR, 2)
    cv2.putText(frame, f'Smile count: {smile_count}', (30, 60), FONT, 1, STAT_TEXT_COLOR, 2)
    cv2.putText(frame, f'Smile time: {live_smile_time:.2f} s', (30, 90),  FONT, 1, STAT_TEXT_COLOR, 2)

    if mic_available:
        cv2.putText(frame, f'Max Loudness: {disp_max_dB:.1f} dBFS', (30, 200), FONT, 1, STAT_TEXT_COLOR, 2)
        cv2.putText(frame, f'Avg Loudness ({BATCH_INTERVAL}s): {disp_avg_dB:.1f} dBFS', (30, 230), FONT, 1, STAT_TEXT_COLOR, 2)

    if SHOW_METRICS and results.multi_face_landmarks:
        cv2.putText(frame, f'EAR: {avg_EAR:.2f}', (30, 120), FONT, 0.75, METRICS_TEXT_COLOR, 2)
        cv2.putText(frame, f'Smile custom metric: {smile_metric:.2f}', (30, 140), FONT, 0.75, METRICS_TEXT_COLOR, 2)

    remaining = max(0, BATCH_INTERVAL - (time.time() - batch_start_time))
    cv2.putText(frame, f'Next batch: {remaining:.0f}s', (30, 260), FONT, 0.7, (200, 200, 200), 1)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# clean up
flush_batch()  # write partial batch

cam.release()

if mic_available:
    audio_stream.stop()
    audio_stream.close()


async def _close_pool():
    if async_pool and not async_pool._closed:
        await async_pool.close()


if db_available:
    try:
        _run_async(_close_pool())
    except Exception:
        pass


db_loop.call_soon_threadsafe(db_loop.stop)
cv2.destroyAllWindows()
