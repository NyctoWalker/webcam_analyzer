from __future__ import annotations

import os
import re
import asyncpg
from datetime import datetime, timedelta, timezone

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://usr:secret@localhost:1984/webcam_stats",
)

SCHEMA_SQL = """
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
"""


# pool
async def init_pool() -> asyncpg.Pool:
    pool = await asyncpg.create_pool(DB_URL, min_size=1, max_size=5)
    async with pool.acquire() as conn:
        await conn.execute(SCHEMA_SQL)
    print("[db] pool ready")
    return pool


async def close_pool(pool: asyncpg.Pool) -> None:
    if pool and not pool._closed:
        await pool.close()


# writes
async def insert_batch(
    *,
    pool: asyncpg.Pool,
    batch_start: datetime,
    batch_end: datetime,
    duration: float,
    blinks: int,
    smiles: int,
    smile_time: float,
    avg_loud: float | None,
    max_loud: float | None,
) -> None:
    async with pool.acquire() as conn:
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


# reads (from fastapi)
async def get_timeseries(
    pool: asyncpg.Pool,
    *,
    start: datetime,
    end: datetime,
    bucket,
) -> list[dict]:
    """agg rows into time buckets with PgSQL date_bin"""
    if isinstance(bucket, str):
        bucket = bucket_to_timedelta(bucket)

    sql = """
        SELECT
            date_bin($1, batch_start, TIMESTAMPTZ '2000-01-01') AS bucket,
            COALESCE(SUM(blink_count), 0)   AS blinks,
            COALESCE(SUM(smile_count), 0)   AS smiles,
            COALESCE(SUM(smile_time_s), 0)  AS smile_time_s,
            COALESCE(AVG(avg_loudness), 0)  AS avg_loudness,
            COALESCE(MAX(max_loudness), 0)  AS max_loudness,
            COALESCE(SUM(duration_s), 0)    AS duration_s,
            COUNT(*)                        AS batch_count
        FROM stats_batch
        WHERE batch_start >= $2 AND batch_start < $3
        GROUP BY bucket
        ORDER BY bucket ASC
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, bucket, start, end)
        return [
            {
                "bucket": r["bucket"].isoformat() if r["bucket"] else None,
                "blinks": r["blinks"],
                "smiles": r["smiles"],
                "smile_time_s": float(r["smile_time_s"]),
                "avg_loudness": float(r["avg_loudness"]) if r["avg_loudness"] is not None else None,
                "max_loudness": float(r["max_loudness"]) if r["max_loudness"] is not None else None,
                "duration_s": float(r["duration_s"]),
                "batch_count": r["batch_count"],
            }
            for r in rows
        ]


async def get_summary(
    pool: asyncpg.Pool,
    *,
    start: datetime,
    end: datetime,
) -> dict:
    """totals for selected period (no buckets)"""
    sql = """
        SELECT
            COALESCE(SUM(blink_count), 0)   AS blinks,
            COALESCE(SUM(smile_count), 0)   AS smiles,
            COALESCE(SUM(smile_time_s), 0)  AS smile_time_s,
            COALESCE(AVG(avg_loudness), 0)  AS avg_loudness,
            COALESCE(MAX(max_loudness), 0)  AS max_loudness,
            COALESCE(SUM(duration_s), 0)    AS duration_s,
            COUNT(*)                        AS batch_count,
            MIN(batch_start)                AS first_batch,
            MAX(batch_end)                  AS last_batch
        FROM stats_batch
        WHERE batch_start >= $1 AND batch_start < $2
    """

    async with pool.acquire() as conn:
        row = await conn.fetchrow(sql, start, end)
        if row is None:
            return {}
        return {
            "blinks": row["blinks"],
            "smiles": row["smiles"],
            "smile_time_s": float(row["smile_time_s"]),
            "avg_loudness": float(row["avg_loudness"]) if row["avg_loudness"] is not None else None,
            "max_loudness": float(row["max_loudness"]) if row["max_loudness"] is not None else None,
            "duration_s": float(row["duration_s"]),
            "batch_count": row["batch_count"],
            "first_batch": row["first_batch"].isoformat() if row["first_batch"] else None,
            "last_batch": row["last_batch"].isoformat() if row["last_batch"] else None,
        }


# bucket selection
def auto_bucket(seconds: float) -> str:
    """pick PgSQL interval for time span (seconds)"""
    if seconds <= 2 * 3600:  # <= 2h > 1 min
        return "1 minute"
    if seconds <= 12 * 3600:  # <= 12h > 5 min
        return "5 minutes"
    if seconds <= 2 * 86400:  # <= 2d > 30 min
        return "30 minutes"
    if seconds <= 8 * 86400:  # <= 8d > 6 hours
        return "6 hours"
    if seconds <= 60 * 86400:  # <= 60d > 1 day
        return "1 day"
    return "1 week"

_BUCKET_RE = re.compile(
    r'^\s*(\d+)\s*(minute|minutes|hour|hours|day|days|week|weeks)\s*$',
    re.IGNORECASE,
)


def bucket_to_timedelta(bucket: str) -> timedelta:
    """Convert a human-readable interval string like '30 minutes' or '1 hour' into a datetime.timedelta."""
    m = _BUCKET_RE.match(bucket)
    if not m:
        raise ValueError(
            f"Invalid bucket interval: {bucket!r}. "
            f"Expected e.g. '1 minute', '30 minutes', '6 hours', '1 day'."
        )
    n = int(m.group(1))
    unit = m.group(2).lower().rstrip('s')  # 'minutes' -> 'minute'
    # timedelta accepts weeks/days/hours/minutes/seconds as kwargs
    return timedelta(**{f"{unit}s": n})
