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
