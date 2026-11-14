CREATE TABLE IF NOT EXISTS chat_runs (
    id UUID PRIMARY KEY,
    position BIGSERIAL UNIQUE NOT NULL,
    user_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    message_id UUID NOT NULL,
    request_id UUID NOT NULL,
    payload JSONB NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled')),
    attempts INTEGER NOT NULL DEFAULT 0,
    cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    UNIQUE (user_id, thread_id, request_id)
);
CREATE INDEX IF NOT EXISTS chat_runs_pending_order ON chat_runs(position)
    WHERE status IN ('queued', 'running');
CREATE INDEX IF NOT EXISTS chat_runs_pending_thread ON chat_runs(user_id, thread_id, position)
    WHERE status IN ('queued', 'running');
CREATE TABLE IF NOT EXISTS chat_run_events (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES chat_runs(id) ON DELETE CASCADE,
    payload JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS chat_run_events_cursor ON chat_run_events(run_id, id);
CREATE INDEX IF NOT EXISTS chat_runs_completed_sessions ON chat_runs(user_id, thread_id, position DESC)
    WHERE status = 'completed';
