CREATE TABLE IF NOT EXISTS watch_log_requests(
    id TEXT PRIMARY KEY,case_id TEXT NOT NULL UNIQUE,user_id INTEGER NOT NULL,
    source_chat INTEGER NOT NULL,title TEXT NOT NULL,submitted_by INTEGER NOT NULL,
    submitted_message INTEGER NOT NULL,created_at INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',approved_by INTEGER,
    review_state TEXT NOT NULL DEFAULT 'queued',review_chat INTEGER,review_message INTEGER,
    review_error TEXT,updated_at INTEGER NOT NULL);
