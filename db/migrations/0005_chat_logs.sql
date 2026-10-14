-- What visitors ask the chat widget and what it answered, so the team can see
-- what people need and which questions the knowledge base can't answer yet.
--
-- Privacy: personal details in the question and answer are masked before a
-- row is written (app/core/redact.py: emails, phone numbers, passport and
-- application numbers, "my name is ..."); no IP address is stored; rows are
-- deleted after CHAT_LOG_RETENTION_DAYS (60 by default). See
-- app/db/chat_log_repo.py.
CREATE TABLE IF NOT EXISTS chat_logs (
    id               BIGSERIAL PRIMARY KEY,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Random per-browser-tab id from the widget: links a conversation's
    -- follow-ups together, identifies no one.
    session_id       TEXT,
    mode             TEXT NOT NULL,          -- 'text' or 'voice'
    is_followup      BOOLEAN NOT NULL DEFAULT false,
    question         TEXT NOT NULL,          -- masked
    search_query     TEXT,                   -- the rewritten search query, masked; NULL when not rewritten
    answer           TEXT,                   -- masked
    -- answered | declined | out_of_scope | smalltalk | guardrail | error
    outcome          TEXT NOT NULL,
    source_urls      TEXT[] NOT NULL DEFAULT '{}',
    retrieved_count  INTEGER,
    latency_ms       INTEGER
);

CREATE INDEX IF NOT EXISTS chat_logs_created_at_idx ON chat_logs (created_at);
CREATE INDEX IF NOT EXISTS chat_logs_outcome_idx ON chat_logs (outcome, created_at);
