-- 0001_init_schema.sql
-- Phase 1: core schema for source registry, version history, chunk metadata
-- mirror, and audit log. This is the Postgres source of truth for
-- "what changed and when" — Qdrant mirrors the chunk payload for retrieval.

CREATE EXTENSION IF NOT EXISTS pgcrypto; -- gives us gen_random_uuid()

-- ---------------------------------------------------------------------------
-- Enums
-- ---------------------------------------------------------------------------

CREATE TYPE source_type AS ENUM ('html', 'pdf');

-- Service categories called out in the project brief. Extend via
-- ALTER TYPE ... ADD VALUE if new categories show up later.
CREATE TYPE service_category AS ENUM (
    'passport',
    'oci',
    'visa',
    'surrender_renunciation',
    'birth_death_certificates',
    'police_clearance',
    'status_tracking',
    'community',
    'misc_services'
);

CREATE TYPE review_status AS ENUM (
    'pending_review',
    'approved',
    'rejected',
    'superseded'
);

CREATE TYPE fetch_result AS ENUM ('success', 'failure');

-- ---------------------------------------------------------------------------
-- sources: the registry of the 47 (and growing) source URLs to track.
-- ---------------------------------------------------------------------------

CREATE TABLE sources (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url                  TEXT NOT NULL UNIQUE,
    source_type          source_type NOT NULL,
    service_category     service_category NOT NULL,
    -- true for gov.in / mea.gov.in official sources, false for VFS/provider pages
    canonical            BOOLEAN NOT NULL DEFAULT false,
    jurisdiction         TEXT[] NOT NULL DEFAULT ARRAY['all'],
    -- e.g. {adult, minor} or {with_passport, without_passport}; empty when not applicable
    applicant_variant    TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    active               BOOLEAN NOT NULL DEFAULT true,
    notes                TEXT,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_sources_service_category ON sources (service_category);
CREATE INDEX idx_sources_active ON sources (active);

-- ---------------------------------------------------------------------------
-- source_versions: one row per successfully fetched, content-changed version
-- of a source. Every fetch *attempt* (success or failure) is logged in
-- audit_log instead — this table only records versions that actually exist.
-- ---------------------------------------------------------------------------

CREATE TABLE source_versions (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id             UUID NOT NULL REFERENCES sources (id) ON DELETE CASCADE,
    version               INTEGER NOT NULL,
    content_hash          TEXT NOT NULL,          -- hash of the extracted region, not raw page
    retrieval_date        TIMESTAMPTZ NOT NULL,
    source_last_modified  TIMESTAMPTZ,             -- from HTTP headers / PDF metadata, if available
    raw_content_path      TEXT NOT NULL,           -- where the raw fetched artifact is stored
    extracted_text_path   TEXT,                    -- populated by Phase 3 extraction step
    fetch_status          fetch_result NOT NULL DEFAULT 'success',
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source_id, version)
);

CREATE INDEX idx_source_versions_source_id ON source_versions (source_id);

-- ---------------------------------------------------------------------------
-- chunks: Postgres mirror of what gets embedded + upserted into Qdrant.
-- Postgres is the source of truth for review/versioning; Qdrant holds the
-- vectors and a copy of this payload for retrieval-time filtering.
-- ---------------------------------------------------------------------------

CREATE TABLE chunks (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id             UUID NOT NULL REFERENCES sources (id) ON DELETE CASCADE,
    source_version_id     UUID NOT NULL REFERENCES source_versions (id) ON DELETE CASCADE,
    chunk_index           INTEGER NOT NULL,        -- order within the document
    chunk_text            TEXT NOT NULL,
    content_hash          TEXT NOT NULL,           -- hash of this chunk's text

    -- required retrieval-time metadata (mirrored into Qdrant payload)
    source_url            TEXT NOT NULL,
    retrieval_date        TIMESTAMPTZ NOT NULL,
    source_last_modified  TIMESTAMPTZ,
    service_category      service_category NOT NULL,
    canonical             BOOLEAN NOT NULL,
    jurisdiction          TEXT[] NOT NULL DEFAULT ARRAY['all'],
    applicant_variant     TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    review_status         review_status NOT NULL DEFAULT 'pending_review',
    version               INTEGER NOT NULL,        -- mirrors source_versions.version

    -- populated by Phase 4 (embedding/indexing)
    embedding_model       TEXT,
    qdrant_point_id       UUID,

    -- when a newer chunk supersedes this one (content change re-review)
    superseded_by         UUID REFERENCES chunks (id),

    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_chunks_source_id ON chunks (source_id);
CREATE INDEX idx_chunks_source_version_id ON chunks (source_version_id);
CREATE INDEX idx_chunks_review_status ON chunks (review_status);
CREATE INDEX idx_chunks_service_category ON chunks (service_category);
CREATE INDEX idx_chunks_canonical ON chunks (canonical);
CREATE INDEX idx_chunks_qdrant_point_id ON chunks (qdrant_point_id);

-- ---------------------------------------------------------------------------
-- audit_log: append-only log of everything that happens to a source/version/
-- chunk — fetch attempts (success AND failure), extraction, chunking,
-- embedding, approve/reject/supersede decisions, manual uploads, etc.
-- ---------------------------------------------------------------------------

CREATE TABLE audit_log (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type   TEXT NOT NULL,          -- 'source' | 'source_version' | 'chunk'
    entity_id     UUID,                   -- nullable: a failed fetch may predate any row existing
    action        TEXT NOT NULL,          -- e.g. 'fetch_attempt', 'extract', 'chunk_created',
                                           --      'embed', 'approved', 'rejected', 'superseded'
    actor         TEXT NOT NULL DEFAULT 'system',  -- 'system' or an admin identifier
    details       JSONB,                  -- freeform: error message, http status, diff, etc.
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_audit_log_entity ON audit_log (entity_type, entity_id);
CREATE INDEX idx_audit_log_action ON audit_log (action);
CREATE INDEX idx_audit_log_created_at ON audit_log (created_at);

-- ---------------------------------------------------------------------------
-- updated_at maintenance trigger
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_sources_updated_at
    BEFORE UPDATE ON sources
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER trg_chunks_updated_at
    BEFORE UPDATE ON chunks
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
