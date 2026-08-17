-- 0003_chunks_used_ocr.sql
-- Phase 4 caught a real mirror inconsistency: Qdrant's payload for each
-- chunk carries `used_ocr` (whether the chunk's text came from OCR rather
-- than a native text layer — lower-confidence content a reviewer should
-- be able to see), but the Postgres `chunks` table — meant to be the
-- system of record for this metadata per the project brief — had nowhere
-- to store it. Adding it here so both stores agree.

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS used_ocr BOOLEAN NOT NULL DEFAULT false;
