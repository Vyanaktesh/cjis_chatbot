-- 0002_source_registry_enrichment.sql
-- Widens service_category to the real taxonomy used by the office's actual
-- 47-source registry (finer-grained than Phase 1's placeholder 9 values),
-- and adds two descriptive columns to `sources` that the real registry
-- data provides but Phase 1's schema didn't anticipate: a human-readable
-- title, and the office's own document grouping (distinct from
-- service_category, which is our retrieval-facing taxonomy).
--
-- Existing enum values are left in place (Postgres cannot cheaply drop enum
-- values, and no data has been written yet, so there's nothing at risk in
-- keeping them available for future use).

ALTER TYPE service_category ADD VALUE IF NOT EXISTS 'passport_lost_damaged';
ALTER TYPE service_category ADD VALUE IF NOT EXISTS 'death_documents';
ALTER TYPE service_category ADD VALUE IF NOT EXISTS 'attestation';
ALTER TYPE service_category ADD VALUE IF NOT EXISTS 'global_entry';
ALTER TYPE service_category ADD VALUE IF NOT EXISTS 'registration';
ALTER TYPE service_category ADD VALUE IF NOT EXISTS 'community_events';
ALTER TYPE service_category ADD VALUE IF NOT EXISTS 'out_of_remit';
ALTER TYPE service_category ADD VALUE IF NOT EXISTS 'fraud_advisory';

ALTER TABLE sources ADD COLUMN IF NOT EXISTS title TEXT;
ALTER TABLE sources ADD COLUMN IF NOT EXISTS source_group TEXT;
