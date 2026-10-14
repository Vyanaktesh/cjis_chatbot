-- Consulate office information (working hours, holidays, contacts, walk-in
-- policy, emergency helpline) added in the Oct 2026 knowledge-base audit.
-- None of the existing categories fit, and the category is shown on the
-- chat's source chips.
ALTER TYPE service_category ADD VALUE IF NOT EXISTS 'office_info';
