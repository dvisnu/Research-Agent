-- Phase 6: a run ranks its candidates by what its own searches said about them.
--
-- A paper row merges every record any run has fetched, so its citation count or
-- abstract can change after a run ranked it (a later search fetched a fresher
-- record). A rerun of the question, reading the same kept responses, then ranked
-- differently. Each run now keeps its own merge of the records it fetched.

-- JSON {source, title, abstract, year, citation_count}; NULL for runs before this migration.
ALTER TABLE run_papers ADD COLUMN seen TEXT;
