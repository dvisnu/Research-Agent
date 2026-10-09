-- Phase 1: the LLM relevance check's label for each screened paper (NULL: not screened).

ALTER TABLE run_papers ADD COLUMN relevance TEXT CHECK (relevance IN ('core', 'related', 'off_topic'));
