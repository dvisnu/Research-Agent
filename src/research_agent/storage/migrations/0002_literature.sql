-- Phase 1: literature acquisition.

ALTER TABLE papers ADD COLUMN title_key TEXT;        -- normalised title, to match versions of one paper
ALTER TABLE papers ADD COLUMN meta_source TEXT;      -- source of the title and abstract: upload, arxiv, s2, openalex
ALTER TABLE papers ADD COLUMN citation_count INTEGER;
ALTER TABLE papers ADD COLUMN pdf_url TEXT;          -- a known open-access PDF
CREATE INDEX papers_title_key ON papers (title_key);

-- The papers each run found, and what it decided about them.
CREATE TABLE run_papers (
    run_id      TEXT NOT NULL REFERENCES runs (run_id),
    paper_id    TEXT NOT NULL REFERENCES papers (paper_id),
    found_by    TEXT NOT NULL DEFAULT '[]',  -- JSON list of {"channel", "rank", ...}: how this run found it
    score       REAL,                         -- relevance to the question
    rank        INTEGER,                      -- 1 = most relevant
    read_level  TEXT CHECK (read_level IN ('full', 'abstract', 'excluded')),
    reason      TEXT,                         -- why it has that read level
    PRIMARY KEY (run_id, paper_id)
);
