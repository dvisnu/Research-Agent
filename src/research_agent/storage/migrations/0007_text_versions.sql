-- Phase 6: every parse of a paper is kept.
--
-- A new parse no longer replaces the paper's sections. It is stored as a new text,
-- numbered after the earlier ones, its sections numbered on from the paper's last
-- section, and the sections of the text it replaces are retired. A sentence address
-- therefore never changes what it points at, and evidence quoting an earlier text
-- (an abstract later read in full, an older arXiv version) keeps its sentences.

CREATE TABLE paper_texts (
    paper_id    TEXT NOT NULL REFERENCES papers (paper_id) ON DELETE CASCADE,
    text_no     INTEGER NOT NULL,              -- 1, 2, ... in the order stored
    source      TEXT NOT NULL,                 -- abstract, arxiv_html, pdf, upload, abstract+arxiv_html, ...
    version     TEXT,                          -- e.g. arXiv v2
    parser      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (paper_id, text_no)
);

-- The text a section was stored with, and the text that retired it (NULL while it
-- belongs to the paper's current text).
ALTER TABLE sections ADD COLUMN text_no INTEGER NOT NULL DEFAULT 1;
ALTER TABLE sections ADD COLUMN retired_by INTEGER;
CREATE INDEX sections_current ON sections (paper_id, retired_by);

-- Texts stored before this migration become text 1 (dated with their paper).
INSERT INTO paper_texts (paper_id, text_no, source, version, parser, created_at)
SELECT paper_id, 1, text_source, text_version, COALESCE(parser, 'unknown'), created_at
FROM papers WHERE text_source IS NOT NULL;
