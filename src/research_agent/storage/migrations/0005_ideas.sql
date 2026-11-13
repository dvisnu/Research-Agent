-- Phase 4: ideas and their novelty assessment.

-- Every idea a run proposed, revised or was given, and what its novelty check found.
-- Ideas of one lineage are versions of one idea: a retry changes one facet of the
-- previous version, which is then rejected (and remembered, so it isn't proposed again).
CREATE TABLE ideas (
    idea_id      TEXT PRIMARY KEY,
    run_id       TEXT NOT NULL REFERENCES runs (run_id),
    lineage      INTEGER NOT NULL,           -- which of the run's checked ideas this is a version of
    round        INTEGER NOT NULL,           -- 1 for a first version; each retry adds one
    parent_id    TEXT REFERENCES ideas (idea_id),
    origin       TEXT NOT NULL CHECK (origin IN ('generated', 'revised', 'user', 'planted')),
    gap_node_id  TEXT,                       -- the gap node it answers; NULL for a user's or planted idea
    kind         TEXT,                       -- fix_limitation, combine_methods, borrow_technique
    title        TEXT NOT NULL,
    purpose      TEXT NOT NULL,
    mechanism    TEXT NOT NULL,
    evaluation   TEXT NOT NULL,
    detail       TEXT NOT NULL DEFAULT '{}', -- JSON: builds_on, feasibility, the gap's text, what a retry changed
    status       TEXT NOT NULL CHECK (status IN ('unchecked', 'pending', 'searched', 'assessed', 'rejected')),
    level        INTEGER CHECK (level BETWEEN 1 AND 5),
    assessment   TEXT,                       -- JSON: the level's reasons, caps and search coverage
    final_rank   INTEGER,                    -- 1 = the run's best idea; NULL for versions not reported
    created_at   TEXT NOT NULL
);
CREATE INDEX ideas_run ON ideas (run_id, status);

-- The papers each idea's novelty search found, and how far each got.
CREATE TABLE idea_candidates (
    idea_id     TEXT NOT NULL REFERENCES ideas (idea_id) ON DELETE CASCADE,
    paper_id    TEXT NOT NULL REFERENCES papers (paper_id),
    found_by    TEXT NOT NULL DEFAULT '[]',  -- JSON list of {"channel", "round", "rank", ...}
    similarity  REAL,                        -- the embedding filter's score against the idea
    position    INTEGER,                     -- 1-10 among the closest work after reranking; NULL if cut
    facet_match TEXT,                        -- the reranker's match level for a kept paper
    PRIMARY KEY (idea_id, paper_id)
);

-- The idea compared with each of its closest papers, facet by facet.
CREATE TABLE comparisons (
    comparison_id   TEXT PRIMARY KEY,
    idea_id         TEXT NOT NULL REFERENCES ideas (idea_id) ON DELETE CASCADE,
    paper_id        TEXT NOT NULL REFERENCES papers (paper_id),
    depth           TEXT NOT NULL CHECK (depth IN ('abstract', 'full')),
    verdict         TEXT NOT NULL,           -- JSON: per-facet match, the paper's own facet, gap, difference
    model           TEXT,
    prompt_version  TEXT,
    created_at      TEXT NOT NULL,
    UNIQUE (idea_id, paper_id)
);

-- Comparisons quote the papers too, so evidence gets a third kind of owner.
CREATE TABLE evidence_new (
    owner_kind   TEXT NOT NULL CHECK (owner_kind IN ('node', 'edge', 'comparison')),
    owner_id     TEXT NOT NULL,
    sentence_id  TEXT NOT NULL REFERENCES sentences (sentence_id) ON DELETE CASCADE,
    quote        TEXT NOT NULL,
    PRIMARY KEY (owner_kind, owner_id, sentence_id)
);
INSERT INTO evidence_new SELECT owner_kind, owner_id, sentence_id, quote FROM evidence;
DROP TABLE evidence;
ALTER TABLE evidence_new RENAME TO evidence;
CREATE INDEX evidence_sentence ON evidence (sentence_id);
