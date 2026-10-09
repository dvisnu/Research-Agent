-- Phase 0: runs, literature, knowledge graph, run trace.

CREATE TABLE runs (
    run_id      TEXT PRIMARY KEY,
    question    TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
    error       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

-- Literature ---------------------------------------------------------------

CREATE TABLE papers (
    paper_id      TEXT PRIMARY KEY,
    title         TEXT NOT NULL,
    abstract      TEXT,
    year          INTEGER,
    venue         TEXT,
    authors       TEXT NOT NULL DEFAULT '[]',   -- JSON list of names
    arxiv_id      TEXT UNIQUE,                  -- without the version suffix
    doi           TEXT UNIQUE,                  -- lower-cased, without https://doi.org/
    s2_id         TEXT UNIQUE,
    openalex_id   TEXT UNIQUE,
    origin        TEXT NOT NULL DEFAULT 'retrieved'
                  CHECK (origin IN ('retrieved', 'user-provided')),
    -- The text the sections and sentences were parsed from; NULL until parsed.
    text_source   TEXT,                         -- e.g. arxiv_html, pdf
    text_version  TEXT,                         -- e.g. arXiv v2
    parser        TEXT,                         -- parser name and version
    created_at    TEXT NOT NULL
);

CREATE TABLE sections (
    section_id  TEXT PRIMARY KEY,               -- <paper_id>:s<idx>
    paper_id    TEXT NOT NULL REFERENCES papers (paper_id) ON DELETE CASCADE,
    idx         INTEGER NOT NULL,
    heading     TEXT,
    text        TEXT NOT NULL,
    UNIQUE (paper_id, idx)
);

CREATE TABLE sentences (
    sentence_id  TEXT PRIMARY KEY,              -- <paper_id>:s<section idx>:<idx>
    section_id   TEXT NOT NULL REFERENCES sections (section_id) ON DELETE CASCADE,
    idx          INTEGER NOT NULL,
    text         TEXT NOT NULL,
    char_start   INTEGER NOT NULL,              -- offsets into sections.text
    char_end     INTEGER NOT NULL,
    UNIQUE (section_id, idx)
);

-- Knowledge graph ----------------------------------------------------------

CREATE TABLE nodes (
    node_id     TEXT PRIMARY KEY,
    node_type   TEXT NOT NULL,                  -- paper, task, method, dataset, metric, claim, ...
    name        TEXT NOT NULL,
    attrs       TEXT NOT NULL DEFAULT '{}',     -- JSON
    created_at  TEXT NOT NULL
);
CREATE INDEX nodes_type_name ON nodes (node_type, name);

CREATE TABLE edges (
    edge_id     TEXT PRIMARY KEY,
    edge_type   TEXT NOT NULL,                  -- proposes, uses, evaluates_with, supports, contradicts, ...
    src_id      TEXT NOT NULL REFERENCES nodes (node_id) ON DELETE CASCADE,
    dst_id      TEXT NOT NULL REFERENCES nodes (node_id) ON DELETE CASCADE,
    attrs       TEXT NOT NULL DEFAULT '{}',     -- JSON
    created_at  TEXT NOT NULL
);
CREATE INDEX edges_src ON edges (src_id, edge_type);
CREATE INDEX edges_dst ON edges (dst_id, edge_type);

-- The sentences a node or edge rests on, with the exact quoted span.
CREATE TABLE evidence (
    owner_kind   TEXT NOT NULL CHECK (owner_kind IN ('node', 'edge')),
    owner_id     TEXT NOT NULL,
    sentence_id  TEXT NOT NULL REFERENCES sentences (sentence_id),
    quote        TEXT NOT NULL,
    PRIMARY KEY (owner_kind, owner_id, sentence_id)
);

-- Run trace ----------------------------------------------------------------

-- Append-only and PROV-DM-shaped: each row is an activity (a pipeline step, or
-- an LLM or API call made inside one). `used` and `generated` hold entity refs
-- ("<kind>:<id>"), `agent` says who performed it, and `dropped` records what was
-- considered but not kept, with the reason.
CREATE TABLE trace_events (
    event_id         TEXT PRIMARY KEY,
    run_id           TEXT NOT NULL REFERENCES runs (run_id),
    seq              INTEGER NOT NULL,          -- write order within the run
    parent_event_id  TEXT,                      -- the step a call was made in
    step             TEXT NOT NULL,             -- pipeline stage
    attempt          INTEGER,                   -- nth execution of this step in the run (steps only)
    activity_type    TEXT NOT NULL,             -- step, llm_call, api_call
    agent            TEXT NOT NULL DEFAULT '{}',
    prompt_version   TEXT,
    used             TEXT NOT NULL DEFAULT '[]',
    generated        TEXT NOT NULL DEFAULT '[]',
    dropped          TEXT NOT NULL DEFAULT '[]', -- JSON list of {"ref", "reason"}
    detail           TEXT NOT NULL DEFAULT '{}', -- JSON: HTTP status, token usage, ...
    status           TEXT NOT NULL CHECK (status IN ('ok', 'error')),
    error            TEXT,
    started_at       TEXT NOT NULL,
    ended_at         TEXT NOT NULL,
    UNIQUE (run_id, seq)
);
CREATE INDEX trace_events_run_step ON trace_events (run_id, step);

CREATE TRIGGER trace_events_no_update BEFORE UPDATE ON trace_events
BEGIN
    SELECT RAISE(ABORT, 'trace_events is append-only');
END;

CREATE TRIGGER trace_events_no_delete BEFORE DELETE ON trace_events
BEGIN
    SELECT RAISE(ABORT, 'trace_events is append-only');
END;
