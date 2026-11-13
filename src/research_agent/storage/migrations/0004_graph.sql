-- Phase 2: the research knowledge graph.

-- Each LLM extraction of one chunk of a paper (a run of whole sections, or the
-- abstract), kept for reuse: a later run that shows the model the same sentences
-- with the same prompt version reuses it instead of calling the model again.
CREATE TABLE extractions (
    extraction_id   TEXT PRIMARY KEY,
    paper_id        TEXT NOT NULL REFERENCES papers (paper_id) ON DELETE CASCADE,
    kind            TEXT NOT NULL CHECK (kind IN ('full', 'abstract')),
    chunk           INTEGER NOT NULL,          -- the chunk's position in the paper
    sections        TEXT NOT NULL,             -- JSON list of the section indexes shown
    input_hash      TEXT NOT NULL,             -- hash of the prompt version and the sentences shown
    prompt_version  TEXT NOT NULL,
    model           TEXT NOT NULL,             -- provider:model that answered
    output          TEXT NOT NULL,             -- JSON: the reply, before the quote check
    created_at      TEXT NOT NULL
);
CREATE INDEX extractions_input ON extractions (input_hash);

-- The extractions each run used, new or reused.
CREATE TABLE run_extractions (
    run_id         TEXT NOT NULL REFERENCES runs (run_id),
    extraction_id  TEXT NOT NULL REFERENCES extractions (extraction_id) ON DELETE CASCADE,
    PRIMARY KEY (run_id, extraction_id)
);

-- Other LLM decisions kept for reuse: name groupings and claim-pair labels.
CREATE TABLE judgments (
    kind        TEXT NOT NULL,                 -- name_groups, claim_pair
    input_hash  TEXT NOT NULL,                 -- hash of the prompt version and the input
    model       TEXT NOT NULL,
    output      TEXT NOT NULL,                 -- JSON
    created_at  TEXT NOT NULL,
    PRIMARY KEY (kind, input_hash, model)
);

-- Text embeddings, kept for reuse. The free tier counts every text as a request.
CREATE TABLE embeddings (
    model       TEXT NOT NULL,                 -- provider:model@dimensions
    text_hash   TEXT NOT NULL,                 -- sha256 of the text sent
    vector      BLOB NOT NULL,                 -- float32, unit length
    created_at  TEXT NOT NULL,
    PRIMARY KEY (model, text_hash)
);

-- Each run builds its own graph from the extractions of its papers.
ALTER TABLE nodes ADD COLUMN run_id TEXT REFERENCES runs (run_id);
ALTER TABLE edges ADD COLUMN run_id TEXT REFERENCES runs (run_id);
CREATE INDEX nodes_run ON nodes (run_id, node_type);
CREATE INDEX edges_run ON edges (run_id, edge_type);

-- Every surface form a merged node was written as.
CREATE TABLE node_aliases (
    node_id  TEXT NOT NULL REFERENCES nodes (node_id) ON DELETE CASCADE,
    alias    TEXT NOT NULL,
    PRIMARY KEY (node_id, alias)
);

-- Evidence now goes when its sentence goes. Addresses are positional, so when a
-- paper is parsed again its old addresses no longer mean what the evidence quoted.
CREATE TABLE evidence_new (
    owner_kind   TEXT NOT NULL CHECK (owner_kind IN ('node', 'edge')),
    owner_id     TEXT NOT NULL,
    sentence_id  TEXT NOT NULL REFERENCES sentences (sentence_id) ON DELETE CASCADE,
    quote        TEXT NOT NULL,
    PRIMARY KEY (owner_kind, owner_id, sentence_id)
);
INSERT INTO evidence_new SELECT owner_kind, owner_id, sentence_id, quote FROM evidence;
DROP TABLE evidence;
ALTER TABLE evidence_new RENAME TO evidence;
CREATE INDEX evidence_sentence ON evidence (sentence_id);
