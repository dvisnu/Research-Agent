-- Phase 5: experiment plans, with three provenance records for each element.

-- The papers a run's plan draws on, and why each is there: the best idea's closest
-- papers, its gap's evidence papers, the related work, or a targeted search for the
-- critique. `detail` holds the search query and channel, and the extractions used.
CREATE TABLE plan_papers (
    run_id    TEXT NOT NULL REFERENCES runs (run_id),
    paper_id  TEXT NOT NULL REFERENCES papers (paper_id),
    role      TEXT NOT NULL CHECK (role IN ('closest', 'gap', 'related', 'search')),
    position  INTEGER,                      -- its rank in that role (closest 1-10, related 1-10, search order)
    detail    TEXT NOT NULL DEFAULT '{}',   -- JSON
    PRIMARY KEY (run_id, paper_id, role)
);

-- Each version of a run's plan: the first draft, then one per revision after a critique.
CREATE TABLE plans (
    plan_id     TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES runs (run_id),
    idea_id     TEXT NOT NULL REFERENCES ideas (idea_id) ON DELETE CASCADE,
    round       INTEGER NOT NULL,             -- 1 for the first draft; each revision adds one
    status      TEXT NOT NULL CHECK (status IN ('draft', 'superseded', 'final')),
    menu        TEXT NOT NULL DEFAULT '[]',   -- JSON: the facts it could cite, each with its evidence and origin
    critique    TEXT,                         -- JSON: the critique of this version (code checks, the verifier's points)
    detail      TEXT NOT NULL DEFAULT '{}',   -- JSON: model, prompt, answers to the previous critique, coverage
    created_at  TEXT NOT NULL,
    UNIQUE (run_id, round)
);

-- One element of a plan, with its three provenance records: its source (evidence rows,
-- owner_kind 'plan_element'), its derivation (how it was produced) and its decision (why
-- it was chosen, over what), plus what the verifier said about it.
CREATE TABLE plan_elements (
    element_id    TEXT PRIMARY KEY,
    plan_id       TEXT NOT NULL REFERENCES plans (plan_id) ON DELETE CASCADE,
    field         TEXT NOT NULL,              -- hypothesis, research_question, method, datasets, ...
    position      INTEGER NOT NULL,           -- its order within the field, from 1
    name          TEXT,                       -- a dataset's, baseline's or metric's name
    text          TEXT NOT NULL,
    basis         TEXT,                       -- what the cited sources say that it rests on
    cites         TEXT NOT NULL DEFAULT '[]', -- JSON list of the facts it cites
    derivation    TEXT NOT NULL DEFAULT '{}', -- JSON: step, round, model, prompt, the cited facts' origins
    decision      TEXT NOT NULL DEFAULT '{}', -- JSON: the reason, the alternatives, the critique points it answers
    verdict       TEXT CHECK (verdict IN ('supported', 'not_supported', 'contradicted', 'no_source', 'unverified')),
    verification  TEXT,                       -- JSON: the verifier's reason and model
    status        TEXT NOT NULL DEFAULT 'kept' CHECK (status IN ('kept', 'flagged', 'removed')),
    UNIQUE (plan_id, field, position)
);

-- Plan elements quote the papers too, so evidence gets a fourth kind of owner.
CREATE TABLE evidence_new (
    owner_kind   TEXT NOT NULL CHECK (owner_kind IN ('node', 'edge', 'comparison', 'plan_element')),
    owner_id     TEXT NOT NULL,
    sentence_id  TEXT NOT NULL REFERENCES sentences (sentence_id) ON DELETE CASCADE,
    quote        TEXT NOT NULL,
    PRIMARY KEY (owner_kind, owner_id, sentence_id)
);
INSERT INTO evidence_new SELECT owner_kind, owner_id, sentence_id, quote FROM evidence;
DROP TABLE evidence;
ALTER TABLE evidence_new RENAME TO evidence;
CREATE INDEX evidence_sentence ON evidence (sentence_id);
