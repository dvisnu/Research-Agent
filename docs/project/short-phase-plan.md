Status (2026-11-13): Phases 0-6 are done and the agent works end to end. What is left is fine-tuning (the final models, the corpus size, retrieval and the known weak spots; see *Remaining work* in `phase-plan.md`), then Phase 7.

Phase 0 (Foundations) sets up the skeleton: a Python project managed with uv, a Typer command-line interface, an LLM interface, API clients for Semantic Scholar, arXiv, OpenAlex and Unpaywall, SQLite storage, a stub LangGraph pipeline, and the provenance trace. It is done when a stub run passes through every stage and writes a trace. Done 2026-10-07.

Phase 1 (Input and literature search) turns the user's question into a research brief, searches for papers, expands the results by following citations, ranks them, and converts each paper into numbered sentences using arXiv HTML or PyMuPDF. Its sources are arXiv and OpenAlex, plus Semantic Scholar once an API key is set. It is done when each test question yields a ranked, parsed set of papers. Done 2026-10-09.

Phase 2 (Knowledge graph) extracts the contents of each paper, checking every quote against the paper's text. It merges duplicate method and dataset names, stores the graph in SQLite and NetworkX, and adds supports and contradicts links between papers. It is done when the graph is built and the quote-check pass rate has been measured. Done 2026-10-10: 98% of 5,310 quotes pass.

Phase 3 (Related work and gaps) builds a grid of methods, datasets and metrics, then finds gaps from stated limitations, empty grid cells and contradictions. It is done when it produces ranked gaps whose evidence points to real sentences. Done 2026-10-10.

Phase 4 (Ideas and novelty) is the first of the project's two main contributions. It runs a separate novelty search and compares each idea with the ten or so closest papers. Each idea gets a novelty level from 1 to 5, assigned by rules, and an unclear result means a lower level. Ideas that score too low go through a retry loop. The output is a Novel Evidence Report. It is done when the system flags most ideas taken from real published papers as duplicates. Done 2026-10-11: all 21 planted ideas were caught.

Phase 5 (Experiment plan) is the second contribution. It fills the plan's fields from the graph and keeps three provenance records for each element. It also adds a critique loop and a separate verifier. It is done when every plan element traces back to a paper section. Done 2026-10-13: 61% of plan elements supported over the dev topics.

Phase 6 (Hardening) makes runs resumable, keeps every version of a paper's text and the LLM replies and search responses so reruns cost little, measures cost per run, adds tests, polishes the reports and adds a local web UI. It is done when all test questions run without manual fixes. Done 2026-10-14.

Phase 7 (Evaluation) uses about 50 topics drawn from papers published after the models' training cutoff. It compares the agent against an LLM alone, plain RAG, and versions of the agent with parts removed. Ideas are judged head-to-head by an LLM, and the provenance metrics are measured. It is done when there is a full evaluation report. Next, after the fine-tuning.
