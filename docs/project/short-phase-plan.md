Phase 0 (Foundations) sets up the skeleton: a Python project managed with uv, a Typer command-line interface, an LLM interface with caching, API clients for Semantic Scholar, arXiv, OpenAlex and Unpaywall, SQLite storage, a stub LangGraph pipeline, and the provenance trace. It is done when a stub run passes through every stage and writes a trace.

Phase 1 (Input and literature search) turns the user's question into a research brief, searches for papers, expands the results by following citations, ranks them, and converts each paper into numbered sentences using arXiv HTML or GROBID/PyMuPDF. It is done when each test question yields a ranked, parsed set of papers.

Phase 2 (Knowledge graph) extracts the contents of each paper, checking every quote against the paper's text. It merges duplicate method and dataset names, stores the graph in SQLite and NetworkX, and adds supports and contradicts links between papers. It is done when the graph is built and the quote-check pass rate has been measured.

Phase 3 (Related work and gaps) builds a grid of methods, datasets and metrics, then finds gaps from stated limitations, empty grid cells and contradictions. It is done when it produces ranked gaps whose evidence points to real sentences.

Phase 4 (Ideas and novelty) is the first of the project's two main contributions. It runs a separate novelty search and compares each idea with the ten or so closest papers. Each idea gets a novelty level from 1 to 5, assigned by rules, and an unclear result means a lower level. Ideas that score too low go through a retry loop. The output is a Novel Evidence Report. It is done when the system flags most ideas taken from real published papers as duplicates.

Phase 5 (Experiment plan) is the second contribution. It fills the plan's fields from the graph and keeps three provenance records for each element. It also adds a critique loop and a separate verifier. It is done when every plan element traces back to a paper section.

Phase 6 (Hardening) makes runs resumable, measures cost per run, adds tests and polishes the reports. It is done when all test questions run without manual fixes.

Phase 7 (Evaluation) uses about 50 topics drawn from papers published after the models' training cutoff. It compares the agent against an LLM alone, plain RAG, and versions of the agent with parts removed. Ideas are judged head-to-head by an LLM, and the provenance metrics are measured. It is done when there is a full evaluation report.