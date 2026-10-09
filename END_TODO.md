# To do at the end

Things deliberately left in a development state. Each is done once the whole
agent works (Phases 1–6), before the Phase 7 evaluation and before shipping.

1. **Swap the LLMs.** Development runs on free tiers (see `.env.example`). Pick
   the final model for each role, keep the verifier in a different model family
   from reasoning, and re-run the checks of every phase.
2. **Restore the corpus size.** `RA_FULL_READ` / `RA_ABSTRACT_ONLY` are 10 + 30
   to fit the free LLM limits (defaults in `config.py` and `.env.example`). Go
   back to 20 + 80, or whatever the re-run checks support.
3. **Run the full key-paper check and tune retrieval with it.** Run
   `scripts/key_paper_recall.py --full-text` on all 18 dev topics (about 2 hours;
   deferred from Phase 1, which was checked on one topic). Use the results to tune:
   - `CITATION_WEIGHT` and `LINK_WEIGHT` in `literature/stages.py`, chosen before
     the LLM relevance check existed;
   - giving full-read places to papers the relevance check labels core first (on
     one topic, 8 of the 10 full-read places went to broad surveys labelled related);
     the stored labels let this be measured without new LLM calls;
   - the brief prompt: the model tends to paraphrase the problem instead of using
     the literature's name for it ("conflicting evidence" for "knowledge conflicts");
   - how strict the arXiv queries are (two ANDed phrases often return nothing).
4. **Remove the development text reuse** (`RA_REUSE_TEXTS`) when the product
   ships: the setting in `config.py`, its branch in `fulltext_stage`, and its
   lines in `.env.example` and `COMMANDS.md`. Wait until the user says it is
   shipping.
5. **Choose the Phase 7 benchmark topics after the final models are fixed.**
   Topics must come from papers published after every model's training cutoff.
