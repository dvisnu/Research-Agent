"""The Phase 1 stages end to end, on canned API responses."""

import json

import httpx
import pytest

from fakes import FakeModel, Router, fake_services
from research_agent.pipeline import RunError, start_run
from research_agent.storage import Database
from research_agent.trace import TraceRecorder
from test_parse import HTML

QUESTION = "parameter-efficient fine-tuning of large language models with low-rank adaptation"
LORA = "LoRA: Low-Rank Adaptation of Large Language Models"


def _atom(*entries: tuple[str, str, str]) -> str:
    items = "".join(
        f"""<entry><id>http://arxiv.org/abs/{aid}v1</id><title>{title}</title><summary>{summary}</summary>
        <published>2021-06-17T00:00:00Z</published><updated>2021-06-17T00:00:00Z</updated>
        <author><name>A. Author</name></author></entry>"""
        for aid, title, summary in entries
    )
    return f'<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">{items}</feed>'


def _s2(paper_id: str, title: str, arxiv: str | None = None, abstract: str | None = None) -> dict:
    return {"paperId": paper_id, "title": title, "abstract": abstract,
            "externalIds": {"ArXiv": arxiv} if arxiv else {}, "authors": [], "year": 2021}


def _router() -> Router:
    router = Router()
    router.add("export.arxiv.org", "/api/query", _atom(
        ("2106.09685", LORA, "We propose low-rank adaptation for large language models."),
        ("2111.06377", "Masked Autoencoders Are Scalable Vision Learners", "Self-supervised vision."),
    ))
    router.add("api.openalex.org", "/works", {"results": [
        # OpenAlex's wrong title for LoRA's arXiv DOI: must merge into LoRA, not replace its title.
        {"id": "https://openalex.org/W1", "display_name": "LoRA Fine-Tuning of a 3B Code LLM",
         "doi": "https://doi.org/10.48550/arxiv.2106.09685"},
        {"id": "https://openalex.org/W2", "display_name": "QLoRA: Efficient Finetuning of Quantized LLMs",
         "doi": "https://doi.org/10.48550/arxiv.2305.14314",
         "abstract_inverted_index": {"Efficient": [0], "finetuning": [1], "of": [2], "quantized": [3], "language": [4], "models": [5]}},
    ]})
    router.add("api.semanticscholar.org", "/graph/v1/paper/search", {"data": [
        _s2("s2lora", LORA, "2106.09685", "Low-rank adaptation of large language models."),
        _s2("s2prefix", "Prefix-Tuning: Optimizing Continuous Prompts for Generation", "2101.00190",
            "Prefix-tuning keeps language model parameters frozen for parameter-efficient fine-tuning."),
    ]})
    router.add("api.semanticscholar.org", "/graph/v1/snippet/search", {"data": [
        {"snippet": {"text": "Adapters add small trainable layers.", "section": "Method"}, "score": 0.9,
         "paper": {"corpusId": 123, "title": "Parameter-Efficient Transfer Learning for NLP"}},
    ]})
    router.add("api.semanticscholar.org", "/graph/v1/paper/batch", [
        _s2("s2adapters", "Parameter-Efficient Transfer Learning for NLP", "1902.00751",
            "Adapter modules yield parameter-efficient fine-tuning of language models."),
    ])
    router.add("api.semanticscholar.org", "/graph/v1/paper/s2lora/references", {"data": [
        {"citedPaper": _s2("s2bitfit", "BitFit: Simple Parameter-efficient Fine-tuning", "2106.10199")},
    ]})
    router.add("arxiv.org", "/html/2106.09685", HTML)
    return router


@pytest.fixture
def small_corpus_settings(settings):
    return settings.model_copy(update={"full_read": 2, "abstract_only": 2})


def _corpus(settings, run_id):
    with Database(settings.db_path) as db:
        rows = db.query(
            "SELECT p.*, r.found_by, r.rank, r.read_level, r.reason FROM run_papers r"
            " JOIN papers p USING (paper_id) WHERE r.run_id = ? ORDER BY r.rank", (run_id,))
        events = {e["step"]: e for e in TraceRecorder(db).events(run_id) if e["activity_type"] == "step"}
        sentences = db.query("SELECT sentence_id, text FROM sentences")
    return rows, events, sentences


def test_search_expand_rank_and_parse(small_corpus_settings, tmp_path):
    settings = small_corpus_settings
    services = fake_services(settings, tmp_path, _router())
    result = start_run(settings, QUESTION, services=services)
    rows, events, sentences = _corpus(settings, result.run_id)
    by_title = {row["title"]: row for row in rows}

    # Six distinct papers: LoRA came from arXiv, OpenAlex and S2 but is one row, with arXiv's title.
    assert len(rows) == 6
    lora = by_title[LORA]
    assert (lora["s2_id"], lora["openalex_id"], lora["arxiv_id"]) == ("s2lora", "W1", "2106.09685")
    channels = {f["channel"] for f in json.loads(lora["found_by"])}
    assert {"arxiv_search", "openalex_search", "s2_search"} <= channels
    snippet_paper = by_title["Parameter-Efficient Transfer Learning for NLP"]
    assert json.loads(snippet_paper["found_by"])[0]["snippet"] == "Adapters add small trainable layers."
    assert json.loads(by_title["BitFit: Simple Parameter-efficient Fine-tuning"]["found_by"])[0] == {
        "channel": "s2_references", "via": lora["paper_id"]}

    # Ranked; the top 2 were meant for full reading, but only LoRA had full text.
    assert [row["rank"] for row in rows] == [1, 2, 3, 4, 5, 6]
    assert rows[0]["title"] == LORA and rows[0]["read_level"] == "full"
    assert rows[1]["read_level"] == "abstract" and rows[1]["reason"].startswith("no usable full text")
    assert [row["read_level"] for row in rows[2:]] == ["abstract", "abstract", "excluded", "excluded"]

    # LoRA's full text is parsed into addressed sentences; abstract-level papers get their abstracts.
    assert lora["text_source"] == "arxiv_html"
    lora_sentences = [s for s in sentences if s["sentence_id"].startswith(lora["paper_id"] + ":")]
    assert lora_sentences[0]["sentence_id"] == f"{lora['paper_id']}:s0:0"
    assert all(row["text_source"] == "abstract" for row in rows[1:4] if row["abstract"])

    # The trace is the retrieval log.
    search = events["search"]["detail"]["queries"]
    assert {q["channel"] for q in search} == {
        f"{channel}:{angle}" for channel in ("arxiv_search", "openalex_search", "s2_search", "s2_snippets") for angle in (0, 1)
    }
    assert len(events["rank"]["dropped"]) == 2
    assert all(d["reason"].startswith("rank ") for d in events["rank"]["dropped"])
    assert events["fulltext"]["detail"]["full_text"][1]["result"] == "demoted to abstract"
    services.sources.close()


def test_without_semantic_scholar_expansion_uses_openalex(small_corpus_settings, tmp_path):
    router = _router()
    router.add("api.openalex.org", "/works/doi:10.48550/arXiv.2106.09685", {
        "id": "https://openalex.org/W1", "display_name": "LoRA Fine-Tuning of a 3B Code LLM",
        "referenced_works": ["https://openalex.org/W9"], "related_works": []})
    router.add("api.openalex.org", "/works", lambda request: httpx.Response(200, json={"results": [
        {"id": "https://openalex.org/W9", "display_name": "Adapter Layers for Transfer Learning"}]})
        if request.url.params.get("filter") == "openalex_id:W9" else httpx.Response(200, json={"results": []}))
    services = fake_services(small_corpus_settings, tmp_path, router, s2=False)
    result = start_run(small_corpus_settings, QUESTION, services=services)
    rows, events, _ = _corpus(small_corpus_settings, result.run_id)
    assert "Adapter Layers for Transfer Learning" in {row["title"] for row in rows}
    assert "no Semantic Scholar key" in str(events["search"]["detail"]["skipped"])
    services.sources.close()


def test_search_fails_when_every_channel_fails(settings, tmp_path):
    router = Router()
    for host in ("export.arxiv.org", "api.openalex.org", "api.semanticscholar.org"):
        router.add(host, "/", httpx.Response(503))
    services = fake_services(settings, tmp_path, router)
    with pytest.raises(RunError, match="AllChannelsFailed"):
        start_run(settings, QUESTION, services=services)
    services.sources.close()


def _stamped_pdf(stamp: str, title: str, body: str) -> bytes:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(10, 100, 60, 750), stamp, fontsize=20, rotate=90)
    page.insert_textbox(pymupdf.Rect(72, 60, 540, 100), title, fontsize=16)
    page.insert_textbox(pymupdf.Rect(72, 120, 540, 160), "1 Introduction", fontsize=12)
    page.insert_textbox(pymupdf.Rect(72, 170, 540, 260), body, fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


def test_each_angle_of_the_brief_is_searched_on_every_channel(small_corpus_settings, tmp_path):
    model = FakeModel(angles=2)
    services = fake_services(small_corpus_settings, tmp_path, _router(), model=model)
    result = start_run(small_corpus_settings, QUESTION, services=services)
    rows, events, _ = _corpus(small_corpus_settings, result.run_id)
    services.sources.close()

    brief = events["brief"]["detail"]["brief"]
    assert [angle["label"] for angle in brief["angles"]] == ["the question as written", "angle 0", "angle 1"]
    queries = {q["channel"]: q["query"] for q in events["search"]["detail"]["queries"]}
    assert len(queries) == 12
    assert queries["openalex_search:2"] == brief["angles"][2]["keyword_query"]
    lora = next(row for row in rows if row["title"] == LORA)
    assert {(f["channel"], f["angle"]) for f in json.loads(lora["found_by"]) if f["channel"] == "arxiv_search"} == {
        ("arxiv_search", 0), ("arxiv_search", 1), ("arxiv_search", 2)}
    # The brief came from the reasoning model, the relevance check from the extraction model.
    assert [call["model"] for call in model.calls] == ["reasoner", "extractor"]
    llm_calls = [e for e in _events(small_corpus_settings, result.run_id) if e["activity_type"] == "llm_call"]
    assert [(e["step"], e["prompt_version"]) for e in llm_calls] == [
        ("brief", "research_brief@v2"), ("rank", "relevance_check@v1")]


def _events(settings, run_id):
    with Database(settings.db_path) as db:
        return TraceRecorder(db).events(run_id)


def test_uploads_are_read_in_full_and_merge_with_their_online_copy(small_corpus_settings, tmp_path):
    settings = small_corpus_settings
    plain = tmp_path / "mine.pdf"
    plain.write_bytes(_stamped_pdf("", "An Uploaded Paper About Adapters", "Our own unpublished notes on adapters. They are short."))
    lora = tmp_path / "lora.pdf"
    lora.write_bytes(_stamped_pdf("arXiv:2106.09685v2 [cs.CL] 16 Oct 2021", "Low Rank Adaptation", "LoRA freezes the weights. It works."))
    services = fake_services(settings, tmp_path, _router())
    result = start_run(settings, QUESTION, services=services, uploads=[plain, lora])
    rows, events, _ = _corpus(settings, result.run_id)
    by_title = {row["title"]: row for row in rows}

    mine = by_title["An Uploaded Paper About Adapters"]
    assert (mine["origin"], mine["text_source"], mine["read_level"]) == ("user-provided", "upload", "full")
    assert "provided by the user" in mine["reason"]
    # The stamped upload merged into the LoRA found online: one row, arXiv's title, the user's text.
    merged = by_title[LORA]
    assert merged["text_source"] == "upload" and merged["read_level"] == "full"
    assert {f["channel"] for f in json.loads(merged["found_by"])} >= {"upload", "arxiv_search"}
    assert len(rows) == 7  # the six papers found online plus the user's own notes
    assert [u["file"] for u in events["search"]["detail"]["uploads"]] == ["mine.pdf", "lora.pdf"]
    services.sources.close()


@pytest.mark.parametrize("reuse", [False, True])
def test_full_texts_from_an_earlier_run_are_reused_only_when_set(small_corpus_settings, tmp_path, reuse):
    settings = small_corpus_settings.model_copy(update={"reuse_texts": reuse})
    router = _router()
    services = fake_services(settings, tmp_path, router)
    start_run(settings, QUESTION, services=services)
    result = start_run(settings, QUESTION, services=services)
    services.sources.close()

    lora_fetches = [r for r in router.requests if r.url.host == "arxiv.org" and r.url.path == "/html/2106.09685"]
    assert len(lora_fetches) == (1 if reuse else 2)
    _, events, _ = _corpus(settings, result.run_id)
    expected = "reused arxiv_html text from an earlier run" if reuse else "parsed arxiv_html"
    assert events["fulltext"]["detail"]["full_text"][0]["result"] == expected


def test_an_unreadable_upload_fails_the_run(settings, tmp_path):
    bad = tmp_path / "notes.pdf"
    bad.write_text("not really a pdf")
    services = fake_services(settings, tmp_path)
    with pytest.raises(RunError, match="could not read uploaded file"):
        start_run(settings, QUESTION, services=services, uploads=[bad])
    services.sources.close()
