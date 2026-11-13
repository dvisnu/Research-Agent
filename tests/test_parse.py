import pymupdf
import pytest

from research_agent.graph.extraction import load_sections
from research_agent.literature.parse import ParseError, parse_arxiv_html, parse_pdf
from research_agent.literature.sentences import split_sentences, split_spans
from research_agent.literature.texts import abstract_text, store_text
from research_agent.literature.papers import PaperRecord, PaperStore

HTML = """<html><body>
<a href="/abs/2401.00001v3">Abstract page</a>
<article class="ltx_document">
<h1 class="ltx_title ltx_title_document">Routing with <math alttext="\\alpha">α</math>-Experts</h1>
<div class="ltx_authors">Alice and Bob</div>
<div class="ltx_abstract"><h6 class="ltx_title ltx_title_abstract">Abstract</h6>
<p class="ltx_p">We study routing. It works well.</p></div>
<section class="ltx_section" id="S1"><h2 class="ltx_title ltx_title_section">1 Introduction</h2>
<div class="ltx_para"><p class="ltx_p">Prior work (Yang et al., 2016) used <math alttext="W_{0}"><mi>W</mi><annotation>W_{0}</annotation></math> as in Eq. 5 of their paper.<span class="ltx_note ltx_role_footnote"><span class="ltx_note_content">A footnote that must not appear.</span></span> We extend it.</p></div>
<div class="ltx_para"><ul class="ltx_itemize"><li class="ltx_item"><div class="ltx_para"><p class="ltx_p">First point.</p></div></li></ul></div>
<figure class="ltx_table"><figcaption class="ltx_caption">Table 1: Results on GLUE.</figcaption><table><tr><td>99.9</td></tr></table></figure>
<section class="ltx_subsection" id="S1.SS1"><h3 class="ltx_title ltx_title_subsection">1.1 Setup</h3>
<div class="ltx_para"><p class="ltx_p">We use eight experts.</p></div></section>
</section>
<section class="ltx_bibliography"><h2 class="ltx_title">References</h2><ul><li class="ltx_bibitem">Yang et al. 2016. A cited paper.</li></ul></section>
</article></body></html>"""


def test_arxiv_html_sections_and_cleanup():
    parsed = parse_arxiv_html(HTML)
    assert parsed.version == "v3"
    assert parsed.title == "Routing with \\alpha-Experts"
    assert [s.heading for s in parsed.sections] == ["Abstract", "1 Introduction", "1.1 Setup"]
    abstract, intro, setup = parsed.sections
    assert abstract.paragraphs == ["We study routing. It works well."]
    assert intro.paragraphs == [
        "Prior work (Yang et al., 2016) used W_{0} as in Eq. 5 of their paper. We extend it.",
        "First point.",  # the list's inner paragraph is read once, as part of the outer one
        "Table 1: Results on GLUE.",  # caption kept, table body dropped
    ]
    assert setup.paragraphs == ["We use eight experts."]
    assert "footnote" not in str(parsed) and "cited paper" not in str(parsed)


def test_page_without_html_is_a_parse_error():
    with pytest.raises(ParseError):
        parse_arxiv_html("<html><body>No HTML for this paper</body></html>")


def test_sentence_splits_survive_scientific_abbreviations():
    paragraph = (
        "Prior work (Yang et al., 2016) used it as in Eq. 5 of their paper. "
        "See Fig. 3 and Sec. 4 for details, e.g. the ablation. We extend it."
    )
    sentences = split_sentences(paragraph)
    assert sentences == [
        "Prior work (Yang et al., 2016) used it as in Eq. 5 of their paper.",
        "See Fig. 3 and Sec. 4 for details, e.g. the ablation.",
        "We extend it.",
    ]
    assert [paragraph[a:b] for a, b in split_spans(paragraph)] == sentences


def test_spans_are_exact_even_when_falling_back():
    from research_agent.literature import sentences as module

    paragraph = "In Proc. the 60th Annual Meeting of the ACL, pages 1-9. Second sentence here."
    original = module._pysbd_spans
    module._pysbd_spans = lambda text: None  # as if pysbd had altered the text
    try:
        spans = split_spans(paragraph)
    finally:
        module._pysbd_spans = original
    assert [paragraph[a:b] for a, b in spans] == [
        "In Proc. the 60th Annual Meeting of the ACL, pages 1-9.",
        "Second sentence here.",
    ]


def _make_pdf() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    blocks = [
        (60, "A Paper About Routing"),
        (110, "Abstract"),
        (140, "We study routing in mixture-of-experts models. It works."),
        (200, "1 Introduction"),
        (230, "Routing decides which expert sees each token, and the"),
        (300, "load must stay balanced. That is the hard part."),
        (370, "References"),
        (400, "[1] A. Author. A cited paper. 2020."),
        (page.rect.height - 20, "3"),  # page number in the bottom margin
    ]
    for y, text in blocks:
        page.insert_textbox(pymupdf.Rect(72, y, 540, y + 40), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def test_pdf_sections_paragraph_joining_and_references():
    parsed = parse_pdf(_make_pdf())
    by_heading = {s.heading: s.paragraphs for s in parsed.sections}
    assert by_heading["Abstract"] == ["We study routing in mixture-of-experts models. It works."]
    # The paragraph broken across blocks is joined back together.
    assert by_heading["1 Introduction"] == [
        "Routing decides which expert sees each token, and the load must stay balanced. That is the hard part."
    ]
    assert "References" not in by_heading
    assert not any("cited paper" in p for s in parsed.sections for p in s.paragraphs)
    assert not any(p == "3" for s in parsed.sections for p in s.paragraphs)


def test_not_a_pdf_is_a_parse_error():
    with pytest.raises(ParseError):
        parse_pdf(b"<html>not a pdf</html>")


def test_stored_sentences_have_addresses_and_exact_spans(db):
    paper_id = PaperStore(db).upsert(PaperRecord(source="arxiv", title="Routing with Experts", arxiv_id="2401.00001"))
    count = store_text(db, paper_id, parse_arxiv_html(HTML), source="arxiv_html", parser="test")
    rows = db.query(
        "SELECT s.sentence_id, s.text, s.char_start, s.char_end, c.text AS section_text"
        " FROM sentences s JOIN sections c USING (section_id) ORDER BY c.idx, s.idx"
    )
    assert count == len(rows) == 7
    assert rows[0]["sentence_id"] == f"{paper_id}:s0:0"
    assert rows[2]["sentence_id"] == f"{paper_id}:s1:0"
    for row in rows:
        assert row["section_text"][row["char_start"] : row["char_end"]] == row["text"]
    paper = db.query("SELECT text_source, text_version, parser FROM papers")[0]
    assert tuple(paper) == ("arxiv_html", "v3", "test")

    # The same parse again is not stored twice.
    store_text(db, paper_id, parse_arxiv_html(HTML), source="arxiv_html", parser="test")
    assert db.query("SELECT COUNT(*) FROM sentences")[0][0] == 7
    assert db.query("SELECT COUNT(*) FROM paper_texts")[0][0] == 1


def test_a_new_parse_is_a_new_text_and_the_old_one_keeps_its_sentences_and_evidence(db):
    paper_id = PaperStore(db).upsert(PaperRecord(source="arxiv", title="Routing with Experts", arxiv_id="2401.00001"))
    store_text(db, paper_id, abstract_text("t", "We route tokens. It works."), source="abstract", parser="abstract")
    db.execute("INSERT INTO evidence (owner_kind, owner_id, sentence_id, quote) VALUES ('node', 'n1', ?, 'We route tokens')",
               (f"{paper_id}:s0:0",))

    count = store_text(db, paper_id, parse_arxiv_html(HTML), source="arxiv_html", parser="test")

    # The new text's sections are numbered on from the abstract; the abstract is retired, not deleted.
    sections = db.query("SELECT idx, text_no, retired_by FROM sections WHERE paper_id = ? ORDER BY idx", (paper_id,))
    assert [tuple(s) for s in sections] == [(0, 1, 2), (1, 2, None), (2, 2, None), (3, 2, None)]
    assert [s.idx for s in load_sections(db, paper_id)] == [1, 2, 3]
    assert sum(len(s.lines) for s in load_sections(db, paper_id)) == count == 7
    assert db.query("SELECT text FROM sentences WHERE sentence_id = ?", (f"{paper_id}:s0:0",))[0]["text"] == "We route tokens."
    assert db.query("SELECT COUNT(*) FROM evidence WHERE owner_id = 'n1'")[0][0] == 1
    texts = db.query("SELECT text_no, source, version FROM paper_texts WHERE paper_id = ? ORDER BY text_no", (paper_id,))
    assert [tuple(t) for t in texts] == [(1, "abstract", None), (2, "arxiv_html", "v3")]
    assert db.query("SELECT text_source FROM papers")[0]["text_source"] == "arxiv_html"

    # A third, different parse retires the second in turn.
    store_text(db, paper_id, abstract_text("t", "Only an abstract now."), source="abstract", parser="abstract")
    assert [s.idx for s in load_sections(db, paper_id)] == [4]
    assert db.query("SELECT COUNT(*) FROM sentences")[0][0] == 2 + 7 + 1
