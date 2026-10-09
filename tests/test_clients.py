import httpx
import pytest
from pydantic import SecretStr

from research_agent.clients import Arxiv, OpenAlex, SemanticScholar, Unpaywall
from research_agent.clients.arxiv import parse_feed
from research_agent.clients.openalex import abstract_from_inverted_index

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2106.09685v2</id>
    <updated>2021-10-16T18:59:58Z</updated>
    <published>2021-06-17T17:37:18Z</published>
    <title>LoRA: Low-Rank Adaptation of
      Large Language Models</title>
    <summary>  An important paradigm of natural language processing ...  </summary>
    <author><name>Edward J. Hu</name></author>
    <author><name>Yelong Shen</name></author>
    <arxiv:doi>10.48550/arXiv.2106.09685</arxiv:doi>
    <link href="http://arxiv.org/abs/2106.09685v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2106.09685v2" rel="related" type="application/pdf"/>
    <arxiv:primary_category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
  <entry>
    <id>http://arxiv.org/api/errors#incorrect_id_format_for_bad</id>
    <title>Error</title>
  </entry>
</feed>"""


def _mock(handler) -> dict:
    return {"transport": httpx.MockTransport(handler), "sleep": lambda seconds: None}


def test_arxiv_feed_is_parsed_and_error_entries_skipped():
    [entry] = parse_feed(ATOM)
    assert entry.arxiv_id == "2106.09685"
    assert entry.version == 2
    assert entry.title == "LoRA: Low-Rank Adaptation of Large Language Models"
    assert entry.summary.startswith("An important paradigm")
    assert entry.authors == ["Edward J. Hu", "Yelong Shen"]
    assert entry.categories == ["cs.CL", "cs.LG"]
    assert entry.primary_category == "cs.CL"
    assert entry.doi == "10.48550/arXiv.2106.09685"
    assert entry.pdf_url == "http://arxiv.org/pdf/2106.09685v2"


def test_arxiv_get_many_sends_an_id_list():
    seen = []

    def handler(request):
        seen.append(request.url.params)
        return httpx.Response(200, text=ATOM)

    entries = Arxiv(**_mock(handler)).get_many(["2106.09685", "bad"])
    assert [e.arxiv_id for e in entries] == ["2106.09685"]
    assert seen[0]["id_list"] == "2106.09685,bad"


def test_semantic_scholar_unwraps_references_and_sends_the_key(settings, tmp_path):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            json={"data": [{"citedPaper": {"paperId": "abc", "title": "Attention Is All You Need"}},
                           {"citedPaper": {"paperId": None, "title": "An unresolved reference"}}]},
        )

    keyed = settings.model_copy(update={"s2_api_key": SecretStr("secret")})
    refs = SemanticScholar(keyed, rate_file=tmp_path / "s2", **_mock(handler)).references("arXiv:2106.09685", limit=2)
    assert [(p.paper_id, p.title) for p in refs] == [
        ("abc", "Attention Is All You Need"),
        (None, "An unresolved reference"),
    ]
    assert seen[0].url.path == "/graph/v1/paper/arXiv:2106.09685/references"
    assert seen[0].headers["x-api-key"] == "secret"


def test_semantic_scholar_missing_paper_is_none(settings, tmp_path):
    client = SemanticScholar(settings, rate_file=tmp_path / "s2", **_mock(lambda request: httpx.Response(404)))
    assert client.paper("arXiv:0000.00000") is None


def test_semantic_scholar_snippets(settings, tmp_path):
    body = {
        "data": [
            {
                "snippet": {"text": "LoRA freezes the weights.", "snippetKind": "body", "section": "Method"},
                "score": 0.7,
                "paper": {"corpusId": 235458009, "title": "LoRA"},
            }
        ]
    }
    [snippet] = SemanticScholar(settings, rate_file=tmp_path / "s2", **_mock(lambda request: httpx.Response(200, json=body))).snippets("lora")
    assert (snippet.text, snippet.section, snippet.kind, snippet.corpus_id) == (
        "LoRA freezes the weights.", "Method", "body", "235458009"
    )


def test_openalex_rebuilds_abstracts():
    assert abstract_from_inverted_index({"learning": [1], "Deep": [0], "works": [2]}) == "Deep learning works"
    assert abstract_from_inverted_index(None) is None


def test_openalex_sends_mailto(settings):
    seen = []

    def handler(request):
        seen.append(request.url.params)
        return httpx.Response(200, json={"id": "W1", "display_name": "Deep learning",
                                         "abstract_inverted_index": {"Deep": [0], "learning": [1]}})

    with_email = settings.model_copy(update={"contact_email": "me@example.org"})
    work = OpenAlex(with_email, **_mock(handler)).work("doi:10.1038/nature14539")
    assert (work.title, work.abstract) == ("Deep learning", "Deep learning")
    assert seen[0]["mailto"] == "me@example.org"


def test_unpaywall_needs_an_email(settings):
    with pytest.raises(ValueError, match="RA_CONTACT_EMAIL"):
        Unpaywall(settings)


def test_unpaywall_lookup(settings):
    body = {"doi": "10.1038/nature14539", "is_oa": True,
            "best_oa_location": {"url_for_pdf": "https://example.org/x.pdf", "host_type": "repository"}}
    responses = {"/v2/10.1038/nature14539": httpx.Response(200, json=body)}
    client = Unpaywall(
        settings.model_copy(update={"contact_email": "me@example.org"}),
        **_mock(lambda request: responses.get(request.url.path, httpx.Response(404))),
    )
    record = client.lookup("10.1038/nature14539")
    assert record.is_oa and record.best_oa_location.url_for_pdf == "https://example.org/x.pdf"
    assert client.lookup("10.0000/missing") is None
