import pytest

from research_agent.literature.papers import (
    PaperRecord,
    PaperStore,
    normalise_arxiv_id,
    normalise_doi,
    title_key,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://doi.org/10.1109/CVPR52688.2022.01553", "10.1109/cvpr52688.2022.01553"),
        ("doi:10.1038/Nature14539", "10.1038/nature14539"),
        ("", None),
        (None, None),
    ],
)
def test_normalise_doi(raw, expected):
    assert normalise_doi(raw) == expected


@pytest.mark.parametrize(
    "raw", ["2106.09685", "2106.09685v2", "arXiv:2106.09685", "https://arxiv.org/abs/2106.09685v1"]
)
def test_normalise_arxiv_id(raw):
    assert normalise_arxiv_id(raw) == "2106.09685"


def test_arxiv_doi_becomes_the_arxiv_id():
    record = PaperRecord(source="openalex", title="x", doi="https://doi.org/10.48550/arxiv.2106.09685")
    assert (record.arxiv_id, record.doi) == ("2106.09685", None)


def _row(db, paper_id):
    return db.query("SELECT * FROM papers WHERE paper_id = ?", (paper_id,))[0]


def test_sources_merge_by_id_and_trusted_title_wins(db):
    store = PaperStore(db)
    lora = "LoRA: Low-Rank Adaptation of Large Language Models"
    first = store.upsert(PaperRecord(source="openalex", title="LoRA Fine-Tuning of a 3B Code LLM",
                                     doi="10.48550/arXiv.2106.09685", openalex_id="https://openalex.org/W3168867926",
                                     citation_count=5))
    second = store.upsert(PaperRecord(source="arxiv", title=lora, arxiv_id="2106.09685v2", abstract="We propose LoRA."))
    third = store.upsert(PaperRecord(source="s2", title=lora, s2_id="abc", arxiv_id="2106.09685", citation_count=9))

    assert first == second == third
    row = _row(db, first)
    assert row["title"] == lora  # arXiv outranks OpenAlex; S2 doesn't outrank arXiv
    assert row["meta_source"] == "arxiv"
    assert (row["arxiv_id"], row["s2_id"], row["openalex_id"]) == ("2106.09685", "abc", "W3168867926")
    assert row["citation_count"] == 9
    assert db.query("SELECT COUNT(*) FROM papers")[0][0] == 1


def test_preprint_and_published_version_merge_by_title(db):
    store = PaperStore(db)
    title = "Masked Autoencoders Are Scalable Vision Learners"
    preprint = store.upsert(PaperRecord(source="openalex", title=title, year=2021, doi="10.48550/arxiv.2111.06377"))
    published = store.upsert(PaperRecord(source="openalex", title=title + ".", year=2022, doi="10.1109/cvpr52688.2022.01553"))
    assert preprint == published
    assert _row(db, preprint)["doi"] == "10.1109/cvpr52688.2022.01553"


def test_short_titles_need_matching_years(db):
    store = PaperStore(db)
    a = store.upsert(PaperRecord(source="s2", title="Attention Is All You Need", year=2017))
    b = store.upsert(PaperRecord(source="s2", title="Attention is all you need", year=2023))
    assert a != b
    assert title_key("Attention Is All You Need") == "attentionisallyouneed"


def test_an_id_held_by_another_paper_is_not_moved(db):
    store = PaperStore(db)
    holder = store.upsert(PaperRecord(source="s2", title="Paper One About Something Long Enough", s2_id="s1", doi="10.1/a"))
    other = store.upsert(PaperRecord(source="arxiv", title="Paper Two Is A Different Paper", arxiv_id="2001.00001"))
    # A record linking the arXiv paper to holder's DOI must not steal the DOI.
    assert store.upsert(PaperRecord(source="s2", title="Paper Two Is A Different Paper", arxiv_id="2001.00001", doi="10.1/a")) == other
    assert _row(db, holder)["doi"] == "10.1/a"
    assert _row(db, other)["doi"] is None
