"""Store a paper's text as sections and addressed sentences.

Every parse of a paper is kept (``paper_texts``). A parse that differs from the
paper's current text becomes a new text: its sections are numbered on from the
paper's last section, and the sections it replaces are retired, not deleted. So a
sentence address never changes what it points at, and evidence that quotes an
earlier text (an abstract later read in full, an older arXiv version) stays valid.
Readers of a paper's text see its current sections only (``retired_by IS NULL``).
"""

import re
import sqlite3

from research_agent.literature.parse import ParsedSection, ParsedText
from research_agent.literature.sentences import split_spans
from research_agent.storage import Database, section_id, sentence_id, utc_now

Layout = list[tuple[str | None, str, list[tuple[str, int, int]]]]  # (heading, text, sentences with spans)


def store_text(
    db: Database, paper_id: str, parsed: ParsedText, *, source: str, parser: str
) -> int:
    """Make ``parsed`` the paper's current text. Returns its sentence count.

    A parse identical to the current text is not stored again; only where it came
    from is updated.
    """
    layout = [(section.heading, *_layout(section)) for section in parsed.sections]
    count = sum(len(sentences) for _, _, sentences in layout)
    with db.transaction() as conn:
        current = _current_text(conn, paper_id)
        if current is not None and _stored_layout(conn, paper_id) == layout:
            conn.execute(
                "UPDATE paper_texts SET source = ?, version = ?, parser = ? WHERE paper_id = ? AND text_no = ?",
                (source, parsed.version, parser, paper_id, current),
            )
            _set_paper_source(conn, paper_id, source, parsed.version, parser)
            return count
        start = _next_section(conn, paper_id)
        text_no = _new_text(conn, paper_id, source, parsed.version, parser)
        conn.execute("UPDATE sections SET retired_by = ? WHERE paper_id = ? AND retired_by IS NULL", (text_no, paper_id))
        _insert_sections(conn, paper_id, text_no, start, layout)
    return count


EXTENDED_PREFIX = "abstract+"  # text_source of an abstract extended with its paper's body
_ABSTRACT_HEADING = re.compile(r"^\s*abstract\s*$", re.IGNORECASE)


def extend_text(db: Database, paper_id: str, parsed: ParsedText, *, source: str, parser: str) -> int:
    """Add a full text's body after the stored abstract, as a new text that keeps the
    abstract's sentences in it: a reader of the paper sees the abstract, then the body.
    The parse's own abstract is left out. Returns the number of sentences added.

    Used by the novelty check, which reads some closest papers in full when other runs
    may already cite their abstracts.
    """
    with db.transaction() as conn:
        row = conn.execute("SELECT text_source FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        if row is None or row["text_source"] != "abstract":
            raise ValueError(f"{paper_id} has no stored abstract-only text to extend")
        body = [section for section in parsed.sections if not (section.heading and _ABSTRACT_HEADING.match(section.heading))]
        layout = [(section.heading, *_layout(section)) for section in body]
        start = _next_section(conn, paper_id)
        text_no = _new_text(conn, paper_id, EXTENDED_PREFIX + source, parsed.version, parser)
        _insert_sections(conn, paper_id, text_no, start, layout)
    return sum(len(sentences) for _, _, sentences in layout)


def _current_text(conn: sqlite3.Connection, paper_id: str) -> int | None:
    return conn.execute("SELECT MAX(text_no) FROM paper_texts WHERE paper_id = ?", (paper_id,)).fetchone()[0]


def _next_section(conn: sqlite3.Connection, paper_id: str) -> int:
    return conn.execute("SELECT COALESCE(MAX(idx), -1) + 1 FROM sections WHERE paper_id = ?", (paper_id,)).fetchone()[0]


def _new_text(conn: sqlite3.Connection, paper_id: str, source: str, version: str | None, parser: str) -> int:
    text_no = (_current_text(conn, paper_id) or 0) + 1
    conn.execute(
        "INSERT INTO paper_texts (paper_id, text_no, source, version, parser, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (paper_id, text_no, source, version, parser, utc_now()),
    )
    _set_paper_source(conn, paper_id, source, version, parser)
    return text_no


def _set_paper_source(conn: sqlite3.Connection, paper_id: str, source: str, version: str | None, parser: str) -> None:
    """The paper row describes its current text."""
    conn.execute(
        "UPDATE papers SET text_source = ?, text_version = ?, parser = ? WHERE paper_id = ?",
        (source, version, parser, paper_id),
    )


def _insert_sections(conn: sqlite3.Connection, paper_id: str, text_no: int, start: int, layout: Layout) -> None:
    for s_idx, (heading, text, sentences) in enumerate(layout, start=start):
        sid = section_id(paper_id, s_idx)
        conn.execute(
            "INSERT INTO sections (section_id, paper_id, idx, heading, text, text_no) VALUES (?, ?, ?, ?, ?, ?)",
            (sid, paper_id, s_idx, heading, text, text_no),
        )
        conn.executemany(
            "INSERT INTO sentences (sentence_id, section_id, idx, text, char_start, char_end)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [(sentence_id(paper_id, s_idx, n), sid, n, sentence, begin, end)
             for n, (sentence, begin, end) in enumerate(sentences)],
        )


def _stored_layout(conn: sqlite3.Connection, paper_id: str) -> Layout:
    """The paper's current sections in the same shape as ``_layout`` gives them."""
    layout: Layout = []
    for section in conn.execute(
        "SELECT section_id, heading, text FROM sections WHERE paper_id = ? AND retired_by IS NULL ORDER BY idx",
        (paper_id,),
    ).fetchall():
        sentences = conn.execute(
            "SELECT text, char_start, char_end FROM sentences WHERE section_id = ? ORDER BY idx",
            (section["section_id"],),
        ).fetchall()
        layout.append((section["heading"], section["text"], [tuple(row) for row in sentences]))
    return layout


def _layout(section: ParsedSection) -> tuple[str, list[tuple[str, int, int]]]:
    """The section text (paragraphs separated by blank lines) and its sentences with spans."""
    text = "\n\n".join(section.paragraphs)
    sentences: list[tuple[str, int, int]] = []
    offset = 0
    for paragraph in section.paragraphs:
        for start, end in split_spans(paragraph):
            sentences.append((paragraph[start:end], offset + start, offset + end))
        offset += len(paragraph) + 2
    return text, sentences


def abstract_text(title: str, abstract: str | None) -> ParsedText | None:
    """The abstract as a one-section text, for papers read by abstract only."""
    if not abstract:
        return None
    return ParsedText(title=title, sections=[ParsedSection("Abstract", [abstract])])
