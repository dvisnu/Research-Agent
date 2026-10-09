"""Store a paper's text as sections and addressed sentences."""

from research_agent.literature.parse import ParsedSection, ParsedText
from research_agent.literature.sentences import split_spans
from research_agent.storage import Database, section_id, sentence_id


def store_text(
    db: Database, paper_id: str, parsed: ParsedText, *, source: str, parser: str
) -> int:
    """Replace the paper's sections and sentences with ``parsed``. Returns the sentence count.

    Addresses are positional, so storing a new parse changes what they point at;
    the paper row records which text and parser they belong to.
    """
    count = 0
    with db.transaction() as conn:
        conn.execute("DELETE FROM sections WHERE paper_id = ?", (paper_id,))  # cascades to sentences
        for s_idx, section in enumerate(parsed.sections):
            text, sentences = _layout(section)
            sid = section_id(paper_id, s_idx)
            conn.execute(
                "INSERT INTO sections (section_id, paper_id, idx, heading, text) VALUES (?, ?, ?, ?, ?)",
                (sid, paper_id, s_idx, section.heading, text),
            )
            conn.executemany(
                "INSERT INTO sentences (sentence_id, section_id, idx, text, char_start, char_end)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (sentence_id(paper_id, s_idx, n), sid, n, sentence, start, end)
                    for n, (sentence, start, end) in enumerate(sentences)
                ],
            )
            count += len(sentences)
        conn.execute(
            "UPDATE papers SET text_source = ?, text_version = ?, parser = ? WHERE paper_id = ?",
            (source, parsed.version, parser, paper_id),
        )
    return count


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
