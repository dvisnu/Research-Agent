"""Turn a paper's full text into sections of paragraphs.

Two inputs: arXiv's HTML (LaTeXML output, the preferred source: real section
structure, LaTeX source for math) and PDFs via PyMuPDF (the fallback: headings
are guessed and math comes out as loose glyphs).
"""

import re
from collections import Counter
from dataclasses import dataclass, field

import pymupdf
from bs4 import BeautifulSoup, Tag


@dataclass
class ParsedSection:
    heading: str | None
    paragraphs: list[str] = field(default_factory=list)


@dataclass
class ParsedText:
    title: str | None
    sections: list[ParsedSection]
    version: str | None = None  # e.g. "v2" for arXiv HTML
    arxiv_id: str | None = None  # from an arXiv PDF's margin stamp


class ParseError(ValueError):
    pass


def _clean(text: str) -> str:
    return " ".join(text.split())


# arXiv HTML -----------------------------------------------------------------

# Elements that each become a section: \section, \subsection, ..., \paragraph, appendices.
_UNITS = {
    "ltx_abstract",
    "ltx_section",
    "ltx_subsection",
    "ltx_subsubsection",
    "ltx_paragraph",
    "ltx_appendix",
}
# Removed before reading: footnotes and thanks (they would land mid-sentence),
# the bibliography, and page furniture.
_DROP = ".ltx_note, .ltx_bibliography, .ltx_TOC, .ltx_page_footer, .ltx_authors, .ltx_dates, nav, script, style"


def parse_arxiv_html(html: str) -> ParsedText:
    soup = BeautifulSoup(html, "lxml")
    article = soup.find("article", class_="ltx_document")
    if not isinstance(article, Tag):
        raise ParseError("no LaTeXML article in the page; arXiv has no HTML for this paper")

    version = re.search(r"/abs/[^\"'\s]+?(v\d+)[\"']", html)  # the page links to its abs page, e.g. /abs/2106.09685v2
    for element in article.select(_DROP):
        element.decompose()
    for math in article.find_all("math"):
        math.replace_with(math.get("alttext") or "")
    for figure in article.find_all("figure"):
        caption = figure.find("figcaption")
        if caption:  # keep the caption as a paragraph; drop the figure or table body
            paragraph = soup.new_tag("div", attrs={"class": "ltx_para"})
            paragraph.string = caption.get_text()
            figure.replace_with(paragraph)
        else:
            figure.decompose()

    title_tag = article.find(class_="ltx_title_document")
    sections: dict[int, ParsedSection] = {}
    order: list[int] = []
    for unit in article.find_all(class_=lambda c: c in _UNITS):
        heading = unit.find(class_="ltx_title", recursive=False) or unit.find(
            re.compile("^h[1-6]$"), recursive=False
        )
        sections[id(unit)] = ParsedSection(_clean(heading.get_text()) if heading else None)
        order.append(id(unit))
        if heading:
            heading.decompose()

    for paragraph in article.find_all(_is_paragraph):
        owner = next((p for p in paragraph.parents if _UNITS & set(p.get("class") or [])), None)
        text = _clean(paragraph.get_text())
        if owner is not None and text:
            sections[id(owner)].paragraphs.append(text)

    return ParsedText(
        title=_clean(title_tag.get_text()) if title_tag else None,
        sections=[sections[key] for key in order if sections[key].paragraphs],
        version=version.group(1) if version else None,
    )


def _is_paragraph(tag: Tag) -> bool:
    """An outermost ltx_para, or a bare <p class="ltx_p"> (abstracts have no ltx_para).
    Paragraphs nested in another, as in lists, are read as part of the outer one."""
    classes = tag.get("class") or []
    is_paragraph = "ltx_para" in classes or (tag.name == "p" and "ltx_p" in classes)
    return is_paragraph and tag.find_parent(class_="ltx_para") is None


# PDF --------------------------------------------------------------------------
#
# Read line by line, since a heading is often the first line of a text block.
# Headings come from the PDF's outline when it has one; otherwise a short line
# that is all bold, or larger than the body text, is a heading.

_KNOWN_HEADING = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*\.?|[A-Z]\.(?:\d+\.?)*|Appendix\s+[A-Z]\.?:?)\s+)?"
    r"(?:Abstract|Introduction|Background|Related Work|Preliminaries|Method(?:s|ology)?|"
    r"Approach|Experiments?|Experimental Setup|Results|Evaluation|Analysis|Discussion|"
    r"Limitations|Conclusions?|Future Work|Acknowledge?ments|References|Bibliography|Appendix)$",
    re.IGNORECASE,
)
_REFERENCES = re.compile(r"^(?:\d+\.?\s+)?(?:References|Bibliography)$", re.IGNORECASE)
# After the references, reading resumes at an appendix.
_APPENDIX = re.compile(r"^(?:Appendix|Appendices|Supplementary|[A-Z](?:\.\d+)*\.?\s+[A-Z])")
_CAPTION = re.compile(r"^(?:Table|Fig\.?|Figure|Algorithm)\s*\d", re.IGNORECASE)
# arXiv stamps its PDFs in the margin: "arXiv:2106.09685v2 [cs.CL] 16 Oct 2021".
_ARXIV_STAMP = re.compile(r"arXiv:(\d{4}\.\d{4,5})(v\d+)")
_SENTENCE_END = re.compile(r"[.!?:][\"')\]]*$")
_LIGATURES = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"})
MAX_HEADING_WORDS = 12
REPEATED_ON_PAGES = 3  # a short line on this many pages is a running header or footer


@dataclass
class _Line:
    block: tuple[int, int]  # (page, block number): lines of one block form a paragraph
    text: str
    size: float
    bold: bool


def parse_pdf(data: bytes) -> ParsedText:
    if not data.lstrip()[:5] == b"%PDF-":  # MuPDF would happily open HTML as a document
        raise ParseError("not a PDF")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ParseError(f"not a readable PDF: {exc}") from exc
    toc = {_heading_key(title) for _, title, _ in doc.get_toc()}
    lines = _read_lines(doc)
    arxiv_stamp = _ARXIV_STAMP.search(doc[0].get_text()) if len(doc) else None
    title = _clean(doc.metadata.get("title") or "") or _largest_line(lines)
    doc.close()
    if not lines:
        raise ParseError("no text found in the PDF")

    body_size = Counter(round(line.size * 2) / 2 for line in lines for _ in line.text).most_common(1)[0][0]
    pages_with = Counter()
    for text in {(line.block[0], line.text) for line in lines}:
        pages_with[text[1]] += 1

    sections = [ParsedSection(None)]
    in_references = False
    current_block = None
    for line in lines:
        repeated = pages_with[line.text] >= REPEATED_ON_PAGES and len(line.text) < 120
        if repeated or _ARXIV_STAMP.match(line.text) or not re.search("[A-Za-z]", line.text):
            continue  # running headers and footers, arXiv's stamp, page numbers, stray numbers
        if _is_heading(line, toc, body_size):
            if in_references and not _APPENDIX.match(line.text):
                continue
            in_references = bool(_REFERENCES.match(line.text))
            if not in_references:
                sections.append(ParsedSection(line.text))
            current_block = None
            continue
        if in_references:
            continue
        paragraphs = sections[-1].paragraphs
        if line.block == current_block:
            paragraphs[-1] = _join_lines(paragraphs[-1], line.text)
        elif paragraphs and not _SENTENCE_END.search(paragraphs[-1]) and line.text[:1].islower():
            paragraphs[-1] = _join_lines(paragraphs[-1], line.text)  # continues across a column or page
        else:
            paragraphs.append(line.text)
        current_block = line.block

    found = [s for s in sections if s.paragraphs]
    if not found:
        raise ParseError("no text found in the PDF")
    return ParsedText(
        title=title,
        sections=found,
        version=arxiv_stamp.group(2) if arxiv_stamp else None,
        arxiv_id=arxiv_stamp.group(1) if arxiv_stamp else None,
    )


def _largest_line(lines: list[_Line]) -> str | None:
    """The title guess: the largest text on the first page, skipping arXiv's margin stamp."""
    first_page = [line for line in lines if line.block[0] == 0 and not _ARXIV_STAMP.search(line.text)]
    return max(first_page, key=lambda line: line.size).text if first_page else None


def _read_lines(doc: pymupdf.Document) -> list[_Line]:
    lines = []
    for page_number, page in enumerate(doc):
        for block_number, block in enumerate(page.get_text("dict")["blocks"]):
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                # Spaces often come as spans of their own: keep them in the text,
                # but judge size and boldness by the spans with letters in them.
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                lines.append(
                    _Line(
                        block=(page_number, block_number),
                        text=_clean("".join(s["text"] for s in line["spans"]).translate(_LIGATURES)),
                        size=max(s["size"] for s in spans),
                        bold=all(s["flags"] & 16 or "bold" in s["font"].lower() for s in spans),
                    )
                )
    return lines


def _heading_key(text: str) -> str:
    """Heading text without numbering, case or punctuation, for matching the outline."""
    return re.sub(r"[^a-z]", "", re.sub(r"^[\dA-Z]?(?:\.\d+)*\.?\s+", "", _clean(text)).lower())


def _is_heading(line: _Line, toc: set[str], body_size: float) -> bool:
    words = len(line.text.split())
    if words > MAX_HEADING_WORDS or _CAPTION.match(line.text):
        return False
    if toc:
        key = _heading_key(line.text)
        return bool(key) and (key in toc or bool(_REFERENCES.match(line.text)))
    if line.size < body_size - 0.25:
        return False
    strong = line.bold or line.size >= body_size + 1
    return (strong and not line.text.endswith((",", ";"))) or bool(_KNOWN_HEADING.match(line.text))


def _join_lines(text: str, line: str) -> str:
    """Join a line onto a paragraph, undoing a hyphenated line break."""
    if text.endswith("-") and line[:1].islower():
        return text[:-1] + line
    return f"{text} {line}"
