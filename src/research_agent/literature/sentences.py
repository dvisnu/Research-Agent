"""Sentence splitting for scientific text.

pysbd proposes the boundaries. A repair pass then rejoins the breaks it makes
after abbreviations common in papers ("Eq. 5", "Yang et al. (2016)", "Fig. 3")
and before a lowercase continuation.

Everything works on character spans of the paragraph, so every sentence is an
exact slice of it. pysbd sometimes alters text; when one of its pieces can't
be found in the paragraph, a simple punctuation splitter is used instead.
"""

import re
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)  # pysbd has invalid escapes on Python 3.12+
    import pysbd

_SEGMENTER = pysbd.Segmenter(language="en", clean=False)
_NO_BREAK_AFTER = re.compile(
    r"(?:\bet al|\be\.g|\bi\.e|\bcf|\bvs|\bresp|\bapprox|\bFigs?|\bTabs?|\bEqs?|\bSecs?|\bApps?|"
    r"\bAppx|\bRefs?|\bThm|\bLem|\bDef|\bProp|\bCor|\bAlg|\bCh|\bNo|\bNos|\bProc|\bVol|\bpp)\.$",
    re.IGNORECASE,
)
# Fallback: a sentence ends at . ! or ? (plus closing quotes or brackets) before
# whitespace and a capital, digit or opening bracket.
_SENTENCE_END = re.compile(r"[.!?][\"'”’)\]]*(?=\s+[A-Z0-9(\[“\"'])")


def split_spans(paragraph: str) -> list[tuple[int, int]]:
    """(start, end) of each sentence in ``paragraph``."""
    spans = _pysbd_spans(paragraph) or _punctuation_spans(paragraph)
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        previous = paragraph[merged[-1][0] : merged[-1][1]] if merged else ""
        if merged and (_NO_BREAK_AFTER.search(previous) or paragraph[start].islower()):
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def split_sentences(paragraph: str) -> list[str]:
    return [paragraph[start:end] for start, end in split_spans(paragraph)]


def _pysbd_spans(paragraph: str) -> list[tuple[int, int]] | None:
    spans = []
    cursor = 0
    for piece in _SEGMENTER.segment(paragraph):
        piece = piece.strip()
        if not piece:
            continue
        index = paragraph.find(piece, cursor)
        if index < 0:
            return None
        spans.append((index, index + len(piece)))
        cursor = index + len(piece)
    return spans


def _punctuation_spans(paragraph: str) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for match in _SENTENCE_END.finditer(paragraph):
        spans.append((start, match.end()))
        start = match.end()
        while start < len(paragraph) and paragraph[start].isspace():
            start += 1
    if start < len(paragraph.rstrip()):
        spans.append((start, len(paragraph.rstrip())))
    return spans
