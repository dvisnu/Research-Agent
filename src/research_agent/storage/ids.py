"""Identifiers, including the sentence address scheme.

Every sentence has an address of the form ``<paper_id>:s<section>:<sentence>``,
with sections and sentences numbered from 0 in reading order. Addresses are
positional, so they hold for one parse of a paper's text; the paper row records
which text and parser produced them (``text_source``, ``text_version``, ``parser``).
"""

import re
import uuid

_SENTENCE_RE = re.compile(r"^(?P<paper>[^:]+):s(?P<section>\d+):(?P<sentence>\d+)$")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def section_id(paper_id: str, section: int) -> str:
    return f"{paper_id}:s{section}"


def sentence_id(paper_id: str, section: int, sentence: int) -> str:
    return f"{paper_id}:s{section}:{sentence}"


def parse_sentence_id(address: str) -> tuple[str, int, int]:
    """Split a sentence address into (paper_id, section, sentence)."""
    match = _SENTENCE_RE.match(address)
    if match is None:
        raise ValueError(f"not a sentence address: {address!r}")
    return match["paper"], int(match["section"]), int(match["sentence"])
