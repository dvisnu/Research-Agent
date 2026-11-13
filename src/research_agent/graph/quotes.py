r"""The quote check: code confirms that every quoted span exists in the paper.

Each piece of evidence names a sentence and quotes it. The quote passes when,
after normalising Unicode, dashes, quotation marks, whitespace and case, it
appears in that sentence; an ellipsis in a quote may skip words. Backslashes
are ignored on both sides: some papers' text keeps LaTeX ("\Delta", "10\%"),
and models copy it back with the backslashes doubled, dropped, spelt out as
"\textbackslash", or read as JSON escapes ("\times" arriving as a tab and
"imes"). Zero-width characters are ignored too. Any other change of wording fails.

A quote found in another sentence of the text the model was shown is moved to
that sentence ("readdressed"). Anything else fails and its evidence is dropped;
an item left with no evidence is dropped too. Missing items stay missing:
nothing is filled in.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

MIN_QUOTE_CHARS = 3  # letters or digits; shorter quotes prove nothing

_TRANSLATE = str.maketrans({
    "\u00ad": None,  # soft hyphen
    "\u200b": None, "\u200c": None, "\u200d": None, "\u2060": None, "\ufeff": None,  # zero-width
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u2032": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2033": '"',
})
# What JSON escapes make of the LaTeX a model copies: "\times" arrives as a tab and "imes".
_ESCAPES = str.maketrans({"\t": "\\t", "\r": "\\r", "\f": "\\f", "\b": "\\b", "\v": "\\v"})
_BACKSLASHES = re.compile(r"\\textbackslash\s*|\\+")
_ELLIPSIS = re.compile(r"\s*(?:\.\.\.|\[\s*\.\.\.\s*\])\s*")
_EDGE = " \"'`.,;:"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_TRANSLATE)
    return " ".join(_BACKSLASHES.sub("", text).split()).casefold()


def _fragments(quote: str) -> list[str]:
    """The quote's parts between ellipses, trimmed of the quotation marks and end
    punctuation models add."""
    parts = _ELLIPSIS.split(normalize(quote.translate(_ESCAPES)))
    return [part for part in (p.strip(_EDGE) for p in parts) if part]


def _contains(sentence: str, fragments: list[str]) -> bool:
    position = 0
    for fragment in fragments:
        found = sentence.find(fragment, position)
        if found < 0:
            return False
        position = found + len(fragment)
    return True


@dataclass
class QuoteStats:
    quotes: int = 0
    exact: int = 0  # found in the sentence it cites
    readdressed: int = 0  # found in another sentence the model was shown
    failed: int = 0
    items: int = 0
    items_kept: int = 0

    def add(self, other: "QuoteStats") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    @property
    def pass_rate(self) -> float | None:
        return (self.exact + self.readdressed) / self.quotes if self.quotes else None

    def as_dict(self) -> dict[str, Any]:
        rate = self.pass_rate
        return {**self.__dict__, "pass_rate": None if rate is None else round(rate, 4)}


@dataclass
class CheckResult:
    output: dict[str, Any]  # the reply with failed evidence and unsupported items removed
    stats: QuoteStats
    dropped: list[dict[str, str]] = field(default_factory=list)  # {"ref", "reason"}


class QuoteChecker:
    """Checks the evidence in one reply against the sentences the model was shown."""

    def __init__(self, sentences: dict[str, str]) -> None:
        self.sentences = {sid: normalize(text) for sid, text in sentences.items()}  # in reading order

    def check(self, output: dict[str, Any]) -> CheckResult:
        stats, dropped = QuoteStats(), []
        checked: dict[str, Any] = {}
        for key, value in output.items():
            if isinstance(value, list):
                checked[key] = [kept for item in value if (kept := self._item(key, item, stats, dropped))]
            elif isinstance(value, dict) and key == "facets":
                checked[key] = {
                    name: self._item(f"facets.{name}", item, stats, dropped) if item else None
                    for name, item in value.items()
                }
            else:
                checked[key] = value
        return CheckResult(checked, stats, dropped)

    def _item(
        self, field_name: str, item: dict[str, Any], stats: QuoteStats, dropped: list[dict[str, str]]
    ) -> dict[str, Any] | None:
        stats.items += 1
        label = f"{field_name}: {item.get('name') or item.get('text') or ''}"[:200]
        kept = []
        for evidence in item.get("evidence", []):
            stats.quotes += 1
            outcome, sentence_id = self.locate(evidence["sentence"], evidence["quote"])
            if outcome == "failed":
                stats.failed += 1
                dropped.append({"ref": f"quote:{evidence['sentence']}", "reason": f"{label} | quote not found: {evidence['quote'][:150]}"})
                continue
            setattr(stats, outcome, getattr(stats, outcome) + 1)
            if all(e["sentence"] != sentence_id for e in kept):
                kept.append({"sentence": sentence_id, "quote": evidence["quote"], "check": outcome})
        if not kept:
            dropped.append({"ref": f"item:{field_name}", "reason": f"{label} | no evidence passed the quote check"})
            return None
        stats.items_kept += 1
        return {**item, "evidence": kept}

    def locate(self, sentence_id: str, quote: str) -> tuple[str, str]:
        """("exact" | "readdressed" | "failed", the sentence the quote is in)."""
        fragments = _fragments(quote)
        if sum(ch.isalnum() for fragment in fragments for ch in fragment) < MIN_QUOTE_CHARS:
            return "failed", sentence_id
        cited = self.sentences.get(sentence_id)
        if cited is not None and _contains(cited, fragments):
            return "exact", sentence_id
        for other, text in self.sentences.items():
            if other != sentence_id and _contains(text, fragments):
                return "readdressed", other
        return "failed", sentence_id
