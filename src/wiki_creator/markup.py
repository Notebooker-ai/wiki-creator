"""Pure helpers for the wiki's inline marker syntax.

The article-writing LLM emits two kinds of inline markers:

* Citations: ``[[cite:s3]]`` where ``s3`` is a source marker from the block set
  sent to the model. Valid markers are rewritten to Obsidian footnote refs
  (``[^s3]``); hallucinated markers are dropped.
* Wikilinks: ``[[Topic Title]]`` or ``[[Topic Title|displayed text]]``,
  rewritten to canonical ``[[Stem]]`` / ``[[Stem|anchor]]`` once resolved to an
  emitted filename stem; unresolved links degrade to their plain anchor text.

Ported from the Laravel app's ``App\\Support\\WikiMarkup`` (same regexes and
drop-on-unknown semantics).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, List, Optional, Tuple

CITE_RE = re.compile(r"\[\[cite:([^\]]+?)\]\]")
WIKILINK_RE = re.compile(r"\[\[(?!cite:)([^\]|]+?)(?:\|([^\]]+?))?\]\]")


def slug(title: str) -> str:
    """URL-safe, lowercase, hyphenated slug (equivalent of Laravel's Str::slug)."""
    normalized = unicodedata.normalize("NFKD", title)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"[^a-zA-Z0-9]+", "-", ascii_text)
    return ascii_text.strip("-").lower()


def parse_citations(body: str) -> List[str]:
    """Citation tokens in order of appearance (with duplicates)."""
    return CITE_RE.findall(body)


def parse_wikilinks(body: str) -> List[Tuple[str, Optional[str]]]:
    """Wikilinks as ``(title, anchor-or-None)`` pairs in order of appearance."""
    out: List[Tuple[str, Optional[str]]] = []
    for m in WIKILINK_RE.finditer(body):
        title = m.group(1).strip()
        anchor = m.group(2).strip() if m.group(2) is not None else None
        out.append((title, anchor))
    return out


def rewrite_citations(body: str, mapping: dict) -> str:
    """Rewrite/drop citation markers.

    ``mapping`` maps each token (e.g. ``"s3"``) to its full replacement text
    (e.g. ``"[^s3]"``) or ``None`` to drop it. Tokens absent from the mapping
    are treated as hallucinated and dropped.
    """

    def _replace(m: "re.Match[str]") -> str:
        replacement = mapping.get(m.group(1))
        return "" if replacement is None else replacement

    return CITE_RE.sub(_replace, body)


def rewrite_wikilinks(body: str, resolver: Callable[[str], Optional[str]]) -> str:
    """Rewrite ``[[Title|anchor]]`` to canonical ``[[Stem]]`` / ``[[Stem|anchor]]``.

    ``resolver`` maps a link title to its target filename stem, or ``None`` to
    drop the brackets (leaving plain anchor text). When the anchor equals the
    stem the short form ``[[Stem]]`` is emitted.
    """

    def _replace(m: "re.Match[str]") -> str:
        title = m.group(1).strip()
        anchor = m.group(2).strip() if m.group(2) is not None else title
        stem = resolver(title)
        if stem is None:
            return anchor
        return f"[[{stem}]]" if anchor == stem else f"[[{stem}|{anchor}]]"

    return WIKILINK_RE.sub(_replace, body)
