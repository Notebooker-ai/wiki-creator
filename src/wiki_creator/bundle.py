"""Parse the host's assembled ContentBundle back into per-source blocks.

``Notebook.get_context()`` (open-notebook) joins per-source/per-note blocks as::

    ## Source: {title}

    {full_text}

    Insights:
    - ...

    ## Note: {title}

    {content}

This module splits the bundle text on those headers, assigns citation markers
``s1..sN`` in block order, and joins blocks to the bundle's ``sources``
provenance (``[{id, title, url}]``) so citations can carry titles/URLs.

Guards (both matter in practice):

* A scraped source body can itself contain a ``## Source:``-looking line, so a
  Source header only starts a new block when its title matches an unclaimed
  provenance entry (when provenance is available). Provenance is matched by
  title with order as the tiebreak — the host's context and provenance queries
  are separate and their ordering can drift.
* When no headers parse at all (explicit-content generation, or the host's
  condense-on-overflow retry which rewrites the text), the whole bundle becomes
  one pseudo-source so generation still works — citations just all point at
  "Notebook content".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from open_notebook_creator_sdk import ContentBundle

HEADER_RE = re.compile(r"^## (Source|Note): (.+)$", re.MULTILINE)

FALLBACK_TITLE = "Notebook content"


@dataclass
class SourceBlock:
    marker: str
    title: str
    text: str
    kind: str  # "source" | "note"
    url: Optional[str] = None
    source_id: Optional[str] = None


def parse_bundle(content: ContentBundle) -> List[SourceBlock]:
    text = content.text or ""
    if not text.strip():
        return []

    provenance = [dict(s) for s in (content.sources or [])]
    have_provenance = bool(provenance)
    claimed = [False] * len(provenance)

    def _claim(title: str) -> Optional[dict]:
        for i, entry in enumerate(provenance):
            if not claimed[i] and (entry.get("title") or "Untitled") == title:
                claimed[i] = True
                return entry
        return None

    # Decide which header matches are real block boundaries.
    boundaries: List[tuple] = []  # (start, body_start, kind, title, prov)
    for m in HEADER_RE.finditer(text):
        kind = "source" if m.group(1) == "Source" else "note"
        title = m.group(2).strip()
        prov = None
        if kind == "source" and have_provenance:
            prov = _claim(title)
            if prov is None:
                # No unclaimed provenance entry with this title — treat as a
                # decoy line inside a scraped body, not a boundary.
                continue
        boundaries.append((m.start(), m.end(), kind, title, prov))

    blocks: List[SourceBlock] = []

    def _append(title: str, body: str, kind: str, prov: Optional[dict]) -> None:
        if not body.strip():
            return
        marker = f"s{len(blocks) + 1}"
        blocks.append(
            SourceBlock(
                marker=marker,
                title=title,
                text=body.strip(),
                kind=kind,
                url=(prov or {}).get("url"),
                source_id=(prov or {}).get("id"),
            )
        )

    if not boundaries:
        _append(FALLBACK_TITLE, text, "source", None)
        return blocks

    # Any non-blank preamble before the first header becomes its own block.
    preamble = text[: boundaries[0][0]]
    _append(FALLBACK_TITLE, preamble, "source", None)

    for i, (_, body_start, kind, title, prov) in enumerate(boundaries):
        body_end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        _append(title, text[body_start:body_end], kind, prov)

    return blocks
