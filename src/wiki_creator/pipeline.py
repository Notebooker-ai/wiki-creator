"""Two-stage wiki generation: topic extraction, then per-topic synthesis.

Ported from open-notebook's ``open_notebook/graphs/wiki.py`` (the account-wide
wiki's stateless generation helpers) and adapted to the creator SDK: source
content arrives as pre-parsed :class:`~wiki_creator.bundle.SourceBlock` items
instead of SurrealDB reads, and per-source incremental categorization becomes a
single global topic-extraction pass (a one-shot creator sees every source at
once, which yields a more coherent topic set).

Synthesis is NEVER truncated — a topic whose combined source text exceeds the
batch budget is MAP-REDUCED: each batch of sources is summarized into
citation-preserving notes, the notes are folded in further rounds while they
remain oversized, then the final article is reduced from them. This lets a
100+-source topic actually be read in full instead of overflowing one prompt's
context window.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from importlib import resources
from typing import List, Optional

from ai_prompter import Prompter
from open_notebook_creator_sdk import ModelRole

from .bundle import SourceBlock

# Topic extraction only needs a sample of each source to pick labels, so it
# reads a bounded excerpt per block. Synthesis map-reduces the FULL content.
CATEGORIZE_CHARS = 6_000
TOPIC_PLAN_MAX_TOKENS = 2_000
# Per-article output budget by configured length.
ARTICLE_MAX_TOKENS = {"concise": 2_000, "standard": 4_000, "comprehensive": 8_000}
# Map-reduce for large topics: ~10k tokens of source text per map batch.
BATCH_CHAR_BUDGET = 40_000
BATCH_NOTES_MAX_TOKENS = 1_800
REDUCE_MAX_TOKENS = 8_000
_BATCH_CONCURRENCY = 4


class TopicPlanError(Exception):
    """The topic-extraction response could not be parsed."""


@dataclass
class TopicPlan:
    name: str
    markers: List[str]


@dataclass
class WikiPlan:
    title: str
    description: Optional[str]
    topics: List[TopicPlan] = field(default_factory=list)


def _render_prompt(name: str, ctx: dict) -> str:
    template = resources.files("wiki_creator.prompts").joinpath(name).read_text()
    return Prompter(template_text=template).render(ctx)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()


def _content(resp: object) -> str:
    return resp.content if hasattr(resp, "content") else str(resp)


def _lede(body: str) -> str:
    """A short plain-text summary: first paragraph, markers/markdown stripped."""
    stripped = re.sub(r"\[\[cite:[^\]]+\]\]", "", body)
    stripped = re.sub(r"\[\^[^\]]+\]", "", stripped)
    for para in stripped.split("\n\n"):
        text = re.sub(r"[#*_`>\[\]]", "", para).strip()
        if text:
            return text[:300]
    return ""


def _labeled_block(block: SourceBlock) -> str:
    return f"[{block.marker}] {block.title}\n{block.text}"


def _split_oversized_block(block: str) -> List[str]:
    """Split one labeled block bigger than BATCH_CHAR_BUDGET into labeled parts,
    preserving the ``[marker] title`` label on each so citations still resolve and
    no content is dropped. Splits on paragraph/line boundaries where possible."""
    label, _, body = block.partition("\n")
    budget = max(1_000, BATCH_CHAR_BUDGET - len(label) - 16)
    parts: List[str] = []
    remaining = body
    while len(remaining) > budget:
        window = remaining[:budget]
        cut = window.rfind("\n\n")
        if cut < budget // 2:
            cut = window.rfind("\n")
        if cut < budget // 2:
            cut = budget
        parts.append(remaining[:cut])
        remaining = remaining[cut:]
    if remaining.strip():
        parts.append(remaining)

    total = len(parts)
    return [
        f"{label} (part {i}/{total})\n{chunk}" for i, chunk in enumerate(parts, 1)
    ]


def _batch_blocks(blocks: List[str]) -> List[List[str]]:
    """Group labeled blocks into batches that each stay under BATCH_CHAR_BUDGET. A
    single block bigger than the budget is split into labeled parts first, so the
    full content is processed without ever truncating a source."""
    units: List[str] = []
    for block in blocks:
        if len(block) > BATCH_CHAR_BUDGET:
            units.extend(_split_oversized_block(block))
        else:
            units.append(block)

    batches: List[List[str]] = []
    current: List[str] = []
    size = 0
    for unit in units:
        if current and size + len(unit) > BATCH_CHAR_BUDGET:
            batches.append(current)
            current, size = [], 0
        current.append(unit)
        size += len(unit)
    if current:
        batches.append(current)
    return batches


async def extract_topics(
    role: ModelRole,
    blocks: List[SourceBlock],
    max_topics: int,
    instructions: Optional[str] = None,
    language: Optional[str] = None,
) -> WikiPlan:
    """Stage 1: one structured pass over every source excerpt to pick the wiki
    title/description and assign each source to the topics it supports.

    Raises :class:`TopicPlanError` when the response isn't parseable JSON (the
    caller surfaces this as a retryable parse failure).
    """
    excerpts = "\n\n".join(
        f"[{b.marker}] {b.title}\n{b.text[:CATEGORIZE_CHARS]}" for b in blocks
    )
    prompt = _render_prompt(
        "topics.jinja",
        {
            "sources": excerpts,
            "max_topics": max_topics,
            "instructions": instructions,
            "language": language,
        },
    )
    llm = role.create_language(structured={"type": "json"}, max_tokens=TOPIC_PLAN_MAX_TOKENS)
    resp = await llm.ainvoke(prompt)
    raw = _content(resp)
    try:
        parsed = json.loads(_strip_fences(raw))
    except json.JSONDecodeError as e:
        raise TopicPlanError(f"invalid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise TopicPlanError("topic plan is not a JSON object")

    valid_markers = {b.marker for b in blocks}
    seen: set = set()
    topics: List[TopicPlan] = []
    for item in parsed.get("topics") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        key = name.lower()
        if not name or key in seen:
            continue
        markers_raw = item.get("source_markers") or []
        markers = [
            str(m).strip()
            for m in markers_raw
            if isinstance(m, (str, int)) and str(m).strip() in valid_markers
        ]
        # Preserve first-seen order, drop duplicate markers.
        markers = list(dict.fromkeys(markers))
        if not markers:
            continue
        seen.add(key)
        topics.append(TopicPlan(name=name, markers=markers))
        if len(topics) >= max_topics:
            break

    title = str(parsed.get("title") or "").strip() or "Wiki"
    description = str(parsed.get("description") or "").strip() or None
    return WikiPlan(title=title, description=description, topics=topics)


async def _summarize_batch(
    role: ModelRole,
    topic: str,
    blocks: List[str],
    semaphore: asyncio.Semaphore,
) -> str:
    """MAP step: distill one batch of source blocks into dense, specific notes that
    PRESERVE the ``[[cite:sN]]`` markers, so the reduce step can cite them."""
    async with semaphore:
        prompt = _render_prompt(
            "topic_batch_notes.jinja", {"topic": topic, "sources": "\n\n".join(blocks)}
        )
        llm = role.create_language(max_tokens=BATCH_NOTES_MAX_TOKENS)
        resp = await llm.ainvoke(prompt)
    return _content(resp)


async def synthesize_topic(
    role: ModelRole,
    topic: str,
    blocks: List[SourceBlock],
    other_topics: List[str],
    article_length: str = "standard",
) -> str:
    """Stage 2: write one topic article summarizing ALL its sources, citing them
    inline as ``[[cite:sN]]`` and cross-linking other wiki topics as
    ``[[Topic Name]]``. Returns the raw markdown body (markers unresolved).

    Small topics are synthesized in a single pass. Large topics are map-reduced
    with citation-preserving notes; oversized note sets are folded in further
    map rounds until the final reduce fits.
    """
    labeled = [_labeled_block(b) for b in blocks]
    if not labeled:
        return ""
    other = "\n".join(f"- {t}" for t in other_topics) or "(none)"
    article_max_tokens = ARTICLE_MAX_TOKENS.get(article_length, 4_000)
    combined = "\n\n".join(labeled)

    # Small topic — one pass over all the source text.
    if len(combined) <= BATCH_CHAR_BUDGET:
        prompt = _render_prompt(
            "topic_article.jinja",
            {"topic": topic, "sources": combined, "other_topics": other},
        )
        llm = role.create_language(max_tokens=article_max_tokens)
        resp = await llm.ainvoke(prompt)
        return _content(resp)

    # Large topic — MAP each batch into notes, then REDUCE into the article.
    # The semaphore is per-call (creators must be stateless across generate()s).
    semaphore = asyncio.Semaphore(_BATCH_CONCURRENCY)
    batches = _batch_blocks(labeled)
    notes = await asyncio.gather(
        *[_summarize_batch(role, topic, batch, semaphore) for batch in batches]
    )
    note_blocks = [
        f"## Source group {i + 1}\n{n}" for i, n in enumerate(notes) if n and n.strip()
    ]
    if not note_blocks:
        return ""

    # The reduce input is itself bounded: a topic with very many sources can
    # produce more batch notes than fit one prompt. Fold the notes in further
    # rounds — same map step on the notes — until they fit the batch budget.
    while len("\n\n".join(note_blocks)) > BATCH_CHAR_BUDGET and len(note_blocks) > 1:
        folded = await asyncio.gather(
            *[
                _summarize_batch(role, topic, batch, semaphore)
                for batch in _batch_blocks(note_blocks)
            ]
        )
        note_blocks = [n for n in folded if n and n.strip()]
        if not note_blocks:
            return ""

    prompt = _render_prompt(
        "topic_reduce.jinja",
        {"topic": topic, "notes": "\n\n".join(note_blocks), "other_topics": other},
    )
    llm = role.create_language(max_tokens=REDUCE_MAX_TOKENS)
    resp = await llm.ainvoke(prompt)
    return _content(resp)
