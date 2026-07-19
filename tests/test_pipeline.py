"""Pure map-reduce helpers ported from the original wiki graph."""

from __future__ import annotations

from wiki_creator import pipeline
from wiki_creator.bundle import SourceBlock


def test_lede_strips_markers_and_markdown():
    body = "**Rosé** is pink.[[cite:s1]][^s2]\n\n## More\ndetail"
    assert pipeline._lede(body) == "Rosé is pink."


def test_lede_skips_empty_paragraphs():
    assert pipeline._lede("\n\n\n> \n\nReal text here.") == "Real text here."


def test_split_oversized_block_preserves_labels_and_content():
    body = "para\n\n" * 30_000  # ~180k chars
    block = f"[s1] Big source\n{body}"
    parts = pipeline._split_oversized_block(block)
    assert len(parts) > 1
    assert all(p.startswith("[s1] Big source (part ") for p in parts)
    # No content dropped: total body length preserved (labels aside).
    rejoined = "".join(p.partition("\n")[2] for p in parts)
    assert rejoined.replace("\n", "") == body.replace("\n", "")


def test_batch_blocks_groups_under_budget():
    small = [f"[s{i}] T\n" + "x" * 15_000 for i in range(6)]
    batches = pipeline._batch_blocks(small)
    assert all(
        sum(len(u) for u in batch) <= pipeline.BATCH_CHAR_BUDGET for batch in batches
    )
    assert sum(len(b) for b in batches) == 6


def test_batch_blocks_splits_single_oversized_block():
    big = ["[s1] T\n" + ("y" * (pipeline.BATCH_CHAR_BUDGET * 2))]
    batches = pipeline._batch_blocks(big)
    assert len(batches) >= 2


async def test_synthesize_empty_blocks_returns_empty():
    class _Role:
        def create_language(self, **_):
            raise AssertionError("should not be called")

    out = await pipeline.synthesize_topic(_Role(), "T", [], [])
    assert out == ""


def _block(marker: str, text: str) -> SourceBlock:
    return SourceBlock(marker=marker, title=f"Title {marker}", text=text, kind="source")


class _CountingLLM:
    def __init__(self, log, reply):
        self._log = log
        self._reply = reply

    async def ainvoke(self, prompt):
        self._log.append(prompt)

        class _R:
            content = self._reply

        return _R()


async def test_large_topic_map_reduces():
    calls: list = []

    class _Role:
        def create_language(self, **kwargs):
            reply = "- note [[cite:s1]]" if kwargs.get("max_tokens") == pipeline.BATCH_NOTES_MAX_TOKENS else "Article body. [[cite:s1]]"
            return _CountingLLM(calls, reply)

    blocks = [
        _block("s1", "z" * 35_000),
        _block("s2", "z" * 35_000),
    ]
    out = await pipeline.synthesize_topic(_Role(), "Topic", blocks, ["Other"])
    assert out == "Article body. [[cite:s1]]"
    # Two map calls (one per batch) + one reduce call.
    assert len(calls) == 3
    assert "BEGIN NOTES" in calls[-1]
