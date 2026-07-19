"""End-to-end WikiCreator tests with a stubbed language model (no network)."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

from open_notebook_creator_sdk import ContentBundle, CreationRequest, ModelRole
from open_notebook_creator_sdk.testing import (
    assert_creator_compliant,
    assert_result_compliant,
)
from wiki_creator import WikiCreator

_PLAN = {
    "title": "My Wiki",
    "description": "What my sources say.",
    "topics": [
        {"name": "Rosé Wine", "source_markers": ["s1", "s9999"]},
        {"name": "Dive Bars", "source_markers": ["s2"]},
    ],
}

# Article bodies keyed by topic (the fake looks the topic up in the prompt).
_ARTICLES = {
    "Rosé Wine": (
        "Rosé is made by limiting skin contact.[[cite:s1]] "
        "Pairs well with a night at [[Dive Bars|the bars]].[[cite:s1]][[cite:s77]] "
        "See also [[Ghost Topic]]."
    ),
    "Dive Bars": "Downtown has the best dives.[[cite:s2]] Sip [[Rosé Wine]] there.",
}

_TEXT = (
    "## Source: Bandol thread\n\nGreat rosé from Tempier.\n\n"
    "## Source: LA guide\n\nDive bars downtown."
)

_PROVENANCE = [
    {"id": "source:1", "title": "Bandol thread", "url": "https://example.com/bandol"},
    {"id": "source:2", "title": "LA guide", "url": None},
]


class _FakeLLM:
    def __init__(self, structured):
        self._structured = structured

    async def ainvoke(self, prompt):
        if self._structured:
            reply = json.dumps(_PLAN)
        else:
            # Dispatch on the "# TOPIC" prompt section — the topic name also
            # appears in every OTHER article's cross-link list.
            reply = next(
                (
                    body
                    for name, body in _ARTICLES.items()
                    if f"# TOPIC\n\n{name}" in prompt
                ),
                "",
            )

        class _Resp:
            content = reply

        return _Resp()


class _FakeRole(ModelRole):
    def create_language(self, **kwargs):
        return _FakeLLM(structured="structured" in kwargs)


def _request(td: str, **overrides) -> CreationRequest:
    base = dict(
        content=ContentBundle(text=_TEXT, sources=_PROVENANCE),
        config={},
        models={"text": _FakeRole(provider="f", model="f")},
        output_dir=td,
        artifact_id="a",
    )
    base.update(overrides)
    return CreationRequest(**base)


def test_static_compliance():
    assert_creator_compliant(WikiCreator())


async def test_generate_full_pipeline():
    creator = WikiCreator()
    with tempfile.TemporaryDirectory() as td:
        result = await creator.generate(_request(td))
        assert result.status == "SUCCESS"
        assert_result_compliant(creator, result)

        data = result.data
        assert data["title"] == "My Wiki"
        assert [t["name"] for t in data["topics"]] == ["Rosé Wine", "Dive Bars"]
        assert data["index_filename"] == "Wiki Index.md"

        rose = data["topics"][0]
        assert rose["slug"] == "rose-wine"
        assert rose["filename"] == "Rosé Wine.md"
        # Citations resolved to footnote refs; hallucinated s77 dropped.
        assert "[^s1]" in rose["body_markdown"]
        assert "[[cite:" not in rose["body_markdown"]
        assert "s77" not in rose["body_markdown"]
        # Hallucinated plan marker s9999 never reaches the source refs.
        assert [s["marker"] for s in rose["sources"]] == ["s1"]
        assert rose["sources"][0]["url"] == "https://example.com/bandol"
        # Cross-topic wikilink resolved to the other topic's filename stem;
        # unresolvable link degraded to plain anchor text.
        assert "[[Dive Bars|the bars]]" in rose["body_markdown"]
        assert "[[Ghost Topic]]" not in rose["body_markdown"]
        assert "Ghost Topic" in rose["body_markdown"]
        assert rose["summary"].startswith("Rosé is made")

        bars = data["topics"][1]
        assert "[[Rosé Wine]]" in bars["body_markdown"]


async def test_files_emitted(tmp_path: Path):
    creator = WikiCreator()
    result = await creator.generate(_request(str(tmp_path)))

    names = [f.filename for f in result.files]
    assert names[0] == "My Wiki.zip"
    assert set(names[1:]) == {"Rosé Wine.md", "Dive Bars.md", "Wiki Index.md"}

    with zipfile.ZipFile(tmp_path / "My Wiki.zip") as zf:
        assert sorted(zf.namelist()) == ["Dive Bars.md", "Rosé Wine.md", "Wiki Index.md"]

    rose_md = (tmp_path / "Rosé Wine.md").read_text()
    assert rose_md.startswith('---\ntitle: "Rosé Wine"\n')
    assert "## References" in rose_md
    assert "[^s1]: [Bandol thread](https://example.com/bandol)" in rose_md

    index_md = (tmp_path / "Wiki Index.md").read_text()
    assert "- [[Rosé Wine]]" in index_md
    assert "- [[Dive Bars]]" in index_md


async def test_index_note_can_be_disabled(tmp_path: Path):
    creator = WikiCreator()
    result = await creator.generate(
        _request(str(tmp_path), config={"include_index_note": False})
    )
    assert result.data["index_filename"] is None
    assert not (tmp_path / "Wiki Index.md").exists()


async def test_no_text_role_is_failure():
    creator = WikiCreator()
    with tempfile.TemporaryDirectory() as td:
        result = await creator.generate(_request(td, models={}))
        assert result.status == "FAILURE"
        assert result.errors[0].phase == "setup"


async def test_empty_content_is_failure():
    creator = WikiCreator()
    with tempfile.TemporaryDirectory() as td:
        result = await creator.generate(
            _request(td, content=ContentBundle(text="  "))
        )
        assert result.status == "FAILURE"
        assert result.errors[0].phase == "setup"


async def test_invalid_plan_json_is_retryable_parse_failure():
    class _BadRole(ModelRole):
        def create_language(self, **kwargs):
            class _LLM:
                async def ainvoke(self, _):
                    class _Resp:
                        content = "not json"

                    return _Resp()

            return _LLM()

    creator = WikiCreator()
    with tempfile.TemporaryDirectory() as td:
        result = await creator.generate(
            _request(td, models={"text": _BadRole(provider="f", model="f")})
        )
        assert result.status == "FAILURE"
        assert result.errors[0].phase == "parse"
        assert result.errors[0].retryable is True


async def test_partial_when_one_topic_produces_nothing():
    class _HalfRole(ModelRole):
        def create_language(self, **kwargs):
            structured = "structured" in kwargs

            class _LLM:
                async def ainvoke(self, prompt):
                    if structured:
                        reply = json.dumps(_PLAN)
                    elif "# TOPIC\n\nRosé Wine" in prompt:
                        reply = _ARTICLES["Rosé Wine"]
                    else:
                        reply = "   "

                    class _Resp:
                        content = reply

                    return _Resp()

            return _LLM()

    creator = WikiCreator()
    with tempfile.TemporaryDirectory() as td:
        result = await creator.generate(
            _request(td, models={"text": _HalfRole(provider="f", model="f")})
        )
        assert result.status == "PARTIAL"
        assert [t["name"] for t in result.data["topics"]] == ["Rosé Wine"]
        assert result.warnings
