"""End-to-end WikiCreator tests with a stubbed language model (no network)."""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

from open_notebook_creator_sdk import (
    ContentBundle,
    CreationRequest,
    ModelRole,
    RenderRequest,
    SourceDoc,
)
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

    names = [f.filename for f in result.files if f.role == "output"]
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


# --- editable source + render ---------------------------------------------


def _render_request(td: str, data: dict, sources: list) -> RenderRequest:
    return RenderRequest(
        sources=[
            SourceDoc(filename=name, content_type="text/markdown", content=body)
            for name, body in sources
        ],
        config={},
        data=data,
        output_dir=td,
        artifact_id="a",
    )


def test_manifest_declares_editable_source():
    assert WikiCreator().manifest.editable_source is True


async def test_generate_emits_one_source_file_per_topic(tmp_path: Path):
    creator = WikiCreator()
    result = await creator.generate(_request(str(tmp_path)))
    assert_result_compliant(creator, result)

    sources = [f for f in result.files if f.role == "source"]
    assert [f.filename for f in sources] == ["source/Rosé Wine.md", "source/Dive Bars.md"]
    assert all(f.path == f.filename for f in sources)
    assert all(f.content_type == "text/markdown" for f in sources)
    assert [f.label for f in sources] == ["Rosé Wine", "Dive Bars"]

    # The source is exactly the raw editable body: no front matter, no References.
    rose = result.data["topics"][0]
    on_disk = (tmp_path / "source" / "Rosé Wine.md").read_text(encoding="utf-8")
    assert on_disk == rose["body_markdown"]
    assert not on_disk.startswith("---")
    assert "## References" not in on_disk

    # Every export file is still a plain output, and the zip holds only outputs.
    outputs = [f for f in result.files if f.role == "output"]
    assert {f.filename for f in outputs} == {
        "My Wiki.zip", "Rosé Wine.md", "Dive Bars.md", "Wiki Index.md"
    }
    with zipfile.ZipFile(tmp_path / "My Wiki.zip") as zf:
        assert not any(n.startswith("source/") for n in zf.namelist())


async def test_render_applies_edited_body_and_rebuilds_exports(tmp_path: Path):
    creator = WikiCreator()
    generated = await creator.generate(_request(str(tmp_path)))
    edited = "Rosé is EDITED by hand.[^s1] Still pairs with [[Dive Bars]].\n\n"

    result = await creator.render(
        _render_request(
            str(tmp_path), generated.data, [("source/Rosé Wine.md", edited)]
        )
    )
    assert result.status == "SUCCESS"
    assert result.schema_id == "wiki.v1"
    assert_result_compliant(creator, result)

    rose = result.data["topics"][0]
    assert rose["body_markdown"] == edited.rstrip()
    assert rose["summary"].startswith("Rosé is EDITED by hand")
    # Unchanged metadata carried forward from the previous data.
    assert result.data["title"] == "My Wiki"
    assert rose["slug"] == "rose-wine"
    assert rose["filename"] == "Rosé Wine.md"
    assert [s["marker"] for s in rose["sources"]] == ["s1"]

    # The exported topic file was regenerated from the edit and still carries
    # the derived front matter + References block.
    rose_md = (tmp_path / "Rosé Wine.md").read_text(encoding="utf-8")
    assert rose_md.startswith('---\ntitle: "Rosé Wine"\n')
    assert "Rosé is EDITED by hand.[^s1]" in rose_md
    assert "[^s1]: [Bandol thread](https://example.com/bandol)" in rose_md
    with zipfile.ZipFile(tmp_path / "My Wiki.zip") as zf:
        assert "EDITED" in zf.read("Rosé Wine.md").decode("utf-8")
    # The index note reflects the recomputed summary.
    assert "Rosé is EDITED by hand" in (tmp_path / "Wiki Index.md").read_text()

    # Full file set: every export output plus the re-declared sources.
    names = [f.filename for f in result.files]
    assert names[0] == "My Wiki.zip"
    assert set(names) == {
        "My Wiki.zip", "Rosé Wine.md", "Dive Bars.md", "Wiki Index.md",
        "source/Rosé Wine.md", "source/Dive Bars.md",
    }
    sources = {f.filename: f for f in result.files if f.role == "source"}
    assert set(sources) == {"source/Rosé Wine.md", "source/Dive Bars.md"}
    assert sources["source/Rosé Wine.md"].label == "Rosé Wine"
    assert (tmp_path / "source" / "Rosé Wine.md").read_text() == edited.rstrip()


async def test_render_leaves_untouched_topics_unchanged(tmp_path: Path):
    creator = WikiCreator()
    generated = await creator.generate(_request(str(tmp_path)))
    before = generated.data["topics"][1]
    bars_md_before = (tmp_path / "Dive Bars.md").read_text(encoding="utf-8")

    result = await creator.render(
        _render_request(
            str(tmp_path), generated.data, [("source/Rosé Wine.md", "Changed.[^s1]")]
        )
    )
    assert result.data["topics"][1] == before
    assert (tmp_path / "Dive Bars.md").read_text(encoding="utf-8") == bars_md_before
    assert (tmp_path / "source" / "Dive Bars.md").read_text() == before["body_markdown"]


async def test_render_with_no_matching_sources_keeps_previous_bodies(tmp_path: Path):
    creator = WikiCreator()
    generated = await creator.generate(_request(str(tmp_path)))

    result = await creator.render(
        _render_request(
            str(tmp_path), generated.data, [("source/Nope.md", "stray edit")]
        )
    )
    assert result.status == "SUCCESS"
    assert result.data["topics"] == generated.data["topics"]
    assert result.warnings and "source/Nope.md" in result.warnings[0]
    assert {f.filename for f in result.files} == {f.filename for f in generated.files}

    empty = await creator.render(_render_request(str(tmp_path), generated.data, []))
    assert empty.status == "SUCCESS"
    assert empty.data == generated.data
    assert not empty.warnings


async def test_render_reresolves_wikilinks_against_existing_topics(tmp_path: Path):
    creator = WikiCreator()
    generated = await creator.generate(_request(str(tmp_path)))
    edited = (
        "Go to [[dive bars|the bars]] or [[Dive Bars]]; see [[Rosé Wine]] "
        "and [[Brand New Topic]].[^s1]"
    )

    result = await creator.render(
        _render_request(
            str(tmp_path), generated.data, [("source/Rosé Wine.md", edited)]
        )
    )
    body = result.data["topics"][0]["body_markdown"]
    # Case-insensitive name/stem matches canonicalise to the target stem…
    assert "[[Dive Bars|the bars]]" in body
    assert "[[Dive Bars]]" in body
    # …a self-link and an unknown link are left as the user wrote them.
    assert "[[Rosé Wine]]" in body
    assert "[[Brand New Topic]]" in body
