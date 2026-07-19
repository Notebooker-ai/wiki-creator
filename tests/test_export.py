"""Obsidian file emission: sanitization, frontmatter, references, zip."""

from __future__ import annotations

import datetime
import zipfile
from pathlib import Path

from open_notebook_creator_sdk.schemas import WikiSourceRefV1, WikiTopicV1, WikiV1
from wiki_creator import export

_CREATED = datetime.date(2026, 7, 19)


def _topic(**overrides) -> WikiTopicV1:
    base = dict(
        name="Rosé Wine",
        slug="rose-wine",
        filename="Rosé Wine.md",
        summary="Pink wine made from red grapes.",
        body_markdown="Rosé is made by limiting skin contact.[^s1] See [[Dive Bars]].",
        sources=[
            WikiSourceRefV1(
                marker="s1", title="Bandol thread", url="https://example.com/bandol"
            ),
            WikiSourceRefV1(marker="s2", title="My uploaded PDF", url=None),
        ],
    )
    base.update(overrides)
    return WikiTopicV1(**base)


def test_sanitize_filename_strips_forbidden_and_obsidian_chars():
    assert export.sanitize_filename('a/b\\c:d*e?f"g<h>i|j') == "a b c d e f g h i j"
    assert export.sanitize_filename("Notes #1 [draft] ^2 | x") == "Notes 1 draft 2 x"
    assert export.sanitize_filename("...dots...") == "dots"
    assert export.sanitize_filename("   ") == "Untitled"
    assert len(export.sanitize_filename("x" * 500)) <= 120


def test_dedupe_stems_case_insensitive():
    assert export.dedupe_stems(["Rose", "rose", "Rose"]) == ["Rose", "rose 2", "Rose 3"]


def test_topic_md_layout():
    md = export.render_topic_md(_topic(), _CREATED)
    assert md.startswith('---\ntitle: "Rosé Wine"\ntags: [wiki]\ncreated: 2026-07-19\nsources: 2\n---\n')
    assert "[^s1]" in md
    assert "## References" in md
    assert "[^s1]: [Bandol thread](https://example.com/bandol)" in md
    assert "[^s2]: My uploaded PDF" in md


def test_index_md_links_topics_by_stem():
    wiki = WikiV1(
        title="My Wiki",
        description="What my sources say.",
        topics=[_topic()],
        index_filename="Wiki Index.md",
    )
    md = export.render_index_md(wiki, _CREATED)
    assert "tags: [wiki, moc]" in md
    assert "# My Wiki" in md
    assert "- [[Rosé Wine]] — Pink wine made from red grapes." in md


def test_write_files_emits_md_index_and_zip(tmp_path: Path):
    wiki = WikiV1(
        title="My Wiki: A/B",
        description=None,
        topics=[_topic(), _topic(name="Dive Bars", slug="dive-bars", filename="Dive Bars.md")],
        index_filename="Wiki Index.md",
    )
    files = export.write_files(tmp_path, wiki, _CREATED)

    assert (tmp_path / "Rosé Wine.md").is_file()
    assert (tmp_path / "Dive Bars.md").is_file()
    assert (tmp_path / "Wiki Index.md").is_file()

    # Zip is first (primary download), named from the sanitized wiki title.
    assert files[0].content_type == "application/zip"
    assert files[0].filename == "My Wiki A B.zip"
    with zipfile.ZipFile(tmp_path / files[0].filename) as zf:
        assert sorted(zf.namelist()) == ["Dive Bars.md", "Rosé Wine.md", "Wiki Index.md"]

    md_files = [f for f in files[1:]]
    assert {f.filename for f in md_files} == {"Rosé Wine.md", "Dive Bars.md", "Wiki Index.md"}
    assert all(f.content_type == "text/markdown" for f in md_files)
    # Paths must be relative (host requires output_dir containment).
    assert all(not f.path.startswith("/") for f in files)


def test_references_omitted_when_no_sources():
    md = export.render_topic_md(_topic(sources=[]), _CREATED)
    assert "## References" not in md
