"""Emit the wiki as Obsidian-ready markdown files plus a vault zip.

Layout per topic file::

    ---
    title: "Rosé Wine"
    tags: [wiki]
    created: 2026-07-19
    sources: 3
    ---
    Lead paragraph… [^s1]

    ## Section
    … see also [[Dive Bars]] …

    ## References

    [^s1]: [Bandol thread — Wine Forum](https://example.com/thread)
    [^s2]: My uploaded PDF

Bodies arrive fully resolved (wikilinks target other topics' filename stems, so
they resolve inside any Obsidian vault the files are dropped into). The
References block is derived here from each topic's ``sources`` — only markers
actually cited get a footnote definition (Obsidian warns on orphan defs).
"""

from __future__ import annotations

import datetime
import re
import zipfile
from pathlib import Path
from typing import Iterable, List

from open_notebook_creator_sdk import CreationFile
from open_notebook_creator_sdk.schemas import WikiTopicV1, WikiV1

# OS-forbidden chars plus the ones that break Obsidian wikilinks (# ^ [ ] |).
_FORBIDDEN_RE = re.compile(r'[/\\:*?"<>|#^\[\]]')

INDEX_FILENAME = "Wiki Index.md"


def sanitize_filename(name: str, fallback: str = "Untitled") -> str:
    """A safe Obsidian note name (no extension): strips forbidden characters,
    leading dots and trailing dots/spaces, collapses whitespace, caps length."""
    cleaned = _FORBIDDEN_RE.sub(" ", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.lstrip(".").rstrip(". ").strip()
    if len(cleaned) > 120:
        cleaned = cleaned[:120].rstrip(". ")
    return cleaned or fallback


def dedupe_stems(stems: Iterable[str]) -> List[str]:
    """Make stems unique (case-insensitive, since vaults land on
    case-insensitive filesystems) by suffixing " 2", " 3", …"""
    out: List[str] = []
    seen: dict = {}
    for stem in stems:
        key = stem.lower()
        n = seen.get(key, 0) + 1
        seen[key] = n
        if n == 1:
            out.append(stem)
            continue
        candidate = f"{stem} {n}"
        while candidate.lower() in seen:
            n += 1
            candidate = f"{stem} {n}"
        seen[candidate.lower()] = 1
        out.append(candidate)
    return out


def _yaml_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _link_label(title: str) -> str:
    """A source title safe inside a markdown link label."""
    return re.sub(r"\s+", " ", title).replace("[", "(").replace("]", ")").strip()


def _references_block(topic: WikiTopicV1) -> str:
    if not topic.sources:
        return ""
    lines = ["## References", ""]
    for ref in topic.sources:
        label = _link_label(ref.title) or "Untitled"
        if ref.url:
            lines.append(f"[^{ref.marker}]: [{label}]({ref.url})")
        else:
            lines.append(f"[^{ref.marker}]: {label}")
    return "\n".join(lines)


def render_topic_md(topic: WikiTopicV1, created: datetime.date) -> str:
    front = "\n".join(
        [
            "---",
            f"title: {_yaml_quote(topic.name)}",
            "tags: [wiki]",
            f"created: {created.isoformat()}",
            f"sources: {len(topic.sources)}",
            "---",
        ]
    )
    parts = [front, "", topic.body_markdown.strip()]
    references = _references_block(topic)
    if references:
        parts += ["", references]
    return "\n".join(parts) + "\n"


def render_index_md(wiki: WikiV1, created: datetime.date) -> str:
    front = "\n".join(
        [
            "---",
            f"title: {_yaml_quote(wiki.title)}",
            "tags: [wiki, moc]",
            f"created: {created.isoformat()}",
            "---",
        ]
    )
    lines = [front, "", f"# {wiki.title}", ""]
    if wiki.description:
        lines += [wiki.description, ""]
    lines.append("## Topics")
    lines.append("")
    for topic in wiki.topics:
        stem = Path(topic.filename).stem
        if topic.summary:
            lines.append(f"- [[{stem}]] — {topic.summary}")
        else:
            lines.append(f"- [[{stem}]]")
    return "\n".join(lines) + "\n"


def write_files(
    output_dir: Path, wiki: WikiV1, created: datetime.date
) -> List[CreationFile]:
    """Write every topic .md, the optional index note, and a vault zip into
    ``output_dir``. Returns the CreationFile list, zip first (primary download)."""
    written: List[str] = []

    for topic in wiki.topics:
        (output_dir / topic.filename).write_text(
            render_topic_md(topic, created), encoding="utf-8"
        )
        written.append(topic.filename)

    if wiki.index_filename:
        (output_dir / wiki.index_filename).write_text(
            render_index_md(wiki, created), encoding="utf-8"
        )
        written.append(wiki.index_filename)

    zip_name = f"{sanitize_filename(wiki.title, fallback='Wiki')}.zip"
    with zipfile.ZipFile(output_dir / zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename in written:
            zf.write(output_dir / filename, arcname=filename)

    files = [
        CreationFile(
            filename=zip_name,
            content_type="application/zip",
            path=zip_name,
            label="Obsidian vault (zip)",
        )
    ]
    for filename in written:
        files.append(
            CreationFile(
                filename=filename,
                content_type="text/markdown",
                path=filename,
                label=Path(filename).stem,
            )
        )
    return files
