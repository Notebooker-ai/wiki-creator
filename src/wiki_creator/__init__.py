"""wiki-creator: an Open Notebook creator that distills a notebook's sources
into a set of interlinked, Wikipedia-style topic articles (emitted as
``wiki.v1``) and exports them as Obsidian-ready markdown files plus a vault zip.

Each topic's raw article body is also attached as an editable ``role="source"``
file (``source/<Topic>.md``) so the host can offer "edit → re-render":
:meth:`WikiCreator.render` folds the edited bodies back into the previous
``wiki.v1`` data and rebuilds the exports without touching an LLM.
"""

from __future__ import annotations

import asyncio
import datetime
from pathlib import Path
from typing import Callable, ClassVar, Dict, List, Literal, Optional

from loguru import logger
from open_notebook_creator_sdk import (
    BaseCreator,
    CreationError,
    CreationFile,
    CreationRequest,
    CreationResult,
    CreatorManifest,
    CreatorView,
    ModelRoleSpec,
    RenderRequest,
)
from open_notebook_creator_sdk.schemas import WikiSourceRefV1, WikiTopicV1, WikiV1
from pydantic import BaseModel, Field

from . import export, markup, pipeline
from .bundle import SourceBlock, parse_bundle

__version__ = "0.2.0"

_TOPIC_CONCURRENCY = 3
#: Subdirectory of ``output_dir`` holding one editable source file per topic.
SOURCE_DIR = "source"


class WikiConfig(BaseModel):
    """Per-generation config; its JSON Schema drives the host's generate form.

    Deliberately no ``count`` field: the host fans ``config.count`` out into N
    whole artifacts for any creator, and N full wiki+zip builds is never wanted.
    """

    max_topics: int = Field(
        default=8, ge=1, le=20, description="Upper bound on topic articles"
    )
    article_length: Literal["concise", "standard", "comprehensive"] = Field(
        default="standard", description="Depth of each topic article"
    )
    include_index_note: bool = Field(
        default=True,
        title="Index note",
        description="Also emit a 'Wiki Index.md' map-of-content note linking every topic",
    )


def _topic_stems(names: List[str]) -> List[str]:
    return export.dedupe_stems([export.sanitize_filename(n) for n in names])


def _unique_slugs(names: List[str]) -> List[str]:
    slugs: List[str] = []
    for name in names:
        base = markup.slug(name) or "topic"
        candidate, n = base, 1
        while candidate in slugs:
            n += 1
            candidate = f"{base}-{n}"
        slugs.append(candidate)
    return slugs


def _source_files(output_dir: Path, wiki: WikiV1) -> List[CreationFile]:
    """Write each topic's raw ``body_markdown`` (no front matter, no References
    block — the editable text) to ``output_dir/source/<topic.filename>`` and
    declare it as a ``role="source"`` file.

    Topic filenames are already unique within the wiki (``dedupe_stems``), so
    the source paths are too.
    """
    (output_dir / SOURCE_DIR).mkdir(parents=True, exist_ok=True)
    files: List[CreationFile] = []
    for topic in wiki.topics:
        rel = f"{SOURCE_DIR}/{topic.filename}"
        (output_dir / rel).write_text(topic.body_markdown, encoding="utf-8")
        files.append(
            CreationFile(
                filename=rel,
                content_type="text/markdown",
                path=rel,
                role="source",
                label=topic.name,
            )
        )
    return files


def _render_link_resolver(wiki: WikiV1) -> Callable[[str], Optional[str]]:
    """Resolver for wikilinks in a hand-edited body: a link whose title matches
    an existing topic's name or filename stem (case-insensitively) is
    canonicalised to that stem; anything else — including self-links — is left
    exactly as the user wrote it (generate-time drop-on-unknown would destroy
    deliberate edits)."""
    stem_by_key: Dict[str, str] = {}
    for topic in wiki.topics:
        stem = Path(topic.filename).stem
        stem_by_key.setdefault(stem.lower(), stem)
        stem_by_key.setdefault(topic.name.strip().lower(), stem)

    def _resolve(title: str) -> Optional[str]:
        title = title.strip()
        return stem_by_key.get(title.lower(), title)

    return _resolve


class WikiCreator(BaseCreator):
    config_model: ClassVar[type] = WikiConfig

    @property
    def manifest(self) -> CreatorManifest:
        return self.build_manifest(
            key="wikis",
            name="Wiki",
            version=__version__,
            description=(
                "Interlinked topic articles with citations, exported as "
                "Obsidian-ready markdown."
            ),
            sdk_compat=">=0.5,<1",
            emits=["wiki.v1"],
            model_roles=[
                ModelRoleSpec(
                    key="text",
                    kind="language",
                    requires=["structured_json"],
                    description="LLM that extracts topics and writes the articles.",
                )
            ],
            # NOT "book-open": textbook-creator already declares that, and the
            # host icon maps key off this name, so the two would render
            # identically side by side in the creator list.
            icon="library-big",
            view=CreatorView(entry="view/index.html"),
            suggestion_hint=(
                "how to split the content into interlinked articles, which topics "
                "deserve their own page, and how deep each should go"
            ),
            editable_source=True,
        )

    async def generate(self, request: CreationRequest) -> CreationResult:
        cfg = WikiConfig.model_validate(request.config)
        role = request.models.get("text")
        if role is None:
            return CreationResult(
                status="FAILURE",
                schema_id="wiki.v1",
                data={},
                errors=[CreationError(phase="setup", message="missing 'text' model role")],
                user_message="No language model was provided for wiki generation.",
            )

        blocks = parse_bundle(request.content)
        if not blocks:
            return CreationResult(
                status="FAILURE",
                schema_id="wiki.v1",
                data={},
                errors=[CreationError(phase="setup", message="notebook has no source content")],
                user_message="This notebook has no source content to build a wiki from.",
            )

        try:
            plan = await pipeline.extract_topics(
                role,
                blocks,
                max_topics=cfg.max_topics,
                instructions=request.instructions,
                language=request.language,
            )
        except pipeline.TopicPlanError as e:
            logger.error(f"wikis: topic extraction failed: {e}")
            return CreationResult(
                status="FAILURE",
                schema_id="wiki.v1",
                data={},
                errors=[CreationError(phase="parse", message=str(e), retryable=True)],
                user_message="The model returned an unparseable topic plan. Please retry.",
            )
        if not plan.topics:
            return CreationResult(
                status="FAILURE",
                schema_id="wiki.v1",
                data={},
                errors=[CreationError(phase="generate", message="no topics extracted")],
                user_message="No wiki topics could be extracted from this content.",
            )

        blocks_by_marker: Dict[str, SourceBlock] = {b.marker: b for b in blocks}
        warnings: List[str] = []
        # The semaphore is per-call (creators must be stateless across generate()s).
        semaphore = asyncio.Semaphore(_TOPIC_CONCURRENCY)

        async def _synthesize(topic: pipeline.TopicPlan) -> Optional[str]:
            topic_blocks = [blocks_by_marker[m] for m in topic.markers]
            other = [t.name for t in plan.topics if t.name != topic.name]
            async with semaphore:
                try:
                    body = await pipeline.synthesize_topic(
                        role, topic.name, topic_blocks, other, cfg.article_length
                    )
                except Exception as e:  # noqa: BLE001 - one topic must not sink the wiki
                    logger.error(f"wikis: synthesis failed for '{topic.name}': {e}")
                    warnings.append(f"Topic '{topic.name}' could not be written: {e}")
                    return None
            if not body.strip():
                warnings.append(f"Topic '{topic.name}' produced no article.")
                return None
            return body

        bodies = await asyncio.gather(*[_synthesize(t) for t in plan.topics])
        realized = [(t, b) for t, b in zip(plan.topics, bodies) if b]
        if not realized:
            return CreationResult(
                status="FAILURE",
                schema_id="wiki.v1",
                data={},
                errors=[CreationError(phase="generate", message="no articles produced")],
                warnings=warnings,
                user_message="No wiki articles could be generated from this content.",
            )

        names = [t.name for t, _ in realized]
        stems = _topic_stems(names)
        slugs = _unique_slugs(names)
        stem_by_name = {name.lower(): stem for name, stem in zip(names, stems)}

        topics: List[WikiTopicV1] = []
        for (topic, body), stem, topic_slug in zip(realized, stems, slugs):
            assigned = {m: f"[^{m}]" for m in topic.markers}
            cited = list(
                dict.fromkeys(
                    m for m in markup.parse_citations(body) if m in assigned
                )
            )
            resolved = markup.rewrite_citations(body, assigned)

            def _resolve(title: str, own_stem: str = stem) -> Optional[str]:
                target = stem_by_name.get(title.strip().lower())
                return None if target is None or target == own_stem else target

            resolved = markup.rewrite_wikilinks(resolved, _resolve).strip()
            topics.append(
                WikiTopicV1(
                    name=topic.name,
                    slug=topic_slug,
                    filename=f"{stem}.md",
                    summary=pipeline._lede(resolved) or None,
                    body_markdown=resolved,
                    sources=[
                        WikiSourceRefV1(
                            marker=m,
                            title=blocks_by_marker[m].title,
                            url=blocks_by_marker[m].url,
                            source_id=blocks_by_marker[m].source_id,
                        )
                        for m in cited
                    ],
                )
            )

        index_filename: Optional[str] = None
        if cfg.include_index_note:
            index_stem = export.dedupe_stems(
                stems + [Path(export.INDEX_FILENAME).stem]
            )[-1]
            index_filename = f"{index_stem}.md"

        wiki = WikiV1(
            title=plan.title,
            description=plan.description,
            topics=topics,
            index_filename=index_filename,
        )
        output_dir = Path(request.output_dir)
        files = export.write_files(output_dir, wiki, datetime.date.today())
        files += _source_files(output_dir, wiki)

        skipped = len(plan.topics) - len(realized)
        return CreationResult(
            status="PARTIAL" if skipped else "SUCCESS",
            schema_id="wiki.v1",
            data=wiki.model_dump(),
            files=files,
            warnings=warnings,
            user_message=(
                f"{skipped} topic(s) could not be written; the rest of the wiki was generated."
                if skipped
                else None
            ),
        )

    async def render(self, request: RenderRequest) -> CreationResult:
        """Rebuild the wiki from user-edited topic bodies. Deterministic: no LLM.

        Each ``source/<topic.filename>`` document replaces that topic's
        ``body_markdown`` (trailing whitespace stripped, wikilinks re-resolved
        against the existing topics, summary recomputed from the new lede).
        Topics without a matching source keep their previous body; sources that
        match no topic are ignored with a warning. Everything else in the
        previous data (title, slugs, filenames, citation sources) is carried
        forward, then the ``.md`` exports and vault zip are rewritten.
        """
        prev = WikiV1.model_validate(request.data)
        edits = {doc.filename: doc.content for doc in request.sources}
        known = {f"{SOURCE_DIR}/{t.filename}" for t in prev.topics}
        warnings = [
            f"Ignored source file '{name}': it does not match any wiki topic."
            for name in edits
            if name not in known
        ]

        resolve = _render_link_resolver(prev)
        topics: List[WikiTopicV1] = []
        for topic in prev.topics:
            edited = edits.get(f"{SOURCE_DIR}/{topic.filename}")
            if edited is None:
                topics.append(topic)
                continue
            body = markup.rewrite_wikilinks(edited, resolve).rstrip()
            topics.append(
                topic.model_copy(
                    update={"body_markdown": body, "summary": pipeline._lede(body) or None}
                )
            )

        wiki = WikiV1(
            title=prev.title,
            description=prev.description,
            topics=topics,
            index_filename=prev.index_filename,
        )
        output_dir = Path(request.output_dir)
        try:
            files = export.write_files(output_dir, wiki, datetime.date.today())
            files += _source_files(output_dir, wiki)
        except OSError as e:
            logger.error(f"wikis: render failed to write files: {e}")
            return CreationResult(
                status="FAILURE",
                schema_id="wiki.v1",
                data=prev.model_dump(),
                errors=[CreationError(phase="render", message=str(e))],
                user_message="The wiki files could not be rewritten.",
            )

        return CreationResult(
            status="SUCCESS",
            schema_id="wiki.v1",
            data=wiki.model_dump(),
            files=files,
            warnings=warnings,
        )
