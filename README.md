# wiki-creator

An [Open Notebook](https://github.com/Notebooker-ai/open-notebook) creator plugin
that distills a notebook's sources into a set of interlinked, Wikipedia-style
topic articles (emitted as `wiki.v1`) — and exports them as **Obsidian-ready
markdown**: one `.md` per topic with `[[wikilinks]]`, YAML frontmatter, footnote
citations back to the original sources, an index/MOC note, and a zip for
dropping straight into a vault.

## How it works

1. **Topic extraction** — one structured-JSON pass over the notebook's sources
   picks mid-level topics and assigns each source to the topics it supports.
2. **Article synthesis** — each topic's full source text is synthesized into a
   grounded article citing sources inline. Topics whose combined source text
   exceeds the batch budget are map-reduced (batch notes → folded → final
   article) so every source is read in full, never truncated.
3. **Linking + export** — citations become `[^sN]` footnotes, cross-topic
   mentions become `[[wikilinks]]` resolved against the emitted filenames, and
   everything is written as markdown files plus a vault zip.

The artifact's `data` carries the full article set, rendered in-app by the
self-contained `view/index.html` bundle (topic index → article pages with
navigable wikilinks). Display-only; no host API access.

## Development

```bash
uv sync --extra dev
uv run pytest -v
```
