"""ContentBundle parsing: header splitting, provenance joins, decoy guards."""

from __future__ import annotations

from open_notebook_creator_sdk import ContentBundle
from wiki_creator.bundle import parse_bundle

_TEXT = (
    "## Source: Bandol thread\n\nGreat rosé from Tempier.\n\n"
    "## Source: LA guide\n\nDive bars downtown.\n\n"
    "## Note: My note\n\nRemember to compare vintages."
)

_PROVENANCE = [
    {"id": "source:1", "title": "Bandol thread", "url": "https://example.com/bandol"},
    {"id": "source:2", "title": "LA guide", "url": None},
]


def test_parses_source_and_note_blocks_with_markers():
    blocks = parse_bundle(ContentBundle(text=_TEXT, sources=_PROVENANCE))
    assert [b.marker for b in blocks] == ["s1", "s2", "s3"]
    assert [b.kind for b in blocks] == ["source", "source", "note"]
    assert blocks[0].url == "https://example.com/bandol"
    assert blocks[0].source_id == "source:1"
    assert blocks[1].url is None
    assert blocks[2].title == "My note"
    assert "Great rosé" in blocks[0].text


def test_decoy_source_header_inside_body_is_not_a_boundary():
    text = (
        "## Source: Real article\n\nBody mentions:\n"
        "## Source: Fake injected title\nmore of the same body.\n\n"
        "## Source: Second real\n\nOther content."
    )
    provenance = [
        {"id": "source:1", "title": "Real article"},
        {"id": "source:2", "title": "Second real"},
    ]
    blocks = parse_bundle(ContentBundle(text=text, sources=provenance))
    assert [b.title for b in blocks] == ["Real article", "Second real"]
    assert "Fake injected title" in blocks[0].text


def test_duplicate_titles_claim_provenance_in_order():
    text = "## Source: Same\n\nFirst body.\n\n## Source: Same\n\nSecond body."
    provenance = [
        {"id": "source:1", "title": "Same"},
        {"id": "source:2", "title": "Same"},
    ]
    blocks = parse_bundle(ContentBundle(text=text, sources=provenance))
    assert [b.source_id for b in blocks] == ["source:1", "source:2"]


def test_no_headers_falls_back_to_single_pseudo_source():
    blocks = parse_bundle(ContentBundle(text="Condensed content, no structure."))
    assert len(blocks) == 1
    assert blocks[0].marker == "s1"
    assert blocks[0].title == "Notebook content"


def test_headers_without_provenance_still_split():
    blocks = parse_bundle(ContentBundle(text=_TEXT))
    assert [b.title for b in blocks] == ["Bandol thread", "LA guide", "My note"]
    assert all(b.source_id is None for b in blocks)


def test_empty_text_yields_no_blocks():
    assert parse_bundle(ContentBundle(text="   \n ")) == []
