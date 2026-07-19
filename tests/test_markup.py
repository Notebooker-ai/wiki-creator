"""Marker-syntax helpers, ported semantics from the Laravel WikiMarkup tests."""

from __future__ import annotations

from wiki_creator import markup


def test_slug():
    assert markup.slug("Rosé Wine") == "rose-wine"
    assert markup.slug("  A  &  B  ") == "a-b"
    assert markup.slug("Dive Bars!") == "dive-bars"


def test_parse_citations_in_order_with_duplicates():
    body = "One. [[cite:s1]] Two. [[cite:s3]][[cite:s1]]"
    assert markup.parse_citations(body) == ["s1", "s3", "s1"]


def test_parse_wikilinks_pairs():
    body = "See [[Rosé Wine]] and [[Dive Bars|the bars]]. Not [[cite:s1]]."
    assert markup.parse_wikilinks(body) == [("Rosé Wine", None), ("Dive Bars", "the bars")]


def test_rewrite_citations_maps_and_drops_hallucinated():
    body = "A.[[cite:s1]] B.[[cite:s9]] C.[[cite:s2]][[cite:s1]]"
    out = markup.rewrite_citations(body, {"s1": "[^s1]", "s2": "[^s2]"})
    assert out == "A.[^s1] B. C.[^s2][^s1]"


def test_rewrite_citations_explicit_none_drops():
    assert markup.rewrite_citations("X.[[cite:s1]]", {"s1": None}) == "X."


def test_rewrite_wikilinks_resolved_and_unresolved():
    stems = {"rosé wine": "Rosé Wine", "dive bars": "Dive Bars"}
    out = markup.rewrite_wikilinks(
        "See [[Rosé Wine]], [[Dive Bars|the bars]], and [[Ghost Topic]].",
        lambda title: stems.get(title.lower()),
    )
    assert out == "See [[Rosé Wine]], [[Dive Bars|the bars]], and Ghost Topic."


def test_rewrite_wikilinks_anchor_equal_to_stem_collapses():
    out = markup.rewrite_wikilinks(
        "[[dive bars|Dive Bars]]", lambda _: "Dive Bars"
    )
    assert out == "[[Dive Bars]]"


def test_rewrite_wikilinks_ignores_cite_markers():
    out = markup.rewrite_wikilinks("[[cite:s1]] and [[T]]", lambda _: None)
    assert out == "[[cite:s1]] and T"
