"""The creator owns its UI: the manifest points at a shipped HTML view bundle."""

from __future__ import annotations

from importlib import resources

from wiki_creator import WikiCreator


def test_manifest_declares_view_bundle_and_it_ships():
    m = WikiCreator().manifest
    assert m.view is not None
    assert m.view.entry == "view/index.html"
    asset = resources.files("wiki_creator").joinpath(m.view.entry)
    assert asset.is_file()
    html = asset.read_text()
    # self-contained + speaks the host handshake + dispatches our schema
    assert "open-notebook:ready" in html
    assert "open-notebook:artifact" in html
    assert "wiki.v1" in html
    assert 'src="http' not in html  # no external scripts (sandbox-safe, offline)
    # sandbox safety: no history/location.hash navigation (throws in opaque origin)
    assert "location.hash" not in html
    assert "pushState" not in html
