"""Tile cache, provider table, and the OSM policy obligations.

No test here touches a real tile server. The fetcher is exercised against a
local ``http.server``, and everything else is pure arithmetic or filesystem
work.
"""

from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from rcta.gui.tiles import (TileCache, TileKey, TileProvider, USER_AGENT,
                            load_providers)


@pytest.fixture
def cache(tmp_path):
    return TileCache(tmp_path / "tiles", budget_bytes=10_000, ttl_days=1,
                     negative_ttl_hours=1)


KEY = TileKey(12, 943, 1670)


# ---- the policy ----------------------------------------------------------


def test_user_agent_names_the_application():
    """A generic agent string is explicitly forbidden by the OSM policy."""
    assert "RadiacodeTrackAnalyzer" in USER_AGENT
    for generic in ("python-requests", "Qt/", "urllib", "Mozilla/5.0"):
        assert generic not in USER_AGENT


def test_user_agent_carries_a_contact_url():
    assert "+http" in USER_AGENT


def test_the_placeholder_repository_url_must_be_replaced_before_release():
    """The cheapest possible guard against shipping a policy violation.

    Delete this test only when the URL below is a real repository that a tile
    server operator could actually use to contact somebody.
    """
    if "github.com/rcta/" in USER_AGENT:
        pytest.xfail("User-Agent still carries the placeholder repository URL")


def test_providers_load_and_osm_is_the_default():
    providers, default = load_providers()
    assert default == "osm"
    assert providers["osm"].attribution == "© OpenStreetMap contributors"


def test_every_provider_declares_attribution_and_a_policy_note():
    providers, _ = load_providers()
    for provider in providers.values():
        assert provider.attribution
        assert provider.attribution_export
        assert provider.policy_note


def test_esri_is_excluded_from_exports():
    """Its terms tie the service to an ArcGIS licence."""
    providers, _ = load_providers()
    assert providers["esri_imagery"].allow_export is False


# ---- the satellite layers -------------------------------------------------
#
# Three are offered because no single one is both worldwide, high resolution
# and publishable. The whole point of carrying more than one is that the
# export flag differs between them, so these tests pin that difference rather
# than the mere presence of the entries.


SATELLITE = ("esri_imagery", "usgs_imagery", "s2cloudless")


@pytest.mark.parametrize("pid", SATELLITE)
def test_the_satellite_layers_are_all_present_and_usable(pid):
    providers, _ = load_providers()
    provider = providers[pid]
    assert provider.available
    assert "{z}" in provider.url


def test_at_least_one_satellite_layer_may_be_exported():
    """Otherwise a figure in a paper could not carry imagery at all."""
    providers, _ = load_providers()
    assert any(providers[p].allow_export for p in SATELLITE)


def test_the_exportable_satellite_layers_are_the_openly_licensed_ones():
    """USGS is public domain and Sentinel-2 cloudless is CC BY 4.0; Esri is
    neither, and the difference must not quietly erode."""
    providers, _ = load_providers()
    assert providers["usgs_imagery"].allow_export is True
    assert providers["s2cloudless"].allow_export is True
    assert providers["esri_imagery"].allow_export is False


def test_usgs_uses_the_arcgis_y_x_ordering():
    """Its URL is {z}/{y}/{x}, not the slippy-map {z}/{x}/{y}. Getting this
    backwards yields tiles that load but show the wrong place."""
    providers, _ = load_providers()
    assert providers["usgs_imagery"].tile_url(TileKey(16, 12397, 24574)) == (
        "https://basemap.nationalmap.gov/arcgis/rest/services/"
        "USGSImageryOnly/MapServer/tile/16/24574/12397")


@pytest.mark.parametrize("pid,zoom", [("usgs_imagery", 16), ("s2cloudless", 16)])
def test_the_satellite_zoom_caps_are_the_measured_ones(pid, zoom):
    """USGS returns 404 above 16; s2cloudless serves higher but upsamples."""
    providers, _ = load_providers()
    assert providers[pid].max_zoom == zoom


def test_a_keyed_provider_is_not_available_without_a_key():
    providers, _ = load_providers()
    carto = providers["carto_positron"]
    assert carto.api_key_setting
    assert not carto.available


def test_an_unsupported_schema_version_raises(tmp_path):
    path = tmp_path / "p.json"
    path.write_text('{"schema_version": 99, "providers": []}')
    with pytest.raises(ValueError, match="schema_version"):
        load_providers(path)


def test_tile_url_substitutes_all_three_coordinates():
    provider = TileProvider(id="t", name="t", url="https://e/{z}/{x}/{y}.png",
                            max_zoom=19, attribution="a", attribution_export="a")
    assert provider.tile_url(TileKey(3, 4, 5)) == "https://e/3/4/5.png"


def test_esri_url_uses_z_y_x_order():
    """Esri's REST endpoint swaps y and x; a naive substitution renders the
    wrong hemisphere."""
    providers, _ = load_providers()
    url = providers["esri_imagery"].tile_url(TileKey(3, 4, 5))
    assert url.endswith("/3/5/4")


# ---- the cache -----------------------------------------------------------


def test_write_then_read_round_trips(cache):
    cache.write("osm", KEY, b"\x89PNG-data")
    assert cache.read("osm", KEY) == b"\x89PNG-data"


def test_a_missing_tile_reads_as_none(cache):
    assert cache.read("osm", KEY) is None


def test_providers_do_not_share_a_namespace(cache):
    cache.write("osm", KEY, b"a")
    assert cache.read("opentopo", KEY) is None


def test_an_expired_tile_is_not_served(cache):
    cache.write("osm", KEY, b"data")
    path = cache._path("osm", KEY)
    stale = time.time() - 2 * cache.ttl_s
    os.utime(path, (stale, stale))
    assert cache.read("osm", KEY) is None


def test_a_negative_entry_suppresses_refetching(cache):
    assert not cache.is_negative("osm", KEY)
    cache.write_negative("osm", KEY)
    assert cache.is_negative("osm", KEY)


def test_a_negative_entry_expires(cache):
    cache.write_negative("osm", KEY)
    path = cache._path("osm", KEY, ".404")
    stale = time.time() - 2 * cache.negative_ttl_s
    os.utime(path, (stale, stale))
    assert not cache.is_negative("osm", KEY)


def test_writes_are_atomic_and_leave_no_partial_files(cache):
    cache.write("osm", KEY, b"x" * 100)
    leftovers = list(cache.root.rglob("*.part"))
    assert leftovers == []


def test_sweep_evicts_oldest_first_until_inside_budget(cache):
    for i in range(20):
        key = TileKey(12, i, 0)
        cache.write("osm", key, b"x" * 1000)
        path = cache._path("osm", key)
        stamp = time.time() - (20 - i) * 10
        os.utime(path, (stamp, stamp))

    assert cache.size_bytes() == 20_000
    cache.sweep()
    assert cache.size_bytes() <= cache.budget_bytes
    # The newest survive; the oldest are gone.
    assert cache.read("osm", TileKey(12, 19, 0)) is not None
    assert cache.read("osm", TileKey(12, 0, 0)) is None


def test_sweep_removes_expired_entries_even_when_inside_budget(cache):
    cache.write("osm", KEY, b"x" * 10)
    path = cache._path("osm", KEY)
    stale = time.time() - 2 * cache.ttl_s
    os.utime(path, (stale, stale))
    removed, _ = cache.sweep()
    assert removed == 1


def test_sweep_on_an_empty_cache_is_harmless(cache):
    assert cache.sweep() == (0, 0)


def test_cache_dir_is_a_cache_location_not_application_data(monkeypatch):
    """Tiles are regenerable derived data: they must not be backed up, and OS
    cache cleaners must be able to reclaim them."""
    monkeypatch.delenv("RCTA_TILE_CACHE", raising=False)
    from rcta.gui.tiles import cache_dir
    path = str(cache_dir()).lower()
    assert "cache" in path
    assert "application support" not in path


# ---- the fetcher ---------------------------------------------------------


@pytest.fixture
def server(tmp_path):
    """A local tile server, so the suite never touches a real one."""
    import http.server
    import socketserver
    import threading

    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d494844520000000100000001080600000"
        "01f15c4890000000a49444154789c6360000002000100" "05fe02fe" "a7d5c1"
        "5f0000000049454e44ae426082")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if "missing" in self.path:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png)))
            self.end_headers()
            self.wfile.write(png)

        def log_message(self, *args):
            pass

    httpd = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def app():
    from PySide6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def fetchers():
    """Keep fetchers alive for the whole test and stop them afterwards.

    A fetcher collected mid-flight would leave replies pending; the production
    code disconnects safely, but leaking sockets between tests makes failures
    non-local and hard to read.
    """
    made = []
    yield made
    for fetcher in made:
        fetcher.cancel_all()


def _fetcher(app, cache, url, registry=None, **kwargs):
    from rcta.gui.tiles import TileFetcher
    provider = TileProvider(id="test", name="test", url=url + "/{z}/{x}/{y}.png",
                            max_zoom=19, attribution="test",
                            attribution_export="test")
    fetcher = TileFetcher(cache, provider, **kwargs)
    if registry is not None:
        registry.append(fetcher)
    return fetcher


def _spin(app, predicate, timeout_s=5.0):
    from PySide6 import QtCore
    deadline = time.monotonic() + timeout_s
    while not predicate() and time.monotonic() < deadline:
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
    return predicate()


def test_a_cached_tile_is_served_without_any_request(fetchers, app, cache):
    """Offline, with the tile already on disk: the map still works."""
    cache.write("test", KEY, b"cached-bytes")
    fetcher = _fetcher(app, cache, "http://127.0.0.1:1", fetchers, offline=True)
    seen = []
    fetcher.tileReady.connect(lambda k, d: seen.append((k, d)))
    fetcher.request([KEY])
    assert seen == [(KEY, b"cached-bytes")]


def test_offline_reports_once_not_once_per_tile(fetchers, app, cache):
    fetcher = _fetcher(app, cache, "http://127.0.0.1:1", fetchers, offline=True)
    messages = []
    fetcher.statusChanged.connect(
        lambda m, n: messages.append(m) if m else None)
    for _ in range(3):
        fetcher.request([TileKey(12, i, 0) for i in range(20)])
    assert len(messages) == 1
    assert "Offline" in messages[0]


def test_a_fetched_tile_is_delivered_and_cached(fetchers, app, cache, server):
    fetcher = _fetcher(app, cache, server, fetchers)
    seen = []
    fetcher.tileReady.connect(lambda k, d: seen.append(k))
    fetcher.request([KEY])
    assert _spin(app, lambda: bool(seen))
    assert cache.read("test", KEY) is not None


def test_a_404_writes_a_negative_entry(fetchers, app, cache, server):
    key = TileKey(1, 0, 0)
    from rcta.gui.tiles import TileFetcher
    provider = TileProvider(id="test", name="t",
                            url=server + "/missing/{z}/{x}/{y}.png",
                            max_zoom=19, attribution="t", attribution_export="t")
    fetcher = TileFetcher(cache, provider)
    fetchers.append(fetcher)
    failures = []
    fetcher.tileFailed.connect(lambda k, r: failures.append(r))
    fetcher.request([key])
    assert _spin(app, lambda: bool(failures))
    assert cache.is_negative("test", key)


def test_a_negative_entry_is_not_refetched(fetchers, app, cache, server):
    cache.write_negative("test", KEY)
    fetcher = _fetcher(app, cache, server, fetchers)
    seen = []
    fetcher.tileReady.connect(lambda k, d: seen.append(k))
    fetcher.request([KEY])
    app.processEvents()
    assert seen == []


def test_concurrency_is_capped(fetchers, app, cache, server):
    """The policy asks for restraint; four is the cap."""
    fetcher = _fetcher(app, cache, server, fetchers, max_concurrent=4)
    fetcher.request([TileKey(12, i, 0) for i in range(40)])
    assert len(fetcher._inflight) <= 4


def test_requesting_a_new_view_drops_the_old_queue(fetchers, app, cache, server):
    """Panning away must not keep fetching tiles nobody is looking at."""
    fetcher = _fetcher(app, cache, server, fetchers, max_concurrent=1)
    fetcher.request([TileKey(12, i, 0) for i in range(30)])
    fetcher.request([TileKey(12, 500, 500)])
    assert fetcher._wanted == {TileKey(12, 500, 500)}
    assert all(k == TileKey(12, 500, 500) for k in fetcher._queue)


def test_there_is_no_prefetch_api():
    """Bulk downloading is prohibited, so the capability does not exist.

    A prefetch method is the kind of thing that gets added for a plausible
    reason and quietly turns a desktop tool into a scraper.
    """
    from rcta.gui import tiles
    source = (tiles.__file__)
    with open(source) as handle:
        text = handle.read()
    assert "def prefetch" not in text
    assert "def download_area" not in text
    for name in dir(tiles.TileFetcher):
        assert "prefetch" not in name.lower()
