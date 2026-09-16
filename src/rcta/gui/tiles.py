"""Slippy-map tile retrieval and an on-disk cache.

**The OpenStreetMap tile usage policy is a hard constraint on this design, not
a footnote.** The standard tile layer is donation-funded infrastructure, and
every clause of its policy maps to a specific mechanism here:

* a unique ``User-Agent`` naming this application (a test fails if the
  placeholder URL is ever shipped);
* a disk cache with a time-to-live well above the seven-day floor;
* **no bulk downloading** -- there is no prefetch, no lookahead ring, and no
  "download this area for offline use" feature anywhere in the program. The
  fetcher only ever accepts the currently visible tile set;
* a concurrency cap of four, and a debounce upstream so that a continuous pan
  issues *zero* requests until the gesture stops;
* attribution that is drawn inside the view and has no visibility toggle.

Retrieval uses ``QNetworkAccessManager`` rather than ``requests`` on a thread
pool, and the reasons are worth recording. It is already asynchronous on the Qt
event loop, so the entire map subsystem adds no threads to a codebase whose
defining property is that it has almost none. ``reply.abort()`` is real,
immediate cancellation, which a blocking ``requests.get`` on a ``QRunnable``
cannot offer -- you can only set a flag and discard the answer after the socket
finishes, which is exactly the behaviour the policy prohibits when the user has
panned away. And it adds no dependency.

Cancellation is layered, cheapest first: replacing the want-set drops queued
tiles before a socket is opened; in-flight replies older than a short grace
period are aborted (a *younger* one is left alone, or a quick pan-and-return
would abort and immediately re-request the same tile, which is the hammering
the policy forbids); and a late arrival is still written to the cache, because
the bytes were paid for and discarding them guarantees a re-request later.

The cache lives under ``QStandardPaths.CacheLocation``, deliberately not
``AppDataLocation``. Tiles are regenerable derived data: putting them in
application data means Time Machine backs up hundreds of megabytes of someone
else's map, and OS cache-cleaning tools that would otherwise reclaim the space
are locked out.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from PySide6 import QtCore, QtNetwork

from rcta import USER_AGENT
from rcta.gui import theme

__all__ = ["TileCache", "TileFetcher", "TileKey", "TileProvider",
           "USER_AGENT", "cache_dir", "load_providers"]


@dataclass(frozen=True)
class TileKey:
    z: int
    x: int
    y: int

    def parent(self, levels: int = 1) -> "TileKey":
        return TileKey(self.z - levels, self.x >> levels, self.y >> levels)


@dataclass(frozen=True)
class TileProvider:
    """A tile source and the terms attached to it."""

    id: str
    name: str
    url: str
    max_zoom: int
    attribution: str
    attribution_export: str
    allow_export: bool = True
    api_key_setting: str = ""
    policy_note: str = ""
    #: How far to desaturate and lift this layer before painting it under the
    #: data. None means the theme default, which is tuned for a cartographic
    #: basemap; satellite layers override it. See :mod:`rcta.gui.basemap`.
    desaturate: float | None = None
    lighten: float | None = None

    @property
    def available(self) -> bool:
        """False when the provider needs a key the program does not have."""
        return bool(self.url) and not self.api_key_setting

    def tile_url(self, key: TileKey) -> str:
        return (self.url.replace("{z}", str(key.z))
                .replace("{x}", str(key.x)).replace("{y}", str(key.y)))


def load_providers(path=None):
    """Read the provider table.

    The URLs live in data rather than in code because the OSM policy asks that
    tile URLs not be hardcoded, so that a server move does not require a
    software update.
    """
    if path is None:
        raw = resources.files("rcta.data").joinpath("providers.json").read_text()
    else:
        raw = Path(path).read_text()
    data = json.loads(raw)
    if data.get("schema_version") != 1:
        raise ValueError(
            f"unsupported providers schema_version {data.get('schema_version')}")
    providers = {p["id"]: TileProvider(**p) for p in data["providers"]}
    return providers, data.get("default", "osm")


def cache_dir() -> Path:
    """Where tiles live. Cache location, never application data."""
    override = os.environ.get("RCTA_TILE_CACHE")
    if override:
        return Path(override)
    base = QtCore.QStandardPaths.writableLocation(
        QtCore.QStandardPaths.StandardLocation.CacheLocation)
    return Path(base or Path.home() / ".cache" / "rcta") / "tiles"


class TileCache:
    """One file per tile on disk. No index database.

    File mtime is the fetch time, which makes both questions answerable with a
    single ``stat``: a tile is fresh if ``now - mtime < ttl``, and eviction
    deletes the oldest first. An atime-based LRU was rejected because most
    Linux filesystems mount ``noatime`` and would silently degrade it to random
    eviction.
    """

    def __init__(self, root, *, budget_bytes=theme.BASEMAP_DISK_BUDGET_BYTES,
                 ttl_days=theme.BASEMAP_TTL_DAYS,
                 negative_ttl_hours=theme.BASEMAP_NEGATIVE_TTL_HOURS) -> None:
        self.root = Path(root)
        self.budget_bytes = int(budget_bytes)
        self.ttl_s = float(ttl_days) * 86400.0
        self.negative_ttl_s = float(negative_ttl_hours) * 3600.0

    def _path(self, provider_id: str, key: TileKey, suffix: str = ".png") -> Path:
        return self.root / provider_id / str(key.z) / str(key.x) / f"{key.y}{suffix}"

    def read(self, provider_id: str, key: TileKey):
        path = self._path(provider_id, key)
        try:
            age = time.time() - path.stat().st_mtime
            if age > self.ttl_s:
                return None
            return path.read_bytes()
        except OSError:
            return None

    def is_negative(self, provider_id: str, key: TileKey) -> bool:
        path = self._path(provider_id, key, ".404")
        try:
            return (time.time() - path.stat().st_mtime) <= self.negative_ttl_s
        except OSError:
            return False

    def write(self, provider_id: str, key: TileKey, data: bytes) -> None:
        path = self._path(provider_id, key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename, so a crash cannot leave a truncated tile that
            # would be served as valid for the next thirty days.
            tmp = path.with_suffix(".part")
            tmp.write_bytes(data)
            os.replace(tmp, path)
        except OSError:
            pass

    def write_negative(self, provider_id: str, key: TileKey) -> None:
        path = self._path(provider_id, key, ".404")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
        except OSError:
            pass

    def sweep(self) -> tuple[int, int]:
        """Delete expired tiles, then the oldest until inside budget."""
        entries = []
        total = 0
        now = time.time()
        removed = freed = 0
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            ttl = self.negative_ttl_s if path.suffix == ".404" else self.ttl_s
            if now - stat.st_mtime > ttl:
                try:
                    path.unlink()
                    removed += 1
                    freed += stat.st_size
                except OSError:
                    pass
                continue
            entries.append((stat.st_mtime, stat.st_size, path))
            total += stat.st_size

        if total <= self.budget_bytes:
            return removed, freed
        for _, size, path in sorted(entries):
            try:
                path.unlink()
            except OSError:
                continue
            removed += 1
            freed += size
            total -= size
            if total <= self.budget_bytes:
                break
        return removed, freed

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*")
                   if p.is_file())


class TileFetcher(QtCore.QObject):
    """On-demand tile retrieval. Cache first, network second, never speculative."""

    tileReady = QtCore.Signal(object, object)      # TileKey, bytes
    tileFailed = QtCore.Signal(object, str)        # TileKey, reason
    statusChanged = QtCore.Signal(str, int)        # message, pending count

    def __init__(self, cache: TileCache, provider: TileProvider, parent=None, *,
                 offline: bool = False,
                 max_concurrent: int = theme.BASEMAP_MAX_CONCURRENT) -> None:
        super().__init__(parent)
        self._cache = cache
        self._provider = provider
        self._offline = bool(offline)
        self._max_concurrent = int(max_concurrent)

        self._manager = QtNetwork.QNetworkAccessManager(self)
        self._queue: list[TileKey] = []
        self._wanted: set[TileKey] = set()
        self._inflight: dict[TileKey, tuple] = {}
        #: Reverse map so the finished handler can be a *bound method* rather
        #: than a lambda closing over the reply. Qt tears a bound-method
        #: connection down automatically when the receiver is destroyed; a
        #: lambda has no receiver, so a reply outliving this object would
        #: deliver its signal into freed memory and segfault the process.
        self._reply_keys: dict[object, TileKey] = {}
        self._backoff_until = 0.0
        self._consecutive_429 = 0
        self._disabled = False
        self._reported_offline = False

    # ---- configuration ----------------------------------------------------

    def set_provider(self, provider: TileProvider) -> None:
        self.cancel_all()
        self._provider = provider
        self._disabled = False
        self._consecutive_429 = 0
        self._backoff_until = 0.0

    def set_offline(self, offline: bool) -> None:
        self._offline = bool(offline)
        if not offline:
            self._reported_offline = False

    @property
    def offline(self) -> bool:
        return self._offline

    @property
    def provider(self) -> TileProvider:
        return self._provider

    # ---- requests ---------------------------------------------------------

    def request(self, keys) -> None:
        """Replace the entire want-set with ``keys``, nearest-to-centre first.

        Replacing rather than appending is what makes cancellation correct: a
        tile that has scrolled off screen is simply absent from the next call
        and is dropped from the queue or aborted.
        """
        keys = [k if isinstance(k, TileKey) else TileKey(*k) for k in keys]
        self._wanted = set(keys)

        pending = []
        for key in keys:
            data = self._cache.read(self._provider.id, key)
            if data is not None:
                self.tileReady.emit(key, data)
                continue
            if self._cache.is_negative(self._provider.id, key):
                continue
            pending.append(key)

        self._queue = [k for k in pending if k not in self._inflight]
        self._abort_stale()
        self._pump()

    def cancel_all(self) -> None:
        self._queue.clear()
        self._wanted.clear()
        for reply, _ in list(self._inflight.values()):
            self._reply_keys.pop(reply, None)
            reply.abort()
        self._inflight.clear()
        self._reply_keys.clear()

    # ---- internals --------------------------------------------------------

    def _abort_stale(self) -> None:
        now = time.monotonic()
        for key, (reply, started) in list(self._inflight.items()):
            if key in self._wanted:
                continue
            # A young reply is left alone: aborting and re-requesting the same
            # tile on a pan-and-return is precisely the hammering the policy
            # asks us to avoid.
            if (now - started) * 1000.0 < theme.ABORT_GRACE_MS:
                continue
            reply.abort()

    def _pump(self) -> None:
        if self._offline or self._disabled or not self._provider.available:
            if self._offline and self._queue and not self._reported_offline:
                self._reported_offline = True
                self.statusChanged.emit("Offline - basemap from cache only", 0)
            self._queue.clear()
            return
        if time.monotonic() < self._backoff_until:
            return

        while self._queue and len(self._inflight) < self._max_concurrent:
            key = self._queue.pop(0)
            if key in self._inflight or key not in self._wanted:
                continue
            self._start(key)
        self.statusChanged.emit("", len(self._queue) + len(self._inflight))

    def _start(self, key: TileKey) -> None:
        request = QtNetwork.QNetworkRequest(QtCore.QUrl(self._provider.tile_url(key)))
        request.setHeader(QtNetwork.QNetworkRequest.KnownHeaders.UserAgentHeader,
                          USER_AGENT)
        request.setAttribute(
            QtNetwork.QNetworkRequest.Attribute.RedirectPolicyAttribute,
            QtNetwork.QNetworkRequest.RedirectPolicy.NoLessSafeRedirectPolicy)
        request.setTransferTimeout(15000)

        reply = self._manager.get(request)
        self._inflight[key] = (reply, time.monotonic())
        self._reply_keys[reply] = key
        reply.finished.connect(self._on_reply_finished)

    def _on_reply_finished(self) -> None:
        reply = self.sender()
        key = self._reply_keys.pop(reply, None)
        if key is None:
            return
        self._on_finished(key, reply)

    def _on_finished(self, key: TileKey, reply) -> None:
        self._inflight.pop(key, None)
        try:
            error = reply.error()
            status = reply.attribute(
                QtNetwork.QNetworkRequest.Attribute.HttpStatusCodeAttribute)
            data = bytes(reply.readAll())
        finally:
            reply.deleteLater()

        if error == QtNetwork.QNetworkReply.NetworkError.OperationCanceledError:
            self._pump()
            return

        if status in (404, 403):
            self._cache.write_negative(self._provider.id, key)
            self.tileFailed.emit(key, f"HTTP {status}")
            self._pump()
            return

        if status in (429, 503):
            self._on_rate_limited(reply)
            return

        if error != QtNetwork.QNetworkReply.NetworkError.NoError or not data:
            if error in (
                    QtNetwork.QNetworkReply.NetworkError.HostNotFoundError,
                    QtNetwork.QNetworkReply.NetworkError.UnknownNetworkError,
                    QtNetwork.QNetworkReply.NetworkError.TemporaryNetworkFailureError):
                self.set_offline(True)
                if not self._reported_offline:
                    self._reported_offline = True
                    self.statusChanged.emit(
                        "Offline - basemap from cache only", 0)
            self.tileFailed.emit(key, str(error))
            self._pump()
            return

        self._consecutive_429 = 0
        # Written even if the key has scrolled out of the want-set: the bytes
        # were paid for, and discarding them guarantees a re-request later.
        self._cache.write(self._provider.id, key, data)
        if key in self._wanted:
            self.tileReady.emit(key, data)
        self._pump()

    def _on_rate_limited(self, reply) -> None:
        self._consecutive_429 += 1
        header = reply.rawHeader(b"Retry-After")
        try:
            delay = float(bytes(header).decode()) if header else 0.0
        except ValueError:
            delay = 0.0
        if delay <= 0:
            delay = min(2.0 * (2 ** (self._consecutive_429 - 1)), 120.0)

        # Suspend *all* fetching for this provider: continuing with other tiles
        # while rate-limited is exactly what the policy forbids.
        self._backoff_until = time.monotonic() + delay
        self._queue.clear()

        if self._consecutive_429 >= 3:
            self._disabled = True
            self.statusChanged.emit(
                "Basemap disabled for this session after repeated rate limits", 0)
        else:
            self.statusChanged.emit(
                f"Basemap rate-limited, retrying in {delay:.0f} s", 0)
