"""Fetching a flight trajectory, and living inside somebody else's API budget.

OpenSky is the default and effectively the only choice, and it is worth
recording why, because the obvious alternatives all fail on the same point.
The whole feature needs *historical* altitude for a flight that has already
landed. Every keyless ADS-B aggregator -- adsb.lol, adsb.fi, airplanes.live --
serves live aircraft only and returns an empty set for a flight that is over.
ADS-B Exchange refuses direct access to its history archive. The commercial
APIs need a paid key. OpenSky serves history, is free for non-commercial and
research use, and asks to be cited; anonymous callers are refused with
``HTTP 403 You cannot access historical flights``, so credentials are not
optional here even though they are for the live endpoints.

That makes the account's metered request budget a design constraint rather
than an afterthought, and three mechanisms follow from it.

*The query is keyed by the track's own departure airport, not by the day.*
OpenSky has no callsign search, and the obvious route -- ``/flights/all``
over the track's time window -- does not work, for a reason that is invisible
until it is measured. That endpoint returns only flights contained *entirely*
within the requested interval, and caps the interval at two hours: across
29,000 flights in six real responses, not one exceeded 1.99 h. So every
flight longer than two hours is unfindable by it at any window size, which is
most of the flights anybody would fly with a detector, and the failure is
silent -- the flight simply is not in the response, and the hint blames the
flight number. ``/flights/departure`` has no containment rule and allows a
seven-day interval, so one request finds a flight of any length. It needs an
airport, and the track's first fix is on the apron of the one it left from;
see :mod:`rcta.core.airports`.

*Every response is cached, and the TTL depends on how old the query is.*
Historical ADS-B is immutable, so a settled window is cached indefinitely --
unlike a map tile, it can never become stale. A window inside the last couple
of days is a different matter: OpenSky backfills as receivers report, so a
track flown this morning may be incomplete now and complete tonight. Those
get a short TTL. Treating both cases the same either re-fetches immutable data
forever or pins an incomplete answer permanently.

*A rate limit stops the whole fetch.* No retry beyond two attempts, and
``Retry-After`` is honoured when offered. Continuing to hammer other endpoints
while one is rate-limiting is the behaviour that gets a research API closed to
everyone.

Nothing here parses a trajectory: the payload goes to
:mod:`rcta.core.io_flight`, so every format this program understands is
testable from a file with no network, and the cached response *is* that file.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

from rcta import USER_AGENT
from rcta.core.airports import find_airport, nearest_airport
from rcta.core.flight import FlightNumber, Trajectory, parse_flight_number
from rcta.core.io_flight import parse_opensky_track

__all__ = ["FlightCache", "FlightFetchError", "FlightProvider",
           "OpenSkyClient", "cache_dir", "credentials_from_env",
           "fetch_trajectory", "load_flight_providers"]

#: Seconds after which a query window is treated as settled and its response
#: cached indefinitely. Two days is comfortably past OpenSky's backfill.
SETTLED_AFTER_S = 2 * 86400.0

#: Cache lifetime for a window that is not yet settled.
UNSETTLED_TTL_S = 3600.0

#: OpenSky caps ``/flights/departure`` at a seven-day interval. Any real
#: search window is far inside this; it is checked so that a track with a
#: corrupt timestamp asks a question the API will answer rather than one it
#: rejects.
MAX_DEPARTURE_WINDOW_S = 7 * 86400.0

#: Margin added either side of the track's own span when searching for the
#: flight. The detector is usually switched on at the gate and off after
#: taxi-in, so the track is a subset of the flight, not the reverse -- but
#: ``firstSeen`` is when ADS-B acquired the aircraft, which can be after
#: pushback, so the window must be generous in both directions.
DEFAULT_SEARCH_MARGIN_S = 3600.0

#: Hard ceiling on requests per fetch, so a mistaken window cannot spend an
#: account's whole daily budget in one command.
MAX_REQUESTS_PER_FETCH = 8

_TIMEOUT_S = 30.0
_MAX_ATTEMPTS = 2


class FlightFetchError(RuntimeError):
    """A trajectory could not be retrieved.

    Carries ``hint`` separately from the message so the CLI can print an
    actionable next step -- register for an account, widen the window, pass a
    local file -- underneath the failure rather than inside it.
    """

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint


# ---------------------------------------------------------------------------
# providers and credentials
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FlightProvider:
    """A trajectory source and the terms attached to it."""

    id: str
    name: str
    base_url: str
    token_url: str = ""
    altitude_kind: str = "unknown"
    needs_credentials: bool = False
    credential_env: tuple[str, ...] = ()
    signup_url: str = ""
    attribution: str = ""
    citation: str = ""
    policy_note: str = ""

    def describe(self) -> str:
        lines = [f"{self.name} ({self.id})"]
        if self.attribution:
            lines.append(f"  {self.attribution}")
        if self.citation:
            lines.append(f"  cite: {self.citation}")
        if self.policy_note:
            lines.append(f"  {self.policy_note}")
        return "\n".join(lines)


def load_flight_providers(path=None) -> tuple[dict[str, FlightProvider], str]:
    """Read the provider table. Returns ``(providers, default_id)``."""
    if path is None:
        raw = resources.files("rcta.data").joinpath(
            "flight_providers.json").read_text()
    else:
        raw = Path(path).read_text()
    data = json.loads(raw)
    if data.get("schema_version") != 1:
        raise ValueError("unsupported flight providers schema_version "
                         f"{data.get('schema_version')}")
    providers = {}
    for entry in data["providers"]:
        entry = dict(entry)
        entry["credential_env"] = tuple(entry.get("credential_env", ()))
        providers[entry["id"]] = FlightProvider(**entry)
    return providers, data.get("default", "opensky")


#: Key spellings accepted for a client id and a secret, in the order tried.
#: The snake_case pair is this program's documented spelling. The camelCase
#: pair is the spelling of the credentials file OpenSky's own account page
#: hands you on download, which is therefore the spelling most people will
#: actually have on disk -- reading only ours means a user who did everything
#: right is told no credentials were found.
_ID_KEYS = ("client_id", "clientId")
_SECRET_KEYS = ("client_secret", "clientSecret")


def _credential_pair(entry) -> tuple[str, str] | None:
    """An ``(id, secret)`` pair out of one mapping, or None if incomplete."""
    if not isinstance(entry, dict):
        return None

    def pick(keys):
        for key in keys:
            value = str(entry.get(key, "") or "").strip()
            if value:
                return value
        return ""

    client_id, secret = pick(_ID_KEYS), pick(_SECRET_KEYS)
    return (client_id, secret) if client_id and secret else None


def credentials_from_env(provider: FlightProvider, env=None
                         ) -> tuple[str, str] | None:
    """OAuth2 client credentials from the environment, or None.

    Environment first and a config file second, because the environment is
    what a CI job and a shell session can both set without a file, and because
    a secret in a file is a secret that gets committed. The values are never
    logged, never written to the cache key, and never included in an error
    message.

    Two shapes of file are read: ours, keyed by provider id, and the flat one
    OpenSky hands out on download. Both key spellings are accepted in either.
    See :data:`_ID_KEYS`.

    Returns None when nothing is configured, but *raises* when a credentials
    file exists and cannot be used. Those two are not the same thing and must
    not read the same: "no credentials were found" sends a user who has not
    registered to the sign-up page, and sends a user whose file is a shape
    this function did not recognise on a search for a problem that is three
    lines from where they are already looking.
    """
    env = os.environ if env is None else env
    if len(provider.credential_env) != 2:
        return None
    client_id = env.get(provider.credential_env[0], "").strip()
    secret = env.get(provider.credential_env[1], "").strip()
    if client_id and secret:
        return (client_id, secret)

    path = Path(env.get("RCTA_FLIGHT_CREDENTIALS")
                or Path.home() / ".config" / "rcta" / "flight_credentials.json")
    try:
        raw = path.read_text()
    except OSError:
        return None                  # absent, or unreadable: not configured
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FlightFetchError(
            f"{path} is not valid JSON: {exc}",
            hint="expected {\"opensky\": {\"client_id\": \"...\", "
                 "\"client_secret\": \"...\"}}") from exc

    found = _credential_pair(
        data.get(provider.id) if isinstance(data, dict) else None)
    if found is None:
        found = _credential_pair(data)
    if found is not None:
        return found

    keys = ", ".join(sorted(data)) if isinstance(data, dict) else "none"
    raise FlightFetchError(
        f"{path} holds no usable {provider.id} credentials",
        hint=f"its top-level keys are: {keys}.  Expected either "
             f'{{"{provider.id}": {{"client_id": "...", "client_secret": '
             '"..."}} or the flat file OpenSky gives you, with clientId '
             "and clientSecret.  Delete the file to fall back to "
             + " and ".join(provider.credential_env) + ".")


def cache_dir() -> Path:
    """Where fetched trajectories live: a cache location, never app data.

    Computed from the platform rather than from ``QStandardPaths`` because
    this package must work with no Qt installed; the GUI's tile cache uses Qt
    and lands in the same parent directory on every platform that matters.
    """
    override = os.environ.get("RCTA_FLIGHT_CACHE")
    if override:
        return Path(override)
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches" / "rcta"
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA",
                                   Path.home() / "AppData" / "Local")) / "rcta"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME",
                                   Path.home() / ".cache")) / "rcta"
    return base / "flights"


# ---------------------------------------------------------------------------
# the cache
# ---------------------------------------------------------------------------

class FlightCache:
    """One JSON file per response, keyed by a hash of the request URL.

    The TTL is a property of the *query*, not of the cache: a window that
    closed days ago describes immutable history and never expires, while a
    recent one is still being backfilled. :meth:`read` therefore takes the end
    of the window it is caching, which is the only way to tell the two apart.
    """

    def __init__(self, root=None, *, settled_after_s: float = SETTLED_AFTER_S,
                 unsettled_ttl_s: float = UNSETTLED_TTL_S) -> None:
        self.root = Path(cache_dir() if root is None else root)
        self.settled_after_s = float(settled_after_s)
        self.unsettled_ttl_s = float(unsettled_ttl_s)

    def _path(self, provider_id: str, url: str) -> Path:
        digest = hashlib.sha256(url.encode()).hexdigest()[:32]
        return self.root / provider_id / f"{digest}.json"

    def ttl_for(self, window_end_s: float) -> float:
        """Cache lifetime for a query whose window ends at ``window_end_s``."""
        age = time.time() - float(window_end_s)
        return float("inf") if age >= self.settled_after_s else self.unsettled_ttl_s

    def read(self, provider_id: str, url: str, window_end_s: float):
        path = self._path(provider_id, url)
        try:
            age = time.time() - path.stat().st_mtime
            if age > self.ttl_for(window_end_s):
                return None
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def write(self, provider_id: str, url: str, payload) -> None:
        path = self._path(provider_id, url)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename: a crash must not leave a truncated response
            # that would be served as a valid trajectory indefinitely.
            tmp = path.with_suffix(".part")
            tmp.write_text(json.dumps(payload))
            os.replace(tmp, path)
        except OSError:
            pass

    def purge(self) -> tuple[int, int]:
        """Delete every cached response. Returns ``(files, bytes)``."""
        removed = freed = 0
        if not self.root.exists():
            return (0, 0)
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            try:
                size = path.stat().st_size
                path.unlink()
            except OSError:
                continue
            removed += 1
            freed += size
        return (removed, freed)

    def size_bytes(self) -> int:
        if not self.root.exists():
            return 0
        return sum(p.stat().st_size for p in self.root.rglob("*")
                   if p.is_file())


# ---------------------------------------------------------------------------
# the client
# ---------------------------------------------------------------------------

@dataclass
class OpenSkyClient:
    """A blocking, cache-first OpenSky client with no threads of its own.

    The access token is held in memory for the life of the object and refreshed
    a minute before it expires. It is never written to the cache: a bearer
    token on disk is a credential on disk, and it would be stale by the time
    anything read it anyway.
    """

    provider: FlightProvider
    credentials: tuple[str, str] | None = None
    cache: FlightCache = field(default_factory=FlightCache)
    #: ``(request, timeout) -> response``, injected by the tests so that no
    #: test in this suite can open a socket. Annotated rather than left as a
    #: bare class attribute because an unannotated default is not a dataclass
    #: field at all, and the constructor would silently reject it.
    opener: object | None = None
    offline: bool = False
    max_requests: int = MAX_REQUESTS_PER_FETCH
    _token: str = field(default="", repr=False, init=False)
    _token_expires: float = field(default=0.0, repr=False, init=False)
    _requests: int = field(default=0, init=False)

    # ---- transport -------------------------------------------------------

    def _open(self, request, timeout=_TIMEOUT_S):
        if self.opener is not None:
            return self.opener(request, timeout)
        return urllib.request.urlopen(request, timeout=timeout)

    def _authorize(self) -> str:
        """A bearer token, fetched or reused. Raises if no credentials exist."""
        if self._token and time.time() < self._token_expires - 60.0:
            return self._token
        if not self.credentials:
            raise FlightFetchError(
                "OpenSky refuses historical data to anonymous callers, and no "
                "credentials were found",
                hint=("register free at " + self.provider.signup_url
                      + ", create an API client, then set "
                      + " and ".join(self.provider.credential_env)
                      + ".  Or skip the network entirely with --trajectory "
                        "FILE."))
        client_id, secret = self.credentials
        body = urllib.parse.urlencode({
            "grant_type": "client_credentials",
            "client_id": client_id, "client_secret": secret,
        }).encode()
        request = urllib.request.Request(
            self.provider.token_url, data=body,
            headers={"User-Agent": USER_AGENT,
                     "Content-Type": "application/x-www-form-urlencoded"})
        try:
            with self._open(request) as response:
                payload = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            # Deliberately does not echo the credentials, not even truncated.
            raise FlightFetchError(
                f"OpenSky rejected the credentials (HTTP {exc.code})",
                hint="check the client id and secret; an OpenSky *website* "
                     "login is not an API client -- the client id and secret "
                     "come from the API Client section of your account page."
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise FlightFetchError(f"cannot reach OpenSky: {exc}",
                                  hint="use --trajectory FILE to work with no "
                                       "network at all") from exc
        token = str(payload.get("access_token", ""))
        if not token:
            raise FlightFetchError("OpenSky returned no access token")
        self._token = token
        self._token_expires = time.time() + float(payload.get("expires_in", 1800))
        return token

    def get(self, endpoint: str, params: dict, *, window_end_s: float):
        """A cached GET against the provider. Returns the decoded payload."""
        query = urllib.parse.urlencode(sorted(params.items()))
        url = f"{self.provider.base_url}{endpoint}?{query}"

        cached = self.cache.read(self.provider.id, url, window_end_s)
        if cached is not None:
            return cached
        if self.offline:
            raise FlightFetchError(
                f"offline, and {endpoint} is not in the cache",
                hint="drop --offline, or pass --trajectory FILE")
        if self._requests >= self.max_requests:
            raise FlightFetchError(
                f"stopping after {self._requests} requests, the per-fetch "
                "ceiling",
                hint="narrow the window, or fetch once and re-use the cached "
                     "trajectory with --trajectory")

        token = self._authorize()
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT,
                          "Authorization": f"Bearer {token}",
                          "Accept": "application/json"})

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            self._requests += 1
            try:
                with self._open(request) as response:
                    body = response.read().decode()
                payload = json.loads(body) if body.strip() else None
                self.cache.write(self.provider.id, url, payload)
                return payload
            except urllib.error.HTTPError as exc:
                detail = _read_error(exc)
                if exc.code in (401, 403):
                    raise FlightFetchError(
                        f"OpenSky refused {endpoint} (HTTP {exc.code}): "
                        f"{detail}",
                        hint="historical endpoints need a registered API "
                             "client; anonymous access is live data only"
                    ) from exc
                if exc.code == 404:
                    return None
                if exc.code in (429, 503) and attempt < _MAX_ATTEMPTS:
                    _sleep_for_retry(exc)
                    continue
                if exc.code in (429, 503):
                    raise FlightFetchError(
                        f"OpenSky is rate-limiting this account "
                        f"(HTTP {exc.code}): {detail}",
                        hint="the request budget is metered per account and "
                             "resets daily; the response cache means a "
                             "re-run of the same query costs nothing"
                    ) from exc
                raise FlightFetchError(
                    f"OpenSky returned HTTP {exc.code} for {endpoint}: "
                    f"{detail}") from exc
            except json.JSONDecodeError as exc:
                raise FlightFetchError(
                    f"OpenSky returned something that is not JSON: {exc}"
                ) from exc
            except (urllib.error.URLError, OSError) as exc:
                if attempt < _MAX_ATTEMPTS:
                    time.sleep(1.0)
                    continue
                raise FlightFetchError(f"cannot reach OpenSky: {exc}",
                                       hint="use --trajectory FILE to work "
                                            "offline") from exc
        return None

    # ---- the two queries -------------------------------------------------

    def find_flight(self, flight: FlightNumber, begin_s: float, end_s: float,
                    *, departure, progress=None) -> dict:
        """Locate the airframe that flew this callsign out of this airport.

        Keyed by departure airport rather than by time alone, because
        ``/flights/all`` cannot answer the question. It returns only flights
        contained entirely within the requested interval and caps that
        interval at two hours, so every flight longer than two hours is
        invisible to it -- silently, and with a hint that blames the flight
        number. ``/flights/departure`` has no containment rule and allows a
        seven-day interval, so one request covers any flight of any length.

        Every candidate spelling of the callsign is tried, because zero
        padding is not consistent between operators.
        """
        candidates = {c.upper() for c in flight.candidates}
        if not candidates:
            raise FlightFetchError(
                f"cannot turn {flight.raw!r} into a callsign: airline code "
                f"{flight.airline_iata!r} is not in the table",
                hint="pass the ICAO callsign instead, e.g. DAL2618")

        icao = getattr(departure, "icao", departure)
        icao = str(icao or "").strip().upper()
        if not icao:
            raise FlightFetchError(
                "no departure airport to search from",
                hint="the flight lookup is keyed by the airport the track "
                     "starts at; name it explicitly if the track does not "
                     "begin on an apron")
        span = float(end_s) - float(begin_s)
        if span > MAX_DEPARTURE_WINDOW_S:
            raise FlightFetchError(
                f"the search window is {span / 86400.0:.1f} days; OpenSky "
                f"allows {MAX_DEPARTURE_WINDOW_S / 86400.0:.0f}",
                hint="narrow the window, or pass a local trajectory file")

        if progress is not None:
            progress(0.4)
        payload = self.get("/flights/departure",
                           {"airport": icao, "begin": int(begin_s),
                            "end": int(end_s)},
                           window_end_s=end_s)

        seen_similar: set[str] = set()
        matches: list[dict] = []
        for entry in payload or ():
            call = str(entry.get("callsign") or "").strip().upper()
            if call in candidates:
                matches.append(entry)
            elif call.startswith(flight.airline_icao):
                seen_similar.add(call)

        if not matches:
            departures = len(payload or ())
            hint = (f"{departures} departures from {icao} between "
                    f"{_iso(begin_s)} and {_iso(end_s)} were searched. ")
            if seen_similar:
                sample = ", ".join(sorted(seen_similar)[:8])
                hint += (f"Other {flight.airline_icao} flights left {icao} in "
                         f"that window ({sample}), so this is the right "
                         "airport and the right day -- check the flight "
                         "number, and note that a codeshare flies under the "
                         "operating carrier's callsign, not the one on your "
                         "ticket.")
            elif departures:
                hint += (f"No {flight.airline_icao} flight left {icao} at all "
                         "in that window, which usually means the number on "
                         "the ticket belongs to a different operating "
                         "carrier.")
            else:
                hint += ("OpenSky recorded no departures at all from there, "
                         "which is a receiver coverage gap rather than "
                         "anything about this flight.")
            raise FlightFetchError(
                f"OpenSky has no flight {'/'.join(sorted(candidates))} "
                f"departing {icao} in that window", hint=hint)

        # Longest overlap with the track's window, not simply the first: a
        # daily flight number can appear twice inside a generous margin, and
        # the right one is the one the detector was actually aboard.
        best = max(matches, key=lambda e: _overlap(
            float(e.get("firstSeen", 0)), float(e.get("lastSeen", 0)),
            begin_s, end_s))
        if len(matches) > 1:
            best = dict(best)
            best["_ambiguous"] = len(matches)
        return best

    def fetch_path(self, icao24: str, at_s: float, *, window_end_s: float):
        """The waypoint path for an airframe at an instant inside its flight."""
        return self.get("/tracks/all",
                        {"icao24": str(icao24).lower(), "time": int(at_s)},
                        window_end_s=window_end_s)


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------

def fetch_trajectory(flight, *, begin_s: float, end_s: float,
                     origin=None, departure_icao: str = "",
                     provider: FlightProvider | None = None,
                     credentials=None, cache: FlightCache | None = None,
                     margin_s: float = DEFAULT_SEARCH_MARGIN_S,
                     offline: bool = False, client: OpenSkyClient | None = None,
                     progress=None) -> Trajectory:
    """Fetch the trajectory of ``flight`` covering ``begin_s .. end_s``.

    ``begin_s`` and ``end_s`` should be the *track's* own first and last
    timestamps, and ``origin`` its first ``(lat, lon)`` fix. The search is
    keyed by departure airport, so the track supplies both when and where;
    see :meth:`OpenSkyClient.find_flight` for why time alone cannot work.

    ``departure_icao`` overrides the airport resolved from ``origin``, for a
    track whose first fix is a hotel car park rather than an apron.
    """
    if isinstance(flight, str):
        flight = parse_flight_number(flight)

    airport = find_airport(departure_icao) if departure_icao else None
    if departure_icao and airport is None:
        raise FlightFetchError(
            f"{departure_icao!r} is not an airport code I know",
            hint="expected an ICAO code such as KAUS, or an IATA code such "
                 "as AUS, for an airport with scheduled service")
    if airport is None and origin is not None:
        airport = nearest_airport(float(origin[0]), float(origin[1]))
        if airport is None:
            raise FlightFetchError(
                f"the track starts at {float(origin[0]):.3f}, "
                f"{float(origin[1]):.3f}, which is not at an airport",
                hint="the lookup is keyed by departure airport. Name it "
                     "explicitly if the detector was switched on away from "
                     "the terminal, or pass a local trajectory file.")

    if provider is None:
        providers, default = load_flight_providers()
        provider = providers[default]
    if provider.id != "opensky":
        raise FlightFetchError(
            f"provider {provider.id!r} cannot fetch: "
            f"{provider.name} reads from disk",
            hint="pass --trajectory FILE for a local trajectory")

    if client is None:
        if credentials is None:
            credentials = credentials_from_env(provider)
        client = OpenSkyClient(provider=provider, credentials=credentials,
                               cache=cache or FlightCache(), offline=offline)

    if airport is None:
        raise FlightFetchError(
            "no departure airport was given and the track's first position "
            "is unknown",
            hint="pass origin=(lat, lon) from the track's first fix, or "
                 "departure_icao explicitly")

    search_begin = float(begin_s) - float(margin_s)
    search_end = float(end_s) + float(margin_s)
    entry = client.find_flight(flight, search_begin, search_end,
                               departure=airport, progress=progress)

    icao24 = str(entry.get("icao24") or "").strip()
    if not icao24:
        raise FlightFetchError("OpenSky matched the callsign but reported no "
                               "airframe for it")
    first_seen = float(entry.get("firstSeen") or begin_s)
    last_seen = float(entry.get("lastSeen") or end_s)

    if progress is not None:
        progress(0.8)
    # Ask at the midpoint of the flight: /tracks/all wants an instant inside
    # the flight and returns the whole path around it, and the midpoint is the
    # value least likely to fall outside on either side.
    payload = client.fetch_path(icao24, 0.5 * (first_seen + last_seen),
                                window_end_s=last_seen)
    if not payload:
        raise FlightFetchError(
            f"OpenSky has no waypoint track for {icao24} on that flight",
            hint="/tracks/all is an experimental endpoint with thinner "
                 "history than /flights/all. The flight was found, so the "
                 "callsign and date are right; there is simply no stored "
                 "path. Try --trajectory with an export from another tracker.")

    trajectory = parse_opensky_track(
        payload, source=f"opensky /tracks/all icao24={icao24}")

    warnings = list(trajectory.warnings)
    if entry.get("_ambiguous"):
        warnings.append(
            f"{entry['_ambiguous']} flights carried this callsign inside the "
            "search window; the one overlapping the track most was used")
    departure = entry.get("estDepartureAirport")
    arrival = entry.get("estArrivalAirport")
    if departure or arrival:
        warnings.append(f"OpenSky calls this {departure or '?'} -> "
                        f"{arrival or '?'}")

    trajectory = replace(
        trajectory,
        callsign=trajectory.callsign or flight.callsign,
        fetched_at=datetime.now(tz=timezone.utc),
        warnings=tuple(warnings))
    if progress is not None:
        progress(1.0)
    return trajectory


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _iso(seconds: float) -> str:
    return datetime.fromtimestamp(float(seconds), tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%MZ")


def _read_error(exc) -> str:
    """The server's own explanation, truncated. OpenSky's are useful."""
    try:
        return exc.read().decode(errors="replace").strip()[:200] or exc.reason
    except Exception:
        return str(getattr(exc, "reason", "")) or "no detail"


def _sleep_for_retry(exc) -> None:
    header = exc.headers.get("Retry-After") if exc.headers else None
    try:
        delay = float(header) if header else 2.0
    except (TypeError, ValueError):
        delay = 2.0
    time.sleep(min(max(delay, 1.0), 30.0))
