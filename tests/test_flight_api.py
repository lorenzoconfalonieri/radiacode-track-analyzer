"""The OpenSky client, driven through an injected opener.

No test here opens a socket. Every response is a fixture, which is the point
of ``OpenSkyClient.opener`` existing: the failure modes worth testing --
rate limiting, a 403 on history, a flight that is not there -- are exactly the
ones that cannot be provoked on demand against the real service.
"""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import pytest

from rcta.core.flight import parse_flight_number
from rcta.net.flights import (
    FlightCache,
    FlightFetchError,
    OpenSkyClient,
    cache_dir,
    credentials_from_env,
    fetch_trajectory,
    load_flight_providers,
)

TOKEN = {"access_token": "t0ken", "expires_in": 1800}

#: The sample track's first fix, on the apron at Austin. The flight lookup is
#: keyed by departure airport, so every fetch needs a position as well as a
#: time; see OpenSkyClient.find_flight for why time alone cannot work.
AUS = (30.2, -97.66)

FLIGHT = {"icao24": "a1b2c3", "callsign": "DAL2618 ",
          "firstSeen": 1788785000, "lastSeen": 1788796000,
          "estDepartureAirport": "KAUS", "estArrivalAirport": "KSLC"}

TRACK = {"icao24": "a1b2c3", "callsign": "DAL2618 ",
         "path": [[1788785639, 30.2, -97.66, 150.0, 0.0, True],
                  [1788790000, 35.0, -102.0, 10668.0, 0.0, False],
                  [1788795393, 40.79, -111.98, 1300.0, 0.0, False]]}


class Response(io.BytesIO):
    """Just enough of an ``http.client.HTTPResponse`` for the client."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def make_opener(routes, log=None):
    """Serve fixtures by URL substring, recording what was asked for."""
    def opener(request, timeout=None):
        url = request.full_url
        if log is not None:
            log.append(url)
        for fragment, payload in routes.items():
            if fragment in url:
                if isinstance(payload, Exception):
                    raise payload
                return Response(json.dumps(payload).encode())
        raise AssertionError(f"unexpected request to {url}")
    return opener


def http_error(code, body=b"", headers=None):
    return urllib.error.HTTPError(
        "https://opensky-network.org/api/x", code, "err",
        headers or {}, io.BytesIO(body))


@pytest.fixture()
def provider():
    providers, default = load_flight_providers()
    return providers[default]


@pytest.fixture()
def cache(tmp_path):
    return FlightCache(tmp_path / "flights")


def client_for(provider, cache, routes, log=None, **kw):
    return OpenSkyClient(provider=provider, credentials=("id", "secret"),
                         cache=cache, opener=make_opener(routes, log), **kw)


# ---- the provider table ---------------------------------------------------

def test_the_provider_table_loads(provider):
    assert provider.id == "opensky"
    assert provider.needs_credentials
    assert provider.credential_env == ("OPENSKY_CLIENT_ID",
                                       "OPENSKY_CLIENT_SECRET")
    assert "opensky.org" in provider.attribution or "opensky" in \
        provider.attribution.lower()


def test_the_file_provider_cannot_fetch(cache):
    providers, _ = load_flight_providers()
    with pytest.raises(FlightFetchError, match="reads from disk"):
        fetch_trajectory("DL2618", begin_s=0, end_s=1,
                         provider=providers["file"], cache=cache)


def test_credentials_come_from_the_environment(provider):
    # Every case points the file away from $HOME. Without that this test reads
    # whatever credentials the developer running it happens to have, and the
    # "nothing configured" assertion below passes or fails by accident.
    assert credentials_from_env(
        provider, {"RCTA_FLIGHT_CREDENTIALS": "/nonexistent"}) is None
    found = credentials_from_env(provider, {
        "OPENSKY_CLIENT_ID": "abc", "OPENSKY_CLIENT_SECRET": "xyz",
        "RCTA_FLIGHT_CREDENTIALS": "/nonexistent"})
    assert found == ("abc", "xyz")


def test_credentials_fall_back_to_a_config_file(provider, tmp_path):
    path = tmp_path / "creds.json"
    path.write_text(json.dumps({"opensky": {"client_id": "a",
                                            "client_secret": "b"}}))
    assert credentials_from_env(
        provider, {"RCTA_FLIGHT_CREDENTIALS": str(path)}) == ("a", "b")


def test_the_file_opensky_hands_you_is_read_as_it_comes(provider, tmp_path):
    """Downloading credentials from the account page is the common path.

    That file is flat and camelCase. Accepting only our own spelling means a
    user who registered, made an API client and saved the file where the
    documentation says is told that no credentials were found.
    """
    path = tmp_path / "creds.json"
    path.write_text(json.dumps({"clientId": "a", "clientSecret": "b"}))
    assert credentials_from_env(
        provider, {"RCTA_FLIGHT_CREDENTIALS": str(path)}) == ("a", "b")


def test_either_spelling_works_under_the_provider_key(provider, tmp_path):
    path = tmp_path / "creds.json"
    path.write_text(json.dumps({"opensky": {"clientId": "a",
                                            "clientSecret": "b"}}))
    assert credentials_from_env(
        provider, {"RCTA_FLIGHT_CREDENTIALS": str(path)}) == ("a", "b")


def test_a_credentials_file_that_cannot_be_used_is_not_silence(provider,
                                                               tmp_path):
    """Unconfigured and misconfigured must not read the same.

    Returning None here would send a user who has already registered back to
    the sign-up page, when the problem is a key name in a file they are
    looking at.
    """
    path = tmp_path / "creds.json"
    path.write_text(json.dumps({"user": "someone", "password": "hunter2"}))
    with pytest.raises(FlightFetchError) as excinfo:
        credentials_from_env(provider, {"RCTA_FLIGHT_CREDENTIALS": str(path)})
    assert "no usable opensky credentials" in str(excinfo.value)
    assert "top-level keys are: password, user" in excinfo.value.hint
    assert "clientId" in excinfo.value.hint
    assert "hunter2" not in excinfo.value.hint + str(excinfo.value)


def test_an_unparseable_credentials_file_says_so(provider, tmp_path):
    path = tmp_path / "creds.json"
    path.write_text("{not json")
    with pytest.raises(FlightFetchError) as excinfo:
        credentials_from_env(provider, {"RCTA_FLIGHT_CREDENTIALS": str(path)})
    assert "not valid JSON" in str(excinfo.value)
    assert "client_secret" in excinfo.value.hint


def test_an_absent_credentials_file_is_simply_unconfigured(provider, tmp_path):
    """The one case that must stay quiet: nothing set up yet."""
    assert credentials_from_env(
        provider, {"RCTA_FLIGHT_CREDENTIALS": str(tmp_path / "nope.json")}
    ) is None


def test_the_environment_wins_over_an_unusable_file(provider, tmp_path):
    """A broken file must not mask credentials that are already working."""
    path = tmp_path / "creds.json"
    path.write_text("{not json")
    assert credentials_from_env(provider, {
        "OPENSKY_CLIENT_ID": "abc", "OPENSKY_CLIENT_SECRET": "xyz",
        "RCTA_FLIGHT_CREDENTIALS": str(path)}) == ("abc", "xyz")


def test_the_cache_is_a_cache_location_not_application_data(monkeypatch):
    monkeypatch.setenv("RCTA_FLIGHT_CACHE", "/tmp/somewhere")
    assert cache_dir() == Path("/tmp/somewhere")
    monkeypatch.delenv("RCTA_FLIGHT_CACHE")
    assert "ache" in str(cache_dir())            # Caches or .cache


# ---- the happy path -------------------------------------------------------

def test_a_flight_is_found_and_its_path_fetched(provider, cache):
    log = []
    client = client_for(provider, cache,
                        {"token": TOKEN, "flights/departure": [FLIGHT],
                         "tracks/all": TRACK}, log)
    trajectory = fetch_trajectory(
        parse_flight_number("DL2618"), origin=AUS, begin_s=1788785639,
        end_s=1788795393, provider=provider, client=client)
    assert trajectory.icao24 == "a1b2c3"
    assert trajectory.n == 3
    assert trajectory.altitude_ceiling_m == 10668.0
    assert trajectory.fetched_at is not None
    assert any("KAUS" in w for w in trajectory.warnings)


def test_the_window_comes_from_the_track_not_the_calendar(provider, cache):
    """One request, keyed by the airport the track starts at."""
    log = []
    client = client_for(provider, cache,
                        {"token": TOKEN, "flights/departure": [FLIGHT],
                         "tracks/all": TRACK}, log)
    fetch_trajectory(parse_flight_number("DL2618"), origin=AUS, begin_s=1788785639,
                     end_s=1788795393, provider=provider, client=client,
                     margin_s=0.0)
    lookups = [u for u in log if "flights/departure" in u]
    assert len(lookups) == 1
    assert "airport=KAUS" in lookups[0]
    assert "begin=1788785639" in lookups[0] and "end=1788795393" in lookups[0]


def test_a_flight_longer_than_two_hours_is_findable(provider, cache):
    """The bug this endpoint was chosen to fix.

    ``/flights/all`` returns only flights contained entirely within the
    requested interval and caps that interval at two hours, so a 2 h 43 min
    flight could never appear in any response and the failure blamed the
    flight number. Nothing here may reintroduce a window narrower than the
    flight.
    """
    log = []
    long_flight = dict(FLIGHT, firstSeen=1788785639, lastSeen=1788795393)
    assert long_flight["lastSeen"] - long_flight["firstSeen"] > 2 * 3600
    client = client_for(provider, cache,
                        {"token": TOKEN, "flights/departure": [long_flight],
                         "tracks/all": TRACK}, log)
    trajectory = fetch_trajectory(
        parse_flight_number("DL2618"), origin=AUS, begin_s=1788785639,
        end_s=1788795393, provider=provider, client=client, margin_s=0.0)
    assert trajectory.n == 3
    assert len([u for u in log if "flights/departure" in u]) == 1


def test_a_response_is_cached_so_a_rerun_costs_nothing(provider, cache):
    log = []
    routes = {"token": TOKEN, "flights/departure": [FLIGHT], "tracks/all": TRACK}
    for _ in range(2):
        fetch_trajectory(parse_flight_number("DL2618"), origin=AUS, begin_s=1788785639,
                         end_s=1788795393, provider=provider,
                         client=client_for(provider, cache, routes, log),
                         margin_s=0.0)
    assert len([u for u in log if "api/" in u]) == 2    # second run: no calls


def test_settled_history_is_cached_indefinitely(cache):
    """Unlike a map tile, a past flight cannot become stale."""
    import time
    assert cache.ttl_for(time.time() - 30 * 86400) == float("inf")
    assert cache.ttl_for(time.time() - 60) < float("inf")


def test_the_cache_can_be_purged(provider, cache):
    fetch_trajectory(parse_flight_number("DL2618"), origin=AUS, begin_s=1788785639,
                     end_s=1788795393, provider=provider,
                     client=client_for(provider, cache,
                                       {"token": TOKEN,
                                        "flights/departure": [FLIGHT],
                                        "tracks/all": TRACK}))
    assert cache.size_bytes() > 0
    removed, freed = cache.purge()
    assert removed > 0 and freed > 0
    assert cache.size_bytes() == 0


# ---- the failures ---------------------------------------------------------

def test_no_credentials_names_the_signup_page(provider, cache):
    client = OpenSkyClient(provider=provider, credentials=None, cache=cache,
                           opener=make_opener({}))
    with pytest.raises(FlightFetchError) as excinfo:
        client.get("/flights/all", {"begin": 1, "end": 2}, window_end_s=2)
    assert "anonymous" in str(excinfo.value)
    assert "register free" in excinfo.value.hint
    assert "--trajectory" in excinfo.value.hint


def test_bad_credentials_do_not_echo_the_secret(provider, cache):
    client = client_for(provider, cache,
                        {"token": http_error(401, b"invalid_client")})
    with pytest.raises(FlightFetchError) as excinfo:
        client.get("/flights/all", {"begin": 1, "end": 2}, window_end_s=2)
    assert "secret" not in str(excinfo.value)
    assert "API Client section" in excinfo.value.hint


def test_a_403_on_history_explains_itself(provider, cache):
    client = client_for(provider, cache, {
        "token": TOKEN,
        "flights/all": http_error(403, b"You cannot access historical flights")})
    with pytest.raises(FlightFetchError) as excinfo:
        client.get("/flights/all", {"begin": 1, "end": 2}, window_end_s=2)
    assert "historical" in str(excinfo.value)


def test_a_flight_that_is_not_there_says_what_was_searched(provider, cache):
    client = client_for(provider, cache, {"token": TOKEN, "flights/departure": []})
    with pytest.raises(FlightFetchError) as excinfo:
        fetch_trajectory(parse_flight_number("DL2618"), origin=AUS, begin_s=1788785639,
                         end_s=1788795393, provider=provider, client=client)
    assert "no flight" in str(excinfo.value)
    assert "2026-09-07" in excinfo.value.hint


def test_other_callsigns_present_means_coverage_is_not_the_problem(provider,
                                                                   cache):
    """The most useful thing the failure can say: check the number, not the sky."""
    others = [dict(FLIGHT, callsign="DAL0099 "), dict(FLIGHT, callsign="DAL777 ")]
    client = client_for(provider, cache,
                        {"token": TOKEN, "flights/departure": others})
    with pytest.raises(FlightFetchError) as excinfo:
        fetch_trajectory(parse_flight_number("DL2618"), origin=AUS, begin_s=1788785639,
                         end_s=1788795393, provider=provider, client=client)
    assert "right airport and the right day" in excinfo.value.hint
    assert "DAL0099" in excinfo.value.hint
    assert "KAUS" in excinfo.value.hint
    # A ticket number is not always the operating carrier's callsign, and
    # that is now the likeliest remaining explanation.
    assert "codeshare" in excinfo.value.hint


def test_a_missing_waypoint_track_is_distinguished_from_a_missing_flight(
        provider, cache):
    client = client_for(provider, cache, {"token": TOKEN,
                                          "flights/departure": [FLIGHT],
                                          "tracks/all": None})
    with pytest.raises(FlightFetchError) as excinfo:
        fetch_trajectory(parse_flight_number("DL2618"), origin=AUS, begin_s=1788785639,
                         end_s=1788795393, provider=provider, client=client)
    assert "no waypoint track" in str(excinfo.value)
    assert "callsign and date are right" in excinfo.value.hint


def test_rate_limiting_stops_the_fetch_rather_than_hammering(provider, cache):
    log = []
    client = client_for(provider, cache,
                        {"token": TOKEN,
                         "flights/all": http_error(429, b"slow down",
                                                   {"Retry-After": "1"})},
                        log)
    with pytest.raises(FlightFetchError) as excinfo:
        client.get("/flights/all", {"begin": 1, "end": 2}, window_end_s=2)
    assert "rate-limiting" in str(excinfo.value)
    assert len([u for u in log if "flights" in u]) == 2      # one retry, no more


def test_offline_uses_the_cache_and_never_the_network(provider, cache):
    routes = {"token": TOKEN, "flights/departure": [FLIGHT], "tracks/all": TRACK}
    fetch_trajectory(parse_flight_number("DL2618"), origin=AUS, begin_s=1788785639,
                     end_s=1788795393, provider=provider,
                     client=client_for(provider, cache, routes), margin_s=0.0)

    offline = OpenSkyClient(provider=provider, credentials=("id", "secret"),
                            cache=cache, offline=True,
                            opener=make_opener({}))     # any call would assert
    trajectory = fetch_trajectory(
        parse_flight_number("DL2618"), origin=AUS, begin_s=1788785639, end_s=1788795393,
        provider=provider, client=offline, margin_s=0.0)
    assert trajectory.n == 3


def test_offline_with_a_cold_cache_says_so(provider, cache):
    offline = OpenSkyClient(provider=provider, credentials=("id", "secret"),
                            cache=cache, offline=True, opener=make_opener({}))
    with pytest.raises(FlightFetchError) as excinfo:
        offline.get("/flights/all", {"begin": 1, "end": 2}, window_end_s=2)
    assert "not in the cache" in str(excinfo.value)


def test_the_per_fetch_request_ceiling_is_enforced(provider, cache):
    """A mistaken window must not spend a whole day's budget.

    A single fetch can no longer reach the ceiling on its own -- the
    departure lookup is one request and the path is a second -- so this
    drives the client directly. The guard stays because the ceiling is a
    property of the client, not of the one function that happens to use it.
    """
    client = client_for(provider, cache, {"token": TOKEN, "flights/departure": []},
                        max_requests=2)
    for i in range(2):
        client.get("/flights/departure", {"airport": "KAUS", "begin": i,
                                          "end": i + 1}, window_end_s=i + 1)
    with pytest.raises(FlightFetchError) as excinfo:
        client.get("/flights/departure", {"airport": "KAUS", "begin": 9,
                                          "end": 10}, window_end_s=10)
    assert "ceiling" in str(excinfo.value)


def test_a_window_wider_than_the_api_allows_is_refused(provider, cache):
    """Seven days is the cap; a corrupt timestamp must not send a bad query."""
    client = client_for(provider, cache, {"token": TOKEN})
    with pytest.raises(FlightFetchError) as excinfo:
        fetch_trajectory(parse_flight_number("DL2618"), origin=AUS, begin_s=0,
                         end_s=30 * 86400.0, provider=provider, client=client,
                         margin_s=0.0)
    assert "days" in str(excinfo.value)


def test_an_unknown_airline_is_refused_before_any_request(provider, cache):
    log = []
    client = client_for(provider, cache, {"token": TOKEN}, log)
    with pytest.raises(FlightFetchError, match="not in the table"):
        fetch_trajectory(parse_flight_number("QQ100"), origin=AUS,
                         begin_s=0, end_s=1,
                         provider=provider, client=client)
    assert log == []


def test_the_longest_overlapping_flight_wins_when_several_match(provider,
                                                                cache):
    """A daily number can appear twice inside a generous margin."""
    wrong_leg = dict(FLIGHT, icao24="ffffff", firstSeen=1788700000,
                     lastSeen=1788710000)
    client = client_for(provider, cache,
                        {"token": TOKEN, "flights/departure": [wrong_leg, FLIGHT],
                         "tracks/all": TRACK})
    trajectory = fetch_trajectory(
        parse_flight_number("DL2618"), origin=AUS, begin_s=1788785639, end_s=1788795393,
        provider=provider, client=client)
    assert trajectory.icao24 == "a1b2c3"
    assert any("carried this callsign" in w for w in trajectory.warnings)
