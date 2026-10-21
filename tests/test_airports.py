"""Resolving the departure airport from a track's first fix."""

from __future__ import annotations

import pytest

from rcta.core.airports import (
    MAX_AIRPORT_DISTANCE_M,
    Airport,
    find_airport,
    load_airports,
    nearest_airport,
)


def test_the_table_loads_and_is_not_a_stub():
    icao, lat, lon, iata, tz = load_airports()
    assert len(icao) > 3000
    assert len(icao) == len(lat) == len(lon) == len(iata) == len(tz)
    assert all(len(code) == 4 for code in icao)


def test_a_position_on_an_apron_resolves_to_that_airport():
    found = nearest_airport(30.202, -97.661)          # Austin, first fix
    assert found is not None
    assert found.icao == "KAUS"
    assert found.iata == "AUS"
    assert found.distance_m < 1000.0


def test_the_other_end_of_the_same_flight_resolves_too():
    found = nearest_airport(40.788, -111.979)         # Salt Lake City
    assert found is not None and found.icao == "KSLC"


@pytest.mark.parametrize("lat,lon,expected", [
    (51.4707, -0.4599, "EGLL"),
    (35.5497, 139.787, "RJTT"),
    (-33.9399, 151.1753, "YSSY"),
    (-22.8100, -43.2506, "SBGL"),
])
def test_airports_outside_north_america_resolve(lat, lon, expected):
    """A world table, not a US one: the tool travels."""
    found = nearest_airport(lat, lon)
    assert found is not None and found.icao == expected


def test_the_middle_of_an_ocean_is_not_an_airport():
    """Nearest is not the same as near, and guessing here queries the wrong
    airport and returns a confident, wrong flight."""
    assert nearest_airport(30.0, -40.0) is None


def test_the_distance_ceiling_is_what_rejects_a_far_position():
    """A point 200 km from a field has a nearest airport; it is not at one."""
    assert nearest_airport(32.0, -97.661) is None
    assert nearest_airport(32.0, -97.661, max_distance_m=3e5) is not None


def test_a_nonsense_position_is_declined_rather_than_raising():
    assert nearest_airport(float("nan"), 0.0) is None
    assert nearest_airport(0.0, float("inf")) is None


def test_lookup_by_icao_and_by_iata():
    assert find_airport("KAUS").icao == "KAUS"
    assert find_airport("kaus").icao == "KAUS"
    assert find_airport("AUS").icao == "KAUS"
    assert find_airport("") is None
    assert find_airport("ZZZZ") is None


def test_an_airport_found_by_code_reports_no_distance():
    """Distance is a property of a search, and there was no position."""
    assert find_airport("KAUS").distance_m == 0.0
    assert "km" not in find_airport("KAUS").describe()
    assert "km" in nearest_airport(30.202, -97.661).describe()


def test_the_default_radius_covers_a_large_field_but_not_the_next_one():
    assert 5_000.0 <= MAX_AIRPORT_DISTANCE_M <= 25_000.0


def test_an_airport_is_frozen():
    from dataclasses import FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        Airport("KAUS", 30.0, -97.0).icao = "KSLC"


# ---- the timezone column --------------------------------------------------
#
# It is here so that a flight date can be read as the calendar day printed on
# a boarding pass rather than as a UTC one; see search_window. The tests below
# are about the table being trustworthy enough to decide that, which means
# both that the zones are right and that a missing one is visibly missing.


@pytest.mark.parametrize("code,zone", [
    ("KSLC", "America/Denver"),
    ("KAUS", "America/Chicago"),
    ("EGLL", "Europe/London"),
    ("RJTT", "Asia/Tokyo"),
    ("YSSY", "Australia/Sydney"),
    ("PANC", "America/Anchorage"),
])
def test_airports_carry_their_iana_zone(code, zone):
    assert find_airport(code).tz == zone


def test_a_zone_survives_a_lookup_by_position_too():
    """The fetch resolves the airport from the first fix, not from a code."""
    assert nearest_airport(40.788, -111.979).tz == "America/Denver"


def test_wide_political_zones_are_taken_from_the_table_not_from_longitude():
    """Longitude is the tempting shortcut and it is wrong exactly here.

    Xinjiang keeps Beijing's clock and Spain keeps Berlin's, so a zone
    derived from longitude is out by two hours in both -- and two hours is
    more than the whole margin at a date boundary.
    """
    assert find_airport("ZWWW").tz == "Asia/Shanghai"      # Urumqi, 87 E
    assert find_airport("LEMD").tz == "Europe/Madrid"      # Madrid, 3 W


def test_every_zone_in_the_table_is_a_real_iana_name():
    """A typo here is a silent one-day error, so the whole column is checked."""
    import zoneinfo

    known = zoneinfo.available_timezones()
    _, _, _, _, tz = load_airports()
    unknown = sorted({z for z in tz if z and z not in known})
    assert not unknown, f"not IANA zone names: {unknown}"


def test_almost_every_airport_has_one():
    _, _, _, _, tz = load_airports()
    assert sum(1 for z in tz if z) / len(tz) > 0.9


def test_an_unsourced_airport_says_so_rather_than_guessing():
    """Empty is a real answer, as a nearest-airport miss is.

    The zone could always be guessed from longitude; the table declines to,
    because a caller that falls back to UTC and reports it is recoverable and
    a caller handed a plausible wrong zone is not.
    """
    _, _, _, _, tz = load_airports()
    assert "" in tz
