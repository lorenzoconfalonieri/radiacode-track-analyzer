"""Turning the first fix of a track into the airport the flight left from.

This exists because of a hard limit in somebody else's API. OpenSky's
``/flights/all`` returns only flights that begin *and* end inside the
requested interval, and it caps that interval at two hours, so a flight
longer than two hours cannot be found by that route at any window size --
not with a wider margin, not with more requests. Measured on a real
response: across 29,000 flights in six windows, the longest was 1.99 h and
none exceeded the window. The departure and arrival endpoints have no such
containment rule and allow a seven-day interval, but they are keyed by
airport, so the flight number alone is not enough to ask the question.

The track already knows the answer. A detector switched on at the gate
records its first fix on the apron, so the nearest airport to that fix is
the departure airport, and no user input is required to find it.

Two design points matter more than the lookup itself.

*A match has to be close, not merely nearest.* The nearest airport to a
point in the North Atlantic is still an airport. A hit is accepted only
within :data:`MAX_AIRPORT_DISTANCE_M`, which is generous enough for the far
end of a big field and its satellite car parks, and far too small to reach
the next city. A track that starts in a car park therefore reports that it
found nothing, which is the truthful answer and the one a caller can act on.

*Only airports with scheduled service are in the table.* Including every
grass strip and heliport would make the nearest-airport answer wrong at
exactly the airports that matter, because large fields have private
helipads inside their own perimeter. It also keeps the table at 4,000
entries instead of 86,000.

The table is a trimmed extract of OurAirports, which is public domain. It
carries coordinates and codes only, no runway or frequency data, because
nothing here needs them -- and one thing that is not geometry at all: the
airport's IANA timezone.

That column is here because a flight number is a thing a human reads off a
boarding pass, and a boarding pass is written in local time. DL2728 leaves
Salt Lake City at 18:06 on a Thursday and is airborne from 00:06 UTC on the
Friday, so a user asking for "the flight on the 10th" and a program working
in UTC disagree about which day it is by a whole calendar day. The departure
airport is the one place that disagreement can be settled, and this table
already had to resolve it for the query itself; see
:func:`rcta.core.flight.search_window`.

The zone is a *lookup*, never a guess from longitude. China spans five solar
hours in one zone and Spain keeps Berlin's, so longitude is wrong by an hour
or more in exactly the populous places, and an hour is the whole margin at a
date boundary. Where no zone could be sourced the column is empty and the
caller falls back to UTC and says so, which is the same contract as
:func:`nearest_airport` returning None: an honest gap beats a plausible
invention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources

import numpy as np

from rcta.core.geo import haversine

__all__ = ["Airport", "MAX_AIRPORT_DISTANCE_M", "find_airport",
           "load_airports", "nearest_airport"]

#: How close the first fix must be to call an airport the departure point.
#: Denver's field is 8 km across, so a 10 km radius covers standing on the
#: far taxiway; the nearest neighbouring airport to any major field is well
#: outside it.
MAX_AIRPORT_DISTANCE_M = 10_000.0


@dataclass(frozen=True)
class Airport:
    """One airport with scheduled service."""

    icao: str
    lat: float
    lon: float
    iata: str = ""
    #: IANA zone name, or "" when the table could not source one. Empty is a
    #: real answer: the caller works in UTC and reports that it did.
    tz: str = ""
    #: Metres from the query point. Zero when the airport was looked up by
    #: code rather than found by position.
    distance_m: float = 0.0

    def describe(self) -> str:
        name = f"{self.icao}/{self.iata}" if self.iata else self.icao
        if self.distance_m:
            return f"{name}, {self.distance_m / 1000.0:.1f} km from the first fix"
        return name


@lru_cache(maxsize=1)
def load_airports() -> tuple[tuple[str, ...], np.ndarray, np.ndarray,
                             tuple[str, ...], tuple[str, ...]]:
    """The table as parallel arrays, parsed once per process.

    Returned as arrays rather than as 4,000 objects because the only two
    operations are a vectorised distance over all of them and a lookup by
    code, and building the objects costs more than either.
    """
    raw = resources.files("rcta.data").joinpath("airports.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    if data.get("schema_version") != 2:
        raise ValueError("unsupported airports schema_version "
                         f"{data.get('schema_version')}")
    rows = data["airports"]
    icao = tuple(str(r[0]) for r in rows)
    lat = np.array([float(r[1]) for r in rows], dtype=float)
    lon = np.array([float(r[2]) for r in rows], dtype=float)
    iata = tuple(str(r[3] or "") for r in rows)
    tz = tuple(str(r[4] or "") for r in rows)
    return icao, lat, lon, iata, tz


def find_airport(code: str) -> Airport | None:
    """An airport by ICAO code, or by IATA code as a convenience."""
    wanted = str(code or "").strip().upper()
    if not wanted:
        return None
    icao, lat, lon, iata, tz = load_airports()
    for i, value in enumerate(icao):
        if value == wanted:
            return Airport(value, float(lat[i]), float(lon[i]), iata[i], tz[i])
    if len(wanted) == 3:
        for i, value in enumerate(iata):
            if value == wanted:
                return Airport(icao[i], float(lat[i]), float(lon[i]), value,
                               tz[i])
    return None


def nearest_airport(lat_deg: float, lon_deg: float, *,
                    max_distance_m: float = MAX_AIRPORT_DISTANCE_M
                    ) -> Airport | None:
    """The airport this position is standing on, or None if it is not at one.

    None is a real answer and the caller must handle it: a track that starts
    on a ramp with no scheduled service, or in the middle of a flight because
    the detector was switched on late, genuinely has no departure airport to
    report, and guessing the nearest one 200 km away would send a query about
    the wrong place and return a confident, wrong flight.
    """
    if not np.isfinite(lat_deg) or not np.isfinite(lon_deg):
        return None
    icao, lat, lon, iata, tz = load_airports()
    distances = haversine(float(lon_deg), float(lat_deg), lon, lat)
    i = int(np.argmin(distances))
    if not np.isfinite(distances[i]) or distances[i] > float(max_distance_m):
        return None
    return Airport(icao[i], float(lat[i]), float(lon[i]), iata[i], tz[i],
                   distance_m=float(distances[i]))
