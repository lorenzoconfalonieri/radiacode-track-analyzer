"""Flight numbers, airlines and the window to search for a flight in.

A boarding pass says DL2618; ADS-B transmits DAL2618; the mapping between
them is a lookup, not a rule (DL to DAL, AA to AAL, BA to BAW share no
pattern), so it lives in a data table. The search window is the other half
of a lookup: the track's own timestamps bound when the flight was in the
air, and a date the user types is a calendar day at the departure airport,
not in UTC, or a late-evening departure is looked for on the wrong day.

These pieces talk to no network and need no trajectory; they only turn what
a user types and what the track recorded into the question a provider is
asked. :mod:`rcta.core.flight` re-exports everything here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from rcta.core.flight_units import _utc

# ---------------------------------------------------------------------------
# flight numbers
# ---------------------------------------------------------------------------

_DESIGNATOR = re.compile(r"^([A-Z][A-Z0-9]|[A-Z0-9][A-Z]|[A-Z]{3})\s*"
                         r"(\d{1,4})([A-Z]?)$")


def load_airlines(path=None) -> dict[str, dict]:
    """The IATA-to-ICAO airline table, keyed by IATA code.

    In data rather than in code because it is a lookup table that grows, and
    because the whole point of the mapping is that it has no rule to encode.
    """
    if path is None:
        raw = resources.files("rcta.data").joinpath("airlines.json").read_text(encoding="utf-8")
    else:
        raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    if data.get("schema_version") != 1:
        raise ValueError("unsupported airlines schema_version "
                         f"{data.get('schema_version')}")
    return {a["iata"]: a for a in data["airlines"]}


@dataclass(frozen=True)
class FlightNumber:
    """A commercial flight number resolved to the callsign ADS-B carries."""

    raw: str
    airline_iata: str
    airline_icao: str
    airline_name: str
    number: str
    suffix: str = ""

    @property
    def known_airline(self) -> bool:
        return bool(self.airline_icao)

    @property
    def callsign(self) -> str:
        """The most likely transmitted callsign, unpadded."""
        return f"{self.airline_icao}{self.number.lstrip('0') or '0'}{self.suffix}"

    @property
    def candidates(self) -> tuple[str, ...]:
        """Every callsign spelling worth matching against, most likely first.

        Zero padding is not consistent between operators -- the same flight
        appears as ``DAL218`` and ``DAL0218`` depending on the flight-management
        system -- so a lookup that tries only one spelling misses flights that
        are present in the data.
        """
        if not self.airline_icao:
            return ()
        stem = self.number.lstrip("0") or "0"
        forms = [stem, stem.zfill(3), stem.zfill(4)]
        seen, out = set(), []
        for form in forms:
            call = f"{self.airline_icao}{form}{self.suffix}"
            if call not in seen:
                seen.add(call)
                out.append(call)
        return tuple(out)

    def describe(self) -> str:
        if not self.known_airline:
            return (f"{self.raw}: airline code {self.airline_iata!r} is not in "
                    "the table; pass the ICAO callsign directly")
        if not self.airline_name:
            return (f"{self.raw}: taken as an ICAO callsign already "
                    f"(operator {self.airline_icao}, flight {self.number})")
        return (f"{self.raw} = {self.airline_name} flight {self.number}, "
                f"callsign {self.callsign}")


def parse_flight_number(text: str, airlines=None) -> FlightNumber:
    """Split ``DL2618`` into its airline and number, and find the callsign.

    A two-character prefix is an IATA code and needs the lookup table; a
    three-letter prefix is already the ICAO designator and is taken as given,
    which is the escape hatch for a carrier the table does not list and for
    the ferry and positioning callsigns that have no IATA number at all.
    """
    raw = str(text).strip()
    match = _DESIGNATOR.match(raw.upper().replace("-", " ").strip())
    if match is None:
        raise ValueError(
            f"cannot read {text!r} as a flight number; expected a form like "
            "DL2618, DL 2618 or the ICAO callsign DAL2618")
    prefix, number, suffix = match.groups()

    if len(prefix) == 3:
        return FlightNumber(raw=raw, airline_iata="", airline_icao=prefix,
                            airline_name="", number=number, suffix=suffix)

    table = load_airlines() if airlines is None else airlines
    entry = table.get(prefix)
    return FlightNumber(
        raw=raw, airline_iata=prefix,
        airline_icao=entry["icao"] if entry else "",
        airline_name=entry["name"] if entry else "",
        number=number, suffix=suffix)


def _calendar_zone(tz: str) -> tuple[object, str, tuple[str, ...]]:
    """A ``tzinfo`` for ``tz``, the name to print, and any warning about it.

    Never raises. A missing zone database is a deployment problem -- Windows
    ships none, so ``zoneinfo`` finds nothing unless the ``tzdata`` wheel is
    installed -- and it must not turn a flight lookup into a traceback. It
    does have to be *said*, though, because the silent fallback to UTC is
    precisely the behaviour this argument exists to remove.
    """
    if not tz:
        return timezone.utc, "UTC", ()
    try:
        return ZoneInfo(tz), tz, ()
    except (ZoneInfoNotFoundError, ValueError):
        return timezone.utc, "UTC", (
            f"no zone database entry for {tz!r}, so the flight date was read "
            "as UTC; on a late-evening departure that is a different calendar "
            "day. Install the 'tzdata' package to fix it",)


def search_window(begin_s: float, end_s: float, date=None, *, tz: str = ""
                  ) -> tuple[float, float, tuple[str, ...]]:
    """The window to search for a flight, and what the date implies about it.

    The window comes from the track, not from the calendar. The detector
    already recorded when it flew, and a provider query derived from the data
    is a couple of requests where a whole-day scan is a dozen -- eleven of
    them covering hours the detector was switched off. So ``date`` is a
    *check*, and it is what the user is actually able to supply confidently:
    they know they flew DL2618 on the 7th, and they do not know the unix
    timestamp of the pushback.

    ``tz`` is the IANA zone of the *departure* airport, and it decides which
    calendar day the track began on. Without it the comparison is made in UTC,
    and UTC is not the clock the date came from: a boarding pass is written in
    local time, so an evening departure is filed under a day that UTC has
    already left. DL2728 pushes back from Salt Lake City at 18:06 on the 10th
    and is airborne from 00:06Z on the 11th; asked in UTC, "the 10th" is a
    day early and finds nothing. The zone comes from
    :mod:`rcta.core.airports`, and is empty for an airport the table could not
    source one for -- in which case this falls back to UTC and says so, rather
    than guessing from longitude, which is wrong by hours across China and
    Spain.

    Whichever zone is used, a disagreement shifts the window by whole days of
    86400 s rather than by calendar days. The model being corrected is a
    detector clock that lost or gained whole days, and such a clock knows
    nothing about a 23-hour spring-forward.

    When the date disagrees with the track it is taken as authoritative and
    the window is shifted. That is the one case where the file's own
    timestamps cannot be believed -- a detector whose real-time clock has
    reset logs a real flight under a wrong date -- and getting a hit at the
    shifted window is itself the evidence that this is what happened, which
    the returned warning says.

    Returns ``(begin_s, end_s, warnings)`` in unix seconds.
    """
    begin, end = float(begin_s), float(end_s)
    if date is None or date == "":
        return begin, end, ()

    if isinstance(date, str):
        try:
            wanted = datetime.strptime(date.strip(), "%Y-%m-%d").date()
        except ValueError:
            raise ValueError(f"cannot read date {date!r}; expected "
                             "YYYY-MM-DD") from None
    elif isinstance(date, datetime):
        wanted = date.date()
    else:
        wanted = date

    zone, zone_name, notes = _calendar_zone(tz)
    recorded = _utc(begin).astimezone(zone).date()
    if wanted == recorded:
        return begin, end, notes

    days = (wanted - recorded).days
    shift = days * 86400.0
    warning = (
        f"the track begins on {recorded} at {zone_name} but the flight date "
        f"given is {wanted}: searching {days:+d} day(s) from the recorded "
        "time. If the flight is found there, the detector's clock was wrong "
        "by that much and so is every timestamp in the file")
    return begin + shift, end + shift, notes + (warning,)


