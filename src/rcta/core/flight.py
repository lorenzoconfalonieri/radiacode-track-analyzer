"""Altitude for a track that has none, and the dose-versus-altitude profile.

The ``.rctrk`` format carries no altitude (see ``docs/FORMAT.md``), which is
awkward, because on an airborne track altitude is the *only* variable that
explains the count rate. Cosmic secondaries are attenuated by the air above
the detector, so the dose rate at cruise is an order of magnitude above the
ground value and every ground-survey assumption in this program -- a stable
background, a localised source -- is false. This module imports the missing
column from an external trajectory and turns it into the one plot that makes
an airborne track interpretable.

Nothing here reaches the network; that is :mod:`rcta.net.flights`, and the
separation is enforced by ``tests/test_architecture.py``. What arrives here is
a :class:`Trajectory`, whether it came from an API, a cached response or a CSV
a user exported by hand, and every check below applies identically to all
three.

Four decisions carry the correctness of the result, and each is the one an
obvious implementation gets wrong.

*The join must be validated against position, not trusted.* A flight number is
a user-supplied string, and the failure mode of getting it wrong is silent: any
trajectory joins onto any track by timestamp and produces a smooth, plausible,
entirely fictional plot. So the trajectory's own latitude and longitude are
interpolated onto the track's clock and compared with the fixes the detector
recorded. Disagreement beyond a tolerance raises :class:`FlightMatchError`
rather than returning a profile, because a wrong altitude axis is worse than
no altitude axis.

*Altitude is never extrapolated and never interpolated across a long gap.*
Outside the trajectory's own span the altitude is NaN, and a sample landing in
a waypoint gap wider than ``max_gap_s`` is NaN too. A ten-minute gap spanning a
top-of-climb, interpolated linearly, invents a smooth ramp through a level-off
and assigns dose from cruise to altitudes the aircraft passed through once.

*Ascent and descent are a control, not a nuisance.* Most of a flight is spent
at one altitude, so a whole-flight scatter is a dense blob at cruise with thin
tails, and any curve through it is set by the tails. Binning the climb and the
descent separately gives two independent measurements of the same profile
hours apart; where they disagree, the disagreement is the error bar that a
single fit does not show. :func:`compare_branches` reports it.

*A poor GPS fix does not invalidate an altitude.* Altitude here comes from
ADS-B, not from the phone, so the 16 % of fixes with accuracy worse than 100 m
still receive a good altitude and still contribute their count rate. They are
excluded from the position check and from nothing else.

Altitude is canonically metres everywhere in this package, as dose rate is
canonically nSv/h. Feet appear only in :meth:`AltitudeProfile.describe` and in
the flight-level labels, because that is how aircrew and every flight tracker
quote it.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from importlib import resources
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

from rcta.core import geo
from rcta.core.track import Track

__all__ = [
    "AltitudeJoin",
    "AltitudeProfile",
    "BRANCH_LABELS",
    "BranchComparison",
    "DEFAULT_ALTITUDE_STEP_M",
    "DEFAULT_FIT_FLOOR_M",
    "DEFAULT_MATCH_TOLERANCE_M",
    "DEFAULT_MAX_GAP_S",
    "FLIGHT_LEVEL_M",
    "FlightMatchError",
    "FlightNumber",
    "FlightVerdict",
    "M_PER_FT",
    "MatchReport",
    "ScaleHeight",
    "Trajectory",
    "bin_by_altitude",
    "compare_branches",
    "fit_scale_height",
    "join_altitude",
    "load_airlines",
    "looks_like_flight",
    "parse_flight_number",
    "search_window",
]

#: Metres per foot, exactly. Altitudes are stored in metres and displayed in
#: both, because a reader who flies thinks in flight levels and a reader who
#: does radiation transport thinks in kilometres of air.
M_PER_FT = 0.3048

#: One flight level is 100 ft of pressure altitude.
FLIGHT_LEVEL_M = 100.0 * M_PER_FT

#: Phase-of-flight names for the codes in :attr:`AltitudeJoin.branch`. Exported
#: alongside every sample so that a reader who bins the data themselves can
#: separate the two traverses without recomputing a vertical rate.
BRANCH_LABELS = {1: "climb", 0: "level", -1: "descent", -128: "unknown"}

#: Vertical rate below which the aircraft is treated as level, in m/s. 1.0 m/s
#: is about 200 ft/min: inside the noise of barometric altitude reported in
#: 25 ft steps, and well below any real climb or descent segment.
_LEVEL_RATE_M_S = 1.0

#: Median ground speed above which a track is confidently airborne. A fast
#: motorway drive peaks near 40 m/s and a high-speed train near 90; 100 m/s
#: sustained as a *median* is not reachable on the ground.
_FLIGHT_MEDIAN_SPEED_M_S = 100.0

#: Ground speed a track must reach at the 95th percentile to be called
#: airborne on the strength of speed alone.
_FLIGHT_P95_SPEED_M_S = 140.0

#: Default tolerance on the median position residual of the join, in metres.
#: Generous on purpose: it exists to catch the wrong flight, the wrong day and
#: the wrong leg -- errors measured in hundreds of kilometres -- and not to
#: audit ADS-B multilateration against a phone GPS. A value tight enough to do
#: the latter would reject good joins over sparse receiver coverage.
DEFAULT_MATCH_TOLERANCE_M = 5000.0

#: Widest waypoint gap that altitude may be interpolated across, in seconds.
DEFAULT_MAX_GAP_S = 300.0

#: Range searched when estimating the detector's clock offset, in seconds.
#: At 233 m/s a one-second offset is 233 m of apparent position residual, so
#: on a fast platform this is worth measuring before blaming the join.
_OFFSET_SEARCH_S = 180.0


def _utc(seconds: float) -> datetime:
    return datetime.fromtimestamp(float(seconds), tz=timezone.utc)


def _fmt_alt(metres: float) -> str:
    """Metres, with the flight level a pilot would recognise beside it."""
    if not np.isfinite(metres):
        return "n/a"
    return f"{metres:.0f} m (FL{metres / FLIGHT_LEVEL_M:.0f})"


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
        raw = resources.files("rcta.data").joinpath("airlines.json").read_text()
    else:
        raw = open(path).read()
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


# ---------------------------------------------------------------------------
# is this even a flight?
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FlightVerdict:
    """Whether a track was recorded airborne, and how sure that is.

    The test has power in one direction only, and saying so is the whole
    reason this is an object rather than a bool. A median ground speed of
    233 m/s cannot be produced on the ground, so ``is_flight`` true is close
    to certain. False means only "not detectable from speed": a glider, a
    holding pattern, a taxi-and-return, or a light aircraft at 50 m/s are all
    genuine flights that look like a fast car. So the flight command warns on
    a false verdict and proceeds, rather than refusing.
    """

    is_flight: bool
    speed_median_m_s: float
    speed_p95_m_s: float
    distance_km: float
    duration_s: float
    reasons: tuple[str, ...] = ()

    def describe(self) -> str:
        verdict = ("airborne" if self.is_flight
                   else "not identifiable as a flight from ground speed alone")
        lines = [f"platform   : {verdict}",
                 f"speed      : median {self.speed_median_m_s:.0f} m/s "
                 f"({self.speed_median_m_s * 3.6:.0f} km/h), "
                 f"p95 {self.speed_p95_m_s:.0f} m/s",
                 f"route      : {self.distance_km:.0f} km in "
                 f"{self.duration_s / 3600.0:.2f} h"]
        lines.extend(f"           : {reason}" for reason in self.reasons)
        return "\n".join(lines)


def looks_like_flight(track: Track, *,
                      median_speed_m_s: float = _FLIGHT_MEDIAN_SPEED_M_S,
                      p95_speed_m_s: float = _FLIGHT_P95_SPEED_M_S
                      ) -> FlightVerdict:
    """Decide from ground speed whether the track was recorded in the air.

    Speed is measured on the jitter-corrected positions, and the median rather
    than the maximum is the primary statistic: a single GPS glitch reaches
    1330 m/s on the bundled sample, so a test on the maximum would call any
    walking survey with one bad fix a flight.
    """
    if track.n < 2:
        return FlightVerdict(is_flight=False, speed_median_m_s=0.0,
                             speed_p95_m_s=0.0, distance_km=0.0, duration_s=0.0,
                             reasons=("track has fewer than two samples",))

    speed = track.speed_m_s[1:]
    median = float(np.median(speed))
    p95 = float(np.percentile(speed, 95))
    distance_km = float(track.distance_along_m[-1]) / 1000.0
    duration = float(track.time_s[-1] - track.time_s[0])

    reasons = []
    by_median = median >= median_speed_m_s
    by_p95 = p95 >= p95_speed_m_s
    if by_median:
        reasons.append(f"median ground speed {median:.0f} m/s is not reachable "
                       "on the ground")
    elif by_p95:
        reasons.append(f"sustained {p95:.0f} m/s at the 95th percentile is "
                       "above any surface vehicle")
    else:
        reasons.append("ground speed is consistent with a surface platform; "
                       "a slow or short flight would look the same, so this "
                       "is not evidence against a flight")
    return FlightVerdict(is_flight=bool(by_median or by_p95),
                         speed_median_m_s=median, speed_p95_m_s=p95,
                         distance_km=distance_km, duration_s=duration,
                         reasons=tuple(reasons))


# ---------------------------------------------------------------------------
# the trajectory
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Trajectory:
    """An external flight path: time, position and altitude, and its origin.

    Struct-of-arrays and read-only for the same reasons as
    :class:`~rcta.core.track.Track`, and sorted by time in ``__post_init__``
    for one more: waypoint lists from an API arrive with duplicate timestamps
    where two receivers reported the same second, and ``np.interp`` on a
    non-monotonic ``xp`` returns silent nonsense rather than raising.

    ``altitude_m`` is allowed to be NaN. A waypoint on the ground often carries
    no barometric altitude at all, and substituting zero would put the aircraft
    at sea level in Denver.
    """

    time_s: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    altitude_m: np.ndarray
    #: "barometric", "geometric" or "unknown". Barometric is the *better*
    #: choice here, not a compromise: it is a pressure measurement, and the
    #: cosmic dose rate is set by the mass of air overhead rather than by
    #: geometric height.
    altitude_kind: str = "unknown"
    on_ground: np.ndarray | None = None
    callsign: str = ""
    icao24: str = ""
    provider: str = ""
    source: str = ""
    fetched_at: datetime | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        cols = {}
        for name in ("time_s", "lat", "lon", "altitude_m"):
            arr = np.asarray(getattr(self, name), dtype=float)
            if arr.ndim != 1:
                raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
            cols[name] = arr

        n = cols["time_s"].size
        if n == 0:
            raise ValueError("a trajectory must contain at least one waypoint")
        for name, arr in cols.items():
            if arr.size != n:
                raise ValueError(f"trajectory column length mismatch: time_s "
                                 f"has {n}, {name} has {arr.size}")
        if not np.isfinite(cols["time_s"]).all():
            raise ValueError("trajectory timestamps must all be finite")

        ground = self.on_ground
        ground = (np.zeros(n, dtype=bool) if ground is None
                  else np.asarray(ground, dtype=bool))
        if ground.size != n:
            raise ValueError(f"on_ground has {ground.size} entries, expected {n}")

        order = np.argsort(cols["time_s"], kind="stable")
        for name in cols:
            cols[name] = cols[name][order]
        ground = ground[order]

        # Collapse duplicate timestamps to the first occurrence. Averaging them
        # would be defensible but hides that the source is duplicating, and
        # np.interp needs strictly increasing xp either way.
        keep = np.concatenate(([True], np.diff(cols["time_s"]) > 0))
        n_dup = int((~keep).sum())
        if n_dup:
            for name in cols:
                cols[name] = cols[name][keep]
            ground = ground[keep]

        for name, arr in cols.items():
            arr = np.ascontiguousarray(arr)
            arr.flags.writeable = False
            object.__setattr__(self, name, arr)
        ground = np.ascontiguousarray(ground)
        ground.flags.writeable = False
        object.__setattr__(self, "on_ground", ground)

        if n_dup:
            object.__setattr__(self, "warnings", tuple(self.warnings) + (
                f"{n_dup} duplicate waypoint timestamp(s) collapsed",))

    # ---- geometry ---------------------------------------------------------

    @property
    def n(self) -> int:
        return int(self.time_s.size)

    @property
    def start(self) -> datetime:
        return _utc(self.time_s[0])

    @property
    def end(self) -> datetime:
        return _utc(self.time_s[-1])

    @property
    def span_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0])

    @cached_property
    def has_altitude(self) -> np.ndarray:
        return np.isfinite(self.altitude_m)

    @cached_property
    def max_gap_s(self) -> float:
        """Widest interval between consecutive altitude-bearing waypoints."""
        t = self.time_s[self.has_altitude]
        return float(np.diff(t).max()) if t.size > 1 else 0.0

    @cached_property
    def altitude_ceiling_m(self) -> float:
        alt = self.altitude_m[self.has_altitude]
        return float(alt.max()) if alt.size else float("nan")

    @cached_property
    def vertical_rate_m_s(self) -> np.ndarray:
        """Rate of climb at each waypoint, NaN where altitude is missing.

        A central difference over the neighbouring *altitude-bearing*
        waypoints, so a missing altitude in the middle of a climb widens the
        interval rather than producing a spike or a hole.
        """
        rate = np.full(self.n, np.nan)
        m = self.has_altitude
        if m.sum() < 2:
            return rate
        t, alt = self.time_s[m], self.altitude_m[m]
        rate[m] = np.gradient(alt, t)
        return rate

    @cached_property
    def unwrapped_lon(self) -> np.ndarray:
        """Longitude made continuous, so interpolation cannot cross zero wrong.

        A flight from 179.9 E to 179.9 W steps by 359.8 degrees in the raw
        column, and linear interpolation across that step sweeps the whole
        globe backwards in one waypoint interval. Unwrapping in radians costs
        nothing and removes the case entirely.
        """
        return np.degrees(np.unwrap(np.radians(self.lon)))

    def describe(self) -> str:
        label = self.callsign.strip() or self.icao24 or "(unidentified)"
        lines = [
            f"flight     : {label}"
            + (f"   icao24 {self.icao24}" if self.icao24 and self.callsign
               else ""),
            f"waypoints  : {self.n}  ({int(self.has_altitude.sum())} with "
            f"altitude)",
            f"span       : {self.start:%Y-%m-%d %H:%M:%S}Z .. "
            f"{self.end:%H:%M:%S}Z   ({self.span_s / 60.0:.0f} min)",
            f"ceiling    : {_fmt_alt(self.altitude_ceiling_m)}   "
            f"[{self.altitude_kind} altitude]",
            f"cadence    : median "
            f"{np.median(np.diff(self.time_s)) if self.n > 1 else 0:.0f} s, "
            f"widest gap {self.max_gap_s:.0f} s",
            f"source     : {self.provider or 'unknown'}"
            + (f"   {self.source}" if self.source else ""),
        ]
        lines.extend(f"warning    : {w}" for w in self.warnings)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# the join
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MatchReport:
    """Whether the trajectory and the track describe the same journey.

    The residual quoted is the *median*, because the mean is dominated by the
    handful of samples whose own GPS fix was hundreds of metres out and by the
    ends of the track where the aircraft is manoeuvring between sparse
    waypoints. A wrong flight is wrong by hundreds of kilometres at the median;
    there is no need for a delicate statistic to see it.
    """

    n_compared: int
    residual_median_m: float
    residual_p95_m: float
    residual_max_m: float
    tolerance_m: float
    clock_offset_s: float
    residual_at_offset_m: float
    matched: bool
    forced: bool = False

    @property
    def offset_matters(self) -> bool:
        """True when aligning the clocks would materially improve the fit."""
        return (abs(self.clock_offset_s) >= 2.0
                and self.residual_at_offset_m < 0.5 * self.residual_median_m)

    def describe(self) -> str:
        if self.n_compared == 0:
            return ("position match : NOT CHECKED\n"
                    "  no sample had both a usable fix and a trajectory "
                    "position to compare it with")
        verdict = "MISMATCH" if not self.matched else "ok"
        if self.matched and self.forced:
            verdict = "MISMATCH, overridden"
        lines = [f"position match : {verdict}",
                 f"  median residual {self.residual_median_m / 1000.0:.2f} km, "
                 f"p95 {self.residual_p95_m / 1000.0:.2f} km, "
                 f"max {self.residual_max_m / 1000.0:.2f} km",
                 f"  over {self.n_compared} compared sample(s), tolerance "
                 f"{self.tolerance_m / 1000.0:.1f} km"]
        if self.offset_matters:
            lines.append(
                f"  a {self.clock_offset_s:+.0f} s clock offset would reduce "
                f"the median to {self.residual_at_offset_m / 1000.0:.2f} km; "
                "the detector clock is probably not GPS-locked. Re-run with "
                f"--time-offset {self.clock_offset_s:+.0f}")
        if not self.matched:
            lines.append("  this is not the flight, the day or the leg that "
                         "produced this track")
        return "\n".join(lines)


class FlightMatchError(ValueError):
    """Raised when a trajectory does not describe the track it was joined to.

    Carries the :class:`MatchReport` so the caller can print the residuals it
    is refusing on, rather than a bare message that leaves the user unable to
    tell a wrong flight number from a wrong day.
    """

    def __init__(self, report: MatchReport) -> None:
        super().__init__(
            f"trajectory does not match this track: median position residual "
            f"{report.residual_median_m / 1000.0:.1f} km exceeds the "
            f"{report.tolerance_m / 1000.0:.1f} km tolerance")
        self.report = report


@dataclass(frozen=True)
class AltitudeJoin:
    """A track with an altitude column imported from a trajectory."""

    track: Track
    #: Length ``track.n``, metres, NaN where no altitude could be justified.
    altitude_m: np.ndarray
    altitude_kind: str
    #: -1 descent, 0 level, +1 climb, and -128 where altitude is unknown.
    branch: np.ndarray
    vertical_rate_m_s: np.ndarray
    match: MatchReport
    trajectory: Trajectory
    max_gap_s: float
    time_offset_s: float
    n_before: int = 0
    n_after: int = 0
    n_gap_masked: int = 0
    widest_bridged_gap_s: float = 0.0
    widest_bridged_change_m: float = 0.0

    @cached_property
    def joined_mask(self) -> np.ndarray:
        """Samples that received an altitude *and* a believable count rate."""
        return np.isfinite(self.altitude_m) & self.track.valid_mask

    @property
    def n_joined(self) -> int:
        return int(self.joined_mask.sum())

    @property
    def coverage(self) -> float:
        return self.n_joined / max(self.track.n, 1)

    @property
    def altitude_span_m(self) -> tuple[float, float]:
        alt = self.altitude_m[self.joined_mask]
        if alt.size == 0:
            return (float("nan"), float("nan"))
        return (float(alt.min()), float(alt.max()))

    def branch_mask(self, branch: str) -> np.ndarray:
        """Samples belonging to a named phase of flight."""
        codes = {"all": None, "climb": 1, "level": 0, "cruise": 0, "descent": -1}
        if branch not in codes:
            raise ValueError(f"unknown branch {branch!r}; expected one of "
                             f"{sorted(codes)}")
        if codes[branch] is None:
            return self.joined_mask
        return self.joined_mask & (self.branch == codes[branch])

    def describe(self) -> str:
        lo, hi = self.altitude_span_m
        n_climb = int(self.branch_mask("climb").sum())
        n_level = int(self.branch_mask("level").sum())
        n_descent = int(self.branch_mask("descent").sum())
        lines = [
            f"joined     : {self.n_joined} of {self.track.n} samples "
            f"({self.coverage * 100.0:.1f} %) carry an altitude",
            f"altitude   : {_fmt_alt(lo)} .. {_fmt_alt(hi)}   "
            f"[{self.altitude_kind}]",
            f"phases     : {n_climb} climbing, {n_level} level, "
            f"{n_descent} descending",
        ]
        dropped = []
        if self.n_before or self.n_after:
            dropped.append(f"{self.n_before + self.n_after} outside the "
                           "trajectory's own span (never extrapolated)")
        if self.n_gap_masked:
            dropped.append(f"{self.n_gap_masked} inside a waypoint gap wider "
                           f"than {self.max_gap_s:.0f} s")
        if dropped:
            lines.append("unjoined   : " + "; ".join(dropped))
        if self.widest_bridged_gap_s > 0:
            lines.append(
                f"bridged    : widest gap actually interpolated across is "
                f"{self.widest_bridged_gap_s:.0f} s, over which the aircraft "
                f"changed altitude by {self.widest_bridged_change_m:.0f} m")
        if self.time_offset_s:
            lines.append(f"clock      : track times shifted by "
                         f"{self.time_offset_s:+.0f} s before joining")
        lines.append(self.match.describe())
        return "\n".join(lines)


def _median_residual(track: Track, trajectory: Trajectory, shift: float,
                     mask: np.ndarray) -> float:
    """Median great-circle residual with the track clock shifted by ``shift``."""
    t = track.time_s[mask] + shift
    lat = np.interp(t, trajectory.time_s, trajectory.lat,
                    left=np.nan, right=np.nan)
    lon = np.interp(t, trajectory.time_s, trajectory.unwrapped_lon,
                    left=np.nan, right=np.nan)
    residual = geo.haversine(track.lon[mask], track.lat[mask], lon, lat)
    residual = residual[np.isfinite(residual)]
    return float(np.median(residual)) if residual.size else float("inf")


def _best_clock_offset(track: Track, trajectory: Trajectory,
                       mask: np.ndarray, base: float) -> tuple[float, float]:
    """The additional clock shift that best aligns track and trajectory.

    Coarse then fine rather than a search at 1 s resolution over the whole
    range, because the residual as a function of offset is smooth and
    single-troughed for any platform that does not retrace its own path, and
    the coarse pass costs 37 evaluations instead of 361.

    Returned for reporting only. Applying it automatically would let a clock
    search absorb a genuine mismatch and hand back a confident plot.
    """
    best_shift, best_value = 0.0, _median_residual(track, trajectory, base, mask)
    for step, span, centre in ((10.0, _OFFSET_SEARCH_S, 0.0),
                               (1.0, 10.0, None)):
        origin = best_shift if centre is None else centre
        for delta in np.arange(origin - span, origin + span + step, step):
            value = _median_residual(track, trajectory, base + float(delta), mask)
            if value < best_value:
                best_shift, best_value = float(delta), value
    return best_shift, best_value


def join_altitude(track: Track, trajectory: Trajectory, *,
                  max_gap_s: float = DEFAULT_MAX_GAP_S,
                  tolerance_m: float = DEFAULT_MATCH_TOLERANCE_M,
                  time_offset_s: float = 0.0,
                  force_match: bool = False,
                  level_rate_m_s: float = _LEVEL_RATE_M_S) -> AltitudeJoin:
    """Interpolate a trajectory's altitude onto a track's samples.

    ``time_offset_s`` is added to the track's clock before joining, for a
    detector whose clock is not GPS-locked. The offset that would minimise the
    residual is measured and reported regardless, but never applied on its own
    initiative: silently sliding the clock until the data fits is how a wrong
    flight is made to look right.

    Raises :class:`FlightMatchError` when the trajectory's positions disagree
    with the track's own fixes beyond ``tolerance_m``, unless ``force_match``.
    """
    if max_gap_s <= 0:
        raise ValueError("max_gap_s must be positive")

    alt_ok = trajectory.has_altitude
    if int(alt_ok.sum()) < 2:
        raise ValueError(
            "trajectory carries fewer than two waypoints with an altitude, so "
            "there is nothing to interpolate")

    t = track.time_s + float(time_offset_s)
    wp_t = trajectory.time_s[alt_ok]
    wp_alt = trajectory.altitude_m[alt_ok]
    wp_rate = trajectory.vertical_rate_m_s[alt_ok]

    altitude = np.interp(t, wp_t, wp_alt, left=np.nan, right=np.nan)
    rate = np.interp(t, wp_t, wp_rate, left=np.nan, right=np.nan)

    n_before = int((t < wp_t[0]).sum())
    n_after = int((t > wp_t[-1]).sum())

    # Which waypoint interval each sample falls in, and how wide it is. A
    # sample inside an interval wider than max_gap_s gets no altitude: the
    # linear ramp across a long gap is an invention, and it is worst exactly
    # where it matters most, at a top of climb.
    inside = np.isfinite(altitude)
    right = np.clip(np.searchsorted(wp_t, t, side="right"), 1, wp_t.size - 1)
    gap = wp_t[right] - wp_t[right - 1]
    change = np.abs(wp_alt[right] - wp_alt[right - 1])
    # A sample landing exactly on a waypoint is not interpolated at all, so a
    # wide interval *after* it says nothing about its altitude. This is not a
    # corner case: both clocks are integer seconds, so on a 30 s cadence
    # roughly one sample in fifteen lands on a waypoint exactly, and masking
    # them would throw away the most accurate altitudes in the join.
    exact = inside & (t == wp_t[right - 1])
    too_wide = inside & (gap > float(max_gap_s)) & ~exact
    n_gap_masked = int(too_wide.sum())
    altitude = np.where(too_wide, np.nan, altitude)
    rate = np.where(too_wide, np.nan, rate)

    bridged = np.isfinite(altitude)
    widest_gap = float(gap[bridged].max()) if bridged.any() else 0.0
    widest_change = float(change[bridged].max()) if bridged.any() else 0.0

    branch = np.full(track.n, -128, dtype=np.int8)
    known = np.isfinite(rate)
    branch[known] = np.where(rate[known] > float(level_rate_m_s), 1,
                             np.where(rate[known] < -float(level_rate_m_s),
                                      -1, 0))

    # ---- the validation --------------------------------------------------
    # Only position-valid samples: a fix with 397 m of accuracy carries a
    # perfectly good count rate but is no evidence about which flight this is.
    check = track.position_mask & np.isfinite(
        np.interp(t, trajectory.time_s, trajectory.lat,
                  left=np.nan, right=np.nan))
    n_compared = int(check.sum())

    if n_compared == 0:
        report = MatchReport(n_compared=0, residual_median_m=float("nan"),
                             residual_p95_m=float("nan"),
                             residual_max_m=float("nan"),
                             tolerance_m=float(tolerance_m),
                             clock_offset_s=0.0,
                             residual_at_offset_m=float("nan"),
                             matched=bool(force_match), forced=bool(force_match))
        if not force_match:
            raise FlightMatchError(report)
    else:
        lat_i = np.interp(t[check], trajectory.time_s, trajectory.lat,
                          left=np.nan, right=np.nan)
        lon_i = np.interp(t[check], trajectory.time_s,
                          trajectory.unwrapped_lon, left=np.nan, right=np.nan)
        residual = geo.haversine(track.lon[check], track.lat[check],
                                 lon_i, lat_i)
        residual = residual[np.isfinite(residual)]

        best_shift, best_value = _best_clock_offset(
            track, trajectory, check, float(time_offset_s))

        median = float(np.median(residual)) if residual.size else float("inf")
        matched = median <= float(tolerance_m)
        report = MatchReport(
            n_compared=n_compared, residual_median_m=median,
            residual_p95_m=float(np.percentile(residual, 95))
            if residual.size else float("nan"),
            residual_max_m=float(residual.max()) if residual.size
            else float("nan"),
            tolerance_m=float(tolerance_m), clock_offset_s=best_shift,
            residual_at_offset_m=best_value,
            matched=bool(matched or force_match),
            forced=bool(force_match and not matched))
        if not matched and not force_match:
            raise FlightMatchError(report)

    altitude = np.ascontiguousarray(altitude)
    altitude.flags.writeable = False
    rate = np.ascontiguousarray(rate)
    rate.flags.writeable = False
    branch = np.ascontiguousarray(branch)
    branch.flags.writeable = False

    return AltitudeJoin(
        track=track, altitude_m=altitude,
        altitude_kind=trajectory.altitude_kind, branch=branch,
        vertical_rate_m_s=rate, match=report, trajectory=trajectory,
        max_gap_s=float(max_gap_s), time_offset_s=float(time_offset_s),
        n_before=n_before, n_after=n_after, n_gap_masked=n_gap_masked,
        widest_bridged_gap_s=widest_gap, widest_bridged_change_m=widest_change)


# ---------------------------------------------------------------------------
# dose against altitude
# ---------------------------------------------------------------------------

#: Default altitude bin width, in metres. 500 m is about FL16, comfortably
#: wider than the barometric quantisation and narrow enough that the climb
#: through the troposphere lands in twenty distinct bins.
DEFAULT_ALTITUDE_STEP_M = 500.0

#: Bins below this many samples are reported but excluded from the fit. Two
#: samples give a standard error, and a standard error from two samples is
#: worse than no point at all when it lands at the end of the lever arm.
_FIT_MIN_SAMPLES = 3

#: Altitude above which the cosmic component dominates the CsI reading, in
#: metres. Below it the terrestrial gamma from soil and building materials is
#: comparable or larger, and it does not vary with altitude at all, so a pure
#: exponential fitted through ground-level points is fitting two physical
#: processes with one parameter. On a synthetic profile with a realistic
#: terrestrial pedestal the floor moves the recovered scale height from 4 %
#: high to within 0.2 % of truth; with a large one, from 28 % to 14 %. It
#: reduces the bias and does not abolish it.
DEFAULT_FIT_FLOOR_M = 3000.0


@dataclass(frozen=True)
class AltitudeProfile:
    """Dose and count rate aggregated onto a regular altitude grid.

    Binned, never interpolated, for the reason given in
    :func:`rcta.core.profile.resample_by_distance`: an occupied-looking bin at
    an altitude the aircraft never held is an invention. Empty bins are NaN
    and the plot breaks across them.

    Two error bars are carried, and the difference between them is the point.
    ``counting_sem_cps`` is what Poisson statistics allow given the counts
    actually accumulated in the bin; ``observed_sem_cps`` is the scatter the
    samples really showed. On a flight the second is reliably larger than the
    first, and that is not a defect: a 500 m altitude bin at cruise also spans
    an hour of flying and a thousand kilometres of geomagnetic latitude, so
    the count rate genuinely varies within it. Quoting only the counting error
    would claim a precision the measurement does not have.
    """

    altitude_m: np.ndarray
    dose_rate: np.ndarray
    count_rate: np.ndarray
    counting_sem_cps: np.ndarray
    observed_sem_cps: np.ndarray
    n_samples: np.ndarray
    live_time_s: np.ndarray
    step_m: float
    statistic: str
    branch: str
    dose_per_count: float
    integration_time_s: float
    altitude_kind: str = "unknown"

    @property
    def occupied(self) -> np.ndarray:
        return self.n_samples > 0

    @property
    def n_empty(self) -> int:
        return int((~self.occupied).sum())

    @cached_property
    def sem_cps(self) -> np.ndarray:
        """The error bar to plot: the larger of counting and observed.

        Taking the larger rather than the counting one alone is the
        conservative choice, and on this data it is also the honest one --
        real variation inside a bin is scatter about any curve fitted through
        the bin centres just as surely as counting noise is.
        """
        return np.fmax(self.counting_sem_cps, self.observed_sem_cps)

    @cached_property
    def dose_sem(self) -> np.ndarray:
        return self.sem_cps * self.dose_per_count

    @property
    def excess_scatter(self) -> float:
        """Median ratio of observed to counting error over occupied bins.

        1.0 means the bins are counting-limited. Above about 2 the spread
        inside a bin is dominated by something real -- latitude, solar
        activity over a long flight, or an altitude bin too wide.
        """
        m = self.occupied & (self.counting_sem_cps > 0)
        if not m.any():
            return float("nan")
        return float(np.median(self.observed_sem_cps[m]
                               / self.counting_sem_cps[m]))

    @property
    def scatter_note(self) -> str:
        """What the observed-to-counting ratio is saying, in words.

        The low branch is the one worth having. Samples cannot scatter *less*
        than Poisson allows, so a ratio well below 1 does not mean the data
        are unusually good -- it means the assumed integration time is too
        short. The counts behind each reading were accumulated over longer
        than ``effective_integration_time_s`` says, every Poisson error bar
        here is correspondingly overstated, and so is every detection limit
        elsewhere in the program. It is the same systematic the track loader
        warns about, showing up somewhere it can actually be measured.
        """
        ratio = self.excess_scatter
        if not np.isfinite(ratio):
            return "no bin carries enough counts to compare the two errors"
        if ratio < 0.7:
            return (f"scatter inside bins is {ratio:.1f}x the counting error "
                    "-- below Poisson, which is not possible for real counts. "
                    f"The assumed integration time of "
                    f"{self.integration_time_s:.1f} s per sample is too short, "
                    "so these error bars are overstated")
        if ratio < 1.5:
            return (f"scatter inside bins is {ratio:.1f}x the counting error "
                    "-- bins are counting-limited")
        return (f"scatter inside bins is {ratio:.1f}x the counting error -- "
                "real variation dominates, so the error bars are the observed "
                "spread, not Poisson")

    def describe(self) -> str:
        m = self.occupied
        if not m.any():
            return f"altitude profile ({self.branch}) is empty"
        lines = [
            f"{int(m.sum())} occupied bins of {self.step_m:.0f} m over "
            f"{self.altitude_m[m].min():.0f} .. {self.altitude_m[m].max():.0f} m "
            f"({self.branch}, {self.statistic}); {self.n_empty} empty",
            f"dose rate {np.nanmin(self.dose_rate[m]):.1f} .. "
            f"{np.nanmax(self.dose_rate[m]):.1f} nSv/h over "
            f"{int(self.n_samples.sum())} samples "
            f"({self.live_time_s.sum() / 60.0:.0f} min of live time)",
            self.scatter_note,
        ]
        if self.statistic != "mean":
            lines.append(f"note: bin values are the {self.statistic}, but the "
                         "error bars describe the mean")
        return "\n".join(lines)


def bin_by_altitude(join: AltitudeJoin, *,
                    step_m: float = DEFAULT_ALTITUDE_STEP_M,
                    statistic: str = "mean",
                    branch: str = "all") -> AltitudeProfile:
    """Aggregate the joined track onto an altitude grid.

    The grid is anchored on a multiple of ``step_m`` rather than on the lowest
    observed altitude, so profiles from different flights, or from the climb
    and the descent of one flight, share bin edges and can be compared bin for
    bin without resampling.
    """
    if step_m <= 0:
        raise ValueError("step_m must be positive")
    reducers = {"mean": np.mean, "median": np.median,
                "max": np.max, "min": np.min}
    if statistic not in reducers:
        raise ValueError(f"unknown statistic {statistic!r}; expected one of "
                         f"{sorted(reducers)}")
    reduce = reducers[statistic]

    mask = join.branch_mask(branch)
    track = join.track
    dt = track.effective_integration_time_s

    if not mask.any():
        empty = np.zeros(0)
        return AltitudeProfile(
            altitude_m=empty, dose_rate=empty, count_rate=empty,
            counting_sem_cps=empty, observed_sem_cps=empty,
            n_samples=np.zeros(0, dtype=np.int64), live_time_s=empty,
            step_m=float(step_m), statistic=statistic, branch=branch,
            dose_per_count=track.dose_per_count, integration_time_s=dt,
            altitude_kind=join.altitude_kind)

    alt = join.altitude_m[mask]
    dose = track.dose_rate[mask]
    cps = track.count_rate[mask]

    lo = math.floor(float(alt.min()) / step_m) * step_m
    hi = math.ceil(float(alt.max()) / step_m) * step_m
    n_bins = max(int(round((hi - lo) / step_m)), 1)
    edges = lo + np.arange(n_bins + 1, dtype=float) * step_m
    index = np.clip(np.searchsorted(edges, alt, side="right") - 1, 0, n_bins - 1)

    dose_out = np.full(n_bins, np.nan)
    cps_out = np.full(n_bins, np.nan)
    counting = np.full(n_bins, np.nan)
    observed = np.full(n_bins, np.nan)
    n_samples = np.zeros(n_bins, dtype=np.int64)
    live = np.zeros(n_bins)

    order = np.argsort(index, kind="stable")
    sorted_index = index[order]
    starts = np.searchsorted(sorted_index, np.arange(n_bins), side="left")
    stops = np.searchsorted(sorted_index, np.arange(n_bins), side="right")
    for b in range(n_bins):
        rows = order[starts[b]:stops[b]]
        if rows.size == 0:
            continue
        n_samples[b] = rows.size
        live[b] = rows.size * dt
        dose_out[b] = reduce(dose[rows])
        cps_out[b] = reduce(cps[rows])
        # Poisson on the counts the bin actually accumulated. sqrt(N) counts
        # over N/live cps means the standard error of the mean rate is
        # sqrt(N)/live, which falls as the bin fills.
        total_counts = float(cps[rows].sum() * dt)
        counting[b] = math.sqrt(max(total_counts, 0.0)) / live[b]
        observed[b] = (float(np.std(cps[rows], ddof=1)) / math.sqrt(rows.size)
                       if rows.size > 1 else counting[b])

    return AltitudeProfile(
        altitude_m=edges[:-1] + step_m / 2.0, dose_rate=dose_out,
        count_rate=cps_out, counting_sem_cps=counting,
        observed_sem_cps=observed, n_samples=n_samples, live_time_s=live,
        step_m=float(step_m), statistic=statistic, branch=branch,
        dose_per_count=track.dose_per_count, integration_time_s=dt,
        altitude_kind=join.altitude_kind)


# ---------------------------------------------------------------------------
# the exponential
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScaleHeight:
    """An exponential fitted to dose rate against altitude.

    Cosmic secondaries are attenuated by the air above the detector, so in the
    troposphere the dose rate goes as ``exp(h / H)`` with an e-folding height
    of roughly 2 km. ``H`` is therefore the one number that summarises the
    whole profile, and its familiar form is the doubling distance: about
    1.5 km of climb for twice the dose rate.

    ``chi2_per_dof`` is reported because it is the only thing here that can
    say the model is wrong. A scale height quoted from a fit with
    ``chi2/dof = 40`` is a number, not a measurement -- it means the profile
    is not a single exponential, which above the tropopause it genuinely is
    not, since the atmospheric depth stops increasing the way the model
    assumes.
    """

    constrained: bool
    scale_height_m: float
    scale_height_sd_m: float
    doubling_m: float
    reference_altitude_m: float
    dose_at_reference: float
    r_squared: float
    chi2_per_dof: float
    n_bins: int
    floor_m: float
    fitted_low_m: float
    fitted_high_m: float
    branch: str = "all"
    note: str = ""

    def describe(self) -> str:
        if not self.constrained:
            return f"scale height : not constrained -- {self.note}"
        lines = [
            f"scale height : {self.scale_height_m / 1000.0:.2f} +- "
            f"{self.scale_height_sd_m / 1000.0:.2f} km e-folding",
            f"             : dose rate doubles every "
            f"{self.doubling_m:.0f} m of climb",
            f"             : {self.dose_at_reference:.0f} nSv/h at "
            f"{_fmt_alt(self.reference_altitude_m)}",
            f"fitted over  : {self.n_bins} bins, "
            f"{self.fitted_low_m:.0f} .. {self.fitted_high_m:.0f} m "
            f"(floor {self.floor_m:.0f} m, branch {self.branch})",
            f"goodness     : R2 = {self.r_squared:.3f}, "
            f"chi2/dof = {self.chi2_per_dof:.1f}"
            + ("   <- a single exponential does not describe this profile"
               if self.chi2_per_dof > 5.0 else ""),
        ]
        if self.note:
            lines.append(f"caveat       : {self.note}")
        return "\n".join(lines)


def fit_scale_height(profile: AltitudeProfile, *,
                     floor_m: float = DEFAULT_FIT_FLOOR_M,
                     min_samples: int = _FIT_MIN_SAMPLES,
                     min_span_m: float = 2000.0) -> ScaleHeight:
    """Weighted least squares of ``log(dose)`` against altitude.

    Fitted to the *binned* profile rather than to the samples, and that is the
    substantive choice here. Most of a flight is spent within a few hundred
    metres of one cruise altitude, so a per-sample fit is weighted by dwell
    time: thousands of points at FL350 and a few dozen through the climb. The
    slope is then set almost entirely by how the single densest altitude
    compares with the sparse ends, which is a two-point measurement wearing
    the error bar of ten thousand samples. One point per occupied altitude bin
    weights the profile by altitude coverage, which is what the question asks
    about.

    ``floor_m`` excludes the low-altitude bins where terrestrial gamma is
    comparable to the cosmic component; see :data:`DEFAULT_FIT_FLOOR_M`.

    The method recovers a known scale height to within about 3 %, and is
    biased low by that much even on a noiseless profile with no terrestrial
    term at all, because the mean of a convex function over a bin exceeds its
    value at the bin centre. Quote the result to two significant figures.
    """
    m = (profile.occupied & (profile.n_samples >= int(min_samples))
         & (profile.altitude_m >= float(floor_m))
         & np.isfinite(profile.dose_rate) & (profile.dose_rate > 0)
         & np.isfinite(profile.dose_sem) & (profile.dose_sem > 0))
    n_bins = int(m.sum())

    def unconstrained(note: str) -> ScaleHeight:
        return ScaleHeight(
            constrained=False, scale_height_m=float("nan"),
            scale_height_sd_m=float("nan"), doubling_m=float("nan"),
            reference_altitude_m=float("nan"),
            dose_at_reference=float("nan"), r_squared=float("nan"),
            chi2_per_dof=float("nan"), n_bins=n_bins, floor_m=float(floor_m),
            fitted_low_m=float("nan"), fitted_high_m=float("nan"),
            branch=profile.branch, note=note)

    if n_bins < 3:
        return unconstrained(
            f"only {n_bins} usable altitude bin(s) above {floor_m:.0f} m; "
            "three are needed for a slope with an error bar")

    x = profile.altitude_m[m]
    y = np.log(profile.dose_rate[m])
    # Uncertainty on a logarithm is the relative uncertainty on the value.
    sigma = profile.dose_sem[m] / profile.dose_rate[m]
    span = float(x.max() - x.min())
    if span < float(min_span_m):
        return unconstrained(
            f"the fitted bins span only {span:.0f} m of altitude; a scale "
            f"height of order 2000 m cannot be measured over less than "
            f"{min_span_m:.0f} m")

    w = 1.0 / sigma ** 2
    s_w = w.sum()
    s_x = float((w * x).sum())
    s_y = float((w * y).sum())
    s_xx = float((w * x * x).sum())
    s_xy = float((w * x * y).sum())
    denom = s_w * s_xx - s_x ** 2
    if denom <= 0:
        return unconstrained("the altitude bins are degenerate")

    slope = (s_w * s_xy - s_x * s_y) / denom
    intercept = (s_xx * s_y - s_x * s_xy) / denom
    slope_var = s_w / denom

    if slope <= 0:
        return unconstrained(
            "the fitted dose rate falls with altitude, which no cosmic "
            "profile does; check the altitude column and the join")

    # H = 1/slope, so sd(H) = sd(slope) / slope^2 by the delta method.
    scale_height = 1.0 / slope
    scale_height_sd = math.sqrt(slope_var) / slope ** 2

    model = intercept + slope * x
    chi2 = float((w * (y - model) ** 2).sum())
    dof = max(n_bins - 2, 1)
    y_bar = s_y / s_w
    ss_tot = float((w * (y - y_bar) ** 2).sum())
    r_squared = 1.0 - chi2 / ss_tot if ss_tot > 0 else float("nan")

    reference = float(0.5 * (x.min() + x.max()))
    notes = []
    if profile.excess_scatter > 2.0:
        notes.append(
            f"scatter inside the altitude bins is {profile.excess_scatter:.1f}x "
            "the counting error, so the error bars -- and this uncertainty -- "
            "are set by real variation, not by statistics")
    notes.append("bins from the same climb are not independent measurements, "
                 "so the quoted sd understates the real uncertainty; the "
                 "climb-versus-descent comparison is the empirical error bar")
    if float(x.max()) > 11000.0:
        notes.append("part of the fitted range is above the tropopause, where "
                     "a single exponential no longer holds")

    return ScaleHeight(
        constrained=True, scale_height_m=float(scale_height),
        scale_height_sd_m=float(scale_height_sd),
        doubling_m=float(scale_height * math.log(2.0)),
        reference_altitude_m=reference,
        dose_at_reference=float(math.exp(intercept + slope * reference)),
        r_squared=float(r_squared), chi2_per_dof=float(chi2 / dof),
        n_bins=n_bins, floor_m=float(floor_m),
        fitted_low_m=float(x.min()), fitted_high_m=float(x.max()),
        branch=profile.branch, note="; ".join(notes))


# ---------------------------------------------------------------------------
# the control: does the climb agree with the descent?
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BranchComparison:
    """Climb against descent, bin for bin: two passes through one atmosphere.

    This is the closest thing a single flight offers to a repeat measurement.
    The aircraft traverses the same altitudes twice, hours apart and a
    continent away, so agreement between the two branches is evidence that the
    profile is a function of altitude, and disagreement bounds how much of it
    is something else -- geomagnetic latitude, a solar event, or a join that
    is quietly wrong at one end.

    The threshold is Bonferroni-corrected over the number of shared bins for
    the same reason the anomaly scan is: comparing twenty bins at 2 sigma and
    reporting the worst one manufactures a disagreement from clean data.
    """

    climb: AltitudeProfile
    descent: AltitudeProfile
    altitude_m: np.ndarray
    difference: np.ndarray
    sigma: np.ndarray
    n_shared: int
    threshold_sigma: float
    max_abs_z: float
    median_abs_difference: float
    median_relative_difference: float
    consistent: bool
    note: str = ""

    def describe(self) -> str:
        if self.n_shared == 0:
            return ("climb vs descent : no shared altitude bin -- "
                    + (self.note or "one branch was not sampled"))
        verdict = ("consistent" if self.consistent
                   else "INCONSISTENT beyond counting statistics")
        lines = [
            f"climb vs descent : {verdict} over {self.n_shared} shared bin(s)",
            f"  median difference {self.median_abs_difference:.1f} nSv/h "
            f"({self.median_relative_difference * 100.0:.1f} % of the level)",
            f"  worst bin {self.max_abs_z:.2f} sigma against a "
            f"{self.threshold_sigma:.2f} sigma threshold "
            f"(Bonferroni over {self.n_shared} bins)",
        ]
        if not self.consistent:
            lines.append("  the profile is not a function of altitude alone; "
                         "treat the fitted scale height as a description of "
                         "this flight, not a measurement of the atmosphere")
        if self.note:
            lines.append(f"  {self.note}")
        return "\n".join(lines)


def compare_branches(join: AltitudeJoin, *, step_m: float = 1000.0,
                     alpha: float = 0.05,
                     min_samples: int = _FIT_MIN_SAMPLES) -> BranchComparison:
    """Bin the climb and the descent separately and compare them bin for bin.

    Wider bins than the headline profile by default: the climb and descent
    together occupy a few minutes of a multi-hour flight, so at 500 m most
    bins would hold one or two samples on at least one branch and the
    comparison would be dominated by bins too thin to compare.
    """
    from rcta.core.anomalies import bonferroni_k

    climb = bin_by_altitude(join, step_m=step_m, branch="climb")
    descent = bin_by_altitude(join, step_m=step_m, branch="descent")

    def empty(note: str) -> BranchComparison:
        return BranchComparison(
            climb=climb, descent=descent, altitude_m=np.zeros(0),
            difference=np.zeros(0), sigma=np.zeros(0), n_shared=0,
            threshold_sigma=float("nan"), max_abs_z=float("nan"),
            median_abs_difference=float("nan"),
            median_relative_difference=float("nan"), consistent=True, note=note)

    if climb.altitude_m.size == 0 or descent.altitude_m.size == 0:
        missing = [name for name, p in (("climb", climb), ("descent", descent))
                   if p.altitude_m.size == 0]
        return empty(f"no {' or '.join(missing)} samples were joined")

    # Bin edges are anchored on multiples of step_m, so the two grids align on
    # a common lattice and can be intersected by altitude rather than by index.
    shared = np.intersect1d(climb.altitude_m[climb.n_samples >= min_samples],
                            descent.altitude_m[descent.n_samples >= min_samples])
    if shared.size == 0:
        return empty(f"the climb and descent share no altitude bin with at "
                     f"least {min_samples} samples on both branches")

    ci = np.searchsorted(climb.altitude_m, shared)
    di = np.searchsorted(descent.altitude_m, shared)
    dose_c, dose_d = climb.dose_rate[ci], descent.dose_rate[di]
    sem_c, sem_d = climb.dose_sem[ci], descent.dose_sem[di]

    difference = dose_c - dose_d
    sigma = np.hypot(sem_c, sem_d)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sigma > 0, difference / sigma, 0.0)
    level = 0.5 * (dose_c + dose_d)

    threshold = bonferroni_k(int(shared.size), alpha)
    max_abs_z = float(np.nanmax(np.abs(z)))
    return BranchComparison(
        climb=climb, descent=descent, altitude_m=shared,
        difference=difference, sigma=sigma, n_shared=int(shared.size),
        threshold_sigma=float(threshold), max_abs_z=max_abs_z,
        median_abs_difference=float(np.median(np.abs(difference))),
        median_relative_difference=float(
            np.median(np.abs(difference) / np.where(level > 0, level, np.nan))),
        consistent=bool(max_abs_z <= threshold))
