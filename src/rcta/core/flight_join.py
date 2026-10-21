"""Is this a flight, and what was its altitude at each sample?

The track format carries no altitude, so an airborne track has lost the
variable that explains its count rate. This module decides whether a track
is airborne at all, holds an imported trajectory, and joins its altitude
onto the track's samples by time -- but only after checking the
trajectory's positions against the track's own GPS fixes. A trajectory that
does not describe the track is refused, with the residuals that prove it,
rather than interpolated into a smooth, plausible and fictional altitude
axis; the clock offset that would make it fit is measured and reported but
never applied unasked, because sliding a clock until the data fit is how a
wrong flight is made to look right. :mod:`rcta.core.flight` re-exports all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property

import numpy as np

from rcta.core import geo
from rcta.core.flight_units import (
    _FLIGHT_MEDIAN_SPEED_M_S,
    _FLIGHT_P95_SPEED_M_S,
    _LEVEL_RATE_M_S,
    _OFFSET_SEARCH_S,
    DEFAULT_MATCH_TOLERANCE_M,
    DEFAULT_MAX_GAP_S,
    _fmt_alt,
    _utc,
)
from rcta.core.track import Track

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
        if report.n_compared == 0:
            # Nothing overlapped in time, so there is no residual to quote;
            # "nan km exceeds the tolerance" blamed positions for a timing
            # problem.
            message = ("trajectory does not overlap this track in time: no "
                       "sample had both a usable fix and a trajectory "
                       "position. Check the date, the flight, or the time "
                       "offset")
        else:
            message = (f"trajectory does not match this track: median "
                       f"position residual "
                       f"{report.residual_median_m / 1000.0:.1f} km exceeds "
                       f"the {report.tolerance_m / 1000.0:.1f} km tolerance")
        super().__init__(message)
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
    if not math.isfinite(float(time_offset_s)):
        raise ValueError(f"time_offset_s must be finite, got {time_offset_s}")

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


