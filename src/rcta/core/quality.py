"""GPS quality assessment: masking bad fixes and splitting the track at dropouts.

The defaults here are deliberately *not* tuned to the bundled sample. That file
is synthetic -- it implies ground speeds up to 1330 m/s and 16 % of its fixes
exceed 100 m of accuracy -- and adjusting thresholds until it comes out clean
would produce a tool that silently accepts nonsense from real surveys. The
numbers below are set from what a handheld detector carried on a person, in a
vehicle or on an aircraft can physically do, and the sample is expected to fail
two of them and say so.

Nothing is ever deleted. Two masks exist because the failure modes are
independent: a fix with 397 m of horizontal accuracy is useless for geometry
but its count rate is a real measurement, so it is excluded from the map and
the distance while remaining in the time series and the statistics.

Speed outliers are the union of two tests, not the intersection. An absolute
ceiling catches what no platform can do; a *robust relative* bound catches what
**this** platform cannot do. Requiring both, as an earlier version did, was
wrong in the direction that matters here: on a walking survey a 100 m GPS
glitch implies 50 m/s, which is nonsense for someone on foot but nowhere near
the 400 m/s ceiling, so it went unflagged. The relative bound uses a
median-plus-MAD estimator rather than a multiple of the median, so a genuinely
fast platform is not flagged end to end.

Both tests run on the *unsmoothed* fixes. A teleport is a property of a raw
fix, and the jitter smoothing that makes distance meaningful (see
:func:`rcta.core.geo.choose_smoothing_window`) would otherwise spread one bad
fix across the whole averaging window and report it as a dozen.

The leg split matters far beyond plotting. A dropout is the boundary at which
distance stops being observed, dose stops being integrable, and a rolling
window stops being contiguous, so every one of those calculations consults
:attr:`QualityReport.leg_index` rather than re-deriving its own idea of a gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rcta.core import geo
from rcta.core.track import Track

__all__ = ["QualityParams", "QualityReport", "assess", "gap_segments", "polyline"]


@dataclass(frozen=True)
class QualityParams:
    """Thresholds for GPS quality."""

    #: Fixes worse than this are excluded from geometry only.
    max_accuracy_m: float = 100.0
    #: An interval above this starts a new leg. None -> max(30, 10 x median).
    gap_threshold_s: float | None = None
    #: Absolute speed ceiling. Above airliner cruise and far below any GPS
    #: glitch, so it catches teleports without assuming the user is walking.
    max_speed_m_s: float = 400.0
    #: A segment is also an outlier if it exceeds this multiple of a robust
    #: upper bound on the track's own speed, which is what catches a glitch
    #: that is impossible for *this* platform but well under the hard ceiling.
    speed_outlier_factor: float = 4.0
    #: Steps shorter than this are never outliers, whatever the ratio says.
    min_jump_m: float = 50.0


@dataclass(frozen=True)
class QualityReport:
    """What the GPS did, and what had to be excluded because of it."""

    n: int
    n_poor_accuracy: int
    n_speed_outliers: int
    n_gaps: int
    n_legs: int
    gap_time_s: float
    gap_distance_m: float
    longest_gap_s: float
    median_dt_s: float
    median_accuracy_m: float
    span_time_s: float
    leg_index: np.ndarray
    gap_after: np.ndarray
    params: QualityParams
    warnings: tuple[str, ...] = ()

    @property
    def covered_time_s(self) -> float:
        """Seconds actually observed, excluding dropouts."""
        return self.span_time_s - self.gap_time_s

    @property
    def coverage(self) -> float:
        """Fraction of the elapsed span during which the detector was logging."""
        return self.covered_time_s / self.span_time_s if self.span_time_s else 1.0

    def describe(self) -> str:
        lines = [
            f"samples    : {self.n}",
            f"accuracy   : median {self.median_accuracy_m:.0f} m, "
            f"{self.n_poor_accuracy} fix(es) worse than "
            f"{self.params.max_accuracy_m:.0f} m",
            f"dropouts   : {self.n_gaps} gap(s) over "
            f"{self._gap_threshold():.0f} s, longest {self.longest_gap_s:.0f} s",
            f"legs       : {self.n_legs}",
            f"lost       : {self.gap_time_s:.0f} s and "
            f"{self.gap_distance_m / 1000.0:.1f} km inside dropouts",
            f"coverage   : {self.coverage * 100.0:.1f} % of the elapsed span",
            f"speed      : {self.n_speed_outliers} implausible segment(s)",
        ]
        for warning in self.warnings:
            lines.append(f"warning    : {warning}")
        return "\n".join(lines)

    def _gap_threshold(self) -> float:
        if self.params.gap_threshold_s is not None:
            return self.params.gap_threshold_s
        return max(30.0, 10.0 * self.median_dt_s)


def assess(track: Track, params: QualityParams | None = None) -> Track:
    """Mask poor fixes, split at dropouts, flag impossible speeds.

    Returns a new Track carrying ``position_mask`` and ``leg_index``; the
    report is attached as :attr:`Track.quality` via the returned track's
    ``meta`` warnings and is also available from :func:`report_for`. Use
    :func:`assess_with_report` when the numbers themselves are wanted.
    """
    return assess_with_report(track, params)[0]


def assess_with_report(track: Track,
                       params: QualityParams | None = None
                       ) -> tuple[Track, QualityReport]:
    """Mask poor fixes, split at dropouts, flag impossible speeds."""
    params = params or QualityParams()
    n = track.n
    warnings: list[str] = []

    median_dt = float(np.median(track.dt_s)) if track.dt_s.size else 0.0
    threshold = (params.gap_threshold_s if params.gap_threshold_s is not None
                 else max(30.0, 10.0 * median_dt))

    position = np.ones(n, dtype=bool)

    poor = track.accuracy_m > params.max_accuracy_m
    position &= ~poor
    n_poor = int(poor.sum())
    if n_poor:
        warnings.append(
            f"{n_poor} fix(es) exceed {params.max_accuracy_m:.0f} m accuracy; "
            "excluded from geometry, kept in the series")

    # Raw fixes, and the union of a hard ceiling with a robust relative bound.
    speed = track.raw_speed_m_s
    step = track.raw_segment_length_m
    outlier = np.zeros(max(n - 1, 0), dtype=bool)
    if speed.size:
        impossible = speed > params.max_speed_m_s
        implausible = speed > params.speed_outlier_factor * geo._robust_max(speed)
        outlier = (impossible | implausible) & (step > params.min_jump_m)
    n_outliers = int(outlier.sum())
    if n_outliers:
        # Only the arrival endpoint is suspect; the departure fix was fine.
        position[1:] &= ~outlier
        worst = float(speed[outlier].max())
        warnings.append(
            f"{n_outliers} segment(s) imply an implausible speed "
            f"(worst {worst:.0f} m/s against a typical "
            f"{np.median(speed):.1f} m/s); arrival fix excluded from geometry")

    gap_after = (track.dt_s > threshold) if track.dt_s.size else np.zeros(0, bool)
    n_gaps = int(gap_after.sum())
    legs = np.zeros(n, dtype=np.int64)
    if n_gaps:
        legs[1:] = np.cumsum(gap_after)
    gap_time = float(track.dt_s[gap_after].sum()) if n_gaps else 0.0
    gap_distance = float(step[gap_after].sum()) if n_gaps else 0.0
    longest = float(track.dt_s.max()) if track.dt_s.size else 0.0

    span = float(track.time_s[-1] - track.time_s[0]) if n > 1 else 0.0
    report = QualityReport(
        n=n, n_poor_accuracy=n_poor, n_speed_outliers=n_outliers,
        n_gaps=n_gaps, n_legs=int(legs.max()) + 1, gap_time_s=gap_time,
        gap_distance_m=gap_distance, longest_gap_s=longest,
        median_dt_s=median_dt, median_accuracy_m=float(np.median(track.accuracy_m)),
        span_time_s=span, leg_index=legs, gap_after=gap_after,
        params=params, warnings=tuple(warnings),
    )
    return track.with_mask(position=position, legs=legs), report


def polyline(track: Track, report: QualityReport | None = None, *,
             mercator: bool = True):
    """x/y arrays for plotting, with NaN inserted at every break.

    Gaps get their own function rather than being handled at the plotting call
    site because the consequence of getting it wrong is a lie rather than an
    inconvenience: the longest dropout in the bundled file spans 763 s and
    103 km, and drawn as a solid line it is indistinguishable from 103 km of
    surveyed track, with a fabricated colour at every pixel along it.
    """
    # The *line* is the route, so it is drawn from jitter-corrected positions;
    # on a walk the raw fixes are a ball of yarn 8 m wide around a 3 m step.
    # The scatter stays on raw positions, because that is where each reading
    # was actually recorded and its uncertainty is shown separately.
    if mercator:
        x, y = track.smooth_merc_x.copy(), track.smooth_merc_y.copy()
    else:
        x, y = track.smooth_lon.copy(), track.smooth_lat.copy()

    hidden = ~track.position_mask
    x[hidden] = np.nan
    y[hidden] = np.nan

    breaks = (report.gap_after if report is not None
              else geo.segment_breaks(track.time_s, track.lon, track.lat))
    if breaks.size and breaks.any():
        # A NaN *between* the two samples, so both endpoints still draw.
        cut = np.flatnonzero(breaks) + 1
        x = np.insert(x, cut, np.nan)
        y = np.insert(y, cut, np.nan)
    return x, y


def gap_segments(track: Track, report: QualityReport) -> list[tuple]:
    """(start_index, stop_index, seconds, metres) for every dropout.

    Lets the map draw a gap as visibly *not* data -- a thin dashed line with a
    duration label -- rather than either a solid coloured track or nothing.
    """
    out = []
    for i in np.flatnonzero(report.gap_after):
        out.append((int(i), int(i) + 1, float(track.dt_s[i]),
                    float(track.segment_length_m[i])))
    return out
