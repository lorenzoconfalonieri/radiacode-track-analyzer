"""Background estimation for a detector that is moving through the world.

This module makes the most consequential choice in the program, and it is
worth being blunt about why: **the background estimator defines what counts as
an anomaly**. Everything downstream -- significances, Currie limits, hotspot
rankings, the entire detections table -- is a statement about the difference
between the data and whatever this module decided the baseline was. Change the
estimator and you change the findings, so the estimator is a parameter of the
result and is reported alongside it, never buried.

The failure this exists to prevent is easiest to see on a synthetic ramp with
a known answer: a background that climbs smoothly by a factor of twelve and
contains no source at all. Estimate the background as a single number for the
whole track and the high end becomes a ten-sigma "detection" spanning the
second half of the survey; estimate it locally and nothing survives. A
regression test pins that inversion, because it is exactly the sort of thing a
well-meaning simplification would quietly undo.

Whether a global estimate is catastrophic on any *particular* file depends on
where it happens to land. On the bundled flight it lands near the cruise-level
population, which is the majority of the samples, so it under-reports rather
than over-reports and both estimators end up finding nothing. That is luck,
not correctness, and it is why the guarantee is demonstrated on data whose
truth is constructed rather than on one convenient real file.

Four methods, and the differences between them are real rather than cosmetic:

*rolling* (default) takes a symmetrically trimmed mean of a centred time
window. It follows slow environmental change, which on any outdoor survey is
the dominant signal. Trimming both tails -- rather than taking a low quantile,
which was the first design here -- is what keeps the estimate robust to a
hotspot inside the window *without* biasing it low. A 0.25 quantile of an
18 cps Poisson background sits 2.0 cps under the true mean, which over a
64-sample window is 256 counts of excess against a 48-count error: a 5.3 sigma
detection on every window, produced entirely by the estimator. See
:func:`rcta.core.statistics.rolling_trimmed_mean`.

*flank* averages two side-bands either side of a guard band. This is the exact
analogue of a linear continuum drawn under a photopeak, and it cancels a
locally linear trend to first order. Use it for a deliberate stop-and-measure
at a suspected source. It is worse than rolling at a step change, because a
step is not linear.

*reference* averages a span the user has marked as clean. It is the only
method that means "background" the way a health physicist means it, and the
only one whose errors are the user's responsibility rather than the tool's.

*global* takes one trimmed mean of everything. Correct only when there is no
environmental gradient -- which for a short walking survey is both true and
the easiest thing to defend in a report.

Rolling and flank both check scale separation against the anomaly scan. If the
background window is comparable to the widest window being tested, the scan is
comparing a signal against itself, every real trend becomes a detection, and
the estimate carries a warning saying so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rcta.core import statistics as st
from rcta.core.track import Track

__all__ = ["BackgroundEstimate", "BackgroundParams", "estimate_background"]

_METHODS = ("rolling", "flank", "reference", "global")


@dataclass(frozen=True)
class BackgroundParams:
    """How to estimate the baseline the anomalies are measured against."""

    #: One of "rolling", "flank", "reference", "global".
    method: str = "rolling"
    #: Rolling or flank window, in seconds. Chosen to clear the
    #: scale-separation check against the default scan's widest window
    #: (64 samples at 2 s = 128 s), so the stock configuration does not warn
    #: about itself.
    window_s: float = 900.0
    #: Fraction trimmed from *each* tail of the window before averaging.
    #: Symmetric, so the estimate stays unbiased; 0.25 tolerates a quarter of
    #: the window being elevated before the background starts to follow it.
    trim_fraction: float = 0.25
    #: Guard band excluded either side of the tested point, in seconds. Stops
    #: a source contaminating the estimate of its own background.
    guard_s: float = 30.0
    #: Windows with fewer contributors than this fall back to the global value.
    min_samples: int = 5
    #: For method="reference", the (t0, t1) span the user marked as clean.
    reference_span: tuple[float, float] | None = None
    #: Background window must exceed this multiple of the widest scan window.
    scale_separation: float = 5.0

    def __post_init__(self) -> None:
        if self.method not in _METHODS:
            raise ValueError(
                f"unknown background method {self.method!r}; "
                f"expected one of {list(_METHODS)}")
        if not 0.0 <= self.trim_fraction < 0.5:
            raise ValueError(
                f"trim_fraction must be in [0, 0.5), got {self.trim_fraction}")
        if self.window_s <= 0:
            raise ValueError("window_s must be positive")
        if self.method == "reference" and self.reference_span is None:
            raise ValueError('method="reference" needs a reference_span')


@dataclass(frozen=True)
class BackgroundEstimate:
    """Per-sample background, with the support behind each value."""

    #: Background count rate at each sample, cps.
    count_rate: np.ndarray
    #: Number of samples contributing to each estimate.
    n_effective: np.ndarray
    #: The same background expressed in nSv/h, via the track's own ratio.
    dose_rate: np.ndarray
    method: str
    params: BackgroundParams
    warnings: tuple[str, ...] = ()

    @property
    def median_count_rate(self) -> float:
        return float(np.median(self.count_rate))

    def describe(self, unit: str = "nSv/h") -> str:
        from rcta.core.dose import format_dose_rate
        lines = [
            f"method     : {self.method}",
            f"background : median {self.median_count_rate:.2f} cps "
            f"({format_dose_rate(float(np.median(self.dose_rate)), unit)}), "
            "range "
            f"{self.count_rate.min():.2f} .. {self.count_rate.max():.2f} cps",
            f"support    : median {np.median(self.n_effective):.0f} samples "
            f"per estimate",
        ]
        for warning in self.warnings:
            lines.append(f"warning    : {warning}")
        return "\n".join(lines)


def estimate_background(track: Track, params: BackgroundParams | None = None, *,
                        quality=None,
                        max_scan_window_s: float | None = None
                        ) -> BackgroundEstimate:
    """Per-sample background count rate.

    ``max_scan_window_s`` lets the caller declare the widest window the
    anomaly scan will test, so that the scale-separation check can fire. The
    facade in :mod:`rcta.core.analysis` always supplies it; a caller who omits
    it simply gets no warning.
    """
    params = params or BackgroundParams()
    warnings: list[str] = []

    # Estimated from the samples that carry a reading, then carried onto the
    # masked ones by interpolation in time, so every sample still has a value
    # and no masked NaN can reach a window.
    valid = track.valid_mask
    rate = track.count_rate[valid]
    time_s = track.time_s[valid]
    n = rate.size

    if params.method == "global":
        value = float(st._trimmed_mean(rate, params.trim_fraction))
        background = np.full(n, value)
        support = np.full(n, n, dtype=np.int64)
        warnings.append(
            "a single background for the whole track assumes no environmental "
            "gradient; on a track that climbs or changes geology this reports "
            "the trend itself as a detection")

    elif params.method == "reference":
        t0, t1 = params.reference_span
        inside = (time_s >= min(t0, t1)) & (time_s <= max(t0, t1))
        if inside.sum() < params.min_samples:
            raise ValueError(
                f"reference span contains {int(inside.sum())} samples, "
                f"need at least {params.min_samples}")
        value = float(rate[inside].mean())
        background = np.full(n, value)
        support = np.full(n, int(inside.sum()), dtype=np.int64)
        warnings.append(
            f"background fixed at {value:.2f} cps from a user-marked span of "
            f"{int(inside.sum())} samples; it does not follow any change "
            "elsewhere on the track")

    elif params.method == "rolling":
        background, support = st.rolling_trimmed_mean(
            rate, time_s, window_s=params.window_s,
            trim_fraction=params.trim_fraction, min_samples=params.min_samples)

    elif params.method == "flank":
        background, support = _flank_background(rate, time_s, params)

    if not valid.all():
        background = np.interp(track.time_s, time_s, background)
        support = np.round(np.interp(track.time_s, time_s,
                                     support)).astype(np.int64)

    thin = int((support[valid] < params.min_samples).sum())
    if thin:
        warnings.append(
            f"{thin} sample(s) had too little local data and fell back to the "
            "global estimate; their significances are not locally referenced")

    if (max_scan_window_s and params.method in ("rolling", "flank")
            and params.window_s < params.scale_separation * max_scan_window_s):
        warnings.append(
            f"background window ({params.window_s:.0f} s) is less than "
            f"{params.scale_separation:g}x the widest scan window "
            f"({max_scan_window_s:.0f} s); the scan is partly comparing the "
            "signal against itself and will under-report broad features")

    if quality is not None and quality.n_gaps and params.method == "rolling":
        warnings.append(
            f"{quality.n_gaps} dropout(s) present; rolling windows are defined "
            "in seconds, so estimates adjacent to a gap draw on fewer samples "
            "rather than on data from the far side of it")

    return BackgroundEstimate(
        count_rate=np.asarray(background, dtype=float),
        n_effective=np.asarray(support, dtype=np.int64),
        dose_rate=np.asarray(background, dtype=float) * track.dose_per_count,
        method=params.method, params=params, warnings=tuple(warnings),
    )


def _flank_background(rate, time_s, params: BackgroundParams):
    """Mean of two side-bands, excluding a guard band around each sample.

    The direct analogue of a linear continuum under a photopeak. The guard
    band is what stops an extended source from being used to estimate its own
    background, which would drive the apparent net signal toward zero exactly
    where it matters most.
    """
    n = time_s.size
    half = params.window_s / 2.0
    guard = params.guard_s

    outer_lo = np.searchsorted(time_s, time_s - half, side="left")
    inner_lo = np.searchsorted(time_s, time_s - guard, side="left")
    inner_hi = np.searchsorted(time_s, time_s + guard, side="right")
    outer_hi = np.searchsorted(time_s, time_s + half, side="right")

    prefix = np.concatenate(([0.0], np.cumsum(rate)))
    left_sum = prefix[inner_lo] - prefix[outer_lo]
    right_sum = prefix[outer_hi] - prefix[inner_hi]
    left_n = inner_lo - outer_lo
    right_n = outer_hi - inner_hi

    total_n = left_n + right_n
    fallback = float(st._trimmed_mean(rate, params.trim_fraction))
    with np.errstate(divide="ignore", invalid="ignore"):
        background = np.where(total_n >= params.min_samples,
                              (left_sum + right_sum) / np.maximum(total_n, 1),
                              fallback)
    return background, total_n.astype(np.int64)
