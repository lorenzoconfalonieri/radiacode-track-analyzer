"""Summaries, distance profiles, distributions and correlations.

The unifying theme is that every function here returns a small frozen result
object with a ``describe()``, rather than a bare number, because on a track
every one of these quantities needs a qualifier to be honest.

*A summary needs its mask.* Quoting a mean dose rate without saying how many
samples were excluded, or over what fraction of the span, invites the reader
to treat a 63 %-covered flight as a continuous survey.

*A distance profile must not interpolate.* Resampling a track onto a regular
distance grid by interpolation invents readings inside dropouts, which is
precisely where there is no data. Binning leaves those cells empty and NaN,
and the plot then breaks across them.

*A distribution needs to know whether it is one distribution.* The bundled
track's dose rate is a mixture of two populations 80 nSv/h apart -- ground and
cruise altitude. Its overall mean and standard deviation describe neither, and
any background estimator that assumes unimodality is wrong on it. So
:func:`bimodality` is offered next to :func:`distribution`, and it uses an
exact deterministic split rather than k-means, because a result that changes
with a random seed is not a diagnostic.

*A correlation needs its effective sample size.* See
:mod:`rcta.core.statistics`; the short version is that consecutive samples two
seconds apart are not independent observations, and pretending otherwise turns
any monotone trend into overwhelming significance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rcta.core import statistics as st
from rcta.core.dose import accumulated_dose
from rcta.core.track import Track

__all__ = [
    "BIMODAL_ASHMAN_D",
    "Bimodality",
    "Correlation",
    "Histogram",
    "Profile",
    "SummaryStats",
    "UNIMODAL_ASHMAN_D",
    "bimodality",
    "correlate",
    "correlate_speed_dose",
    "distribution",
    "resample_by_distance",
    "summarise",
]


@dataclass(frozen=True)
class SummaryStats:
    """Headline numbers for a track or a selection."""

    n: int
    n_valid: int
    n_poor_position: int
    duration_s: float
    distance_m: float
    dose_min: float
    dose_max: float
    dose_mean: float
    dose_median: float
    dose_p95: float
    count_min: float
    count_max: float
    count_mean: float
    count_median: float
    speed_median: float
    speed_max: float
    accuracy_median: float
    accuracy_max: float
    accumulated_dose_nsv: float
    coverage: float
    contiguous: bool = True

    def describe(self, unit: str = "nSv/h") -> str:
        from rcta.core.dose import (accumulated_unit, convert_dose_rate,
                                    format_accumulated_dose)
        hours, rem = divmod(int(self.duration_s), 3600)
        minutes, seconds = divmod(rem, 60)
        lo, hi, mean, median, p95 = (
            float(convert_dose_rate(v, unit)) for v in
            (self.dose_min, self.dose_max, self.dose_mean, self.dose_median,
             self.dose_p95))
        digits = 2 if unit in ("nSv/h", "uR/h") else 4
        lines = [
            f"samples    : {self.n} ({self.n_valid} valid, "
            f"{self.n_poor_position} with poor position)",
            f"duration   : {hours} h {minutes} m {seconds} s"
            f"   coverage {self.coverage * 100.0:.1f} %",
            f"distance   : {self.distance_m / 1000.0:.2f} km"
            + ("" if self.contiguous else "   (selection is not contiguous)"),
            f"dose rate  : {lo:.{digits}f} .. {hi:.{digits}f} {unit}   "
            f"mean {mean:.{digits}f}, median {median:.{digits}f}, "
            f"p95 {p95:.{digits}f}",
            f"count rate : {self.count_min:.2f} .. {self.count_max:.2f} cps   "
            f"mean {self.count_mean:.2f}, median {self.count_median:.2f}",
            f"speed      : median {self.speed_median:.1f} m/s, "
            f"max {self.speed_max:.1f} m/s",
            f"accuracy   : median {self.accuracy_median:.0f} m, "
            f"max {self.accuracy_max:.0f} m",
            f"dose       : "
            f"{format_accumulated_dose(self.accumulated_dose_nsv, unit)} "
            "accumulated",
        ]
        return "\n".join(lines)


def _stat(reduce, values) -> float:
    """``reduce(values)`` as a float, or NaN when there is nothing to reduce."""
    return float(reduce(values)) if values.size else float("nan")


def summarise(track: Track, *, mask=None, quality=None) -> SummaryStats:
    """Headline statistics over the whole track or a selection.

    One function, used by the status bar, the CLI and the report alike, so
    they cannot drift apart and quote three different means.
    """
    contiguous = True
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            raise ValueError("cannot summarise an empty selection")
        # A non-contiguous selection has no meaningful path length; say so
        # rather than quoting the sum of unrelated jumps.
        contiguous = bool(np.diff(np.flatnonzero(mask)).max(initial=1) == 1)
        sub = track.select(mask)
    else:
        sub = track

    try:
        dose = accumulated_dose(sub, quality=quality if mask is None else None)
        accumulated, coverage = dose.total_nsv, dose.coverage
    except ValueError:
        accumulated, coverage = 0.0, 1.0

    distance = float(sub.distance_along_m[-1]) if contiguous else float("nan")
    # Readings only from samples that have one; a selection of nothing but
    # masked samples reports NaN rather than a fabricated zero.
    dose = sub.dose_rate[sub.valid_mask]
    cps = sub.count_rate[sub.valid_mask]
    accuracy = sub.accuracy_m[np.isfinite(sub.accuracy_m)]
    return SummaryStats(
        n=sub.n,
        n_valid=int(sub.valid_mask.sum()),
        n_poor_position=int((~sub.position_mask).sum()),
        duration_s=float(sub.time_s[-1] - sub.time_s[0]) if sub.n > 1 else 0.0,
        distance_m=distance,
        dose_min=_stat(np.min, dose), dose_max=_stat(np.max, dose),
        dose_mean=_stat(np.mean, dose),
        dose_median=_stat(np.median, dose),
        dose_p95=_stat(lambda v: np.percentile(v, 95), dose),
        count_min=_stat(np.min, cps), count_max=_stat(np.max, cps),
        count_mean=_stat(np.mean, cps),
        count_median=_stat(np.median, cps),
        speed_median=float(np.median(sub.speed_m_s)),
        speed_max=float(sub.speed_m_s.max()),
        accuracy_median=_stat(np.median, accuracy),
        accuracy_max=_stat(np.max, accuracy),
        accumulated_dose_nsv=accumulated,
        coverage=coverage,
        contiguous=contiguous,
    )


@dataclass(frozen=True)
class Profile:
    """A track aggregated onto a regular distance-along-track grid."""

    distance_m: np.ndarray
    dose_rate: np.ndarray
    count_rate: np.ndarray
    speed_m_s: np.ndarray
    n_samples: np.ndarray
    step_m: float
    statistic: str

    @property
    def n_empty(self) -> int:
        return int((self.n_samples == 0).sum())

    def describe(self) -> str:
        occupied = self.n_samples > 0
        return (f"{self.distance_m.size} bins of {self.step_m:.0f} m "
                f"({self.statistic}); {self.n_empty} empty, "
                f"median {np.median(self.n_samples[occupied]):.0f} samples per "
                f"occupied bin" if occupied.any() else "profile is empty")


def resample_by_distance(track: Track, *, step_m: float = 100.0,
                         statistic: str = "mean") -> Profile:
    """Bin samples by distance along the track. Deliberately not interpolation.

    Interpolating onto a distance grid manufactures readings inside dropouts,
    which is exactly where the detector was not reading. Binning leaves those
    cells at ``n_samples == 0`` with a NaN value, so the profile plot breaks
    across them and the absence is visible.

    Note that on a fast platform most bins are empty by construction: at
    233 m/s with 2 s logging, consecutive samples are 470 m apart, so a 100 m
    grid cannot be full. That is the honest picture, not a defect.
    """
    if step_m <= 0:
        raise ValueError("step_m must be positive")
    reducers = {"mean": np.mean, "median": np.median, "max": np.max, "min": np.min}
    if statistic not in reducers:
        raise ValueError(f"unknown statistic {statistic!r}; "
                         f"expected one of {sorted(reducers)}")
    reduce = reducers[statistic]

    d = track.distance_along_m
    total = float(d[-1])
    n_bins = max(int(np.ceil(total / step_m)), 1)
    edges = np.arange(n_bins + 1, dtype=float) * step_m
    index = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, n_bins - 1)

    dose = np.full(n_bins, np.nan)
    count = np.full(n_bins, np.nan)
    speed = np.full(n_bins, np.nan)
    n_samples = np.zeros(n_bins, dtype=np.int64)
    order = np.argsort(index, kind="stable")
    sorted_index = index[order]
    starts = np.searchsorted(sorted_index, np.arange(n_bins), side="left")
    stops = np.searchsorted(sorted_index, np.arange(n_bins), side="right")
    valid = track.valid_mask
    for b in range(n_bins):
        rows = order[starts[b]:stops[b]]
        rows = rows[valid[rows]]
        if rows.size == 0:
            continue
        n_samples[b] = rows.size
        dose[b] = reduce(track.dose_rate[rows])
        count[b] = reduce(track.count_rate[rows])
        speed[b] = reduce(track.speed_m_s[rows])

    return Profile(distance_m=edges[:-1] + step_m / 2.0, dose_rate=dose,
                   count_rate=count, speed_m_s=speed, n_samples=n_samples,
                   step_m=float(step_m), statistic=statistic)


@dataclass(frozen=True)
class Histogram:
    counts: np.ndarray
    edges: np.ndarray
    unit: str = ""

    @property
    def centres(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    def describe(self) -> str:
        return (f"{self.counts.sum()} values in {self.counts.size} bins over "
                f"{self.edges[0]:.2f} .. {self.edges[-1]:.2f} {self.unit}".strip())


def distribution(values, *, bins="fd", value_range=None, unit="") -> Histogram:
    """A histogram with Freedman-Diaconis binning by default.

    FD rather than Sturges or a fixed count because it is driven by the
    interquartile range, so a bimodal or heavy-tailed distribution does not
    get smoothed into a single lump by a bin width chosen from the sample size
    alone.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return Histogram(counts=np.zeros(0, dtype=np.int64),
                         edges=np.zeros(1), unit=unit)
    counts, edges = np.histogram(values, bins=bins, range=value_range)
    return Histogram(counts=counts, edges=edges, unit=unit)


#: Ashman's D produced by hard-splitting a single Gaussian at its mean. The
#: two lobes are half-normals with means +-sqrt(2/pi) and variance 1 - 2/pi,
#: giving D = sqrt(2) * 2 sqrt(2/pi) / sqrt(2 (1 - 2/pi)) = 2.647. This is the
#: floor any 2-means split clears for free, so it is the number to beat.
UNIMODAL_ASHMAN_D = 2.6467

#: Threshold for calling a distribution genuinely bimodal, set clear of the
#: unimodal floor above rather than at the textbook 2.0.
BIMODAL_ASHMAN_D = 3.0


@dataclass(frozen=True)
class Bimodality:
    """An exact two-means split of a one-dimensional distribution."""

    threshold: float
    mu_low: float
    sd_low: float
    n_low: int
    mu_high: float
    sd_high: float
    n_high: int
    ashman_d: float

    @property
    def is_bimodal(self) -> bool:
        """True only when the split beats what a *single* Gaussian would give.

        The textbook "Ashman D > 2" criterion applies to a fitted two-component
        mixture, and applying it to a hard partition is a trap: 2-means always
        splits, so cutting a perfectly unimodal Gaussian at its mean yields two
        half-normal lobes with D = 2.647 (see :data:`UNIMODAL_ASHMAN_D`, which
        is analytic, not fitted). Every unimodal distribution would therefore
        be reported as bimodal. The threshold here sits above that baseline.
        """
        return self.ashman_d > BIMODAL_ASHMAN_D

    def describe(self) -> str:
        verdict = ("clearly bimodal" if self.is_bimodal
                   else f"not clearly bimodal (a single Gaussian scores "
                        f"{UNIMODAL_ASHMAN_D:.2f} here)")
        return (f"split at {self.threshold:.2f}: "
                f"{self.mu_low:.2f} +- {self.sd_low:.2f} (n={self.n_low}) vs "
                f"{self.mu_high:.2f} +- {self.sd_high:.2f} (n={self.n_high}); "
                f"Ashman D = {self.ashman_d:.2f}, {verdict}")


def bimodality(values) -> Bimodality:
    """Exact one-dimensional 2-means, by scanning every split of the sorted data.

    Deterministic and O(n log n): no Lloyd iterations, no initialisation, no
    seed. A diagnostic whose answer moves when you re-run it is not a
    diagnostic, and on a question as consequential as "is my background
    estimator being applied to one population or two" that matters.

    The within-cluster sum of squares for every candidate split is obtained
    from prefix sums, so the whole scan is three vectorised passes.
    """
    values = np.sort(np.asarray(values, dtype=float))
    n = values.size
    if n < 4:
        mean = float(values.mean()) if n else 0.0
        return Bimodality(threshold=mean, mu_low=mean, sd_low=0.0, n_low=n,
                          mu_high=mean, sd_high=0.0, n_high=0, ashman_d=0.0)

    prefix = np.concatenate(([0.0], np.cumsum(values)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(values * values)))
    k = np.arange(1, n)                       # size of the low cluster
    low_sum, high_sum = prefix[k], prefix[n] - prefix[k]
    low_sq, high_sq = prefix_sq[k], prefix_sq[n] - prefix_sq[k]
    # Within-cluster SS = sum(x^2) - (sum x)^2 / count, for each side.
    wcss = (low_sq - low_sum ** 2 / k) + (high_sq - high_sum ** 2 / (n - k))
    best = int(np.argmin(wcss)) + 1

    n_low, n_high = best, n - best
    mu_low = float(low_sum[best - 1] / n_low)
    mu_high = float(high_sum[best - 1] / n_high)
    var_low = max(float(low_sq[best - 1] / n_low - mu_low ** 2), 0.0)
    var_high = max(float(high_sq[best - 1] / n_high - mu_high ** 2), 0.0)
    sd_low, sd_high = np.sqrt(var_low), np.sqrt(var_high)

    denom = var_low + var_high
    ashman = (float(np.sqrt(2.0) * abs(mu_high - mu_low) / np.sqrt(denom))
              if denom > 0 else 0.0)
    return Bimodality(
        threshold=float(0.5 * (values[best - 1] + values[best])),
        mu_low=mu_low, sd_low=float(sd_low), n_low=n_low,
        mu_high=mu_high, sd_high=float(sd_high), n_high=n_high,
        ashman_d=ashman,
    )


def _format_p(p: float) -> str:
    """Format a p-value without ever printing a bare zero.

    Underflow to 0.0 is a floating-point artefact, not a claim of absolute
    certainty, and printing "0" invites exactly that reading.
    """
    return "< 1e-300" if p <= 0.0 else f"{p:.3g}"


@dataclass(frozen=True)
class Correlation:
    """An association between two series, with an honest p-value."""

    label_x: str
    label_y: str
    spearman_rho: float
    pearson_r: float
    n: int
    n_effective: float
    p_naive: float
    p_effective: float
    p_block_bootstrap: float
    block_s: float
    #: Free-text caveat about confounding that no p-value can express.
    caveat: str = ""

    def describe(self) -> str:
        lines = [
            f"{self.label_x} vs {self.label_y}: "
            f"rho = {self.spearman_rho:+.4f}, r = {self.pearson_r:+.4f}",
            f"samples    : {self.n} nominal, {self.n_effective:.1f} effective "
            f"after the serial-dependence correction",
            f"p (naive)  : {_format_p(self.p_naive)}   <- assumes independence; "
            "on a 2 s cadence it does not hold",
            f"p (honest) : {_format_p(self.p_effective)}   <- use this one",
            f"p (block)  : {_format_p(self.p_block_bootstrap)}   "
            f"(moving block, {self.block_s:.0f} s; anticonservative on a trend)",
        ]
        if self.caveat:
            lines.append(f"caveat     : {self.caveat}")
        return "\n".join(lines)


def correlate(x, y, time_s, *, label_x="x", label_y="y", block_s: float = 300.0,
              n_boot: int = 2000, seed: int = 0, caveat: str = "") -> Correlation:
    """Correlate two series sampled from the same track.

    Reports three p-values on purpose. The naive one is included only so the
    reader can see the size of the correction: on serially dependent data it
    can be wrong by two hundred orders of magnitude, and a program that showed
    only the honest number would leave the user wondering why it disagreed
    with whatever they computed in a spreadsheet.
    """
    from scipy import stats

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    time_s = np.asarray(time_s, dtype=float)

    # A constant series has no correlation to report, and scipy warns rather
    # than erroring; saying "no association" plainly is more useful than a NaN.
    degenerate = (x.size <= 2 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0)
    if degenerate:
        return Correlation(
            label_x=label_x, label_y=label_y, spearman_rho=0.0, pearson_r=0.0,
            n=int(x.size), n_effective=float(x.size), p_naive=1.0,
            p_effective=1.0, p_block_bootstrap=1.0, block_s=float(block_s),
            caveat=caveat or "one of the series is constant",
        )

    rho = st.spearman_rho(x, y)
    pearson = stats.pearsonr(x, y)
    n_eff = st.bartlett_effective_n(x, y)

    p_naive = float(stats.spearmanr(x, y).pvalue)
    return Correlation(
        label_x=label_x, label_y=label_y,
        spearman_rho=rho, pearson_r=float(pearson[0]),
        n=int(x.size), n_effective=n_eff,
        p_naive=p_naive,
        p_effective=st.effective_p(rho, n_eff),
        p_block_bootstrap=st.block_bootstrap_p(
            x, y, time_s, block_s=block_s, n_boot=n_boot, seed=seed),
        block_s=float(block_s), caveat=caveat,
    )


def correlate_speed_dose(track: Track, **kw) -> Correlation:
    """Speed against dose rate, with the relevant confound stated up front.

    The caveat is chosen from the platform, because the wrong one is worse
    than none: telling somebody on a walking survey that their result is an
    altitude effect is simply false, and invites them to dismiss a real
    finding. Above vehicle speeds the confound is altitude driving both cosmic
    dose and ground speed; on foot it is that you slow down near whatever you
    stopped to look at, which correlates dwell time with dose for reasons that
    have nothing to do with the source.
    """
    typical = float(np.median(track.speed_m_s)) if track.n > 1 else 0.0
    if typical > 20.0:
        caveat = ("speed and dose rate are both driven by altitude on an "
                  "airborne or mountain track; this is confounding, not "
                  "causation")
    else:
        caveat = ("on foot, slowing down near something of interest couples "
                  "speed to dwell time and so to the samples collected; treat "
                  "this as confounding, not causation")
    kw.setdefault("caveat", caveat)
    # The first speed is a copy of the second, and masked samples have none.
    keep = track.valid_mask.copy()
    keep[0] = False
    return correlate(track.speed_m_s[keep], track.dose_rate[keep],
                     track.time_s[keep], label_x="speed", label_y="dose rate",
                     **kw)
