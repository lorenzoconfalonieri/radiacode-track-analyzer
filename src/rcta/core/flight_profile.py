"""Dose rate against altitude: bins, a scale height, and its control.

Once altitude is joined, the profile bins the samples by altitude, fits an
exponential atmosphere to the bins above a floor (below it, terrestrial
gamma does not vary with altitude and bends the fit), and compares the climb
with the descent. That comparison is the honest error bar on the fit: two
independent traverses of the same air hours apart should agree, and how far
they do not is reported beside the scale height rather than hidden in it.

Bins report both the counting error and the observed scatter, because a
ratio well below one means the assumed integration time is wrong.
:mod:`rcta.core.flight` re-exports everything here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from rcta.core.dose import format_dose_rate
from rcta.core.flight_join import AltitudeJoin
from rcta.core.flight_units import _fmt_alt

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

    def describe(self, unit: str = "nSv/h") -> str:
        m = self.occupied
        if not m.any():
            return f"altitude profile ({self.branch}) is empty"
        lines = [
            f"{int(m.sum())} occupied bins of {self.step_m:.0f} m over "
            f"{self.altitude_m[m].min():.0f} .. {self.altitude_m[m].max():.0f} m "
            f"({self.branch}, {self.statistic}); {self.n_empty} empty",
            f"dose rate {format_dose_rate(np.nanmin(self.dose_rate[m]), unit)} .. "
            f"{format_dose_rate(np.nanmax(self.dose_rate[m]), unit)} over "
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

    def describe(self, unit: str = "nSv/h") -> str:
        if not self.constrained:
            return f"scale height : not constrained -- {self.note}"
        lines = [
            f"scale height : {self.scale_height_m / 1000.0:.2f} +- "
            f"{self.scale_height_sd_m / 1000.0:.2f} km e-folding",
            f"             : dose rate doubles every "
            f"{self.doubling_m:.0f} m of climb",
            f"             : {format_dose_rate(self.dose_at_reference, unit)} at "
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

    def describe(self, unit: str = "nSv/h") -> str:
        if self.n_shared == 0:
            return ("climb vs descent : no shared altitude bin -- "
                    + (self.note or "one branch was not sampled"))
        verdict = ("consistent" if self.consistent
                   else "INCONSISTENT beyond counting statistics")
        lines = [
            f"climb vs descent : {verdict} over {self.n_shared} shared bin(s)",
            f"  median difference "
            f"{format_dose_rate(self.median_abs_difference, unit)} "
            f"({self.median_relative_difference * 100.0:.1f} % of the level)",
            f"  worst bin {self.max_abs_z:.2f} sigma against a "
            f"{self.threshold_sigma:.2f} sigma threshold "
            f"(two-sided, Bonferroni over {self.n_shared} bins)",
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
    # Matched on the lattice index, not on float equality, which a step such
    # as 304.8 m (1000 ft) would break.
    def lattice(profile):
        return np.round(profile.altitude_m / step_m * 2.0).astype(np.int64)

    c_key, d_key = lattice(climb), lattice(descent)
    shared_key = np.intersect1d(c_key[climb.n_samples >= min_samples],
                                d_key[descent.n_samples >= min_samples])
    if shared_key.size == 0:
        return empty(f"the climb and descent share no altitude bin with at "
                     f"least {min_samples} samples on both branches")

    ci = np.searchsorted(c_key, shared_key)
    di = np.searchsorted(d_key, shared_key)
    shared = climb.altitude_m[ci]
    dose_c, dose_d = climb.dose_rate[ci], descent.dose_rate[di]
    sem_c, sem_d = climb.dose_sem[ci], descent.dose_sem[di]

    difference = dose_c - dose_d
    sigma = np.hypot(sem_c, sem_d)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sigma > 0, difference / sigma, 0.0)
    level = 0.5 * (dose_c + dose_d)

    # Two-sided: the descent reading high is as much a disagreement as the
    # climb reading high, so each bin spends alpha / 2 in each tail.
    threshold = bonferroni_k(int(shared.size), alpha / 2.0)
    max_abs_z = float(np.nanmax(np.abs(z)))
    return BranchComparison(
        climb=climb, descent=descent, altitude_m=shared,
        difference=difference, sigma=sigma, n_shared=int(shared.size),
        threshold_sigma=float(threshold), max_abs_z=max_abs_z,
        median_abs_difference=float(np.nanmedian(np.abs(difference))),
        median_relative_difference=float(
            np.nanmedian(np.abs(difference) / np.where(level > 0, level, np.nan))),
        consistent=bool(max_abs_z <= threshold))
