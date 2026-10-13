"""Counting statistics for a moving detector: Currie limits and honest p-values.

The Currie algebra here is the standard formulation, used unchanged, and
that is deliberate rather than lazy. Nothing about
sigma_0^2 = B(1 + n_p/n_b) cares whether the "peak region" is a band of energy
channels or a window of time, or whether the "side-band" is adjacent channels
or adjacent seconds. Keeping the two implementations character-identical means
a detection limit quoted by this program is directly comparable with one
quoted by the spectrum tool, and a bug fixed in one is fixed in both.

What *is* new here is everything that follows from samples being ordered in
time rather than in energy, and it is where the statistical traps live.

*Windows must be measured in seconds, not in samples.* A track logs every 2 s
until it drops out for 763 s. An index-based rolling window silently becomes a
twenty-minute window across that dropout, so the "local" background is drawn
from data twelve kilometres away.

*Consecutive samples are not independent.* The bundled track's count rate has
a lag-1 autocorrelation of 0.9985, which for 2964 samples is an effective
sample size of about two. Any test that assumes independence -- which is every
textbook correlation p-value -- will report overwhelming significance for
nothing at all. :func:`autocorr_time` measures it and
:func:`block_bootstrap_p` works around it by resampling contiguous blocks
rather than points, so the null distribution inherits the same serial
structure as the data.

*A low quantile is not the median.* Background is estimated with a low
quantile of a window precisely because the median of a window that is half
hotspot is the hotspot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "CurrieLimits",
    "K_95",
    "autocorr_time",
    "block_bootstrap_p",
    "effective_p",
    "currie_limits",
    "net_significance",
    "rolling_quantile",
    "rolling_trimmed_mean",
    "spearman_rho",
]

#: One-sided normal quantile for 95 % confidence (alpha = beta = 0.05).
K_95 = 1.6448536269514722


@dataclass(frozen=True)
class CurrieLimits:
    """Currie (1968) decision thresholds for a window, in counts.

    ``critical_level`` (L_C) is the decision threshold: net counts above it are
    called a detection at the stated false-positive rate. ``detection_limit``
    (L_D) is the true net signal needed to be found with probability
    ``1 - beta``, and it is L_D -- not L_C -- that converts into an MDA.
    """

    critical_level: float
    detection_limit: float
    background_counts: float
    n_peak_channels: int
    n_background_channels: int
    k_alpha: float
    k_beta: float
    live_time: float

    @property
    def mda_cps(self) -> float:
        """Minimum detectable net count rate over this window."""
        return self.detection_limit / self.live_time

    def mda_dose_rate(self, dose_per_count: float) -> float:
        """MDA expressed in nSv/h, given the track's own nSv/h per cps.

        There is no efficiency curve in this program and no way to produce one
        from a track, so the conversion is the empirical ratio the detector
        itself reported -- see :attr:`rcta.core.track.Track.dose_per_count`.
        """
        return self.mda_cps * float(dose_per_count)

    def describe(self) -> str:
        return (f"B = {self.background_counts:.1f} counts, "
                f"L_C = {self.critical_level:.1f}, "
                f"L_D = {self.detection_limit:.1f}, "
                f"MDA = {self.mda_cps:.3f} cps")


def currie_limits(
    background_counts: float,
    n_peak_channels: int,
    n_background_channels: int | None = None,
    *,
    live_time: float = 1.0,
    k_alpha: float = K_95,
    k_beta: float = K_95,
) -> CurrieLimits:
    """Critical level and detection limit for a window of a count-rate series.

    ``background_counts`` is the background integrated *over the tested
    window*, not the total of the reference region. ``n_peak_channels`` is the
    number of samples in the window and ``n_background_channels`` the number
    the background estimate was drawn from; ``None`` means the background is
    known exactly, which is an idealisation no real track supplies.

    The null-hypothesis variance of the net signal is

        sigma_0^2 = B * (1 + n_p / n_b)

    so a background estimated from a short reference window costs real
    sensitivity. Then ``L_C = k_alpha * sigma_0`` and

        L_D = k_alpha * k_beta + (k_alpha + k_beta) * sigma_0

    which at the usual 95 %/95 % choice is the familiar ``2.71 + 3.29 sigma_0``.
    """
    if background_counts < 0:
        raise ValueError("background_counts must be non-negative")
    if n_peak_channels <= 0:
        raise ValueError("n_peak_channels must be positive")
    if live_time <= 0:
        raise ValueError("live_time must be positive")

    if n_background_channels is None:
        ratio = 0.0
        n_bg = 0
    else:
        if n_background_channels <= 0:
            raise ValueError("n_background_channels must be positive or None")
        ratio = n_peak_channels / n_background_channels
        n_bg = int(n_background_channels)

    sigma_0 = float(np.sqrt(background_counts * (1.0 + ratio)))
    return CurrieLimits(
        critical_level=k_alpha * sigma_0,
        detection_limit=k_alpha * k_beta + (k_alpha + k_beta) * sigma_0,
        background_counts=float(background_counts),
        n_peak_channels=int(n_peak_channels),
        n_background_channels=n_bg,
        k_alpha=k_alpha,
        k_beta=k_beta,
        live_time=live_time,
    )


def net_significance(gross_counts, background_counts, n_peak_channels,
                     n_background_channels=None):
    """Net signal divided by its null-hypothesis standard deviation.

    Uses the *gross* counts for the window variance and the reference-window
    scaling for the background, so the result is directly comparable with the
    Currie limits above rather than with the naive ``net / sqrt(net)``.
    Vectorised over arrays so a whole scan is one call.
    """
    gross = np.asarray(gross_counts, dtype=float)
    background = np.asarray(background_counts, dtype=float)
    n_peak = np.asarray(n_peak_channels, dtype=float)
    if n_background_channels is None:
        ratio = np.zeros_like(gross)
    else:
        n_bg = np.asarray(n_background_channels, dtype=float)
        ratio = np.where(n_bg > 0, n_peak / np.maximum(n_bg, 1e-12), 0.0)

    net = gross - background
    var = np.maximum(gross, 0.0) + np.maximum(background, 0.0) * ratio
    out = np.where(var > 0, net / np.sqrt(np.maximum(var, 1e-300)), 0.0)
    return float(out) if out.ndim == 0 else out


# ---- time-series helpers --------------------------------------------------


def rolling_quantile(values, time_s, *, window_s: float, q: float = 0.25,
                     min_samples: int = 5):
    """Centred rolling quantile over a *time* window.

    Returns ``(quantile, n_effective)``. The window is defined in seconds, not
    in samples, because a track's spacing varies by three orders of magnitude:
    an index window of 300 samples is ten minutes during normal logging and
    several hours if it straddles a dropout, and the resulting "local"
    background would be drawn from a different state entirely.

    Windows with fewer than ``min_samples`` contributors fall back to the
    global quantile and report their true (small) count, so the caller can see
    which estimates are unsupported rather than trusting a quantile of two
    points.
    """
    values = np.asarray(values, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    n = values.size
    if n == 0:
        return np.zeros(0), np.zeros(0, dtype=np.int64)

    half = float(window_s) / 2.0
    # searchsorted on a sorted time column is O(n log n) overall and, unlike a
    # stride trick, is correct when the spacing is irregular.
    left = np.searchsorted(time_s, time_s - half, side="left")
    right = np.searchsorted(time_s, time_s + half, side="right")
    counts = (right - left).astype(np.int64)

    fallback = float(np.quantile(values, q)) if n else 0.0
    out = np.full(n, fallback, dtype=float)
    for i in range(n):
        lo, hi = left[i], right[i]
        if hi - lo >= min_samples:
            out[i] = np.quantile(values[lo:hi], q)
    return out, counts


def rolling_trimmed_mean(values, time_s, *, window_s: float,
                         trim_fraction: float = 0.25, min_samples: int = 5):
    """Centred rolling symmetrically-trimmed mean over a *time* window.

    This is the background estimator, and it is a trimmed mean rather than a
    low quantile for a reason that was measured rather than assumed.

    A low quantile is robust to hotspots but badly *biased*: the 0.25 quantile
    of a Poisson background at 18 cps sits 2.0 cps below its mean. Over a
    64-sample window at 2 s that is 256 counts of excess that no source put
    there, against a counting error of 48 counts -- a 5.3 sigma detection
    manufactured out of the estimator's own bias, on every window, everywhere
    on the track. Robustness is worthless if it is paid for in false positives.

    Trimming the same fraction from *both* tails keeps the robustness and
    removes the bias, because the remaining sample is centred. Measured on
    Poisson data at 18 cps: bias -0.07 cps on a clean window (against -2.05
    for the quantile), and +0.40 cps when a tenth of the window is a fivefold
    hotspot (against +7.20 for an untrimmed mean).

    Returns ``(estimate, n_contributing)``. Time-based windowing, for the same
    reason as :func:`rolling_quantile`: sample spacing varies by three orders
    of magnitude across a dropout.
    """
    values = np.asarray(values, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    n = values.size
    if n == 0:
        return np.zeros(0), np.zeros(0, dtype=np.int64)
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError(
            f"trim_fraction must be in [0, 0.5), got {trim_fraction}")

    half = float(window_s) / 2.0
    left = np.searchsorted(time_s, time_s - half, side="left")
    right = np.searchsorted(time_s, time_s + half, side="right")
    counts = (right - left).astype(np.int64)

    fallback = float(_trimmed_mean(values, trim_fraction))
    out = np.full(n, fallback, dtype=float)
    for i in range(n):
        lo, hi = left[i], right[i]
        if hi - lo >= min_samples:
            out[i] = _trimmed_mean(values[lo:hi], trim_fraction)
    return out, counts


def _trimmed_mean(values: np.ndarray, fraction: float) -> float:
    """Mean after dropping ``fraction`` of the sample from each tail."""
    ordered = np.sort(values)
    k = int(round(fraction * ordered.size))
    kept = ordered[k:ordered.size - k] if ordered.size - 2 * k > 0 else ordered
    return float(kept.mean())


def autocorr_time(values, time_s=None):
    """``(lag-1 autocorrelation, effective sample size)``.

    The effective size is the standard AR(1) correction ``N (1-r)/(1+r)``. On
    the bundled track this returns about 2 for 2964 samples, which is the
    single most important number to look at before quoting any p-value.
    """
    values = np.asarray(values, dtype=float)
    n = values.size
    if n < 3:
        return 0.0, float(n)
    centred = values - values.mean()
    denom = float(np.dot(centred, centred))
    if denom <= 0:
        return 0.0, float(n)
    r = float(np.dot(centred[:-1], centred[1:]) / denom)
    r = min(max(r, -0.999999), 0.999999)
    return r, float(n * (1.0 - r) / (1.0 + r))


def spearman_rho(x, y) -> float:
    """Spearman rank correlation, with average ranks for ties."""
    from scipy.stats import rankdata

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rx = rankdata(x)
    ry = rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt(np.dot(rx, rx) * np.dot(ry, ry)))
    return float(np.dot(rx, ry) / denom) if denom > 0 else 0.0


def effective_p(rho: float, n_effective: float) -> float:
    """Two-sided p-value for a correlation, corrected for serial dependence.

    This is the number to trust, and it is the one :class:`Correlation`
    reports as its headline. The Bartlett/Dawdy-Matalas correction replaces
    the nominal sample size with the effective one implied by both series'
    autocorrelation, then applies the ordinary Student-t test on the reduced
    degrees of freedom.

    The correction is brutal by design. On the bundled track the effective
    size is about two, which leaves zero degrees of freedom and a p-value of
    one: a correlation of 0.64 over 2964 serially dependent samples is not
    evidence of anything. Reporting the nominal p of 1e-300 instead would
    manufacture a confident physical claim out of two independent
    observations.
    """
    df = float(n_effective) - 2.0
    if df <= 0 or not np.isfinite(rho):
        return 1.0
    rho = min(max(float(rho), -0.999999), 0.999999)
    from scipy import stats

    t = rho * np.sqrt(df / (1.0 - rho * rho))
    return float(2.0 * stats.t.sf(abs(t), df))


def bartlett_effective_n(x, y) -> float:
    """Effective sample size for a correlation between two serial series.

    ``N (1 - r1 r2) / (1 + r1 r2)`` using each series' own lag-1
    autocorrelation. Using only one series' autocorrelation, as a naive
    correction does, understates the penalty whenever both are smooth -- which
    on a radiation track they invariably are.
    """
    n = int(np.asarray(x).size)
    r1, _ = autocorr_time(x)
    r2, _ = autocorr_time(y)
    product = min(max(r1 * r2, -0.999999), 0.999999)
    return float(n * (1.0 - product) / (1.0 + product))


def block_bootstrap_p(x, y, time_s, *, statistic=None, block_s: float = 300.0,
                      n_boot: int = 2000, seed: int = 0) -> float:
    """Two-sided p-value for an association between two serial time series.

    Contiguous blocks of ``block_s`` seconds of ``y`` are shuffled -- a
    permutation of whole blocks, not a resample with replacement -- against
    an unchanged ``x`` to build the null distribution. Each surrogate thus
    inherits the same short-range serial structure as the data and only the
    *pairing* between the two series is destroyed. Shuffling individual points
    instead would destroy the autocorrelation as well, and would reproduce
    exactly the false confidence this function exists to avoid.

    ``block_s`` should comfortably exceed the autocorrelation time; the default
    of five minutes is several times the two-second logging cadence and is
    reported alongside the result so the choice is visible.

    **Known limitation, and the reason this is not the headline p-value.**
    Block resampling preserves dependence only up to the block length. For a
    series with a near-unit root -- a random walk, or a dose rate climbing
    monotonically with altitude -- the long-range trend is destroyed by the
    resampling, the surrogates are far less correlated than the data, and the
    test is anticonservative: it will call two independent random walks
    significant. :func:`effective_p` does not have that failure mode, and
    :class:`~rcta.core.profile.Correlation` reports it as the primary result
    with this one alongside for comparison. There is a regression test that
    pins this difference so nobody later "fixes" it by promoting the bootstrap.
    """
    statistic = statistic or spearman_rho
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    n = x.size
    if n < 8:
        return 1.0

    observed = abs(statistic(x, y))
    span = float(time_s[-1] - time_s[0])
    block_s = float(min(max(block_s, 1e-9), max(span, 1e-9)))
    n_blocks = max(int(np.ceil(span / block_s)), 1)
    if n_blocks < 2:
        return 1.0

    # Block boundaries by time, so blocks are equal in seconds not in samples.
    edges = np.searchsorted(time_s, time_s[0] + np.arange(n_blocks + 1) * block_s)
    edges[-1] = n
    blocks = [(edges[i], edges[i + 1]) for i in range(n_blocks)
              if edges[i + 1] > edges[i]]
    if len(blocks) < 2:
        return 1.0

    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(int(n_boot)):
        order = rng.permutation(len(blocks))
        pieces = [y[blocks[j][0]:blocks[j][1]] for j in order]
        surrogate = np.concatenate(pieces)
        # Block lengths differ slightly; trim to the common length.
        m = min(surrogate.size, n)
        if abs(statistic(x[:m], surrogate[:m])) >= observed:
            exceed += 1
    # Add-one smoothing: a bootstrap can never justify reporting p = 0.
    return (exceed + 1) / (int(n_boot) + 1)
