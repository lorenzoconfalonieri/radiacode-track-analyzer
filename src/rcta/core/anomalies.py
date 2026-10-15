"""Multi-scale Currie scan for localised count-rate excesses.

The job is to find stretches of track where the count rate is higher than the
local background by more than counting statistics can explain, and -- far more
importantly -- to *not* find them anywhere else. Three design choices carry
that second half, and each is a response to a way of generating hundreds of
confident nonsense detections.

*Multi-scale, because sensitivity depends entirely on window width.* At the
bundled track's cruise background of about 18 cps with a 2 s integration, a
single sample has an MDA of roughly 10 cps: a source would have to double the
count rate to be seen. Integrate sixty samples and the MDA falls by nearly an
order of magnitude. A single-width scan is therefore blind to either brief
intense sources or extended weak ones, depending on which width was chosen.
Geometric widths mean any real feature is within a factor of root two of a
matched window.

*Bonferroni, because a scan is not one test.* Scanning 2964 samples at seven
widths is 20 748 tests. At a nominal one-sided alpha of 0.05 the threshold is
1.64 sigma, and a perfectly clean track will hand back on the order of a
thousand "detections". Correcting moves the threshold to about 4.4 sigma and
the count to zero. A synthetic pure-background regression test pins this.

*Trends are flagged, not suppressed.* A twenty-minute monotone climb in cosmic
dose really is an excess over any local baseline, and hiding it would be
dishonest; calling it a hotspot would be worse. So a feature that a local
straight line already explains gets a ``trend`` flag and is excluded from the
significant set, while remaining visible in the table with its reason attached.

On the bundled file this module's correct output is *nothing*. The only
features above five sigma are the climb-out and the descent, and both come
back flagged. The MDA at each scale is the result: when a search finds
nothing, the detection limit is the finding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rcta.core import statistics as st
from rcta.core.background import BackgroundEstimate
from rcta.core.statistics import CurrieLimits
from rcta.core.track import Track

__all__ = ["Anomaly", "ScanParams", "bonferroni_k", "scan", "scan_mda",
           "significance_trace"]


@dataclass(frozen=True)
class ScanParams:
    """Configuration for the multi-scale scan."""

    #: Window widths in samples. Geometric, so cost is O(N log N) and any
    #: feature is within sqrt(2) of a matched width.
    scales: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    #: Nominal one-sided false-positive rate, before correction.
    alpha: float = 0.05
    #: Correct for the number of windows tested. Without this a clean track
    #: yields roughly a thousand detections.
    bonferroni: bool = True
    #: Merge overlapping detections across scales, keeping the strongest.
    merge_overlapping: bool = True
    #: A window whose excess is at least this well explained by a local
    #: straight line is flagged as a trend rather than called a hotspot.
    trend_flag_ratio: float = 0.5
    #: Windows whose GPS is mostly untrustworthy get a flag, not exclusion.
    poor_gps_fraction: float = 0.5

    def __post_init__(self) -> None:
        if not self.scales or any(s < 1 for s in self.scales):
            raise ValueError("scales must be positive sample counts")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")


@dataclass(frozen=True)
class Anomaly:
    """A localised count-rate excess over the local background."""

    start_index: int
    stop_index: int          #: exclusive
    scale: int
    time_s: float
    duration_s: float
    lat: float
    lon: float
    distance_along_m: float
    length_m: float
    gross_counts: float
    background_counts: float
    net_counts: float
    net_count_rate: float
    net_dose_rate: float
    peak_dose_rate: float
    significance: float
    threshold_sigma: float
    limits: CurrieLimits
    n_background: float
    flags: tuple[str, ...] = ()

    @property
    def is_significant(self) -> bool:
        """Both criteria, because either alone is too easy to satisfy.

        Net counts must clear the Currie critical level *and* the significance
        must clear the (corrected) threshold. They can disagree when the
        background support is thin, and a detection either test rejects is not
        one worth reporting.
        """
        return (not self.flags_block_detection
                and self.net_counts > self.limits.critical_level
                and self.significance > self.threshold_sigma)

    @property
    def flags_block_detection(self) -> bool:
        return "trend" in self.flags

    def describe(self) -> str:
        flags = f"  [{', '.join(self.flags)}]" if self.flags else ""
        return (f"{self.duration_s:6.0f} s  "
                f"{abs(self.lat):8.5f}{'N' if self.lat >= 0 else 'S'} "
                f"{abs(self.lon):9.5f}{'E' if self.lon >= 0 else 'W'}  "
                f"net {self.net_count_rate:+7.2f} cps "
                f"({self.net_dose_rate:+8.1f} nSv/h)  "
                f"{self.significance:6.2f} sigma{flags}")


def bonferroni_k(n_tests: int, alpha: float = 0.05) -> float:
    """One-sided normal quantile for ``alpha / n_tests``."""
    from scipy import stats

    n_tests = max(int(n_tests), 1)
    return float(stats.norm.isf(alpha / n_tests))


def _window_sums(values: np.ndarray, width: int) -> np.ndarray:
    """Sliding sums of ``width`` consecutive samples, length n - width + 1."""
    prefix = np.concatenate(([0.0], np.cumsum(values)))
    return prefix[width:] - prefix[:-width]


def _rates(track: Track) -> np.ndarray:
    """Count rates with masked samples zeroed, so a prefix sum stays finite.

    The zeros are never tested: :func:`_blocked` excludes every window that
    contains one.
    """
    return np.where(track.valid_mask, track.count_rate, 0.0)


def _blocked(track: Track, width: int) -> np.ndarray:
    """Windows of ``width`` that contain a masked sample and so are not tested."""
    return _window_sums((~track.valid_mask).astype(float), width) > 0


def _threshold(track: Track, params: ScanParams) -> float:
    """The significance a window must clear, corrected for every window tested.

    Only windows that are actually tested count, so a masked sample removes
    its windows from the correction as well as from the scan.
    """
    if not params.bonferroni:
        return bonferroni_k(1, params.alpha)
    n_tests = sum(int((~_blocked(track, s)).sum())
                  for s in sorted(set(params.scales)) if s <= track.n)
    return bonferroni_k(n_tests, params.alpha)


def significance_trace(track: Track, background: BackgroundEstimate, *,
                       scale: int = 8) -> np.ndarray:
    """Per-sample significance at one width, for the GUI's strip plot.

    It shows *why* a stretch of track was or was not called an anomaly,
    rather than leaving the threshold invisible and the verdict unexplained.
    """
    scale = max(int(scale), 1)
    live = track.effective_integration_time_s
    gross = _window_sums(_rates(track), scale) * live
    back = _window_sums(background.count_rate, scale) * live
    n_bg = np.maximum(_window_sums(background.n_effective.astype(float), scale)
                      / scale, 1.0)

    sig = st.net_significance(gross, back, np.full(gross.size, scale), n_bg)
    sig[_blocked(track, scale)] = 0.0
    # Centre each window's value on its own midpoint so the strip lines up
    # with the series above it.
    out = np.zeros(track.n)
    offset = scale // 2
    out[offset:offset + sig.size] = sig
    return out


def scan(track: Track, background: BackgroundEstimate,
         params: ScanParams | None = None, *, quality=None,
         progress=None, cancelled=None) -> list[Anomaly]:
    """Multi-scale Currie scan. The single entry point for CLI, GUI and tests.

    ``progress`` and ``cancelled`` are plain callables -- ``(fraction) -> None``
    and ``() -> bool``. They let the GUI drive this from a worker thread
    without this module ever learning that Qt exists, which is what keeps the
    architecture test satisfied.
    """
    params = params or ScanParams()
    live = track.effective_integration_time_s
    n = track.n

    usable_scales = [s for s in sorted(set(params.scales)) if s <= n]
    if not usable_scales:
        return []

    threshold = _threshold(track, params)
    rates = _rates(track)

    found: list[Anomaly] = []
    for step, width in enumerate(usable_scales):
        if cancelled is not None and cancelled():
            return []
        if progress is not None:
            progress(step / len(usable_scales))

        gross = _window_sums(rates, width) * live
        back = _window_sums(background.count_rate, width) * live
        support = np.maximum(
            _window_sums(background.n_effective.astype(float), width) / width, 1.0)
        widths = np.full(gross.size, width)
        sig = st.net_significance(gross, back, widths, support)
        sig[_blocked(track, width)] = -np.inf

        candidates = np.flatnonzero(sig > threshold)
        for start in candidates:
            found.append(_build(track, background, int(start), width, live,
                                float(gross[start]), float(back[start]),
                                float(support[start]), float(sig[start]),
                                threshold, params, quality))

    if progress is not None:
        progress(1.0)
    if params.merge_overlapping:
        found = _merge(found)
    found.sort(key=lambda a: a.significance, reverse=True)
    return found


def _build(track, background, start, width, live, gross, back, support, sig,
           threshold, params: ScanParams, quality) -> Anomaly:
    stop = start + width
    rows = slice(start, stop)
    net = gross - back

    counts = np.maximum(track.count_rate[rows], 0.0)
    weight = counts.sum()
    if weight > 0:
        lat = float(np.dot(track.lat[rows], counts) / weight)
        lon = float(np.dot(track.lon[rows], counts) / weight)
    else:
        lat = float(track.lat[rows].mean())
        lon = float(track.lon[rows].mean())

    duration = float(track.time_s[stop - 1] - track.time_s[start]) + live
    length = float(track.distance_along_m[stop - 1] - track.distance_along_m[start])

    flags: list[str] = []
    if width == 1:
        flags.append("single-sample")
    if _is_trend(track, background, start, stop, params.trend_flag_ratio):
        flags.append("trend")
    if (~track.position_mask[rows]).mean() > params.poor_gps_fraction:
        flags.append("poor-gps")
    if quality is not None and quality.gap_after.size:
        near = quality.gap_after[max(start - 1, 0):min(stop, quality.gap_after.size)]
        if near.size and near.any():
            flags.append("gap-adjacent")

    limits = st.currie_limits(max(back, 0.0), width, max(int(round(support)), 1),
                              live_time=width * live,
                              k_alpha=threshold, k_beta=st.K_95)
    return Anomaly(
        start_index=start, stop_index=stop, scale=width,
        time_s=float(track.time_s[start]), duration_s=duration,
        lat=lat, lon=lon,
        distance_along_m=float(track.distance_along_m[start]), length_m=length,
        gross_counts=gross, background_counts=back, net_counts=net,
        net_count_rate=net / (width * live),
        net_dose_rate=net / (width * live) * track.dose_per_count,
        peak_dose_rate=float(track.dose_rate[rows].max()),
        significance=sig, threshold_sigma=threshold, limits=limits,
        n_background=support, flags=tuple(flags),
    )


def _is_trend(track, background, start, stop, ratio: float) -> bool:
    """Is this excess already explained by a straight line through its flanks?

    A least-squares line is fitted to the count rate on *either side* of the
    tested window and extrapolated across it. The window's remaining excess
    over that line is compared with its excess over the flat background.

    The flanks-only fit is the whole trick, and getting it wrong is subtle:
    fitting the line through the window as well lets an isolated source lift
    the line to meet itself, so every genuine detection explains itself away
    and is dismissed as a trend. Excluding the window means a monotone climb
    is still predicted by its own surroundings, while a source sitting on a
    flat baseline is not predicted at all.
    """
    width = stop - start
    lo = max(start - 2 * width, 0)
    hi = min(stop + 2 * width, track.n)
    left = np.arange(lo, start)
    right = np.arange(stop, hi)
    flanks = np.concatenate([left, right])
    flanks = flanks[track.valid_mask[flanks]]
    if flanks.size < 4:
        return False

    origin = float(track.time_s[start])
    t = track.time_s[flanks].astype(float) - origin
    if float(t.max() - t.min()) <= 0:
        return False

    slope, intercept = np.polyfit(t, track.count_rate[flanks], 1)
    predicted = slope * (track.time_s[start:stop] - origin) + intercept

    observed = track.count_rate[start:stop]
    over_background = float((observed - background.count_rate[start:stop]).sum())
    if over_background <= 0:
        return False
    over_trend = float((observed - predicted).sum())
    explained = (over_background - over_trend) / over_background
    return explained >= ratio


def _merge(found: list[Anomaly]) -> list[Anomaly]:
    """Collapse overlapping windows by non-maximum suppression.

    Take the most significant candidate, emit it, discard everything that
    overlaps it, repeat. Its flags are its own and are not pooled across the
    group.

    Both of those details are corrections to the obvious implementation.
    Grouping by adjacency instead chains transitively -- with a dense scan,
    every candidate overlaps its neighbour and the whole track collapses into
    a single "anomaly", merging genuinely distinct sources. And pooling flags
    lets one broad trend-flagged window in the chain mark a sharp, clean
    detection as a trend, which is how a real source ends up dismissed.
    """
    remaining = sorted(found, key=lambda a: a.significance, reverse=True)
    kept: list[Anomaly] = []
    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        remaining = [a for a in remaining
                     if a.stop_index <= best.start_index
                     or a.start_index >= best.stop_index]
    return kept


def scan_mda(track: Track, background: BackgroundEstimate,
             params: ScanParams | None = None) -> list[tuple[int, float, float]]:
    """``(width, duration_s, MDA in nSv/h)`` at each scale.

    When the scan finds nothing -- which is the correct outcome for the
    bundled track -- this table *is* the result, and it is what the CLI prints
    in place of an empty detections list.
    """
    params = params or ScanParams()
    live = track.effective_integration_time_s
    threshold = _threshold(track, params)

    out = []
    median_support = max(float(np.median(background.n_effective)), 1.0)
    typical = float(np.median(background.count_rate))
    for width in sorted(set(params.scales)):
        # A scale at which every window holds a masked sample was never
        # tested, so it has no detection limit: quoting one would claim a
        # sensitivity for a measurement the scan did not make.
        if width > track.n or _blocked(track, width).all():
            continue
        back = typical * width * live
        limits = st.currie_limits(back, width, int(round(median_support)),
                                  live_time=width * live,
                                  k_alpha=threshold, k_beta=st.K_95)
        out.append((width, width * live,
                    limits.mda_dose_rate(track.dose_per_count)))
    return out
