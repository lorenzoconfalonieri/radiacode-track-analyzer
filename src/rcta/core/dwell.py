"""Stop-and-measure dwells: the most sensitive points on any walk.

When the detector stands still, every sample in the stretch looks at the same
patch of ground, so the counts add into one long measurement. Its detection
limit falls as ``1/sqrt(t)``: a 200 s dwell at 10 s logging integrates twenty
samples and reaches a limit no single moving sample can. Surveyors stop
deliberately for exactly this reason, and the multi-scale scan does not see
it -- its windows are a fixed number of samples, and a dwell is a span of
*standing still* of whatever length it was.

A dwell is a run of consecutive valid samples, inside one leg, whose smoothed
speed stays below ``max_speed_m_s`` for at least ``min_duration_s``. Smoothed
speed, not raw: on a walk the raw fixes jitter by metres per sample, which is
a walking pace of pure noise.

Each dwell is tested once against the background estimate with the same
Currie framework as the scan. The dwells of a track are a family of tests, so
the critical level is Bonferroni-corrected over how many there are; with one
dwell that is the plain one-sided threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rcta.core import statistics as st
from rcta.core.anomalies import bonferroni_k
from rcta.core.track import Track

__all__ = ["Dwell", "DwellParams", "find_dwells"]


@dataclass(frozen=True)
class DwellParams:
    """What counts as standing still."""

    #: Smoothed ground speed below which the detector is taken as stationary.
    #: A slow walk is about 1 m/s; GPS jitter left after smoothing is well
    #: under this on a handheld fix.
    max_speed_m_s: float = 0.3
    #: Shortest stretch reported. Below about two minutes a dwell adds little
    #: that the scan's own windows do not already cover.
    min_duration_s: float = 120.0
    #: False-positive rate over all the dwells of a track together.
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if self.max_speed_m_s <= 0:
            raise ValueError("max_speed_m_s must be positive")
        if self.min_duration_s <= 0:
            raise ValueError("min_duration_s must be positive")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")


@dataclass(frozen=True)
class Dwell:
    """One stationary stretch, measured as a single long count."""

    start_index: int
    stop_index: int               #: exclusive
    start_time_s: float
    duration_s: float             #: elapsed, first sample to last plus one
    n_samples: int
    live_time_s: float
    lat: float
    lon: float
    gross_counts: float
    background_counts: float
    net_counts: float
    net_count_rate: float
    net_dose_rate: float          #: nSv/h
    mean_dose_rate: float         #: nSv/h, over the dwell's readings
    significance: float
    limits: st.CurrieLimits
    dose_per_count: float

    @property
    def detected(self) -> bool:
        return self.net_counts > self.limits.critical_level

    @property
    def mda_dose_rate(self) -> float:
        """Smallest net dose rate this dwell would detect, nSv/h."""
        return self.limits.mda_dose_rate(self.dose_per_count)

    def describe(self, unit: str = "nSv/h") -> str:
        from rcta.core.dose import format_dose_rate, format_signed_dose_rate
        verdict = ("DETECTED" if self.detected else "nothing above L_C")
        return (f"{self.duration_s:5.0f} s at "
                f"{abs(self.lat):.5f}{'N' if self.lat >= 0 else 'S'} "
                f"{abs(self.lon):.5f}{'E' if self.lon >= 0 else 'W'}  "
                f"{self.n_samples} samples, {self.live_time_s:.0f} s live  "
                f"mean {format_dose_rate(self.mean_dose_rate, unit)}  "
                f"net {format_signed_dose_rate(self.net_dose_rate, unit)} "
                f"({self.significance:+.2f} sigma)  "
                f"MDA {format_dose_rate(self.mda_dose_rate, unit)}  {verdict}")


def find_dwells(track: Track, background, *, quality=None,
                params: DwellParams | None = None) -> list[Dwell]:
    """Every stationary stretch, in time order, each tested as one count.

    ``background`` is the track's :class:`BackgroundEstimate`; ``quality``
    supplies the leg boundaries, so a dwell never spans a dropout.
    """
    params = params or DwellParams()
    legs = quality.leg_index if quality is not None else track.leg_index
    slow = (track.speed_m_s < params.max_speed_m_s) & track.valid_mask

    runs = []
    start = None
    for i in range(track.n + 1):
        keep = (i < track.n and slow[i]
                and (start is None or legs[i] == legs[start]))
        if keep and start is None:
            start = i
        elif not keep and start is not None:
            runs.append((start, i))
            start = i if (i < track.n and slow[i]) else None

    live = track.effective_integration_time_s
    long_enough = [(a, b) for a, b in runs
                   if track.time_s[b - 1] - track.time_s[a] + live
                   >= params.min_duration_s]
    if not long_enough:
        return []
    k_alpha = bonferroni_k(len(long_enough), params.alpha)
    return [_measure(track, background, a, b, live, k_alpha)
            for a, b in long_enough]


def _measure(track, background, start, stop, live, k_alpha) -> Dwell:
    rows = slice(start, stop)
    n = stop - start
    gross = float(np.sum(track.count_rate[rows]) * live)
    back = float(np.sum(background.count_rate[rows]) * live)
    support = max(int(round(float(np.median(background.n_effective[rows])))), 1)
    net = gross - back
    limits = st.currie_limits(max(back, 0.0), n, support,
                              live_time=n * live, k_alpha=k_alpha,
                              k_beta=st.K_95)
    sig = float(st.net_significance(gross, back, n, support))
    positioned = track.position_mask[rows]
    lat = track.lat[rows][positioned] if positioned.any() else track.lat[rows]
    lon = track.lon[rows][positioned] if positioned.any() else track.lon[rows]
    rate = net / (n * live)
    return Dwell(
        start_index=start, stop_index=stop,
        start_time_s=float(track.time_s[start]),
        duration_s=float(track.time_s[stop - 1] - track.time_s[start]) + live,
        n_samples=n, live_time_s=n * live,
        lat=float(np.mean(lat)), lon=float(np.mean(lon)),
        gross_counts=gross, background_counts=back, net_counts=net,
        net_count_rate=rate, net_dose_rate=rate * track.dose_per_count,
        mean_dose_rate=float(np.mean(track.dose_rate[rows])),
        significance=sig, limits=limits,
        dose_per_count=float(track.dose_per_count),
    )
