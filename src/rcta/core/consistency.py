"""Detector consistency: do the dose rate and the count rate agree?

The two columns of a track are not independent measurements. The dose rate
is the count rate times an energy-dependent factor the device applies, so
their ratio drifts slowly -- 13 nSv/h per cps on the ground, 51 at cruise --
but cannot jump. A sample that breaks that relation is a logging fault, not
radiation: a dose rate with no counts behind it, counts with no dose rate,
or a ratio several times off its neighbours'.

The check reports and never masks. Whether a sample is broken is a judgement
about the device, and silently dropping readings because they look odd is
how a genuine spike would be thrown away. A warning names how many samples
disagree and where, so the user can look.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rcta.core.track import Track

__all__ = ["ConsistencyReport", "check_consistency"]

#: A ratio this many times off its local median is a jump, not a drift.
RATIO_JUMP_FACTOR = 3.0
#: Samples either side in the local median of the dose/count ratio.
_HALF_WINDOW = 15


@dataclass(frozen=True)
class ConsistencyReport:
    """Samples whose dose rate and count rate disagree."""

    dose_without_counts: tuple[int, ...]
    counts_without_dose: tuple[int, ...]
    ratio_jumps: tuple[int, ...]

    @property
    def n_inconsistent(self) -> int:
        return len(set(self.dose_without_counts) | set(self.counts_without_dose)
                   | set(self.ratio_jumps))

    @property
    def warnings(self) -> tuple[str, ...]:
        out = []
        for indices, what in (
                (self.dose_without_counts, "a dose rate but zero counts"),
                (self.counts_without_dose, "counts but a zero dose rate"),
                (self.ratio_jumps,
                 f"a dose/count ratio over {RATIO_JUMP_FACTOR:g}x off its "
                 "neighbours'")):
            if indices:
                shown = ", ".join(str(i) for i in indices[:8])
                more = f" and {len(indices) - 8} more" if len(indices) > 8 else ""
                out.append(f"{len(indices)} sample(s) report {what} "
                           f"(indices {shown}{more}); likely logging faults, "
                           "kept in the analysis")
        return tuple(out)


def check_consistency(track: Track) -> ConsistencyReport:
    """Flag samples whose dose and count rates cannot both be right."""
    ok = track.valid_mask
    dose, counts = track.dose_rate, track.count_rate
    dose_only = np.flatnonzero(ok & (dose > 0) & (counts <= 0))
    counts_only = np.flatnonzero(ok & (counts > 0) & (dose <= 0))

    both = ok & (dose > 0) & (counts > 0)
    ratio = np.full(track.n, np.nan)
    ratio[both] = dose[both] / counts[both]
    jumps = np.zeros(0, dtype=np.int64)
    if both.sum() > 2 * _HALF_WINDOW:
        padded = np.pad(ratio, _HALF_WINDOW, mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(
            padded, 2 * _HALF_WINDOW + 1)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                # An all-NaN window (a run of masked samples) has no median.
                warnings.simplefilter("ignore", RuntimeWarning)
                local = np.nanmedian(windows, axis=1)
            relative = ratio / local
        off = both & np.isfinite(relative) & (
            (relative > RATIO_JUMP_FACTOR) | (relative < 1.0 / RATIO_JUMP_FACTOR))
        jumps = np.flatnonzero(off)
    return ConsistencyReport(
        dose_without_counts=tuple(int(i) for i in dose_only),
        counts_without_dose=tuple(int(i) for i in counts_only),
        ratio_jumps=tuple(int(i) for i in jumps))
