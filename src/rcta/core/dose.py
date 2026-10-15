"""Dose-rate units and gap-aware integration of accumulated dose.

Two independent things live here, and both exist because of a specific way of
getting the answer wrong.

*Units.* Dose rate is canonically nSv/h throughout this package, and this is
the only module permitted to convert. Scattering ``/ 1000`` through the GUI is
how a program ends up displaying microsieverts on one panel and nanosieverts
on another, and how an exported CSV ends up with a column whose unit depends
on which dialog the user last opened. Every conversion goes through
:func:`convert_dose_rate`, and every exported or displayed number carries its
unit string.

*Integration.* Accumulated dose is the integral of dose rate over time, which
sounds unambiguous until the detector stops logging for thirteen minutes. The
bundled track spans 9754 s but only observed 6126 s of it. Integrating
straight across every dropout assumes the detector kept reading while it was
off; excluding the dropouts assumes those seconds never happened. Neither is
"the" dose, they differ by 24 % on this file, and a program that silently
picks one is asserting something about the world that it cannot know.

So :class:`DoseSummary` carries both totals and the coverage fraction, and
:meth:`DoseSummary.describe` prints all three and refuses to nominate a
winner. The caller may choose; the library will not choose for them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rcta.core.track import Track

__all__ = [
    "DOSE_UNITS",
    "DoseSummary",
    "accumulated_dose",
    "accumulated_unit",
    "convert_dose_rate",
    "format_accumulated_dose",
    "format_dose_rate",
]

#: Multipliers from the canonical nSv/h. The roentgen equivalence is the
#: conventional survey-meter one for gamma, 1 R = 10 mSv, so 1 uR/h = 10 nSv/h.
DOSE_UNITS: dict[str, float] = {
    "nSv/h": 1.0,
    "uSv/h": 1e-3,
    "mSv/h": 1e-6,
    "uR/h": 0.1,
}

#: Sensible display precision per unit, so a microsievert reading does not
#: render as "0.09" and throw away two significant figures.
_PRECISION = {"nSv/h": 2, "uSv/h": 4, "mSv/h": 7, "uR/h": 3}


def convert_dose_rate(values, to_unit: str = "nSv/h"):
    """Convert from the canonical nSv/h to ``to_unit``."""
    try:
        factor = DOSE_UNITS[to_unit]
    except KeyError:
        raise ValueError(
            f"unknown dose unit {to_unit!r}; expected one of "
            f"{sorted(DOSE_UNITS)}") from None
    return np.asarray(values, dtype=float) * factor


def format_dose_rate(value: float, unit: str = "nSv/h") -> str:
    """One dose rate as text, with its unit and a sensible precision."""
    converted = float(convert_dose_rate(value, unit))
    return f"{converted:.{_PRECISION.get(unit, 3)}f} {unit}"


def accumulated_unit(rate_unit: str = "nSv/h") -> str:
    """The dose unit belonging to a dose *rate* unit: ``nSv/h`` -> ``nSv``.

    An accumulated dose is a dose, not a rate, so it cannot simply carry the
    rate's unit string -- but it must track the same choice, or a report ends
    up quoting a total in nanosieverts beside a mean in microsieverts. The
    multiplier is the same one either way: nSv/h -> uSv/h and nSv -> uSv are
    both 1e-3, and nSv -> uR is the 0.1 that 1 uR = 10 nSv gives.
    """
    if rate_unit not in DOSE_UNITS:
        raise ValueError(
            f"unknown dose unit {rate_unit!r}; expected one of "
            f"{sorted(DOSE_UNITS)}")
    return rate_unit[:-len("/h")]


def format_accumulated_dose(value_nsv: float, rate_unit: str = "nSv/h") -> str:
    """One accumulated dose as text, in the unit matching ``rate_unit``.

    One digit finer than the rate: a total is integrated over the whole track
    and carries correspondingly more significant figures than an instantaneous
    reading of the same magnitude.
    """
    converted = float(convert_dose_rate(value_nsv, rate_unit))
    precision = _PRECISION.get(rate_unit, 3) + 1
    return f"{converted:.{precision}f} {accumulated_unit(rate_unit)}"


@dataclass(frozen=True)
class DoseSummary:
    """Integrated dose, reported with the coverage that qualifies it."""

    #: Integrated over observed intervals only, nSv.
    total_nsv: float
    #: Integrated straight across every dropout as though logging continued.
    naive_total_nsv: float
    covered_time_s: float
    span_time_s: float
    n_gaps_excluded: int
    max_dose_rate: float
    unit: str = "nSv/h"

    @property
    def uncovered_time_s(self) -> float:
        return self.span_time_s - self.covered_time_s

    @property
    def coverage(self) -> float:
        return self.covered_time_s / self.span_time_s if self.span_time_s else 1.0

    @property
    def mean_dose_rate(self) -> float:
        """Time-weighted mean over the intervals actually observed, nSv/h.

        NaN when no interval was observed: there is no mean of nothing, and
        0.0 would read as a measured zero dose rate.
        """
        if self.covered_time_s <= 0:
            return float("nan")
        return self.total_nsv / (self.covered_time_s / 3600.0)

    def describe(self) -> str:
        """Both totals, the coverage, and no recommendation between them."""
        lines = [
            f"observed   : {format_accumulated_dose(self.total_nsv, self.unit)}"
            f" over {self.covered_time_s:.0f} s actually logged",
            f"naive      : "
            f"{format_accumulated_dose(self.naive_total_nsv, self.unit)}"
            f" integrating straight across {self.n_gaps_excluded} dropout(s)",
            f"coverage   : {self.coverage * 100.0:.1f} % of a "
            f"{self.span_time_s:.0f} s span",
            "mean rate  : "
            + (format_dose_rate(self.mean_dose_rate, self.unit)
               if self.covered_time_s > 0 else "undefined, nothing observed")
            + f"   peak {format_dose_rate(self.max_dose_rate, self.unit)}",
        ]
        if self.n_gaps_excluded and self.naive_total_nsv > 0:
            spread = abs(self.naive_total_nsv - self.total_nsv)
            percent = spread / self.naive_total_nsv * 100.0
            lines.append(
                f"note       : the two totals differ by {percent:.0f} %. Neither "
                "is 'the' dose -- the first assumes nothing was measured during "
                "the dropouts, the second that the rate held through them.")
        return "\n".join(lines)


def accumulated_dose(track: Track, *, quality=None, max_gap_s: float = 30.0,
                     mask=None, unit: str = "nSv/h") -> DoseSummary:
    """Trapezoidal integration of dose rate over time, split at dropouts.

    Integration is over *time*, so it is deliberately unaffected by the GPS
    position mask: a fix with 397 m of horizontal error still measured a real
    dose rate, and discarding it would understate the exposure for a reason
    that has nothing to do with the detector.

    ``quality`` supplies the dropout boundaries when it is available, so the
    definition of a gap matches the one the map and the legs already use. When
    it is absent, ``max_gap_s`` is applied directly.
    """
    if unit not in DOSE_UNITS:
        raise ValueError(f"unknown dose unit {unit!r}")

    time_s = track.time_s
    rate = track.dose_rate
    valid = track.valid_mask
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.size != track.n:
            raise ValueError(f"mask has {mask.size} entries, expected {track.n}")
        if mask.sum() < 2:
            raise ValueError("need at least two samples to integrate a dose")
        time_s, rate, valid = time_s[mask], rate[mask], valid[mask]

    n = time_s.size
    peak = float(rate[valid].max()) if valid.any() else 0.0
    if n < 2:
        return DoseSummary(total_nsv=0.0, naive_total_nsv=0.0, covered_time_s=0.0,
                           span_time_s=0.0, n_gaps_excluded=0,
                           max_dose_rate=peak, unit=unit)

    dt = np.diff(time_s)
    # Trapezoid over each interval: nSv/h * s / 3600 = nSv.
    per_interval = 0.5 * (rate[:-1] + rate[1:]) * dt / 3600.0

    if quality is not None and mask is None and quality.gap_after.size == dt.size:
        is_gap = quality.gap_after
    else:
        is_gap = dt > max_gap_s

    # An interval with a masked end has no reading to integrate, so it is
    # unobserved time exactly like a dropout. The naive total bridges it from
    # the readings on either side, as it bridges a dropout.
    observed = ~is_gap & valid[:-1] & valid[1:]
    t_ok, r_ok = time_s[valid], rate[valid]
    naive = float((0.5 * (r_ok[:-1] + r_ok[1:]) * np.diff(t_ok)).sum() / 3600.0)

    return DoseSummary(
        total_nsv=float(per_interval[observed].sum()),
        naive_total_nsv=naive,
        covered_time_s=float(dt[observed].sum()),
        span_time_s=float(time_s[-1] - time_s[0]),
        n_gaps_excluded=int(is_gap.sum()),
        max_dose_rate=peak,
        unit=unit,
    )
