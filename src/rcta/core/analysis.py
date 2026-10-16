"""The single analysis entry point shared by the CLI, the GUI and the tests.

This module exists to stop three callers from slowly growing three different
analyses. The reasoning is the ordinary one for a facade: the moment the GUI
assembles its own sequence
of core calls, it acquires its own defaults, its own ordering, and eventually
its own answers, and a number quoted in the status bar stops matching the same
number in an exported report.

So the whole pipeline is one function with one parameter object.
:func:`analyze_track` runs quality assessment, background estimation, the
multi-scale scan, hotspot clustering, dose integration and the summary in a
fixed order, and returns everything -- including the parameters it used and
every warning raised along the way -- in one frozen :class:`TrackAnalysis`.

The parameters are part of the result rather than an input the caller is
trusted to remember. A detection limit without the assumptions behind it is
not reproducible, and the most consequential assumption on a track file --
the per-sample integration time -- is one the file does not record at all and
the program has to infer.

The spatial grid is deliberately *not* computed here. It is the only step that
costs seconds rather than milliseconds, it is not needed to answer "is there
anything here", and making it opt-in keeps the facade fast enough that the GUI
can call it synchronously on load.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from rcta.core.anomalies import Anomaly, ScanParams, scan, scan_mda
from rcta.core.background import (BackgroundEstimate, BackgroundParams,
                                  estimate_background)
from rcta.core.dose import (DoseSummary, accumulated_dose,
                            format_accumulated_dose, format_dose_rate)
from rcta.core.profile import (Bimodality, Correlation, SummaryStats,
                               bimodality, correlate_speed_dose, summarise)
from rcta.core.quality import QualityParams, QualityReport, assess_with_report
from rcta.core.spatial import Hotspot, find_hotspots
from rcta.core.track import Track

__all__ = ["AnalysisParams", "TrackAnalysis", "analyze_track"]


@dataclass(frozen=True)
class AnalysisParams:
    """Everything that can change the answer, in one place."""

    quality: QualityParams = field(default_factory=QualityParams)
    background: BackgroundParams = field(default_factory=BackgroundParams)
    scan: ScanParams = field(default_factory=ScanParams)
    #: Detector live time per sample. None means infer from the median spacing
    #: and warn -- see Track.effective_integration_time_s.
    integration_time_s: float | None = None
    #: Display unit. The canonical stored unit is always nSv/h.
    unit: str = "nSv/h"
    #: Cluster radius for hotspots; None derives it from the GPS accuracy.
    cluster_radius_m: float | None = None
    #: Bootstrap resamples for the speed/dose correlation.
    n_boot: int = 500


@dataclass(frozen=True)
class TrackAnalysis:
    """Everything the program concluded, plus how it concluded it."""

    track: Track
    params: AnalysisParams
    quality: QualityReport
    background: BackgroundEstimate
    anomalies: tuple[Anomaly, ...]
    hotspots: tuple[Hotspot, ...]
    dose: DoseSummary
    summary: SummaryStats
    dose_distribution: Bimodality
    speed_dose: Correlation
    #: (window_samples, seconds, MDA in nSv/h) at each scanned scale.
    mda_table: tuple[tuple[int, float, float], ...]
    warnings: tuple[str, ...] = ()

    @property
    def significant(self) -> tuple[Anomaly, ...]:
        return tuple(a for a in self.anomalies if a.is_significant)

    @property
    def found_nothing(self) -> bool:
        return not self.significant

    def describe(self) -> str:
        """The whole result as text. The CLI prints this; the GUI logs it."""
        unit = self.params.unit
        blocks = [
            self.track.describe(unit),
            "",
            "-- gps quality " + "-" * 55,
            self.quality.describe(),
            "",
            "-- dose " + "-" * 62,
            self.dose.describe(),
            "",
            "-- distribution " + "-" * 54,
            self.dose_distribution.describe(),
            "",
            self.speed_dose.describe(),
            "",
            "-- background " + "-" * 56,
            self.background.describe(unit),
            "",
            "-- detections " + "-" * 56,
        ]
        if self.significant:
            blocks += [a.describe(unit) for a in self.significant]
        elif self.nothing_tested:
            # Not a null result: no window was tested, so there is neither a
            # detection nor a detection limit to report.
            blocks.append("nothing was tested: no scan window fits this track")
        else:
            blocks.append(
                f"no anomalies above L_C at "
                f"{self.anomalies[0].threshold_sigma:.2f} sigma"
                if self.anomalies else "no anomalies above L_C")
            blocks.append("")
            blocks.append("minimum detectable dose rate, by window:")
            for width, seconds, mda in self.mda_table:
                blocks.append(f"  {width:3d} samples ({seconds:6.0f} s): "
                              f"{format_dose_rate(mda, unit):>14}")

        flagged = [a for a in self.anomalies if not a.is_significant]
        if flagged:
            blocks += ["", f"{len(flagged)} candidate(s) excluded by flags:"]
            blocks += [f"  {a.describe(unit)}" for a in flagged[:10]]

        if self.hotspots:
            blocks += ["", "-- hotspots " + "-" * 58]
            blocks += [h.describe(unit) for h in self.hotspots]

        if self.warnings:
            blocks += ["", "-- warnings " + "-" * 58]
            blocks += [f"! {w}" for w in self.warnings]
        return "\n".join(blocks)

    @property
    def nothing_tested(self) -> bool:
        """No scan window fitted the track: neither a detection nor a limit."""
        return not self.anomalies and not self.mda_table

    def status_line(self, unit: str | None = None) -> str:
        """One line for the GUI status bar, dose in ``unit`` (default: the
        analysis's own)."""
        if self.significant:
            verdict = (f"{len(self.significant)} anomal"
                       f"{'y' if len(self.significant) == 1 else 'ies'} "
                       "above L_C")
        elif self.nothing_tested:
            verdict = "nothing tested: no scan window fits"
        else:
            verdict = "no anomalies above L_C"
        return (f"{self.track.n} samples  |  "
                f"{self.summary.distance_m / 1000.0:.2f} km  |  "
                f"{self.quality.n_legs} legs  |  "
                f"{format_accumulated_dose(self.dose.total_nsv, unit or self.dose.unit)}"
                f"  |  {verdict}")


def analyze_track(track: Track, params: AnalysisParams | None = None, *,
                  progress=None, cancelled=None) -> TrackAnalysis:
    """Run the full pipeline in a fixed order and return everything.

    ``progress`` and ``cancelled`` are plain callables, so a Qt worker can
    drive this without :mod:`rcta.core` ever importing Qt.
    """
    params = params or AnalysisParams()

    if params.integration_time_s is not None:
        track = track.with_mask(integration_time_s=params.integration_time_s)

    track, quality = assess_with_report(track, params.quality)

    live = track.effective_integration_time_s
    # Only scales that fit the track are scanned, so only they may size the
    # background: --scales 100000 on a 348-sample walk once widened it to
    # 5 000 000 s for a window that was never tested.
    fitting = [s for s in params.scan.scales if s <= track.n]
    max_scan_window_s = max(fitting or [1]) * live

    # Scale separation depends on the logging cadence, which the caller cannot
    # know in advance. A device sampling every 10 s makes the default 64-sample
    # scale 640 s wide, so the stock 900 s background window would warn about
    # the stock settings. Widen it unless the user picked a window themselves.
    background_params = params.background
    required = background_params.scale_separation * max_scan_window_s
    left_at_default = background_params.window_s == BackgroundParams().window_s
    widened_from = None
    if left_at_default and background_params.window_s < required:
        widened_from = background_params.window_s
        background_params = replace(background_params, window_s=required)

    background = estimate_background(track, background_params, quality=quality,
                                     max_scan_window_s=max_scan_window_s)

    anomalies = scan(track, background, params.scan, quality=quality,
                     progress=progress, cancelled=cancelled)
    hotspots = find_hotspots(track, anomalies, background,
                             cluster_radius_m=params.cluster_radius_m)

    # The dose integral breaks at the same dropouts as the map and the legs,
    # so its gap rule is QualityParams.gap_threshold_s and nothing else.
    dose = accumulated_dose(track, quality=quality, unit=params.unit)
    summary = summarise(track, quality=quality)

    warnings = list(track.meta.warnings) + list(quality.warnings) \
        + list(background.warnings)
    # The widened window is used everywhere, but the GUI's Window box and the
    # command line still show the default the user left alone; without this
    # line the only record of the window actually applied was in the JSON.
    if widened_from is not None:
        warnings.append(
            f"background window widened from {widened_from:.0f} s to "
            f"{background_params.window_s:.0f} s: the widest scan window is "
            f"{max_scan_window_s:.0f} s at this cadence and the background "
            f"must be {background_params.scale_separation:g}x wider; set a "
            "window explicitly to override")
    if not fitting:
        warnings.append(
            f"no scan window fits: every scale "
            f"({', '.join(str(s) for s in sorted(set(params.scan.scales)))} "
            f"samples) exceeds the track's {track.n}, so nothing was tested")
    # Warn only when the time really was inferred: a Track built in code may
    # carry its own, and calling that "inferred" would be false.
    if track.integration_time_s is None:
        warnings.append(
            f"per-sample integration time is {live:.1f} s, "
            f"{track.integration_time_source}; the file does not record it, "
            "and every detection limit scales with it")

    return TrackAnalysis(
        track=track, params=params, quality=quality, background=background,
        anomalies=tuple(anomalies), hotspots=tuple(hotspots), dose=dose,
        summary=summary,
        dose_distribution=bimodality(track.dose_rate[track.valid_mask]),
        speed_dose=correlate_speed_dose(track, n_boot=params.n_boot),
        mda_table=tuple(scan_mda(track, background, params.scan)),
        warnings=tuple(warnings),
    )
