"""Dose rate and count rate must tell the same story."""

from __future__ import annotations

import numpy as np

from rcta.core.consistency import check_consistency
from rcta.core.track import Track


def _track(dose, counts):
    n = len(dose)
    return Track(time_s=np.arange(n) * 2.0, lat=np.full(n, 30.0),
                 lon=np.linspace(-97.0, -96.99, n), accuracy_m=np.full(n, 5.0),
                 dose_rate=np.asarray(dose, float),
                 count_rate=np.asarray(counts, float))


def test_a_clean_track_has_nothing_to_report(sample, walk_path):
    from rcta.core.io_radiacode import load_track
    for track in (sample, load_track(walk_path)):
        report = check_consistency(track)
        assert report.n_inconsistent == 0 and report.warnings == ()


def test_each_kind_of_disagreement_is_found_and_reported():
    counts = np.full(100, 10.0)
    dose = counts * 13.0
    dose[10], counts[10] = 70.0, 0.0          # dose, no counts
    dose[20] = 0.0                            # counts, no dose
    dose[50] = counts[50] * 13.0 * 6.0        # ratio jump
    report = check_consistency(_track(dose, counts))
    assert report.dose_without_counts == (10,)
    assert report.counts_without_dose == (20,)
    assert report.ratio_jumps == (50,)
    assert len(report.warnings) == 3
    assert all("kept in the analysis" in w for w in report.warnings)


def test_a_slow_drift_in_the_ratio_is_not_a_jump():
    """Climbing a flight, the ratio goes from 13 to 51: a drift, not a fault."""
    counts = np.full(400, 10.0)
    dose = counts * np.linspace(13.0, 51.0, 400)
    assert check_consistency(_track(dose, counts)).ratio_jumps == ()


def test_inconsistent_samples_reach_the_analysis_warnings_and_report():
    import json
    from rcta.core.analysis import AnalysisParams, analyze_track
    from rcta.core.io_export import report_to_json
    counts = np.full(300, 10.0)
    dose = counts * 13.0
    dose[150] = counts[150] * 13.0 * 6.0
    a = analyze_track(_track(dose, counts), AnalysisParams(n_boot=10))
    assert any("dose/count ratio" in w for w in a.warnings)
    assert json.loads(report_to_json(a))["consistency"]["ratio_jumps"] == [150]
