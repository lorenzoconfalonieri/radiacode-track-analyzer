"""The shared facade: one pipeline, one set of parameters, one result."""

from __future__ import annotations

import pytest
import synthetic

from rcta.core.analysis import AnalysisParams, analyze_track
from rcta.core.anomalies import ScanParams
from rcta.core.background import BackgroundParams


def test_the_facade_produces_every_section():
    track, _ = synthetic.make(n=600, shuffle=True, seed=1)
    a = analyze_track(track, AnalysisParams(n_boot=50))
    assert a.quality.n == track.n
    assert a.background.count_rate.size == track.n
    assert a.summary.n == track.n
    assert a.mda_table
    assert a.dose.span_time_s > 0


def test_parameters_are_carried_on_the_result():
    """A detection limit without its assumptions is not reproducible."""
    params = AnalysisParams(integration_time_s=1.5, unit="uSv/h", n_boot=50)
    track, _ = synthetic.make(n=400, shuffle=False)
    a = analyze_track(track, params)
    assert a.params is params
    assert a.track.effective_integration_time_s == 1.5


def test_an_inferred_integration_time_is_warned_about():
    track, _ = synthetic.make(n=400, shuffle=False)
    a = analyze_track(track, AnalysisParams(n_boot=50))
    assert any("integration time was inferred" in w for w in a.warnings)


def test_a_supplied_integration_time_is_not_warned_about():
    track, _ = synthetic.make(n=400, shuffle=False)
    a = analyze_track(track, AnalysisParams(integration_time_s=2.0, n_boot=50))
    assert not any("integration time was inferred" in w for w in a.warnings)


def test_integration_time_scales_the_detection_limits():
    """The single largest systematic in the whole program, made visible."""
    track, _ = synthetic.make(n=600, shuffle=False)
    short = analyze_track(track, AnalysisParams(integration_time_s=1.0, n_boot=50))
    long = analyze_track(track, AnalysisParams(integration_time_s=4.0, n_boot=50))
    assert short.mda_table[0][2] > long.mda_table[0][2]


def test_warnings_are_collected_from_every_stage():
    """Parse, quality and background warnings all reach the one result object.

    Note the shuffled synthetic track raises no sort warning: Track sorts
    silently as an invariant, and it is the *loader* that counts inversions and
    reports them, because only the loader knows what the file claimed. The
    real-file case is covered in test_io.
    """
    track, _ = synthetic.make(n=600, shuffle=True, accuracy_outliers=50,
                              gaps=[(300, 600.0)], seed=2)
    a = analyze_track(track, AnalysisParams(n_boot=50))
    assert any("accuracy" in w for w in a.warnings)
    assert any("dropout" in w for w in a.warnings)
    assert any("integration time" in w for w in a.warnings)


def test_loader_warnings_reach_the_analysis(sample):
    a = analyze_track(sample, AnalysisParams(n_boot=50))
    assert any("not in time order" in w for w in a.warnings)


def test_a_clean_track_finds_nothing_and_says_so():
    track, _ = synthetic.make(n=1200, background_cps=18.0, shuffle=False, seed=3)
    a = analyze_track(track, AnalysisParams(n_boot=50))
    assert a.found_nothing
    assert "no anomalies above L_C" in a.describe()
    assert "minimum detectable dose rate" in a.describe()


def test_a_source_is_found_and_becomes_a_hotspot():
    track, _ = synthetic.make(n=1200, background_cps=18.0, shuffle=False, seed=4,
                              sources=[(600, 110.0, 6.0)])
    a = analyze_track(track, AnalysisParams(n_boot=50))
    assert not a.found_nothing
    assert a.hotspots


def test_status_line_is_one_line_and_carries_the_verdict():
    track, _ = synthetic.make(n=600, shuffle=False)
    line = analyze_track(track, AnalysisParams(n_boot=50)).status_line()
    assert "\n" not in line
    assert "samples" in line and "km" in line and "L_C" in line


def test_the_facade_can_be_cancelled():
    track, _ = synthetic.make(n=800, shuffle=False)
    a = analyze_track(track, AnalysisParams(n_boot=50), cancelled=lambda: True)
    assert a.anomalies == ()


def test_the_background_window_is_checked_against_the_scan_width():
    """The facade is what wires scale separation up; a bare call cannot."""
    track, _ = synthetic.make(n=800, shuffle=False)
    a = analyze_track(track, AnalysisParams(
        background=BackgroundParams(window_s=60.0),
        scan=ScanParams(scales=(1, 2, 4, 8, 16, 32, 64)), n_boot=50))
    assert any("widest scan window" in w for w in a.background.warnings)


# ---- the real file -------------------------------------------------------


def test_sample_analysis_is_a_null_result(sample):
    a = analyze_track(sample, AnalysisParams(n_boot=100))
    assert a.found_nothing
    assert a.hotspots == ()
    assert a.quality.n_legs == 14
    assert a.dose.total_nsv == pytest.approx(1234.50, rel=1e-5)


def test_sample_describe_leads_with_the_limits_not_an_empty_table(sample):
    a = analyze_track(sample, AnalysisParams(n_boot=100))
    text = a.describe()
    assert "no anomalies above L_C" in text
    assert "minimum detectable dose rate" in text
    # The widest window is the most sensitive, so it must be the smallest MDA.
    assert a.mda_table[-1][2] < a.mda_table[0][2]
    assert f"{a.mda_table[-1][2]:.2f}" in text


def test_sample_status_line(sample):
    line = analyze_track(sample, AnalysisParams(n_boot=100)).status_line()
    assert "2964 samples" in line
    assert "1792.21 km" in line
    assert "14 legs" in line
