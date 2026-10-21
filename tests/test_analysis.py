"""The shared facade: one pipeline, one set of parameters, one result."""

from __future__ import annotations

import numpy as np
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
    track, _ = synthetic.make(n=400, shuffle=False, from_file=True)
    a = analyze_track(track, AnalysisParams(n_boot=50))
    assert any("integration time is" in w and "inferred from median spacing" in w
               for w in a.warnings)


def test_a_track_that_carries_its_integration_time_is_not_warned_about():
    track, _ = synthetic.make(n=400, shuffle=False)
    assert track.integration_time_s is not None
    a = analyze_track(track, AnalysisParams(n_boot=50))
    assert not any("integration time is" in w for w in a.warnings)


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
                              gaps=[(300, 600.0)], seed=2, from_file=True)
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


# ---- masked samples ------------------------------------------------------


@pytest.fixture(scope="module")
def corrupted(sample_path, tmp_path_factory):
    """The real walk with every 97th marker broken, and the same walk without them.

    Masking must change nothing a reading feeds: the masked analysis and an
    analysis of the file with those markers deleted outright have to agree on
    every statistic computed from readings.
    """
    import json

    from rcta.core.io_radiacode import load_track

    raw = json.loads(sample_path.read_text())
    broken = dict(raw, markers=[
        {k: v for k, v in m.items() if k != "doseRate"} if i % 97 == 0 else m
        for i, m in enumerate(raw["markers"])])
    dropped = dict(raw, markers=[m for i, m in enumerate(raw["markers"])
                                 if i % 97 != 0])
    folder = tmp_path_factory.mktemp("masked")
    (folder / "broken.rctrk").write_text(json.dumps(broken))
    (folder / "dropped.rctrk").write_text(json.dumps(dropped))
    return (load_track(folder / "broken.rctrk"),
            load_track(folder / "dropped.rctrk"))


def test_masked_samples_do_not_move_the_statistics(corrupted):
    broken, dropped = corrupted
    assert int((~broken.valid_mask).sum()) == 31
    params = AnalysisParams(n_boot=50, integration_time_s=10.0)
    a = analyze_track(broken, params)
    b = analyze_track(dropped, params)

    for name in ("dose_min", "dose_max", "dose_mean", "dose_median",
                 "dose_p95", "count_mean", "count_median"):
        assert getattr(a.summary, name) == pytest.approx(
            getattr(b.summary, name)), name
    assert a.dose_distribution.threshold == pytest.approx(
        b.dose_distribution.threshold)
    assert a.dose.max_dose_rate == pytest.approx(b.dose.max_dose_rate)
    assert a.speed_dose.spearman_rho == pytest.approx(b.speed_dose.spearman_rho)


def test_masked_samples_leave_no_nan_in_the_results(corrupted):
    import json

    from rcta.core.io_export import report_to_json, track_to_csv

    broken, _ = corrupted
    a = analyze_track(broken, AnalysisParams(n_boot=50))
    assert np.isfinite(a.background.count_rate).all()
    assert np.isfinite(a.dose.total_nsv) and np.isfinite(a.dose.naive_total_nsv)
    assert all(np.isfinite(mda) for _, _, mda in a.mda_table)
    assert "nan" not in a.describe().lower()
    json.loads(report_to_json(a))
    csv = track_to_csv(a.track, quality=a.quality, background=a.background)
    assert "nan" not in csv.lower()


def test_a_masked_interval_is_unobserved_time(corrupted):
    """No reading, no dose: the covered time shrinks rather than bridging it."""
    broken, _ = corrupted
    clean = analyze_track(broken.with_mask(valid=np.ones(broken.n, bool)),
                          AnalysisParams(n_boot=50))
    masked = analyze_track(broken, AnalysisParams(n_boot=50))
    assert masked.dose.covered_time_s < clean.quality.covered_time_s


@pytest.mark.parametrize("dt_s", [2.0, 5.0, 10.0])
def test_the_inferred_integration_time_recovers_the_cadence(dt_s):
    """With jitter and a long dropout, the median spacing is still the cadence,
    so a file-like track gets the same detection limits as one that states
    its integration time."""
    kwargs = dict(n=500, dt_s=dt_s, dt_jitter=0.1 * dt_s, gaps=[(250, 900.0)],
                  shuffle=False, seed=3)
    inferred, _ = synthetic.make(from_file=True, **kwargs)
    stated, _ = synthetic.make(**kwargs)
    a = analyze_track(inferred, AnalysisParams(n_boot=20))
    b = analyze_track(stated, AnalysisParams(n_boot=20))
    assert a.track.effective_integration_time_s == pytest.approx(dt_s, rel=0.05)
    for (wa, _, mda_a), (wb, _, mda_b) in zip(a.mda_table, b.mda_table):
        assert wa == wb
        assert mda_a == pytest.approx(mda_b, rel=0.05)


def test_a_widened_background_window_is_reported():
    """The window applied differed from the one shown, and only the JSON said."""
    track, _ = synthetic.make(n=400, dt_s=10.0, shuffle=False)
    a = analyze_track(track, AnalysisParams(n_boot=20))
    assert a.background.params.window_s > 900.0
    assert any("background window widened from 900 s" in w for w in a.warnings)


def test_an_explicit_window_is_not_reported_as_widened():
    from rcta.core.background import BackgroundParams
    track, _ = synthetic.make(n=400, dt_s=10.0, shuffle=False)
    a = analyze_track(track, AnalysisParams(
        background=BackgroundParams(window_s=1200.0), n_boot=20))
    assert not any("widened" in w for w in a.warnings)


def test_scan_scales_longer_than_the_track_are_not_a_null_result():
    """Nothing was tested, so 'no anomalies above L_C' would be a claim."""
    from rcta.core.anomalies import ScanParams
    track, _ = synthetic.make(n=200, dt_s=10.0, shuffle=False, from_file=True)
    a = analyze_track(track, AnalysisParams(scan=ScanParams(scales=(500,)),
                                            n_boot=20))
    text = a.describe()
    assert "nothing was tested" in text
    assert "no anomalies above L_C" not in text
    assert any("no scan window fits" in w for w in a.warnings)
    assert a.nothing_tested
    assert "nothing tested" in a.status_line()
    assert "no anomalies" not in a.status_line()
    # The unusable scale no longer drives the background window.
    assert a.background.params.window_s == 900.0


# ---- figures quoted in the paper and README -------------------------------


@pytest.fixture(scope="module")
def walk_analysis(walk_path):
    from rcta.core.io_radiacode import load_track
    return analyze_track(load_track(walk_path))


def test_the_walk_figures_quoted_in_the_paper_still_hold(walk_analysis):
    """paper.tex Fig. 1 and section 3 quote these numbers, and the docs have
    drifted from the program before (1160.02 vs 1159.97). This test is the
    tripwire: if it fails, fix the paper and the README in the same commit."""
    a = walk_analysis
    assert a.track.n == 348
    assert round(a.summary.distance_m / 1000.0, 2) == 3.15
    assert round(a.background.median_count_rate, 2) == 5.31
    assert round(float(np.median(a.background.dose_rate)), 1) == 70.1
    sigmas = sorted((round(h.significance, 1) for h in a.hotspots), reverse=True)
    assert sigmas[:2] == [7.4, 4.6]
    assert [h.evidence for h in a.hotspots[:2]] == ["strong", "possible"]
    assert round(a.dose.total_nsv, 1) == 80.9
    assert round(a.dose.naive_total_nsv, 1) == 153.8
    assert round(a.dose.covered_time_s) == 3856
    assert round(a.speed_dose.n_effective) == 168
    assert round(a.speed_dose.p_naive, 3) == 0.084
    assert round(a.speed_dose.p_effective, 3) == 0.232
    # Section 3: "the walk's single 80 s dwell reaches 12 nSv/h".
    assert len(a.dwells) == 1
    assert round(a.dwells[0].live_time_s) == 80
    assert round(a.dwells[0].mda_dose_rate) == 12


def test_the_walk_legs_and_dropouts_add_up_to_its_distance(walk_analysis):
    """Smoothing used to freeze each leg's last samples in place: the legs
    summed to 2.64 km and the dropouts to 0.09 km of a 2.99 km walk, the
    connectors across the gaps having picked up the missing path."""
    a = walk_analysis
    legs = a.quality.leg_index
    same = legs[1:] == legs[:-1]
    in_legs = float(a.track.segment_length_m[same].sum())
    assert in_legs + a.quality.gap_distance_m == pytest.approx(
        a.summary.distance_m, rel=0.01)


def test_the_flight_detection_limits_quoted_in_the_paper_still_hold(sample):
    """paper.tex section 3: 1160 nSv/h in one 2 s sample, 133 over 128 s."""
    table = {width: mda for width, _, mda in
             analyze_track(sample, AnalysisParams(n_boot=20)).mda_table}
    assert round(table[1], 2) == 1159.97
    assert round(table[64], 2) == 132.56
