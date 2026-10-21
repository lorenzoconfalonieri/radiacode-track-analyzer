"""The anomaly scan: what it must find, and what it must never find."""

from __future__ import annotations

import numpy as np
import pytest
import synthetic

from rcta.core.anomalies import (ScanParams, bonferroni_k, scan, scan_mda,
                                 significance_trace)
from rcta.core.background import BackgroundParams, estimate_background
from rcta.core.quality import assess_with_report


def _scan(track, params=None, **bg_kw):
    bg = estimate_background(track, BackgroundParams(**bg_kw))
    return scan(track, bg, params or ScanParams()), bg


# ---- configuration -------------------------------------------------------


def test_bonferroni_raises_the_threshold_a_lot():
    """1.64 sigma to 4.57 sigma is the difference between a thousand
    detections and none."""
    assert bonferroni_k(1, 0.05) == pytest.approx(1.6449, abs=1e-3)
    assert bonferroni_k(20748, 0.05) == pytest.approx(4.57, abs=0.02)


def test_invalid_scales_raise():
    with pytest.raises(ValueError, match="scales"):
        ScanParams(scales=())
    with pytest.raises(ValueError, match="scales"):
        ScanParams(scales=(0, 4))


def test_invalid_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        ScanParams(alpha=1.5)


# ---- the false-positive guarantee ----------------------------------------

def test_bonferroni_suppresses_false_positives_on_pure_background():
    """Forty independent clean tracks; a detection in any of them is a bug.

    This is the test that justifies correcting for multiplicity at all. Run
    the same tracks without the correction and the assertion below inverts.
    """
    detections = 0
    for seed in range(40):
        track, _ = synthetic.make(n=1500, background_cps=18.0, shuffle=False,
                                  seed=seed)
        found, _ = _scan(track)
        detections += len([a for a in found if a.is_significant])
    assert detections == 0, f"{detections} false detections over 40 clean tracks"


def test_without_bonferroni_the_same_clean_tracks_light_up():
    """The counterfactual, so the correction's value is measured not asserted."""
    uncorrected = 0
    for seed in range(10):
        track, _ = synthetic.make(n=1500, background_cps=18.0, shuffle=False,
                                  seed=seed)
        found, _ = _scan(track, ScanParams(bonferroni=False))
        uncorrected += len([a for a in found if a.is_significant])
    assert uncorrected > 20, uncorrected


# ---- recovery ------------------------------------------------------------


def test_a_strong_source_is_recovered_at_the_right_place():
    track, truth = synthetic.make(
        n=1200, background_cps=18.0, shuffle=False, seed=11,
        sources=[(600, 110.0, 6.0)])
    found, _ = _scan(track)
    significant = [a for a in found if a.is_significant]
    assert significant, "a 6x background source must be found"

    best = max(significant, key=lambda a: a.significance)
    source = truth["sources"][0]
    assert abs(best.lat - source["lat"]) < 0.002
    assert best.start_index <= 600 < best.stop_index


def test_recovered_net_counts_are_close_to_the_injected_truth():
    track, truth = synthetic.make(
        n=1200, background_cps=18.0, shuffle=False, seed=12,
        sources=[(600, 110.0, 6.0)])
    found, _ = _scan(track)
    best = max((a for a in found if a.is_significant),
               key=lambda a: a.net_counts)
    injected = truth["sources"][0]["net_counts"]
    # The matched window captures most but not all of a Gaussian's tails.
    assert 0.5 * injected < best.net_counts < 1.3 * injected


def test_a_source_below_the_detection_limit_is_not_recovered():
    """The limit has to bind in both directions, or it is not a limit."""
    track, _ = synthetic.make(n=1200, background_cps=18.0, shuffle=False,
                              seed=13, sources=[(600, 1.2, 4.0)])
    found, _ = _scan(track)
    assert not [a for a in found if a.is_significant]


def test_a_wide_weak_source_needs_the_wide_scales():
    """Demonstrates why the scan is multi-scale rather than single-width."""
    track, _ = synthetic.make(n=1500, background_cps=18.0, shuffle=False,
                              seed=14, sources=[(750, 4.0, 40.0)])
    narrow, _ = _scan(track, ScanParams(scales=(1, 2)))
    wide, _ = _scan(track, ScanParams(scales=(1, 2, 4, 8, 16, 32, 64, 128)))
    assert not [a for a in narrow if a.is_significant]
    assert [a for a in wide if a.is_significant]


# ---- trends --------------------------------------------------------------


def test_a_smooth_ramp_is_flagged_as_a_trend_not_called_a_hotspot():
    track, _ = synthetic.make(n=1200, drift=(5.0, 60.0), shuffle=False, seed=15)
    found, _ = _scan(track, method="global")
    assert found, "a global background should raise candidates here"
    assert all("trend" in a.flags for a in found)
    assert not [a for a in found if a.is_significant]


def test_a_trend_flagged_feature_is_still_reported():
    """Hiding it would be as dishonest as calling it a source."""
    track, _ = synthetic.make(n=1200, drift=(5.0, 60.0), shuffle=False, seed=16)
    found, _ = _scan(track, method="global")
    assert len(found) > 0
    assert all(not a.is_significant for a in found)


def test_a_source_on_a_flat_baseline_is_not_flagged_as_a_trend():
    track, _ = synthetic.make(n=1200, background_cps=18.0, shuffle=False,
                              seed=17, sources=[(600, 110.0, 5.0)])
    found, _ = _scan(track)
    best = max(found, key=lambda a: a.significance)
    assert "trend" not in best.flags


# ---- flags and merging ---------------------------------------------------


def test_overlapping_windows_are_merged_to_the_strongest():
    track, _ = synthetic.make(n=1200, background_cps=18.0, shuffle=False,
                              seed=18, sources=[(600, 110.0, 8.0)])
    merged, _ = _scan(track, ScanParams(merge_overlapping=True))
    raw, _ = _scan(track, ScanParams(merge_overlapping=False))
    assert len(merged) < len(raw)


def test_poor_gps_is_flagged_not_excluded():
    track, _ = synthetic.make(n=1200, background_cps=18.0, shuffle=False,
                              seed=19, sources=[(600, 110.0, 8.0)])
    accuracy = np.array(track.accuracy_m)
    accuracy[560:660] = 400.0
    track = track.with_mask(position=accuracy <= 100.0)
    found, _ = _scan(track)
    hits = [a for a in found if a.start_index <= 600 < a.stop_index]
    assert hits, "the source is still detected"
    assert any("poor-gps" in a.flags for a in hits)


def test_single_sample_detections_are_flagged_as_such():
    track, _ = synthetic.make(n=800, background_cps=18.0, shuffle=False,
                              seed=20, sources=[(400, 400.0, 0.4)])
    found, _ = _scan(track, ScanParams(scales=(1,)))
    assert found and all("single-sample" in a.flags for a in found)


# ---- traces and MDA ------------------------------------------------------


def test_significance_trace_aligns_with_the_track():
    track, _ = synthetic.make(n=600, shuffle=False, seed=21)
    bg = estimate_background(track)
    trace = significance_trace(track, bg, scale=8)
    assert trace.size == track.n


def test_significance_trace_peaks_at_the_source():
    track, _ = synthetic.make(n=1200, background_cps=18.0, shuffle=False,
                              seed=22, sources=[(600, 110.0, 6.0)])
    bg = estimate_background(track)
    trace = significance_trace(track, bg, scale=8)
    assert abs(int(np.argmax(trace)) - 600) < 20


def test_mda_improves_with_window_width():
    """The whole justification for scanning at several widths."""
    track, _ = synthetic.make(n=1200, background_cps=18.0, shuffle=False, seed=23)
    table = scan_mda(track, estimate_background(track))
    widths = [row[0] for row in table]
    mdas = [row[2] for row in table]
    assert widths == sorted(widths)
    assert all(b < a for a, b in zip(mdas, mdas[1:])), mdas


# ---- the real file -------------------------------------------------------


def test_sample_yields_no_unflagged_detections(sample):
    """The headline result for the bundled track is a null result.

    Everything above threshold is the climb or the descent, and every one of
    them comes back flagged as the trend it is.
    """
    track, quality = assess_with_report(sample)
    bg = estimate_background(track, quality=quality)
    found = scan(track, bg, ScanParams(), quality=quality)
    assert not [a for a in found if a.is_significant]
    assert found, "the climb and descent should still appear as candidates"
    assert all("trend" in a.flags for a in found)


def test_sample_mda_table_is_the_result(sample):
    """When nothing is detected, this table is what the report says instead."""
    track, quality = assess_with_report(sample)
    table = scan_mda(track, estimate_background(track, quality=quality))
    assert len(table) == 7
    assert table[0][2] > table[-1][2] > 0.0


def test_scan_can_be_cancelled(sample):
    """The GUI must be able to abandon a scan without this module knowing Qt."""
    track, quality = assess_with_report(sample)
    bg = estimate_background(track, quality=quality)
    assert scan(track, bg, ScanParams(), cancelled=lambda: True) == []


def test_scan_reports_progress(sample):
    seen = []
    track, quality = assess_with_report(sample)
    bg = estimate_background(track, quality=quality)
    scan(track, bg, ScanParams(), progress=seen.append)
    assert seen and seen[-1] == 1.0
    assert seen == sorted(seen)


def test_no_detection_limit_is_quoted_for_a_scale_the_scan_never_tested():
    """With every 3rd sample masked, no window of 4 or more is clean."""
    import synthetic
    from rcta.core.anomalies import ScanParams, scan_mda
    from rcta.core.background import estimate_background
    track, _ = synthetic.make(n=300, shuffle=False)
    valid = np.ones(track.n, dtype=bool)
    valid[::3] = False
    track = track.with_mask(valid=valid)
    background = estimate_background(track)
    table = scan_mda(track, background, ScanParams(scales=(1, 2, 4, 8)))
    assert [width for width, _, _ in table] == [1, 2]


def test_an_excess_where_the_gps_fix_collapses_is_flagged_as_a_transition():
    """Walking indoors changes shielding and fix together; say so."""
    import synthetic
    from rcta.core.analysis import AnalysisParams, analyze_track
    from rcta.core.track import Track
    track, _ = synthetic.make(n=600, shuffle=False, sources=[(300, 40.0, 6)])
    accuracy = np.full(track.n, 5.0)
    accuracy[305:] = 60.0                       # indoors from here on
    track = Track(time_s=track.time_s, lat=track.lat, lon=track.lon,
                  accuracy_m=accuracy, dose_rate=track.dose_rate,
                  count_rate=track.count_rate,
                  integration_time_s=track.integration_time_s)
    a = analyze_track(track, AnalysisParams(n_boot=10))
    assert a.significant, "the injected source must still be detected"
    assert "gps-transition" in a.significant[0].flags


def test_steady_gps_raises_no_transition_flag():
    import synthetic
    from rcta.core.analysis import AnalysisParams, analyze_track
    track, _ = synthetic.make(n=600, shuffle=False, sources=[(300, 40.0, 6)])
    a = analyze_track(track, AnalysisParams(n_boot=10))
    assert a.significant
    assert all("gps-transition" not in x.flags for x in a.anomalies)
