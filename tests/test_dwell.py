"""Stop-and-measure dwells: found where the detector stood still, tested once."""

from __future__ import annotations

import numpy as np
import pytest

from rcta.core.analysis import AnalysisParams, analyze_track
from rcta.core.dwell import DwellParams, find_dwells
from rcta.core.track import Track


def _walk_with_stop(n=600, dt=2.0, stop=(200, 320), source_cps=0.0, seed=1):
    """A straight walk that stands still for ``stop`` samples, optionally
    over a source that only the stationary samples see."""
    rng = np.random.default_rng(seed)
    step = np.full(n, 1.4 * dt)
    step[stop[0]:stop[1]] = 0.0
    north = np.concatenate(([0.0], np.cumsum(step[1:])))
    lat = 30.2 + north / 111_195.0
    lon = np.full(n, -97.66)
    true = np.full(n, 10.0)
    true[stop[0]:stop[1]] += source_cps
    counts = rng.poisson(true * dt) / dt
    return Track(time_s=np.arange(n) * dt, lat=lat, lon=lon,
                 accuracy_m=np.full(n, 3.0), dose_rate=counts * 13.0,
                 count_rate=counts, integration_time_s=dt)


def test_a_stop_is_found_and_a_walk_is_not():
    a = analyze_track(_walk_with_stop(), AnalysisParams(n_boot=10))
    assert len(a.dwells) == 1
    dwell = a.dwells[0]
    assert dwell.start_index == pytest.approx(200, abs=6)
    assert dwell.stop_index == pytest.approx(320, abs=6)
    assert dwell.live_time_s == pytest.approx(dwell.n_samples * 2.0)


def test_a_clean_dwell_detects_nothing_and_quotes_a_better_limit():
    a = analyze_track(_walk_with_stop(), AnalysisParams(n_boot=10))
    dwell = a.dwells[0]
    assert not dwell.detected
    # ~115 samples counted together beat the scan's widest (64) window.
    widest = a.mda_table[-1][2]
    assert dwell.mda_dose_rate < widest


def test_a_weak_source_under_a_dwell_is_detected():
    """1.5 cps over a 10 cps background: lost per sample, plain over 230 s."""
    a = analyze_track(_walk_with_stop(source_cps=1.5), AnalysisParams(n_boot=10))
    assert a.dwells[0].detected
    assert a.dwells[0].significance > 3.0


def test_short_or_moving_stretches_are_not_dwells():
    track = _walk_with_stop(stop=(200, 230))          # 60 s: too short
    a = analyze_track(track, AnalysisParams(n_boot=10))
    assert a.dwells == ()
    assert find_dwells(a.track, a.background, quality=a.quality,
                       params=DwellParams(min_duration_s=30.0))


def test_dwell_params_validate():
    with pytest.raises(ValueError):
        DwellParams(max_speed_m_s=0)
    with pytest.raises(ValueError):
        DwellParams(alpha=1.5)


def test_the_flight_has_no_dwells_and_the_walk_has_one(sample, walk_path):
    from rcta.core.io_radiacode import load_track
    assert analyze_track(sample, AnalysisParams(n_boot=10)).dwells == ()
    walk = analyze_track(load_track(walk_path), AnalysisParams(n_boot=10))
    assert len(walk.dwells) == 1


def test_dwells_reach_the_summary_and_the_report():
    import json

    from rcta.core.io_export import report_to_json
    a = analyze_track(_walk_with_stop(source_cps=1.5), AnalysisParams(n_boot=10))
    assert "-- dwells" in a.describe() and "DETECTED" in a.describe()
    report = json.loads(report_to_json(a))
    assert report["dwells"][0]["detected"] is True
    assert report["parameters"]["dwell"]["min_duration_s"] == 120.0
