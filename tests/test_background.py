"""Background estimation, and the inversion that justifies the default."""

from __future__ import annotations

import numpy as np
import pytest
import synthetic

from rcta.core.anomalies import ScanParams, scan
from rcta.core.background import BackgroundParams, estimate_background
from rcta.core.quality import assess_with_report


# ---- parameter validation ------------------------------------------------


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown background method"):
        BackgroundParams(method="magic")


@pytest.mark.parametrize("f", [0.5, 0.9, -0.1])
def test_trim_fraction_must_be_valid(f):
    with pytest.raises(ValueError, match="trim_fraction"):
        BackgroundParams(trim_fraction=f)


def test_reference_method_requires_a_span():
    with pytest.raises(ValueError, match="reference_span"):
        BackgroundParams(method="reference")


# ---- behaviour of each method --------------------------------------------


def test_rolling_follows_a_ramp():
    """The defining capability: a background that changes with the world."""
    track, _ = synthetic.make(n=900, drift=(5.0, 60.0), shuffle=False, seed=1)
    bg = estimate_background(track, BackgroundParams(method="rolling",
                                                     window_s=200.0))
    assert bg.count_rate[50] < bg.count_rate[-50]
    assert bg.count_rate[-50] / max(bg.count_rate[50], 1e-9) > 3.0


def test_global_does_not_follow_a_ramp():
    track, _ = synthetic.make(n=900, drift=(5.0, 60.0), shuffle=False, seed=1)
    bg = estimate_background(track, BackgroundParams(method="global"))
    assert bg.count_rate.min() == pytest.approx(bg.count_rate.max())


def test_global_warns_that_it_reports_trends_as_detections():
    track, _ = synthetic.make(n=200, shuffle=False)
    bg = estimate_background(track, BackgroundParams(method="global"))
    assert any("gradient" in w for w in bg.warnings)


def test_reference_uses_only_the_marked_span():
    track, _ = synthetic.make(n=600, background_cps=18.0, shuffle=False, seed=2)
    t0 = float(track.time_s[10])
    t1 = float(track.time_s[100])
    bg = estimate_background(
        track, BackgroundParams(method="reference", reference_span=(t0, t1)))
    assert bg.count_rate.min() == pytest.approx(bg.count_rate.max())
    assert bg.count_rate[0] == pytest.approx(18.0, rel=0.15)


def test_reference_with_too_few_samples_raises():
    track, _ = synthetic.make(n=200, shuffle=False)
    with pytest.raises(ValueError, match="reference span contains"):
        estimate_background(track, BackgroundParams(
            method="reference", reference_span=(0.0, 1.0)))


def test_flank_excludes_the_guard_band():
    """A source must not be allowed to estimate its own background."""
    track, _ = synthetic.make(n=600, background_cps=18.0, poisson=False,
                              sources=[(300, 40.0, 3.0)], shuffle=False)
    guarded = estimate_background(track, BackgroundParams(
        method="flank", window_s=200.0, guard_s=40.0))
    unguarded = estimate_background(track, BackgroundParams(
        method="flank", window_s=200.0, guard_s=0.0))
    assert guarded.count_rate[300] < unguarded.count_rate[300]


def test_trimming_resists_a_hotspot_inside_the_window():
    """A source inside the window must not raise its own background."""
    track, _ = synthetic.make(n=600, background_cps=18.0, poisson=False,
                              sources=[(300, 60.0, 5.0)], shuffle=False)
    trimmed = estimate_background(track, BackgroundParams(trim_fraction=0.25,
                                                          window_s=300.0))
    plain = estimate_background(track, BackgroundParams(trim_fraction=0.0,
                                                        window_s=300.0))
    assert trimmed.count_rate[300] == pytest.approx(18.0, abs=0.1)
    assert plain.count_rate[300] > 22.0


def test_trimming_has_a_breakdown_point_and_it_is_documented():
    """Beyond a quarter of the window, the source wins. Stated as a test so
    the guarantee is not mistaken for an unconditional one: a stop-and-measure
    over a source needs the flank or reference method, not rolling."""
    track, _ = synthetic.make(n=600, background_cps=18.0, poisson=False,
                              sources=[(300, 60.0, 40.0)], shuffle=False)
    bg = estimate_background(track, BackgroundParams(trim_fraction=0.25,
                                                     window_s=300.0))
    assert bg.count_rate[300] > 40.0, "the background has followed the source"


def test_the_background_estimator_is_unbiased_on_clean_data():
    """The bug this estimator replaced: a 0.25 quantile sits 2 cps low on an
    18 cps Poisson background, which is 5.3 sigma of pure artefact over a
    64-sample window. Anything biased more than a tenth of a count per sample
    will manufacture detections, so the bias is pinned here directly."""
    track, _ = synthetic.make(n=3000, background_cps=18.0, shuffle=False, seed=99)
    bg = estimate_background(track, BackgroundParams(window_s=900.0))
    assert abs(float(np.median(bg.count_rate)) - 18.0) < 0.25


def test_dose_units_track_the_count_rate_estimate():
    track, truth = synthetic.make(n=300, shuffle=False)
    bg = estimate_background(track)
    assert np.allclose(bg.dose_rate, bg.count_rate * track.dose_per_count)


# ---- scale separation ----------------------------------------------------


def test_a_background_window_close_to_the_scan_window_warns():
    track, _ = synthetic.make(n=600, shuffle=False)
    bg = estimate_background(track, BackgroundParams(window_s=100.0),
                             max_scan_window_s=90.0)
    assert any("scale" in w or "comparing the signal against itself" in w
               for w in bg.warnings)


def test_the_default_configuration_does_not_warn_about_itself():
    """A stock run must not lecture the user about the stock settings."""
    track, _ = synthetic.make(n=600, dt_s=2.0, shuffle=False)
    bg = estimate_background(track, BackgroundParams(),
                             max_scan_window_s=64 * 2.0)
    assert not any("scale separation" in w or "widest scan window" in w
                   for w in bg.warnings)


# ---- the inversion that defines the default ------------------------------


def test_global_background_turns_a_ramp_into_a_huge_false_detection():
    """The central failure mode, on synthetic data with a known answer.

    A smooth environmental ramp and no source at all. A global background
    calls the high end an enormous anomaly; a rolling one finds nothing that
    is not flagged as the trend it is.
    """
    track, _ = synthetic.make(n=1200, drift=(5.0, 60.0), shuffle=False, seed=3)
    params = ScanParams()

    glob = scan(track, estimate_background(
        track, BackgroundParams(method="global")), params)
    roll = scan(track, estimate_background(
        track, BackgroundParams(method="rolling", window_s=300.0)), params)

    assert max((a.significance for a in glob), default=0.0) > 10.0
    assert not [a for a in roll if a.is_significant]


def test_sample_finds_nothing_under_either_estimator(sample):
    """On the real file both estimators agree: there is nothing here.

    Note what this test does *not* claim. A global background is not
    catastrophic on this particular track, because the trimmed global estimate
    lands near the cruise-level population that makes up most of the samples,
    so it under-reports rather than over-reports. That is a property of this
    file, not a general guarantee, which is why the estimator comparison is
    made on synthetic data above where the truth is constructed.
    """
    track, quality = assess_with_report(sample)
    params = ScanParams()
    for method in ("global", "rolling", "flank"):
        bg = estimate_background(track, BackgroundParams(method=method),
                                 quality=quality)
        found = scan(track, bg, params, quality=quality)
        assert not [a for a in found if a.is_significant], method


def test_sample_rolling_background_tracks_the_climb(sample):
    track, quality = assess_with_report(sample)
    bg = estimate_background(track, quality=quality)
    assert bg.count_rate.min() < 5.0, "ground level"
    assert bg.count_rate.max() > 15.0, "cruise level"


def test_sample_background_warns_about_dropouts(sample):
    track, quality = assess_with_report(sample)
    bg = estimate_background(track, quality=quality)
    assert any("dropout" in w for w in bg.warnings)
