"""GPS quality: what gets masked, what gets split, and what must not."""

from __future__ import annotations

import numpy as np
import pytest

from rcta.core.quality import QualityParams, assess_with_report, gap_segments, polyline
from rcta.core.track import Track


def _track(n=50, dt=2.0, accuracy=8.0, speed_m_s=1.4):
    t = np.arange(n) * dt
    lat = 30.0 + np.arange(n) * speed_m_s * dt / 111195.0
    return Track(time_s=t, lat=lat, lon=np.full(n, -97.0),
                 accuracy_m=np.full(n, accuracy),
                 dose_rate=np.full(n, 90.0), count_rate=np.full(n, 18.0))


# ---- masking -------------------------------------------------------------


def test_a_poor_fix_loses_its_position_but_keeps_its_reading():
    """The independence of the two masks, stated as a test."""
    t = _track(n=10)
    acc = t.accuracy_m.copy()
    acc[3] = 300.0
    t = Track(time_s=t.time_s, lat=t.lat, lon=t.lon, accuracy_m=acc,
              dose_rate=t.dose_rate, count_rate=t.count_rate)
    out, report = assess_with_report(t)
    assert out.position_mask[3] is np.False_ or not out.position_mask[3]
    assert out.valid_mask[3], "the count rate at a bad fix is still a measurement"
    assert report.n_poor_accuracy == 1


def test_nothing_is_ever_dropped():
    t = _track(n=10)
    acc = t.accuracy_m.copy()
    acc[:5] = 500.0
    t = Track(time_s=t.time_s, lat=t.lat, lon=t.lon, accuracy_m=acc,
              dose_rate=t.dose_rate, count_rate=t.count_rate)
    out, report = assess_with_report(t)
    assert out.n == t.n == report.n


# ---- legs ----------------------------------------------------------------


def test_a_dropout_starts_a_new_leg():
    t = _track(n=10)
    time_s = t.time_s.copy()
    time_s[5:] += 600.0
    t = Track(time_s=time_s, lat=t.lat, lon=t.lon, accuracy_m=t.accuracy_m,
              dose_rate=t.dose_rate, count_rate=t.count_rate)
    out, report = assess_with_report(t)
    assert report.n_gaps == 1
    assert report.n_legs == 2
    assert out.leg_index.tolist() == [0] * 5 + [1] * 5


def test_a_slow_logging_cadence_is_not_a_track_full_of_dropouts():
    """A deliberate 60 s cadence must not read as 50 consecutive gaps."""
    t = _track(n=50, dt=60.0)
    _, report = assess_with_report(t)
    assert report.n_gaps == 0
    assert report.n_legs == 1


def test_coverage_is_one_when_there_are_no_gaps():
    _, report = assess_with_report(_track())
    assert report.coverage == pytest.approx(1.0)


# ---- speed ---------------------------------------------------------------


def test_a_teleport_on_a_walking_survey_is_flagged():
    t = _track(n=40, speed_m_s=1.4)
    lat = t.lat.copy()
    lat[20:] += 3.0            # ~333 km in one 2 s step
    t = Track(time_s=t.time_s, lat=lat, lon=t.lon, accuracy_m=t.accuracy_m,
              dose_rate=t.dose_rate, count_rate=t.count_rate)
    out, report = assess_with_report(t)
    assert report.n_speed_outliers == 1
    assert not out.position_mask[20]
    assert out.position_mask[19], "the departure fix was fine"


def test_a_uniformly_fast_platform_is_not_flagged_wholesale():
    """A 250 m/s aircraft is fast, not broken. The robust relative bound is
    what keeps a consistently quick platform out of the outlier list."""
    t = _track(n=60, speed_m_s=250.0)
    _, report = assess_with_report(t)
    assert report.n_speed_outliers == 0


def test_a_walking_speed_glitch_is_caught():
    """The case the old conjunction rule missed entirely.

    A 100 m jump in 2 s is 50 m/s: obvious nonsense for someone on foot, but
    an eighth of the 400 m/s hard ceiling, so requiring *both* tests let it
    through. Walking is the normal use of this tool, so this is the case that
    matters most.
    """
    t = _track(n=60, speed_m_s=1.4)
    lat = t.lat.copy()
    lat[30:] += 100.0 / 111195.0
    t = Track(time_s=t.time_s, lat=lat, lon=t.lon, accuracy_m=t.accuracy_m,
              dose_rate=t.dose_rate, count_rate=t.count_rate)
    _, report = assess_with_report(t)
    assert report.n_speed_outliers == 1


def test_a_glitch_is_detected_on_raw_fixes_not_smoothed_ones():
    """Smoothing spreads one bad fix across the averaging window; detecting on
    the smoothed path reported a single teleport as seventeen."""
    t = _track(n=200, speed_m_s=1.4, accuracy=8.0)
    lat = t.lat.copy()
    lat[100:] += 500.0 / 111195.0
    t = Track(time_s=t.time_s, lat=lat, lon=t.lon, accuracy_m=t.accuracy_m,
              dose_rate=t.dose_rate, count_rate=t.count_rate)
    _, report = assess_with_report(t)
    assert t.smoothing_window > 1, "this track should be smoothed"
    assert report.n_speed_outliers == 1


def test_one_bad_fix_masks_only_itself():
    """A spike makes both of its segments implausible; the fix after it was fine."""
    t = _track(n=40, speed_m_s=1.4)
    lat = t.lat.copy()
    lat[20] += 0.01            # ~1.1 km out and straight back
    t = Track(time_s=t.time_s, lat=lat, lon=t.lon, accuracy_m=t.accuracy_m,
              dose_rate=t.dose_rate, count_rate=t.count_rate)
    out, report = assess_with_report(t)
    assert report.n_speed_outliers == 2
    assert np.flatnonzero(~out.position_mask).tolist() == [20]


def test_a_glitch_adds_no_distance():
    """Bypassed, not smeared: the route is as long as it would be without it."""
    clean = _track(n=40, speed_m_s=1.4)
    lat = clean.lat.copy()
    lat[20] += 0.01
    glitched = Track(time_s=clean.time_s, lat=lat, lon=clean.lon,
                     accuracy_m=clean.accuracy_m, dose_rate=clean.dose_rate,
                     count_rate=clean.count_rate)
    out, _ = assess_with_report(glitched.with_mask(smoothing=1))
    assert float(out.distance_along_m[-1]) == pytest.approx(1.4 * 2.0 * 39,
                                                            rel=1e-3)
    assert out.speed_m_s.max() < 2.0


def test_a_jump_within_one_second_is_not_stationary():
    """dt = 0 used to read as zero speed, so a same-second teleport slipped by."""
    t = _track(n=40, speed_m_s=1.4)
    time_s, lat = t.time_s.copy(), t.lat.copy()
    time_s[20] = time_s[19]
    lat[20] += 0.01
    t = Track(time_s=time_s, lat=lat, lon=t.lon, accuracy_m=t.accuracy_m,
              dose_rate=t.dose_rate, count_rate=t.count_rate)
    out, report = assess_with_report(t)
    assert report.n_speed_outliers >= 1
    assert not out.position_mask[20]


def test_an_imprecise_fix_still_traces_the_route():
    """Hidden from the map, but smoothing is what handles its noise."""
    t = _track(n=10)
    acc = t.accuracy_m.copy()
    acc[3] = 300.0
    t = Track(time_s=t.time_s, lat=t.lat, lon=t.lon, accuracy_m=acc,
              dose_rate=t.dose_rate, count_rate=t.count_rate)
    out, _ = assess_with_report(t)
    assert not out.position_mask[3]
    assert out.trusted_position[3]


# ---- polyline ------------------------------------------------------------


def test_polyline_breaks_at_a_dropout():
    t = _track(n=10)
    time_s = t.time_s.copy()
    time_s[5:] += 600.0
    t = Track(time_s=time_s, lat=t.lat, lon=t.lon, accuracy_m=t.accuracy_m,
              dose_rate=t.dose_rate, count_rate=t.count_rate)
    out, report = assess_with_report(t)
    x, _ = polyline(out, report)
    assert int(np.isnan(x).sum()) == 1
    assert x.size == out.n + 1


def test_polyline_hides_position_masked_samples():
    t = _track(n=10)
    acc = t.accuracy_m.copy()
    acc[4] = 900.0
    t = Track(time_s=t.time_s, lat=t.lat, lon=t.lon, accuracy_m=acc,
              dose_rate=t.dose_rate, count_rate=t.count_rate)
    out, report = assess_with_report(t)
    x, _ = polyline(out, report)
    assert np.isnan(x[4])


def test_gap_segments_report_duration_and_distance():
    t = _track(n=10)
    time_s = t.time_s.copy()
    time_s[5:] += 600.0
    t = Track(time_s=time_s, lat=t.lat, lon=t.lon, accuracy_m=t.accuracy_m,
              dose_rate=t.dose_rate, count_rate=t.count_rate)
    out, report = assess_with_report(t)
    gaps = gap_segments(out, report)
    assert len(gaps) == 1
    start, stop, seconds, _ = gaps[0]
    assert (start, stop) == (4, 5)
    assert seconds == pytest.approx(602.0)


# ---- goldens against the real sample -------------------------------------


def test_sample_quality_goldens(sample):
    _, report = assess_with_report(sample)
    assert report.n_poor_accuracy == 475
    assert report.n_gaps == 13
    assert report.n_legs == 14
    assert report.gap_time_s == pytest.approx(3628.0)
    assert report.gap_distance_m == pytest.approx(536_258.0, rel=1e-4)
    assert report.longest_gap_s == pytest.approx(763.0)
    assert report.coverage == pytest.approx(6126.0 / 9754.0)


def test_sample_flags_its_two_impossible_segments_and_no_more(sample):
    """1330 m/s is Mach 3.9 -- implausible even for the flight this file is.

    Only two of 2963 segments are flagged, so a genuinely fast platform is not
    masked wholesale; that is the robust relative bound doing its job.
    """
    _, report = assess_with_report(sample)
    assert report.n_speed_outliers == 2
    assert int((sample.raw_speed_m_s > 400.0).sum()) == 2
    assert report.n_speed_outliers / sample.n < 0.001


def test_sample_distance_survives_masking(sample):
    """If masking moved the distance, the masking would itself be corrupting
    the geometry it is supposed to protect."""
    out, _ = assess_with_report(sample)
    before = float(sample.distance_along_m[-1])
    after = float(out.distance_along_m[-1])
    assert abs(after - before) / before < 0.0005


def test_sample_polyline_has_one_nan_per_gap_plus_one_per_masked_fix(sample):
    out, report = assess_with_report(sample)
    x, _ = polyline(out, report)
    # 475 poor fixes + 13 dropout separators + 2 impossible segments.
    assert int(np.isnan(x).sum()) == 475 + 13 + 2


def test_a_stricter_accuracy_threshold_masks_more(sample):
    _, loose = assess_with_report(sample, QualityParams(max_accuracy_m=200.0))
    _, strict = assess_with_report(sample, QualityParams(max_accuracy_m=20.0))
    assert strict.n_poor_accuracy > loose.n_poor_accuracy


def _paused_track(pause_s=10.0, n=200, dt=2.0):
    """Two recording periods, with a pause shorter than the gap threshold."""
    from datetime import datetime, timezone

    from rcta.core.track import Period, Track, TrackMeta
    time_s = np.arange(n) * dt + 1_788_800_000.0
    time_s[n // 2:] += pause_s
    def utc(t):
        return datetime.fromtimestamp(t, tz=timezone.utc)
    periods = (Period(distance_m=100.0, start=utc(time_s[0]), end=utc(time_s[n // 2 - 1] + 1)),
               Period(distance_m=100.0, start=utc(time_s[n // 2]), end=utc(time_s[-1] + 1)))
    return Track(time_s=time_s, lat=np.linspace(30.0, 30.01, n),
                 lon=np.full(n, -97.0), accuracy_m=np.full(n, 5.0),
                 dose_rate=np.full(n, 90.0), count_rate=np.full(n, 7.0),
                 meta=TrackMeta(title="paused", periods=periods))


def test_a_short_device_pause_still_splits_the_legs():
    """10 s is under the 30 s gap rule, but the device says it paused."""
    track, report = assess_with_report(_paused_track(pause_s=10.0))
    assert report.n_legs == 2
    assert report.gap_after[99] and report.gap_after.sum() == 1


def test_samples_outside_every_recording_period_are_reported():
    from dataclasses import replace
    track = _paused_track()
    shifted = replace(track, time_s=track.time_s + 3600.0)
    _, report = assess_with_report(shifted)
    assert any("outside the device's own recording periods" in w
               for w in report.warnings)


def test_the_bundled_tracks_lie_inside_their_single_period(sample):
    _, report = assess_with_report(sample)
    assert not any("recording periods" in w for w in report.warnings)
