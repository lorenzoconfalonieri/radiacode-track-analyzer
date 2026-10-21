"""The data model, and the geometry goldens that guard against regressions."""

from __future__ import annotations

import numpy as np
import pytest

from rcta.core.track import Track


def _track(**kw):
    base = dict(time_s=[0.0, 2.0, 4.0], lat=[30.0, 30.001, 30.002],
                lon=[-97.0, -97.0, -97.0], accuracy_m=[5.0, 5.0, 5.0],
                dose_rate=[90.0, 95.0, 100.0], count_rate=[18.0, 19.0, 20.0])
    base.update(kw)
    return Track(**base)


# ---- invariants ----------------------------------------------------------


def test_columns_are_sorted_together_not_independently():
    """Sorting one column and not the rest would silently scramble the data."""
    t = _track(time_s=[4.0, 0.0, 2.0])
    assert t.time_s.tolist() == [0.0, 2.0, 4.0]
    assert t.dose_rate.tolist() == [95.0, 100.0, 90.0]
    assert t.lat.tolist() == [30.001, 30.002, 30.0]


def test_masks_follow_the_sort():
    t = _track(time_s=[4.0, 0.0, 2.0], valid_mask=[False, True, True])
    assert t.valid_mask.tolist() == [True, True, False]


def test_columns_are_read_only():
    """Cached geometry would go stale, and workers share these arrays."""
    t = _track()
    with pytest.raises(ValueError):
        t.dose_rate[0] = 1.0


def test_mismatched_column_lengths_raise():
    with pytest.raises(ValueError, match="length mismatch"):
        _track(dose_rate=[1.0, 2.0])


def test_empty_track_raises():
    with pytest.raises(ValueError, match="at least one sample"):
        Track(time_s=[], lat=[], lon=[], accuracy_m=[], dose_rate=[],
              count_rate=[])


def test_non_positive_integration_time_raises():
    with pytest.raises(ValueError, match="must be positive"):
        _track(integration_time_s=0.0)


def test_with_mask_does_not_mutate_the_original():
    t = _track()
    other = t.with_mask(position=[True, False, True])
    assert t.position_mask.all()
    assert other.position_mask.tolist() == [True, False, True]
    assert other.dose_rate is t.dose_rate


# ---- derived geometry ----------------------------------------------------


def test_speed_aligns_with_the_columns_and_has_no_leading_gap():
    t = _track()
    assert t.speed_m_s.size == t.n
    assert t.speed_m_s[0] == pytest.approx(t.speed_m_s[1])


def test_distance_starts_at_zero_and_is_monotone():
    t = _track()
    assert t.distance_along_m[0] == 0.0
    assert np.all(np.diff(t.distance_along_m) >= 0)


def test_single_sample_track_has_zero_length_geometry():
    t = Track(time_s=[5.0], lat=[30.0], lon=[-97.0], accuracy_m=[5.0],
              dose_rate=[90.0], count_rate=[18.0])
    assert t.distance_along_m.tolist() == [0.0]
    assert t.speed_m_s.tolist() == [0.0]
    assert t.bbox == (-97.0, 30.0, -97.0, 30.0)


def test_bbox_ignores_position_masked_samples():
    t = _track(lat=[30.0, 89.0, 30.002],
               position_mask=[True, False, True])
    assert t.bbox[3] == pytest.approx(30.002)


def test_select_returns_a_track_over_the_subset():
    t = _track()
    sub = t.select([True, False, True])
    assert sub.n == 2
    assert sub.dose_rate.tolist() == [90.0, 100.0]


def test_select_inherits_the_resolved_integration_time():
    """Otherwise a two-sample selection would re-infer a different live time
    and its Currie limits would not match the parent track's."""
    t = _track()
    sub = t.select([True, False, True])
    assert sub.effective_integration_time_s == t.effective_integration_time_s


def test_select_rejects_an_empty_mask():
    with pytest.raises(ValueError, match="empty subset"):
        _track().select([False, False, False])


def test_point_round_trips_a_sample():
    t = _track()
    p = t.point(1)
    assert p.index == 1
    assert p.dose_rate == pytest.approx(95.0)
    assert "nSv/h" in p.describe()


# ---- integration time ----------------------------------------------------


def test_integration_time_is_inferred_from_the_median_spacing():
    t = _track()
    assert t.effective_integration_time_s == pytest.approx(2.0)


def test_integration_time_is_not_inflated_by_a_dropout():
    """Multiplying a count rate by a 763 s gap would invent 16 000 counts.

    The median is what protects against this, so the track needs enough
    intervals for a median to mean anything -- with only two, the median of
    [2 s, 763 s] is 382 s and no estimator could do better.
    """
    time_s = np.arange(60, dtype=float) * 2.0
    time_s[30:] += 763.0
    t = _track(time_s=time_s, lat=np.linspace(30.0, 30.01, 60),
               lon=np.full(60, -97.0), accuracy_m=np.full(60, 5.0),
               dose_rate=np.full(60, 90.0), count_rate=np.full(60, 18.0))
    assert t.effective_integration_time_s == pytest.approx(2.0)


def test_a_pathological_log_is_still_capped():
    """The bound is a backstop, not the main defence. It was 8 s, which
    silently clipped a real 10 s walking cadence and lost 20 % of the counts
    behind every detection limit."""
    time_s = np.arange(20, dtype=float) * 4000.0
    t = _track(time_s=time_s, lat=np.linspace(30.0, 30.1, 20),
               lon=np.full(20, -97.0), accuracy_m=np.full(20, 5.0),
               dose_rate=np.full(20, 90.0), count_rate=np.full(20, 18.0))
    assert t.effective_integration_time_s == 60.0


def test_a_ten_second_logging_cadence_is_taken_at_face_value():
    """The real RadiaCode walking log samples every 10 s."""
    time_s = np.arange(60, dtype=float) * 10.0
    t = _track(time_s=time_s, lat=np.linspace(30.0, 30.01, 60),
               lon=np.full(60, -97.0), accuracy_m=np.full(60, 14.0),
               dose_rate=np.full(60, 7.0), count_rate=np.full(60, 5.3))
    assert t.effective_integration_time_s == pytest.approx(10.0)


def test_supplied_integration_time_wins():
    t = _track(integration_time_s=1.0)
    assert t.effective_integration_time_s == 1.0
    assert t.counts.tolist() == [18.0, 19.0, 20.0]


# ---- goldens against the real sample -------------------------------------


def test_sample_distance_matches_the_device_odometer(sample):
    """The strongest single guard against a geometry regression: an external
    number the program did not compute."""
    measured = float(sample.distance_along_m[-1])
    assert measured == pytest.approx(1_792_210.09, rel=1e-6)
    declared = sample.meta.declared_distance_m
    assert abs(measured - declared) / declared < 0.002


def test_sample_file_order_distance_would_be_absurd(sample):
    """Quantifies what the sort is protecting against: three orders of
    magnitude, not a rounding difference."""
    from rcta.core import geo
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(sample.n)
    scrambled = geo.cumulative_distance(sample.lon[shuffled], sample.lat[shuffled])
    assert float(scrambled[-1]) > 100 * float(sample.distance_along_m[-1])


def test_sample_sampling_interval(sample):
    assert float(np.median(sample.dt_s)) == 2.0
    assert float(sample.dt_s.max()) == 763.0


def test_sample_dose_and_count_ranges(sample):
    # The file stores uR/h; these are the values after conversion to nSv/h.
    assert float(sample.dose_rate.min()) == pytest.approx(27.2)
    assert float(sample.dose_rate.max()) == pytest.approx(1233.4)
    assert float(np.median(sample.dose_rate)) == pytest.approx(905.55)
    assert float(sample.count_rate.min()) == pytest.approx(0.50)
    assert float(sample.count_rate.max()) == pytest.approx(21.07)
    assert float(np.median(sample.count_rate)) == pytest.approx(18.34)


def test_sample_accuracy(sample):
    assert float(np.median(sample.accuracy_m)) == 28.0
    assert float(sample.accuracy_m.max()) == 397.0


def test_sample_bbox(sample):
    assert sample.bbox == pytest.approx(
        (-111.978687, 30.122992, -97.661445, 40.790785))


def test_sample_integration_time_is_two_seconds(sample):
    assert sample.effective_integration_time_s == pytest.approx(2.0)


def test_sample_dose_per_count(sample):
    assert sample.dose_per_count == pytest.approx(51.29, rel=1e-3)


def test_sample_describe_reports_the_odometer_agreement(sample):
    text = sample.describe()
    assert "1792.21 km" in text
    assert "0.16 % apart" in text
    assert "not in time order" in text


# ---- GPS jitter on a walking survey ---------------------------------------


def _walk(n=1200, dt=2.0, speed_m_s=1.4, accuracy_m=8.0, seed=0):
    """A walk with realistic per-fix GPS noise on both axes."""
    rng = np.random.default_rng(seed)
    metres = 111195.0
    north = np.arange(n) * speed_m_s * dt
    lat = 30.2 + north / metres + rng.normal(0, accuracy_m, n) / metres
    lon = (-97.66 + rng.normal(0, accuracy_m, n)
           / (metres * np.cos(np.radians(30.2))))
    true_km = speed_m_s * dt * (n - 1) / 1000.0
    return Track(time_s=np.arange(n) * dt, lat=lat, lon=lon,
                 accuracy_m=np.full(n, accuracy_m),
                 dose_rate=np.full(n, 15.0), count_rate=np.full(n, 3.0)), true_km


def test_a_walk_measures_its_true_distance_not_its_gps_jitter():
    """The bug this correction exists for.

    At 1.4 m/s with 2 s logging the true step is 2.8 m and a good fix is 8 m,
    so raw summed distance is mostly noise -- measured at 5.2x the truth.
    """
    track, true_km = _walk()
    assert track.smoothing_window > 1
    assert float(track.distance_along_m[-1]) / 1000.0 == pytest.approx(
        true_km, rel=0.10)
    raw_km = float(track.raw_distance_along_m[-1]) / 1000.0
    assert raw_km > 3.0 * true_km, "the raw path really is that inflated"


@pytest.mark.parametrize("accuracy", [3.0, 8.0, 20.0])
def test_the_correction_holds_across_gps_quality(accuracy):
    track, true_km = _walk(accuracy_m=accuracy)
    assert float(track.distance_along_m[-1]) / 1000.0 == pytest.approx(
        true_km, rel=0.12)


def test_walking_speed_is_believable_after_correction():
    """Speed inherits the distance error, so a 1.4 m/s walk reads at 7 m/s."""
    track, _ = _walk()
    assert float(np.median(track.speed_m_s)) == pytest.approx(1.4, abs=0.4)


def _staircase(leg_m, accuracy_m, seed, n=1200, dt=2.0, speed_m_s=1.4):
    """A walk of right-angle turns every ``leg_m``, alternately N and E."""
    rng = np.random.default_rng(seed)
    metres = 111195.0
    along = np.arange(n) * speed_m_s * dt
    k = (along // leg_m).astype(int)
    f = along - k * leg_m
    half = k // 2
    north = np.where(k % 2 == 0, half * leg_m + f, (half + 1) * leg_m)
    east = np.where(k % 2 == 0, half * leg_m, half * leg_m + f)
    lat = 30.2 + (north + rng.normal(0, accuracy_m, n)) / metres
    lon = -97.66 + (east + rng.normal(0, accuracy_m, n)) / (
        metres * np.cos(np.radians(30.2)))
    return Track(time_s=np.arange(n) * dt, lat=lat, lon=lon,
                 accuracy_m=np.full(n, accuracy_m),
                 dose_rate=np.full(n, 15.0), count_rate=np.full(n, 3.0)), \
        speed_m_s * dt * (n - 1) / 1000.0


def test_smoothing_cuts_corners_as_the_readme_says():
    """The README's limitation, pinned: a turn every 100 m at 8 m accuracy
    reads about 5 % short; every 50 m, about 20 %."""
    for leg_m, low, high in ((100.0, 0.92, 0.98), (50.0, 0.75, 0.86)):
        ratios = []
        for seed in range(8):
            track, true_km = _staircase(leg_m, 8.0, seed)
            ratios.append(float(track.distance_along_m[-1]) / 1000.0 / true_km)
        assert low <= float(np.mean(ratios)) <= high, (leg_m, np.mean(ratios))


def test_a_vehicle_track_is_left_alone():
    """Smoothing must not touch data whose steps already dominate the noise."""
    track, true_km = _walk(speed_m_s=25.0, accuracy_m=8.0)
    assert track.smoothing_window == 1
    assert np.array_equal(track.smooth_lat, track.lat)


def test_the_sample_flight_is_not_smoothed(sample):
    """466 m steps against a 28 m fix: nothing to correct, and the odometer
    agreement proves it."""
    assert sample.smoothing_window == 1
    assert float(sample.distance_along_m[-1]) == pytest.approx(
        float(sample.raw_distance_along_m[-1]))


def test_describe_reports_how_much_of_the_raw_path_was_jitter():
    track, _ = _walk()
    text = track.describe()
    assert "smoothing" in text
    assert "GPS jitter, not travel" in text


def test_a_selection_inherits_the_parent_smoothing_window():
    """Otherwise a short selection re-infers a different window and its
    distance stops being comparable with the whole track's."""
    track, _ = _walk()
    sub = track.select(np.arange(track.n) < 300)
    assert sub.smoothing_window == track.smoothing_window


# ---- geometry from trusted fixes only ------------------------------------


def test_smoothing_does_not_reach_across_a_dropout():
    """Two legs 100 km apart: the first leg's end must stay where it was."""
    walk, _ = _walk(n=400)
    lat = walk.lat.copy()
    lat[200:] += 1.0
    time_s = walk.time_s.copy()
    time_s[200:] += 3600.0
    legs = np.r_[np.zeros(200), np.ones(200)]
    track = Track(time_s=time_s, lat=lat, lon=walk.lon,
                  accuracy_m=walk.accuracy_m, dose_rate=walk.dose_rate,
                  count_rate=walk.count_rate, leg_index=legs)
    assert track.smoothing_window > 1
    assert abs(track.smooth_lat[199] - lat[150:200].mean()) < 0.001


def test_smoothing_survives_the_antimeridian():
    """Averaging 179.9 and -179.9 gives zero unless longitude is unwrapped."""
    n = 200

    def along(lon):
        return Track(time_s=np.arange(n) * 2.0, lat=np.full(n, 10.0), lon=lon,
                     accuracy_m=np.full(n, 8.0), dose_rate=np.full(n, 90.0),
                     count_rate=np.full(n, 18.0), position_smoothing=9)

    crossing = along((np.linspace(179.5, 180.5, n) + 180.0) % 360.0 - 180.0)
    control = along(np.linspace(-0.5, 0.5, n))
    assert np.all(np.abs(crossing.smooth_lon) > 179.0)
    assert crossing.distance_along_m[-1] == pytest.approx(
        control.distance_along_m[-1], rel=1e-6)


def test_a_masked_sample_takes_no_part_in_the_geometry():
    walk, _ = _walk(n=100)
    lat = walk.lat.copy()
    lat[50] = 0.0              # a borrowed place on the far side of the world
    valid = np.ones(100, bool)
    valid[50] = False
    track = Track(time_s=walk.time_s, lat=lat, lon=walk.lon,
                  accuracy_m=walk.accuracy_m, dose_rate=walk.dose_rate,
                  count_rate=walk.count_rate, valid_mask=valid)
    assert track.speed_m_s.max() < 5.0
    assert track.bbox[1] > 29.0


def test_a_point_describes_itself_in_the_requested_unit(sample):
    text = sample.point(0).describe("uSv/h")
    assert "uSv/h" in text and "nSv/h" not in text


def _timed(times):
    n = len(times)
    return Track(time_s=np.asarray(times, float), lat=np.full(n, 40.0),
                 lon=np.full(n, -111.9), accuracy_m=np.full(n, 8.0),
                 dose_rate=np.full(n, 70.0), count_rate=np.full(n, 5.0))


def test_a_single_sample_does_not_claim_an_inferred_integration_time():
    track = _timed([0.0])
    assert "no spacing to infer from" in track.integration_time_source
    assert "median spacing" not in track.describe()


def test_a_clipped_integration_time_says_it_was_clipped():
    assert track_source([0.0, 0.2, 0.4]).startswith(
        "inferred from median spacing (0.2 s), clipped")
    assert track_source([0.0, 2.0, 4.0]) == "inferred from median spacing"


def track_source(times):
    return _timed(times).integration_time_source


@pytest.mark.parametrize("accuracy, raw_factor", [(3.0, 2.1), (8.0, 5.2)])
def test_the_readme_jitter_figures_hold(accuracy, raw_factor):
    """README, 'Walking surveys': raw distance overstates a walk 5.2x at 8 m
    and 2.1x at 3 m, and the correction lands within 3 % across 3-20 m.
    The tests above allow 10-12 %; this pins the published claim, averaged
    over seeds so it is the figure and not one draw. The correction leaves
    straight walks 2-5 % long (held leg ends used to hide that by cutting
    about as much off the ends)."""
    raw = [(_walk(accuracy_m=accuracy, seed=s)) for s in range(8)]
    ratio = np.mean([t.raw_distance_along_m[-1] / 1000.0 / km for t, km in raw])
    assert ratio == pytest.approx(raw_factor, abs=0.15)


@pytest.mark.parametrize("accuracy", [3.0, 8.0, 20.0])
def test_the_correction_is_within_five_percent_as_published(accuracy):
    for seed in range(8):
        track, true_km = _walk(accuracy_m=accuracy, seed=seed)
        ratio = float(track.distance_along_m[-1]) / 1000.0 / true_km
        # Residual jitter only ever lengthens a straight path.
        assert 1.0 <= ratio <= 1.05
