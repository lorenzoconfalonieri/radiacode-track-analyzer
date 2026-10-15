"""Projection and geodesy, checked against values known independently."""

from __future__ import annotations

import numpy as np
import pytest

from rcta.core import geo


def test_origin_projects_to_the_origin():
    x, y = geo.lonlat_to_web_mercator(0.0, 0.0)
    assert float(x) == pytest.approx(0.0, abs=1e-9)
    assert float(y) == pytest.approx(0.0, abs=1e-9)


def test_world_corner_matches_the_epsg_definition():
    """The defining constant of EPSG:3857, to the metre it is published at."""
    x, y = geo.lonlat_to_web_mercator(180.0, geo.MAX_LAT)
    assert float(x) == pytest.approx(20037508.342789244, abs=1e-6)
    assert float(y) == pytest.approx(20037508.342789244, abs=1e-3)


@pytest.mark.parametrize("lon", [-179.9, -104.5, 0.0, 12.3, 179.9])
@pytest.mark.parametrize("lat", [-84.0, -33.9, 0.0, 36.15, 84.0])
def test_projection_round_trips(lon, lat):
    back = geo.web_mercator_to_lonlat(*geo.lonlat_to_web_mercator(lon, lat))
    assert float(back[0]) == pytest.approx(lon, abs=1e-9)
    assert float(back[1]) == pytest.approx(lat, abs=1e-9)


def test_latitude_beyond_the_mercator_limit_is_clamped_not_infinite():
    """A pole would project to infinity and poison every downstream autoscale."""
    _, y = geo.lonlat_to_web_mercator(0.0, 90.0)
    assert np.isfinite(y)


def test_one_degree_of_latitude_is_about_111_km():
    assert float(geo.haversine(0, 0, 0, 1)) == pytest.approx(111195.0, abs=1.0)


def test_a_degree_of_longitude_shrinks_with_latitude():
    """cos(60 deg) is exactly 0.5, so this is an analytic check, not a fixture."""
    equator = float(geo.haversine(0, 0, 1, 0))
    sixty = float(geo.haversine(0, 60, 1, 60))
    assert sixty == pytest.approx(equator * 0.5, rel=1e-4)


def test_haversine_is_symmetric_and_zero_on_itself():
    assert float(geo.haversine(10, 20, 10, 20)) == pytest.approx(0.0, abs=1e-9)
    assert float(geo.haversine(1, 2, 3, 4)) == pytest.approx(
        float(geo.haversine(3, 4, 1, 2)))


def test_bearing_cardinal_directions():
    assert float(geo.bearing(0, 0, 0, 1)) == pytest.approx(0.0, abs=1e-6)
    assert float(geo.bearing(0, 0, 1, 0)) == pytest.approx(90.0, abs=1e-6)
    assert float(geo.bearing(0, 0, 0, -1)) == pytest.approx(180.0, abs=1e-6)


def test_mercator_scale_is_one_at_the_equator_and_two_at_sixty():
    assert float(geo.mercator_scale(0.0)) == pytest.approx(1.0)
    assert float(geo.mercator_scale(60.0)) == pytest.approx(2.0, rel=1e-9)


def test_cumulative_distance_starts_at_zero_and_is_monotone():
    lon = np.linspace(0, 1, 20)
    lat = np.zeros(20)
    d = geo.cumulative_distance(lon, lat)
    assert d[0] == 0.0
    assert np.all(np.diff(d) >= 0)
    assert d[-1] == pytest.approx(float(geo.haversine(0, 0, 1, 0)), rel=1e-9)


def test_point_segment_distance_uses_the_segment_not_its_endpoints():
    """The whole reason this exists: a point beside a long segment is close to
    it, even though it is far from either vertex."""
    d = geo.point_segment_distance([500.0], [10.0], 0.0, 0.0, 1000.0, 0.0)
    assert float(d[0]) == pytest.approx(10.0)
    ends = min(np.hypot(500.0, 10.0), np.hypot(500.0, 10.0))
    assert float(d[0]) < ends


def test_point_segment_distance_clamps_beyond_the_ends():
    d = geo.point_segment_distance([-30.0], [40.0], 0.0, 0.0, 100.0, 0.0)
    assert float(d[0]) == pytest.approx(50.0)


def test_degenerate_segment_falls_back_to_point_distance():
    d = geo.point_segment_distance([3.0], [4.0], 0.0, 0.0, 0.0, 0.0)
    assert float(d[0]) == pytest.approx(5.0)


# ---- tiles ---------------------------------------------------------------


@pytest.mark.parametrize("z", [0, 1, 8, 14, 19])
def test_zoom_for_resolution_inverts_tile_resolution(z):
    assert geo.zoom_for_resolution(geo.tile_resolution(z), max_zoom=19) == z


def test_zoom_is_clamped_to_the_providers_maximum():
    assert geo.zoom_for_resolution(1e-6, max_zoom=17) == 17
    assert geo.zoom_for_resolution(1e12, max_zoom=17) == 0


def test_tile_zero_covers_the_whole_world():
    x0, y0, x1, y1 = geo.tile_bounds_3857(0, 0, 0)
    assert x0 == pytest.approx(-geo.WORLD_M)
    assert y1 == pytest.approx(geo.WORLD_M)
    assert x1 == pytest.approx(geo.WORLD_M)
    assert y0 == pytest.approx(-geo.WORLD_M)


def test_tile_row_zero_is_the_north_not_the_south():
    """Slippy-map rows count down from the pole; the plot's y counts up. This
    is the single easiest sign error to make in the whole map subsystem."""
    _, y0_north, _, y1_north = geo.tile_bounds_3857(1, 0, 0)
    _, y0_south, _, y1_south = geo.tile_bounds_3857(1, 0, 1)
    assert y1_north > y1_south
    assert y0_north == pytest.approx(0.0)


def test_tiles_for_bounds_covers_the_rectangle_centre_first():
    keys = geo.tiles_for_bounds(-1e6, -1e6, 1e6, 1e6, 6)
    assert keys, "a non-empty rectangle must yield tiles"
    assert all(k[0] == 6 for k in keys)
    # Every returned tile must actually intersect the request.
    for _, x, y in keys:
        bx0, by0, bx1, by1 = geo.tile_bounds_3857(6, x, y)
        assert bx1 >= -1e6 and bx0 <= 1e6
        assert by1 >= -1e6 and by0 <= 1e6


def test_tiles_for_bounds_clamps_at_the_world_edge():
    keys = geo.tiles_for_bounds(-3e7, -3e7, 3e7, 3e7, 2)
    assert len(keys) == 16
    assert all(0 <= x < 4 and 0 <= y < 4 for _, x, y in keys)


# ---- gaps ----------------------------------------------------------------


def test_a_long_pause_without_movement_is_not_a_break():
    """A stationary measurement is not a dropout, and joining it is correct."""
    t = np.array([0.0, 1.0, 600.0, 601.0])
    lon = np.zeros(4)
    lat = np.zeros(4)
    assert not geo.segment_breaks(t, lon, lat).any()


def test_a_long_gap_with_movement_breaks_the_line():
    t = np.array([0.0, 2.0, 800.0, 802.0])
    lat = np.array([0.0, 0.0, 1.0, 1.0])
    lon = np.zeros(4)
    breaks = geo.segment_breaks(t, lon, lat)
    assert breaks[1]


def test_a_teleport_in_two_seconds_breaks_the_line():
    """A time-only rule would draw straight through this."""
    t = np.arange(0.0, 10.0, 2.0)
    lat = np.array([0.0, 0.0001, 5.0, 5.0001, 5.0002])
    lon = np.zeros(5)
    assert geo.segment_breaks(t, lon, lat)[1]


def test_connect_array_marks_the_break_and_matches_point_count():
    breaks = np.array([False, True, False])
    connect = geo.connect_array(breaks)
    assert connect.size == 4
    assert connect.dtype == np.uint8
    assert connect[1] == 0
    assert connect[0] == 1 and connect[2] == 1


def test_segment_breaks_handles_a_single_sample():
    assert geo.segment_breaks([0.0], [0.0], [0.0]).size == 0


def test_a_stop_start_drive_produces_no_false_breaks():
    """The regression that killed the percentile-based bound.

    Half the samples are stationary, so the median speed is near zero. A rule
    keyed to the median alone would call every moving segment a teleport.
    """
    n, dt = 400, 2.0
    moving = np.tile([0, 0, 1, 1], n // 4).astype(bool)
    step = np.where(moving, 30.0 * dt, 0.2)
    lat = 30.0 + np.cumsum(step) / 111195.0
    t = np.arange(n) * dt
    assert not geo.segment_breaks(t, np.zeros(n), lat).any()


def test_robust_bound_is_not_raised_by_the_outlier_it_must_catch():
    """A high percentile over few segments *is* the teleport."""
    speeds = np.array([5.0, 5.0, 5.0, 277000.0])
    assert geo._robust_max(speeds) < 100.0
    assert float(np.percentile(speeds, 95)) > 1e5


# ---- position smoothing ----------------------------------------------------


def _walk(n=600, step_m=2.8, sigma_m=8.0, seed=1):
    """A straight north-going walk with Gaussian fix noise, in degrees."""
    rng = np.random.default_rng(seed)
    north = np.arange(n) * step_m + rng.normal(0, sigma_m, n)
    east = rng.normal(0, sigma_m, n)
    lat = 40.0 + north / 111_320.0
    lon = -111.9 + east / (111_320.0 * np.cos(np.radians(40.0)))
    return lat, lon


def test_smoothing_with_window_one_is_the_identity():
    from rcta.core.geo import smooth_positions
    lat, lon = _walk(50)
    out_lat, out_lon = smooth_positions(lat, lon, 1)
    np.testing.assert_array_equal(out_lat, lat)
    np.testing.assert_array_equal(out_lon, lon)


def test_a_window_wider_than_the_track_leaves_it_raw():
    from rcta.core.geo import smooth_positions
    lat, lon = _walk(5)
    out_lat, _ = smooth_positions(lat, lon, 9)
    np.testing.assert_array_equal(out_lat, lat)


def test_smoothing_is_centred_so_a_straight_line_stays_put():
    """An off-centre kernel would shift every point along the track."""
    from rcta.core.geo import smooth_positions
    lat = np.linspace(40.0, 40.01, 101)
    lon = np.linspace(-111.9, -111.89, 101)
    out_lat, out_lon = smooth_positions(lat, lon, 7)
    np.testing.assert_allclose(out_lat[3:-3], lat[3:-3], rtol=0, atol=1e-12)
    np.testing.assert_allclose(out_lon[3:-3], lon[3:-3], rtol=0, atol=1e-12)
    # Endpoints are held at the first and last fully averaged value.
    assert (out_lat[:3] == out_lat[3]).all()
    assert (out_lat[-3:] == out_lat[-4]).all()


def test_smoothing_removes_most_of_the_cross_track_noise():
    from rcta.core.geo import smooth_positions
    lat, lon = _walk(sigma_m=8.0)
    _, out_lon = smooth_positions(lat, lon, 15)
    assert np.std(out_lon[20:-20]) < np.std(lon) / 3.0


def test_the_window_is_always_odd():
    """np.convolve(mode="same") is only centred for an odd kernel."""
    from rcta.core.geo import choose_smoothing_window
    for sigma in (3.0, 8.0, 20.0):
        lat, lon = _walk(sigma_m=sigma)
        window = choose_smoothing_window(lat, lon, np.full(lat.size, sigma))
        assert window % 2 == 1


def test_a_walk_needs_smoothing_and_a_fast_platform_does_not():
    from rcta.core.geo import choose_smoothing_window
    lat, lon = _walk(step_m=2.8, sigma_m=8.0)
    assert choose_smoothing_window(lat, lon, np.full(lat.size, 8.0)) > 1
    # 100 m per fix: a highway at 10 s logging, or anything airborne.
    lat, lon = _walk(step_m=100.0, sigma_m=8.0)
    assert choose_smoothing_window(lat, lon, np.full(lat.size, 8.0)) == 1


def test_no_recorded_accuracy_means_no_smoothing():
    from rcta.core.geo import choose_smoothing_window
    lat, lon = _walk()
    assert choose_smoothing_window(lat, lon, np.full(lat.size, np.nan)) == 1


def test_a_platform_that_never_moved_gets_the_widest_window():
    from rcta.core.geo import choose_smoothing_window
    lat, lon = _walk(step_m=0.0, sigma_m=8.0)
    assert choose_smoothing_window(lat, lon, np.full(lat.size, 8.0),
                                   max_window=121) == 121
