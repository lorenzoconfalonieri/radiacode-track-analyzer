"""The interpolated surface, its masks, and hotspot clustering."""

from __future__ import annotations

import numpy as np
import pytest
import synthetic

from rcta.core import geo
from rcta.core.anomalies import ScanParams, scan
from rcta.core.background import BackgroundParams, estimate_background
from rcta.core.quality import assess_with_report
from rcta.core.spatial import GridParams, find_hotspots, interpolate


def _analysed(track, params=None):
    bg = estimate_background(track, BackgroundParams())
    return bg, scan(track, bg, params or ScanParams())


# ---- parameters ----------------------------------------------------------


def test_invalid_value_field_raises():
    with pytest.raises(ValueError, match='"dose" or "excess"'):
        GridParams(value="counts")


def test_excess_without_a_background_raises():
    track, _ = synthetic.make(n=200, shuffle=False)
    with pytest.raises(ValueError, match="needs a background"):
        interpolate(track, GridParams(value="excess"))


def test_too_few_positioned_samples_raises():
    track, _ = synthetic.make(n=200, shuffle=False)
    blind = track.with_mask(position=np.zeros(track.n, dtype=bool))
    with pytest.raises(ValueError, match="at least three positioned"):
        interpolate(blind)


# ---- the masks, which are the point of the module ------------------------


def test_cells_far_from_the_track_are_blank_not_extrapolated():
    """The central guarantee: no confident colour over unvisited ground."""
    track, _ = synthetic.make(n=400, path="line", speed_m_s=1.4, shuffle=False)
    grid = interpolate(track, GridParams(bandwidth_m=50.0, max_distance_m=100.0,
                                         padding_bandwidths=8.0))
    far = grid.distance_to_track_m > 100.0
    assert far.any(), "the padded grid must extend beyond the mask"
    assert np.isnan(grid.values[far]).all()


def test_the_geometric_mask_agrees_with_a_brute_force_distance():
    """Independently recomputed, so a bug in the KD-tree path cannot hide."""
    track, _ = synthetic.make(n=200, path="line", speed_m_s=1.4, shuffle=False)
    grid = interpolate(track, GridParams(bandwidth_m=60.0, grid_size=48))

    xc = 0.5 * (grid.x_edges[:-1] + grid.x_edges[1:])
    yc = 0.5 * (grid.y_edges[:-1] + grid.y_edges[1:])
    x, y = track.merc_x, track.merc_y
    brute = np.full((yc.size, xc.size), np.inf)
    for i in range(track.n - 1):
        for j, cy in enumerate(yc):
            d = geo.point_segment_distance(xc, np.full(xc.size, cy),
                                           x[i], y[i], x[i + 1], y[i + 1])
            brute[j] = np.minimum(brute[j], d)
    inflation = float(geo.mercator_scale(float(np.median(track.lat))))
    assert np.allclose(grid.distance_to_track_m, brute / inflation,
                       atol=grid.bandwidth_m * 0.6)


def test_a_lawnmower_block_is_covered_between_its_passes():
    """Adjacent passes 30 m apart with a 100 m bandwidth: the interior is
    genuinely surveyed and must not be punched full of holes."""
    track, _ = synthetic.make(n=800, path="lawnmower", speed_m_s=1.4,
                              shuffle=False, seed=5)
    grid = interpolate(track, GridParams(bandwidth_m=100.0))
    assert grid.coverage > 0.5


def test_a_fast_straight_leg_is_not_masked_into_a_dotted_line():
    """Vertex distance would leave gaps between fixes 470 m apart; segment
    distance does not. This is the reason the mask densifies segments."""
    track, _ = synthetic.make(n=120, path="line", speed_m_s=235.0, dt_s=2.0,
                              shuffle=False)
    grid = interpolate(track, GridParams(bandwidth_m=100.0))

    xc = 0.5 * (grid.x_edges[:-1] + grid.x_edges[1:])
    yc = 0.5 * (grid.y_edges[:-1] + grid.y_edges[1:])
    # Sample the field directly on the path, midway between two fixes.
    mid_x = float((track.merc_x[60] + track.merc_x[61]) / 2)
    mid_y = float((track.merc_y[60] + track.merc_y[61]) / 2)
    j = int(np.argmin(np.abs(xc - mid_x)))
    i = int(np.argmin(np.abs(yc - mid_y)))
    assert np.isfinite(grid.values[i, j]), "a point on the track must be supported"


def test_the_mask_does_not_extend_across_a_dropout():
    """The ground inside a 763 s gap was never surveyed, so a surface over it
    would be an invention."""
    track, _ = synthetic.make(n=200, path="line", speed_m_s=200.0, dt_s=2.0,
                              shuffle=False, gaps=[(100, 900.0)])
    grid = interpolate(track, GridParams(bandwidth_m=100.0))
    mid_x = float((track.merc_x[100] + track.merc_x[101]) / 2)
    mid_y = float((track.merc_y[100] + track.merc_y[101]) / 2)
    xc = 0.5 * (grid.x_edges[:-1] + grid.x_edges[1:])
    yc = 0.5 * (grid.y_edges[:-1] + grid.y_edges[1:])
    j = int(np.argmin(np.abs(xc - mid_x)))
    i = int(np.argmin(np.abs(yc - mid_y)))
    assert np.isnan(grid.values[i, j])


def test_weight_floor_blanks_thinly_supported_cells():
    track, _ = synthetic.make(n=300, path="line", shuffle=False)
    generous = interpolate(track, GridParams(bandwidth_m=60.0, min_weight=1e-9,
                                             max_distance_m=1e9))
    strict = interpolate(track, GridParams(bandwidth_m=60.0, min_weight=2.0,
                                           max_distance_m=1e9))
    assert strict.coverage < generous.coverage


# ---- the field itself ----------------------------------------------------


def test_a_constant_field_is_recovered():
    track, _ = synthetic.make(n=400, path="line", background_cps=18.0,
                              poisson=False, shuffle=False)
    grid = interpolate(track, GridParams(bandwidth_m=60.0))
    finite = grid.values[grid.supported]
    assert np.allclose(finite, finite[0], rtol=1e-6)


def test_the_grid_maximum_lands_near_an_injected_source():
    track, truth = synthetic.make(n=800, path="line", speed_m_s=1.4,
                                  background_cps=18.0, shuffle=False, seed=6,
                                  sources=[(400, 200.0, 5.0)])
    grid = interpolate(track, GridParams(bandwidth_m=50.0))
    yi, xi = np.unravel_index(np.nanargmax(grid.values), grid.values.shape)
    xc = 0.5 * (grid.x_edges[:-1] + grid.x_edges[1:])
    yc = 0.5 * (grid.y_edges[:-1] + grid.y_edges[1:])
    lon, lat = geo.web_mercator_to_lonlat(xc[xi], yc[yi])
    source = truth["sources"][0]
    assert float(geo.haversine(lon, lat, source["lon"], source["lat"])) < 150.0


def test_excess_field_is_centred_on_zero_away_from_sources():
    track, _ = synthetic.make(n=600, path="line", background_cps=18.0,
                              shuffle=False, seed=7)
    bg = estimate_background(track)
    grid = interpolate(track, GridParams(bandwidth_m=60.0, value="excess"),
                       background=bg)
    finite = grid.values[grid.supported]
    assert abs(float(np.median(finite))) < 3.0
    assert "net" in grid.unit


def test_levels_are_quantiles_and_strictly_increasing():
    track, _ = synthetic.make(n=400, path="line", shuffle=False, seed=8)
    levels = interpolate(track, GridParams(bandwidth_m=60.0)).levels(6)
    assert levels.size >= 2
    assert np.all(np.diff(levels) > 0)


def test_interpolation_can_be_cancelled():
    track, _ = synthetic.make(n=2000, path="line", shuffle=False)
    with pytest.raises(InterruptedError):
        interpolate(track, GridParams(bandwidth_m=50.0), cancelled=lambda: True)


def test_interpolation_reports_progress():
    seen = []
    track, _ = synthetic.make(n=1500, path="line", shuffle=False)
    interpolate(track, GridParams(bandwidth_m=60.0), progress=seen.append)
    assert seen and seen[-1] == 1.0


def test_a_wide_latitude_span_warns_about_mercator_distortion():
    track, _ = synthetic.make(n=400, path="line", speed_m_s=3000.0, dt_s=2.0,
                              origin=(30.0, -97.0), shuffle=False)
    grid = interpolate(track, GridParams(bandwidth_m=5000.0))
    assert any("Mercator" in w for w in grid.warnings)


# ---- hotspots ------------------------------------------------------------


def test_no_significant_anomalies_means_no_hotspots():
    track, _ = synthetic.make(n=800, background_cps=18.0, shuffle=False, seed=9)
    bg, found = _analysed(track)
    assert find_hotspots(track, found, bg) == []


def test_a_source_becomes_a_hotspot_at_the_right_place():
    track, truth = synthetic.make(n=1200, path="line", speed_m_s=1.4,
                                  background_cps=18.0, shuffle=False, seed=10,
                                  sources=[(600, 110.0, 6.0)])
    bg, found = _analysed(track)
    hotspots = find_hotspots(track, found, bg)
    assert hotspots
    source = truth["sources"][0]
    assert float(geo.haversine(hotspots[0].lon, hotspots[0].lat,
                               source["lon"], source["lat"])) < 200.0


def test_hotspot_centroids_do_not_move_with_the_grid_bandwidth():
    """The justification for clustering samples rather than grid maxima: a
    grid maximum is a property of the bandwidth and moves when you halve it."""
    track, _ = synthetic.make(n=1200, path="line", speed_m_s=1.4,
                              background_cps=18.0, shuffle=False, seed=10,
                              sources=[(600, 110.0, 6.0)])
    bg, found = _analysed(track)
    coarse = find_hotspots(track, found, bg, bandwidth_m=200.0)
    fine = find_hotspots(track, found, bg, bandwidth_m=50.0)
    assert coarse and fine
    assert float(geo.haversine(coarse[0].lon, coarse[0].lat,
                               fine[0].lon, fine[0].lat)) < 30.0


def test_repeat_visits_are_counted_and_rated_strong():
    """Three passes over the same ground is qualitatively better evidence
    than one, and only that earns the 'strong' label."""
    track, _ = synthetic.make(
        n=1800, path="lawnmower", speed_m_s=1.4, background_cps=18.0,
        shuffle=False, seed=11,
        sources=[(200, 150.0, 4.0), (700, 150.0, 4.0), (1200, 150.0, 4.0)])
    bg, found = _analysed(track)
    hotspots = find_hotspots(track, found, bg, cluster_radius_m=5000.0)
    assert hotspots
    assert hotspots[0].n_passes >= 2
    assert hotspots[0].evidence == "strong"


def test_a_single_pass_source_is_only_possible_not_strong():
    track, _ = synthetic.make(n=1200, path="line", speed_m_s=1.4,
                              background_cps=18.0, shuffle=False, seed=12,
                              sources=[(600, 110.0, 6.0)])
    bg, found = _analysed(track)
    hotspots = find_hotspots(track, found, bg)
    assert hotspots and hotspots[0].evidence == "possible"


def test_hotspots_are_ranked_by_integrated_excess_not_peak():
    """A brief intense spike must not outrank a sustained plateau."""
    track, _ = synthetic.make(
        n=2000, path="line", speed_m_s=1.4, background_cps=18.0,
        shuffle=False, seed=13,
        sources=[(500, 400.0, 0.8), (1500, 60.0, 30.0)])
    bg, found = _analysed(track)
    hotspots = find_hotspots(track, found, bg, cluster_radius_m=300.0)
    assert len(hotspots) >= 2
    assert hotspots[0].net_counts > hotspots[1].net_counts
    assert hotspots[0].score >= hotspots[1].score


# ---- the real file -------------------------------------------------------


def test_sample_yields_no_hotspots(sample):
    """Consistent with the null result: nothing to cluster."""
    track, quality = assess_with_report(sample)
    bg = estimate_background(track, quality=quality)
    found = scan(track, bg, ScanParams(), quality=quality)
    assert find_hotspots(track, found, bg) == []


def test_sample_grid_is_mostly_unsupported(sample):
    """A 1200 km transect through the Southwest surveys a line, not a region.
    Anything but a low coverage figure here would mean the mask had failed."""
    track, quality = assess_with_report(sample)
    grid = interpolate(track, GridParams(bandwidth_m=2000.0, grid_size=96))
    assert grid.coverage < 0.35
    assert np.isnan(grid.values).any()


def test_a_single_anomaly_hotspot_has_the_radius_of_its_samples():
    """Measured on anomaly centres, one anomaly always gave radius 0."""
    import synthetic
    from rcta.core.analysis import AnalysisParams, analyze_track
    track, _ = synthetic.make(n=500, shuffle=False, sources=[(250, 80.0, 4)])
    a = analyze_track(track, AnalysisParams(n_boot=20))
    assert len(a.hotspots) == 1
    hotspot = a.hotspots[0]
    member = a.significant[0]
    span = a.track.distance_along_m[member.stop_index - 1] \
        - a.track.distance_along_m[member.start_index]
    # Roughly half the feature's length either side of its centre.
    assert 0.3 * span < hotspot.radius_m < 1.2 * span
