"""The selection mask, and the disjoint cases that motivate it."""

from __future__ import annotations

import numpy as np
import pytest

from rcta.core.selection import (
    Selection,
    select_by_distance,
    select_by_polygon,
    select_by_time,
    select_by_value,
)
from rcta.core.track import Track


def _track(n=100):
    return Track(time_s=np.arange(n, dtype=float) * 2.0,
                 lat=30.0 + np.arange(n) * 1e-4, lon=np.full(n, -97.0),
                 accuracy_m=np.full(n, 8.0),
                 dose_rate=np.linspace(10.0, 110.0, n),
                 count_rate=np.linspace(2.0, 22.0, n))


def test_empty_and_all():
    assert Selection.empty(10).count == 0
    assert Selection.all(10).count == 10
    assert Selection.empty(10).is_empty


def test_the_mask_is_read_only():
    s = Selection.all(5)
    with pytest.raises(ValueError):
        s.mask[0] = False


def test_from_interval_is_inclusive_and_records_its_provenance():
    t = _track()
    s = Selection.from_interval(t.time_s, 0.0, 10.0)
    assert s.count == 6
    assert s.source == "series"
    assert s.interval == (0.0, 10.0)


def test_from_interval_accepts_reversed_bounds():
    t = _track()
    assert (Selection.from_interval(t.time_s, 50.0, 10.0).count
            == Selection.from_interval(t.time_s, 10.0, 50.0).count)


def test_from_rect_ignores_non_finite_coordinates():
    x = np.array([0.0, 1.0, np.nan])
    y = np.array([0.0, 1.0, 0.5])
    assert Selection.from_rect(x, y, -5, -5, 5, 5).count == 2


# ---- the disjoint case, which is the reason a mask is used ---------------


def test_a_contiguous_selection_is_an_interval():
    mask = np.zeros(20, dtype=bool)
    mask[5:12] = True
    s = Selection(mask=mask)
    assert s.is_interval
    assert s.runs() == [(5, 12)]


def test_a_disjoint_selection_is_not_an_interval():
    """An out-and-back route lassoed once yields two separated runs, which no
    time interval can express."""
    mask = np.zeros(20, dtype=bool)
    mask[2:5] = True
    mask[14:18] = True
    s = Selection(mask=mask)
    assert not s.is_interval
    assert s.runs() == [(2, 5), (14, 18)]
    assert "2 intervals" in s.describe()


def test_runs_are_capped_so_a_panel_cannot_be_flooded():
    mask = np.zeros(2000, dtype=bool)
    mask[::2] = True
    assert len(Selection(mask=mask).runs(limit=64)) == 64


def test_time_span_is_the_hull_and_not_the_selection():
    """Documents the difference the series panel has to respect."""
    t = _track()
    mask = np.zeros(t.n, dtype=bool)
    mask[0] = True
    mask[-1] = True
    s = Selection(mask=mask)
    assert s.time_span(t.time_s) == (0.0, float(t.time_s[-1]))
    assert s.count == 2, "the hull spans the track; the selection is two samples"


def test_time_span_of_nothing_is_none():
    assert Selection.empty(10).time_span(np.arange(10.0)) is None


# ---- set operations ------------------------------------------------------


def test_intersection_union_and_difference():
    a = Selection(mask=np.array([True, True, False, False]))
    b = Selection(mask=np.array([False, True, True, False]))
    assert (a & b).mask.tolist() == [False, True, False, False]
    assert (a | b).mask.tolist() == [True, True, True, False]
    assert a.without(b).mask.tolist() == [True, False, False, False]
    assert (~a).mask.tolist() == [False, False, True, True]


def test_set_operations_drop_the_interval_provenance():
    """After combining, the recorded interval no longer describes the mask."""
    t = _track()
    a = Selection.from_interval(t.time_s, 0.0, 20.0)
    b = Selection.from_interval(t.time_s, 10.0, 40.0)
    assert (a & b).interval is None


# ---- selectors -----------------------------------------------------------


def test_select_by_time_and_distance():
    t = _track()
    assert select_by_time(t, 0.0, 20.0).count == 11
    total = float(t.distance_along_m[-1])
    assert 0 < select_by_distance(t, 0.0, total / 2).count < t.n


def test_select_by_value_combines_constraints():
    t = _track()
    both = select_by_value(t, dose_range=(50.0, 80.0), count_range=(0.0, 14.0))
    dose_only = select_by_value(t, dose_range=(50.0, 80.0))
    assert both.count <= dose_only.count


def test_select_by_value_can_filter_on_accuracy():
    t = _track()
    accuracy = t.accuracy_m.copy()
    accuracy[:10] = 500.0
    t = Track(time_s=t.time_s, lat=t.lat, lon=t.lon, accuracy_m=accuracy,
              dose_rate=t.dose_rate, count_rate=t.count_rate)
    assert select_by_value(t, accuracy_max=100.0).count == t.n - 10


def test_polygon_selects_what_is_inside_it():
    t = _track()
    x, y = t.merc_x, t.merc_y
    pad = 1.0
    poly = np.array([[x.min() - pad, y.min() - pad],
                     [x.max() + pad, y.min() - pad],
                     [x.max() + pad, y[t.n // 2]],
                     [x.min() - pad, y[t.n // 2]]])
    s = select_by_polygon(t, poly)
    assert 0 < s.count < t.n
    assert s.mask[0] and not s.mask[-1]


def test_polygon_excludes_position_masked_samples():
    """A 397 m fix has no business being lassoed into a 50 m polygon."""
    t = _track()
    x, y = t.merc_x, t.merc_y
    poly = np.array([[x.min() - 1, y.min() - 1], [x.max() + 1, y.min() - 1],
                     [x.max() + 1, y.max() + 1], [x.min() - 1, y.max() + 1]])
    masked = t.with_mask(position=np.zeros(t.n, dtype=bool))
    assert select_by_polygon(t, poly).count == t.n
    assert select_by_polygon(masked, poly).count == 0


def test_a_degenerate_polygon_raises():
    with pytest.raises(ValueError, match="at least 3 vertices"):
        select_by_polygon(_track(), np.array([[0.0, 0.0], [1.0, 1.0]]))
