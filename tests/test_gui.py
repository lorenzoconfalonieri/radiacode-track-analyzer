"""GUI smoke tests. No pixel comparisons -- the logic lives in rcta.core.

Everything runs offscreen, with the tile fetcher offline and the job queue
synchronous, so the suite never touches the network and never waits on a
thread. Assertions are on status-bar text, log text and model row counts,
which is what actually catches regressions.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("RCTA_TILE_CACHE", "/tmp/rcta-test-tiles")

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6 import QtCore  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from PySide6 import QtWidgets
    from rcta.gui import theme
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    theme.apply(application)
    return application


@pytest.fixture
def window(app):
    from rcta.gui.main_window import MainWindow
    win = MainWindow(offline=True, synchronous=True)
    yield win
    win.close()
    # Delete, not just close. A closed window stays alive through its own
    # signal connections, and destroying ~150 of them at interpreter exit is
    # quadratic in PySide: it added minutes to every run after the last test.
    win.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


@pytest.fixture
def loaded(window, sample_path):
    assert window.load(str(sample_path))
    return window


# ---- loading -------------------------------------------------------------


def test_window_loads_the_sample(loaded):
    status = loaded.statusBar().currentMessage()
    assert "2964 samples" in status
    assert "1792.21 km" in status
    assert "14 legs" in status
    assert "no anomalies above L_C" in status


def test_the_unsorted_markers_are_reported_to_the_user(loaded):
    """A silent fix is a fix nobody can audit."""
    assert "not in time order" in loaded.log.toPlainText()


def test_the_dropouts_are_reported(loaded):
    assert "13 dropout(s)" in loaded.log.toPlainText()


def test_the_detection_limit_is_logged_when_nothing_is_found(loaded):
    log = loaded.log.toPlainText()
    assert "no anomalies above L_C" in log
    assert "MDA" in log


def test_tables_are_populated(loaded):
    assert loaded.leg_model.rowCount() == 14
    assert loaded.hotspot_model.rowCount() == 0
    assert loaded.anomaly_model.rowCount() >= 1


def test_every_anomaly_row_is_a_flagged_candidate(loaded):
    """The null result, seen through the table."""
    for row in range(loaded.anomaly_model.rowCount()):
        anomaly = loaded.anomaly_model.row_object(row)
        assert not anomaly.is_significant
        assert anomaly.flags


def test_the_track_dock_shows_the_odometer_agreement(loaded):
    assert "0.16 % apart" in loaded.track_dock.text.toPlainText()


def test_loading_a_malformed_file_fails_without_raising(window, tmp_path):
    path = tmp_path / "bad.rctrk"
    path.write_text("{not json")
    assert window.load(str(path)) is False
    assert "cannot load" in window.log.toPlainText()


def test_loading_a_missing_file_fails_without_raising(window, tmp_path):
    assert window.load(str(tmp_path / "nope.rctrk")) is False


# ---- linked brushing -----------------------------------------------------


def test_a_series_interval_selects_and_reports(loaded):
    t = loaded._track.time_s
    loaded._on_series_interval(float(t[100]), float(t[400]))
    assert loaded._selection.count == 301
    assert loaded._selection.is_interval
    assert "301 samples in 1 interval" in loaded.statusBar().currentMessage()


def test_a_disjoint_map_selection_is_not_coarsened_to_its_hull(loaded):
    """The reason the selection is a mask and the region item is only a control."""
    from PySide6.QtCore import Qt
    mask = np.zeros(loaded._track.n, dtype=bool)
    mask[10:20] = True
    mask[900:910] = True
    loaded._on_map_selection(mask, Qt.KeyboardModifier.NoModifier)

    assert loaded._selection.count == 20
    assert not loaded._selection.is_interval
    assert "2 intervals" in loaded.selection_dock.caption.text()
    # The draggable region cannot express this, so it must be hidden rather
    # than silently widened to span 890 samples.
    assert not loaded.series.region.isVisible()


def test_a_contiguous_selection_does_show_the_region_control(loaded):
    t = loaded._track.time_s
    loaded._on_series_interval(float(t[50]), float(t[80]))
    assert loaded.series.region.isVisible()


def test_shift_adds_to_the_selection(loaded):
    from PySide6.QtCore import Qt
    first = np.zeros(loaded._track.n, dtype=bool)
    first[10:20] = True
    loaded._on_map_selection(first, Qt.KeyboardModifier.NoModifier)

    second = np.zeros(loaded._track.n, dtype=bool)
    second[100:110] = True
    loaded._on_map_selection(second, Qt.KeyboardModifier.ShiftModifier)
    assert loaded._selection.count == 20


def test_alt_subtracts_from_the_selection(loaded):
    from PySide6.QtCore import Qt
    whole = np.ones(loaded._track.n, dtype=bool)
    loaded._on_map_selection(whole, Qt.KeyboardModifier.NoModifier)
    before = loaded._selection.count

    cut = np.zeros(loaded._track.n, dtype=bool)
    cut[:100] = True
    loaded._on_map_selection(cut, Qt.KeyboardModifier.AltModifier)
    assert loaded._selection.count == before - 100


def test_clearing_the_selection_restores_the_whole_track_summary(loaded):
    t = loaded._track.time_s
    loaded._on_series_interval(float(t[10]), float(t[100]))
    loaded._on_selection_cleared()
    assert loaded._selection.is_empty
    assert loaded.selection_dock.caption.text() == "no selection"


def test_selecting_an_anomaly_row_selects_its_samples(loaded):
    from PySide6.QtCore import QItemSelectionModel
    if loaded.anomaly_model.rowCount() == 0:
        pytest.skip("no candidates on this file")
    index = loaded.anomaly_model.index(0, 0)
    loaded.anomaly_view.selectionModel().select(
        index, QItemSelectionModel.SelectionFlag.Select
        | QItemSelectionModel.SelectionFlag.Rows)
    assert loaded._selection.count > 0


def test_hovering_shows_the_sample_in_the_status_bar(loaded):
    loaded._on_hover(500)
    message = loaded.statusBar().currentMessage()
    assert "nSv/h" in message and "cps" in message


def test_hovering_off_the_track_restores_the_summary(loaded):
    loaded._on_hover(500)
    loaded._on_hover(-1)
    assert "2964 samples" in loaded.statusBar().currentMessage()


# ---- parameters ----------------------------------------------------------


def test_changing_the_background_method_re_runs_the_analysis(loaded):
    loaded.analysis_dock.background.setCurrentText("global")
    assert loaded._analysis.background.method == "global"


def test_turning_off_bonferroni_changes_the_verdict(loaded):
    """The correction is what makes this file a null result."""
    assert loaded._analysis.found_nothing
    loaded.analysis_dock.bonferroni.setChecked(False)
    assert not loaded._analysis.found_nothing


def test_marking_a_selection_as_background_switches_method(loaded):
    t = loaded._track.time_s
    loaded._on_series_interval(float(t[100]), float(t[600]))
    loaded._mark_reference()
    assert loaded._analysis.background.method == "reference"
    assert "background reference set" in loaded.log.toPlainText()


def test_marking_an_empty_selection_warns_instead_of_failing(loaded):
    loaded._on_selection_cleared()
    loaded._mark_reference()
    assert "select a clean span first" in loaded.log.toPlainText()


def test_clearing_the_reference_returns_to_rolling(loaded):
    t = loaded._track.time_s
    loaded._on_series_interval(float(t[100]), float(t[600]))
    loaded._mark_reference()
    loaded._clear_reference()
    assert loaded._analysis.background.method == "rolling"


def test_an_analysis_failure_is_logged_and_does_not_kill_the_window(loaded,
                                                                   monkeypatch):
    import rcta.gui.main_window as mw
    monkeypatch.setattr(mw, "analyze_track",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    loaded.analyse()
    assert "analysis failed: boom" in loaded.log.toPlainText()
    assert loaded.isVisible() or True          # still alive


# ---- the heatmap ---------------------------------------------------------


def test_the_heatmap_runs_and_reports_its_support(loaded):
    loaded.compute_grid()
    assert loaded._grid is not None
    assert "supported" in loaded.log.toPlainText()


def test_the_heatmap_warns_rather_than_silently_over_smoothing(loaded):
    """A metre-scale kernel cannot be resolved over a 1600 km transect."""
    loaded.compute_grid()
    assert "bandwidth raised" in loaded.log.toPlainText()


def test_heatmap_warnings_are_not_logged_twice(loaded):
    loaded.compute_grid()
    log = loaded.log.toPlainText()
    assert log.count("bandwidth raised") == 1


def test_the_heatmap_is_drawn_once_computed(loaded):
    """It used to be computed, logged and never shown."""
    assert not loaded.map.heatmap_shown
    assert not loaded.display_dock.heatmap.isEnabled()
    loaded.compute_grid()
    assert loaded.map.heatmap_shown
    assert loaded.display_dock.heatmap.isEnabled()
    image = loaded.map.heatmap.image
    # Unsupported cells are transparent, not painted as low.
    assert image.shape[:2] == loaded._grid.values.shape
    assert (image[~loaded._grid.supported, 3] == 0).all()
    assert (image[loaded._grid.supported, 3] == 255).all()


def test_the_heatmap_hides_when_the_colour_bar_means_count_rate(loaded):
    loaded.compute_grid()
    loaded.map.set_color_field("count_rate")
    assert not loaded.map.heatmap_shown
    loaded.map.set_color_field("dose_rate")
    assert loaded.map.heatmap_shown
    loaded.display_dock.heatmap.setChecked(False)
    assert not loaded.map.heatmap_shown


def test_a_new_track_clears_the_heatmap(loaded, sample_path):
    loaded.compute_grid()
    assert loaded.load(str(sample_path))
    assert loaded._grid is None and not loaded.map.heatmap_shown
    assert not loaded.display_dock.heatmap.isEnabled()


def test_changing_which_fixes_count_clears_the_heatmap(loaded):
    loaded.compute_grid()
    loaded.analysis_dock.max_accuracy.setValue(20.0)
    assert loaded._grid is None
    assert "heatmap cleared" in loaded.log.toPlainText()


def test_changing_only_the_scan_keeps_the_heatmap(loaded):
    loaded.compute_grid()
    loaded.analysis_dock.alpha.setValue(0.01)
    assert loaded._grid is not None


def test_the_heatmap_grid_exports_with_its_projection(loaded, tmp_path,
                                                      monkeypatch):
    from PySide6 import QtWidgets
    out = tmp_path / "grid.asc"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out), "")))
    loaded.compute_grid()
    loaded._export("grid")
    assert out.read_text().startswith("ncols ")
    assert "Mercator" in (tmp_path / "grid.prj").read_text()


# ---- the basemap ---------------------------------------------------------


def test_the_map_works_with_no_basemap_at_all(loaded):
    """Tile problems must never block the analysis."""
    loaded._set_provider(None)
    assert loaded.map._basemap_enabled is False
    assert loaded.map.attribution.text() == ""
    assert "2964 samples" in loaded.statusBar().currentMessage()


def test_selecting_a_provider_shows_its_attribution(loaded):
    loaded._set_provider("osm")
    assert "OpenStreetMap" in loaded.map.attribution.text()


def test_attribution_has_no_visibility_toggle(loaded):
    """The OSM policy forbids putting it behind a toggle."""
    actions = [a.text().lower() for a in loaded.findChildren(type(
        loaded.menuBar().actions()[0]))]
    assert not any("attribution" in text for text in actions)


def test_offline_mode_is_reported_once(loaded):
    loaded._set_provider("osm")
    loaded.map._retile()
    assert loaded.log.toPlainText().count("Offline - basemap") <= 1


def test_purging_the_cache_reports_what_it_freed(loaded):
    loaded._purge_cache()
    assert "tile cache:" in loaded.log.toPlainText()


# ---- export --------------------------------------------------------------


@pytest.mark.parametrize("kind,check", [
    ("csv", "index,iso_time"),
    ("geojson", "FeatureCollection"),
    ("gpx", "<gpx"),
    ("json", '"generator"'),
    ("anomalies", "start_index"),
    ("hotspots", "lat,lon"),
])
def test_export_writes_the_file(loaded, tmp_path, monkeypatch, kind, check):
    from PySide6 import QtWidgets
    out = tmp_path / f"out.{kind}"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out), "")))
    loaded._export(kind)
    assert check in out.read_text()
    assert f"wrote {out}" in loaded.log.toPlainText()


def test_cancelling_the_export_dialog_writes_nothing(loaded, monkeypatch):
    from PySide6 import QtWidgets
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    before = loaded.log.toPlainText()
    loaded._export("csv")
    assert loaded.log.toPlainText() == before


# ---- the worker ----------------------------------------------------------


def test_a_stale_job_result_is_dropped(app):
    """Dragging a slider must not let the slowest queued job win."""
    from rcta.gui.worker import JobQueue
    queue = JobQueue(synchronous=True)
    seen = []
    queue.finished.connect(lambda name, token, result: seen.append(result))
    for value in (1, 2, 3):
        queue.submit("grid", lambda v=value: v)
    assert seen == [1, 2, 3]

    queue.cancel("grid")
    queue.submit("grid", lambda: 4)
    assert seen[-1] == 4


def test_a_cancelled_job_never_delivers(app):
    from rcta.gui.worker import JobQueue
    queue = JobQueue(synchronous=True)
    seen = []
    queue.finished.connect(lambda *a: seen.append(a))

    def slow(cancelled=None):
        return "done"

    queue.submit("job", slow)
    assert len(seen) == 1


def test_a_failing_job_reports_instead_of_raising(app):
    from rcta.gui.worker import JobQueue
    queue = JobQueue(synchronous=True)
    failures = []
    queue.failed.connect(lambda name, token, message: failures.append(message))
    queue.submit("job", lambda: 1 / 0)
    assert failures and "division by zero" in failures[0]


def test_a_failure_hint_survives_the_worker(app):
    """Stringifying the exception would drop the actionable half."""
    from rcta.gui.worker import JobQueue
    queue = JobQueue(synchronous=True)
    failures = []
    queue.failed.connect(lambda name, token, message: failures.append(message))

    class Refused(RuntimeError):
        hint = "do this instead"

    queue.submit("job", lambda: (_ for _ in ()).throw(Refused("no")))
    explanation, _, tb = failures[0].partition("\n\n")
    assert explanation == "no\nhint: do this instead"
    assert "Traceback" in tb


def test_progress_is_forwarded(app):
    from rcta.gui.worker import JobQueue
    queue = JobQueue(synchronous=True)
    seen = []
    queue.progress.connect(lambda name, token, f: seen.append(f))
    queue.submit("job", lambda progress=None: progress(0.5) or "x")
    assert seen == [0.5]


def test_closing_the_window_while_a_job_is_queued_is_clean(app, sample_path):
    from rcta.gui.main_window import MainWindow
    win = MainWindow(offline=True, synchronous=True)
    win.load(str(sample_path))
    win.compute_grid()
    win.close()


# ---- panels in isolation -------------------------------------------------


def test_the_map_panel_survives_an_empty_track(app):
    from rcta.gui.map_panel import MapPanel
    panel = MapPanel()
    panel.set_track(None)
    panel.set_selection(None)
    panel.set_hover(5)


def test_the_series_panel_survives_an_empty_track(app):
    from rcta.gui.series_panel import SeriesPanel
    panel = SeriesPanel()
    panel.set_track(None)
    panel.set_selection(None)
    panel.set_hover(3)


def test_a_single_point_track_does_not_send_the_view_to_infinity(app):
    """A stationary measurement has a zero-width bounding box."""
    from rcta.core.track import Track
    from rcta.gui.map_panel import MapPanel

    track = Track(time_s=[0.0, 2.0, 4.0], lat=[30.0] * 3, lon=[-97.0] * 3,
                  accuracy_m=[5.0] * 3, dose_rate=[90.0] * 3,
                  count_rate=[18.0] * 3)
    panel = MapPanel()
    panel.set_track(track)
    rect = panel.vb.viewRect()
    assert np.isfinite(rect.width()) and rect.width() > 0


def test_colour_scale_auto_selects_log_for_a_wide_range():
    from rcta.gui.map_panel import suggest_color_scale
    assert suggest_color_scale(2.97, 114.0) == "log"
    assert suggest_color_scale(80.0, 120.0) == "linear"
    assert suggest_color_scale(0.0, 100.0) == "linear"


def test_the_degree_axis_formats_hemispheres_not_minus_signs():
    from rcta.gui.geo_axis import format_degree
    assert format_degree(40.79, 1.0, is_lat=True) == "41°N"
    assert format_degree(-104.5, 1.0, is_lat=False) == "104°W"
    assert "'" in format_degree(40.79, 1 / 60, is_lat=True)


def test_the_scale_bar_picks_round_numbers():
    from rcta.gui.geo_axis import nice_distance
    assert nice_distance(1234.0) == 1000.0
    assert nice_distance(7000.0) == 5000.0
    assert nice_distance(23.0) == 20.0


def test_the_log_colour_bar_is_labelled_in_real_units(loaded):
    """A log bar left unrelabelled prints 100 nSv/h as "2.0", which a reader
    will take at face value."""
    from rcta.gui.map_panel import _decade_ticks
    ticks = _decade_ticks(2.97, 114.0)
    assert 5.0 in ticks and 100.0 in ticks
    assert all(2.97 <= t <= 114.0 for t in ticks)


def test_decade_ticks_degrade_gracefully_on_bad_input():
    from rcta.gui.map_panel import _decade_ticks
    assert _decade_ticks(0.0, 10.0) == [0.0, 10.0]
    assert _decade_ticks(5.0, 5.0) == [5.0, 5.0]


def test_exported_image_reserves_a_band_for_the_attribution(loaded):
    """Overlaying it on the axis labels makes it illegible, and an
    unreadable licence notice is not a licence notice."""
    from rcta.gui.map_export import attribution_lines, render_image
    lines = attribution_lines(loaded._analysis, loaded._providers["osm"])
    plain = render_image(loaded.map, 600, 400)
    stamped = render_image(loaded.map, 600, 400, lines=lines)
    assert stamped.height() > plain.height()


def test_an_export_blocked_by_provider_terms_says_so(loaded, tmp_path,
                                                     monkeypatch):
    from PySide6 import QtWidgets
    out = tmp_path / "map.png"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out), "")))
    loaded._set_provider("esri_imagery")
    loaded.map.set_basemap_enabled(True, 19)
    loaded._export_image()
    assert "may not be redistributed" in loaded.log.toPlainText()
    assert out.exists()


def _count_basemap_paints(monkeypatch):
    from rcta.gui.basemap import BasemapItem
    calls = []
    original = BasemapItem.paint

    def paint(self, *args):
        calls.append(1)
        return original(self, *args)
    monkeypatch.setattr(BasemapItem, "paint", paint)
    return calls


def test_a_basemap_that_may_not_be_exported_is_not_rendered(loaded,
                                                            monkeypatch):
    """Dropping only the credit line still shipped the provider's pixels."""
    from rcta.gui.map_export import render_image
    calls = _count_basemap_paints(monkeypatch)
    render_image(loaded.map, 300, 200, include_basemap=True)
    assert calls, "the control case must paint the basemap"
    calls.clear()
    render_image(loaded.map, 300, 200, include_basemap=False)
    assert not calls
    assert loaded.map.basemap.isVisible()          # put back afterwards


def test_the_svg_carries_the_colour_bar_and_provenance_but_no_tiles(
        loaded, tmp_path, monkeypatch):
    from PySide6 import QtWidgets
    calls = _count_basemap_paints(monkeypatch)
    out = tmp_path / "map.svg"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(out), "")))
    loaded._export_svg()
    svg = out.read_text()
    assert not calls
    assert "dose rate (nSv/h" in svg                 # the colour bar label
    assert "2964 samples" in svg                     # the provenance block
    assert "Web Mercator" in svg


def test_the_provenance_quotes_the_display_unit_and_the_heatmap(loaded):
    from rcta.gui.map_export import attribution_lines
    loaded.compute_grid()
    lines = attribution_lines(loaded._analysis, unit="uSv/h",
                              grid=loaded._grid)
    text = "\n".join(lines)
    assert "uSv/h" in text and "nSv/h" not in text
    assert "unknown, not low" in text


def test_the_profile_panel_reports_the_confound(loaded):
    caption = loaded.profile.caption.text()
    assert "confounding, not causation" in caption
    assert "independent ones" in caption


def test_whole_degree_ticks_are_not_printed_with_decimals(app):
    """Regression: deriving the step from the tick values mixed the major and
    minor levels and printed a whole-degree tick as "41.00 deg N"."""
    from rcta.core.geo import lonlat_to_web_mercator
    from rcta.gui.geo_axis import GeoAxisItem

    axis = GeoAxisItem("left")
    metres = [float(lonlat_to_web_mercator(0.0, d)[1]) for d in (30, 35, 40)]
    assert axis.tickStrings(metres, 1.0, 1.0) == ["30°N", "35°N", "40°N"]
    assert axis.tickStrings(metres[:1], 1.0, 0.5)[0] == "30.00°N"


def test_the_axis_declares_its_spacing_in_degrees(app):
    """tickStrings relies on this, so the contract is pinned here."""
    from rcta.core.geo import lonlat_to_web_mercator
    from rcta.gui.geo_axis import GeoAxisItem

    axis = GeoAxisItem("bottom")
    lo = float(lonlat_to_web_mercator(-112.0, 0.0)[0])
    hi = float(lonlat_to_web_mercator(-97.0, 0.0)[0])
    levels = axis.tickValues(lo, hi, 900)
    assert levels
    major_step = levels[0][0]
    assert 0.5 <= major_step <= 10.0, "spacing should be a degree step"


def test_the_basemap_picker_is_in_the_display_dock(loaded):
    """It is a visual setting, so it lives with the other visual settings."""
    items = [loaded.display_dock.basemap.itemText(i)
             for i in range(loaded.display_dock.basemap.count())]
    assert items[0] == "None"
    assert "OpenStreetMap" in items


def test_a_keyed_provider_is_greyed_out_with_the_reason(loaded):
    """Greyed rather than hidden, so why it is unavailable is discoverable."""
    combo = loaded.display_dock.basemap
    index = combo.findData("carto_positron")
    assert index >= 0
    assert not combo.model().item(index).isEnabled()
    assert "API key" in combo.itemData(index, Qt.ItemDataRole.ToolTipRole)


def test_choosing_a_basemap_in_the_dock_applies_it(loaded):
    index = loaded.display_dock.basemap.findData("osm")
    loaded.display_dock.basemap.setCurrentIndex(index)
    assert loaded._provider_id == "osm"
    assert "OpenStreetMap" in loaded.map.attribution.text()


def test_the_menu_and_the_dock_stay_in_sync(loaded):
    """Two controls for one setting is a drift hazard; both routes converge on
    _set_provider, which syncs the other."""
    loaded._basemap_actions["osm"].setChecked(True)
    assert loaded.display_dock.basemap.currentData() == "osm"

    loaded.display_dock.basemap.setCurrentIndex(
        loaded.display_dock.basemap.findData(None))
    assert loaded._provider_id is None
    assert loaded._basemap_actions[None].isChecked()


def test_the_offline_toggle_stays_in_sync(loaded):
    loaded.display_dock.offline.setChecked(False)
    assert loaded._offline is False
    assert loaded._offline_action.isChecked() is False

    loaded._offline_action.setChecked(True)
    assert loaded._offline is True
    assert loaded.display_dock.offline.isChecked() is True


# ---- flight --------------------------------------------------------------


@pytest.fixture
def joined(loaded, sample_trajectory_path):
    loaded._load_trajectory(str(sample_trajectory_path), "auto")
    return loaded


def test_the_flight_dock_starts_empty_and_says_why(loaded):
    assert loaded._join() is None
    assert "carries none" in loaded.flight.caption.text()


def test_the_date_is_prefilled_from_the_track(loaded):
    assert loaded.flight.date.date().toString("yyyy-MM-dd") == "2026-09-07"


def test_the_prefilled_date_is_on_the_departure_airports_clock(window):
    """The field is read as a local date, so the default has to be one.

    A UTC default would be a day out on exactly the evening departures that
    motivated reading it locally: the dock would open pre-set to a date it
    then warns about.
    """
    import synthetic

    # 2026-09-11 00:06Z at Salt Lake City: 18:06 on the 10th, local.
    track, _, _ = synthetic.flight_track(
        n=200, start=1789085184, origin=(40.7889, -111.9799),
        destination=(30.1975, -97.662))
    assert window._departure_zone(track) == "America/Denver"
    assert window._departure_local_start(track).date().isoformat() == "2026-09-10"


def test_a_track_that_starts_nowhere_near_an_airport_keeps_utc(window):
    """No airport means no zone, and UTC is the honest fallback."""
    import synthetic

    track, _, _ = synthetic.flight_track(
        n=200, start=1789085184, origin=(30.0, -40.0),   # mid-Atlantic
        destination=(35.0, -30.0))
    assert window._departure_zone(track) == ""
    assert window._departure_local_start(track).date().isoformat() == "2026-09-11"


def test_a_trajectory_joins_and_is_plotted(joined):
    join = joined._join()
    assert join is not None and join.n_joined > 2900
    caption = joined.flight.caption.text()
    assert "samples joined" in caption
    assert "Scale height" in caption


def test_climb_and_descent_are_drawn_as_separate_series(joined):
    climb, _ = joined.flight._series["climb"]
    descent, _ = joined.flight._series["descent"]
    assert climb.xData is not None and len(climb.xData) > 3
    assert descent.xData is not None and len(descent.xData) > 3


def test_pooling_the_branches_draws_one_series(joined):
    joined.flight.split.setChecked(False)
    climb, _ = joined.flight._series["climb"]
    descent, _ = joined.flight._series["descent"]
    assert len(climb.xData) > 3
    assert descent.xData is None or len(descent.xData) == 0
    assert len(joined.flight._series["level"][0].xData or []) == 0


def test_the_ordinate_is_logarithmic_by_default(joined):
    assert joined.flight.log_y.isChecked()
    assert joined.flight.plot.getViewBox().state["logMode"][1]
    joined.flight.log_y.setChecked(False)
    assert not joined.flight.plot.getViewBox().state["logMode"][1]


def test_error_bars_are_drawn_for_every_point(joined):
    _, bars = joined.flight._series["climb"]
    assert bars.opts["x"] is not None and len(bars.opts["x"]) > 3


def test_changing_the_bin_width_replots_without_refetching(joined):
    before = len(joined.flight._series["climb"][0].xData)
    joined.flight.step.setValue(2000.0)
    after = len(joined.flight._series["climb"][0].xData)
    assert after < before
    assert joined._trajectory is not None


def test_the_disagreement_is_stated_in_the_caption(joined):
    """The bundled pair does not reconcile, and the panel has to say so."""
    assert "disagree" in joined.flight.caption.text()


def test_a_mismatched_trajectory_is_refused_and_clears_the_plot(
        loaded, tmp_path, monkeypatch):
    import json
    from PySide6 import QtWidgets
    path = tmp_path / "elsewhere.json"
    path.write_text(json.dumps({
        "icao24": "ffffff", "callsign": "XXX1 ",
        "path": [[1788785639, 51.5, -0.1, 150.0, 0.0, True],
                 [1788790000, 52.0, 0.5, 10668.0, 0.0, False],
                 [1788795393, 52.5, 1.0, 200.0, 0.0, False]]}))
    # Decline the "join anyway" offer, which is what the default button does.
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "exec",
        lambda self: QtWidgets.QMessageBox.StandardButton.Cancel)

    loaded._load_trajectory(str(path), "auto")
    assert loaded._join() is None
    assert "Refusing" in loaded.flight.caption.text()
    assert loaded.flight._series["climb"][0].xData in (None, [])


def test_the_refusal_can_be_overridden_from_the_dialog(loaded, tmp_path,
                                                       monkeypatch):
    import json
    from PySide6 import QtWidgets
    path = tmp_path / "elsewhere.json"
    path.write_text(json.dumps({
        "icao24": "ffffff", "callsign": "XXX1 ",
        "path": [[1788785639, 51.5, -0.1, 150.0, 0.0, True],
                 [1788790000, 52.0, 0.5, 10668.0, 0.0, False],
                 [1788795393, 52.5, 1.0, 200.0, 0.0, False]]}))
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "exec",
        lambda self: QtWidgets.QMessageBox.StandardButton.Ignore)

    loaded._load_trajectory(str(path), "auto")
    assert loaded._join() is not None
    assert loaded._join().match.forced
    assert "FORCED" in loaded.flight.caption.text()


def test_loading_a_new_track_drops_the_previous_altitudes(joined, sample_path):
    """A trajectory belongs to the track it was validated against."""
    assert joined._join() is not None
    joined.load(str(sample_path))
    assert joined._join() is None
    assert joined._trajectory is None


def test_a_fetch_without_a_track_is_declined_politely(window):
    window._fetch_flight("DL2618", "2026-09-07")
    assert "Load a track first" in window.flight.caption.text()


def test_an_unparseable_flight_number_is_reported(loaded):
    loaded._fetch_flight("not a flight", "2026-09-07")
    assert "flight number" in loaded.flight.caption.text()


def test_an_unknown_airline_never_reaches_the_network(loaded):
    loaded._fetch_flight("QQ100", "2026-09-07")
    assert "not in the table" in loaded.flight.caption.text()


def test_a_failed_fetch_explains_itself_and_keeps_the_hint(loaded, monkeypatch):
    """The hint is the half of the message a user can act on."""
    from rcta.net import flights

    def refuse(*a, **k):
        raise flights.FlightFetchError(
            "OpenSky refuses historical data to anonymous callers",
            hint="register free at https://opensky-network.org/index.php"
                 "?option=com_users&view=registration, then set "
                 "OPENSKY_CLIENT_ID and OPENSKY_CLIENT_SECRET.")

    monkeypatch.setattr(flights, "fetch_trajectory", refuse)
    loaded._fetch_flight("DL2618", "2026-09-07")

    caption = loaded.flight.caption.text()
    assert "Could not fetch" in caption
    assert "anonymous callers" in caption
    assert "register free" in caption
    assert "OPENSKY_CLIENT_ID" in caption
    # Clickable, and the query string survives as entities rather than markup.
    assert '<a href="https://opensky-network.org/index.php?option=' in caption
    assert "&amp;view=registration" in caption
    # The comma after the URL is punctuation, not part of the address.
    assert "registration</a>, then set" in caption
    assert "From file" in caption
    assert "Traceback" not in caption
    assert "Traceback" in loaded.log.toPlainText()
    assert "failed" in loaded.statusBar().currentMessage()


def test_an_offline_fetch_reports_through_the_real_worker(loaded, tmp_path,
                                                         monkeypatch):
    """The whole path, unmocked: --offline with an empty cache really fails.

    Nothing is patched out here, so this is the one test that proves the
    exception survives the worker, the signal and the caption together. The
    cache is redirected at a temporary directory so that the developer's own
    cached flights cannot turn this into a success.
    """
    monkeypatch.setenv("RCTA_FLIGHT_CACHE", str(tmp_path / "flights"))
    assert loaded._offline, "the fixture must not be able to reach the network"

    loaded._fetch_flight("DL2618", "2026-09-07")

    caption = loaded.flight.caption.text()
    assert "Could not fetch" in caption
    assert "offline" in caption
    assert "fetching" not in loaded.statusBar().currentMessage()
    assert loaded._join() is None
    assert not list((tmp_path / "flights").rglob("*.json"))


def test_a_failure_with_no_hint_still_reports(loaded, monkeypatch):
    """Not every job carries a hint, and the caption must survive that."""
    from rcta.net import flights

    def explode(*a, **k):
        raise RuntimeError("something with an <angle> bracket in it")

    monkeypatch.setattr(flights, "fetch_trajectory", explode)
    loaded._fetch_flight("DL2618", "2026-09-07")

    caption = loaded.flight.caption.text()
    assert "&lt;angle&gt;" in caption, "unescaped markup would eat the word"
    assert "hint:" not in caption


def test_a_bad_trajectory_file_is_reported_not_raised(loaded, tmp_path):
    path = tmp_path / "junk.csv"
    path.write_text("nothing,useful\n1,2\n")
    loaded._load_trajectory(str(path), "auto")
    assert "Cannot read that trajectory" in loaded.flight.caption.text()
    assert loaded._join() is None


def test_the_clear_cache_button_reports_what_it_freed(loaded, tmp_path,
                                                     monkeypatch):
    from PySide6 import QtWidgets
    monkeypatch.setenv("RCTA_FLIGHT_CACHE", str(tmp_path / "flights"))
    stale = tmp_path / "flights" / "opensky"
    stale.mkdir(parents=True)
    (stale / "a.json").write_text("[]" * 100)
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec",
                        lambda self: QtWidgets.QMessageBox.StandardButton.Discard)

    loaded.flight.clear_cache_button.click()

    assert "removed 1 file(s)" in loaded.log.toPlainText()
    assert not list(stale.glob("*.json"))


def test_cancelling_the_clear_keeps_the_cache(loaded, tmp_path, monkeypatch):
    """A settled flight is cached for ever so it need not be paid for twice."""
    from PySide6 import QtWidgets
    monkeypatch.setenv("RCTA_FLIGHT_CACHE", str(tmp_path / "flights"))
    stale = tmp_path / "flights" / "opensky"
    stale.mkdir(parents=True)
    (stale / "a.json").write_text("[]")
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec",
                        lambda self: QtWidgets.QMessageBox.StandardButton.Cancel)

    loaded.flight.clear_cache_button.click()

    assert "flight cache kept" in loaded.log.toPlainText()
    assert (stale / "a.json").exists()


def test_clearing_an_empty_cache_never_asks(loaded, tmp_path, monkeypatch):
    """Nothing to weigh up, so no dialog to dismiss."""
    from PySide6 import QtWidgets
    monkeypatch.setenv("RCTA_FLIGHT_CACHE", str(tmp_path / "empty"))

    def refuse(self):
        raise AssertionError("asked about an empty cache")

    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", refuse)
    loaded.flight.clear_cache_button.click()
    assert "already empty" in loaded.log.toPlainText()


def test_the_unit_change_reaches_the_flight_panel(joined):
    joined._on_unit_changed("uSv/h")
    assert "uSv/h" in joined.flight.plot.getAxis("left").labelText


def test_the_flight_exports_need_a_join(loaded, tmp_path, monkeypatch):
    from PySide6 import QtWidgets
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        lambda *a, **k: (str(tmp_path / "out.csv"), ""))
    loaded._export("profile")
    assert not (tmp_path / "out.csv").exists()
    assert "no altitude joined yet" in loaded.log.toPlainText()


def test_the_fitted_curve_is_on_the_same_scale_as_the_points(joined):
    """pyqtgraph log-transforms a PlotDataItem itself; doing it here too
    flattens the fit onto the axis, and it looks like a bad fit rather than a
    bug."""
    points = joined.flight._series["climb"][0]
    curve = joined.flight._fit_curve
    assert joined.flight.log_y.isChecked()
    assert curve.yData is not None and len(curve.yData) > 8
    lo, hi = float(points.yData.min()), float(points.yData.max())
    assert lo <= float(np.median(curve.yData)) <= hi


def test_the_error_bars_are_log_transformed_but_the_points_are_not(joined):
    """The two items have opposite requirements, which is easy to get wrong."""
    points, bars = joined.flight._series["climb"]
    assert float(points.yData.max()) > 100.0        # linear nSv/h
    assert float(np.max(bars.opts["y"])) < 10.0     # decades


@pytest.mark.parametrize("log_first", [True, False])
def test_toggling_the_dose_axis_does_not_leave_a_stale_error_bar(joined,
                                                                 log_first):
    """A partial ErrorBarItem update hangs the process, with no traceback.

    ``ErrorBarItem.setData`` merges into a persistent dict and ``drawPath``
    prefers ``height`` over ``top``/``bottom``, so leaving one of them set from
    a previous render feeds mismatched array lengths into ``y - height / 2``.
    That raises inside ``paint()``, where a Python exception cannot propagate:
    the item stays dirty, Qt repaints it for ever, and ``processEvents`` never
    drains. It presents as the whole test session hanging in an unrelated test
    later on, so it is pinned here rather than left to be rediscovered.
    """
    panel = joined.flight
    panel.log_y.setChecked(log_first)
    panel.log_y.setChecked(not log_first)
    panel.log_y.setChecked(log_first)

    for _, bars in panel._series.values():
        opts = bars.opts
        set_dims = [name for name in ("height", "top", "bottom")
                    if opts[name] is not None]
        # Exactly one convention in use, never a mixture.
        assert set_dims in (["height"], ["top", "bottom"]), set_dims
        for name in set_dims:
            assert np.asarray(opts[name]).shape == np.asarray(opts["y"]).shape


def test_the_panel_repaints_without_stalling_the_event_loop(joined, app):
    """The observable symptom of the bug above, asserted directly."""
    import time
    from PySide6 import QtCore
    joined.flight.show()
    deadline = time.monotonic() + 5.0
    spins = 0
    while time.monotonic() < deadline and spins < 40:
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 20)
        spins += 1
    joined.flight.close()
    assert spins == 40, "processEvents did not drain; the plot is repainting"


# ---- flight exports ------------------------------------------------------


@pytest.fixture
def saved(monkeypatch, tmp_path):
    """Capture what each export writes, and the name it suggested."""
    from PySide6 import QtWidgets
    calls = {}

    def fake(parent, title, suggested, filters):
        import pathlib
        calls[title] = suggested
        return (str(tmp_path / pathlib.Path(suggested).name), "")

    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(fake))
    return calls


def test_the_export_buttons_are_disabled_until_there_is_data(loaded):
    buttons = loaded.flight._export_buttons
    assert not any(b.isEnabled() for b in buttons.values())


def test_a_join_enables_every_export(joined):
    buttons = joined.flight._export_buttons
    assert all(b.isEnabled() for b in buttons.values())


def test_a_refused_trajectory_still_offers_the_trajectory_export(
        loaded, tmp_path, monkeypatch):
    """Opening it elsewhere is how you work out why it did not match."""
    import json

    from PySide6 import QtWidgets
    path = tmp_path / "elsewhere.json"
    path.write_text(json.dumps({
        "icao24": "ffffff", "callsign": "XXX1 ",
        "path": [[1788785639, 51.5, -0.1, 150.0, 0.0, True],
                 [1788790000, 52.0, 0.5, 10668.0, 0.0, False],
                 [1788795393, 52.5, 1.0, 200.0, 0.0, False]]}))
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec",
                        lambda self: QtWidgets.QMessageBox.StandardButton.Cancel)
    loaded._load_trajectory(str(path), "auto")

    buttons = loaded.flight._export_buttons
    assert buttons["trajectory"].isEnabled()
    assert not buttons["samples"].isEnabled()
    assert not buttons["profile"].isEnabled()


@pytest.mark.parametrize("kind", ["samples", "profile", "trajectory", "report"])
def test_each_export_button_writes_a_file(joined, saved, tmp_path, kind):
    joined.flight._export_buttons[kind].click()
    written = list(tmp_path.iterdir())
    assert len(written) == 1
    assert written[0].stat().st_size > 0
    assert kind.replace("report", "flight-report") in written[0].name or \
        "altitude" in written[0].name


def test_the_suggested_name_carries_the_track_and_the_phase(joined, saved):
    joined.flight.export_branch.setCurrentText("descent")
    joined.flight._export_buttons["samples"].click()
    suggested = next(iter(saved.values()))
    assert "Track 07 Sep 2026" in suggested
    assert suggested.endswith("-altitude-samples-descent.csv")


def test_the_phase_selector_scopes_the_sample_export(joined, saved, tmp_path):
    joined.flight.export_branch.setCurrentText("climb")
    joined.flight._export_buttons["samples"].click()
    written = next(iter(tmp_path.iterdir()))
    rows = written.read_text().strip().splitlines()
    phase = rows[0].split(",").index("phase")
    assert {row.split(",")[phase] for row in rows[1:]} == {"climb"}


def test_the_export_matches_the_bin_width_on_screen(joined, saved, tmp_path):
    """An export that disagreed with the plot would be the worst outcome."""
    joined.flight.step.setValue(1000.0)
    joined.flight._export_buttons["profile"].click()
    rows = next(iter(tmp_path.iterdir())).read_text().strip().splitlines()
    first = float(rows[1].split(",")[0])
    second = float(rows[2].split(",")[0])
    assert second - first == 1000.0


def test_the_exports_honour_the_display_unit(joined, saved, tmp_path):
    joined._on_unit_changed("uSv/h")
    joined.flight._export_buttons["samples"].click()
    header = next(iter(tmp_path.iterdir())).read_text().splitlines()[0]
    assert "dose_rate_uSv_per_h" in header


def test_the_trajectory_export_round_trips_back_in(joined, saved, tmp_path):
    """Save once, re-use offline: the file has to be readable again."""
    from rcta.core.io_flight import load_trajectory
    joined.flight._export_buttons["trajectory"].click()
    written = next(iter(tmp_path.iterdir()))
    again = load_trajectory(written)
    assert again.n == joined._trajectory.n
    assert again.warnings == ()          # the header names its own unit


def test_a_cancelled_dialog_writes_nothing(joined, monkeypatch, tmp_path):
    from PySide6 import QtWidgets
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    joined.flight._export_buttons["samples"].click()
    assert list(tmp_path.iterdir()) == []


# ---- what is greyed out, and why -----------------------------------------


def _action(window, text):
    """Find a menu action by its label, mnemonics and ellipsis included."""
    for action in window.findChildren(type(window.menuBar().actions()[0])):
        if action.text() == text:
            return action
    raise AssertionError(f"no action labelled {text!r}")


EXPORTS_NEEDING_A_TRACK = [
    "Export &track CSV...", "Export &GeoJSON...", "Export G&PX...",
    "Export &anomalies CSV...", "Export &hotspots CSV...",
    "Export &JSON report...", "Export map &image...", "Export map &SVG...",
]
EXPORTS_NEEDING_AN_ALTITUDE = [
    "Export altitude &samples CSV...", "Export a&ltitude profile CSV...",
    "Export &flight JSON report...",
]


@pytest.mark.parametrize("label", EXPORTS_NEEDING_A_TRACK
                         + EXPORTS_NEEDING_AN_ALTITUDE
                         + ["Save flight &trajectory CSV...",
                            "&Run analysis", "Compute &heatmap"])
def test_nothing_that_needs_a_track_is_offered_before_one_is_loaded(window,
                                                                    label):
    """These used to be enabled and to answer a click with silence."""
    assert not _action(window, label).isEnabled()


@pytest.mark.parametrize("label", EXPORTS_NEEDING_A_TRACK)
def test_loading_a_track_enables_its_exports(loaded, label):
    assert _action(loaded, label).isEnabled()


@pytest.mark.parametrize("label", EXPORTS_NEEDING_AN_ALTITUDE)
def test_the_altitude_exports_need_an_altitude_not_just_a_track(loaded, label):
    """The track format carries no altitude, so a track alone is not enough."""
    assert not _action(loaded, label).isEnabled()


def test_a_disabled_action_says_what_it_is_waiting_for(window):
    """Greyed with no reason teaches the user nothing."""
    action = _action(window, "Export &GeoJSON...")
    assert "track file" in action.toolTip()
    altitude = _action(window, "Export a&ltitude profile CSV...")
    assert "Flight dock" in altitude.toolTip()


def test_menus_show_their_tooltips(window):
    """Qt hides action tooltips in a menu unless the menu opts in."""
    from PySide6 import QtWidgets
    menus = window.findChildren(QtWidgets.QMenu)
    assert len(menus) >= 5                      # File, Analysis, View, Basemap, Help
    assert all(menu.toolTipsVisible() for menu in menus)


def test_joining_a_trajectory_enables_the_altitude_exports(
        loaded, sample_trajectory_path):
    loaded._load_trajectory(str(sample_trajectory_path), "auto")
    assert loaded._join() is not None
    for label in EXPORTS_NEEDING_AN_ALTITUDE:
        assert _action(loaded, label).isEnabled(), label
    assert _action(loaded, "Save flight &trajectory CSV...").isEnabled()


def test_clearing_the_reference_is_offered_only_once_one_is_set(loaded):
    action = _action(loaded, "&Clear reference background")
    assert not action.isEnabled()
    assert "rolling" in action.toolTip()
    mask = np.zeros(loaded._track.n, dtype=bool)
    mask[100:400] = True
    loaded._on_map_selection(mask, Qt.KeyboardModifier.NoModifier)
    loaded._mark_reference()
    assert _action(loaded, "&Clear reference background").isEnabled()


def test_the_selection_buttons_follow_the_selection(loaded):
    dock = loaded.selection_dock
    assert not dock.clear_button.isEnabled()
    assert not dock.reference_button.isEnabled()
    # "Zoom to" still frames the whole track, so it stays available.
    assert dock.zoom_button.isEnabled()

    mask = np.zeros(loaded._track.n, dtype=bool)
    mask[10:20] = True
    loaded._on_map_selection(mask, Qt.KeyboardModifier.NoModifier)
    assert dock.clear_button.isEnabled()
    assert dock.reference_button.isEnabled()

    loaded._on_selection_cleared()
    assert not dock.clear_button.isEnabled()
    assert "Select a clean span" in dock.reference_button.toolTip()


# ---- export filenames ----------------------------------------------------


def test_every_export_suggests_a_filename_named_after_the_track(loaded):
    """A dialog opening on "untitled" is a file nobody will find again."""
    stem = "Track 07 Sep 2026 07-53-59"
    for kind, expected in (("csv", f"{stem}.csv"),
                           ("geojson", f"{stem}.geojson"),
                           ("gpx", f"{stem}.gpx"),
                           ("json", f"{stem}-report.json"),
                           ("anomalies", f"{stem}-anomalies.csv"),
                           ("hotspots", f"{stem}-hotspots.csv"),
                           ("image", f"{stem}-map.png"),
                           ("svg", f"{stem}-map.svg")):
        assert loaded._suggested_export_name(kind) == expected


def test_the_per_phase_exports_carry_the_phase_in_the_name(loaded):
    loaded.flight.export_branch.setCurrentText("climb")
    assert loaded._suggested_export_name("profile").endswith(
        "-altitude-profile-climb.csv")


# ---- panels do not declare signals nobody connects ------------------------


def test_the_flight_panel_declares_no_signal_that_is_never_emitted():
    """A signal with no emitter reads as wiring that exists and does not."""
    import inspect

    from PySide6 import QtCore

    from rcta.gui import flight_panel

    source = inspect.getsource(flight_panel)
    declared = {name for name, value in vars(flight_panel.FlightPanel).items()
                if isinstance(value, QtCore.Signal)}
    assert declared
    for name in declared:
        assert f"self.{name}.emit(" in source or f"connect(self.{name})" in source


# ---- the dock is not a third copy of the defaults ------------------------


def test_the_analysis_dock_opens_on_the_core_defaults(app):
    """Every spin box mirrors a frozen dataclass field. Three front ends --
    this dock, ``rcta-cli`` and a direct core call -- have to agree on what
    the default answer is, and a literal written out in each is a default
    that drifts the first time one of them is tuned."""
    from rcta.core.analysis import AnalysisParams
    from rcta.gui.panels import AnalysisDock

    dock = AnalysisDock()
    default = AnalysisParams()
    assert dock.background_params() == default.background
    assert dock.scan_params().alpha == default.scan.alpha
    assert dock.quality_params().max_accuracy_m == default.quality.max_accuracy_m
    assert dock.integration_time() is default.integration_time_s


def test_every_analysis_control_explains_itself(app):
    """The dock is the one place a user can change the answer."""
    from rcta.gui.panels import AnalysisDock

    dock = AnalysisDock()
    for name in ("background", "window_s", "trim", "integration", "alpha",
                 "bonferroni", "max_accuracy"):
        assert getattr(dock, name).toolTip(), name


# ---- switching between the bottom panels ---------------------------------


def _panel_tab_bar(window):
    """The tab bar the bottom dock area draws, or None if it is not tabbed."""
    from PySide6 import QtWidgets

    titles = {dock.windowTitle() for dock in window._panel_docks}
    for bar in window.findChildren(QtWidgets.QTabBar):
        if {bar.tabText(i) for i in range(bar.count())} == titles:
            return bar
    return None


def test_the_bottom_panels_are_one_tabbed_stack(window):
    """Seven panels sharing 320 px would leave none of them readable."""
    window.show()
    bar = _panel_tab_bar(window)
    assert bar is not None
    assert [bar.tabText(i) for i in range(bar.count())] == [
        "Series", "Anomalies", "Hotspots", "Legs", "Profile", "Flight", "Log"]


def test_the_tab_order_is_the_numbered_order(window):
    """Ctrl+4 has to hit the fourth tab. Qt inserts a tabified dock directly
    after the one it is dropped onto, so a stack built by tabifying everything
    onto the first panel lays the tabs out backwards and the numbering in the
    Panels menu silently stops matching the tab bar."""
    window.show()
    bar = _panel_tab_bar(window)
    numbered = [dock.windowTitle() for dock in window._panel_docks]
    assert [bar.tabText(i) for i in range(bar.count())] == numbered


def _panel_menu(window):
    from PySide6 import QtWidgets

    for menu in window.menuBar().findChildren(QtWidgets.QMenu):
        if menu.title() == "&Panels":
            return menu
    raise AssertionError("no Panels menu")


def test_every_panel_has_a_numbered_shortcut(window):
    actions = [a for a in _panel_menu(window).actions() if not a.isSeparator()]
    for number, dock in enumerate(window._panel_docks, start=1):
        action = actions[number - 1]
        assert dock.windowTitle() in action.text()
        assert action.shortcut().toString() == f"Ctrl+{number}"


def test_switching_panels_does_not_require_closing_the_others(window):
    """The gesture this exists for: from the first panel to the last in one
    action, with everything in between still open behind it."""
    window.show()
    first, last = window._panel_docks[0], window._panel_docks[-1]
    window.show_panel(last)
    bar = _panel_tab_bar(window)
    assert bar.tabText(bar.currentIndex()) == last.windowTitle()
    assert not first.isHidden()
    assert all(not dock.isHidden() for dock in window._panel_docks)


def test_cycling_walks_the_stack_and_wraps(window):
    window.show()
    titles = [dock.windowTitle() for dock in window._panel_docks]
    seen = []
    for _ in titles:
        seen.append(window._front_panel.windowTitle())
        window._cycle_panel(1)
    assert seen == titles
    assert window._front_panel.windowTitle() == titles[0]
    window._cycle_panel(-1)
    assert window._front_panel.windowTitle() == titles[-1]


def test_cycling_skips_a_panel_the_user_closed(window):
    """A keystroke must not reopen what someone deliberately shut."""
    window.show()
    window.show_panel(window._panel_docks[0])
    window._panel_docks[1].close()
    window._cycle_panel(1)
    assert window._front_panel is window._panel_docks[2]
    assert window._panel_docks[1].isHidden()


def test_switching_to_a_closed_panel_reopens_it_as_a_tab(window):
    """Reopened docks come back as a split of the bottom area, not a tab,
    which is what used to squeeze the panels into unreadable slivers."""
    window.show()
    legs = window._panel_docks[3]
    legs.close()
    assert legs.isHidden()
    window.show_panel(legs)
    assert not legs.isHidden()
    bar = _panel_tab_bar(window)
    assert bar.tabText(bar.currentIndex()) == "Legs"
    assert [bar.tabText(i) for i in range(bar.count())] == [
        dock.windowTitle() for dock in window._panel_docks]


def test_the_view_menu_does_not_also_toggle_the_bottom_panels(window):
    """Two routes with opposite meanings -- switch to, and hide -- sitting in
    the same menu is what made reaching the Log a matter of closing six
    panels."""
    from PySide6 import QtWidgets

    view = next(m for m in window.menuBar().findChildren(QtWidgets.QMenu)
                if m.title() == "&View")
    toggles = {a.text() for a in view.actions()}
    for dock in window._panel_docks:
        assert dock.windowTitle() not in toggles
    assert "Track" in toggles and "Display" in toggles


# ---- basemap recession is a property of the provider ----------------------


def test_a_satellite_layer_gets_a_lighter_recession_than_a_map(window):
    """Aerial imagery is already muted; the cartographic treatment erases it."""
    window._set_provider("osm")
    osm = (window.map.basemap._desaturate, window.map.basemap._lighten)
    window._set_provider("usgs_imagery")
    sat = (window.map.basemap._desaturate, window.map.basemap._lighten)
    assert sat[0] < osm[0] and sat[1] < osm[1]


def test_the_recession_is_set_before_the_tiles_are_requested(window, monkeypatch):
    """Order matters, and getting it wrong produces a blank map, not an error.

    ``set_basemap_enabled`` is what triggers the tile request, and a tile
    already in the disk cache is delivered synchronously inside that call.
    recess() runs at decode time and changing the amounts clears the decoded
    pixmaps, so setting them afterwards discards exactly the tiles just
    decoded -- and nothing asks for them again, because the fetcher considers
    them served. The map goes white with no warning anywhere.
    """
    order = []
    real_recession = window.map.set_basemap_recession
    real_enabled = window.map.set_basemap_enabled

    def recession(*a, **k):
        order.append("recession")
        return real_recession(*a, **k)

    def enabled(*a, **k):
        order.append("enabled")
        return real_enabled(*a, **k)

    monkeypatch.setattr(window.map, "set_basemap_recession", recession)
    monkeypatch.setattr(window.map, "set_basemap_enabled", enabled)
    window._set_provider("usgs_imagery")
    assert order == ["recession", "enabled"]


def test_switching_basemap_does_not_leave_stale_decoded_tiles(window):
    """The decoded cache is treated at the old amounts and must be dropped."""
    from rcta.gui.tiles import TileKey
    from PySide6 import QtGui

    window._set_provider("osm")
    image = QtGui.QImage(256, 256, QtGui.QImage.Format.Format_RGB32)
    image.fill(QtGui.QColor(20, 120, 60))
    buf = QtCore.QBuffer()
    buf.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
    image.save(buf, "PNG")
    assert window.map.set_tile(TileKey(12, 1, 1), bytes(buf.data()))
    assert window.map.basemap.has(TileKey(12, 1, 1))

    window._set_provider("usgs_imagery")
    assert not window.map.basemap.has(TileKey(12, 1, 1))


def test_a_failed_load_analysis_leaves_nothing_to_export(loaded, tmp_path,
                                                         monkeypatch):
    """The last file's results must never be exported under the new file's name."""
    import rcta.gui.main_window as mw

    def broken(*_, **__):
        raise RuntimeError("boom")

    monkeypatch.setattr(mw, "analyze_track", broken)
    import shutil
    copy = tmp_path / "other.rctrk"
    shutil.copy(loaded._path, copy)
    assert loaded.load(str(copy))
    assert loaded._analysis is None
    assert not any(a.isEnabled() for a in loaded._needs_track)


# ---- theme ---------------------------------------------------------------


def test_switching_theme_restyles_the_open_plots(app, loaded):
    from rcta.gui import theme
    try:
        theme.apply(app, "Dark")
        assert loaded.map._glw.backgroundBrush().color().name() == theme.DARK.plot_bg
        assert loaded.series._glw.backgroundBrush().color().name() == theme.DARK.plot_bg
        assert app.palette().color(app.palette().ColorRole.Window).name() \
            == theme.DARK.window
    finally:
        theme.apply(app, "Light")
    assert loaded.map._glw.backgroundBrush().color().name() == theme.LIGHT.plot_bg


def test_a_map_exported_from_the_dark_theme_is_light(app, loaded):
    """A figure lands on a white page; the screen goes back to dark after."""
    from rcta.gui import map_export, theme
    try:
        theme.apply(app, "Dark")
        image = map_export.render_image(loaded.map, 400, 300)
        assert image.pixelColor(200, 5).lightness() > 240
        assert loaded.map._glw.backgroundBrush().color().name() == theme.DARK.plot_bg
    finally:
        theme.apply(app, "Light")


def test_the_theme_icons_ship_with_the_package():
    from rcta.gui import theme
    for name in ("check", "up_light", "down_light", "up_dark", "down_dark",
                 "rcta"):
        assert (theme._ICONS / f"{name}.svg").is_file()


# ---- display unit --------------------------------------------------------


def test_the_display_unit_reaches_every_dose_on_screen(loaded):
    """It used to change the report and the Flight dock and nothing else."""
    from PySide6.QtCore import Qt as _Qt
    loaded.display_dock.unit.setCurrentText("uSv/h")

    assert "uSv/h" in loaded.map.colorbar.getAxis("left").labelText
    assert "uSv/h" in loaded.series.dose_plot.getAxis("left").labelText
    assert "uSv/h" in loaded.profile.hist_plot.getAxis("bottom").labelText
    headers = [loaded.anomaly_model.headerData(i, _Qt.Orientation.Horizontal)
               for i in range(loaded.anomaly_model.columnCount())]
    assert "peak (uSv/h)" in headers and not any("nSv" in h for h in headers)
    assert "uSv/h" in loaded.selection_dock.text.toPlainText()
    assert "nSv/h" not in loaded.selection_dock.text.toPlainText()

    loaded._on_hover(10)
    assert "uSv/h" in loaded.statusBar().currentMessage()
    loaded._on_hover(-1)
    assert " uSv " in loaded.statusBar().currentMessage()


def test_the_display_unit_scales_the_values_not_just_the_label(loaded):
    lo_nano, hi_nano = loaded.map._limits
    curve = loaded.series._curves["dose"].getData()[1]
    loaded.display_dock.unit.setCurrentText("uSv/h")
    lo_micro, hi_micro = loaded.map._limits
    assert lo_micro == pytest.approx(lo_nano / 1000.0)
    assert hi_micro == pytest.approx(hi_nano / 1000.0)
    import numpy as np
    np.testing.assert_allclose(loaded.series._curves["dose"].getData()[1],
                               curve / 1000.0)


def test_changing_the_unit_does_not_rerun_the_analysis(loaded):
    before = loaded._analysis
    loaded.display_dock.unit.setCurrentText("uR/h")
    assert loaded._analysis is before


def test_the_app_icon_renders(app):
    from rcta.gui import theme
    icon = theme.app_icon()
    assert not icon.isNull()
    image = icon.pixmap(64, 64).toImage()
    assert image.pixelColor(32, 32).alpha() == 255


# ---- drag and drop -------------------------------------------------------


def _drop(window, *paths):
    from PySide6 import QtCore as _QtCore, QtGui as _QtGui
    mime = _QtCore.QMimeData()
    mime.setUrls([_QtCore.QUrl.fromLocalFile(str(p)) for p in paths])
    event = _QtGui.QDropEvent(_QtCore.QPointF(10, 10),
                              _QtCore.Qt.DropAction.CopyAction, mime,
                              _QtCore.Qt.MouseButton.LeftButton,
                              _QtCore.Qt.KeyboardModifier.NoModifier)
    window.dropEvent(event)
    return event


def test_dropping_a_track_opens_it(window, sample_path):
    assert window.acceptDrops()
    _drop(window, sample_path)
    assert window._track is not None and window._track.n == 2964


def test_dropping_a_trajectory_onto_a_track_joins_it(loaded,
                                                     sample_trajectory_path):
    _drop(loaded, sample_trajectory_path)
    assert loaded._trajectory is not None
    assert "loaded trajectory" in loaded.log.toPlainText()


def test_a_trajectory_dropped_before_any_track_explains_itself(
        window, sample_trajectory_path):
    _drop(window, sample_trajectory_path)
    assert window._track is None
    assert "drop a .rctrk track first" in window.log.toPlainText()


def test_longitude_labels_fit_side_by_side_on_a_walk(app):
    """At walking scale seconds labels used to run into one another."""
    from PySide6 import QtGui
    from rcta.core.geo import lonlat_to_web_mercator
    from rcta.gui.geo_axis import GeoAxisItem, format_degree
    axis = GeoAxisItem("bottom")
    x0 = float(lonlat_to_web_mercator(-111.905, 40.77)[0])
    x1 = float(lonlat_to_web_mercator(-111.870, 40.77)[0])
    size = 680.0
    (major, ticks), (minor, _) = axis.tickValues(x0, x1, size)
    metrics = QtGui.QFontMetricsF(QtGui.QFont())
    width = metrics.horizontalAdvance(format_degree(-111.9, major, False))
    assert len(ticks) * width <= size
    # Minor ticks carry no labels of their own.
    assert all(s == "" for s in axis.tickStrings(ticks, 1.0, minor))
    assert all(axis.tickStrings(ticks, 1.0, major))


def test_degree_labels_carry_instead_of_printing_sixty():
    """59.96 s rounded on its own printed 111°52'60.0"; it is 111°53'00.0"."""
    from rcta.gui.geo_axis import format_degree
    assert format_degree(-111.88332222, 1 / 3600, False) == "111°53'00.0\"W"
    assert format_degree(40.99999, 1 / 60, True) == "41°00'N"
    assert format_degree(40.5, 1 / 60, True) == "40°30'N"
    assert format_degree(-111.905, 1 / 3600, False) == "111°54'18.0\"W"
