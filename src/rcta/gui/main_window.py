"""The main window: docks, menus, and the one place everything is wired.

Structurally conventional, and deliberately so: a ``QMainWindow`` with a
central plot, dock widgets around it, mnemonic menus, no toolbar and no
top-level tabs, and the status bar as the primary summary surface.

The rule that keeps it comprehensible is that ``MainWindow`` owns every
connection and makes every call into :mod:`rcta.core`. Panels emit gestures and
render what they are handed; they never talk to each other and never analyse
anything. In particular there is exactly one place a
:class:`~rcta.core.selection.Selection` is constructed -- :meth:`_broadcast` --
so the map, the series and the tables cannot end up disagreeing about what is
selected.
"""

from __future__ import annotations

import html
import re
import traceback
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from rcta.core.analysis import AnalysisParams, analyze_track
from rcta.core.anomalies import significance_trace
from rcta.core.io_export import (altitude_profile_to_csv,
                                 altitude_samples_to_csv, anomalies_to_csv,
                                 flight_report_to_json, hotspots_to_csv,
                                 report_to_json, track_to_csv,
                                 track_to_geojson, track_to_gpx)
from rcta.core.io_flight import trajectory_to_csv
from rcta.core.flight import FlightMatchError, join_altitude, looks_like_flight
from rcta.core.io_radiacode import TrackParseError, load_track
from rcta.core.profile import summarise
from rcta.core.selection import Selection
from rcta.core.spatial import GridParams, interpolate
from rcta.gui import theme
from rcta.gui.map_panel import MapPanel
from rcta.gui.models import AnomalyTableModel, HotspotTableModel, LegTableModel
from rcta.gui.panels import (AnalysisDock, DisplayDock, LogDock, SelectionDock,
                             TrackDock)
from rcta.gui.flight_panel import FlightPanel
from rcta.gui.profile_panel import ProfilePanel
from rcta.gui.series_panel import SeriesPanel

Qt = QtCore.Qt

__all__ = ["MainWindow"]

LIMITATIONS = """
<h3>Radiacode Track Analyzer 0.1.0</h3>
<p>Analysis of GPS radiation tracks from RadiaCode 100-series detectors.</p>
<h4>Known limitations, all of them deliberate</h4>
<ol>
<li><b>The per-sample integration time is inferred.</b> The file records
sample timestamps, not detector live times. Every detection limit scales
directly with the assumed value, and it is the largest systematic here. Set it
explicitly if you know it.</li>
<li><b>The background estimator defines what counts as an anomaly.</b> Change
the method and the findings change. It is reported with every result for that
reason.</li>
<li><b>The interpolated surface is a measurement of a line, not an area.</b>
Cells more than a couple of bandwidths from the track are left blank rather
than extrapolated. Do not read the blank regions as low.</li>
<li><b>Correlations on a track are serially dependent.</b> Naive p-values are
meaningless at a two-second cadence; the corrected one is reported alongside
and is the one to use. Neither can detect confounding.</li>
<li><b>No source localisation or activity estimate.</b> This tool reports
where the count rate was elevated, not what caused it or how strong it was.</li>
<li><b>No spectra.</b> The track format carries none, so no nuclide
identification is possible from a track alone.</li>
<li><b>Altitude is imported, not measured.</b> The track format carries no
altitude, so the Flight dock joins one from an external trajectory matched by
time. It is validated against the track's own GPS fixes and refused when they
disagree, but a trajectory that matches in position can still be wrong in
altitude, and barometric altitude is a pressure measurement rather than a
height. Fetching history needs free OpenSky credentials
(<tt>OPENSKY_CLIENT_ID</tt> and <tt>OPENSKY_CLIENT_SECRET</tt>); a local
trajectory file needs nothing.</li>
<li><b>A scale height fitted to one flight describes that flight.</b> Cruise
dominates the samples, so the exponential is set by the climb and descent,
which are also the least well covered. Whether the two agree is reported
beside the fit and is the honest error bar on it.</li>
</ol>
"""


#: A bare URL. Ends on a non-punctuation character so that the comma after
#: the sign-up link in a provider hint is not swallowed into the href.
_URL_RE = re.compile(r'https?://[^\s<>"\')]*[^\s<>"\').,;:!?]')


def _rich(text: str) -> str:
    """Plain failure text as rich text: escaped, wrapped, and with live links.

    Escaping is not optional here. The text includes a server's own error
    string, and OpenSky's carry angle brackets; a caption set to RichText
    would swallow them and, on a bad day, the sentence around them.

    Links are made clickable because the useful half of the credentials hint
    is a URL, and a caption is neither selectable nor copyable by default --
    an address a user can read but not reach is a dead end. Escaped inside
    :func:`html.escape` rather than around it because the sign-up URL carries
    a query string, and ``&`` in an href must be an entity too.
    """
    out = []
    cursor = 0
    for match in _URL_RE.finditer(text):
        out.append(html.escape(text[cursor:match.start()]))
        url = html.escape(match.group(0), quote=True)
        out.append(f'<a href="{url}">{url}</a>')
        cursor = match.end()
    out.append(html.escape(text[cursor:]))
    return "<br>".join("".join(out).splitlines())


class MainWindow(QtWidgets.QMainWindow):
    """Track analysis front end."""

    def __init__(self, parent=None, *, offline: bool = False,
                 synchronous: bool = False) -> None:
        super().__init__(parent)
        self.setWindowTitle("Radiacode Track Analyzer")
        self.resize(1500, 950)

        self._track = None
        self._analysis = None
        self._selection = None
        self._grid = None
        self._reference_span = None
        self._trajectory = None
        self._path = ""
        self._unit = "nSv/h"
        self._applying = False
        self._offline = bool(offline)
        # Declared before the menus exist: creating the offline action with
        # checked=True fires its toggled signal, which lands in _set_offline
        # before _make_basemap_menu has finished assigning these.
        self._basemap_actions: dict = {}
        self._offline_action = None

        self._setup_tiles()

        from rcta.gui.worker import JobQueue
        self.jobs = JobQueue(self, synchronous=synchronous)
        self.jobs.finished.connect(self._on_job_finished)
        self.jobs.failed.connect(self._on_job_failed)
        self.jobs.progress.connect(self._on_job_progress)

        self.map = MapPanel()
        self.setCentralWidget(self.map)

        self._make_docks()
        self._make_menus()
        self._wire()

        self.statusBar().showMessage("Open a .rctrk track file to begin")

    # ---- construction -----------------------------------------------------

    def _add_dock(self, title, widget, area):
        dock = QtWidgets.QDockWidget(title, self)
        dock.setObjectName(f"dock_{title}")
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _action(self, menu, text, slot, shortcut=None, checkable=False,
                checked=False):
        action = QtGui.QAction(text, self)
        if checkable:
            action.setCheckable(True)
            action.setChecked(checked)
            action.toggled.connect(slot)
        else:
            action.triggered.connect(slot)
        if shortcut:
            action.setShortcut(shortcut)
        # Qt suppresses action tooltips inside a menu unless the menu asks for
        # them. Without this the basemap providers' policy notes and the
        # "why is this greyed out" reasons below are set but never shown.
        menu.setToolTipsVisible(True)
        menu.addAction(action)
        return action

    def _make_docks(self) -> None:
        left = Qt.DockWidgetArea.LeftDockWidgetArea
        right = Qt.DockWidgetArea.RightDockWidgetArea
        bottom = Qt.DockWidgetArea.BottomDockWidgetArea

        self.track_dock = TrackDock()
        self._add_dock("Track", self.track_dock, left)

        self.analysis_dock = AnalysisDock()
        self._add_dock("Analysis", self.analysis_dock, left)

        self.display_dock = DisplayDock()
        self.display_dock.set_providers(self._providers)
        # The window may have been constructed offline; the dock must show that
        # rather than defaulting to unchecked and disagreeing with the fetcher.
        self.display_dock.set_offline(self._offline)
        self._add_dock("Display", self.display_dock, right)

        self.selection_dock = SelectionDock()
        self._add_dock("Selection", self.selection_dock, right)

        self.series = SeriesPanel()
        series_dock = self._add_dock("Series", self.series, bottom)

        self.anomaly_model = AnomalyTableModel()
        self.anomaly_view = self._make_table(self.anomaly_model)
        anomaly_dock = self._add_dock("Anomalies", self.anomaly_view, bottom)

        self.hotspot_model = HotspotTableModel()
        self.hotspot_view = self._make_table(self.hotspot_model)
        hotspot_dock = self._add_dock("Hotspots", self.hotspot_view, bottom)

        self.leg_model = LegTableModel()
        self.leg_view = self._make_table(self.leg_model)
        leg_dock = self._add_dock("Legs", self.leg_view, bottom)

        self.profile = ProfilePanel()
        profile_dock = self._add_dock("Profile", self.profile, bottom)

        self.flight = FlightPanel()
        flight_dock = self._add_dock("Flight", self.flight, bottom)

        self.log = LogDock()
        log_dock = self._add_dock("Log", self.log, bottom)

        # Canonical order of the bottom panels. Everything that stacks, cycles
        # or numbers them reads this one list, so the tab order, the Panels
        # menu and Ctrl+1..7 can never drift apart.
        self._panel_docks = [series_dock, anomaly_dock, hotspot_dock, leg_dock,
                             profile_dock, flight_dock, log_dock]
        # Tabs above the panel rather than below it: they are the handle for
        # the gesture the user makes most often down here, and at the bottom
        # of the window they sit under the panel content and read as a
        # footer. North puts them against the map, where a tab bar is looked
        # for.
        self.setTabPosition(bottom, QtWidgets.QTabWidget.TabPosition.North)
        self._restack_panels()
        # Which panel is in front. Qt will not say -- a tabified dock that is
        # behind another still reports isVisible() -- so the one signal that
        # does fire on a tab click keeps it current, and cycling then starts
        # from where the user actually is rather than from the first tab.
        self._front_panel = series_dock
        for dock in self._panel_docks:
            dock.visibilityChanged.connect(
                lambda shown, d=dock: self._note_front_panel(d, shown))
        series_dock.raise_()
        self.resizeDocks([series_dock], [320], Qt.Orientation.Vertical)

    # ---- the bottom panels ------------------------------------------------

    def _restack_panels(self) -> None:
        """Lay the bottom panels out as one tab stack, in canonical order.

        Called again whenever a closed panel is reopened. Qt drops a reopened
        dock back into the bottom area as a *split*, not a tab, which is how
        the area ends up with several panels fighting over 320 px and why
        switching used to mean closing everything in front. Restacking is
        cheap and idempotent, and it also restores the numbering after the
        user has dragged tabs around.
        """
        # Chained pairwise rather than all onto the first: Qt inserts the new
        # tab immediately after the one it is dropped onto, so tabifying
        # everything onto the Series panel would lay the tabs out backwards
        # and the numbering below would not match what the tab bar shows.
        for host, dock in zip(self._panel_docks, self._panel_docks[1:]):
            self.tabifyDockWidget(host, dock)

    def show_panel(self, dock) -> None:
        """Bring one bottom panel to the front, opening it if it was closed."""
        if dock.isFloating():
            dock.setFloating(False)
        if dock.isHidden():
            dock.show()
            self._restack_panels()
        dock.raise_()
        self._front_panel = dock
        widget = dock.widget()
        if widget is not None:
            widget.setFocus(Qt.FocusReason.TabFocusReason)

    def _note_front_panel(self, dock, shown: bool) -> None:
        if shown:
            self._front_panel = dock

    def _cycle_panel(self, step: int) -> None:
        """Move ``step`` panels along the tab bar, wrapping at both ends.

        Only open panels take part: one the user closed stays closed rather
        than being resurrected by a keystroke.
        """
        open_docks = [d for d in self._panel_docks if not d.isHidden()]
        if not open_docks:
            self.show_panel(self._panel_docks[0])
            return
        current = (self._front_panel if self._front_panel in open_docks
                   else open_docks[0])
        index = open_docks.index(current)
        self.show_panel(open_docks[(index + step) % len(open_docks)])

    def _make_table(self, model):
        view = QtWidgets.QTableView()
        view.setModel(model)
        view.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        view.setSortingEnabled(False)
        view.verticalHeader().setVisible(False)
        view.horizontalHeader().setStretchLastSection(True)
        view.setAlternatingRowColors(True)
        return view

    def _make_menus(self) -> None:
        # Actions that cannot do anything yet, grouped by what they are waiting
        # for. Every one of these used to be permanently enabled and to return
        # in silence when the precondition was missing, so "Export GeoJSON..."
        # on an empty window was indistinguishable from a failed write.
        self._needs_track: list = []
        self._needs_join: list = []
        self._needs_trajectory: list = []
        self._needs_selection: list = []

        file_menu = self.menuBar().addMenu("&File")
        self._action(file_menu, "&Open track...", self.open_dialog, "Ctrl+O")
        self._needs_track.append(self._action(
            file_menu, "Open flight &trajectory...",
            self._open_trajectory_dialog, "Ctrl+Shift+O"))
        file_menu.addSeparator()
        for text, kind in (("Export &track CSV...", "csv"),
                           ("Export &GeoJSON...", "geojson"),
                           ("Export G&PX...", "gpx"),
                           ("Export &anomalies CSV...", "anomalies"),
                           ("Export &hotspots CSV...", "hotspots"),
                           ("Export &JSON report...", "json")):
            self._needs_track.append(self._action(
                file_menu, text, lambda _=False, k=kind: self._export(k)))
        self._needs_join.append(self._action(
            file_menu, "Export altitude &samples CSV...",
            lambda: self._export("samples")))
        self._needs_join.append(self._action(
            file_menu, "Export a&ltitude profile CSV...",
            lambda: self._export("profile")))
        self._needs_trajectory.append(self._action(
            file_menu, "Save flight &trajectory CSV...",
            lambda: self._export("trajectory")))
        self._needs_join.append(self._action(
            file_menu, "Export &flight JSON report...",
            lambda: self._export("report")))
        file_menu.addSeparator()
        self._needs_track.append(self._action(
            file_menu, "Export map &image...", self._export_image))
        self._needs_track.append(self._action(
            file_menu, "Export map &SVG...", self._export_svg))
        file_menu.addSeparator()
        self._action(file_menu, "&Quit", self.close, "Ctrl+Q")

        analysis_menu = self.menuBar().addMenu("&Analysis")
        self._needs_track.append(self._action(
            analysis_menu, "&Run analysis", self.analyse, "Ctrl+R"))
        self._needs_track.append(self._action(
            analysis_menu, "Compute &heatmap", self.compute_grid, "Ctrl+H"))
        analysis_menu.addSeparator()
        self._needs_selection.append(self._action(
            analysis_menu, "Use selection as &background",
            self._mark_reference))
        self._reference_action = self._action(
            analysis_menu, "&Clear reference background", self._clear_reference)

        view_menu = self.menuBar().addMenu("&View")
        self._make_basemap_menu(view_menu.addMenu("&Basemap"))
        self._needs_track.append(self._action(
            view_menu, "Zoom to &track", self.map.zoom_to_track))
        self._needs_track.append(self._action(
            view_menu, "Zoom to &selection", self._zoom_to_selection))
        self._action(view_menu, "&Select mode (or hold Shift)",
                     self.map.set_select_mode, "Ctrl+E", checkable=True)
        view_menu.addSeparator()
        self._make_theme_menu(view_menu.addMenu("T&heme"))
        self._make_panel_menu(view_menu.addMenu("&Panels"))
        view_menu.addSeparator()
        for dock in self.findChildren(QtWidgets.QDockWidget):
            if dock not in self._panel_docks:
                view_menu.addAction(dock.toggleViewAction())

        help_menu = self.menuBar().addMenu("&Help")
        self._action(help_menu, "&About / limitations", self._about)

        self._update_actions()

    def _make_theme_menu(self, menu) -> None:
        menu.setToolTipsVisible(True)
        group = QtGui.QActionGroup(self)
        for name in theme.THEMES:
            action = QtGui.QAction(name, self, checkable=True,
                                   checked=name == theme.choice())
            action.triggered.connect(
                lambda _=False, n=name: self._set_theme(n))
            group.addAction(action)
            menu.addAction(action)

    def _set_theme(self, name: str) -> None:
        QtCore.QSettings().setValue("ui/theme", name)
        theme.apply(QtWidgets.QApplication.instance(), name)

    def _update_actions(self) -> None:
        """Enable exactly the actions whose precondition is satisfied.

        Greyed rather than hidden, and each with the reason attached, for the
        same reason the unavailable basemap providers are: a control the user
        cannot find teaches them nothing about what the program can do. Qt
        falls back to an action's own label when its tooltip is empty, so the
        reason has to be cleared again once the action becomes available.
        """
        has_selection = (self._selection is not None
                         and not self._selection.is_empty)
        groups = (
            (self._needs_track, self._analysis is not None,
             "Open a .rctrk track file first."),
            (self._needs_join, self._join() is not None,
             "No altitude yet. The track format carries none: fetch a flight "
             "or load a trajectory in the Flight dock."),
            (self._needs_trajectory, self._trajectory is not None,
             "No trajectory loaded. Fetch a flight or load one from a file in "
             "the Flight dock."),
            (self._needs_selection, has_selection,
             "Select a span on the map or the series plot first."),
            ([self._reference_action], self._reference_span is not None,
             "No reference background is set; the estimator is rolling."),
        )
        for actions, ready, reason in groups:
            for action in actions:
                action.setEnabled(ready)
                action.setToolTip("" if ready else reason)

        self.selection_dock.set_selection_available(has_selection)

    def _wire(self) -> None:
        self.map.selectionDrawn.connect(self._on_map_selection)
        self.map.selectionCleared.connect(self._on_selection_cleared)
        self.map.pointHovered.connect(self._on_hover)
        self.map.colorLimitsChanged.connect(self.map.set_color_limits)
        self.map.tilesRequested.connect(self._on_tiles_requested)

        self.series.intervalSelected.connect(self._on_series_interval)
        self.series.intervalCleared.connect(self._on_selection_cleared)
        self.series.pointHovered.connect(self._on_hover)

        self.analysis_dock.changed.connect(self.analyse)
        self.analysis_dock.runRequested.connect(self.analyse)
        self.analysis_dock.gridRequested.connect(self.compute_grid)

        self.display_dock.colormapChanged.connect(self.map.set_colormap)
        self.display_dock.colorScaleChanged.connect(self.map.set_color_scale)
        self.display_dock.colorFieldChanged.connect(self.map.set_color_field)
        self.display_dock.accuracyThresholdChanged.connect(
            self.map.set_accuracy_threshold)
        self.display_dock.gapConnectorsToggled.connect(self.map.set_gap_connectors)
        self.display_dock.unitChanged.connect(self._on_unit_changed)
        self.display_dock.basemapChanged.connect(self._set_provider)
        self.display_dock.offlineToggled.connect(self._set_offline)

        self.flight.fetchRequested.connect(self._fetch_flight)
        self.flight.cacheClearRequested.connect(self._clear_flight_cache)
        self.flight.fileRequested.connect(self._load_trajectory)
        self.flight.exportRequested.connect(self._export)

        self.selection_dock.zoomRequested.connect(self._zoom_to_selection)
        self.selection_dock.clearRequested.connect(self._on_selection_cleared)
        self.selection_dock.markReferenceRequested.connect(self._mark_reference)

        self.anomaly_view.selectionModel().selectionChanged.connect(
            self._on_anomaly_row)
        self.hotspot_view.selectionModel().selectionChanged.connect(
            self._on_hotspot_row)

    # ---- basemap ----------------------------------------------------------

    def _setup_tiles(self) -> None:
        from rcta.gui.tiles import TileCache, TileFetcher, cache_dir, load_providers

        self._providers, default_id = load_providers()
        self._provider_id = default_id
        self._cache = TileCache(cache_dir())
        self.tiles = TileFetcher(self._cache, self._providers[default_id], self,
                                 offline=self._offline)
        self.tiles.tileReady.connect(self._on_tile_ready)
        self.tiles.statusChanged.connect(self._on_tile_status)

    def _make_panel_menu(self, menu) -> None:
        """Numbered *switch to* actions for the bottom panels.

        The panels have always been tabbed, but the only keyboard route to
        them was the View menu's toggles, and a toggle hides a panel rather
        than switching to it -- so reaching the Log meant closing the six
        panels stacked in front of it. These actions show the panel if it was
        closed, restack it as a tab and raise it: one keystroke, whatever
        state the bottom of the window is in.
        """
        for number, dock in enumerate(self._panel_docks, start=1):
            title = dock.windowTitle()
            action = self._action(
                menu, f"&{number}  {title}",
                lambda _=False, d=dock: self.show_panel(d),
                f"Ctrl+{number}")
            action.setToolTip(f"Switch to the {title} panel")
        menu.addSeparator()
        # Two sequences each: the page keys are the Qt convention, and the
        # brackets are what a Mac laptop without them can actually reach.
        for text, step, keys in (
                ("&Next panel", 1, ("Ctrl+PgDown", "Ctrl+]")),
                ("&Previous panel", -1, ("Ctrl+PgUp", "Ctrl+[")),):
            action = self._action(menu, text,
                                  lambda _=False, s=step: self._cycle_panel(s))
            action.setShortcuts([QtGui.QKeySequence(k) for k in keys])

    def _make_basemap_menu(self, menu) -> None:
        """Mirrors the Display dock's picker. The dock is the primary control;
        both routes call :meth:`_set_provider`, which syncs the other."""
        group = QtGui.QActionGroup(self)
        group.setExclusive(True)
        self._basemap_actions = {}

        none_action = self._action(menu, "&No basemap",
                                   lambda on: on and self._set_provider(None),
                                   checkable=True, checked=True)
        group.addAction(none_action)
        self._basemap_actions[None] = none_action

        for provider in self._providers.values():
            action = self._action(
                menu, provider.name,
                lambda on, pid=provider.id: on and self._set_provider(pid),
                checkable=True)
            action.setEnabled(provider.available)
            tip = provider.policy_note
            if not provider.available:
                tip = f"Needs an API key. {tip}"
            elif not provider.allow_export:
                tip = f"{tip} Excluded from image exports."
            action.setToolTip(tip)
            group.addAction(action)
            self._basemap_actions[provider.id] = action

        menu.addSeparator()
        self._offline_action = self._action(
            menu, "Work &offline (cache only)", self._set_offline,
            checkable=True, checked=self._offline)
        self._action(menu, "&Purge tile cache", self._purge_cache)

    def _set_provider(self, provider_id) -> None:
        # Reached from the Display dock's picker and from the View menu, so it
        # syncs both back afterwards rather than letting them drift apart.
        self._provider_id = provider_id
        self.display_dock.set_basemap(provider_id)
        action = self._basemap_actions.get(provider_id)
        if action is not None and not action.isChecked():
            blocked = action.blockSignals(True)
            try:
                action.setChecked(True)
            finally:
                action.blockSignals(blocked)

        if provider_id is None:
            self.map.set_basemap_enabled(False)
            self.map.set_attribution("")
            self.log.log("basemap off")
            return
        provider = self._providers[provider_id]
        self.tiles.set_provider(provider)
        self.map.clear_basemap()
        # Strictly before enabling, which is what requests the tiles. recess()
        # runs at decode time, and a tile already on disk is delivered
        # synchronously inside that request -- so setting the amounts
        # afterwards would discard the tiles just decoded and never ask again.
        self.map.set_basemap_recession(provider.desaturate, provider.lighten)
        self.map.set_basemap_enabled(True, provider.max_zoom)
        # Attribution is drawn inside the view and has no visibility toggle.
        self.map.set_attribution(provider.attribution)
        self.log.log(f"basemap: {provider.name}")

    def _set_offline(self, offline: bool) -> None:
        self._offline = bool(offline)
        self.display_dock.set_offline(self._offline)
        if self._offline_action is not None and \
                self._offline_action.isChecked() != self._offline:
            blocked = self._offline_action.blockSignals(True)
            try:
                self._offline_action.setChecked(self._offline)
            finally:
                self._offline_action.blockSignals(blocked)
        self.tiles.set_offline(self._offline)
        self.log.log("working offline; basemap served from cache only"
                     if offline else "basemap online")

    def _purge_cache(self) -> None:
        removed, freed = self._cache.sweep()
        self.log.log(f"tile cache: removed {removed} file(s), "
                     f"freed {freed / 1e6:.1f} MB")

    def _on_tiles_requested(self, keys, zoom) -> None:
        self.tiles.request(keys)

    def _on_tile_ready(self, key, data) -> None:
        if not self.map.set_tile(key, data):
            # 200 with something that is not an image: a captive portal, most
            # likely. Never cached as terrain.
            self.log.log("basemap returned a non-image response; "
                         "you may be behind a captive portal", "warning")

    def _on_tile_status(self, message, pending) -> None:
        if message:
            self.log.log(message, "warning")
            self.statusBar().showMessage(message)

    # ---- loading ----------------------------------------------------------

    def open_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open track", "", "RadiaCode track (*.rctrk);;All files (*)")
        if path:
            self.load(path)

    def load(self, path) -> bool:
        """Load a track file. Returns True on success."""
        try:
            track = load_track(path)
        except TrackParseError as exc:
            self.log.log(f"cannot load {path}: {exc}", "error")
            self.statusBar().showMessage(f"failed to load {Path(path).name}")
            return False
        except OSError as exc:
            self.log.log(f"cannot read {path}: {exc}", "error")
            return False

        # Anything still computing belongs to the previous file, and so does
        # its analysis: if the new file's analysis fails, the exports must
        # have nothing to write rather than the last file's results under
        # this file's name.
        self.jobs.cancel_all()
        self._analysis = None
        self._track = track
        self._selection = Selection.empty(track.n)
        self._grid = None
        self._reference_span = None
        # A trajectory belongs to the track it was joined to, so a new file
        # must not inherit the last one's altitudes.
        self._trajectory = None
        self._path = str(path)
        self.flight.set_trajectory_available(False)
        self.flight.set_default_date(self._departure_local_start(track))
        # Status first, then the join: clearing the join restores the panel's
        # own "there is no altitude yet" message, and wiping the caption
        # afterwards would leave the dock silently blank.
        self.flight.set_status("")
        self.flight.set_join(None)
        self.log.log(f"loaded {Path(path).name}: {track.n} samples")
        for warning in track.meta.warnings:
            self.log.log(warning, "warning")
        self.setWindowTitle(f"Radiacode Track Analyzer - {Path(path).name}")

        # Said once, on load, because a user with a flight track has no reason
        # to guess that a dock exists to import the altitude the file lacks.
        verdict = looks_like_flight(track)
        if verdict.is_flight:
            self.log.log(
                f"this track is airborne (median {verdict.speed_median_m_s:.0f} "
                "m/s): the format carries no altitude, so use the Flight dock "
                "to import one and profile dose against it")

        self.analyse()
        # analyse() returns early if the analysis raised, in which case
        # _apply_analysis never ran and the action states are still the
        # previous file's.
        self._update_actions()
        return True

    # ---- analysis ---------------------------------------------------------

    def _params(self) -> AnalysisParams:
        return AnalysisParams(
            quality=self.analysis_dock.quality_params(),
            background=self.analysis_dock.background_params(self._reference_span),
            scan=self.analysis_dock.scan_params(),
            integration_time_s=self.analysis_dock.integration_time(),
            unit=self._unit,
        )

    def analyse(self) -> None:
        if self._track is None:
            return
        try:
            analysis = analyze_track(self._track, self._params())
        except Exception as exc:                      # keep the GUI alive
            self.log.log(f"analysis failed: {exc}", "error")
            self.log.log(traceback.format_exc(limit=3), "error")
            return
        self._apply_analysis(analysis)

    def _apply_analysis(self, analysis) -> None:
        self._analysis = analysis
        track = analysis.track

        self.track_dock.set_text(track.describe())
        for warning in analysis.background.warnings:
            self.log.log(warning, "warning")

        threshold = (analysis.anomalies[0].threshold_sigma
                     if analysis.anomalies else 0.0)
        trace = significance_trace(track, analysis.background, scale=8)

        self.map.set_track(track, analysis.quality)
        self.series.set_track(track, analysis.background, trace, threshold)
        self.profile.set_track(track, analysis.speed_dose)

        self.anomaly_model.set_rows(analysis.anomalies)
        self.hotspot_model.set_rows(analysis.hotspots)
        self.leg_model.set_rows(LegTableModel.rows_from(track, analysis.quality))
        for view in (self.anomaly_view, self.hotspot_view, self.leg_view):
            view.resizeColumnsToContents()

        if self._selection is None or self._selection.n != track.n:
            self._selection = Selection.empty(track.n)
        self._broadcast(self._selection)
        self._update_actions()

        if analysis.found_nothing:
            widest = analysis.mda_table[-1] if analysis.mda_table else None
            if widest:
                self.log.log(
                    f"no anomalies above L_C; MDA {widest[2]:.1f} nSv/h over "
                    f"{widest[1]:.0f} s")
        else:
            self.log.log(f"{len(analysis.significant)} anomal"
                         f"{'y' if len(analysis.significant) == 1 else 'ies'} "
                         "above L_C")

    def compute_grid(self) -> None:
        """The one slow step, and the only one that goes to a worker."""
        if self._analysis is None:
            return
        self.statusBar().showMessage("computing heatmap...")
        self.jobs.submit(
            "grid", interpolate, self._analysis.track,
            GridParams(value="dose"), background=self._analysis.background)

    # ---- flight -----------------------------------------------------------

    def _join(self):
        """The current altitude join, or None. Owned by the panel."""
        return self.flight._join

    def _flight_profile(self):
        """The profile exactly as the Flight dock is drawing it.

        Read back from the panel's own controls rather than re-derived with
        defaults, so an export can never disagree with the plot the user is
        looking at. The phase comes from the export row's own selector, so a
        reader can take the climb and the descent as separate files.
        """
        from rcta.core.flight import bin_by_altitude
        return bin_by_altitude(self._join(), step_m=float(self.flight.step.value()),
                               statistic=self.flight.statistic.currentText(),
                               branch=self.flight.export_branch.currentText())

    def _open_trajectory_dialog(self) -> None:
        """Raise the Flight dock and ask for a file, so the result is visible.

        Choosing a trajectory from the File menu while the Flight dock is
        hidden behind four other tabs would silently do all the work somewhere
        the user cannot see.
        """
        for dock in self.findChildren(QtWidgets.QDockWidget):
            if dock.widget() is self.flight:
                dock.show()
                dock.raise_()
        self.flight.choose_file()

    @staticmethod
    def _departure_zone(track) -> str:
        """IANA zone of the airport the track left from, or "" for UTC.

        The date in the dock is the one on a boarding pass, and a boarding
        pass is written in local time, so the day it names is a day at the
        departure airport rather than a day in UTC. For an evening departure
        those are different days; see
        :func:`rcta.core.flight.search_window`.
        """
        from rcta.core.airports import nearest_airport

        airport = nearest_airport(float(track.lat[0]), float(track.lon[0]))
        return airport.tz if airport is not None else ""

    def _departure_local_start(self, track):
        """When the track begins, on the departure airport's own clock.

        What the date field is pre-filled with, and it has to agree with
        :meth:`_departure_zone` or the default would be a day out on exactly
        the flights that motivated reading the date locally at all -- the
        dock would open pre-set to a date it then warns about.
        """
        start = datetime.fromtimestamp(float(track.time_s[0]), tz=timezone.utc)
        tz = self._departure_zone(track)
        if not tz:
            return start
        try:
            return start.astimezone(ZoneInfo(tz))
        except (ZoneInfoNotFoundError, ValueError):
            return start

    def _fetch_flight(self, text: str, date_iso: str) -> None:
        """Look a flight up with the provider, off the GUI thread.

        The only network call in this program that is not a map tile, and the
        only one that can block for seconds, so it goes to the worker. Import
        is deferred to the call: a user who never opens the Flight dock should
        not pay for an HTTP stack at startup, and the GUI must still run if
        ``rcta.net`` is unavailable.
        """
        if self._analysis is None:
            self.flight.set_status("Load a track first: the search window "
                                   "comes from its own timestamps.")
            return

        from rcta.core.flight import parse_flight_number, search_window
        from rcta.net.flights import fetch_trajectory, load_flight_providers

        track = self._analysis.track
        try:
            flight = parse_flight_number(text)
            begin, end, warnings = search_window(
                float(track.time_s[0]), float(track.time_s[-1]), date_iso,
                tz=self._departure_zone(track))
        except ValueError as exc:
            self.flight.set_status(str(exc))
            self.log.log(f"flight lookup: {exc}", "error")
            return

        if not flight.known_airline:
            self.flight.set_status(flight.describe())
            self.log.log(flight.describe(), "warning")
            return
        for warning in warnings:
            self.log.log(warning, "warning")

        providers, default = load_flight_providers()
        provider = providers[default]
        self.log.log(f"flight lookup: {flight.describe()}")
        self.flight.set_status(
            f"Looking up {flight.callsign} with {provider.name}...")
        self.statusBar().showMessage(f"fetching {flight.callsign}...")
        # The lookup is keyed by departure airport, so the track supplies
        # where as well as when; see OpenSkyClient.find_flight.
        self.jobs.submit("flight", fetch_trajectory, flight, begin_s=begin,
                         end_s=end,
                         origin=(float(track.lat[0]), float(track.lon[0])),
                         provider=provider, offline=self._offline)

    def _clear_flight_cache(self) -> None:
        """Delete every cached provider response, after saying what it costs.

        Confirmed, unlike the tile cache purge beside it in the Basemap menu,
        because the two are not equivalent. A discarded map tile costs a
        request to somebody's free tile server; a discarded flight costs
        requests out of a metered account, and a settled flight is cached
        indefinitely precisely so that it never has to be paid for twice.
        The dialog names the size so the trade is a decision rather than a
        guess. Imported here rather than at module scope for the same reason
        as the fetch: the GUI has to run with no HTTP stack installed.
        """
        from rcta.net.flights import FlightCache

        cache = FlightCache()
        size = cache.size_bytes()
        if not size:
            self.log.log("flight cache is already empty")
            self.statusBar().showMessage("flight cache is already empty")
            return

        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Question)
        box.setWindowTitle("Clear flight cache")
        box.setText(f"Delete {size / 1e6:.1f} MB of cached trajectory "
                    "responses?")
        box.setInformativeText(
            "A flight whose window has settled is cached for ever, because "
            "its history cannot change. Clearing it means spending those "
            "metered requests again the next time the same flight is "
            "fetched.\n\n"
            "Worth doing to reclaim the space, or to re-fetch a flight from "
            "the last day or two that OpenSky may still be backfilling.")
        box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        if box.exec() != QtWidgets.QMessageBox.StandardButton.Discard:
            self.log.log("flight cache kept")
            return

        removed, freed = cache.purge()
        self.log.log(f"flight cache: removed {removed} file(s), "
                     f"freed {freed / 1e6:.1f} MB")
        self.statusBar().showMessage(
            f"flight cache cleared: {freed / 1e6:.1f} MB")

    def _load_trajectory(self, path: str, altitude_unit: str) -> None:
        """Read a trajectory from disk. No network, no account, no worker."""
        from rcta.core.io_flight import TrajectoryParseError, load_trajectory
        try:
            trajectory = load_trajectory(path, altitude_unit=altitude_unit)
        except (TrajectoryParseError, OSError, ValueError) as exc:
            self.log.log(f"cannot read trajectory: {exc}", "error")
            self.flight.set_status(f"Cannot read that trajectory: {exc}")
            return
        self.log.log(f"loaded trajectory {Path(path).name}: "
                     f"{trajectory.n} waypoints")
        self._apply_trajectory(trajectory)

    def _apply_trajectory(self, trajectory, *, force: bool = False) -> None:
        """Join a trajectory onto the track, or refuse and explain why.

        Refusing is the whole point of the position check, so the panel is
        cleared rather than left showing a previous result that the new,
        rejected trajectory might be mistaken for.
        """
        if self._analysis is None:
            return
        self._trajectory = trajectory
        self.flight.set_trajectory_available(True)
        for warning in trajectory.warnings:
            self.log.log(warning, "warning")

        try:
            join = join_altitude(self._analysis.track, trajectory,
                                 force_match=force)
        except FlightMatchError as exc:
            report = exc.report
            self.flight.set_join(None)
            self.flight.set_status(
                f"<b style='color:{theme.current().bad}'>Refusing this trajectory.</b> "
                + report.describe().replace("\n", "<br>"))
            self.log.log(str(exc), "error")
            self.log.log(report.describe(), "warning")
            self._update_actions()
            self._offer_force(report)
            return
        except ValueError as exc:
            self.flight.set_join(None)
            self.flight.set_status(str(exc))
            self.log.log(f"cannot join trajectory: {exc}", "error")
            self._update_actions()
            return

        self.flight.set_unit(self._unit)
        self.flight.set_join(join)
        self.log.log(trajectory.describe())
        self.log.log(join.describe())
        self._update_actions()
        self.statusBar().showMessage(
            f"altitude joined onto {join.n_joined} of {join.track.n} samples")

    def _offer_force(self, report) -> None:
        """Ask once, in a dialog that states what forcing actually means."""
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle("Trajectory does not match this track")
        box.setText(
            f"The trajectory sits {report.residual_median_m / 1000.0:.0f} km "
            "from the positions this track recorded.")
        detail = (
            "That is almost always the wrong flight number, the wrong day, or "
            "the wrong leg of a multi-leg rotation.\n\n"
            "Joining anyway produces a smooth, plausible and entirely "
            "fictional altitude axis. Every result derived from it will be "
            "labelled as forced.")
        if report.offset_matters:
            detail += (f"\n\nA {report.clock_offset_s:+.0f} s clock offset "
                       "would reduce the residual to "
                       f"{report.residual_at_offset_m / 1000.0:.2f} km, so the "
                       "detector's clock may simply be wrong rather than the "
                       "flight.")
        box.setInformativeText(detail)
        box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Cancel
                               | QtWidgets.QMessageBox.StandardButton.Ignore)
        box.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        button = box.button(QtWidgets.QMessageBox.StandardButton.Ignore)
        button.setText("Join anyway")
        if box.exec() == QtWidgets.QMessageBox.StandardButton.Ignore:
            self.log.log("position match overridden by the user", "warning")
            self._apply_trajectory(self._trajectory, force=True)

    # ---- selection --------------------------------------------------------

    def _on_map_selection(self, mask, modifiers) -> None:
        if self._track is None:
            return
        drawn = Selection(mask=mask, source="map")
        current = self._selection or Selection.empty(self._track.n)
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            drawn = current | drawn
        elif modifiers & Qt.KeyboardModifier.AltModifier:
            drawn = current.without(drawn)
        self._broadcast(drawn)

    def _on_series_interval(self, t0: float, t1: float) -> None:
        if self._track is None:
            return
        self._broadcast(Selection.from_interval(self._track.time_s, t0, t1))

    def _on_selection_cleared(self) -> None:
        if self._track is not None:
            self.series.clear_selection_control()
            self._broadcast(Selection.empty(self._track.n))

    def _broadcast(self, selection) -> None:
        """The single point of truth for what is selected."""
        self._selection = selection
        self._applying = True
        try:
            self.map.set_selection(selection)
            self.series.set_selection(selection)
        finally:
            self._applying = False

        if selection.is_empty:
            self.selection_dock.set_summary(
                "no selection", self._analysis.summary.describe()
                if self._analysis else "")
        else:
            try:
                stats = summarise(self._track, mask=selection.mask)
                self.selection_dock.set_summary(selection.describe(),
                                                stats.describe())
            except ValueError as exc:
                self.selection_dock.set_summary(selection.describe(), str(exc))
        self._update_actions()
        self._update_status()

    def _on_hover(self, index: int) -> None:
        self.map.set_hover(index)
        self.series.set_hover(index)
        if self._track is not None and 0 <= index < self._track.n:
            self.statusBar().showMessage(self._track.point(index).describe())
        else:
            self._update_status()

    def _on_anomaly_row(self, *_) -> None:
        if self._applying or self._track is None:
            return
        rows = self.anomaly_view.selectionModel().selectedRows()
        if not rows:
            return
        mask = np.zeros(self._track.n, dtype=bool)
        for index in rows:
            anomaly = self.anomaly_model.row_object(index.row())
            if anomaly is not None:
                mask[anomaly.start_index:anomaly.stop_index] = True
        self._broadcast(Selection(mask=mask, source="table"))

    def _on_hotspot_row(self, *_) -> None:
        if self._applying or self._track is None:
            return
        rows = self.hotspot_view.selectionModel().selectedRows()
        if not rows:
            return
        hotspot = self.hotspot_model.row_object(rows[0].row())
        if hotspot is None:
            return
        mask = np.zeros(self._track.n, dtype=bool)
        for start in hotspot.anomaly_indices:
            mask[start] = True
        self._broadcast(Selection(mask=mask, source="table"))

    def _zoom_to_selection(self) -> None:
        if self._selection is not None and not self._selection.is_empty:
            self.map.zoom_to_selection(self._selection)
        else:
            self.map.zoom_to_track()

    # ---- background reference --------------------------------------------

    def _mark_reference(self) -> None:
        if self._track is None or self._selection is None or self._selection.is_empty:
            self.log.log("select a clean span first", "warning")
            return
        span = self._selection.time_span(self._track.time_s)
        self._reference_span = span
        self.analysis_dock.background.setCurrentText("reference")
        self.log.log(f"background reference set to {span[1] - span[0]:.0f} s "
                     f"of track ({self._selection.count} samples)")
        self._update_actions()
        self.analyse()

    def _clear_reference(self) -> None:
        self._reference_span = None
        self.analysis_dock.background.setCurrentText("rolling")
        self.log.log("reference background cleared")
        self._update_actions()
        self.analyse()

    # ---- worker results ---------------------------------------------------

    def _on_job_finished(self, name, token, result) -> None:
        if name == "flight":
            self._apply_trajectory(result)
            return
        if name == "grid":
            self._grid = result
            # describe() already includes the warnings; logging them again
            # would double every one of them.
            self.log.log(result.describe())
            self.statusBar().showMessage(
                f"heatmap: {result.coverage * 100:.0f} % of cells supported")

    def _on_job_failed(self, name, token, message) -> None:
        self.log.log(f"{name} failed: {message}", "error")
        if name == "flight":
            self._report_fetch_failure(message)

    def _report_fetch_failure(self, message: str) -> None:
        """Say in the dock that the lookup failed, and what to do about it.

        A failed fetch used to leave the first line of the exception in the
        caption and the status bar still reading "fetching DL2618...", which
        together suggest a slow request rather than one that is over. Both are
        corrected here, and the hint the worker preserved is shown, because
        the commonest failure by far -- no credentials -- is not a fault at
        all but a step the user has not taken yet, and the message alone does
        not say that the account is free.
        """
        # The worker puts the explanation before the first blank line and the
        # traceback after it. "hint:" is left spelled as the CLI spells it,
        # so the same failure reads the same way in both front ends.
        explanation = message.partition("\n\n")[0].strip()
        self.flight.set_status(
            f"<b style='color:{theme.current().bad}'>Could not fetch that flight.</b> "
            + _rich(explanation)
            + "<br>The <b>From file...</b> button beside <b>Fetch</b> reads a "
              "trajectory already on disk, which needs no account and no "
              "network. The full failure is in the Log dock.")
        self.statusBar().showMessage("flight lookup failed")

    def _on_job_progress(self, name, token, fraction) -> None:
        self.statusBar().showMessage(f"{name}: {fraction * 100:.0f} %")

    # ---- misc -------------------------------------------------------------

    def _on_unit_changed(self, unit: str) -> None:
        self._unit = unit
        self.flight.set_unit(unit)
        self.analyse()

    def _update_status(self) -> None:
        if self._analysis is None:
            return
        line = self._analysis.status_line()
        if self._selection is not None and not self._selection.is_empty:
            line += f"  |  {self._selection.describe()}"
        self.statusBar().showMessage(line)

    def _export(self, kind: str) -> None:
        if self._analysis is None:
            return
        # The altitude exports describe a join, not a track, so they are
        # unavailable until one exists rather than silently writing an empty
        # profile that looks like a measurement of nothing. The trajectory is
        # the exception: a refused one is still worth saving.
        if kind in ("samples", "profile", "report") and self._join() is None:
            self.log.log("no altitude joined yet: fetch a flight or load a "
                         "trajectory in the Flight dock first", "warning")
            return
        if kind == "trajectory" and self._trajectory is None:
            self.log.log("no trajectory loaded yet", "warning")
            return

        filters = {"csv": "CSV (*.csv)", "anomalies": "CSV (*.csv)",
                   "hotspots": "CSV (*.csv)", "geojson": "GeoJSON (*.geojson)",
                   "gpx": "GPX (*.gpx)", "json": "JSON (*.json)",
                   "samples": "CSV (*.csv)", "profile": "CSV (*.csv)",
                   "trajectory": "CSV (*.csv)", "report": "JSON (*.json)"}
        suggested = self._suggested_export_name(kind)
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, f"Export {kind}", suggested, filters[kind])
        if not path:
            return
        a = self._analysis
        writers = {
            "csv": lambda: track_to_csv(a.track, quality=a.quality,
                                        background=a.background, unit=self._unit),
            "geojson": lambda: track_to_geojson(a.track, quality=a.quality,
                                                unit=self._unit),
            "gpx": lambda: track_to_gpx(a.track, quality=a.quality),
            "anomalies": lambda: anomalies_to_csv(a.anomalies, unit=self._unit),
            "hotspots": lambda: hotspots_to_csv(a.hotspots, unit=self._unit),
            "json": lambda: report_to_json(a),
            "samples": lambda: altitude_samples_to_csv(
                self._join(), unit=self._unit,
                branch=self.flight.export_branch.currentText()),
            "profile": lambda: altitude_profile_to_csv(
                self._flight_profile(), unit=self._unit),
            "trajectory": lambda: trajectory_to_csv(self._trajectory),
            "report": lambda: flight_report_to_json(
                self._join(), self._flight_profile(), unit=self._unit,
                floor_m=float(self.flight.floor.value())),
        }
        try:
            Path(path).write_text(writers[kind](), encoding="utf-8")
        except OSError as exc:
            self.log.log(f"could not write {path}: {exc}", "error")
            return
        self.log.log(f"wrote {path}")

    def _suggested_export_name(self, kind: str) -> str:
        """A filename that says what the file is and which phase it covers.

        Named after the track, because these are files somebody will keep
        alongside it and will not thank us for calling "untitled". The phase is
        included for the two per-phase exports, because the whole reason that
        selector exists is to produce several files from one track and three
        files called ``samples.csv`` are worse than none.
        """
        stem = Path(self._path).stem if self._path else "track"
        branch = self.flight.export_branch.currentText()
        phase = "" if branch == "all" else f"-{branch}"
        names = {
            "csv": f"{stem}.csv",
            "geojson": f"{stem}.geojson",
            "gpx": f"{stem}.gpx",
            "json": f"{stem}-report.json",
            "anomalies": f"{stem}-anomalies.csv",
            "hotspots": f"{stem}-hotspots.csv",
            "samples": f"{stem}-altitude-samples{phase}.csv",
            "profile": f"{stem}-altitude-profile{phase}.csv",
            "trajectory": f"{stem}-trajectory.csv",
            "report": f"{stem}-flight-report.json",
            "image": f"{stem}-map.png",
            "svg": f"{stem}-map.svg",
        }
        return names.get(kind, "")

    def _export_image(self) -> None:
        """Render the map to a PNG with attribution baked into the pixels."""
        if self._analysis is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export map image", self._suggested_export_name("image"),
            "PNG image (*.png)")
        if not path:
            return

        from rcta.gui.map_export import attribution_lines, render_image

        provider = (self._providers.get(self._provider_id)
                    if self._provider_id else None)
        include_basemap = bool(provider and provider.allow_export
                               and self.map._basemap_enabled)
        if provider and self.map._basemap_enabled and not provider.allow_export:
            self.log.log(
                f"{provider.name} may not be redistributed in an exported "
                "image under its terms; exporting the data layers only",
                "warning")

        lines = attribution_lines(
            self._analysis, provider,
            colormap=self.display_dock.colormap.currentText(),
            scale=self.map._scale, include_basemap=include_basemap)

        size = self.map.size()
        image = render_image(self.map, size.width() * 2, size.height() * 2,
                             lines=lines)
        if image.save(path):
            self.log.log(f"wrote {path}")
        else:
            self.log.log(f"could not write {path}", "error")

    def _export_svg(self) -> None:
        """Vector export of the data layers. The raster basemap cannot come."""
        if self._analysis is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export map SVG", self._suggested_export_name("svg"),
            "SVG image (*.svg)")
        if not path:
            return
        from rcta.gui.map_export import render_svg
        try:
            render_svg(self.map, path)
        except Exception as exc:
            self.log.log(f"SVG export failed: {exc}", "error")
            return
        self.log.log(f"wrote {path} (data layers only; vector export cannot "
                     "include raster basemap tiles)")

    def _about(self) -> None:
        QtWidgets.QMessageBox.about(self, "About Radiacode Track Analyzer",
                                    LIMITATIONS)

    def closeEvent(self, event) -> None:
        self.tiles.cancel_all()
        self.jobs.shutdown()
        super().closeEvent(event)
