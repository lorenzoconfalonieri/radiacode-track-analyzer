"""The track map: projected metres, degree axes, and a colour-coded route.

Plots in EPSG:3857 metres so that a 1:1 aspect lock is exactly correct (the
projection is conformal) and so the basemap layer shares the coordinate
system. The axes are relabelled in degrees by :mod:`rcta.gui.geo_axis`, so the
map is fully usable with no basemap at all -- graticule, scale bar, and
readable coordinates -- which is deliberate: tile problems can never block the
analysis.

Two rendering decisions carry most of the performance and most of the honesty.

*One scatter item with a quantised shared-brush palette.* Giving every point
its own ``QBrush`` makes pyqtgraph allocate one symbol-atlas entry per distinct
brush: measured at 100 000 points that is 2.1 s to build, against 0.19 s when
the colours are quantised to 32 shared brushes. Splitting the track across one
item per colour bin is equally fast but scatters the point-to-index mapping
across items, which is precisely what hit-testing and linked brushing depend
on.

*The line breaks at dropouts.* A solid line across the bundled track's 763 s
gap would assert 103 km of surveyed route that was never observed, with a
fabricated colour at every pixel. ``arrayToQPath`` with a connect array breaks
it, and an optional off-by-default dotted connector shows *that* a gap existed
without claiming a path through it.
"""

from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from rcta.core import geo
from rcta.core.quality import polyline
from rcta.gui import theme
from rcta.gui.basemap import BasemapItem
from rcta.gui.geo_axis import GeoAxisItem, GroundScaleBar

Qt = QtCore.Qt

__all__ = ["MapPanel", "suggest_color_scale"]


def suggest_color_scale(lo: float, hi: float) -> str:
    """Log when the data actually spans decades, linear otherwise.

    The bundled flight runs 2.97 to 114 nSv/h between its first and
    ninety-ninth percentiles: on a linear ramp every ground-level sample
    collapses into the bottom three per cent of the colour axis. A walking
    survey spanning 80 to 120 nSv/h has the opposite problem, so this is not a
    setting that can simply be hardcoded either way.
    """
    return "log" if (lo > 0 and hi / lo >= 10.0) else "linear"


def _decade_ticks(lo: float, hi: float):
    """1/2/5 x 10^n values inside [lo, hi], for relabelling a log colour bar."""
    if lo <= 0 or hi <= lo:
        return [lo, hi]
    ticks = []
    exponent = math.floor(math.log10(lo))
    while True:
        base = 10.0 ** exponent
        for factor in (1.0, 2.0, 5.0):
            value = factor * base
            if value > hi:
                return ticks or [lo, hi]
            if value >= lo:
                ticks.append(value)
        exponent += 1
        if exponent > 12:
            return ticks or [lo, hi]


class _MapViewBox(pg.ViewBox):
    """Left-drag pans, as on every map. Shift or Alt drag selects."""

    rectDrawn = QtCore.Signal(object, object)   # QRectF, Qt.KeyboardModifiers

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.select_mode = False

    def mouseDragEvent(self, ev, axis=None):
        modifiers = ev.modifiers()
        selecting = self.select_mode or bool(
            modifiers & (Qt.KeyboardModifier.ShiftModifier
                         | Qt.KeyboardModifier.AltModifier))
        if ev.button() != Qt.MouseButton.LeftButton or not selecting:
            return super().mouseDragEvent(ev, axis)

        ev.accept()
        if ev.isFinish():
            self.rbScaleBox.hide()
            p0 = self.mapToView(ev.buttonDownPos())
            p1 = self.mapToView(ev.pos())
            self.rectDrawn.emit(QtCore.QRectF(p0, p1).normalized(), modifiers)
        else:
            self.updateScaleBox(ev.buttonDownPos(), ev.pos())


class MapPanel(QtWidgets.QWidget):
    """Track on aspect-locked projected axes, optionally over a basemap."""

    #: boolean numpy mask over the track's time-sorted sample index
    selectionDrawn = QtCore.Signal(object, object)   # mask, modifiers
    selectionCleared = QtCore.Signal()
    pointHovered = QtCore.Signal(int)
    viewChanged = QtCore.Signal(object)              # (lon0, lat0, lon1, lat1)
    colorLimitsChanged = QtCore.Signal(float, float)
    #: (message, pending tile count) for the status bar
    basemapStatus = QtCore.Signal(str, int)
    #: tile keys the view now needs, nearest the centre first
    tilesRequested = QtCore.Signal(object, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._track = None
        self._quality = None
        self._values = None
        self._x = np.zeros(0)
        self._y = np.zeros(0)
        self._limits = (0.0, 1.0)
        self._scale_mode = "auto"
        self._scale = "linear"
        self._field = "dose_rate"
        self._colormap = theme.DOSE_COLORMAP_DEFAULT
        self._accuracy_threshold = theme.ACCURACY_THRESHOLD_M
        self._show_gap_connectors = False
        self._palette: list = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self._glw)

        self.vb = _MapViewBox()
        self.plot = self._glw.addPlot(
            row=0, col=0, viewBox=self.vb,
            axisItems={"bottom": GeoAxisItem("bottom"),
                       "left": GeoAxisItem("left")})
        self.plot.showGrid(x=True, y=True, alpha=theme.GRATICULE_ALPHA)
        # Mercator is conformal, so 1:1 is exactly right at every latitude.
        self.vb.setAspectLocked(True, ratio=1.0)
        self.vb.setLimits(xMin=-geo.WORLD_M, xMax=geo.WORLD_M,
                          yMin=-geo.WORLD_M, yMax=geo.WORLD_M, minXRange=20.0)
        self.vb.rectDrawn.connect(self._on_rect)

        self._build_palette(self._colormap)
        self._make_items()

        self.colorbar = pg.ColorBarItem(
            colorMap=theme.dose_colormap(self._colormap), interactive=True,
            label="dose rate (nSv/h)")
        self._glw.addItem(self.colorbar, row=0, col=1)
        self.colorbar.sigLevelsChanged.connect(self._on_levels_changed)
        self._applying_levels = False

        self.basemap = BasemapItem()
        self.plot.addItem(self.basemap)
        self._basemap_enabled = False
        self._max_zoom = 19
        self._last_want = None

        # A continuous drag issues zero requests until the hand stops, which is
        # precisely what the OSM policy asks for. 120 ms is below the threshold
        # at which a pause feels deliberate.
        self._retile_timer = QtCore.QTimer(self)
        self._retile_timer.setSingleShot(True)
        self._retile_timer.setInterval(theme.RETILE_DEBOUNCE_MS)
        self._retile_timer.timeout.connect(self._retile)

        self.scalebar = GroundScaleBar()
        self.scalebar.setParentItem(self.vb)
        self.scalebar.anchor((1, 1), (1, 1), offset=(-25, -25))

        self.attribution = QtWidgets.QGraphicsSimpleTextItem("")
        self.attribution.setBrush(pg.mkBrush(theme.ATTRIBUTION_COLOR))
        self.attribution.setParentItem(self.vb)
        self.attribution.setZValue(1000)
        self.attribution.hide()

        self.vb.sigRangeChanged.connect(self._on_range_changed)
        self._glw.scene().sigMouseMoved.connect(self._on_mouse_moved)

    # ---- construction -----------------------------------------------------

    def _make_items(self) -> None:
        self.line = QtWidgets.QGraphicsPathItem()
        self.line.setPen(pg.mkPen(theme.TRACK_LINE_COLOR,
                                  width=theme.TRACK_LINE_WIDTH, cosmetic=True))
        self.line.setZValue(5)
        self.plot.addItem(self.line)

        self.gap_line = QtWidgets.QGraphicsPathItem()
        self.gap_line.setPen(pg.mkPen(theme.TRACK_GAP_COLOR, width=1.0,
                                      style=Qt.PenStyle.DotLine, cosmetic=True))
        self.gap_line.setZValue(4)
        self.gap_line.hide()
        self.plot.addItem(self.gap_line)

        self.halo = pg.ScatterPlotItem(
            size=theme.POINT_SIZE + 5, brush=None,
            pen=pg.mkPen(theme.SELECTION_COLOR, width=theme.SELECTION_HALO_WIDTH))
        self.halo.setZValue(8)
        self.plot.addItem(self.halo)

        self.points = pg.ScatterPlotItem(size=theme.POINT_SIZE, pen=None)
        self.points.setZValue(10)
        self.plot.addItem(self.points)

        # Poor fixes are hollow: the reading is real, the position is not.
        self.uncertain = pg.ScatterPlotItem(
            size=theme.POINT_SIZE, brush=None,
            pen=pg.mkPen(theme.SERIES_COLORS[4], width=theme.UNCERTAIN_PEN_WIDTH))
        self.uncertain.setZValue(11)
        self.plot.addItem(self.uncertain)

        self.hover = pg.ScatterPlotItem(
            size=theme.POINT_SIZE + 8, brush=None,
            pen=pg.mkPen(theme.HOVER_COLOR, width=1.5))
        self.hover.setZValue(20)
        self.plot.addItem(self.hover)

    def _build_palette(self, name: str) -> None:
        lut = theme.dose_colormap(name).getLookupTable(
            nPts=theme.DOSE_COLOR_LEVELS, alpha=True)
        self._palette = [pg.mkBrush(*rgba) for rgba in lut]

    # ---- public API (never emits) -----------------------------------------

    def set_track(self, track, quality=None) -> None:
        self._track = track
        self._quality = quality
        if track is None:
            for item in (self.points, self.uncertain, self.halo, self.hover):
                item.setData([], [])
            self.line.setPath(QtGui.QPainterPath())
            return

        self._x = track.merc_x
        self._y = track.merc_y
        self._update_values()
        self._rebuild_line()
        self._recolour()
        self.zoom_to_track()

    def set_color_field(self, name: str) -> None:
        self._field = name
        self._update_values()
        self._recolour()

    def set_colormap(self, name: str) -> None:
        self._colormap = name
        self._build_palette(name)
        self.colorbar.setColorMap(theme.dose_colormap(name))
        self._recolour()

    def set_color_scale(self, mode: str) -> None:
        self._scale_mode = mode
        self._update_values()
        self._recolour()

    def set_color_limits(self, lo: float, hi: float) -> None:
        self._limits = (float(lo), float(hi))
        self._recolour()

    def set_accuracy_threshold(self, metres: float) -> None:
        self._accuracy_threshold = float(metres)
        self._recolour()

    def set_gap_connectors(self, show: bool) -> None:
        self._show_gap_connectors = bool(show)
        self.gap_line.setVisible(bool(show))

    def set_selection(self, selection) -> None:
        if self._track is None or selection is None or selection.is_empty:
            self.halo.setData([], [])
            return
        rows = selection.mask & self._track.position_mask
        self.halo.setData(self._x[rows], self._y[rows])

    def set_hover(self, index: int) -> None:
        if (self._track is None or index < 0 or index >= self._track.n
                or not self._track.position_mask[index]):
            self.hover.setData([], [])
            return
        self.hover.setData([self._x[index]], [self._y[index]])

    def set_attribution(self, text: str) -> None:
        """Basemap attribution. Deliberately has no visibility toggle."""
        self.attribution.setText(text)
        self.attribution.setVisible(bool(text))
        if text:
            self.attribution.setPos(8, 8)

    def set_basemap_enabled(self, enabled: bool, max_zoom: int = 19) -> None:
        self._basemap_enabled = bool(enabled)
        self._max_zoom = int(max_zoom)
        if not enabled:
            self.basemap.clear()
            self._last_want = None
        else:
            self._retile()

    def set_tile(self, key, data: bytes) -> bool:
        return self.basemap.set_tile(key, data)

    def clear_basemap(self) -> None:
        self.basemap.clear()
        self._last_want = None

    def zoom_to_track(self) -> None:
        if self._track is None:
            return
        mask = self._track.position_mask
        if not mask.any():
            return
        x, y = self._x[mask], self._y[mask]
        self._set_view(float(x.min()), float(y.min()),
                       float(x.max()), float(y.max()))

    def zoom_to_selection(self, selection) -> None:
        if self._track is None or selection is None or selection.is_empty:
            return
        rows = selection.mask & self._track.position_mask
        if not rows.any():
            return
        x, y = self._x[rows], self._y[rows]
        self._set_view(float(x.min()), float(y.min()),
                       float(x.max()), float(y.max()))

    def set_select_mode(self, on: bool) -> None:
        self.vb.select_mode = bool(on)

    def view_bounds_lonlat(self):
        rect = self.vb.viewRect()
        lon0, lat0 = geo.web_mercator_to_lonlat(rect.left(), rect.bottom())
        lon1, lat1 = geo.web_mercator_to_lonlat(rect.right(), rect.top())
        return (float(lon0), float(lat0), float(lon1), float(lat1))

    def metres_per_device_pixel(self) -> float:
        width_px = max(self.vb.width(), 1.0)
        ratio = self.devicePixelRatioF() if hasattr(self, "devicePixelRatioF") else 1.0
        return float(self.vb.viewRect().width()) / (width_px * max(ratio, 1.0))

    # ---- internals --------------------------------------------------------

    def _set_view(self, x0, y0, x1, y1) -> None:
        # A stationary survey has a zero-width bbox, which would send
        # autoRange to infinity; expand it about its centre instead.
        if x1 - x0 < theme.MIN_EXTENT_M:
            cx = 0.5 * (x0 + x1)
            x0, x1 = cx - theme.MIN_EXTENT_M / 2, cx + theme.MIN_EXTENT_M / 2
        if y1 - y0 < theme.MIN_EXTENT_M:
            cy = 0.5 * (y0 + y1)
            y0, y1 = cy - theme.MIN_EXTENT_M / 2, cy + theme.MIN_EXTENT_M / 2
        pad = 0.05 * max(x1 - x0, y1 - y0)
        self.vb.setRange(QtCore.QRectF(x0 - pad, y0 - pad,
                                       (x1 - x0) + 2 * pad, (y1 - y0) + 2 * pad),
                         padding=0.0)

    def _update_values(self) -> None:
        if self._track is None:
            return
        if self._field == "count_rate":
            self._values = self._track.count_rate
            label, unit = "count rate", "cps"
        else:
            self._values = self._track.dose_rate
            label, unit = "dose rate", "nSv/h"

        finite = self._values[np.isfinite(self._values)]
        if finite.size:
            # Robust percentiles, so one spurious sample cannot consume the ramp.
            lo = float(np.percentile(finite, 1))
            hi = float(np.percentile(finite, 99))
            if hi <= lo:
                lo, hi = float(finite.min()), float(finite.max()) or 1.0
            self._limits = (lo, hi)

        self._scale = (suggest_color_scale(*self._limits)
                       if self._scale_mode == "auto" else self._scale_mode)
        self.colorbar.setLabel("left", f"{label} ({unit}, {self._scale})")

    def _rebuild_line(self) -> None:
        track, quality = self._track, self._quality
        x, y = polyline(track, quality)
        finite = np.isfinite(x) & np.isfinite(y)
        connect = finite.astype(np.uint8)
        # Never join across an inserted NaN or a hidden sample.
        connect[:-1] &= finite[1:]
        path = pg.functions.arrayToQPath(np.nan_to_num(x), np.nan_to_num(y),
                                         connect=connect)
        self.line.setPath(path)
        self._rebuild_gap_connectors()

    def _rebuild_gap_connectors(self) -> None:
        track, quality = self._track, self._quality
        if track is None or quality is None or not quality.gap_after.any():
            self.gap_line.setPath(QtGui.QPainterPath())
            return
        path = QtGui.QPainterPath()
        for i in np.flatnonzero(quality.gap_after):
            if not (track.position_mask[i] and track.position_mask[i + 1]):
                continue
            path.moveTo(float(self._x[i]), float(self._y[i]))
            path.lineTo(float(self._x[i + 1]), float(self._y[i + 1]))
        self.gap_line.setPath(path)

    def _recolour(self) -> None:
        if self._track is None or self._values is None:
            return
        lo, hi = self._limits
        values = self._values

        if self._scale == "log":
            positive = values[values > 0]
            floor = float(positive.min()) if positive.size else 1e-6
            safe_lo = max(lo, floor)
            with np.errstate(divide="ignore", invalid="ignore"):
                u = ((np.log10(np.maximum(values, floor)) - math.log10(safe_lo))
                     / max(math.log10(hi) - math.log10(safe_lo), 1e-9))
        else:
            u = (values - lo) / max(hi - lo, 1e-9)

        index = np.clip((np.nan_to_num(u) * (theme.DOSE_COLOR_LEVELS - 1) + 0.5),
                        0, theme.DOSE_COLOR_LEVELS - 1).astype(np.intp)

        good = self._track.position_mask & (
            self._track.accuracy_m <= self._accuracy_threshold)
        poor = self._track.position_mask & ~good

        brushes = [self._palette[i] for i in index[good]]
        self.points.setData(self._x[good], self._y[good], brush=brushes,
                            pen=None, size=theme.POINT_SIZE)

        pens = [pg.mkPen(self._palette[i].color(),
                         width=theme.UNCERTAIN_PEN_WIDTH) for i in index[poor]]
        self.uncertain.setData(self._x[poor], self._y[poor], brush=None, pen=pens,
                               size=theme.POINT_SIZE)

        self._applying_levels = True
        try:
            if self._scale == "log":
                self.colorbar.setLevels(low=math.log10(max(lo, 1e-12)),
                                        high=math.log10(max(hi, 1e-11)))
                # The bar's own axis is in log10 units. Left alone it labels a
                # 100 nSv/h tick "2.0", which a reader will take at face value.
                self.colorbar.axis.setTicks(
                    [[(math.log10(v), f"{v:g}") for v in _decade_ticks(lo, hi)],
                     []])
            else:
                self.colorbar.setLevels(low=lo, high=hi)
                self.colorbar.axis.setTicks(None)
        finally:
            self._applying_levels = False

    def _on_levels_changed(self) -> None:
        if self._applying_levels:
            return
        lo, hi = self.colorbar.levels()
        # In log mode the bar's handles live in log10 units; undo that before
        # the value leaves the panel.
        if self._scale == "log":
            lo, hi = 10.0 ** float(lo), 10.0 ** float(hi)
        self.colorLimitsChanged.emit(float(lo), float(hi))

    def _on_range_changed(self, *_) -> None:
        self.scalebar.update_for(self.vb.viewRect())
        # Repaint immediately with whatever is already decoded, and only ask
        # the network for anything once the gesture has settled.
        self.basemap.update()
        if self._basemap_enabled:
            self._retile_timer.start()
        self.viewChanged.emit(self.view_bounds_lonlat())

    def _retile(self) -> None:
        if not self._basemap_enabled:
            return
        zoom = geo.zoom_for_resolution(self.metres_per_device_pixel(),
                                       self._max_zoom)
        rect = self.vb.viewRect()
        keys = geo.tiles_for_bounds(rect.left(), rect.bottom(),
                                    rect.right(), rect.top(), zoom)
        if not keys:
            return
        # An unchanged want-set means a pure repaint -- a resize, a selection
        # change -- and must cost nothing.
        signature = (zoom, keys[0], keys[-1], len(keys))
        if signature == self._last_want:
            return
        self._last_want = signature

        from rcta.gui.tiles import TileKey
        tile_keys = [TileKey(*k) for k in keys]
        self.basemap.set_visible(zoom, tile_keys)
        self.tilesRequested.emit(tile_keys, zoom)

    def _on_rect(self, rect, modifiers) -> None:
        if self._track is None:
            return
        from rcta.core.selection import Selection

        selection = Selection.from_rect(self._x, self._y, rect.left(),
                                        rect.bottom(), rect.right(), rect.top())
        mask = selection.mask & self._track.position_mask
        if not mask.any() and not (modifiers & (Qt.KeyboardModifier.ShiftModifier
                                                | Qt.KeyboardModifier.AltModifier)):
            self.selectionCleared.emit()
            return
        self.selectionDrawn.emit(mask, modifiers)

    def _on_mouse_moved(self, position) -> None:
        if self._track is None:
            return
        if not self.plot.sceneBoundingRect().contains(position):
            self.pointHovered.emit(-1)
            return
        point = self.vb.mapSceneToView(position)
        mask = self._track.position_mask
        if not mask.any():
            return
        rows = np.flatnonzero(mask)
        d2 = ((self._x[rows] - point.x()) ** 2 + (self._y[rows] - point.y()) ** 2)
        nearest = int(rows[int(np.argmin(d2))])
        # Only report a hover when the cursor is genuinely near the track.
        tolerance = 12.0 * self.metres_per_device_pixel()
        self.pointHovered.emit(nearest if math.sqrt(float(d2.min())) < tolerance
                               else -1)
