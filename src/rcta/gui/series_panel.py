"""Dose and count rate against time, with the background and significance shown.

Three stacked, x-linked plots: the dose/count series, the count rate with its
estimated background drawn over it, and a significance strip with the decision
threshold marked. The strip exists so the user can see *why* a stretch of
track was or was not called an anomaly, rather than being handed a verdict
with the threshold invisible.

The panel is dumb: it emits what the user did and renders what it is told. It
never calls into the analysis layer, and its ``set_*`` methods never emit, so
there is no echo for ``MainWindow`` to guard against.

The one subtlety is the draggable region. It is a *control*, not the model. A
lasso on the map can select several disjoint stretches of an out-and-back
route, and a single region item cannot represent that. Rather than silently
coarsening the selection to its hull -- which would quietly include kilometres
the user did not select -- the region is hidden and the selected samples are
drawn as a highlight overlay with a caption saying how many intervals there
are.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from rcta.core.dose import convert_dose_rate
from rcta.gui import theme

Qt = QtCore.Qt

__all__ = ["SeriesPanel"]

#: Beyond this many contiguous runs, drawing a band per run costs more than it
#: communicates and the overlay alone carries the message.
MAX_RUN_BANDS = 64


class SeriesPanel(QtWidgets.QWidget):
    """Time series with a region control and a significance strip."""

    intervalSelected = QtCore.Signal(float, float)   # unix seconds
    intervalCleared = QtCore.Signal()
    pointHovered = QtCore.Signal(int)                # sample index, or -1

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._track = None
        self._background = None
        self._trace = None
        self._threshold = 0.0
        self._run_bands: list[pg.LinearRegionItem] = []
        self._applying = False
        self._unit = "nSv/h"
        self._selection = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self._glw)

        self.dose_plot = self._glw.addPlot(row=0, col=0)
        self.dose_plot.showGrid(x=True, y=True, alpha=0.15)
        self.dose_plot.setLabel("left", "dose rate (nSv/h)")
        self.dose_plot.getAxis("left").enableAutoSIPrefix(False)
        self.dose_plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.dose_plot.addLegend(offset=(-10, 10))

        self.count_plot = self._glw.addPlot(row=1, col=0)
        self.count_plot.showGrid(x=True, y=True, alpha=0.15)
        self.count_plot.setLabel("left", "count rate (cps)")
        self.count_plot.getAxis("left").enableAutoSIPrefix(False)
        self.count_plot.setXLink(self.dose_plot)

        self.sig_plot = self._glw.addPlot(row=2, col=0)
        self.sig_plot.showGrid(x=True, y=True, alpha=0.15)
        self.sig_plot.setLabel("left", "sigma")
        self.sig_plot.setLabel("bottom", "elapsed (s)")
        self.sig_plot.getAxis("left").enableAutoSIPrefix(False)
        self.sig_plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.sig_plot.setXLink(self.dose_plot)
        self.sig_plot.setMaximumHeight(130)

        # A fixed axis width keeps the three stacked plots' left edges aligned
        # and stops their labels colliding when the value ranges differ.
        for plot in (self.dose_plot, self.count_plot, self.sig_plot):
            plot.getAxis("left").setWidth(62)

        self._glw.ci.layout.setRowStretchFactor(0, 4)
        self._glw.ci.layout.setRowStretchFactor(1, 3)
        self._glw.ci.layout.setRowStretchFactor(2, 1)

        self._curves: dict[str, pg.PlotDataItem] = {}
        self._make_items()

        self._caption = QtWidgets.QLabel("")
        self._caption.setContentsMargins(6, 0, 6, 2)
        layout.addWidget(self._caption)
        self._show_plots = theme.empty_state(
            layout, self._glw,
            "Dose rate, count rate and significance against time appear "
            "here once a track is open.")
        theme.follow(self)

    def restyle(self, t) -> None:
        """Colours that have to change with the theme to stay visible."""
        theme.style_graphics_view(self._glw, t)
        self._curves["dose"].setPen(pg.mkPen(t.dose, width=1.2))
        self._curves["count"].setPen(pg.mkPen(t.count, width=1.2))
        self._hover_line.setPen(pg.mkPen(t.hover, width=0.8))

    # ---- construction -----------------------------------------------------

    def _make_items(self) -> None:
        self._curves["dose"] = self.dose_plot.plot(
            pen=pg.mkPen(theme.DOSE_COLOR, width=1.2), name="dose rate")
        self._curves["dose_selected"] = self.dose_plot.plot(
            pen=pg.mkPen(theme.SELECTION_COLOR, width=2.0), name="selection")
        self._curves["count"] = self.count_plot.plot(
            pen=pg.mkPen(theme.COUNT_COLOR, width=1.2))
        self._curves["background"] = self.count_plot.plot(
            pen=pg.mkPen(theme.BACKGROUND_COLOR, width=1.2,
                         style=Qt.PenStyle.DashLine))
        self._curves["significance"] = self.sig_plot.plot(
            pen=pg.mkPen(theme.SERIES_COLORS[2], width=1.0))

        self.region = pg.LinearRegionItem(brush=pg.mkBrush(213, 94, 0, 40),
                                          pen=pg.mkPen(theme.SELECTION_COLOR))
        self.region.setZValue(-10)
        self.region.hide()
        self.dose_plot.addItem(self.region)
        self.region.sigRegionChangeFinished.connect(self._on_region_changed)

        self._threshold_line = pg.InfiniteLine(
            angle=0, pen=pg.mkPen(theme.THRESHOLD_COLOR, width=1.0,
                                  style=Qt.PenStyle.DashLine))
        self.sig_plot.addItem(self._threshold_line)
        self._threshold_line.hide()

        self._hover_line = pg.InfiniteLine(
            angle=90, pen=pg.mkPen(theme.HOVER_COLOR, width=0.8))
        self.dose_plot.addItem(self._hover_line)
        self._hover_line.hide()

        self._glw.scene().sigMouseMoved.connect(self._on_mouse_moved)

    # ---- public API (never emits) -----------------------------------------

    def set_track(self, track, background=None, trace=None,
                  threshold: float = 0.0) -> None:
        self._track = track
        self._background = background
        self._trace = trace
        self._threshold = float(threshold)
        self._show_plots(track is not None)

        if track is None:
            for curve in self._curves.values():
                curve.setData([], [])
            self.region.hide()
            self._caption.setText("")
            return

        t = track.elapsed_s
        self._curves["dose"].setData(t, self._dose(track.dose_rate))
        self._curves["count"].setData(t, track.count_rate)
        self._curves["dose_selected"].setData([], [])

        if background is not None:
            self._curves["background"].setData(t, background.count_rate)
        else:
            self._curves["background"].setData([], [])

        if trace is not None:
            self._curves["significance"].setData(t, trace)
            self._threshold_line.setValue(self._threshold)
            self._threshold_line.setVisible(self._threshold > 0)
        else:
            self._curves["significance"].setData([], [])
            self._threshold_line.hide()

        self.dose_plot.enableAutoRange()
        self._caption.setText("")

    def set_unit(self, unit: str) -> None:
        """Display unit for the dose plot. Redraws; never emits."""
        self._unit = unit
        self.dose_plot.setLabel("left", f"dose rate ({unit})")
        if self._track is not None:
            self._curves["dose"].setData(self._track.elapsed_s,
                                         self._dose(self._track.dose_rate))
            self.set_selection(self._selection)

    def _dose(self, values):
        return convert_dose_rate(values, self._unit)

    def set_selection(self, selection) -> None:
        """Render a selection. The region item is a control, not the model."""
        self._selection = selection
        self._clear_run_bands()
        if self._track is None or selection is None or selection.is_empty:
            self.region.hide()
            self._curves["dose_selected"].setData([], [])
            self._caption.setText("")
            return

        t = self._track.elapsed_s
        # Break the highlight between runs so disjoint stretches are not joined.
        y = np.where(selection.mask, self._dose(self._track.dose_rate), np.nan)
        self._curves["dose_selected"].setData(t, y, connect="finite")

        self._applying = True
        try:
            if selection.is_interval:
                (start, stop), = selection.runs(limit=1)
                self.region.setRegion((float(t[start]), float(t[stop - 1])))
                self.region.show()
            else:
                # No single region can express this; showing one would silently
                # widen the selection to its hull.
                self.region.hide()
                self._draw_run_bands(t, selection)
        finally:
            self._applying = False

        self._caption.setText(selection.describe())

    def set_hover(self, index: int) -> None:
        if self._track is None or index < 0 or index >= self._track.n:
            self._hover_line.hide()
            return
        self._hover_line.setValue(float(self._track.elapsed_s[index]))
        self._hover_line.show()

    def clear_selection_control(self) -> None:
        self.region.hide()

    # ---- internals --------------------------------------------------------

    def _draw_run_bands(self, t, selection) -> None:
        for start, stop in selection.runs(limit=MAX_RUN_BANDS):
            band = pg.LinearRegionItem(
                values=(float(t[start]), float(t[stop - 1])),
                brush=pg.mkBrush(213, 94, 0, 25), pen=pg.mkPen(None),
                movable=False)
            band.setZValue(-20)
            self.dose_plot.addItem(band)
            self._run_bands.append(band)

    def _clear_run_bands(self) -> None:
        for band in self._run_bands:
            self.dose_plot.removeItem(band)
        self._run_bands.clear()

    def _on_region_changed(self) -> None:
        # Only a genuine user gesture emits; set_selection sets _applying.
        if self._applying or self._track is None:
            return
        lo, hi = self.region.getRegion()
        origin = float(self._track.time_s[0])
        self.intervalSelected.emit(origin + float(lo), origin + float(hi))

    def _on_mouse_moved(self, position) -> None:
        if self._track is None:
            return
        vb = self.dose_plot.vb
        if not self.dose_plot.sceneBoundingRect().contains(position):
            self.pointHovered.emit(-1)
            return
        x = float(vb.mapSceneToView(position).x())
        index = int(np.searchsorted(self._track.elapsed_s, x))
        index = max(0, min(index, self._track.n - 1))
        self.pointHovered.emit(index)
