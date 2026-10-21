"""Distance profile, distributions, and the speed/dose scatter.

Three views that answer questions the map and the time series cannot.

The distance profile re-expresses the survey against ground covered rather than
time elapsed, which is what matters when the platform stopped: an hour parked
at a junction is one point on this axis and a wide plateau on the time axis.
Empty bins are left as gaps rather than interpolated, for the reason given in
:func:`rcta.core.profile.resample_by_distance` -- a filled bin where no sample
exists is an invention.

The scatter carries its own health warning. A correlation between speed and
dose rate is real on almost any airborne or mountain track and means nothing
causal, because altitude drives both. The caption states the corrected p-value
and the confound together, because a scatter plot with a visible trend is the
single most persuasive way to mislead someone in this program.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from rcta.core.dose import convert_dose_rate
from rcta.core.profile import correlate_speed_dose, distribution, resample_by_distance
from rcta.gui import theme
from rcta.gui.geo_axis import nice_distance

Qt = QtCore.Qt

__all__ = ["ProfilePanel"]


def suggest_step(track) -> float:
    """A distance bin that suits the track: about 150 bins over the route,
    and never so fine that most bins are empty.

    A fixed 100 m left the bundled 1800 km flight with 17 923 bins, 15 118 of
    them empty; the profile, drawn with gaps at empty bins, was invisible.
    """
    along = track.distance_along_m
    total = float(along[-1]) if along.size else 0.0
    steps = np.diff(along)
    spacing = float(np.median(steps[steps > 0])) if (steps > 0).any() else 0.0
    return nice_distance(max(total / 150.0, 3.0 * spacing, 10.0))


class ProfilePanel(QtWidgets.QWidget):
    """Distance profile, histograms and the speed/dose scatter."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._unit = "nSv/h"
        self._track = None
        self._correlation = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        controls = QtWidgets.QHBoxLayout()
        controls.setContentsMargins(6, 4, 6, 0)
        controls.addWidget(QtWidgets.QLabel("Bin"))
        self.step = QtWidgets.QDoubleSpinBox()
        self.step.setRange(1.0, 100_000.0)
        self.step.setValue(100.0)
        self.step.setSuffix(" m")
        self.step.setSingleStep(50.0)
        self.step.valueChanged.connect(self._refresh)
        controls.addWidget(self.step)

        controls.addWidget(QtWidgets.QLabel("Statistic"))
        self.statistic = QtWidgets.QComboBox()
        self.statistic.addItems(["mean", "median", "max", "min"])
        self.statistic.currentTextChanged.connect(self._refresh)
        controls.addWidget(self.statistic)
        controls.addStretch(1)
        layout.addLayout(controls)

        self._glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self._glw)

        self.profile_plot = self._glw.addPlot(row=0, col=0, colspan=2)
        self.profile_plot.showGrid(x=True, y=True, alpha=0.15)
        self.profile_plot.setLabel("left", "dose rate (nSv/h)")
        self.profile_plot.setLabel("bottom", "distance along track (km)")
        for axis in ("left", "bottom"):
            self.profile_plot.getAxis(axis).enableAutoSIPrefix(False)

        self.hist_plot = self._glw.addPlot(row=1, col=0)
        self.hist_plot.showGrid(x=True, y=True, alpha=0.15)
        self.hist_plot.setLabel("left", "samples")
        self.hist_plot.setLabel("bottom", "dose rate (nSv/h)")
        for axis in ("left", "bottom"):
            self.hist_plot.getAxis(axis).enableAutoSIPrefix(False)

        self.scatter_plot = self._glw.addPlot(row=1, col=1)
        self.scatter_plot.showGrid(x=True, y=True, alpha=0.15)
        self.scatter_plot.setLabel("left", "dose rate (nSv/h)")
        self.scatter_plot.setLabel("bottom", "speed (m/s)")
        for axis in ("left", "bottom"):
            self.scatter_plot.getAxis(axis).enableAutoSIPrefix(False)

        self._profile_curve = self.profile_plot.plot(
            pen=pg.mkPen(theme.DOSE_COLOR, width=1.4))
        self._hist_curve = self.hist_plot.plot(
            stepMode="center", fillLevel=0,
            brush=pg.mkBrush(0, 114, 178, 90),
            pen=pg.mkPen(theme.DOSE_COLOR, width=1.0))
        self._scatter = pg.ScatterPlotItem(size=3, pen=None,
                                           brush=pg.mkBrush(0, 114, 178, 70))
        self.scatter_plot.addItem(self._scatter)

        self.caption = QtWidgets.QLabel("")
        self.caption.setWordWrap(True)
        self.caption.setContentsMargins(6, 2, 6, 4)
        layout.addWidget(self.caption)
        self._show_plots = theme.empty_state(
            layout, self._glw,
            "Dose rate along the route, its distribution and its relation "
            "to speed appear here once a track is open.")
        theme.follow(self)

    def restyle(self, t) -> None:
        """Colours that have to change with the theme to stay visible."""
        theme.style_graphics_view(self._glw, t)
        fill = pg.mkColor(t.dose)
        self._profile_curve.setPen(pg.mkPen(t.dose, width=1.4))
        fill.setAlpha(90)
        self._hist_curve.setPen(pg.mkPen(t.dose, width=1.0))
        self._hist_curve.setBrush(pg.mkBrush(fill))
        fill.setAlpha(70)
        self._scatter.setBrush(pg.mkBrush(fill))

    # ---- public API (never emits) -----------------------------------------

    def set_unit(self, unit: str) -> None:
        """Display unit for dose rate. Redraws; never emits."""
        self._unit = unit
        self._refresh()

    def set_track(self, track, correlation=None) -> None:
        """Draw ``track``; ``correlation`` is the analysis's own speed/dose
        result, quoted rather than recomputed so the caption cannot disagree
        with the report."""
        self._track = track
        self._show_plots(track is not None)
        if track is not None:
            blocked = self.step.blockSignals(True)
            try:
                self.step.setValue(suggest_step(track))
            finally:
                self.step.blockSignals(blocked)
        self._correlation = (correlation if correlation is not None or
                             track is None else correlate_speed_dose(track))
        self._refresh()

    def _refresh(self, *_) -> None:
        track = self._track
        if track is None:
            self._profile_curve.setData([], [])
            self._hist_curve.setData([], [])
            self._scatter.setData([], [])
            self.caption.setText("")
            return

        profile = resample_by_distance(track, step_m=self.step.value(),
                                       statistic=self.statistic.currentText())
        # NaN plus connect="finite" leaves empty bins as visible gaps rather
        # than drawing a line through ground that was never sampled.
        unit = self._unit
        for plot, axis in ((self.profile_plot, "left"), (self.hist_plot, "bottom"),
                           (self.scatter_plot, "left")):
            plot.setLabel(axis, f"dose rate ({unit})")
        dose = convert_dose_rate(track.dose_rate, unit)
        self._profile_curve.setData(profile.distance_m / 1000.0,
                                    convert_dose_rate(profile.dose_rate, unit),
                                    connect="finite")

        readings = track.valid_mask
        histogram = distribution(dose[readings], unit=unit)
        if histogram.counts.size:
            self._hist_curve.setData(histogram.edges, histogram.counts)

        readings = readings.copy()
        readings[0] = False                 # its speed is a copy of the next
        self._scatter.setData(track.speed_m_s[readings], dose[readings])

        correlation = self._correlation
        self.caption.setText(
            f"{profile.describe()}   |   "
            f"speed vs dose: rho = {correlation.spearman_rho:+.3f}, "
            f"p = {correlation.p_effective:.3g} after correcting for "
            f"{correlation.n} samples worth {correlation.n_effective:.0f} "
            f"independent ones. {correlation.caveat}")
