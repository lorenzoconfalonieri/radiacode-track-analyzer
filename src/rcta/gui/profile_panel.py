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

from rcta.core.profile import (correlate_speed_dose, distribution,
                               resample_by_distance)
from rcta.gui import theme

Qt = QtCore.Qt

__all__ = ["ProfilePanel"]


class ProfilePanel(QtWidgets.QWidget):
    """Distance profile, histograms and the speed/dose scatter."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._track = None

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

    # ---- public API (never emits) -----------------------------------------

    def set_track(self, track) -> None:
        self._track = track
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
        self._profile_curve.setData(profile.distance_m / 1000.0,
                                    profile.dose_rate, connect="finite")

        histogram = distribution(track.dose_rate, unit="nSv/h")
        if histogram.counts.size:
            self._hist_curve.setData(histogram.edges, histogram.counts)

        speed = track.speed_m_s[1:]
        dose = track.dose_rate[1:]
        self._scatter.setData(speed, dose)

        correlation = correlate_speed_dose(track, n_boot=200)
        self.caption.setText(
            f"{profile.describe()}   |   "
            f"speed vs dose: rho = {correlation.spearman_rho:+.3f}, "
            f"p = {correlation.p_effective:.3g} after correcting for "
            f"{correlation.n} samples worth {correlation.n_effective:.0f} "
            f"independent ones. {correlation.caveat}")
