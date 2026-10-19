"""Dose rate against altitude, and the controls that decide it may be drawn.

Dumb like every other panel: it emits gestures, renders what it is handed, and
makes no core call of its own. ``MainWindow`` owns the fetch and the join.

Three choices here are about not misleading the eye, which on this particular
plot is easy to do.

*The ordinate is logarithmic by default.* The profile is an exponential over
more than a decade, so on a linear axis the entire troposphere below cruise is
pressed into the bottom two rows of pixels and the plot says only "high at the
top". On a log axis a constant scale height is a straight line, so the fitted
number and the picture check each other, and a departure from a single
exponential is visible as curvature rather than hidden in the floor.

*Climb and descent are drawn as separate series, not pooled.* They are two
traverses of the same atmosphere hours apart, and the honest error bar on this
measurement is how far apart they land. Averaging them into one curve destroys
exactly the information that says whether the profile is a function of
altitude at all.

*Error bars are always drawn.* Without them a reader cannot tell the cruise
bin -- thousands of samples, a fraction of a per cent -- from a climb bin
holding four, and those are adjacent points on the same curve. The bars are
the larger of the counting and observed errors, for the reason given in
:class:`rcta.core.flight.AltitudeProfile`.

The panel refuses to render an unvalidated join for the same reason the CLI
refuses to print one: a wrong altitude axis produces a smooth, plausible and
entirely fictional curve, and a plot is far more persuasive than a table.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from rcta.core.dose import convert_dose_rate
from rcta.core.flight import (DEFAULT_ALTITUDE_STEP_M, DEFAULT_FIT_FLOOR_M,
                              bin_by_altitude, compare_branches,
                              fit_scale_height)
from rcta.gui import theme

Qt = QtCore.Qt

__all__ = ["FlightPanel"]

#: Okabe-Ito assignments for the three phases, kept distinct from the dose
#: colormap used on the map so the two never look like the same encoding.
_BRANCH_STYLE = {
    "climb": (theme.DOSE_COLOR, "o", "climb"),
    "descent": (theme.COUNT_COLOR, "t", "descent"),
    "level": (theme.EVIDENCE_COLORS["strong"], "s", "level"),
}


def _set_bars(bars, x, y, *, height=None, top=None, bottom=None) -> None:
    """Set an ``ErrorBarItem``'s geometry, clearing whatever it held before.

    ``ErrorBarItem.setData`` *merges* into a persistent options dict, and
    ``drawPath`` gives ``height`` precedence over ``top``/``bottom``. So a
    partial update leaves the previous call's arrays in place: clearing the
    plot with ``height`` and then drawing with ``top``/``bottom`` leaves a
    stale, differently-shaped ``height``, and ``y - height / 2`` raises inside
    ``paint()``. A Python exception in a Qt paint handler does not propagate --
    the item simply stays dirty and is repainted for ever, which shows up as
    ``processEvents`` never draining and the application, or a test session,
    hanging with no traceback. Passing every dimension on every call, ``None``
    included, is the only safe way to use this item.
    """
    bars.setData(x=x, y=y, height=height, top=top, bottom=bottom)


class FlightPanel(QtWidgets.QWidget):
    """Flight lookup, and the dose-versus-altitude profile it produces."""

    #: (flight_number, date_iso) -- fetch this flight from the provider.
    fetchRequested = QtCore.Signal(str, str)
    #: A local trajectory file was chosen.
    fileRequested = QtCore.Signal(str, str)
    #: An export was asked for: "samples", "profile", "trajectory" or "report".
    exportRequested = QtCore.Signal(str)
    #: The cached provider responses should be deleted.
    cacheClearRequested = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._join = None
        self._unit = "nSv/h"
        self._start_folder = ""

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self._make_lookup_row())
        layout.addLayout(self._make_display_row())
        layout.addLayout(self._make_export_row())

        self._glw = pg.GraphicsLayoutWidget()
        # A log ordinate spanning two decades needs vertical room: at the
        # height the tabbed docks open to, the decade labels overlap into an
        # unreadable stack and the whole climb is a few pixels tall. Asking for
        # a floor makes the dock open large enough to be read.
        self._glw.setMinimumHeight(300)
        layout.addWidget(self._glw, 1)
        self.plot = self._glw.addPlot(row=0, col=0)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setLabel("bottom", "barometric altitude (m)")
        self.plot.setLabel("left", f"dose rate ({self._unit})")
        self.plot.getAxis("bottom").enableAutoSIPrefix(False)
        self.plot.getAxis("left").enableAutoSIPrefix(False)
        self.plot.setLogMode(x=False, y=True)
        self.legend = self.plot.addLegend(offset=(10, 10))

        self._series = {}
        for branch, (colour, symbol, label) in _BRANCH_STYLE.items():
            item = pg.PlotDataItem(
                pen=None, symbol=symbol, symbolSize=6,
                symbolBrush=pg.mkBrush(colour), symbolPen=None, name=label)
            self.plot.addItem(item)
            bars = pg.ErrorBarItem(pen=pg.mkPen(colour, width=1.0))
            self.plot.addItem(bars)
            self._series[branch] = (item, bars)

        self._fit_curve = self.plot.plot(
            pen=pg.mkPen(theme.LIGHT.fit, width=1.2, style=Qt.PenStyle.DashLine),
            name="fitted exponential")

        self.caption = QtWidgets.QLabel("")
        self.caption.setWordWrap(True)
        # Set explicitly rather than relying on auto-detection: most captions
        # here carry only entities such as &plusmn;, which does not look like
        # markup to Qt's heuristic, so the plain-text branch would render the
        # entity literally on some captions and not others.
        self.caption.setTextFormat(Qt.TextFormat.RichText)
        # A failed fetch reports here, and the actionable half of that report
        # is a sign-up URL. A default QLabel is neither selectable nor
        # clickable, so the address would be readable and unreachable.
        self.caption.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.LinksAccessibleByMouse)
        self.caption.setOpenExternalLinks(True)
        self.caption.setContentsMargins(6, 2, 6, 4)
        layout.addWidget(self.caption)
        self.set_join(None)
        theme.follow(self)

    def restyle(self, t) -> None:
        """Colours that have to change with the theme to stay visible."""
        theme.style_graphics_view(self._glw, t)
        self._fit_curve.setPen(pg.mkPen(t.fit, width=1.2,
                                        style=Qt.PenStyle.DashLine))

    # ---- construction -----------------------------------------------------

    def _make_lookup_row(self):
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(6, 4, 6, 0)

        row.addWidget(QtWidgets.QLabel("Flight"))
        self.flight = QtWidgets.QLineEdit()
        self.flight.setPlaceholderText("DL2618")
        self.flight.setMaximumWidth(90)
        self.flight.returnPressed.connect(self._emit_fetch)
        row.addWidget(self.flight)

        row.addWidget(QtWidgets.QLabel("Date"))
        self.date = QtWidgets.QDateEdit()
        self.date.setDisplayFormat("yyyy-MM-dd")
        self.date.setCalendarPopup(True)
        self.date.setMaximumWidth(110)
        self.date.setToolTip(
            "The day of the flight at the departure airport -- the date on "
            "the boarding pass, not the UTC one. They differ for an evening "
            "departure. Pre-filled from the track, so it is already right "
            "unless the detector's clock was wrong.")
        # QDateEdit defaults to the year 2000, which reads as a real choice.
        # Overwritten by set_default_date as soon as a track is loaded.
        self.date.setDate(QtCore.QDate.currentDate())
        row.addWidget(self.date)

        self.fetch_button = QtWidgets.QPushButton("Fetch")
        self.fetch_button.setToolTip(
            "Look the flight up with the trajectory provider. Historical data "
            "needs free OpenSky credentials; see Help.")
        self.fetch_button.clicked.connect(self._emit_fetch)
        row.addWidget(self.fetch_button)

        self.file_button = QtWidgets.QPushButton("From file...")
        self.file_button.setToolTip(
            "Read a trajectory already on disk. Needs no account and no "
            "network.")
        self.file_button.clicked.connect(self.choose_file)
        row.addWidget(self.file_button)

        self.clear_cache_button = QtWidgets.QPushButton("Clear cache")
        self.clear_cache_button.setToolTip(
            "Delete every cached provider response. A settled flight is "
            "cached for ever because its history cannot change, so clearing "
            "it means paying for those requests again -- worth it to reclaim "
            "the space, or to re-fetch a recent flight OpenSky is still "
            "backfilling.")
        self.clear_cache_button.clicked.connect(self.cacheClearRequested)
        row.addWidget(self.clear_cache_button)

        row.addWidget(QtWidgets.QLabel("Altitude in"))
        self.altitude_unit = QtWidgets.QComboBox()
        self.altitude_unit.addItems(["auto", "m", "ft"])
        self.altitude_unit.setToolTip(
            "Unit of the altitude column in the file. Most flight trackers "
            "export feet; OpenSky exports metres.")
        row.addWidget(self.altitude_unit)
        row.addStretch(1)
        return row

    def _make_display_row(self):
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(6, 0, 6, 0)

        row.addWidget(QtWidgets.QLabel("Bin"))
        self.step = QtWidgets.QDoubleSpinBox()
        self.step.setRange(50.0, 5000.0)
        self.step.setValue(DEFAULT_ALTITUDE_STEP_M)
        self.step.setSingleStep(250.0)
        self.step.setSuffix(" m")
        self.step.valueChanged.connect(self._refresh)
        row.addWidget(self.step)

        row.addWidget(QtWidgets.QLabel("Statistic"))
        self.statistic = QtWidgets.QComboBox()
        self.statistic.addItems(["mean", "median", "max", "min"])
        self.statistic.currentTextChanged.connect(self._refresh)
        row.addWidget(self.statistic)

        row.addWidget(QtWidgets.QLabel("Fit floor"))
        self.floor = QtWidgets.QDoubleSpinBox()
        self.floor.setRange(0.0, 12000.0)
        self.floor.setValue(DEFAULT_FIT_FLOOR_M)
        self.floor.setSingleStep(500.0)
        self.floor.setSuffix(" m")
        self.floor.setToolTip(
            "Bins below this are excluded from the exponential fit, where "
            "terrestrial gamma is comparable to the cosmic component.")
        self.floor.valueChanged.connect(self._refresh)
        row.addWidget(self.floor)

        self.log_y = QtWidgets.QCheckBox("Log dose")
        self.log_y.setChecked(True)
        self.log_y.setToolTip(
            "A constant scale height is a straight line on a log ordinate. "
            "On a linear one the whole climb is pressed into the floor.")
        self.log_y.toggled.connect(self._set_log)
        row.addWidget(self.log_y)

        self.split = QtWidgets.QCheckBox("Split climb/descent")
        self.split.setChecked(True)
        self.split.setToolTip(
            "Two traverses of the same atmosphere hours apart. How far apart "
            "they land is the honest error bar on this measurement.")
        self.split.toggled.connect(self._refresh)
        row.addWidget(self.split)
        row.addStretch(1)
        return row

    def _make_export_row(self):
        """Buttons, because this plot is not meant to be the final figure.

        The binned profile the panel draws embeds three choices -- bin width,
        statistic, error bar -- and somebody preparing a figure for print will
        want their own. So the per-sample export comes first and is the wide
        button: it is the one that hands over the data with none of those
        decisions baked in.
        """
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(6, 0, 6, 4)
        row.addWidget(QtWidgets.QLabel("Export"))

        self._export_buttons = {}
        for kind, label, tip in (
            ("samples", "Samples CSV...",
             "One row per joined sample: altitude, dose rate, count rate, "
             "estimated counts, integration time and phase of flight. Bin it, "
             "fit it and plot it however you like."),
            ("profile", "Profile CSV...",
             "The binned profile as drawn, with both the counting and the "
             "observed error on every bin."),
            ("trajectory", "Trajectory CSV...",
             "The flight path itself. Save it once and re-use it with no "
             "network and no account."),
            ("report", "Report JSON...",
             "Everything numeric, including the position-match residuals that "
             "say whether this join can be trusted."),
        ):
            button = QtWidgets.QPushButton(label)
            button.setToolTip(tip)
            # Disabled until there is something to write. A QPushButton is
            # enabled by default, so the empty state has to be set explicitly.
            button.setEnabled(False)
            button.clicked.connect(
                lambda _=False, k=kind: self.exportRequested.emit(k))
            row.addWidget(button)
            self._export_buttons[kind] = button

        row.addSpacing(12)
        row.addWidget(QtWidgets.QLabel("Phase"))
        self.export_branch = QtWidgets.QComboBox()
        self.export_branch.addItems(["all", "climb", "descent", "level"])
        self.export_branch.setToolTip(
            "Restrict the sample and profile exports to one phase of flight. "
            "The climb and the descent are independent traverses of the same "
            "atmosphere, so exporting them separately is often the point.")
        row.addWidget(self.export_branch)
        row.addStretch(1)
        return row

    # ---- gestures ---------------------------------------------------------

    def _emit_fetch(self) -> None:
        text = self.flight.text().strip()
        if not text:
            # Used to return in silence, so the button looked broken.
            self.set_status("Type a flight number first, such as DL2618 or "
                            "DAL2618, or load a trajectory with "
                            "<b>From file...</b>")
            self.flight.setFocus(Qt.FocusReason.OtherFocusReason)
            return
        self.fetchRequested.emit(text, self.date.date().toString("yyyy-MM-dd"))

    def set_start_folder(self, folder: str) -> None:
        """Where the trajectory dialog opens: beside the loaded track, where a
        trajectory exported for that flight is most likely to have been
        saved. Set by the window, which knows the path; never emits."""
        self._start_folder = str(folder)

    def choose_file(self) -> None:
        """Public: the File menu offers the same gesture."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open flight trajectory", self._start_folder,
            "Trajectory (*.json *.csv *.tsv *.txt);;All files (*)")
        if path:
            self.fileRequested.emit(path, self.altitude_unit.currentText())

    def _set_log(self, on: bool) -> None:
        self.plot.setLogMode(x=False, y=bool(on))
        self._refresh()

    # ---- rendering (never emits) ------------------------------------------

    def set_unit(self, unit: str) -> None:
        self._unit = unit
        self.plot.setLabel("left", f"dose rate ({unit})")
        self._refresh()

    def set_default_date(self, when) -> None:
        """Pre-fill the date from the track, so it is right by default.

        ``when`` is expected on the departure airport's clock, not in UTC:
        the field is read as a local date, so a UTC default would be a day
        out on exactly the evening departures that motivated reading it
        locally.
        """
        blocked = self.date.blockSignals(True)
        try:
            self.date.setDate(QtCore.QDate(when.year, when.month, when.day))
        finally:
            self.date.blockSignals(blocked)

    def set_status(self, message: str) -> None:
        self.caption.setText(message)

    def set_join(self, join) -> None:
        """Render a validated join, or clear the plot when given None."""
        self._join = join
        # Greyed out rather than writing an empty file: these three name a
        # measurement, and there is nothing to name until a join exists.
        for kind in ("samples", "profile", "report"):
            self._export_buttons[kind].setEnabled(join is not None)
        if join is not None:
            self.set_trajectory_available(True)
        self._refresh()

    def set_trajectory_available(self, available: bool) -> None:
        """Enable the trajectory export, which does not need a valid join.

        Deliberately separate from :meth:`set_join`. A trajectory that was
        *refused* -- wrong flight, wrong day -- is still worth saving, because
        opening it in something else is how a user works out why it did not
        match. Tying this button to the join would remove the file exactly when
        it is most wanted.
        """
        self._export_buttons["trajectory"].setEnabled(bool(available))

    def _clear(self) -> None:
        empty = np.zeros(0)
        for item, bars in self._series.values():
            item.setData(empty, empty)
            _set_bars(bars, empty, empty)
        self._fit_curve.setData(empty, empty)

    def _refresh(self, *_) -> None:
        join = self._join
        if join is None:
            self._clear()
            if not self.caption.text():
                self.caption.setText(
                    "No altitude yet. The track format carries none: name the "
                    "flight and fetch it, or load a trajectory file.")
            return

        step = float(self.step.value())
        statistic = self.statistic.currentText()
        split = self.split.isChecked()

        def profile_for(branch):
            return bin_by_altitude(join, step_m=step, statistic=statistic,
                                   branch=branch)

        headline = profile_for("all")
        # Clear every series first. Populating only the ones in use would
        # leave the descent curve from the previous, split, render sitting on
        # the plot after the user pools the branches -- drawn in the descent
        # colour, with a legend entry, and stale.
        self._clear()
        drawn = ({name: profile_for(name)
                  for name in ("climb", "descent", "level")} if split
                 else {"climb": headline})

        for name, profile in drawn.items():
            item, bars = self._series[name]
            if profile.altitude_m.size == 0:
                continue
            m = profile.occupied & np.isfinite(profile.dose_rate)
            if not m.any():
                continue
            x = profile.altitude_m[m]
            y = convert_dose_rate(profile.dose_rate[m], self._unit)
            err = convert_dose_rate(profile.dose_sem[m], self._unit)
            item.setData(x, y)
            # pyqtgraph draws error bars in data coordinates, so under a log
            # ordinate they must be given in decades or they land in the wrong
            # place -- symmetric in dose is asymmetric in log(dose).
            if self.log_y.isChecked():
                # Asymmetric in log space: a bar symmetric in dose is not.
                # Clamped at 5 % of the value so a bin whose error exceeds its
                # own level cannot ask for log10 of something non-positive.
                centre = np.log10(np.maximum(y, 1e-12))
                lo = np.log10(np.maximum(y - err, y * 0.05))
                hi = np.log10(y + err)
                _set_bars(bars, x, centre, top=hi - centre, bottom=centre - lo)
            else:
                _set_bars(bars, x, y, height=2.0 * err)
        fit = fit_scale_height(headline, floor_m=float(self.floor.value()))
        self._draw_fit(fit, headline)
        self._caption(join, headline, fit)

    def _draw_fit(self, fit, profile) -> None:
        """Draw the fitted exponential in *linear* dose, always.

        The curve is a ``PlotDataItem``, and pyqtgraph applies the plot's log
        mode to those itself. Taking the logarithm here as well -- which the
        error bars below genuinely need, because ``ErrorBarItem`` is drawn in
        raw view coordinates and is not log-aware -- would apply it twice and
        flatten the fit onto the bottom of the axis.
        """
        if not fit.constrained:
            self._fit_curve.setData(np.zeros(0), np.zeros(0))
            return
        x = np.linspace(fit.fitted_low_m, fit.fitted_high_m, 64)
        y = fit.dose_at_reference * np.exp(
            (x - fit.reference_altitude_m) / fit.scale_height_m)
        self._fit_curve.setData(x, convert_dose_rate(y, self._unit))

    def _caption(self, join, profile, fit) -> None:
        match = join.match
        parts = []
        if match.forced:
            parts.append(
                f"<b style='color:{theme.current().bad}'>Position match FORCED</b>: the "
                f"trajectory sits {match.residual_median_m / 1000.0:.0f} km "
                "from the recorded fixes. This is very likely not the flight "
                "that produced this track, and everything below is fiction.")
        parts.append(
            f"{join.n_joined} of {join.track.n} samples joined "
            f"({join.coverage * 100.0:.0f} %), "
            f"{profile.altitude_m.size} bins of {profile.step_m:.0f} m. "
            f"{profile.scatter_note}.")
        if fit.constrained:
            parts.append(
                f"Scale height {fit.scale_height_m / 1000.0:.2f} "
                f"&plusmn; {fit.scale_height_sd_m / 1000.0:.2f} km "
                f"(doubles every {fit.doubling_m:.0f} m), "
                f"R&sup2; = {fit.r_squared:.3f}, "
                f"&chi;&sup2;/dof = {fit.chi2_per_dof:.1f}.")
            if fit.chi2_per_dof > 5.0:
                parts.append("A single exponential does not describe this "
                             "profile.")
        else:
            parts.append(f"Scale height not constrained: {fit.note}")

        comparison = compare_branches(join, step_m=max(profile.step_m, 1000.0))
        if comparison.n_shared:
            verdict = ("agree" if comparison.consistent
                       else "<b>disagree</b>")
            parts.append(
                f"Climb and descent {verdict} over {comparison.n_shared} "
                f"shared bins: median difference "
                f"{comparison.median_relative_difference * 100.0:.0f} %, "
                f"worst {comparison.max_abs_z:.1f}&sigma; against a "
                f"{comparison.threshold_sigma:.1f}&sigma; threshold.")
        self.caption.setText(" ".join(parts))
