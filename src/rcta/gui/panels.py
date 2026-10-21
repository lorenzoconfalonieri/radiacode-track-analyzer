"""Dock widgets. Dumb by design: they emit gestures and render what they are told.

None of these import :mod:`rcta.core` analysis functions -- only the frozen
parameter dataclasses they assemble. ``MainWindow`` owns every connection and
makes every core call, so there is exactly one place where the analysis is
driven and exactly one place a selection is created.

A panel emits only in response to a *user* gesture. Its ``set_*`` methods never
emit, which is what removes the need for ``blockSignals`` gymnastics anywhere
in the wiring.
"""

from __future__ import annotations

from datetime import datetime, timezone

from PySide6 import QtCore, QtWidgets

from rcta.core.anomalies import ScanParams
from rcta.core.background import BackgroundParams
from rcta.core.dose import DOSE_UNITS
from rcta.core.quality import QualityParams
from rcta.gui import theme

Qt = QtCore.Qt

__all__ = ["AnalysisDock", "DisplayDock", "LogDock", "SelectionDock", "TrackDock"]

# The initial value of every control that mirrors an analysis parameter is read
# off the frozen dataclass rather than written out again here. Three front ends
# -- this dock, ``rcta-cli`` and a direct core call -- must agree on what the
# default answer is, and a literal repeated in each is a default that drifts
# silently the first time one of them is tuned.
_QUALITY_DEFAULTS = QualityParams()
_BACKGROUND_DEFAULTS = BackgroundParams()
_SCAN_DEFAULTS = ScanParams()


class TrackDock(QtWidgets.QWidget):
    """What was loaded, and what the file said about itself."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(theme.mono_font())
        self.text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.text)

    def set_text(self, text: str) -> None:
        self.text.setPlainText(text)


class AnalysisDock(QtWidgets.QWidget):
    """Every parameter that can change the answer."""

    changed = QtCore.Signal()
    runRequested = QtCore.Signal()
    gridRequested = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        form = QtWidgets.QFormLayout(self)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.background = QtWidgets.QComboBox()
        self.background.addItems(["rolling", "flank", "reference", "global"])
        self.background.setToolTip(
            "The estimator defines what counts as an anomaly.\n"
            "rolling: follows environmental change (default)\n"
            "flank: side-bands with a guard band, for stop-and-measure\n"
            "reference: a span you mark as clean\n"
            "global: one value; only valid without a gradient")
        form.addRow("Background", self.background)

        self.window_s = QtWidgets.QDoubleSpinBox()
        self.window_s.setRange(10.0, 100_000.0)
        self.window_s.setValue(_BACKGROUND_DEFAULTS.window_s)
        self.window_s.setSuffix(" s")
        self.window_s.setSingleStep(60.0)
        self.window_s.setToolTip(
            "Span the background is estimated over.\n"
            "Comfortably wider than the widest window scanned, or the\n"
            "scan is partly comparing a source against itself.")
        form.addRow("Window", self.window_s)
        # What the analysis actually applied, when that differs: the default
        # is widened on a slow cadence, and writing the widened value back
        # into the box would turn an automatic choice into a fixed one.
        self.window_applied = QtWidgets.QLabel("")
        theme.polish(self.window_applied, "muted")
        self.window_applied.setWordWrap(True)
        self.window_applied.hide()
        form.addRow("", self.window_applied)

        self.trim = QtWidgets.QDoubleSpinBox()
        self.trim.setRange(0.0, 0.49)
        self.trim.setValue(_BACKGROUND_DEFAULTS.trim_fraction)
        self.trim.setSingleStep(0.05)
        self.trim.setToolTip(
            "Fraction trimmed from each tail before averaging.\n"
            "Symmetric, so the estimate stays unbiased; a low quantile\n"
            "would sit below the true mean and manufacture detections.")
        form.addRow("Trim", self.trim)

        self.integration = QtWidgets.QDoubleSpinBox()
        self.integration.setRange(0.0, 3600.0)
        self.integration.setValue(0.0)
        self.integration.setSpecialValueText("infer")
        self.integration.setSuffix(" s")
        self.integration.setToolTip(
            "Detector live time behind each reported rate.\n"
            "The file does not record it. Every detection limit scales\n"
            "with this number, so it is the largest systematic here.")
        form.addRow("Integration", self.integration)

        self.alpha = QtWidgets.QDoubleSpinBox()
        self.alpha.setRange(0.0001, 0.5)
        self.alpha.setDecimals(4)
        self.alpha.setValue(_SCAN_DEFAULTS.alpha)
        self.alpha.setToolTip(
            "False positive rate for the whole scan, not per window,\n"
            "when the correction below is on. With it off this is the\n"
            "rate per window and thousands of tests are run.")
        form.addRow("Alpha", self.alpha)

        self.bonferroni = QtWidgets.QCheckBox("Correct for multiple windows")
        self.bonferroni.setChecked(True)
        self.bonferroni.setToolTip(
            "A scan is tens of thousands of tests. Without this a clean\n"
            "track returns on the order of a thousand false detections.")
        form.addRow("", self.bonferroni)

        self.max_accuracy = QtWidgets.QDoubleSpinBox()
        self.max_accuracy.setRange(1.0, 10_000.0)
        self.max_accuracy.setValue(_QUALITY_DEFAULTS.max_accuracy_m)
        self.max_accuracy.setSuffix(" m")
        self.max_accuracy.setToolTip(
            "Fixes worse than this are excluded from anything positional.\n"
            "The default is set for a vehicle or a flight and is generous\n"
            "for a walk: on a 3 km route 100 m is 3 % of the whole survey,\n"
            "so try 20-30 m if your fixes are good.")
        form.addRow("Max accuracy", self.max_accuracy)

        self.auto_run = QtWidgets.QCheckBox("Re-analyse on change")
        self.auto_run.setChecked(True)
        form.addRow("", self.auto_run)

        self.run_button = QtWidgets.QPushButton("Analyse now")
        theme.polish(self.run_button, "primary")
        self.run_button.clicked.connect(self.runRequested)
        form.addRow("", self.run_button)

        self.grid_button = QtWidgets.QPushButton("Compute heatmap")
        self.grid_button.setToolTip(
            "The only slow step. Off by default so loading stays instant.")
        self.grid_button.clicked.connect(self.gridRequested)
        form.addRow("", self.grid_button)

        for widget in (self.window_s, self.trim, self.integration, self.alpha,
                       self.max_accuracy):
            widget.valueChanged.connect(self._maybe_emit)
        self.background.currentTextChanged.connect(self._maybe_emit)
        self.bonferroni.toggled.connect(self._maybe_emit)

    def _maybe_emit(self, *_) -> None:
        if self.auto_run.isChecked():
            self.changed.emit()

    # ---- parameter assembly ----------------------------------------------

    def quality_params(self) -> QualityParams:
        return QualityParams(max_accuracy_m=self.max_accuracy.value())

    def background_params(self, reference_span=None) -> BackgroundParams:
        method = self.background.currentText()
        if method == "reference" and reference_span is None:
            method = "rolling"
        return BackgroundParams(
            method=method, window_s=self.window_s.value(),
            trim_fraction=self.trim.value(),
            reference_span=reference_span if method == "reference" else None)

    def scan_params(self) -> ScanParams:
        return ScanParams(alpha=self.alpha.value(),
                          bonferroni=self.bonferroni.isChecked())

    def set_applied_window(self, applied_s: float) -> None:
        """Show the window the analysis used, if not the one in the box."""
        differs = abs(applied_s - self.window_s.value()) > 0.5
        # Short enough for the dock's width; the reason is the tooltip.
        self.window_applied.setText(
            f"applied: {applied_s:.0f} s (widened)" if differs else "")
        self.window_applied.setToolTip(
            "The default window was widened to fit the widest scan window at "
            "this logging cadence. Set a window explicitly to override; the "
            "log has the details." if differs else "")
        self.window_applied.setVisible(differs)

    def integration_time(self):
        value = self.integration.value()
        return None if value <= 0 else value


class DisplayDock(QtWidgets.QWidget):
    """Purely visual settings. None of these change a number."""

    colormapChanged = QtCore.Signal(str)
    colorScaleChanged = QtCore.Signal(str)
    colorFieldChanged = QtCore.Signal(str)
    unitChanged = QtCore.Signal(str)
    accuracyThresholdChanged = QtCore.Signal(float)
    gapConnectorsToggled = QtCore.Signal(bool)
    heatmapToggled = QtCore.Signal(bool)
    heatmapOpacityChanged = QtCore.Signal(float)
    #: provider id, or None for no basemap
    basemapChanged = QtCore.Signal(object)
    offlineToggled = QtCore.Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        form = QtWidgets.QFormLayout(self)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.field = QtWidgets.QComboBox()
        self.field.addItems(["dose_rate", "count_rate"])
        self.field.currentTextChanged.connect(self.colorFieldChanged)
        form.addRow("Colour by", self.field)

        self.colormap = QtWidgets.QComboBox()
        self.colormap.addItems(theme.DOSE_COLORMAPS)
        self.colormap.setToolTip(
            "All perceptually uniform and colour-blind safe.\n"
            "jet and turbo are deliberately absent: their banding\n"
            "invents structure that is not in the data.")
        self.colormap.currentTextChanged.connect(self.colormapChanged)
        form.addRow("Colourmap", self.colormap)

        self.scale = QtWidgets.QComboBox()
        self.scale.addItems(["auto", "linear", "log"])
        self.scale.currentTextChanged.connect(self.colorScaleChanged)
        form.addRow("Colour scale", self.scale)

        self.unit = QtWidgets.QComboBox()
        self.unit.addItems(sorted(DOSE_UNITS))
        self.unit.setCurrentText("nSv/h")
        self.unit.currentTextChanged.connect(self.unitChanged)
        form.addRow("Display unit", self.unit)

        self.accuracy = QtWidgets.QDoubleSpinBox()
        self.accuracy.setRange(1.0, 10_000.0)
        self.accuracy.setValue(theme.ACCURACY_THRESHOLD_M)
        self.accuracy.setSuffix(" m")
        self.accuracy.setToolTip(
            "Fixes worse than this are drawn as hollow rings:\n"
            "the reading is real, the position is uncertain.")
        self.accuracy.valueChanged.connect(self.accuracyThresholdChanged)
        form.addRow("Ring above", self.accuracy)

        self.basemap = QtWidgets.QComboBox()
        self.basemap.setToolTip(
            "Tiles are fetched only for the view you are looking at.\n"
            "Attribution is drawn on the map and cannot be switched off.")
        self.basemap.currentIndexChanged.connect(self._emit_basemap)
        form.addRow("Basemap", self.basemap)

        self.offline = QtWidgets.QCheckBox("Work offline (cache only)")
        self.offline.setToolTip(
            "Serve tiles already on disk and make no network requests.\n"
            "The map stays fully usable without a basemap.")
        self.offline.toggled.connect(self.offlineToggled)
        form.addRow("", self.offline)

        self.gap_connectors = QtWidgets.QCheckBox("Show gap connectors")
        self.gap_connectors.setToolTip(
            "Draws a dotted line across dropouts, so a gap is visible\n"
            "without asserting a route that was never measured.")
        self.gap_connectors.toggled.connect(self.gapConnectorsToggled)
        form.addRow("", self.gap_connectors)

        self.heatmap = QtWidgets.QCheckBox("Show heatmap")
        self.heatmap.setChecked(True)
        self.heatmap.setToolTip(
            "The interpolated surface, on the points' own colour scale.\n"
            "Blank cells were not constrained by the track; they are\n"
            "unknown, not low. Drawn only when colouring by dose rate.")
        self.heatmap.toggled.connect(self.heatmapToggled)
        form.addRow("", self.heatmap)

        self.heatmap_opacity = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.heatmap_opacity.setRange(10, 100)
        self.heatmap_opacity.setValue(round(theme.HEATMAP_OPACITY * 100))
        self.heatmap_opacity.setToolTip("Heatmap opacity")
        self.heatmap_opacity.valueChanged.connect(
            lambda v: self.heatmapOpacityChanged.emit(v / 100.0))
        form.addRow("Opacity", self.heatmap_opacity)
        self.set_heatmap_available(False)


    def set_heatmap_available(self, available: bool) -> None:
        """Grey the heatmap controls until there is a surface to show."""
        for widget in (self.heatmap, self.heatmap_opacity):
            widget.setEnabled(bool(available))
        self.heatmap.setText("Show heatmap" if available
                             else "Show heatmap (compute it first)")

    # ---- basemap -----------------------------------------------------

    def set_providers(self, providers) -> None:
        """Populate the picker. MainWindow owns the provider table."""
        blocked = self.basemap.blockSignals(True)
        try:
            self.basemap.clear()
            self.basemap.addItem("None", None)
            for provider in providers.values():
                self.basemap.addItem(provider.name, provider.id)
                index = self.basemap.count() - 1
                tip = provider.policy_note
                if not provider.available:
                    # Greyed rather than hidden, so the reason is discoverable.
                    self.basemap.model().item(index).setEnabled(False)
                    tip = f"Needs an API key. {tip}"
                elif not provider.allow_export:
                    tip = f"{tip} Excluded from image exports."
                self.basemap.setItemData(index, tip, Qt.ItemDataRole.ToolTipRole)
        finally:
            self.basemap.blockSignals(blocked)

    def set_basemap(self, provider_id) -> None:
        """Programmatic sync from the View menu. Never emits."""
        index = self.basemap.findData(provider_id)
        if index < 0:
            return
        blocked = self.basemap.blockSignals(True)
        try:
            self.basemap.setCurrentIndex(index)
        finally:
            self.basemap.blockSignals(blocked)

    def set_offline(self, offline: bool) -> None:
        blocked = self.offline.blockSignals(True)
        try:
            self.offline.setChecked(bool(offline))
        finally:
            self.offline.blockSignals(blocked)

    def _emit_basemap(self, index: int) -> None:
        self.basemapChanged.emit(self.basemap.itemData(index))


class SelectionDock(QtWidgets.QWidget):
    """Statistics for the current selection, or the whole track."""

    zoomRequested = QtCore.Signal()
    clearRequested = QtCore.Signal()
    markReferenceRequested = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)

        self.caption = QtWidgets.QLabel("no selection")
        layout.addWidget(self.caption)

        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(theme.mono_font())
        self.text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.text)

        buttons = QtWidgets.QHBoxLayout()
        self.zoom_button = QtWidgets.QPushButton("Zoom to")
        self.zoom_button.clicked.connect(self.zoomRequested)
        buttons.addWidget(self.zoom_button)

        self.reference_button = QtWidgets.QPushButton("Use as background")
        self.reference_button.clicked.connect(self.markReferenceRequested)
        buttons.addWidget(self.reference_button)

        self.clear_button = QtWidgets.QPushButton("Clear")
        self.clear_button.clicked.connect(self.clearRequested)
        buttons.addWidget(self.clear_button)
        layout.addLayout(buttons)

        self.set_selection_available(False)

    def set_summary(self, caption: str, text: str) -> None:
        self.caption.setText(caption)
        self.text.setPlainText(text)

    def set_selection_available(self, available: bool, *,
                                reference_reason: str | None = None) -> None:
        """Grey the three buttons that need a selection to mean anything.

        "Zoom to" survives, because with nothing selected it is still a useful
        gesture -- it frames the whole track. The other two have no reading at
        all without a span, and used to answer a click with a line in the log
        dock, which is not where a user who just pressed a button is looking.
        """
        available = bool(available)
        if reference_reason is None:
            reference_reason = "" if available else "Select a clean span first."
        self.clear_button.setEnabled(available)
        self.reference_button.setEnabled(not reference_reason)
        self.reference_button.setToolTip(
            reference_reason or
            "Mark this span as clean and estimate the background from it.")
        self.clear_button.setToolTip("" if available else "Nothing is selected.")


class LogDock(QtWidgets.QPlainTextEdit):
    """Parser and analysis messages, in the order they happened."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(theme.mono_font())
        self.setMaximumBlockCount(5000)

    def log(self, message: str, level: str = "info") -> None:
        stamp = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
        colour = theme.log_color(level)
        prefix = {"warning": "! ", "error": "x "}.get(level, "  ")
        for line in str(message).rstrip().splitlines() or [""]:
            self.appendHtml(f'<span style="color:{colour}">'
                            f'{stamp}{prefix}{_escape(line)}</span>')


def _escape(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace(" ", "&nbsp;"))
