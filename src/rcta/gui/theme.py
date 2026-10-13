"""Light and dark themes: Qt palette, application stylesheet, plot colours.

The widget look -- Fusion, a soft window background, flat rounded cards,
underlined tabs, one accent colour -- is shared with the author's other tools
(DepthVision, cztcal), so they read as one family.

Data colours do not depend on the theme where they carry meaning: viridis is
viridis and a strong detection is green in both. The exceptions are a few
line colours that would vanish against a dark plot (the hover cursor, the
track line, the fitted curve), which :class:`Theme` carries per theme. Map
exports are always rendered with the light theme, because a figure is pasted
onto a white page whatever the screen looked like.

Widgets read colours through :func:`current` rather than hard-coding them, and
listen to :data:`notifier` so the theme can be switched while the program runs
(View > Theme).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pyqtgraph as pg
from PySide6 import QtCore, QtGui

Qt = QtCore.Qt

#: Okabe-Ito, colour-blind safe and distinguishable in greyscale.
SERIES_COLORS = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000",
]

DOSE_COLOR = "#0072B2"
COUNT_COLOR = "#D55E00"
BACKGROUND_COLOR = "#888888"
THRESHOLD_COLOR = "#B00020"
SELECTION_COLOR = "#D55E00"
HOVER_COLOR = "#000000"

# ---- track overlay --------------------------------------------------------

#: Perceptually uniform, monotonic in lightness, colour-blind safe and legible
#: in greyscale print. jet and turbo are deliberately absent: their yellow-cyan
#: banding invents apparent structure, which on a radiation map is not a
#: cosmetic problem.
DOSE_COLORMAPS = ("viridis", "cividis", "magma", "inferno")
DOSE_COLORMAP_DEFAULT = "viridis"

#: Quantisation levels for per-point colour. Beyond about twenty the eye
#: cannot discriminate steps on a continuous ramp, and sharing 32 brushes keeps
#: pyqtgraph's symbol atlas at 32 entries rather than one per point -- the
#: difference between 0.19 s and 2.1 s to build a 100 000-point scatter.
DOSE_COLOR_LEVELS = 32

POINT_SIZE = 6
TRACK_LINE_COLOR = "#9AA0A6"
TRACK_LINE_WIDTH = 1.0
TRACK_GAP_COLOR = "#C8CCD0"
SELECTION_HALO_WIDTH = 2.0
UNCERTAIN_PEN_WIDTH = 1.2
ACCURACY_THRESHOLD_M = 50.0
DECIMATE_ABOVE = 50_000

#: The surface is an inference drawn under the measured points; opaque
#: enough to read, translucent enough that the basemap still locates it.
HEATMAP_OPACITY = 0.6

# ---- basemap --------------------------------------------------------------

#: Desaturate, then lift toward white. Opacity alone preserves hue, so OSM's
#: green landuse and blue water would still read as data colours next to the
#: viridis ramp. Desaturating frees the whole colour axis for dose rate; the
#: lift keeps roads and coastlines legible as grey structure.
BASEMAP_DESATURATE = 0.85
BASEMAP_LIGHTEN = 0.55
BASEMAP_MEMORY_TILES = 256
BASEMAP_DISK_BUDGET_BYTES = 512 << 20
BASEMAP_TTL_DAYS = 30                  # the OSM policy floor is 7
BASEMAP_NEGATIVE_TTL_HOURS = 24
BASEMAP_MAX_CONCURRENT = 4
RETILE_DEBOUNCE_MS = 120
ABORT_GRACE_MS = 250
DECODE_PER_TICK = 8

# ---- map chrome -----------------------------------------------------------

GRATICULE_ALPHA = 0.12
ATTRIBUTION_COLOR = "#404040"
ATTRIBUTION_BG = (255, 255, 255, 190)
MIN_EXTENT_M = 50.0

# ---- semantic ------------------------------------------------------------

EVIDENCE_COLORS = {
    "strong": "#009E73",
    "possible": "#E69F00",
    "weak": "#888888",
}

FLAG_COLORS = {
    "trend": "#888888",
    "poor-gps": "#E69F00",
    "gap-adjacent": "#CC79A7",
    "single-sample": "#56B4E9",
}

def log_color(level: str) -> str:
    """Log text colour for the current theme."""
    t = _current
    return {"warning": t.warn, "error": t.bad}.get(level, t.text)


# ---- widget themes --------------------------------------------------------

THEMES = ("System", "Light", "Dark")


@dataclass(frozen=True)
class Theme:
    name: str
    dark: bool
    window: str          # page background
    panel: str           # cards, docks, tab pages
    base: str            # inputs, tables
    alt: str             # alternating table rows
    text: str
    muted: str
    border: str
    accent: str
    accent_text: str
    good: str
    bad: str
    warn: str
    banner_bg: str
    banner_fg: str
    banner_border: str
    plot_bg: str
    plot_fg: str
    plot_axis: str
    hover: str           # cursor line and hover ring
    track_line: str
    track_gap: str
    fit: str             # fitted curves drawn over data
    dose: str
    count: str


LIGHT = Theme(
    name="Light", dark=False,
    window="#eef0f3", panel="#f9fafb", base="#ffffff", alt="#f2f4f7",
    text="#1f2328", muted="#667085", border="#d0d5dd",
    accent="#2563eb", accent_text="#ffffff",
    good="#15803d", bad="#d92d20", warn="#b54708",
    banner_bg="#eaf1ff", banner_fg="#1e3a8a", banner_border="#bcd0fb",
    plot_bg="#ffffff", plot_fg="#344054", plot_axis="#d0d5dd",
    hover=HOVER_COLOR, track_line=TRACK_LINE_COLOR, track_gap=TRACK_GAP_COLOR,
    fit="#444444", dose=DOSE_COLOR, count=COUNT_COLOR,
)

DARK = Theme(
    name="Dark", dark=True,
    window="#16191d", panel="#1e2227", base="#14171b", alt="#262b31",
    text="#e4e7eb", muted="#98a2b3", border="#343a42",
    accent="#4c8dff", accent_text="#ffffff",
    good="#3fb950", bad="#ff6b5f", warn="#e3a23a",
    banner_bg="#1c2a44", banner_fg="#cfe0ff", banner_border="#2f4a7a",
    plot_bg="#111316", plot_fg="#c3cad3", plot_axis="#3a4048",
    hover="#e4e7eb", track_line="#6b7280", track_gap="#4b5563",
    fit="#c3cad3", dose="#56B4E9", count="#FF8A5B",
)


class _Notifier(QtCore.QObject):
    changed = QtCore.Signal(object)


#: Emits the new :class:`Theme` after every switch.
notifier = _Notifier()
_current: Theme = LIGHT
_choice = "Light"

_ICONS = Path(__file__).with_name("icons")


def current() -> Theme:
    return _current


def choice() -> str:
    """What the user picked, which may be "System" rather than a theme."""
    return _choice


def _system_is_dark() -> bool:
    try:
        scheme = QtGui.QGuiApplication.styleHints().colorScheme()
        return scheme == Qt.ColorScheme.Dark
    except AttributeError:          # Qt < 6.5
        window = QtGui.QGuiApplication.palette().color(QtGui.QPalette.ColorRole.Window)
        return window.lightness() < 128


def resolve(name: str) -> Theme:
    if name == "Dark":
        return DARK
    if name == "Light":
        return LIGHT
    return DARK if _system_is_dark() else LIGHT


def palette(t: Theme) -> QtGui.QPalette:
    role = QtGui.QPalette.ColorRole
    p = QtGui.QPalette()
    for r, colour in {
        role.Window: t.window, role.WindowText: t.text, role.Base: t.base,
        role.AlternateBase: t.alt, role.Text: t.text, role.Button: t.panel,
        role.ButtonText: t.text, role.ToolTipBase: t.panel,
        role.ToolTipText: t.text, role.Highlight: t.accent,
        role.HighlightedText: t.accent_text, role.PlaceholderText: t.muted,
        role.Link: t.accent, role.BrightText: t.bad, role.Light: t.alt,
        role.Midlight: t.border, role.Mid: t.border, role.Dark: t.border,
        role.Shadow: "#000000",
    }.items():
        p.setColor(r, QtGui.QColor(colour))
    for r in (role.WindowText, role.Text, role.ButtonText):
        p.setColor(QtGui.QPalette.ColorGroup.Disabled, r, QtGui.QColor(t.muted))
    return p


def stylesheet(t: Theme) -> str:
    check = (_ICONS / "check.svg").as_posix()
    v = "dark" if t.dark else "light"
    up = (_ICONS / f"up_{v}.svg").as_posix()
    down = (_ICONS / f"down_{v}.svg").as_posix()
    box = "#5a626d" if t.dark else "#98a2b3"
    hover = "#2d333b" if t.dark else "#e4e9f2"
    pressed = "#353c45" if t.dark else "#d5dceb"
    accent_hover = "#6aa0ff" if t.dark else "#1d4fd8"
    return f"""
QMainWindow, QDialog {{ background: {t.window}; }}
QMainWindow::separator {{ background: transparent; width: 5px; height: 5px; }}
QMainWindow::separator:hover {{ background: {t.accent}; }}
QToolTip {{ background: {t.panel}; color: {t.text}; border: 1px solid {t.border}; padding: 4px 6px; }}

QTabWidget::pane {{ border: 1px solid {t.border}; border-radius: 6px; background: {t.panel}; top: -1px; }}
QTabBar::tab {{ background: transparent; color: {t.muted}; padding: 6px 14px; margin-right: 2px;
               border: none; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {t.text}; border-bottom: 2px solid {t.accent}; font-weight: 600; }}
QTabBar::tab:hover:!selected {{ color: {t.text}; border-bottom: 2px solid {t.border}; }}

QDockWidget {{ color: {t.text}; font-weight: 600; }}
QDockWidget::title {{ background: {t.window}; padding: 6px 8px; text-align: left; }}

QGroupBox {{ background: {t.panel}; border: 1px solid {t.border}; border-radius: 6px;
            margin-top: 18px; padding: 8px 6px 6px 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; left: 8px; top: 2px;
                   color: {t.text}; font-weight: 600; }}

QPushButton {{ background: {t.base}; color: {t.text}; border: 1px solid {t.border}; border-radius: 5px;
              padding: 4px 10px; min-height: 18px; }}
QPushButton:hover {{ background: {hover}; }}
QPushButton:pressed, QPushButton:checked {{ background: {pressed}; }}
QPushButton:disabled {{ color: {t.muted}; background: {t.alt}; }}
QPushButton[class="primary"] {{ background: {t.accent}; color: {t.accent_text}; border: 1px solid {t.accent};
                                font-weight: 600; padding: 5px 14px; }}
QPushButton[class="primary"]:hover {{ background: {accent_hover}; border-color: {accent_hover}; }}
QPushButton[class="primary"]:disabled {{ background: {t.border}; border-color: {t.border}; color: {t.muted}; }}

QToolButton {{ background: {t.base}; color: {t.text}; border: 1px solid {t.border}; border-radius: 5px;
              padding: 3px 8px; }}
QToolButton:hover {{ background: {hover}; }}
QToolButton:checked {{ background: {t.accent}; color: {t.accent_text}; border-color: {t.accent}; }}

QCheckBox::indicator, QTableView::indicator, QGroupBox::indicator {{
    width: 14px; height: 14px; border: 1px solid {box}; border-radius: 3px; background: {t.base}; }}
QCheckBox::indicator:hover {{ border-color: {t.accent}; }}
QCheckBox::indicator:checked, QTableView::indicator:checked {{
    background: {t.accent}; border-color: {t.accent}; image: url({check}); }}
QCheckBox::indicator:disabled {{ background: {t.alt}; border-color: {t.border}; }}
QAbstractSpinBox, QLineEdit {{ border: 1px solid {box if t.dark else t.border}; border-radius: 4px;
    background: {t.base}; padding: 1px 4px; min-height: 18px; selection-background-color: {t.accent}; }}
QAbstractSpinBox:focus, QLineEdit:focus {{ border-color: {t.accent}; }}
QAbstractSpinBox::up-button {{ subcontrol-origin: border; subcontrol-position: top right; width: 16px;
    border: none; background: transparent; }}
QAbstractSpinBox::down-button {{ subcontrol-origin: border; subcontrol-position: bottom right; width: 16px;
    border: none; background: transparent; }}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{ background: {hover}; }}
QAbstractSpinBox::up-arrow {{ image: url({up}); width: 8px; height: 5px; }}
QAbstractSpinBox::down-arrow {{ image: url({down}); width: 8px; height: 5px; }}
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::down-arrow:disabled {{ image: none; }}

QSplitter::handle {{ background: transparent; }}
QSplitter::handle:hover {{ background: {t.accent}; }}

QHeaderView::section {{ background: {t.alt}; color: {t.muted}; border: none;
                       border-right: 1px solid {t.border}; border-bottom: 1px solid {t.border};
                       padding: 3px 6px; font-weight: 600; }}
QTableView, QPlainTextEdit {{ background: {t.base}; gridline-color: {t.alt};
                       selection-background-color: {t.accent}; selection-color: {t.accent_text}; }}
QDockWidget QTableView, QDockWidget QPlainTextEdit {{ border: 1px solid {t.border}; border-radius: 5px; }}
QTableCornerButton::section {{ background: {t.alt}; border: none; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle {{ background: {t.border}; border-radius: 4px; min-height: 24px; min-width: 24px; margin: 2px; }}
QScrollBar::handle:hover {{ background: {t.muted}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QStatusBar {{ background: {t.window}; color: {t.muted}; border-top: 1px solid {t.border}; }}
QStatusBar::item {{ border: none; }}
QProgressBar {{ background: {t.alt}; border: 1px solid {t.border}; border-radius: 5px; text-align: center;
               color: {t.text}; min-height: 14px; }}
QProgressBar::chunk {{ background: {t.accent}; border-radius: 4px; }}
QMenu {{ background: {t.panel}; color: {t.text}; border: 1px solid {t.border}; padding: 4px; }}
QMenu::item {{ padding: 4px 18px; border-radius: 4px; }}
QMenu::item:selected {{ background: {t.accent}; color: {t.accent_text}; }}
QMenu::item:disabled {{ color: {t.muted}; }}

QLabel[class="banner"] {{ background: {t.banner_bg}; color: {t.banner_fg}; border: 1px solid {t.banner_border};
                         border-radius: 6px; padding: 7px 12px; margin: 6px 8px; }}
QLabel[class="muted"] {{ color: {t.muted}; }}
QLabel[class="title"] {{ font-weight: 600; }}
QLabel[class="chip-good"] {{ background: {t.good}; color: white; border-radius: 9px; padding: 2px 8px;
                            font-weight: 600; }}
QLabel[class="chip-bad"] {{ background: {t.bad}; color: white; border-radius: 9px; padding: 2px 8px;
                           font-weight: 600; }}
QLabel[class="chip-warn"] {{ background: {t.warn}; color: white; border-radius: 9px; padding: 2px 8px;
                            font-weight: 600; }}
QLabel[class="chip-muted"] {{ background: {t.alt}; color: {t.muted}; border: 1px solid {t.border};
                             border-radius: 9px; padding: 2px 8px; }}
"""


def polish(widget, cls: str):
    """Set a stylesheet class (QSS ``[class="..."]``) and re-polish."""
    widget.setProperty("class", cls)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    return widget


# ---- plots -----------------------------------------------------------------

def style_plot_item(item, t: Theme | None = None) -> None:
    """Axes, title and legend colours of one pyqtgraph PlotItem."""
    t = t or _current
    fg = pg.mkColor(t.plot_fg)
    for name in ("left", "bottom", "right", "top"):
        if name not in item.axes:
            continue
        axis = item.getAxis(name)
        axis.setPen(pg.mkPen(t.plot_axis))
        axis.setTextPen(pg.mkPen(fg))
        if axis.label is not None:
            axis.label.setDefaultTextColor(fg)
    if item.titleLabel is not None:
        item.titleLabel.setAttr("color", t.plot_fg)
        item.titleLabel.setText(item.titleLabel.text)
    if item.legend is not None:
        style_legend(item.legend, t)


def style_legend(legend, t: Theme | None = None) -> None:
    t = t or _current
    background = QtGui.QColor(t.panel)
    background.setAlpha(215)
    legend.setBrush(pg.mkBrush(background))
    legend.setPen(pg.mkPen(t.border))
    legend.setLabelTextColor(t.plot_fg)
    for _sample, label in legend.items:
        label.setAttr("color", t.plot_fg)
        label.setText(label.text)


def style_graphics_view(view, t: Theme | None = None) -> None:
    """Background, axes and legends of every plot in one GraphicsView."""
    t = t or _current
    view.setBackground(t.plot_bg)
    for item in view.scene().items():
        if isinstance(item, pg.PlotItem):
            style_plot_item(item, t)
        elif isinstance(item, pg.AxisItem) and not isinstance(
                item.parentItem(), pg.PlotItem):
            item.setTextPen(pg.mkPen(t.plot_fg))     # colour bar axis
            item.setPen(pg.mkPen(t.plot_axis))
            if item.label is not None:
                item.label.setDefaultTextColor(pg.mkColor(t.plot_fg))


@contextmanager
def plots_in(panel, t: Theme):
    """Draw one panel in another theme for the duration, e.g. for export.

    ``panel`` is anything with a ``restyle(theme)`` method; only that panel
    changes, and it is put back whatever happens inside the block.
    """
    if t is _current:
        yield
        return
    panel.restyle(t)
    try:
        yield
    finally:
        panel.restyle(_current)


def follow(panel) -> None:
    """Restyle ``panel`` now and after every theme switch."""
    panel.restyle(_current)
    notifier.changed.connect(panel.restyle)


# ---- apply -----------------------------------------------------------------

def apply(app, name: str = "Light") -> Theme:
    """Install a theme on the application and restyle what already exists."""
    global _current, _choice
    _choice = name if name in THEMES else "Light"
    _current = t = resolve(_choice)
    app.setStyle("Fusion")
    app.setPalette(palette(t))
    app.setStyleSheet(stylesheet(t))
    pg.setConfigOptions(antialias=True, background=t.plot_bg,
                        foreground=t.plot_fg, imageAxisOrder="row-major")
    app.setProperty("rcta.monoFont", mono_font())
    for widget in app.topLevelWidgets():
        for view in widget.findChildren(pg.GraphicsView):
            style_graphics_view(view, t)
    notifier.changed.emit(t)
    return t


def follow_system(app) -> None:
    """Re-apply when the OS switches light/dark, if the choice is "System"."""
    try:
        hints = QtGui.QGuiApplication.styleHints()
        hints.colorSchemeChanged.connect(
            lambda *_: apply(app, _choice) if _choice == "System" else None)
    except AttributeError:
        pass


def mono_font() -> QtGui.QFont:
    """Fixed-pitch font so digits line up in the tables."""
    font = QtGui.QFont("Menlo")
    if not QtGui.QFontInfo(font).fixedPitch():
        font = QtGui.QFont("Monaco")
    font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
    font.setPointSize(11)
    return font


def dose_colormap(name: str = DOSE_COLORMAP_DEFAULT):
    """A pyqtgraph colormap by name, falling back to the default."""
    try:
        return pg.colormap.get(name)
    except (KeyError, ValueError, FileNotFoundError):
        return pg.colormap.get(DOSE_COLORMAP_DEFAULT)
