"""Plot and table styling: a light, print-friendly scientific look.

One palette for every plot and table in the program, so a figure exported
from one panel and a figure exported from another cannot disagree about what
a colour means. Extended with what a map needs.
"""

from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtGui

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

LOG_COLORS = {
    "info": "#404040",
    "warning": "#B36B00",
    "error": "#B00020",
}


def apply(app) -> None:
    """Configure global pyqtgraph and font defaults."""
    pg.setConfigOptions(
        antialias=True, background="w", foreground="k", imageAxisOrder="row-major",
    )
    app.setProperty("rcta.monoFont", mono_font())


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
