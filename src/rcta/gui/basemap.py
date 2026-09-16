"""Painting slippy-map tiles beneath the data.

A single custom ``GraphicsObject`` that paints every visible tile in its
``paint``, rather than one ``ImageItem`` per tile or a composited numpy mosaic.

*Not an item per tile.* The tile set changes on every pan, so that approach
adds and removes forty scene items per gesture, dirtying Qt's scene index for
no benefit; and adjacent tiles' float rectangles, mapped independently to
device pixels, produce hairline seams at some zoom and pan combinations.

*Not a numpy mosaic.* One ``ImageItem`` avoids the seams but re-blits and
re-uploads a multi-megabyte array on *every single tile arrival*, and the
lower-zoom placeholder has to be resampled in numpy by hand.

Painting directly costs about a hundred lines and buys two things that matter:
each tile is blitted to whole device pixels, which removes the seams and hits
Qt's fast 1:1 path whenever the view scale matches the tile scale; and the
parent-zoom fallback is one extra ``drawPixmap`` argument, so a freshly opened
deep view shows a blurry but correctly georeferenced map within one frame
instead of white.

The basemap is made recessive by desaturating and lifting toward white at
decode time, not by lowering opacity. Opacity preserves hue, so OpenStreetMap's
green landuse and blue water would still read as *data* colours competing with
the viridis ramp. Desaturating frees the entire colour axis for dose rate,
while the contrast lift keeps roads and coastlines legible as grey structure.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui

from rcta.core.geo import TILE_PX, WORLD_M, tile_bounds_3857
from rcta.gui import theme

__all__ = ["BasemapItem", "recess"]

#: How many zoom levels to climb looking for a usable placeholder.
MAX_PARENT_LEVELS = 5


def recess(image: QtGui.QImage) -> QtGui.QPixmap:
    """Desaturate and lift toward white so the basemap cannot compete."""
    img = image.convertToFormat(QtGui.QImage.Format.Format_RGB32).copy()
    array = pg.imageToArray(img, copy=False, transpose=False)
    rgb = array[..., :3].astype(np.float32)
    # BGRA memory order, hence the reversed luminance weights.
    grey = rgb @ np.array([0.0722, 0.7152, 0.2126], dtype=np.float32)
    out = rgb + (grey[..., None] - rgb) * theme.BASEMAP_DESATURATE
    out = 255.0 - (255.0 - out) * (1.0 - theme.BASEMAP_LIGHTEN)
    array[..., :3] = np.clip(out, 0, 255).astype(np.uint8)
    return QtGui.QPixmap.fromImage(img)


class BasemapItem(pg.GraphicsObject):
    """Tiles painted beneath the track, in EPSG:3857 metres."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setZValue(-100)
        self._zoom = None
        self._visible_keys: list = []
        self._pixmaps: OrderedDict = OrderedDict()
        self._lighten = theme.BASEMAP_LIGHTEN

    # ---- required by GraphicsObject ---------------------------------------

    def boundingRect(self) -> QtCore.QRectF:
        # The whole Mercator world: always correct, never needs invalidating.
        return QtCore.QRectF(-WORLD_M, -WORLD_M, 2 * WORLD_M, 2 * WORLD_M)

    def paint(self, painter, option, widget=None) -> None:
        if self._zoom is None or not self._visible_keys:
            return
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform,
                              True)
        to_device = painter.transform()
        # Blit in device pixels, not data units: independently rounded float
        # rects are what produce hairline seams between adjacent tiles.
        painter.resetTransform()

        for key in self._visible_keys:
            pixmap, source = self._pixmap_for(key)
            if pixmap is None:
                continue
            x0, y0, x1, y1 = tile_bounds_3857(key.z, key.x, key.y)
            top_left = to_device.map(QtCore.QPointF(x0, y1))
            bottom_right = to_device.map(QtCore.QPointF(x1, y0))
            target = QtCore.QRectF(
                QtCore.QPointF(round(top_left.x()), round(top_left.y())),
                QtCore.QPointF(round(bottom_right.x()), round(bottom_right.y())))
            painter.drawPixmap(target, pixmap, source)

        painter.restore()

    # ---- content ----------------------------------------------------------

    def set_visible(self, zoom: int, keys) -> None:
        self._zoom = int(zoom)
        self._visible_keys = list(keys)
        self.update()

    def set_tile(self, key, data: bytes) -> bool:
        """Decode and store one tile. Returns False if the bytes are not an image.

        A captive portal answers 200 with HTML; treating that as a tile would
        cache a login page as terrain.
        """
        image = QtGui.QImage()
        if not image.loadFromData(data):
            return False
        self._pixmaps[key] = recess(image)
        self._pixmaps.move_to_end(key)
        while len(self._pixmaps) > theme.BASEMAP_MEMORY_TILES:
            self._pixmaps.popitem(last=False)
        self.update()
        return True

    def set_lighten(self, value: float) -> None:
        """Change the recessive treatment. Invalidates the decoded cache."""
        self._lighten = float(value)
        theme.BASEMAP_LIGHTEN = float(value)
        self._pixmaps.clear()
        self.update()

    def clear(self) -> None:
        self._pixmaps.clear()
        self._visible_keys = []
        self._zoom = None
        self.update()

    def has(self, key) -> bool:
        return key in self._pixmaps

    # ---- internals --------------------------------------------------------

    def _pixmap_for(self, key):
        """The tile, or the matching quadrant of the nearest cached ancestor."""
        pixmap = self._pixmaps.get(key)
        if pixmap is not None:
            return pixmap, QtCore.QRectF(0, 0, TILE_PX, TILE_PX)

        for levels in range(1, MAX_PARENT_LEVELS + 1):
            if key.z - levels < 0:
                break
            parent = key.parent(levels)
            pixmap = self._pixmaps.get(parent)
            if pixmap is None:
                continue
            span = 1 << levels
            size = TILE_PX / span
            return pixmap, QtCore.QRectF((key.x % span) * size,
                                         (key.y % span) * size, size, size)
        return None, None
