"""Rendering the map to an image, with attribution baked into the pixels.

Attribution is composited into the bitmap rather than left as a layer, and it
is not optional. An exported map that loses its attribution becomes a licence
violation the moment somebody pastes it into a report, and the person pasting
it will not be the person who knew that.

Two consequences of provider terms are enforced here rather than documented and
hoped for.

*Some providers may not be exported at all.* Esri's World Imagery terms tie the
service to an ArcGIS licence, so :attr:`TileProvider.allow_export` is false for
it and the basemap is dropped from the image with the reason stated.

*Export must not silently fetch.* Rendering at a sharper zoom for a
higher-resolution figure would issue a burst of requests for tiles nobody is
looking at, which is precisely the bulk downloading the OpenStreetMap policy
prohibits. The export therefore uses the tiles already on screen and upscales
them -- slightly soft, and honestly labelled -- rather than reaching for
sharper ones behind the user's back.

SVG export deliberately excludes the basemap. A custom ``paint`` is not
representable as vector primitives, and embedding forty base64 PNGs of someone
else's tiles in an SVG is both ugly and the least defensible thing to do with
them. An SVG carries the track, graticule, scale bar and colour bar, which for
a journal figure is usually what is actually wanted.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui

from rcta.gui import theme

__all__ = ["attribution_lines", "compose_attribution", "render_image",
           "render_svg"]


def attribution_lines(analysis, provider=None, *, colormap: str = "viridis",
                      scale: str = "linear", include_basemap: bool = True):
    """The provenance block stamped into an exported image."""
    lines = []
    if provider is not None and include_basemap:
        lines.append(provider.attribution_export)

    track = analysis.track
    start = QtCore.QDateTime.fromSecsSinceEpoch(
        int(track.time_s[0]), QtCore.QTimeZone.utc()).toString("yyyy-MM-dd HH:mm")
    end = QtCore.QDateTime.fromSecsSinceEpoch(
        int(track.time_s[-1]), QtCore.QTimeZone.utc()).toString("HH:mm")
    device = ", ".join(track.meta.devices) or "unknown device"
    lines.append(f"{track.meta.title or 'track'} · {device} · "
                 f"{track.n} samples · {start}–{end} UTC")

    lo = float(track.dose_rate.min())
    hi = float(track.dose_rate.max())
    lines.append(f"Dose rate {lo:.1f}–{hi:.0f} nSv/h, {colormap}, {scale} scale · "
                 "Web Mercator (EPSG:3857)")
    return lines


def compose_attribution(image: QtGui.QImage, lines) -> None:
    """Bake attribution into the pixels, bottom-left, on a translucent plate."""
    if not lines:
        return
    painter = QtGui.QPainter(image)
    try:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        font = QtGui.QFont()
        font.setPointSizeF(max(image.height() / 90.0, 7.0))
        painter.setFont(font)
        metrics = QtGui.QFontMetricsF(font)

        pad = metrics.height() * 0.4
        height = metrics.height() * len(lines) + 2 * pad
        width = max(metrics.horizontalAdvance(line) for line in lines) + 2 * pad
        rect = QtCore.QRectF(pad, image.height() - height - pad, width, height)

        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(*theme.ATTRIBUTION_BG))
        painter.drawRoundedRect(rect, pad * 0.5, pad * 0.5)

        painter.setPen(QtGui.QColor(theme.ATTRIBUTION_COLOR))
        y = rect.top() + pad + metrics.ascent()
        for line in lines:
            painter.drawText(QtCore.QPointF(rect.left() + pad, y), line)
            y += metrics.height()
    finally:
        painter.end()


def render_image(map_panel, width: int, height: int, *, lines=()) -> QtGui.QImage:
    """Render the map scene into an image and stamp the attribution on it."""
    lines = list(lines)
    # Reserve a band for the attribution rather than overlaying it on the axis
    # labels: the text has to stay legible for the licence to be satisfied.
    band = int(max(height / 90.0, 7.0) * 1.9) * max(len(lines), 1) if lines else 0

    image = QtGui.QImage(int(width), int(height) + band,
                         QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QtGui.QColor("white"))

    view = map_panel._glw
    # The source must be the *view's visible region* mapped into scene
    # coordinates, not scene.sceneRect(). The scene rect is whatever bounding
    # box the items happen to occupy, so using it renders the track as a
    # postage stamp in the corner of an otherwise empty page.
    source = view.mapToScene(view.viewport().rect()).boundingRect()
    if source.isEmpty():
        source = view.scene().sceneRect()

    # A figure goes onto a white page, so it is drawn in the light theme
    # whatever the screen is showing.
    painter = QtGui.QPainter(image)
    try:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        target = QtCore.QRectF(0, 0, image.width(), image.height() - band)
        with theme.plots_in(map_panel, theme.LIGHT):
            view.scene().render(painter, target, source,
                                QtCore.Qt.AspectRatioMode.IgnoreAspectRatio)
    finally:
        painter.end()

    compose_attribution(image, lines)
    return image


def render_svg(map_panel, path) -> None:
    """Vector export of the data layers only. The basemap cannot come along."""
    from pyqtgraph import exporters

    exporter = exporters.SVGExporter(map_panel.plot)
    with theme.plots_in(map_panel, theme.LIGHT):
        exporter.export(str(path))
