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

import re
from contextlib import contextmanager
from pathlib import Path
from xml.sax.saxutils import escape

from PySide6 import QtCore, QtGui

from rcta.core.dose import convert_dose_rate
from rcta.gui import theme

__all__ = ["attribution_lines", "compose_attribution", "render_image",
           "render_svg"]


def attribution_lines(analysis, provider=None, *, colormap: str = "viridis",
                      scale: str = "linear", include_basemap: bool = True,
                      unit: str = "nSv/h", grid=None):
    """The provenance block stamped into an exported image.

    ``grid`` is the heatmap when it is drawn, so the figure states how far
    the surface was allowed to reach from the track it was inferred from.
    """
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

    readings = convert_dose_rate(track.dose_rate[track.valid_mask], unit)
    if readings.size:
        lines.append(f"Dose rate {readings.min():.3g}–{readings.max():.3g} "
                     f"{unit}, {colormap}, {scale} scale · "
                     "Web Mercator (EPSG:3857)")
    if grid is not None:
        reach = grid.params.max_distance_m or 2.0 * grid.bandwidth_m
        lines.append(f"Heatmap: Gaussian kernel, bandwidth {grid.bandwidth_m:.0f} m, "
                     f"blank beyond {reach:.0f} m of the track (unknown, not low)")
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
        metrics = QtGui.QFontMetricsF(font, image)
        # Sized from the height alone, a tall narrow export ran the lines off
        # the right edge and clipped the very credit the licence requires.
        # Shrink until the widest line fits, padding included. A loop, not
        # one proportional step: hinted glyph widths do not scale linearly
        # at small sizes.
        for _ in range(20):
            widest = max(metrics.horizontalAdvance(line) for line in lines)
            room = image.width() - 3.0 * metrics.height() * 0.4
            if widest <= room or font.pointSizeF() <= 3.0:
                break
            font.setPointSizeF(max(font.pointSizeF() * 0.92, 3.0))
            metrics = QtGui.QFontMetricsF(font, image)
        painter.setFont(font)

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


@contextmanager
def _without_basemap(map_panel, hide: bool):
    """Take the basemap out of the scene for the duration of a render.

    Dropping only the credit line, as an earlier version did, still wrote
    the provider's pixels into the file, which is exactly what its terms
    forbid; the tiles themselves have to leave the scene.
    """
    item = getattr(map_panel, "basemap", None)
    was_visible = item is not None and item.isVisible()
    if hide and was_visible:
        item.hide()
    try:
        yield
    finally:
        if hide and was_visible:
            item.show()


def render_image(map_panel, width: int, height: int, *, lines=(),
                 include_basemap: bool = True) -> QtGui.QImage:
    """Render the map scene into an image and stamp the attribution on it.

    ``include_basemap=False`` renders the data layers alone, for a provider
    whose terms do not allow its tiles in an exported image.
    """
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
        with theme.plots_in(map_panel, theme.LIGHT), \
                _without_basemap(map_panel, not include_basemap):
            view.scene().render(painter, target, source,
                                QtCore.Qt.AspectRatioMode.IgnoreAspectRatio)
    finally:
        painter.end()

    compose_attribution(image, lines)
    return image


def render_svg(map_panel, path, *, lines=()) -> None:
    """Vector export of the data layers and colour bar, with provenance.

    The whole scene is exported rather than the plot alone, so the colour bar
    comes along: a coloured track without its scale is a picture, not a
    figure. The provenance lines are appended below it as real SVG text, as
    they are baked into the PNG.
    """
    from pyqtgraph import exporters

    exporter = exporters.SVGExporter(map_panel._glw.scene())
    # The basemap paints raster tiles, which the SVG generator would embed as
    # images; it is removed rather than trusted to be skipped.
    with theme.plots_in(map_panel, theme.LIGHT), \
            _without_basemap(map_panel, True):
        svg = exporter.export(toBytes=True).decode("utf-8")
    Path(path).write_text(_append_lines(svg, list(lines)), encoding="utf-8")


def _append_lines(svg: str, lines) -> str:
    """Grow the SVG's viewBox downward and write ``lines`` into the band."""
    match = re.search(r'viewBox\s*=\s*"([\d.\s-]+)"', svg)
    if not lines or match is None:
        return svg
    x, y, width, height = (float(v) for v in match.group(1).split())
    size, step = 11.0, 14.0
    band = step * len(lines) + 8.0
    svg = (svg[:match.start()]
           + f'viewBox="{x:g} {y:g} {width:g} {height + band:g}"'
           + svg[match.end():])
    text = [f'<g font-family="sans-serif" font-size="{size:g}" '
            f'fill="{theme.ATTRIBUTION_COLOR}">']
    for i, line in enumerate(lines):
        text.append(f'<text x="{x + 6:g}" y="{height + step * (i + 1):g}">'
                    f'{escape(line)}</text>')
    text.append("</g>")
    close = svg.rindex("</svg>")
    return svg[:close] + "\n".join(text) + "\n" + svg[close:]
