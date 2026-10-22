"""Render docs/social-preview.png, the card GitHub shows when the repo is shared.

Not part of the package: run by hand, then upload the PNG under the
repository's Settings -> General -> Social preview (GitHub has no API for it).
1280x640 is GitHub's recommended size; everything stays inside a 40 px margin
because some sites crop the edges.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtGui, QtSvg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SVG = ROOT / "src" / "rcta" / "gui" / "icons" / "rcta.svg"
OUT = ROOT / "docs" / "social-preview.png"
W, H = 1280, 640
# The icon's own gradient, darkened so white text reads on it.
TOP, BOTTOM = QtGui.QColor("#1e3a8a"), QtGui.QColor("#0b1a40")
VIRIDIS = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]


def font(px: int, weight=QtGui.QFont.Weight.Normal) -> QtGui.QFont:
    f = QtGui.QFont()
    f.setFamilies(["Helvetica Neue", "Segoe UI", "Inter", "Arial"])
    f.setPixelSize(px)
    f.setWeight(weight)
    return f


def main() -> int:
    _app = QtGui.QGuiApplication(sys.argv)  # noqa: F841 (fonts need one)
    image = QtGui.QImage(W, H, QtGui.QImage.Format.Format_RGB32)
    p = QtGui.QPainter(image)
    p.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing
                     | QtGui.QPainter.RenderHint.TextAntialiasing)

    sky = QtGui.QLinearGradient(0, 0, W, H)
    sky.setColorAt(0, TOP)
    sky.setColorAt(1, BOTTOM)
    p.fillRect(0, 0, W, H, sky)

    # A faint map grid, as on the icon.
    p.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 18), 2))
    for x in range(0, W, 160):
        p.drawLine(x, 0, x, H)
    for y in range(0, H, 160):
        p.drawLine(0, y, W, y)

    # A survey track wandering along the foot of the card, coloured through
    # viridis. Its darkest purple is skipped: it vanishes on the navy.
    import math
    stops = [QtGui.QColor(c) for c in VIRIDIS[1:]]

    def colour_at(t: float) -> QtGui.QColor:
        x = t * (len(stops) - 1)
        i = min(int(x), len(stops) - 2)
        a, b, f = stops[i], stops[i + 1], x - i
        return QtGui.QColor(int(a.red() + (b.red() - a.red()) * f),
                            int(a.green() + (b.green() - a.green()) * f),
                            int(a.blue() + (b.blue() - a.blue()) * f))

    points = [QtCore.QPointF(x, 548 + 22 * math.sin(x / 130.0)
                             + 9 * math.sin(x / 41.0))
              for x in range(60, W - 59, 8)]
    for i in range(len(points) - 1):
        p.setPen(QtGui.QPen(colour_at(i / (len(points) - 2)), 5,
                            c=QtCore.Qt.PenCapStyle.RoundCap))
        p.drawLine(points[i], points[i + 1])

    QtSvg.QSvgRenderer(str(SVG)).render(p, QtCore.QRectF(90, 120, 340, 340))

    white = QtGui.QColor("#ffffff")
    soft = QtGui.QColor(255, 255, 255, 200)
    left, width = 480, W - 480 - 60
    p.setPen(white)
    p.setFont(font(64, QtGui.QFont.Weight.Bold))
    p.drawText(QtCore.QRectF(left, 150, width, 160),
               QtCore.Qt.TextFlag.TextWordWrap, "Radiacode Track Analyzer")
    p.setPen(soft)
    p.setFont(font(30))
    p.drawText(QtCore.QRectF(left, 320, width, 90),
               QtCore.Qt.TextFlag.TextWordWrap,
               "Open-source analysis of GPS-tagged radiation tracks "
               "from RadiaCode dosimeters")
    p.setPen(QtGui.QColor("#fde725"))
    p.setFont(font(24, QtGui.QFont.Weight.DemiBold))
    p.drawText(QtCore.QRectF(left, 420, width, 40), 0,
               "Detection limits, not just detections")
    p.end()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not image.save(str(OUT)):
        print(f"could not write {OUT}", file=sys.stderr)
        return 1
    print(f"wrote {OUT} ({W}x{H}, {OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
