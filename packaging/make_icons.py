"""Render the app icon (src/rcta/gui/icons/rcta.svg) to .icns and .ico.

Run before PyInstaller; the spec reads build/icons/. Qt renders the SVG, the
same renderer the running app uses, and Pillow writes both containers.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image  # noqa: E402
from PySide6 import QtCore, QtGui, QtSvg  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SVG = ROOT / "src" / "rcta" / "gui" / "icons" / "rcta.svg"
OUT = ROOT / "build" / "icons"


def render(size: int) -> Image.Image:
    image = QtGui.QImage(size, size, QtGui.QImage.Format.Format_RGBA8888)
    image.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    QtSvg.QSvgRenderer(str(SVG)).render(painter)
    painter.end()
    data = bytes(image.constBits())
    return Image.frombytes("RGBA", (size, size), data)


def main() -> int:
    _app = QtGui.QGuiApplication(sys.argv)  # noqa: F841 (Qt needs one)
    OUT.mkdir(parents=True, exist_ok=True)
    big = render(1024)
    big.save(OUT / "rcta.png")
    big.save(OUT / "rcta.icns")
    big.save(OUT / "rcta.ico",
             sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
                    (128, 128), (256, 256)])
    print(f"wrote {OUT}/rcta.png, rcta.icns, rcta.ico")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
