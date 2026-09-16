"""Regenerate the documentation screenshots against a live basemap.

Not part of the package: a maintenance script run by hand when a figure needs
refreshing. It drives the real MainWindow offscreen rather than compositing
anything, so what lands in docs/ is exactly what a user sees.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, "src")

from PySide6 import QtCore, QtWidgets                              # noqa: E402

from rcta.gui import theme                                         # noqa: E402
from rcta.gui.main_window import MainWindow                        # noqa: E402

WALK = "sample-data/Track 07 Sep 2026 14-30-57.rctrk"
FLIGHT = "sample-data/Track 07 Sep 2026 07-53-59.rctrk"
PROVIDER = "usgs_imagery"          # public domain; the only kind we may print


def pump(ms: int) -> None:
    """Run the event loop for a fixed wall-clock interval."""
    loop = QtCore.QEventLoop()
    QtCore.QTimer.singleShot(ms, loop.quit)
    loop.exec()


def settle(win, *, quiet_ms: int = 1500, timeout_ms: int = 60000) -> int:
    """Pump until the tile fetcher has been idle for ``quiet_ms``."""
    pending = {"n": -1}
    win.tiles.statusChanged.connect(lambda _m, n: pending.__setitem__("n", n))
    waited = idle = 0
    while waited < timeout_ms:
        pump(250)
        waited += 250
        idle = idle + 250 if pending["n"] in (0, -1) else 0
        if idle >= quiet_ms and waited > 2000:
            break
    return waited


def shoot(win, path: str) -> None:
    pump(400)
    win.grab().save(path)
    print(f"  wrote {path}  ({win.width()}x{win.height()})")


def size_to(win, w: int, h: int) -> None:
    """Resize until grab() actually yields w x h.

    The offscreen platform ignores propagateSizeHints, so the first resize
    lands a few tens of pixels out; correcting by the observed error converges
    in one step and keeps the figure aspect ratio reproducible.
    """
    win.resize(w, h)
    pump(200)
    dw, dh = w - win.width(), h - win.height()
    if dw or dh:
        win.resize(w + dw, h + dh)
        pump(200)


def main() -> int:
    out = os.environ.get("RCTA_SHOT_OUT", "docs/screenshot-walk.png")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    theme.apply(app)

    win = MainWindow(offline=False)
    win.show()
    size_to(win, 1500, 950)

    if not win.load(WALK):
        print("could not load the walking track", file=sys.stderr)
        return 1
    win._set_provider(PROVIDER)
    win.map.zoom_to_track()
    waited = settle(win)
    prov = win._providers[PROVIDER]
    print(f"  tiles settled after {waited/1000:.1f}s on {prov.name} "
          f"(desaturate={prov.desaturate}, lighten={prov.lighten})")
    shoot(win, out)

    win.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
