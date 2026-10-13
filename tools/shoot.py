"""Regenerate the documentation screenshots against a live basemap.

Not part of the package: a maintenance script run by hand when a figure needs
refreshing. It drives the real MainWindow offscreen rather than compositing
anything, so what lands in docs/ is exactly what a user sees.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

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


def _synthetic_flight(folder: Path) -> tuple[Path, Path]:
    """tests/synthetic.flight_track() written out as a track and trajectory.

    The figure is "a profile with a known answer": 900 samples built from a
    2000 m scale height. Going through files rather than injecting objects
    means the figure exercises the same parser and join a user's file does.
    """
    sys.path.insert(0, "tests")
    import synthetic

    track, trajectory, _truth = synthetic.flight_track()
    rows = ["Track: synthetic flight",
            "Timestamp\tLatitude\tLongitude\tAccuracy\tDoseRate\tCountRate"]
    for i in range(track.n):
        # The TSV dose column is in the device's display unit, a tenth of
        # nSv/h; see docs/FORMAT.md.
        rows.append(f"{int(track.time_s[i])}\t{track.lat[i]:.6f}\t"
                    f"{track.lon[i]:.6f}\t{track.accuracy_m[i]:.0f}\t"
                    f"{track.dose_rate[i] / 10:.3f}\t{track.count_rate[i]:.3f}")
    track_path = folder / "synthetic-flight.rctrk"
    track_path.write_text("\n".join(rows), encoding="utf-8")
    trajectory_path = folder / "synthetic-flight-trajectory.csv"
    trajectory_path.write_text(
        "time,lat,lon,altitude_m\n" + "\n".join(
            f"{int(t)},{la:.6f},{lo:.6f},{a:.1f}" for t, la, lo, a in zip(
                trajectory.time_s, trajectory.lat, trajectory.lon,
                trajectory.altitude_m)), encoding="utf-8")
    return track_path, trajectory_path


def shoot_flight_fit(path: str) -> int:
    """The Flight dock alone: altitude profile, fit and its caption."""
    with tempfile.TemporaryDirectory() as folder:
        track_path, trajectory_path = _synthetic_flight(Path(folder))
        win = MainWindow(offline=True)
        win.show()
        size_to(win, 1500, 950)
        if not win.load(str(track_path)):
            print("could not load the synthetic flight", file=sys.stderr)
            return 1
        win._load_trajectory(str(trajectory_path), "m")
    # Out of its dock, so the figure has a fixed size whatever the window
    # layout does.
    panel = win.flight
    panel.setParent(None)
    panel.resize(1200, 640)
    panel.show()
    pump(600)
    panel.grab().save(path)
    print(f"  wrote {path}  ({panel.width()}x{panel.height()})")
    win.close()
    return 0


def main() -> int:
    out = os.environ.get("RCTA_SHOT_OUT", "docs/screenshot-walk.png")
    fit_out = os.environ.get("RCTA_SHOT_FIT_OUT",
                             "docs/screenshot-flight-fit.png")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    theme.apply(app, "Light")       # a figure for a white page

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
    return shoot_flight_fit(fit_out)


if __name__ == "__main__":
    raise SystemExit(main())
