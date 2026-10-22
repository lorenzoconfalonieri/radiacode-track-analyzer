"""GUI entry point."""

from __future__ import annotations

import argparse
import sys

#: What to tell someone who installed the package without its GUI extra.
INSTALL_HINT = (
    "rcta: the desktop application needs PySide6 and pyqtgraph, which are not "
    "installed ({missing}).\n"
    "Install them with:  pip install 'rcta[gui]'\n"
    "The command-line tool works without them:  rcta-cli --help")


def parse_args(argv):
    """``(options, qt_argv)`` from a full ``sys.argv``-style list.

    Arguments this parser does not know are left for Qt, returned with the
    program name in front. A Qt option that takes a value must be written
    joined (``-style=fusion``): split, the value would be taken for a track.
    Parsed before Qt is imported, so ``--help`` and ``--version`` work
    without the GUI extra.
    """
    from rcta import __version__

    parser = argparse.ArgumentParser(
        prog="rcta", description="Radiacode Track Analyzer, desktop application.")
    parser.add_argument("track", nargs="?",
                        help="a .rctrk file (or exported track) to open")
    parser.add_argument("--offline", action="store_true",
                        help="never fetch basemap tiles")
    parser.add_argument("--version", action="version",
                        version=f"rcta {__version__}")
    # For release builds: proves a frozen bundle can load, analyse and export.
    parser.add_argument("--smoke-test", metavar="DIR", help=argparse.SUPPRESS)
    options, rest = parser.parse_known_args(argv[1:])
    return options, [argv[0] if argv else "rcta", *rest]


def main(argv=None) -> int:
    options, qt_argv = parse_args(list(sys.argv if argv is None else argv))
    # Imported here, not at module level: the `rcta` script imports this
    # module before calling main(), so a top-level import of a missing Qt
    # would end in a traceback instead of this explanation.
    try:
        import pyqtgraph  # noqa: F401
        from PySide6 import QtCore, QtWidgets
    except ImportError as exc:
        print(INSTALL_HINT.format(missing=exc.name or exc), file=sys.stderr)
        return 1

    from rcta.gui import theme
    from rcta.gui.main_window import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(qt_argv)
    app.setApplicationName("Radiacode Track Analyzer")
    app.setOrganizationName("rcta")
    app.setWindowIcon(theme.app_icon())
    theme.apply(app, str(QtCore.QSettings().value("ui/theme", "System")))
    theme.follow_system(app)

    if options.smoke_test:
        return _smoke_test(options.track, options.smoke_test)

    window = MainWindow(offline=options.offline, settings=QtCore.QSettings())
    window.show()
    if options.track:
        window.load(options.track)
    return app.exec()


def _smoke_test(track, out_dir) -> int:
    """Load ``track``, analyse it, export a PNG and an SVG; 0 if all worked.

    Run by the release workflow against each built bundle, where a module or
    data file PyInstaller failed to collect shows up only at runtime.
    """
    from pathlib import Path

    from rcta.gui.main_window import MainWindow
    from rcta.gui.map_export import render_image, render_svg

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    window = MainWindow(offline=True, synchronous=True)
    window.resize(1200, 800)
    if not track or not window.load(track) or window._analysis is None:
        print("smoke test: the track did not load and analyse", file=sys.stderr)
        return 1
    if not render_image(window.map, 800, 600).save(str(out / "map.png")):
        print("smoke test: PNG export failed", file=sys.stderr)
        return 1
    render_svg(window.map, out / "map.svg")
    print(f"smoke test: ok, {window._track.n} samples, "
          f"{len(window._analysis.hotspots)} hotspot(s), exports in {out}")
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
