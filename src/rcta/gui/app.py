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

    app = QtWidgets.QApplication(qt_argv)
    app.setApplicationName("Radiacode Track Analyzer")
    app.setOrganizationName("rcta")
    app.setWindowIcon(theme.app_icon())
    theme.apply(app, str(QtCore.QSettings().value("ui/theme", "System")))
    theme.follow_system(app)

    window = MainWindow(offline=options.offline, settings=QtCore.QSettings())
    window.show()
    if options.track:
        window.load(options.track)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
