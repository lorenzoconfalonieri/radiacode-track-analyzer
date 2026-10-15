"""GUI entry point."""

from __future__ import annotations

import sys

#: What to tell someone who installed the package without its GUI extra.
INSTALL_HINT = (
    "rcta: the desktop application needs PySide6 and pyqtgraph, which are not "
    "installed ({missing}).\n"
    "Install them with:  pip install 'rcta[gui]'\n"
    "The command-line tool works without them:  rcta-cli --help")


def main(argv=None) -> int:
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

    argv = list(sys.argv if argv is None else argv)
    app = QtWidgets.QApplication(argv)
    app.setApplicationName("Radiacode Track Analyzer")
    app.setOrganizationName("rcta")
    app.setWindowIcon(theme.app_icon())
    theme.apply(app, str(QtCore.QSettings().value("ui/theme", "System")))
    theme.follow_system(app)

    window = MainWindow(offline="--offline" in argv,
                        settings=QtCore.QSettings())
    window.show()

    for arg in argv[1:]:
        if not arg.startswith("-"):
            window.load(arg)
            break
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
