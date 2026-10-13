"""GUI entry point."""

from __future__ import annotations

import sys

from PySide6 import QtCore, QtWidgets

from rcta.gui import theme
from rcta.gui.main_window import MainWindow


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    app = QtWidgets.QApplication(argv)
    app.setApplicationName("Radiacode Track Analyzer")
    app.setOrganizationName("rcta")
    app.setWindowIcon(theme.app_icon())
    theme.apply(app, str(QtCore.QSettings().value("ui/theme", "System")))
    theme.follow_system(app)

    window = MainWindow(offline="--offline" in argv)
    window.show()

    for arg in argv[1:]:
        if not arg.startswith("-"):
            window.load(arg)
            break
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
