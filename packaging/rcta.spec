# PyInstaller spec for the desktop application.
#
#   python packaging/make_icons.py
#   pyinstaller --noconfirm packaging/rcta.spec
#
# Produces dist/rcta/ (Windows, Linux) or dist/Radiacode Track Analyzer.app
# (macOS). Run on the platform you are building for; PyInstaller does not
# cross-compile.
import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

ROOT = Path(SPECPATH).parent
ICONS = ROOT / "build" / "icons"
VERSION = re.search(r'__version__\s*=\s*"([^"]+)"',
                    (ROOT / "src" / "rcta" / "__init__.py").read_text()).group(1)
NAME = "Radiacode Track Analyzer"

# Qt modules the app never imports. Left in, WebEngine alone adds ~200 MB.
EXCLUDES = [
    "tkinter", "matplotlib", "IPython", "pytest", "PyQt5", "PyQt6",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick", "PySide6.QtWebChannel", "PySide6.QtWebSockets",
    "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuickWidgets",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.QtMultimedia",
    "PySide6.QtPdf", "PySide6.QtPdfWidgets", "PySide6.QtCharts",
    "PySide6.QtDataVisualization", "PySide6.QtBluetooth",
    "PySide6.QtPositioning", "PySide6.QtLocation", "PySide6.QtSql",
    "PySide6.QtDesigner", "PySide6.QtHelp", "PySide6.QtUiTools",
]

a = Analysis(
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT / "src")],
    datas=collect_data_files("rcta"),
    hiddenimports=["PySide6.QtSvg"],
    excludes=EXCLUDES,
    noarchive=False,
)
pyz = PYZ(a.pure)

icon = str(ICONS / ("rcta.icns" if sys.platform == "darwin" else "rcta.ico"))
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="rcta",
    console=False,
    icon=icon,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="rcta", upx=False)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{NAME}.app",
        icon=icon,
        bundle_identifier="io.github.lorenzoconfalonieri.rcta",
        version=VERSION,
        info_plist={
            "CFBundleDisplayName": NAME,
            "CFBundleShortVersionString": VERSION,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.0",
        },
    )
