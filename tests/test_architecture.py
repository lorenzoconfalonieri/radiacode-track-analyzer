"""Structural rules, enforced mechanically rather than remembered.

Every rule here encodes a mistake that is easy to make and expensive to find
later. A stray ``from PySide6 import ...`` inside an analysis module does not
fail any functional test -- it just makes the core untestable headlessly and
un-reusable from the CLI, and it is discovered months later by someone trying
to run an analysis on a server. Likewise a ``urllib`` import inside an
estimator makes results silently depend on network reachability.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "rcta"

QT_MODULES = {"PySide6", "PyQt5", "PyQt6", "pyqtgraph"}
NETWORK_MODULES = {"urllib", "http", "socket", "ssl", "requests", "ftplib", "telnetlib"}
THREAD_MODULES = {"threading", "concurrent", "multiprocessing", "asyncio"}

#: Threading is a maintenance cost worth paying in as few places as possible.
#: It is allowed in exactly one module, which exists to keep it contained.
THREAD_ALLOWED = {"gui/worker.py"}


def _core_files() -> list[Path]:
    return sorted((SRC / "core").rglob("*.py"))


def _gui_files() -> list[Path]:
    return sorted((SRC / "gui").rglob("*.py"))


def _imported_roots(path: Path) -> set[str]:
    """Top-level module names imported by ``path``."""
    tree = ast.parse(path.read_text(), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _sources() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _rel(path: Path) -> str:
    return path.relative_to(SRC).as_posix()


@pytest.mark.parametrize("path", _core_files(), ids=_rel)
def test_core_never_imports_qt(path):
    """The core must run headless, on a server, and from the CLI."""
    offenders = _imported_roots(path) & QT_MODULES
    assert not offenders, f"{_rel(path)} imports {sorted(offenders)}"


@pytest.mark.parametrize("path", _core_files(), ids=_rel)
def test_core_never_imports_matplotlib(path):
    """Plotting is pyqtgraph's job. The core returns numbers."""
    assert "matplotlib" not in _imported_roots(path), _rel(path)


@pytest.mark.parametrize("path", _core_files(), ids=_rel)
def test_core_never_reaches_the_network(path):
    """An estimator that fetches something is not reproducible offline.

    Tile retrieval lives in ``gui/tiles.py`` precisely so that this rule can be
    absolute: no module under ``core/`` may import a network library at all.
    """
    offenders = _imported_roots(path) & NETWORK_MODULES
    assert not offenders, f"{_rel(path)} imports {sorted(offenders)}"


@pytest.mark.parametrize("path", _core_files() + _gui_files(), ids=_rel)
def test_threading_is_confined_to_the_worker(path):
    """Concurrency is allowed in one module, so its bugs have one home."""
    if _rel(path) in THREAD_ALLOWED:
        return
    offenders = _imported_roots(path) & THREAD_MODULES
    assert not offenders, f"{_rel(path)} imports {sorted(offenders)}"


@pytest.mark.parametrize("path", _core_files(), ids=_rel)
def test_core_modules_have_essay_docstrings(path):
    """A core module explains the failure mode its design prevents.

    Enforcing a length floor mechanically is cheaper than remembering to, and
    it catches the module that was written in a hurry and never documented.
    """
    if path.name == "__init__.py":
        return
    doc = ast.get_docstring(ast.parse(path.read_text(), filename=str(path)))
    assert doc, f"{_rel(path)} has no module docstring"
    assert len(doc.splitlines()) >= 10, (
        f"{_rel(path)} docstring is {len(doc.splitlines())} lines; "
        "explain what the design prevents, not just what it does"
    )


def _net_files() -> list[Path]:
    return sorted((SRC / "net").rglob("*.py"))


#: Network access is allowed in exactly two places, and the point of naming
#: them is that the list is short enough to audit. ``gui/tiles.py`` is Qt-based
#: because the map is interactive; ``net/`` is stdlib because the CLI needs it
#: too. Anywhere else is a mistake.
NETWORK_ALLOWED = {"gui/tiles.py"}


@pytest.mark.parametrize("path", _net_files(), ids=_rel)
def test_net_never_imports_qt(path):
    """``net/`` must work with no Qt installed.

    The whole reason the flight client is not built on ``QNetworkAccessManager``
    like the tile fetcher is that ``rcta-cli flight`` has to run headless, in
    CI and on a server where PySide6 is not present. A stray Qt import here
    would not fail any functional test on a developer's machine and would
    break the CLI everywhere else.
    """
    offenders = _imported_roots(path) & QT_MODULES
    assert not offenders, f"{_rel(path)} imports {sorted(offenders)}"


@pytest.mark.parametrize("path", _net_files(), ids=_rel)
def test_net_modules_have_essay_docstrings(path):
    """The same standard as ``core/``: explain the failure mode prevented."""
    if path.name == "__init__.py":
        return
    doc = ast.get_docstring(ast.parse(path.read_text(), filename=str(path)))
    assert doc, f"{_rel(path)} has no module docstring"
    assert len(doc.splitlines()) >= 10, (
        f"{_rel(path)} docstring is {len(doc.splitlines())} lines")


@pytest.mark.parametrize(
    "path", _core_files() + _gui_files() + _net_files(), ids=_rel)
def test_network_access_is_confined(path):
    """Only the two named modules may import a network library.

    ``test_core_never_reaches_the_network`` states the rule for ``core/``;
    this states it for the whole package, so that a new panel or a new
    exporter cannot quietly acquire a dependency on a server being up.
    """
    rel = _rel(path)
    if rel in NETWORK_ALLOWED or rel.startswith("net/"):
        return
    offenders = _imported_roots(path) & NETWORK_MODULES
    assert not offenders, f"{rel} imports {sorted(offenders)}"


@pytest.mark.parametrize("path", _net_files(), ids=_rel)
def test_net_adds_no_threads(path):
    """Concurrency still has exactly one home, and it is not this package."""
    offenders = _imported_roots(path) & THREAD_MODULES
    assert not offenders, f"{_rel(path)} imports {sorted(offenders)}"


def test_every_text_read_and_write_names_its_encoding():
    """Without one, Python uses the locale's: on Windows (cp1252) every map
    attribution read from providers.json came out as "Â© OpenStreetMap"."""
    import ast
    offenders = []
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name not in ("read_text", "write_text", "open"):
                continue
            if name == "open" and not isinstance(node.func, ast.Name):
                continue                      # urlopen, QFile.open, ...
            keywords = {k.arg for k in node.keywords}
            modes = [a.value for a in node.args[1:2]
                     if isinstance(a, ast.Constant)]
            if "encoding" in keywords or any("b" in str(m) for m in modes):
                continue
            offenders.append(f"{_rel(path)}:{node.lineno} {name}()")
    assert not offenders, "text I/O without an encoding: " + ", ".join(offenders)
