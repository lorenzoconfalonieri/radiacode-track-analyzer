"""The only threading in this program, and the fence that keeps it contained.

No threading at all is the right default for a program this size, and that
restraint is worth keeping wherever it is affordable. Two jobs go to a worker:
computing the interpolated grid, which scales into seconds on a long survey,
and fetching a flight trajectory, which waits on someone else's server.
Loading a file and running the analysis stay on the GUI thread; both finish
in well under a second on the bundled tracks, and a synchronous load keeps
"the window shows the file you opened" trivially true. Tile retrieval does
*not* need a worker either -- ``QNetworkAccessManager`` is already
asynchronous on the Qt event loop, so the map adds no threads.

Three rules make this safe rather than merely functional.

*A job is a plain callable from* :mod:`rcta.core` *plus plain arguments.* The
worker never touches a widget, and the core never learns that Qt exists: long
functions accept ``progress`` and ``cancelled`` as ordinary callables, which is
exactly why the architecture test still passes with this module present.

*Every job is submitted under a name carrying a monotonic token.* A result
whose token is stale is dropped. Without this, dragging the bandwidth slider
queues five grid computations and the slowest one wins, so the map ends up
showing a bandwidth the user moved past four seconds ago.

*Tracks and grids are frozen dataclasses over arrays with* ``writeable=False``.
A worker and the GUI thread can therefore hold the same object with no lock.
That property is load-bearing, not incidental.
"""

from __future__ import annotations

import inspect
import traceback

from PySide6 import QtCore

__all__ = ["JobQueue"]


def _describe(exc: BaseException) -> str:
    """A failure message whose first paragraph is fit to put in front of a user.

    :class:`rcta.net.flights.FlightFetchError` deliberately carries the
    actionable next step -- register for an account, widen the window, load a
    local file -- in a separate ``hint`` attribute, because the message alone
    says only that something failed. The CLI prints both. Stringifying the
    exception here would drop the hint and hand the GUI the half that cannot
    be acted on, which is how "OpenSky refuses historical data to anonymous
    callers" ends up on screen with no mention that the account is free.

    Read by duck typing rather than by importing the class: this module knows
    nothing about the network, and any job may raise.

    Everything before the first blank line is the explanation and everything
    after it is the traceback. That split is the contract callers render
    against -- a dock caption gets the paragraph, the log gets all of it.
    """
    hint = str(getattr(exc, "hint", "") or "").strip()
    head = f"{exc}\nhint: {hint}" if hint else f"{exc}"
    return f"{head}\n\n{traceback.format_exc(limit=3)}"


class _JobSignals(QtCore.QObject):
    finished = QtCore.Signal(str, int, object)    # name, token, result
    failed = QtCore.Signal(str, int, str)         # name, token, message
    progress = QtCore.Signal(str, int, float)     # name, token, fraction


class _Job(QtCore.QRunnable):
    """Runs one callable, injecting progress/cancelled only if it accepts them."""

    def __init__(self, name, token, fn, args, kwargs, signals, is_cancelled):
        super().__init__()
        self._name = name
        self._token = token
        self._fn = fn
        self._args = args
        self._kwargs = dict(kwargs)
        self._signals = signals
        self._is_cancelled = is_cancelled
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            accepted = inspect.signature(self._fn).parameters
            if "progress" in accepted:
                self._kwargs["progress"] = self._emit_progress
            if "cancelled" in accepted:
                self._kwargs["cancelled"] = self._cancelled
            result = self._fn(*self._args, **self._kwargs)
        except InterruptedError:
            return                       # a cancelled job is not a failure
        except Exception as exc:         # keep the GUI alive, always
            self._signals.failed.emit(self._name, self._token, _describe(exc))
            return
        self._signals.finished.emit(self._name, self._token, result)

    def _cancelled(self) -> bool:
        return self._is_cancelled(self._name, self._token)

    def _emit_progress(self, fraction: float) -> None:
        if not self._cancelled():
            self._signals.progress.emit(self._name, self._token, float(fraction))


class JobQueue(QtCore.QObject):
    """Named background jobs with staleness tokens. Owned by MainWindow.

    ``synchronous=True`` runs everything inline on the calling thread, which is
    what the test suite uses: no thread to wait for, no flake, and the same
    signals are emitted in the same order.
    """

    finished = QtCore.Signal(str, int, object)
    failed = QtCore.Signal(str, int, str)
    progress = QtCore.Signal(str, int, float)

    def __init__(self, parent=None, *, synchronous: bool = False,
                 max_threads: int = 0) -> None:
        super().__init__(parent)
        self._synchronous = bool(synchronous)
        self._tokens: dict[str, int] = {}
        self._active: dict[str, int] = {}
        self._pool = QtCore.QThreadPool()
        if max_threads:
            self._pool.setMaxThreadCount(int(max_threads))

        self._signals = _JobSignals()
        self._signals.finished.connect(self._on_finished)
        self._signals.failed.connect(self._on_failed)
        self._signals.progress.connect(self._on_progress)

    # ---- submission -------------------------------------------------------

    def submit(self, name: str, fn, *args, **kwargs) -> int:
        """Queue ``fn`` under ``name``, superseding any earlier job of that name."""
        token = self._tokens.get(name, 0) + 1
        self._tokens[name] = token
        self._active[name] = token

        job = _Job(name, token, fn, args, kwargs, self._signals, self._is_stale_check)
        if self._synchronous:
            job.run()
        else:
            self._pool.start(job)
        return token

    def cancel(self, name: str) -> None:
        """Abandon the current job of that name; its result will be dropped."""
        self._active.pop(name, None)

    def cancel_all(self) -> None:
        self._active.clear()

    def is_running(self, name: str) -> bool:
        return name in self._active

    def shutdown(self, timeout_ms: int = 2000) -> None:
        """Stop accepting results and wait briefly for workers to notice."""
        self.cancel_all()
        self._pool.waitForDone(int(timeout_ms))

    # ---- staleness --------------------------------------------------------

    def _is_stale_check(self, name: str, token: int) -> bool:
        return self._active.get(name) != token

    def _current(self, name: str, token: int) -> bool:
        return self._active.get(name) == token

    def _on_finished(self, name, token, result) -> None:
        if self._current(name, token):
            self._active.pop(name, None)
            self.finished.emit(name, token, result)

    def _on_failed(self, name, token, message) -> None:
        if self._current(name, token):
            self._active.pop(name, None)
            self.failed.emit(name, token, message)

    def _on_progress(self, name, token, fraction) -> None:
        if self._current(name, token):
            self.progress.emit(name, token, fraction)
