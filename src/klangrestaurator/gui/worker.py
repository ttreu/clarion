"""Run long jobs off the GUI thread with progress and cancellation."""
from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

import shiboken6
from PySide6.QtCore import QObject, QThread, Signal

from ..pipeline import Cancelled


class Job(QObject):
    progress = Signal(str, float)
    done = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, fn: Callable[..., Any]):
        super().__init__()
        self._fn = fn
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def is_cancelled(self) -> bool:
        return self._cancel

    def run(self) -> None:
        try:
            res = self._fn(progress=lambda s, v: self.progress.emit(s, float(v)), cancel=self.is_cancelled)
        except Cancelled:
            self.cancelled.emit()
        except InterruptedError:
            self.cancelled.emit()
        except Exception as e:  # noqa: BLE001 – show every failure to the user
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")
        else:
            self.done.emit(res)


class Handle(QObject):
    """Same signals as Job, but living on the caller's (GUI) thread.

    PySide calls plain Python callables (lambdas, closures) connected to a Job signal directly on the
    worker thread. Relaying through this QObject delivers every signal on the GUI thread instead, so
    callers may connect anything and touch widgets safely.
    """

    progress = Signal(str, float)
    done = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, job: Job, parent: QObject):
        super().__init__(parent)
        self._jobref = job
        for name in ("progress", "done", "failed", "cancelled"):
            getattr(job, name).connect(getattr(self, name))  # queued: receiver is on the GUI thread

    def cancel(self) -> None:
        self._jobref.cancel()


# Python owns each Job. Using deleteLater on it as well caused a double delete (segfault) once the
# caller dropped its reference, so we hold (thread, job) here and release finished ones on the GUI
# thread at the next start().
_alive: list[tuple[QThread, Job, Handle]] = []


def _reap() -> None:
    for entry in list(_alive):
        th = entry[0]
        if shiboken6.isValid(th) and not th.isFinished():
            continue
        _alive.remove(entry)  # finished, or already deleted together with its parent
        for obj in (th, entry[2]):
            if shiboken6.isValid(obj):
                obj.deleteLater()


def start(parent: QObject, fn: Callable[..., Any]) -> tuple[QThread, Handle]:
    """Create thread + job; caller connects to the returned handle's signals, then calls thread.start().

    All handle signals arrive on the thread of ``parent`` (the GUI thread).
    """
    _reap()
    th = QThread(parent)
    job = Job(fn)
    job.moveToThread(th)
    th.started.connect(job.run)
    for sig in (job.done, job.failed, job.cancelled):
        sig.connect(th.quit)
    handle = Handle(job, parent)
    _alive.append((th, job, handle))
    return th, handle


def wait_all(timeout_ms: int = 10000) -> None:
    """Block until running jobs have stopped (call after cancelling, e.g. on window close)."""
    for th, *_ in _alive:
        if shiboken6.isValid(th):
            th.wait(timeout_ms)
