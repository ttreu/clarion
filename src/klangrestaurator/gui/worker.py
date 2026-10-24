"""Run long jobs off the GUI thread with progress and cancellation."""
from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

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


# Python owns each Job. Using deleteLater on it as well caused a double delete (segfault) once the
# caller dropped its reference, so we hold (thread, job) here and release finished ones on the GUI
# thread at the next start().
_alive: list[tuple[QThread, Job]] = []


def _reap() -> None:
    for pair in [p for p in _alive if p[0].isFinished()]:
        _alive.remove(pair)
        pair[0].deleteLater()


def start(parent: QObject, fn: Callable[..., Any]) -> tuple[QThread, Job]:
    """Create thread + job; caller connects signals, then calls thread.start()."""
    _reap()
    th = QThread(parent)
    job = Job(fn)
    job.moveToThread(th)
    th.started.connect(job.run)
    for sig in (job.done, job.failed, job.cancelled):
        sig.connect(th.quit)
    _alive.append((th, job))
    return th, job


def wait_all(timeout_ms: int = 10000) -> None:
    """Block until running jobs have stopped (call after cancelling, e.g. on window close)."""
    for th, _ in _alive:
        th.wait(timeout_ms)
