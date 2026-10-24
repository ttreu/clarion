"""Job callbacks must arrive on the GUI thread, even plain lambdas/closures."""
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtCore = pytest.importorskip("PySide6.QtCore")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from klangrestaurator.gui import worker  # noqa: E402
from klangrestaurator.pipeline import Cancelled  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _run(app, fn):
    parent = QtCore.QObject()
    th, h = worker.start(parent, fn)
    seen = {}
    gui = QtCore.QThread.currentThread()
    h.progress.connect(lambda *_: seen.setdefault("progress", QtCore.QThread.currentThread() is gui))
    for name in ("done", "failed", "cancelled"):
        getattr(h, name).connect(lambda *_, n=name: seen.__setitem__(n, QtCore.QThread.currentThread() is gui))
    th.start()
    t = time.time()
    while not th.isFinished() and time.time() - t < 5:
        app.processEvents()
    for _ in range(20):
        app.processEvents()
    worker.wait_all()
    return seen


def test_done_on_gui_thread(app):
    def fn(progress, cancel):
        progress("x", 0.5)
        return 42

    assert _run(app, fn) == {"progress": True, "done": True}


def test_failed_and_cancelled_on_gui_thread(app):
    def boom(progress, cancel):
        raise ValueError("x")

    def stop(progress, cancel):
        raise Cancelled()

    assert _run(app, boom) == {"failed": True}
    assert _run(app, stop) == {"cancelled": True}
