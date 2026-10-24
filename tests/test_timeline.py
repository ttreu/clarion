"""Dragging a section boundary in the timeline (headless)."""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtCore = pytest.importorskip("PySide6.QtCore")
QtGui = pytest.importorskip("PySide6.QtGui")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from klangrestaurator.analysis import MIX, SPEECH, Segment  # noqa: E402
from klangrestaurator.gui.timeline import Timeline  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _mouse(w, kind, x, buttons):
    pos = QtCore.QPointF(x, 30)
    btn = QtCore.Qt.LeftButton if kind != QtCore.QEvent.MouseMove else QtCore.Qt.NoButton
    ev = QtGui.QMouseEvent(kind, pos, w.mapToGlobal(pos), btn, buttons, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(w, ev)


def test_drag_boundary_emits_move_and_keeps_project_list(app):
    w = Timeline()
    w.resize(1000, 160)
    segs = [Segment(0, 30, SPEECH), Segment(30, 60, MIX), Segment(60, 100, SPEECH)]
    w.set_data(np.full(400, -30.0), segs, 100.0)
    got = []
    w.boundary_moved.connect(lambda a, b: got.append((a, b)))
    sel = []
    w.selection_changed.connect(lambda a, b: sel.append((a, b)))
    L, none = QtCore.Qt.LeftButton, QtCore.Qt.NoButton
    _mouse(w, QtCore.QEvent.MouseButtonPress, 302, L)       # 2 px next to the boundary at 30 s
    _mouse(w, QtCore.QEvent.MouseMove, 400, L)
    _mouse(w, QtCore.QEvent.MouseButtonRelease, 400, none)
    assert len(got) == 1 and got[0][0] == 30 and abs(got[0][1] - 40) < 0.2
    assert segs[0].end == 30                                 # the caller's list is not modified
    assert sel == []                                         # no selection was made
    # away from a boundary, dragging still selects
    _mouse(w, QtCore.QEvent.MouseButtonPress, 700, L)
    _mouse(w, QtCore.QEvent.MouseMove, 800, L)
    _mouse(w, QtCore.QEvent.MouseButtonRelease, 800, none)
    assert len(got) == 1 and sel and abs(sel[-1][0] - 70) < 0.2 and abs(sel[-1][1] - 80) < 0.2


def test_loudness_checkbox_disables_controls(app, monkeypatch):
    monkeypatch.setenv("KLANGRESTAURATOR_NO_AUDIO", "1")
    from klangrestaurator.gui.app import MainWindow
    w = MainWindow()
    try:
        was = w.chk_norm.isChecked()
        w.chk_norm.setChecked(False)
        assert not w.cmb_loud.isEnabled()
        assert w.settings().effective().loudness_target is None
        w.chk_norm.setChecked(True)
        assert w.cmb_loud.isEnabled() and w.settings().normalize
        w.chk_norm.setChecked(was)                     # leave the user's preference as it was
    finally:
        w.close()


def test_previews_are_remembered_and_reused(app, monkeypatch, tmp_path, reading_file, no_dfn):
    """Switching back to an already computed setting must not compute again."""
    import time

    monkeypatch.setenv("KLANGRESTAURATOR_NO_AUDIO", "1")
    from klangrestaurator import pipeline
    from klangrestaurator.gui.app import MainWindow
    calls = []
    orig = pipeline.Project.write_preview
    monkeypatch.setattr(pipeline.Project, "write_preview",
                        lambda self, st, a, b, cancel=None, slot=0: (calls.append(st.normalize),
                                                                    orig(self, st, a, b, cancel, slot))[1])
    w = MainWindow()
    was = w.chk_norm.isChecked()

    def pump(cond, limit=30):
        t = time.time()
        while time.time() - t < limit and not cond():
            app.processEvents()
            time.sleep(0.01)

    try:
        w.chk_norm.setChecked(True)
        path, _ = reading_file
        w.load(path)
        pump(lambda: w.project is not None and w.project.analysed and w._job is None)
        w.timeline.sel = (2.0, 8.0)
        w.make_preview()
        pump(lambda: w._job is None and w.files_ab is not None)
        for on in (False, True, False):
            w.chk_norm.setChecked(on)
            pump(lambda: not w._preview_stale and w._job is None)
        assert calls == [True, False]           # only the first visit of each state was computed
        assert len(w._previews) == 2
        assert w._auto_timer.interval() == 400     # sliders keep their debounce
    finally:
        w.chk_norm.setChecked(was)
        w.close()
