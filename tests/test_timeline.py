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
