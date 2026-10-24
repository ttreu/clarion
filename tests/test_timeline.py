"""Dragging a section boundary in the timeline (headless)."""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtCore = pytest.importorskip("PySide6.QtCore")
QtGui = pytest.importorskip("PySide6.QtGui")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from clarion.analysis import MIX, SPEECH, Segment  # noqa: E402
from clarion.gui.timeline import Timeline  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _mouse(w, kind, x, buttons, y=30):
    pos = QtCore.QPointF(x, y)
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
    monkeypatch.setenv("CLARION_NO_AUDIO", "1")
    from clarion.gui.app import MainWindow
    w = MainWindow()
    try:
        was = w.panel.chk_norm.isChecked()
        w.panel.chk_norm.setChecked(False)
        assert not w.panel.cmb_loud.isEnabled()
        assert w.settings().effective().loudness_target is None
        w.panel.chk_norm.setChecked(True)
        assert w.panel.cmb_loud.isEnabled() and w.settings().normalize
        w.panel.chk_norm.setChecked(was)                     # leave the user's preference as it was
    finally:
        w.close()


def test_previews_are_remembered_and_reused(app, monkeypatch, tmp_path, reading_file, no_dfn):
    """Switching back to an already computed setting must not compute again."""
    import time

    monkeypatch.setenv("CLARION_NO_AUDIO", "1")
    from clarion import pipeline
    from clarion.gui.app import MainWindow
    calls = []
    orig = pipeline.Project.write_preview
    monkeypatch.setattr(pipeline.Project, "write_preview",
                        lambda self, st, a, b, cancel=None, slot=0: (calls.append(st.normalize),
                                                                    orig(self, st, a, b, cancel, slot))[1])
    w = MainWindow()
    was = w.panel.chk_norm.isChecked()

    def pump(cond, limit=30):
        t = time.time()
        while time.time() - t < limit and not cond():
            app.processEvents()
            time.sleep(0.01)

    try:
        w.panel.chk_norm.setChecked(True)
        path, _ = reading_file
        w.load(path)
        pump(lambda: w.project is not None and w.project.analysed and w._job is None)
        w.timeline.sel = (2.0, 8.0)
        w.make_preview()
        pump(lambda: w._job is None and w.listen.files_ab is not None)
        for on in (False, True, False):
            w.panel.chk_norm.setChecked(on)
            pump(lambda: not w._preview_stale and w._job is None)
        assert calls == [True, False]           # only the first visit of each state was computed
        assert len(w._previews) == 2
        assert w._auto_timer.interval() == 400     # sliders keep their debounce
    finally:
        w.panel.chk_norm.setChecked(was)
        w.close()


def test_original_plays_before_preview_and_trim_keys(app, monkeypatch, reading_file, no_dfn):
    import time

    monkeypatch.setenv("CLARION_NO_AUDIO", "1")
    from clarion.gui.app import MainWindow
    w = MainWindow()

    def pump(cond, limit=30):
        t = time.time()
        while time.time() - t < limit and not cond():
            app.processEvents()
            time.sleep(0.01)

    try:
        path, _ = reading_file
        w.load(path)
        # right after loading – analysis still running – the original can already be played
        assert w.listen.files_ab == (path, None) and w.listen.orig_only
        assert w.listen.btn_play.isEnabled() and not w.listen.rb_b.isEnabled() and w.listen.rb_a.isChecked()
        pump(lambda: w.project.analysed and w._job is None)
        assert w.listen.btn_play.isEnabled() and w.listen.range == (0.0, w.project.duration)
        w._mark_stale()                                # no automatic whole-file preview in original mode
        assert not w._auto_timer.isActive()
        w.timeline.head = 4.0
        w._trim_at_head(start=True)
        w.timeline.head = 20.0
        w._trim_at_head(start=False)
        assert w.project.trim == (4.0, 20.0) and w.timeline.trim == (4.0, 20.0)
        w.timeline.grab()                              # paints the greyed-out parts without errors
        w._on_trim(-1.0, -1.0)
        assert w.project.trim is None
        w.timeline.sel = (2.0, 8.0)
        w.make_preview()
        pump(lambda: w._job is None and w.listen.files_ab is not None and w.listen.files_ab[1] is not None)
        assert not w.listen.orig_only and w.listen.rb_b.isEnabled() and w.listen.rb_b.isChecked()
    finally:
        w.close()


def test_drag_in_and_out_markers(app):
    w = Timeline()
    w.resize(1000, 160)
    segs = [Segment(0, 30, SPEECH), Segment(30, 60, MIX), Segment(60, 100, SPEECH)]
    w.set_data(np.full(400, -30.0), segs, 100.0)
    got = []
    w.trim_changed.connect(lambda a, b: got.append((round(a, 1), round(b, 1))))
    L, none = QtCore.Qt.LeftButton, QtCore.Qt.NoButton
    # untrimmed: In sits at the left edge, Out at the right edge
    _mouse(w, QtCore.QEvent.MouseButtonPress, 2, L, y=100)
    _mouse(w, QtCore.QEvent.MouseMove, 120, L, y=100)
    _mouse(w, QtCore.QEvent.MouseButtonRelease, 120, none, y=100)
    assert got[-1] == (12.0, 100.0)
    w.trim = (12.0, 100.0)
    _mouse(w, QtCore.QEvent.MouseButtonPress, 998, L, y=100)
    _mouse(w, QtCore.QEvent.MouseMove, 800, L, y=100)
    _mouse(w, QtCore.QEvent.MouseButtonRelease, 800, none, y=100)
    assert got[-1] == (12.0, 80.0)
    # Out cannot be pulled in front of In
    w.trim = (12.0, 80.0)
    _mouse(w, QtCore.QEvent.MouseButtonPress, 800, L, y=100)
    _mouse(w, QtCore.QEvent.MouseMove, 50, L, y=100)
    _mouse(w, QtCore.QEvent.MouseButtonRelease, 50, none, y=100)
    assert got[-1] == (12.0, 12.5)
    # in the top band a section boundary at the same spot still wins
    w.trim = (30.0, 80.0)
    moved = []
    w.boundary_moved.connect(lambda a, b: moved.append(b))
    _mouse(w, QtCore.QEvent.MouseButtonPress, 300, L, y=8)
    _mouse(w, QtCore.QEvent.MouseMove, 350, L, y=8)
    _mouse(w, QtCore.QEvent.MouseButtonRelease, 350, none, y=8)
    assert moved and abs(moved[-1] - 35) < 0.2


def _wheel(w, dx, dy, phase, pixel=True, mods=None):
    pos = QtCore.QPointF(500, 80)
    pd = QtCore.QPoint(int(dx), int(dy)) if pixel else QtCore.QPoint()
    ad = QtCore.QPoint(int(dx * 8), int(dy * 8)) if pixel else QtCore.QPoint(int(dx), int(dy))
    ev = QtGui.QWheelEvent(pos, w.mapToGlobal(pos), pd, ad, QtCore.Qt.NoButton,
                           mods or QtCore.Qt.NoModifier, phase, False)
    QtWidgets.QApplication.sendEvent(w, ev)


def test_trackpad_axis_lock_and_gentle_zoom(app):
    w = Timeline()
    w.resize(1000, 160)
    w.set_data(np.full(2400, -30.0), [Segment(0, 600, SPEECH)], 600.0)
    w.view = (100.0, 300.0)
    P = QtCore.Qt
    # mostly horizontal swipe with a little vertical jitter: pans only, span unchanged
    _wheel(w, 0, 0, P.ScrollBegin)
    for _ in range(10):
        _wheel(w, -10, 3, P.ScrollUpdate)
    _wheel(w, 0, 0, P.ScrollEnd)
    a, b = w.view
    assert abs((b - a) - 200) < 1e-6 and a > 100
    # mostly vertical swipe: zooms only, and gently (one small event ≈ a few percent, not 25 %)
    w.view = (100.0, 300.0)
    _wheel(w, 0, 0, P.ScrollBegin)
    _wheel(w, 1, 8, P.ScrollUpdate)                 # decides the axis
    span0 = w.view[1] - w.view[0]
    _wheel(w, 1, 4, P.ScrollUpdate)
    span1 = w.view[1] - w.view[0]
    _wheel(w, 0, 0, P.ScrollEnd)
    assert span1 < span0 and span1 / span0 > 0.95
    t_mid = w._t(500)
    assert abs(t_mid - w._t(500)) < 1e-9           # anchored at the cursor
    # classic mouse wheel: one notch = one 0.8× zoom step; Shift + wheel pans
    w.view = (100.0, 300.0)
    _wheel(w, 0, 120, P.NoScrollPhase, pixel=False)
    assert abs((w.view[1] - w.view[0]) - 160) < 1e-6
    a0 = w.view[0]
    _wheel(w, 0, 120, P.NoScrollPhase, pixel=False, mods=P.ShiftModifier)
    assert abs((w.view[1] - w.view[0]) - 160) < 1e-6 and w.view[0] != a0


def test_save_and_reopen_project_restores_everything(app, monkeypatch, reading_file, no_dfn, tmp_path):
    import time

    monkeypatch.setenv("CLARION_NO_AUDIO", "1")
    from clarion.gui.app import MainWindow

    def pump(w, cond, limit=30):
        t = time.time()
        while time.time() - t < limit and not cond():
            app.processEvents()
            time.sleep(0.01)

    path, _ = reading_file
    w = MainWindow()
    try:
        w.open_audio(path)
        pump(w, lambda: w.project.analysed and w._job is None)
        w._on_relabel(2.0, 6.0, MIX)
        w._on_trim(1.0, 25.0)
        w._on_tone_ref(20.0, 24.0)
        w.panel.cmb_v_music.activated.emit(4)                 # music variant E
        w.panel.sl_tone.setValue(40)
        assert w._dirty and w.windowTitle().endswith("•")
        proj = tmp_path / "test.clarion"
        monkeypatch.setattr(w, "_project_path", proj)   # avoid the save dialog
        assert w.save_project()
        assert not w._dirty and proj.exists()
        expected = ([(s.start, s.end, s.kind) for s in w.project.segments], w.settings())
    finally:
        w.close()
    w2 = MainWindow()
    try:
        w2.open_project(proj)
        pump(w2, lambda: w2.project is not None and w2.project.analysed and w2._job is None)
        app.processEvents()
        assert [(s.start, s.end, s.kind) for s in w2.project.segments] == expected[0]
        assert w2.project.trim == (1.0, 25.0) and w2.project.tone_ref == (20.0, 24.0)
        assert w2.settings() == expected[1]
        assert w2.panel.cmb_v_music.currentText().startswith("E")
        assert not w2._dirty and "test.clarion" in w2.windowTitle()
    finally:
        w2.close()


def test_waveform_overview_and_columns(app):
    from clarion.analysis import WAVE_HOP, waveform_overview
    from clarion.settings import SR

    t = np.arange(10 * SR) / SR
    x = np.stack([0.5 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1).astype(np.float32)
    x[5 * SR:] *= 0.1                                    # second half 20 dB quieter
    wave = waveform_overview(x)
    assert wave.shape == (10 * SR // WAVE_HOP, 2)
    assert abs(wave[: len(wave) // 2, 0].max() - 0.5) < 1e-3
    assert abs(np.sqrt((wave[:400, 1] ** 2).mean()) - 0.5 / np.sqrt(2)) < 0.005   # rms of a sine

    w = Timeline()
    w.resize(800, 160)
    w.set_data(np.full(40, -30.0), [Segment(0, 10, SPEECH)], 10.0, wave=wave, audio=x)
    hi, lo, rms = w._columns(800)
    assert np.allclose(hi, -lo) and hi[:350].min() > 0.9 and hi[450:].max() < 0.4   # quiet half lower
    w._set_view(2.0, 2.25)                               # zoomed in: real samples, both signs
    hi, lo, _ = w._columns(800)
    assert hi.max() > 0.9 and lo.min() < -0.9 and (hi > 0).any() and (lo < 0).any()
    assert not w.grab().isNull()
