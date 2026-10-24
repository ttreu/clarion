"""Level overview with speech/music segments, playhead, selection and zoom."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QMenu, QSizePolicy, QWidget

from ..analysis import KIND_LABEL, MIX, MUSIC, SPEECH, Segment

COL_SPEECH = QColor(70, 130, 220, 70)
COL_MUSIC = QColor(235, 160, 40, 90)
COL_MIX = QColor(150, 90, 200, 80)
KIND_COL = {SPEECH: COL_SPEECH, MUSIC: COL_MUSIC, MIX: COL_MIX}
COL_WAVE = QColor(60, 60, 70)
COL_SEL = QColor(40, 170, 120, 70)
COL_HEAD = QColor(220, 40, 60)
COL_REF = QColor(30, 160, 90)


def fmt(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


class Timeline(QWidget):
    seek = Signal(float)                      # user clicked a position (s)
    selection_changed = Signal(float, float)  # (start, end) s, end<=start means none
    relabel = Signal(float, float, str)       # mark range as speech/music/mix
    tone_ref = Signal(float, float)           # use range as tone reference; (-1, -1) = automatic
    boundary_moved = Signal(float, float)     # a boundary between two sections was dragged: (old, new) s

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.levels: np.ndarray | None = None  # dB per 0.25 s
        self.segments: list[Segment] = []
        self.duration = 0.0
        self.view = (0.0, 1.0)  # visible seconds
        self.head = 0.0
        self.sel = (0.0, 0.0)
        self._drag_from: float | None = None
        self._hover: float | None = None
        self.ref: tuple[float, float] | None = None  # tone reference range (shown as a green bar)
        self._bound: int | None = None      # index i of the dragged boundary (between segments i and i+1)
        self._bound_from = 0.0

    # ---------------------------------------------------------------- data
    def set_data(self, levels: np.ndarray, segments: list[Segment], duration: float) -> None:
        self.levels, self.segments, self.duration = levels, segments, duration
        self.view = (0.0, max(duration, 1.0))
        self.sel = (0.0, 0.0)
        self.update()

    def set_segments(self, segments: list[Segment]) -> None:
        self.segments = segments
        self.update()

    def set_head(self, t: float) -> None:
        self.head = t
        a, b = self.view
        if t > b or t < a:  # keep playhead visible when zoomed
            w = b - a
            self.view = (max(0.0, min(self.duration - w, t - w * 0.1)), 0)
            self.view = (self.view[0], self.view[0] + w)
        self.update()

    # ---------------------------------------------------------------- geometry
    def _x(self, t: float) -> float:
        a, b = self.view
        return (t - a) / (b - a) * self.width()

    def _t(self, x: float) -> float:
        a, b = self.view
        return float(np.clip(a + x / max(1, self.width()) * (b - a), 0, self.duration))

    # ---------------------------------------------------------------- painting
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), self.palette().base())
        if self.levels is None:
            p.setPen(self.palette().placeholderText().color())
            p.drawText(self.rect(), Qt.AlignCenter, "Audiodatei hierher ziehen oder „Öffnen…“ wählen")
            return
        a, b = self.view
        lane = 18
        for s in self.segments:  # segment bands
            if s.end < a or s.start > b:
                continue
            x0, x1 = self._x(s.start), self._x(s.end)
            col = KIND_COL.get(s.kind, COL_SPEECH)
            p.fillRect(QRectF(x0, lane, x1 - x0, h - lane), col)
            solid = QColor(col)
            solid.setAlpha(230)
            p.fillRect(QRectF(x0, 0, x1 - x0, lane - 2), solid)
            if s.manual:
                p.fillRect(QRectF(x0, lane, x1 - x0, h - lane), QBrush(QColor(0, 0, 0, 40), Qt.BDiagPattern))
            if x1 - x0 > 60:
                p.setPen(Qt.white)
                p.drawText(QRectF(x0 + 4, 0, x1 - x0 - 8, lane - 2), Qt.AlignVCenter | Qt.AlignLeft,
                           KIND_LABEL.get(s.kind, s.kind))

        for i in range(len(self.segments) - 1):  # grips on the boundaries (draggable)
            t = self.segments[i].end
            if a <= t <= b:
                x = self._x(t)
                active = self._bound == i
                p.fillRect(QRectF(x - (2 if active else 1), 0, 4 if active else 2, lane - 2),
                           QColor(255, 255, 255, 255 if active else 200))

        # level envelope (dB mapped -70..0)
        lv = self.levels
        i0, i1 = max(0, int(a * 4)), min(len(lv), int(b * 4) + 1)
        if i1 > i0:
            seg = lv[i0:i1]
            cols = max(1, w)
            idx = np.linspace(0, len(seg), cols + 1).astype(int)
            mx = np.array([seg[idx[k]:max(idx[k] + 1, idx[k + 1])].max() for k in range(cols)])
            amp = np.clip((mx + 70) / 70, 0, 1) * (h - lane - 6) / 2
            mid = lane + (h - lane) / 2
            path = QPainterPath(QPointF(0, mid))
            for k in range(cols):
                path.lineTo(k, mid - amp[k])
            for k in range(cols - 1, -1, -1):
                path.lineTo(k, mid + amp[k])
            p.fillPath(path, COL_WAVE)

        s0, s1 = self.sel
        if s1 > s0:
            p.fillRect(QRectF(self._x(s0), lane, self._x(s1) - self._x(s0), h - lane), COL_SEL)
        if self.ref:  # tone reference
            r0, r1 = self._x(self.ref[0]), self._x(self.ref[1])
            p.fillRect(QRectF(r0, h - 24, max(2.0, r1 - r0), 4), COL_REF)
            if r1 - r0 > 90:
                p.setPen(COL_REF)
                p.drawText(QRectF(r0 + 2, h - 40, r1 - r0, 14), Qt.AlignLeft, "Klangreferenz")
        p.setPen(QPen(COL_HEAD, 2))
        xh = self._x(self.head)
        p.drawLine(QPointF(xh, 0), QPointF(xh, h))
        p.setPen(self.palette().text().color())
        p.drawText(QRectF(4, h - 18, 200, 16), Qt.AlignLeft, fmt(a))
        p.drawText(QRectF(w - 204, h - 18, 200, 16), Qt.AlignRight, fmt(b))
        if self._hover is not None:
            p.drawText(QRectF(self._x(self._hover) + 4, lane + 2, 120, 16), Qt.AlignLeft, fmt(self._hover))

    # ---------------------------------------------------------------- interaction
    def _boundary_at(self, x: float, tol: float = 5.0) -> int | None:
        """Index i of the boundary between segments i and i+1 within ``tol`` pixels of x."""
        best, dist = None, tol
        for i in range(len(self.segments) - 1):
            d = abs(self._x(self.segments[i].end) - x)
            if d <= dist:
                best, dist = i, d
        return best

    def mousePressEvent(self, e):
        if self.levels is None or e.button() != Qt.LeftButton:
            return
        i = self._boundary_at(e.position().x())
        if i is not None:  # grab a section boundary
            # work on copies: the list belongs to the project until the move is confirmed
            self.segments = [Segment(s.start, s.end, s.kind, s.manual) for s in self.segments]
            self._bound, self._bound_from = i, self.segments[i].end
            self.update()
            return
        self._drag_from = self._t(e.position().x())
        self.sel = (0.0, 0.0)
        self.update()

    def mouseMoveEvent(self, e):
        t = self._t(e.position().x())
        self._hover = t if self.levels is not None else None
        if self._bound is not None:
            left, right = self.segments[self._bound], self.segments[self._bound + 1]
            t = float(np.clip(t, left.start + 0.3, right.end - 0.3))
            left.end = right.start = t
            self._hover = t
        elif self._drag_from is not None:
            self.sel = (min(self._drag_from, t), max(self._drag_from, t))
        elif self.levels is not None:
            near = self._boundary_at(e.position().x()) is not None
            self.setCursor(Qt.SizeHorCursor if near else Qt.ArrowCursor)
        self.update()

    def mouseReleaseEvent(self, e):
        if self._bound is not None:
            new = self.segments[self._bound].end
            self._bound = None
            if abs(new - self._bound_from) > 1e-3:
                self.boundary_moved.emit(self._bound_from, new)
            self.update()
            return
        if self._drag_from is None:
            return
        t = self._t(e.position().x())
        if abs(self._x(t) - self._x(self._drag_from)) < 3:  # plain click = seek
            self.sel = (0.0, 0.0)
            self.head = t
            self.seek.emit(t)
        self._drag_from = None
        self.selection_changed.emit(*self.sel)
        self.update()

    def leaveEvent(self, _):
        self._hover = None
        self.update()

    def wheelEvent(self, e):
        if self.levels is None:
            return
        a, b = self.view
        t = self._t(e.position().x())
        dy = e.angleDelta().y()
        dx = e.angleDelta().x()
        if dx and not dy:  # horizontal scroll pans
            shift = -dx / 120 * (b - a) * 0.1
            a, b = a + shift, b + shift
        else:
            f = 0.8 if dy > 0 else 1.25
            a, b = t - (t - a) * f, t + (b - t) * f
        span = min(max(b - a, 5.0), self.duration)
        a = float(np.clip(a, 0, self.duration - span))
        self.view = (a, a + span)
        self.update()

    def contextMenuEvent(self, e):
        if self.levels is None:
            return
        s0, s1 = self.sel
        if s1 - s0 < 0.2:  # no selection: act on the segment under the mouse
            t = self._t(e.pos().x())
            seg = next((s for s in self.segments if s.start <= t < s.end), None)
            if seg is None:
                return
            s0, s1 = seg.start, seg.end
        m = QMenu(self)
        for label, kind in (("Als Sprache behandeln", SPEECH), ("Als Musik behandeln", MUSIC),
                            ("Als Sprache + Musik behandeln", MIX)):
            act = QAction(f"{label}  ({fmt(s0)} – {fmt(s1)})", m)
            act.triggered.connect(lambda _=False, k=kind: self.relabel.emit(s0, s1, k))
            m.addAction(act)
        m.addSeparator()
        act = QAction(f"Als Klangreferenz für die Stimme verwenden  ({fmt(s0)} – {fmt(s1)})", m)
        act.triggered.connect(lambda: self.tone_ref.emit(s0, s1))
        m.addAction(act)
        if self.ref:
            act = QAction("Klangreferenz: automatisch (trockenste Stellen)", m)
            act.triggered.connect(lambda: self.tone_ref.emit(-1.0, -1.0))
            m.addAction(act)
        m.addSeparator()
        zoom = QAction("Ganze Datei anzeigen", m)
        zoom.triggered.connect(self._zoom_all)
        m.addAction(zoom)
        m.exec(e.globalPos())

    def _zoom_all(self):
        self.view = (0.0, max(self.duration, 1.0))
        self.update()
