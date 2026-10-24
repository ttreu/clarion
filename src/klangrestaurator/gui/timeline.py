"""Level overview with speech/music segments, playhead, selection and zoom."""
from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QLinearGradient, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QMenu, QSizePolicy, QWidget

from ..analysis import KIND_LABEL, MIX, MUSIC, SPEECH, WAVE_HOP, Segment
from ..settings import SR
from .style import C


def _c(hexcol: str, alpha: int = 255) -> QColor:
    q = QColor(hexcol)
    q.setAlpha(alpha)
    return q


COL_SPEECH = _c(C.speech, 26)
COL_MUSIC = _c(C.music, 26)
COL_MIX = _c(C.mix, 30)
KIND_COL = {SPEECH: COL_SPEECH, MUSIC: COL_MUSIC, MIX: COL_MIX}
# the waveform itself takes the colour of its section, a little lighter than the band
WAVE_COL = {SPEECH: QColor(C.speech).lighter(120), MUSIC: QColor(C.music),
            MIX: QColor(C.mix).lighter(115)}
COL_WAVE = _c(C.wave)
COL_SEL = _c("#ffffff", 34)
COL_HEAD = _c("#ffffff", 235)
COL_REF = _c(C.good)
COL_WAVE_OFF = _c(C.faint)                  # waveform outside the In/Out range
COL_TRIM = _c(C.accent)                     # In/Out markers


def _disp(v: np.ndarray, ref: float) -> np.ndarray:
    """Signed amplitude → −1..1 display height. Square-root scale relative to the file's peak: quiet
    speech stays visible (unlike linear) while the wave keeps its shape (unlike dB, which turns
    everything into a solid block and squares off waves when zoomed in)."""
    return np.sign(v) * np.sqrt(np.clip(np.abs(v) / ref, 0.0, 1.0))


def fmt(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


class Timeline(QWidget):
    seek = Signal(float)                      # user clicked a position (s)
    selection_changed = Signal(float, float)  # (start, end) s, end<=start means none
    relabel = Signal(float, float, str)       # mark range as speech/music/mix
    tone_ref = Signal(float, float)           # use range as tone reference; (-1, -1) = automatic
    boundary_moved = Signal(float, float)     # a boundary between two sections was dragged: (old, new) s
    trim_changed = Signal(float, float)       # new export range (start, end) s; (-1, -1) = no trim

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
        self.trim: tuple[float, float] | None = None  # export range; outside is greyed out
        self._trim_drag: str | None = None   # "in" / "out" while a marker is dragged
        self._axis: str | None = None        # trackpad gesture locked to "pan" or "zoom"
        self._axis_acc = (0.0, 0.0)          # movement collected before the axis is decided
        self._last_wheel = 0.0
        self._trim_from: tuple[float, float] | None = None
        self._bound: int | None = None      # index i of the dragged boundary (between segments i and i+1)
        self._bound_from = 0.0
        self._wave: np.ndarray | None = None  # (peak, rms) linear per bin
        self._hop = 0.25                     # bin length (s)
        self.audio: np.ndarray | None = None
        self._ref = 1.0                      # amplitude shown at full height (file peak + a little room)
        self._cols_key = None                # cache of the rendered waveform layer (see _wave_layer)
        self._cols: QPixmap | None = None

    # ---------------------------------------------------------------- data
    def set_data(self, levels: np.ndarray, segments: list[Segment], duration: float,
                 wave: np.ndarray | None = None, audio: np.ndarray | None = None) -> None:
        """``wave``: (peak, rms) per 5 ms (analysis.waveform_overview); ``audio``: the 48 kHz source
        for sample-exact drawing when zoomed far in. Without them the 0.25 s levels are drawn."""
        self.levels, self.segments, self.duration = levels, segments, duration
        if wave is None:
            lin = (10 ** (np.asarray(levels, np.float32) / 20))[:, None].repeat(2, axis=1)
            self._wave, self._hop = lin, 0.25
        else:
            self._wave, self._hop = np.asarray(wave, np.float32), WAVE_HOP / SR
        self.audio = audio
        self._ref = max(float(self._wave[:, 0].max()) if len(self._wave) else 0.0, 1e-4) / 0.96
        self._cols_key = None
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
            p.drawText(self.rect(), Qt.AlignCenter, "Drop an audio file or a project here, or choose “Open…”")
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
                p.fillRect(QRectF(x0, lane, x1 - x0, h - lane), QBrush(QColor(255, 255, 255, 28), Qt.BDiagPattern))
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

        # waveform: peak (light) and RMS (solid), coloured by section, grey outside In/Out
        t_in, t_out = self.trim or (0.0, self.duration)
        x_in, x_out = self._x(t_in), self._x(t_out)
        wash = QColor(self.palette().base().color())
        wash.setAlpha(180)
        outside = [r for r in (QRectF(0, 0, max(0.0, x_in), h), QRectF(x_out, 0, max(0.0, w - x_out), h))
                   if r.width() > 0]
        for r in outside:
            p.fillRect(r, wash)
        mid = lane + (h - lane) / 2
        half = (h - lane) / 2 - 4
        p.setPen(QPen(_c(C.line), 1))
        p.drawLine(QPointF(0, mid), QPointF(w, mid))
        p.drawPixmap(0, 0, self._wave_layer(w, h, mid, half, x_in, x_out))

        s0, s1 = self.sel
        if s1 > s0:
            p.fillRect(QRectF(self._x(s0), lane, self._x(s1) - self._x(s0), h - lane), COL_SEL)

        # In / Out markers (always there; drag them to cut start and end)
        for which, x, label in (("in", x_in, "In"), ("out", x_out, "Out")):
            if -6 <= x <= w + 6:
                xx = min(max(x, 1.0), w - 1.0)
                active = self._trim_drag == which
                p.setPen(QPen(COL_TRIM, 3 if active else 2))
                p.drawLine(QPointF(xx, lane), QPointF(xx, h))
                fw = 28
                fx = xx if which == "in" else xx - fw
                flag = QRectF(fx, h - 24, fw, 16)
                p.fillRect(flag, COL_TRIM)
                p.setPen(Qt.white)
                p.drawText(flag, Qt.AlignCenter, label)
        if self.ref:  # tone reference
            r0, r1 = self._x(self.ref[0]), self._x(self.ref[1])
            p.fillRect(QRectF(r0, h - 24, max(2.0, r1 - r0), 4), COL_REF)
            if r1 - r0 > 90:
                p.setPen(COL_REF)
                p.drawText(QRectF(r0 + 2, h - 40, r1 - r0, 14), Qt.AlignLeft, "Tone reference")
        p.setPen(QPen(COL_HEAD, 2))
        xh = self._x(self.head)
        p.drawLine(QPointF(xh, 0), QPointF(xh, h))
        p.setPen(self.palette().text().color())
        p.drawText(QRectF(34, h - 18, 200, 16), Qt.AlignLeft, fmt(a))
        p.drawText(QRectF(w - 234, h - 18, 200, 16), Qt.AlignRight, fmt(b))
        if self._hover is not None:
            p.drawText(QRectF(self._x(self._hover) + 4, lane + 2, 120, 16), Qt.AlignLeft, fmt(self._hover))

    def _wave_regions(self, x_in: float, x_out: float, w: float):
        """(x0, x1, colour) strips: each section's colour inside In/Out, grey outside."""
        out = [(0.0, max(0.0, x_in), _c(C.faint)), (min(w, x_out), float(w), _c(C.faint))]
        x = max(0.0, x_in)
        for seg in sorted(self.segments, key=lambda q: q.start):
            a, b = max(self._x(seg.start), x_in, 0.0), min(self._x(seg.end), x_out, w)
            if b <= a:
                continue
            if a > x:
                out.append((x, a, COL_WAVE))
            out.append((a, b, WAVE_COL.get(seg.kind, COL_WAVE)))
            x = max(x, b)
        if x < min(w, x_out):
            out.append((x, min(w, x_out), COL_WAVE))
        return [r for r in out if r[1] > r[0]]

    def _columns(self, n: int):
        """Per display column: (hi, lo, rms) in −1..1 display units, or None."""
        a, b = self.view
        edges = a + (b - a) * np.arange(n + 1) / n
        spp = (b - a) * SR / n                       # source samples per column
        if self.audio is not None and spp < 2 * WAVE_HOP:
            # zoomed far in: min/max of the real samples, so the actual wave shape shows
            i0, i1 = max(0, int(a * SR)), min(len(self.audio), int(np.ceil(b * SR)) + 1)
            if i1 - i0 < 2:
                return None
            y = np.asarray(self.audio[i0:i1], np.float32)
            st = np.clip((edges[:-1] * SR).astype(int) - i0, 0, len(y) - 1)
            hi = np.maximum.reduceat(y.max(axis=1), st)
            lo = np.minimum.reduceat(y.min(axis=1), st)
            cnt = np.maximum(np.diff(np.append(st, len(y))), 1)
            rms = np.sqrt(np.add.reduceat((y ** 2).mean(axis=1), st) / cnt)
            return _disp(hi, self._ref), _disp(lo, self._ref), _disp(rms, self._ref)
        wv, hop = self._wave, self._hop
        if wv is None or len(wv) == 0:
            return None
        pos = edges / hop
        if spp >= hop * SR:                          # ≥ 1 bin per column: peak = max, rms = power mean
            j0 = int(np.clip(pos[0], 0, len(wv) - 1))
            j1 = int(np.clip(np.ceil(pos[-1]), j0 + 1, len(wv)))
            sub = wv[j0:j1]
            st = np.clip(pos[:-1].astype(int) - j0, 0, len(sub) - 1)
            pk = np.maximum.reduceat(sub[:, 0], st)
            cnt = np.maximum(np.diff(np.append(st, len(sub))), 1)
            rms = np.sqrt(np.add.reduceat(sub[:, 1] ** 2, st) / cnt)
        else:                                        # finer than the bins: smooth interpolation
            c = (pos[:-1] + pos[1:]) / 2 - 0.5
            k = np.arange(len(wv))
            pk, rms = np.interp(c, k, wv[:, 0]), np.interp(c, k, wv[:, 1])
        d = _disp(pk, self._ref)
        return d, -d, _disp(rms, self._ref)

    def _wave_layer(self, w: int, h: int, mid: float, half: float, x_in: float, x_out: float) -> QPixmap:
        """The waveform as a transparent pixmap, cached until view, size, sections or trim change
        (the playhead repaints many times a second; the waveform should not be redrawn each time)."""
        dpr = max(1.0, self.devicePixelRatioF())
        n = max(2, int(w * dpr))
        segs = tuple((q.start, q.end, q.kind) for q in self.segments)
        key = (self.view, w, h, dpr, round(x_in, 1), round(x_out, 1), segs, id(self._wave), id(self.audio))
        if key == self._cols_key and self._cols is not None:
            return self._cols
        pm = QPixmap(int(w * dpr), int(h * dpr))
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.transparent)
        cols = self._columns(n)
        if cols is not None:
            hi, lo, r = cols
            r_hi = np.minimum(r, np.maximum(hi, 0))
            r_lo = np.maximum(-r, np.minimum(lo, 0))
            xs = (np.arange(n) + 0.5) / dpr
            xs[0], xs[-1] = 0.0, float(w)

            def poly(k0, k1, top, bot):
                x = xs[k0:k1]
                ys = np.concatenate([mid - top[k0:k1] * half, (mid - bot[k0:k1] * half)[::-1]])
                xx = np.concatenate([x, x[::-1]])
                return QPolygonF([QPointF(float(a), float(b)) for a, b in zip(xx, ys, strict=True)])

            p = QPainter(pm)
            p.setRenderHint(QPainter.Antialiasing)
            p.setPen(Qt.NoPen)
            for x0, x1, col in self._wave_regions(x_in, x_out, w):
                # only the columns of this strip (+1 on each side), clipped exactly to the strip
                k0 = max(0, int(np.searchsorted(xs, x0)) - 1)
                k1 = min(n, int(np.searchsorted(xs, x1)) + 1)
                if k1 - k0 < 2:
                    continue
                p.save()
                p.setClipRect(QRectF(x0, 0, x1 - x0, h))
                light = QColor(col)
                light.setAlpha(110)
                p.setBrush(light)
                p.drawPolygon(poly(k0, k1, hi, lo))
                g = QLinearGradient(0, mid - half, 0, mid + half)
                edge, core = QColor(col).lighter(115), QColor(col)
                edge.setAlpha(235)
                core.setAlpha(200)
                g.setColorAt(0.0, edge)
                g.setColorAt(0.5, core)
                g.setColorAt(1.0, edge)
                p.setBrush(QBrush(g))
                p.drawPolygon(poly(k0, k1, r_hi, r_lo))
                p.restore()
            p.end()
        self._cols_key, self._cols = key, pm
        return pm

    # ---------------------------------------------------------------- interaction
    def _boundary_at(self, x: float, tol: float = 5.0) -> int | None:
        """Index i of the boundary between segments i and i+1 within ``tol`` pixels of x."""
        best, dist = None, tol
        for i in range(len(self.segments) - 1):
            d = abs(self._x(self.segments[i].end) - x)
            if d <= dist:
                best, dist = i, d
        return best

    LANE = 18

    def _trim_at(self, x: float, y: float, tol: float = 6.0) -> str | None:
        """"in" / "out" if (x, y) is on a marker; only below the section band (boundaries win there)."""
        if y < self.LANE:
            return None
        t_in, t_out = self.trim or (0.0, self.duration)
        d_in, d_out = abs(self._x(t_in) - x), abs(self._x(t_out) - x)
        if min(d_in, d_out) > tol:
            # the markers sit at the very edge of the view when untrimmed: grab them a bit generously
            if t_in <= 0 and x <= tol:
                return "in"
            if t_out >= self.duration and x >= self.width() - tol:
                return "out"
            return None
        return "in" if d_in <= d_out else "out"

    def mousePressEvent(self, e):
        if self.levels is None or e.button() != Qt.LeftButton:
            return
        which = self._trim_at(e.position().x(), e.position().y())
        if which is not None:  # grab the In or Out marker
            self._trim_drag = which
            self._trim_from = self.trim or (0.0, self.duration)
            self.trim = self._trim_from
            self.update()
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
        if self._trim_drag is not None:
            t_in, t_out = self.trim
            if self._trim_drag == "in":
                t_in = float(np.clip(t, 0.0, t_out - 0.5))
                self._hover = t_in
            else:
                t_out = float(np.clip(t, t_in + 0.5, self.duration))
                self._hover = t_out
            self.trim = (t_in, t_out)
        elif self._bound is not None:
            left, right = self.segments[self._bound], self.segments[self._bound + 1]
            t = float(np.clip(t, left.start + 0.3, right.end - 0.3))
            left.end = right.start = t
            self._hover = t
        elif self._drag_from is not None:
            self.sel = (min(self._drag_from, t), max(self._drag_from, t))
        elif self.levels is not None:
            x, y = e.position().x(), e.position().y()
            near = self._trim_at(x, y) is not None or self._boundary_at(x) is not None
            self.setCursor(Qt.SizeHorCursor if near else Qt.ArrowCursor)
        self.update()

    def mouseReleaseEvent(self, e):
        if self._trim_drag is not None:
            self._trim_drag = None
            if self.trim != self._trim_from:
                self.trim_changed.emit(*self.trim)
            self.update()
            return
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

    # ---------------------------------------------------------------- zoom / pan
    MIN_SPAN_S = 0.25

    def _set_view(self, a: float, b: float) -> None:
        span = min(max(b - a, self.MIN_SPAN_S), self.duration)
        a = float(np.clip(a, 0, self.duration - span))
        self.view = (a, a + span)
        self.update()

    def _zoom(self, factor: float, x: float) -> None:
        """Zoom by ``factor`` (< 1 = in) keeping the time under pixel ``x`` in place."""
        a, b = self.view
        t = self._t(x)
        self._set_view(t - (t - a) * factor, t + (b - t) * factor)

    def _pan(self, dx_px: float) -> None:
        a, b = self.view
        dt = -dx_px / max(1, self.width()) * (b - a)
        self._set_view(a + dt, b + dt)

    def event(self, e):
        # trackpad pinch = zoom around the cursor; two-finger double tap = whole file
        if e.type() == QEvent.NativeGesture and self.levels is not None:
            if e.gestureType() == Qt.ZoomNativeGesture:
                self._zoom(1.0 / max(0.2, 1.0 + e.value()), e.position().x())
                return True
            if e.gestureType() == Qt.SmartZoomNativeGesture:
                self._zoom_all()
                return True
        return super().event(e)

    def wheelEvent(self, e):
        """Trackpad: two-finger swipe sideways = pan, up/down = smooth zoom. The first movement of a
        gesture decides which (axis lock until the gesture ends), so a slightly diagonal swipe does
        not do both. Mouse wheel: zoom one step per notch, Shift + wheel = pan."""
        if self.levels is None:
            return
        x = e.position().x()
        pd = e.pixelDelta()
        trackpad = not pd.isNull() or e.phase() != Qt.NoScrollPhase
        now = time.monotonic()
        if e.phase() == Qt.ScrollBegin or now - self._last_wheel > 0.3:
            self._axis, self._axis_acc = None, (0.0, 0.0)
        self._last_wheel = now
        if trackpad:
            dx, dy = (pd.x(), pd.y()) if not pd.isNull() else (e.angleDelta().x() / 8, e.angleDelta().y() / 8)
            if self._axis is None:
                ax, ay = self._axis_acc[0] + abs(dx), self._axis_acc[1] + abs(dy)
                self._axis_acc = (ax, ay)
                if ax + ay < 6:            # not enough movement yet to tell the direction
                    return
                self._axis = "pan" if ax > ay else "zoom"
            if self._axis == "pan":
                self._pan(dx)
            else:
                self._zoom(float(np.exp(-dy * 0.006)), x)
            if e.phase() == Qt.ScrollEnd:
                self._axis = None
            return
        dy, dx = e.angleDelta().y(), e.angleDelta().x()
        if e.modifiers() & Qt.ShiftModifier or (dx and not dy):
            self._pan((dx or dy) / 120 * self.width() * 0.1)
        else:
            self._zoom(0.8 ** (dy / 120), x)

    def contextMenuEvent(self, e):
        if self.levels is None:
            return
        t_mouse = self._t(e.pos().x())
        s0, s1 = self.sel
        if s1 - s0 < 0.2:  # no selection: act on the segment under the mouse
            t = self._t(e.pos().x())
            seg = next((s for s in self.segments if s.start <= t < s.end), None)
            if seg is None:
                return
            s0, s1 = seg.start, seg.end
        m = QMenu(self)
        for label, kind in (("Treat as speech", SPEECH), ("Treat as music", MUSIC),
                            ("Treat as speech + music", MIX)):
            act = QAction(f"{label}  ({fmt(s0)} – {fmt(s1)})", m)
            act.triggered.connect(lambda _=False, k=kind: self.relabel.emit(s0, s1, k))
            m.addAction(act)
        m.addSeparator()
        act = QAction(f"Use as tone reference for the voice  ({fmt(s0)} – {fmt(s1)})", m)
        act.triggered.connect(lambda: self.tone_ref.emit(s0, s1))
        m.addAction(act)
        if self.ref:
            act = QAction("Tone reference: automatic (driest parts)", m)
            act.triggered.connect(lambda: self.tone_ref.emit(-1.0, -1.0))
            m.addAction(act)
        m.addSeparator()
        cur0, cur1 = self.trim or (0.0, self.duration)
        for label, rng in ((f"Set In here  ({fmt(t_mouse)})", (t_mouse, cur1)),
                           (f"Set Out here  ({fmt(t_mouse)})", (cur0, t_mouse))):
            act = QAction(label, m)
            act.setEnabled(rng[1] - rng[0] >= 0.5)
            act.triggered.connect(lambda _=False, r=rng: self.trim_changed.emit(*r))
            m.addAction(act)
        if self.sel[1] - self.sel[0] >= 0.5:
            a, b = self.sel
            act = QAction(f"Trim to selection  ({fmt(a)} – {fmt(b)})", m)
            act.triggered.connect(lambda: self.trim_changed.emit(a, b))
            m.addAction(act)
        if self.trim:
            act = QAction("Remove trim", m)
            act.triggered.connect(lambda: self.trim_changed.emit(-1.0, -1.0))
            m.addAction(act)
        m.addSeparator()
        zoom = QAction("Show whole file", m)
        zoom.triggered.connect(self._zoom_all)
        m.addAction(zoom)
        m.exec(e.globalPos())

    def _zoom_all(self):
        self.view = (0.0, max(self.duration, 1.0))
        self.update()
