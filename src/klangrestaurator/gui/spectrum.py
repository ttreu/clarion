"""Live spectrum of what is playing, in the style of Audition's multiband display: log frequency axis,
dB scale, and the multiband compressor's crossover frequencies as lines.

Cheap: one 4096-point FFT per meter tick (~30 per second, well under 1 ms each), drawn as one path.
"""
from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from .style import C

N_FFT = 4096
F_LO, F_HI = 20.0, 20_000.0
DB_LO, DB_HI = -90.0, 0.0
N_BINS = 160                     # display columns, log spaced (~1/16 octave)
FREQ_TICKS = (30, 40, 50, 60, 100, 200, 300, 500, 1000, 2000, 3000, 4000, 6000, 10_000)
FREQ_LABELS = {100: "100", 1000: "1k", 10_000: "10k", 50: "50", 300: "300", 3000: "3k"}


class Spectrum(QWidget):
    FALL_DB_S = 40.0             # how fast the curve sinks when the sound gets quieter
    HOLD_S = 1.2                 # peak-hold line

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setToolTip("Spectrum of what is playing. Lines: crossover frequencies of the broadcast compressor.")
        self.crossovers: tuple[float, ...] = ()
        self.cross_active = False
        self.edges = np.geomspace(F_LO, F_HI, N_BINS + 1)
        self.level = np.full(N_BINS, DB_LO)
        self.hold = np.full(N_BINS, DB_LO)
        self.hold_t = np.zeros(N_BINS)
        self._win = np.hanning(N_FFT).astype(np.float32)
        self._norm = 2.0 / self._win.sum()          # full-scale sine → 0 dB
        self._sr = None
        self._idx = None
        self._last = time.monotonic()
        self._timer = QTimer(self, interval=33, timeout=self._fall)

    # ------------------------------------------------------------------ feeding
    def set_crossovers(self, freqs, active: bool) -> None:
        self.crossovers, self.cross_active = tuple(freqs), active
        self.update()

    def set_audio(self, x: np.ndarray, sr: int) -> None:
        """``x``: the most recent samples (n, channels) of what is playing."""
        if len(x) < 256:
            return
        mono = x.mean(axis=1) if x.ndim == 2 else x
        mono = mono[-N_FFT:]
        if len(mono) < N_FFT:
            mono = np.concatenate([np.zeros(N_FFT - len(mono), np.float32), mono])
        mag = np.abs(np.fft.rfft(mono * self._win)) * self._norm
        if sr != self._sr:                            # FFT bin → display column
            self._sr = sr
            f = np.fft.rfftfreq(N_FFT, 1 / sr)
            self._idx = np.searchsorted(self.edges, f) - 1
        col = np.full(N_BINS, 0.0)
        k = (self._idx >= 0) & (self._idx < N_BINS)
        np.maximum.at(col, self._idx[k], mag[k])
        # columns narrower than one FFT bin (low end) borrow from the neighbour: interpolate
        empty = col == 0
        if empty.any() and (~empty).any():
            centres = np.sqrt(self.edges[:-1] * self.edges[1:])
            col[empty] = np.interp(np.log(centres[empty]), np.log(centres[~empty]), col[~empty])
        db = 20 * np.log10(np.maximum(col, 1e-9))
        now = time.monotonic()
        self.level = np.maximum(self.level, db)
        up = db >= self.hold
        old = now - self.hold_t > self.HOLD_S
        self.hold = np.where(up | old, db, self.hold)
        self.hold_t = np.where(up | old, now, self.hold_t)
        if not self._timer.isActive():
            self._last = now
            self._timer.start()
        self.update()

    def _fall(self):
        now = time.monotonic()
        d = (now - self._last) * self.FALL_DB_S
        self._last = now
        self.level = np.maximum(DB_LO, self.level - d)
        old = now - self.hold_t > self.HOLD_S
        self.hold = np.where(old, np.maximum(DB_LO, self.hold - d), self.hold)
        if self.level.max() <= DB_LO and self.hold.max() <= DB_LO:
            self._timer.stop()
        self.update()

    def reset(self):
        self.level[:] = DB_LO
        self.hold[:] = DB_LO
        self.update()

    # ------------------------------------------------------------------ painting
    def _geom(self):
        return QRectF(4, 4, self.width() - 34, self.height() - 22)    # plot area (scale right, Hz below)

    def _fx(self, f: float, r: QRectF) -> float:
        return r.left() + np.log(f / F_LO) / np.log(F_HI / F_LO) * r.width()

    def _dy(self, db, r: QRectF):
        return r.top() + (DB_HI - np.clip(db, DB_LO, DB_HI)) / (DB_HI - DB_LO) * r.height()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self._geom()
        p.fillRect(r, QColor("#0f1013"))
        f = QFont(self.font())
        f.setPointSizeF(8.5)
        p.setFont(f)
        # grid
        for db in range(-10, int(DB_LO), -10):
            y = float(self._dy(db, r))
            p.setPen(QPen(QColor(C.line), 1))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            if db % 20 == 0:
                p.setPen(QColor(C.muted))
                p.drawText(QRectF(r.right() + 4, y - 7, 30, 14), Qt.AlignLeft | Qt.AlignVCenter, str(db))
        for fq in FREQ_TICKS:
            x = self._fx(fq, r)
            major = fq in (100, 1000, 10_000)
            p.setPen(QPen(QColor(C.faint if major else C.line), 1))
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            if fq in FREQ_LABELS:
                p.setPen(QColor(C.text if major else C.muted))
                p.drawText(QRectF(x - 16, r.bottom() + 2, 32, 14), Qt.AlignCenter, FREQ_LABELS[fq])
        p.setPen(QColor(C.muted))
        p.drawText(QRectF(r.left(), r.bottom() + 2, 20, 14), Qt.AlignLeft, "Hz")
        p.drawText(QRectF(r.right() + 4, r.top() - 2, 30, 14), Qt.AlignLeft, "dB")

        # spectrum: filled area + outline, peak-hold line
        centres = np.sqrt(self.edges[:-1] * self.edges[1:])
        xs = r.left() + np.log(centres / F_LO) / np.log(F_HI / F_LO) * r.width()
        ys = self._dy(self.level, r)
        if self.level.max() > DB_LO:
            p.save()
            p.setClipRect(r)
            area = QPainterPath(QPointF(xs[0], r.bottom()))
            for x, y in zip(xs, ys, strict=True):
                area.lineTo(float(x), float(y))
            area.lineTo(float(xs[-1]), r.bottom())
            area.closeSubpath()
            g = QLinearGradient(0, r.top(), 0, r.bottom())
            top, bottom = QColor(C.accent), QColor(C.accent)
            top.setAlpha(150)
            bottom.setAlpha(25)
            g.setColorAt(0, top)
            g.setColorAt(1, bottom)
            p.fillPath(area, g)
            line = QPainterPath(QPointF(float(xs[0]), float(ys[0])))
            for x, y in zip(xs[1:], ys[1:], strict=True):
                line.lineTo(float(x), float(y))
            p.setPen(QPen(QColor(C.accent_hi), 1.4))
            p.drawPath(line)
            hy = self._dy(self.hold, r)
            hold = QPainterPath(QPointF(float(xs[0]), float(hy[0])))
            for x, y in zip(xs[1:], hy[1:], strict=True):
                hold.lineTo(float(x), float(y))
            hc = QColor(C.text)
            hc.setAlpha(90)
            p.setPen(QPen(hc, 1))
            p.drawPath(hold)
            p.restore()

        # crossover lines (bright when the broadcast compressor is on)
        for fq in self.crossovers:
            x = self._fx(fq, r)
            col = QColor(C.text if self.cross_active else C.faint)
            col.setAlpha(200 if self.cross_active else 140)
            p.setPen(QPen(col, 1.2, Qt.SolidLine if self.cross_active else Qt.DashLine))
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
        p.setPen(QPen(QColor(C.line), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRect(r)
