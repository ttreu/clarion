"""Stereo level meter: peak + RMS bars, peak hold and a clip lamp (click to reset)."""
from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget

from .style import C

FLOOR_DB = -60.0
TICKS = (-60, -40, -30, -20, -12, -6, -3, 0)


def to_db(v: float) -> float:
    return 20 * np.log10(max(v, 1e-6))


class LevelMeter(QWidget):
    HOLD_S = 1.5
    FALL_DB_S = 24.0         # how fast the bars fall back (dB per second)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(46)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setToolTip("Playback level (peak light, RMS solid). Red lamp = clipping "
                        "(≥ −0.1 dBFS), click to reset.")
        self.peak = [FLOOR_DB, FLOOR_DB]
        self.rms = [FLOOR_DB, FLOOR_DB]
        self.hold = [FLOOR_DB, FLOOR_DB]
        self.hold_t = [0.0, 0.0]
        self.clip = False
        self._last = time.monotonic()
        self._timer = QTimer(self, interval=33, timeout=self._fall)

    # ------------------------------------------------------------------ feeding
    def set_levels(self, peak: tuple[float, float], rms: tuple[float, float]) -> None:
        """Linear peak and RMS per channel of the audio just played."""
        now = time.monotonic()
        for c in range(2):
            pdb, rdb = max(FLOOR_DB, to_db(peak[c])), max(FLOOR_DB, to_db(rms[c]))
            self.peak[c] = max(pdb, self.peak[c])
            self.rms[c] = max(rdb, self.rms[c])
            if pdb >= self.hold[c] or now - self.hold_t[c] > self.HOLD_S:
                self.hold[c], self.hold_t[c] = pdb, now
            if peak[c] >= 10 ** (-0.1 / 20):
                self.clip = True
        if not self._timer.isActive():
            self._last = now
            self._timer.start()
        self.update()

    def _fall(self):
        now = time.monotonic()
        d = (now - self._last) * self.FALL_DB_S
        self._last = now
        for c in range(2):
            self.peak[c] = max(FLOOR_DB, self.peak[c] - d)
            self.rms[c] = max(FLOOR_DB, self.rms[c] - d)
            if now - self.hold_t[c] > self.HOLD_S:
                self.hold[c] = max(FLOOR_DB, self.hold[c] - d)
        if max(self.peak + self.hold) <= FLOOR_DB:
            self._timer.stop()
        self.update()

    def reset(self):
        self.peak = [FLOOR_DB, FLOOR_DB]
        self.rms = [FLOOR_DB, FLOOR_DB]
        self.hold = [FLOOR_DB, FLOOR_DB]
        self.clip = False
        self.update()

    def mousePressEvent(self, _):
        self.clip = False
        self.update()

    # ------------------------------------------------------------------ painting
    def _x(self, db: float, x0: float, w: float) -> float:
        return x0 + (min(0.0, max(FLOOR_DB, db)) - FLOOR_DB) / -FLOOR_DB * w

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        lamp = 12
        x0, x1 = 16.0, w - lamp - 10.0
        bw = x1 - x0
        bar_h, gap, top = 9.0, 4.0, 4.0
        f = QFont(self.font())
        f.setPointSizeF(8.5)
        p.setFont(f)
        grad = QLinearGradient(x0, 0, x1, 0)
        for db, col in ((FLOOR_DB, C.good), (-18, C.good), (-9, C.warn), (-3, C.warn), (-1, C.bad), (0, C.bad)):
            grad.setColorAt((db - FLOOR_DB) / -FLOOR_DB, QColor(col))
        for c, label in enumerate(("L", "R")):
            y = top + c * (bar_h + gap)
            p.setPen(QColor(C.muted))
            p.drawText(QRectF(0, y - 2, 12, bar_h + 4), Qt.AlignCenter, label)
            p.fillRect(QRectF(x0, y, bw, bar_h), QColor(C.card_hi))
            xp = self._x(self.peak[c], x0, bw)
            xr = self._x(self.rms[c], x0, bw)
            p.setOpacity(0.45)
            p.fillRect(QRectF(x0, y, xp - x0, bar_h), grad)       # peak: light
            p.setOpacity(1.0)
            p.fillRect(QRectF(x0, y, xr - x0, bar_h), grad)       # RMS: solid
            if self.hold[c] > FLOOR_DB:
                xh = self._x(self.hold[c], x0, bw)
                p.fillRect(QRectF(xh - 1, y, 2, bar_h), QColor(C.text))
        # scale
        ys = top + 2 * (bar_h + gap)
        for db in TICKS:
            x = self._x(db, x0, bw)
            p.setPen(QColor(C.faint))
            p.drawLine(int(x), int(ys - 2), int(x), int(ys + 1))
            p.setPen(QColor(C.muted))
            p.drawText(QRectF(x - 14, ys + 1, 28, 12), Qt.AlignCenter, str(db))
        # clip lamp
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(C.bad if self.clip else C.card_hi))
        p.drawEllipse(QRectF(w - lamp - 2, top + (2 * bar_h + gap - lamp) / 2, lamp, lamp))
