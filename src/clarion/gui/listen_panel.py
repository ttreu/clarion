"""The Listen group: preview button, A/B choice, transport with loop, level meter – and the player.

It plays either the original source file (before the first preview: ``orig_only``) or a preview pair
``files_ab`` = (A original, B restored) that covers ``range`` (seconds in the recording). The window
decides *what* to play; this panel handles *how*: loading without audible glitches, A/B switching at
the same position, looping, and feeding the level meter and spectrum.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6.QtCore import QSettings, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QButtonGroup,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QRadioButton,
    QToolButton,
    QVBoxLayout,
)

from .. import compat
from ..settings import SR
from . import style
from .meter import LevelMeter
from .spectrum import N_FFT
from .timeline import fmt

PREVIEW_S = 20.0
SKIP_MS = 5000  # transport back/forward step


class ListenPanel(QGroupBox):
    preview_requested = Signal()
    head = Signal(float)              # play position in the recording (s)
    samples = Signal(object, int)     # recent audio (n, 2) and its rate, for the spectrum
    status = Signal(str)              # message for the status bar

    def __init__(self, prefs: QSettings, parent=None):
        super().__init__("LISTEN", parent)
        self.prefs = prefs
        self.player: QMediaPlayer | None = None    # created lazily (audio init can be slow)
        self.files_ab: tuple[Path, Path | None] | None = None
        self.range: tuple[float, float] | None = None
        self.orig_only = False
        # provided by the window
        self.selection: Callable[[], tuple[float, float]] = lambda: (0.0, 0.0)
        self.decoded_source: Callable[[], np.ndarray | None] = lambda: None
        self.pending_load: tuple[int, bool] | None = None   # (position ms, play) once media is loaded
        self._pending_armed = False  # set once the new source started loading (see _on_media_status)
        self._in_set_source = False
        self._meter_file: Path | None = None
        self._meter_src: tuple[Path, sf.SoundFile] | None = None
        self._loop_seek_until = 0.0

        tv = QVBoxLayout(self)
        tv.setSpacing(10)
        # first step of the listening workflow, so it sits on top and stands out
        self.btn_preview = QPushButton("  Compute preview")
        self.btn_preview.setObjectName("preview")
        self.btn_preview.setIcon(style.icon("headphones", style.C.accent_hi, 20))
        self.btn_preview.setIconSize(QSize(20, 20))
        self.btn_preview.setCursor(Qt.PointingHandCursor)
        self.btn_preview.setToolTip(f"Restores {int(PREVIEW_S)} s from the cursor, or the selection, "
                                    "so you can compare A (original) and B (restored).")
        self.btn_preview.clicked.connect(self.preview_requested)
        tv.addWidget(self.btn_preview)
        self.lbl_prev = QLabel(f"{int(PREVIEW_S)} s from the cursor, or the selection")
        self.lbl_prev.setObjectName("muted")
        self.lbl_prev.setWordWrap(True)
        tv.addWidget(self.lbl_prev)
        tv.addSpacing(4)
        tv.addLayout(self._ab_row())
        tv.addLayout(self._transport())
        self.meter = LevelMeter()
        tv.addWidget(self.meter)
        tv.addStretch(1)
        # meter + spectrum: fed from the file being played, in step with the play position
        self._meter_timer = QTimer(self, interval=30, timeout=self._feed_meter)
        # selection loop: watch the position more often than positionChanged fires
        self._loop_timer = QTimer(self, interval=10, timeout=self._check_loop)

    # ------------------------------------------------------------------ widgets
    def _ab_row(self) -> QHBoxLayout:
        ab = QHBoxLayout()
        self.rb_a = QRadioButton("A  Original")
        self.rb_b = QRadioButton("B  Restored")
        self.rb_b.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.rb_a)
        grp.addButton(self.rb_b)
        self.rb_a.toggled.connect(self._switch_ab)
        for rb in (self.rb_a, self.rb_b):
            rb.setMinimumWidth(rb.sizeHint().width() + 16)
        ab.addWidget(self.rb_a)
        ab.addSpacing(14)
        ab.addWidget(self.rb_b)
        ab.addStretch(1)
        return ab

    def _transport(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)
        sec = SKIP_MS // 1000

        def btn(kind, tip, fn, size=34):
            b = QToolButton()
            b.setIcon(style.icon(kind))
            b.setIconSize(QSize(18, 18))
            b.setFixedSize(size, size)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
            return b

        self.btn_start = btn("start", "To start (Home)", self.to_start)
        self.btn_back = btn("back", f"Back {sec} s (←)", lambda: self.skip(-SKIP_MS))
        self.btn_play = btn("play", "Play / pause (space)", self.toggle_play, size=40)
        self.btn_play.setObjectName("play")
        self.btn_play.setIcon(style.icon("play", "#1b1406"))
        self.btn_stop = btn("stop", "Stop (.)", self.stop)
        self.btn_fwd = btn("fwd", f"Forward {sec} s (→)", lambda: self.skip(SKIP_MS))
        self.btn_loop = QToolButton()
        self.btn_loop.setIcon(style.icon("loop"))
        self.btn_loop.setIconSize(QSize(18, 18))
        self.btn_loop.setFixedHeight(34)
        self.btn_loop.setToolTip("Loop on/off (L) · arrow: repeat the whole preview or only the selection")
        self.btn_loop.setCheckable(True)
        self.btn_loop.setChecked(self.prefs.value("loop", False, type=bool))
        self.btn_loop.toggled.connect(self._set_loop)
        self.btn_loop.setPopupMode(QToolButton.MenuButtonPopup)
        menu = QMenu(self.btn_loop)
        grp = QActionGroup(menu)
        self.act_loop_all = QAction("Repeat whole preview", menu, checkable=True)
        self.act_loop_sel = QAction("Repeat selection only", menu, checkable=True)
        for act in (self.act_loop_all, self.act_loop_sel):
            grp.addAction(act)
            menu.addAction(act)
        (self.act_loop_sel if self.prefs.value("loop_mode", "all") == "sel" else self.act_loop_all).setChecked(True)
        grp.triggered.connect(self._set_loop_mode)
        self.btn_loop.setMenu(menu)
        row.addSpacing(8)
        row.addWidget(self.btn_loop)
        self.transport = (self.btn_start, self.btn_back, self.btn_play, self.btn_stop, self.btn_fwd, self.btn_loop)
        self.lbl_time = QLabel("–:––")
        self.lbl_time.setObjectName("time")
        self.lbl_time.setToolTip("Playback position / length")
        row.addSpacing(12)
        row.addWidget(self.lbl_time)
        row.addStretch(1)
        return row

    def set_busy(self, busy: bool, can_preview: bool, job_is_preview: bool) -> None:
        self.btn_preview.setEnabled(can_preview and not busy)
        for b in self.transport:  # previews (and the original) can be listened to during jobs
            b.setEnabled(self.files_ab is not None and (not busy or job_is_preview or self.orig_only))
        self.rb_b.setEnabled(not self.orig_only)

    # ------------------------------------------------------------------ what to play
    def play_original(self, path: Path, duration: float) -> None:
        """Make the transport play the original file (until a preview exists)."""
        self.orig_only = True
        self.files_ab = (path, None)
        self.range = (0.0, duration)
        self.rb_a.setChecked(True)
        self.lbl_prev.setText("Original · space = play · compute a preview to hear B (restored)")
        self.load_current(0)

    def show_preview(self, files: tuple[Path, Path], rng: tuple[float, float], resume: bool) -> None:
        """Switch to a new preview. With ``resume`` playback goes on at the same moment of the recording."""
        a, b = rng
        playing = self.is_playing()
        pos = self.player.position() if (resume and self.player) else 0
        if resume and self.pending_load:  # previous swap still loading: carry its intent over
            pos, playing = self.pending_load[0], self.pending_load[1] or playing
        if resume and self.range and self.range != rng:
            # new range: keep the same moment in the recording if it is inside, else start at a
            t = self.range[0] + pos / 1000
            pos = int((t - a) * 1000) if a <= t < b else 0
        if self.orig_only:            # first preview: from now on A/B of the preview range
            self.orig_only = False
            pos, playing = 0, playing or not resume
            self.rb_a.blockSignals(True)
            self.rb_b.setChecked(True)
            self.rb_a.blockSignals(False)
        self.files_ab = files
        self.range = rng
        self.lbl_prev.setText(f"Preview {fmt(a)} – {fmt(b)} · space = play, Tab = A/B")
        self.load_current(pos, play=playing or not resume)

    def forget(self) -> None:
        """No file any more (a new one is being opened)."""
        self.stop_playback()
        self.files_ab = None

    def release(self) -> None:
        """Let go of the files (window closing)."""
        if self.player:
            self.player.stop()
            self.player.setSource(QUrl())
        self.files_ab = None

    # ------------------------------------------------------------------ player
    def _ensure_player(self) -> QMediaPlayer | None:
        if self.player is None and not compat.env("NO_AUDIO"):
            self.out = QAudioOutput(self)
            self.player = QMediaPlayer(self)
            self.player.setAudioOutput(self.out)
            self.player.mediaStatusChanged.connect(self._on_media_status)
            self.player.positionChanged.connect(self._on_play_pos)
            self.player.durationChanged.connect(lambda _: self._update_time())
            self.player.playbackStateChanged.connect(self._on_state)
        return self.player

    def is_playing(self) -> bool:
        return bool(self.player) and self.player.playbackState() == QMediaPlayer.PlayingState

    def stop_playback(self) -> None:
        if self.player:
            self.player.stop()

    def load_current(self, pos_ms: int, play: bool = False) -> None:
        """Load the A or B file; seek and (optionally) play once it is loaded, so nothing is heard early."""
        if not self.files_ab or not self._ensure_player():
            return
        f = self.files_ab[0] if (self.rb_a.isChecked() or self.files_ab[1] is None) else self.files_ab[1]
        self.pending_load = (pos_ms, play)
        # Status signals emitted inside setSource() are unreliable: it reports LoadedMedia for the old
        # file first, and when called during a load it reports LoadedMedia for the new one right away.
        # So decide by the status afterwards: still loading -> wait for LoadedMedia, else apply now.
        self._pending_armed = False
        self._in_set_source = True
        try:
            self.player.setSource(QUrl.fromLocalFile(str(f)))
            self._meter_file = Path(f)
        finally:
            self._in_set_source = False
        if self.player.mediaStatus() == QMediaPlayer.LoadingMedia:
            self._pending_armed = True
        elif self.player.mediaStatus() == QMediaPlayer.LoadedMedia:
            self._apply_pending_load()

    def _apply_pending_load(self) -> None:
        if self.pending_load is None:
            return
        pos, play = self.pending_load
        self.pending_load = None
        if pos:
            self.player.setPosition(pos)
        if play:
            self.player.play()

    def _switch_ab(self, *_) -> None:
        if not self.files_ab or not self.player or self.files_ab[1] is None:
            return
        self.load_current(self.player.position(), play=self.is_playing())

    def toggle_ab(self) -> None:
        if self.rb_b.isEnabled():
            (self.rb_a if self.rb_b.isChecked() else self.rb_b).setChecked(True)

    def toggle_play(self) -> None:
        if not self.files_ab or not self._ensure_player():
            return
        if self.is_playing():
            self.player.pause()
        else:
            r = self._loop_region()
            if r and not r[0] <= self.player.position() < r[1]:
                self._loop_seek(r[0])
            self.player.play()

    def stop(self) -> None:
        """Stop and return to the start of the preview."""
        if self.player and self.files_ab:
            self.player.stop()
            self.player.setPosition(0)
            self._on_play_pos(0)

    def to_start(self) -> None:
        if self.player and self.files_ab:
            self.player.setPosition(0)

    def skip(self, delta_ms: int) -> None:
        if not self.player or not self.files_ab:
            return
        dur = self.player.duration()
        pos = max(0, self.player.position() + delta_ms)
        if dur > 0:
            pos = min(pos, dur - 1)
        self.player.setPosition(pos)

    def seek_to(self, t: float) -> None:
        """Jump to second ``t`` of the recording, if it is inside what is loaded."""
        r = self.range
        if self.player and self.files_ab and r and r[0] <= t < r[1]:
            self.player.setPosition(int((t - r[0]) * 1000))

    # ------------------------------------------------------------------ loop
    def _set_loop(self, on: bool) -> None:
        self.prefs.setValue("loop", on)
        if on:
            self.loop_hint()

    def _set_loop_mode(self, act: QAction) -> None:
        self.prefs.setValue("loop_mode", "sel" if act is self.act_loop_sel else "all")
        self.btn_loop.setChecked(True)  # picking a mode means "loop like this"
        self.loop_hint()

    def _loop_region(self) -> tuple[int, int] | None:
        """Selection loop range in ms within the loaded file, or None to loop all of it."""
        if not (self.btn_loop.isChecked() and self.act_loop_sel.isChecked() and self.range):
            return None
        p0, p1 = self.range
        s0, s1 = self.selection()
        a, b = max(s0, p0), min(s1, p1)
        if b - a < 0.2:
            return None
        return int((a - p0) * 1000), int((b - p0) * 1000)

    def loop_hint(self) -> None:
        if not (self.btn_loop.isChecked() and self.act_loop_sel.isChecked()):
            return
        r = self._loop_region()
        if r:
            p0 = self.range[0]
            self.status.emit(f"Looping selection {fmt(p0 + r[0] / 1000)} – {fmt(p0 + r[1] / 1000)}")
        else:
            self.status.emit("Loop: no selection inside the preview – repeating the whole preview")

    def _loop_seek(self, ms: int) -> None:
        self.player.setPosition(ms)
        self._loop_seek_until = time.monotonic() + 0.15  # position() lags behind a seek for a moment

    def _check_loop(self) -> None:
        r = self._loop_region()
        if not r or not self.player or time.monotonic() < self._loop_seek_until:
            return
        if not r[0] <= self.player.position() < r[1]:
            self._loop_seek(r[0])

    # ------------------------------------------------------------------ player events
    def _on_media_status(self, status) -> None:
        if self._in_set_source:
            pass  # see load_current
        elif status == QMediaPlayer.LoadingMedia and self.pending_load:
            self._pending_armed = True
        elif status == QMediaPlayer.LoadedMedia and self._pending_armed:
            self._pending_armed = False
            self._apply_pending_load()
        # Loop by hand: QMediaPlayer.setLoops() keeps repeating for a while after switching it off.
        if status == QMediaPlayer.EndOfMedia and self.btn_loop.isChecked():
            r = self._loop_region()
            self._loop_seek(r[0] if r else 0)
            self.player.play()

    def _on_state(self, state) -> None:
        playing = state == QMediaPlayer.PlayingState
        for timer in (self._loop_timer, self._meter_timer):
            timer.start() if playing else timer.stop()
        self.btn_play.setIcon(style.icon("pause" if playing else "play", "#1b1406"))

    def _update_time(self) -> None:
        if self.player:
            self.lbl_time.setText(f"{fmt(self.player.position() / 1000)} / {fmt(self.player.duration() / 1000)}")

    def _on_play_pos(self, ms: int) -> None:
        if self.range:
            self.head.emit(self.range[0] + ms / 1000)
        self._update_time()

    def _feed_meter(self) -> None:
        """Level meter (last 40 ms) and spectrum (last ~85 ms) before the play position, read from the
        file being played."""
        if not self.player or self._meter_file is None:
            return
        pos = self.player.position() / 1000
        f = self._meter_file
        try:
            if self._meter_src is None or self._meter_src[0] != f:
                self._meter_src = (f, sf.SoundFile(str(f)))
            snd = self._meter_src[1]
            sr = snd.samplerate
            end = int(pos * sr)
            snd.seek(max(0, end - N_FFT))
            x = snd.read(min(N_FFT, end) or N_FFT, dtype="float32", always_2d=True)
        except Exception:  # noqa: BLE001 – e.g. m4a/mp3 originals: use the decoded copy if there is one
            src = self.decoded_source()
            if src is None or not self.orig_only:
                return
            sr, end = SR, int(pos * SR)
            x = np.asarray(src[max(0, end - N_FFT):end], dtype=np.float32)
        if len(x) == 0:
            return
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        self.samples.emit(x[:, :2], sr)
        x = x[-int(0.04 * sr):, :2]
        peak = np.abs(x).max(axis=0)
        rms = np.sqrt((x ** 2).mean(axis=0))
        self.meter.set_levels((float(peak[0]), float(peak[1])), (float(rms[0]), float(rms[1])))
