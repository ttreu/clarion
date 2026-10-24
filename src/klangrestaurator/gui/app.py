"""Main window."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..analysis import MUSIC
from ..pipeline import Project, default_output_name
from ..settings import PRESETS, SR, Settings
from . import worker
from .timeline import Timeline, fmt

AUDIO_EXT = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".m4a", ".ogg", ".w64"}
PREVIEW_S = 20.0
SKIP_MS = 5000  # transport back/forward step


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Klangrestaurator")
        self.resize(1100, 640)
        self.setAcceptDrops(True)
        self.prefs = QSettings("Klangrestaurator", "Klangrestaurator")
        self.project: Project | None = None
        self._job = None
        self._preview_range: tuple[float, float] | None = None
        self._preview_stale = True

        # --- players for A/B -------------------------------------------------
        self.player: QMediaPlayer | None = None  # created lazily (audio init can be slow)
        self.files_ab: tuple[Path, Path] | None = None

        # --- widgets -----------------------------------------------------------
        root = QWidget()
        self.setCentralWidget(root)
        v = QVBoxLayout(root)

        top = QHBoxLayout()
        self.btn_open = QPushButton("Öffnen…")
        self.btn_open.clicked.connect(self.open_dialog)
        self.lbl_file = QLabel("Keine Datei")
        self.lbl_file.setTextInteractionFlags(Qt.TextSelectableByMouse)
        top.addWidget(self.btn_open)
        top.addWidget(self.lbl_file, 1)
        v.addLayout(top)

        self.timeline = Timeline()
        self.timeline.seek.connect(self._on_seek)
        self.timeline.relabel.connect(self._on_relabel)
        self.timeline.selection_changed.connect(self._on_selection)
        v.addWidget(self.timeline, 1)
        hint = QLabel("Klicken = Position · Ziehen = Bereich wählen · Rechtsklick = Sprache/Musik korrigieren · "
                      "Mausrad = Zoom")
        hint.setStyleSheet("color: gray; font-size: 11px;")
        v.addWidget(hint)

        mid = QHBoxLayout()
        v.addLayout(mid)

        # settings
        g = QGroupBox("Einstellungen")
        f = QFormLayout(g)
        self.cmb_preset = QComboBox()
        self.cmb_preset.addItems(list(PRESETS))
        self.cmb_preset.currentTextChanged.connect(self._apply_preset)
        f.addRow("Voreinstellung", self.cmb_preset)
        self.sl_speech, self.lbl_speech = self._slider(0, 40, "dB")
        f.addRow("Sprache: Rauschreduktion", self._pair(self.sl_speech, self.lbl_speech))
        self.chk_wpe = QCheckBox("Zusätzliche Hallreduktion (langsamer, trockener)")
        self.chk_wpe.toggled.connect(self._mark_stale)
        f.addRow("", self.chk_wpe)
        self.sl_music, self.lbl_music = self._slider(0, 100, "%")
        f.addRow("Musik: Rauschreduktion", self._pair(self.sl_music, self.lbl_music))
        self.cmb_rate = QComboBox()
        self.cmb_rate.addItems(["48 kHz (Video)", "Original-Samplerate"])
        f.addRow("Ausgabe (24 bit WAV)", self.cmb_rate)
        mid.addWidget(g, 3)

        # preview / transport
        t = QGroupBox("Vorhören")
        tv = QVBoxLayout(t)
        self.btn_preview = QPushButton(f"Vorschau berechnen ({int(PREVIEW_S)} s ab Cursor / Auswahl)")
        self.btn_preview.clicked.connect(self.make_preview)
        tv.addWidget(self.btn_preview)
        ab = QHBoxLayout()
        self.rb_a = QRadioButton("A  Original")
        self.rb_b = QRadioButton("B  Restauriert")
        self.rb_b.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.rb_a)
        grp.addButton(self.rb_b)
        self.rb_a.toggled.connect(self._switch_ab)
        ab.addWidget(self.rb_a)
        ab.addWidget(self.rb_b)
        tv.addLayout(ab)
        tv.addLayout(self._transport())
        self.lbl_prev = QLabel("")
        self.lbl_prev.setStyleSheet("color: gray;")
        tv.addWidget(self.lbl_prev)
        tv.addStretch(1)
        mid.addWidget(t, 2)

        bottom = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setFormat("")
        self.btn_cancel = QPushButton("Abbrechen")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_job)
        self.btn_export = QPushButton("Exportieren…")
        self.btn_export.setDefault(True)
        self.btn_export.clicked.connect(self.export_dialog)
        bottom.addWidget(self.progress, 1)
        bottom.addWidget(self.btn_cancel)
        bottom.addWidget(self.btn_export)
        v.addLayout(bottom)

        # menu
        m = self.menuBar().addMenu("Datei")
        a = QAction("Öffnen…", self, shortcut=QKeySequence.Open, triggered=self.open_dialog)
        m.addAction(a)
        a = QAction("Exportieren…", self, shortcut=QKeySequence("Ctrl+E"), triggered=self.export_dialog)
        m.addAction(a)
        a = QAction("Erkennung neu berechnen", self, triggered=self.reclassify)
        m.addAction(a)
        for name, key, fn in (("Abspielen/Pause", Qt.Key_Space, self.toggle_play),
                              ("Stopp", Qt.Key_Period, self.stop),
                              ("Zurück", Qt.Key_Left, lambda: self.skip(-SKIP_MS)),
                              ("Vor", Qt.Key_Right, lambda: self.skip(SKIP_MS)),
                              ("Zum Anfang", Qt.Key_Home, self.to_start),
                              ("Schleife", Qt.Key_L, self.btn_loop.toggle)):
            self.addAction(QAction(name, self, shortcut=QKeySequence(key), triggered=fn))
        ab_toggle = QAction("A/B", self, shortcut=QKeySequence(Qt.Key_Tab),
                            triggered=lambda: (self.rb_a if self.rb_b.isChecked() else self.rb_b).setChecked(True))
        self.addAction(ab_toggle)
        hm = self.menuBar().addMenu("Hilfe")
        hm.addAction(QAction("Über Klangrestaurator", self, triggered=self.about))

        self._apply_preset(self.prefs.value("preset", "Natürlich"))
        self.cmb_preset.setCurrentText(self.prefs.value("preset", "Natürlich"))
        self._set_busy(False)
        self.statusBar().showMessage("Bereit")

    # ------------------------------------------------------------------ helpers
    def _transport(self) -> QHBoxLayout:
        row = QHBoxLayout()
        icon = self.style().standardIcon
        sec = SKIP_MS // 1000

        def btn(sp, tip, fn):
            b = QToolButton()
            b.setIcon(icon(sp))
            b.setToolTip(tip)
            b.setAutoRaise(False)
            b.clicked.connect(fn)
            row.addWidget(b)
            return b

        self.btn_start = btn(QStyle.SP_MediaSkipBackward, "Zum Anfang (Pos1)", self.to_start)
        self.btn_back = btn(QStyle.SP_MediaSeekBackward, f"{sec} s zurück (←)", lambda: self.skip(-SKIP_MS))
        self.btn_play = btn(QStyle.SP_MediaPlay, "Abspielen / Pause (Leertaste)", self.toggle_play)
        self.btn_stop = btn(QStyle.SP_MediaStop, "Stopp (.)", self.stop)
        self.btn_fwd = btn(QStyle.SP_MediaSeekForward, f"{sec} s vor (→)", lambda: self.skip(SKIP_MS))
        self.btn_loop = QToolButton()
        self.btn_loop.setText("↻")
        self.btn_loop.setToolTip("Schleife: Vorschau endlos wiederholen (L)")
        self.btn_loop.setCheckable(True)
        self.btn_loop.setChecked(self.prefs.value("loop", False, type=bool))
        self.btn_loop.toggled.connect(self._set_loop)
        row.addSpacing(8)
        row.addWidget(self.btn_loop)
        self.transport = (self.btn_start, self.btn_back, self.btn_play, self.btn_stop, self.btn_fwd, self.btn_loop)
        self.lbl_time = QLabel("–:–– / –:––")
        self.lbl_time.setStyleSheet("font-family: Menlo, monospace;")
        row.addSpacing(8)
        row.addWidget(self.lbl_time)
        row.addStretch(1)
        return row

    def _slider(self, lo, hi, unit):
        s = QSlider(Qt.Horizontal)
        s.setRange(lo, hi)
        lab = QLabel()
        lab.setMinimumWidth(48)
        s.valueChanged.connect(lambda val: lab.setText(f"{val} {unit}"))
        s.valueChanged.connect(self._mark_stale)
        return s, lab

    @staticmethod
    def _pair(a, b):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(a, 1)
        h.addWidget(b)
        return w

    def settings(self) -> Settings:
        base = PRESETS.get(self.cmb_preset.currentText(), Settings())
        return base.with_(speech_reduction_db=float(self.sl_speech.value()),
                          speech_dereverb=self.chk_wpe.isChecked(),
                          music_reduction=self.sl_music.value() / 100)

    def _apply_preset(self, name: str):
        st = PRESETS.get(name)
        if not st:
            return
        self.sl_speech.setValue(int(st.speech_reduction_db))
        self.sl_music.setValue(int(round(st.music_reduction * 100)))
        self.chk_wpe.setChecked(st.speech_dereverb)
        self.prefs.setValue("preset", name)
        self._mark_stale()

    def _mark_stale(self, *_):
        self._preview_stale = True
        if self.files_ab:
            self.lbl_prev.setText("Einstellungen geändert – Vorschau neu berechnen")

    def _set_busy(self, busy: bool):
        has = self.project is not None and self.project.analysed
        for wdg in (self.btn_preview, self.btn_export):
            wdg.setEnabled(has and not busy)
        self.btn_open.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        for b in self.transport:
            b.setEnabled(self.files_ab is not None and not busy)

    def _progress(self, stage: str, v: float):
        self.progress.setValue(int(v * 1000))
        self.progress.setMaximum(1000)
        self.progress.setFormat(f"{stage}  %p%")

    def _run(self, fn, on_done, label: str):
        th, job = worker.start(self, fn)
        for sig in (job.done, job.failed, job.cancelled):
            sig.connect(self._clear_job)
        job.progress.connect(self._progress)
        job.done.connect(on_done)
        job.failed.connect(self._failed)
        job.cancelled.connect(lambda: (self.statusBar().showMessage("Abgebrochen"), self._set_busy(False)))
        self._job = job
        self._set_busy(True)
        self.statusBar().showMessage(label)
        th.start()

    def _clear_job(self, *_):
        self._job = None
        self.progress.reset()
        self.progress.setFormat("")

    def cancel_job(self):
        if self._job:
            self._job.cancel()
            self.statusBar().showMessage("Wird abgebrochen…")

    def _failed(self, msg: str):
        self._set_busy(False)
        self.progress.setFormat("Fehler")
        QMessageBox.critical(self, "Fehler", msg)

    # ------------------------------------------------------------------ file handling
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            p = Path(u.toLocalFile())
            if p.suffix.lower() in AUDIO_EXT:
                self.load(p)
                return

    def open_dialog(self):
        start = self.prefs.value("last_dir", str(Path.home()))
        fn, _ = QFileDialog.getOpenFileName(self, "Audiodatei öffnen", start,
                                            "Audio (*.wav *.aif *.aiff *.flac *.mp3 *.m4a *.ogg *.w64)")
        if fn:
            self.load(Path(fn))

    def load(self, path: Path):
        self._stop()
        self.files_ab = None
        self.prefs.setValue("last_dir", str(path.parent))
        self.project = Project(path)
        self.lbl_file.setText(str(path))
        st = self.settings()
        proj = self.project
        self._run(lambda progress, cancel: proj.prepare(st, progress=progress, cancel=cancel),
                  self._analysed, "Analysiere… (beim ersten Öffnen einer Datei dauert das einige Minuten)")

    def reclassify(self):
        if not self.project:
            return
        st, proj = self.settings(), self.project
        self._run(lambda progress, cancel: proj.prepare(st, progress=progress, cancel=cancel, reclassify=True),
                  self._analysed, "Erkenne Sprache/Musik neu…")

    def _analysed(self, _):
        p = self.project
        self.timeline.set_data(p.levels, p.segments, p.duration)
        n_music = sum(1 for s in p.segments if s.kind == MUSIC)
        t_music = sum(s.duration for s in p.segments if s.kind == MUSIC)
        self.statusBar().showMessage(f"{fmt(p.duration)} min · {n_music} Musikpassagen ({fmt(t_music)}) erkannt")
        self._set_busy(False)

    # ------------------------------------------------------------------ segments
    def _on_relabel(self, a: float, b: float, kind: str):
        if not self.project:
            return
        self.project.set_kind(a, b, kind)
        self.timeline.set_segments(self.project.segments)
        self._mark_stale()

    def _on_selection(self, a: float, b: float):
        if b > a:
            self.statusBar().showMessage(f"Auswahl {fmt(a)} – {fmt(b)} ({b - a:.1f} s)")

    def _on_seek(self, t: float):
        r = self._preview_range
        if self.player and self.files_ab and r and r[0] <= t < r[1]:
            self.player.setPosition(int((t - self._preview_range[0]) * 1000))

    # ------------------------------------------------------------------ preview
    def make_preview(self):
        if not self.project:
            return
        a, b = self.timeline.sel
        if b - a < 1:
            a = self.timeline.head
            b = a + PREVIEW_S
        b = min(b, a + 120, self.project.duration)  # keep previews short, stay inside the file
        st, proj = self.settings(), self.project
        self._stop()

        def job(progress, cancel):
            progress("Vorschau berechnen", 0.2)
            r = proj.write_preview(st, a, b, cancel=cancel)
            progress("Vorschau berechnen", 1)
            return r

        def done(files):
            self.files_ab = files
            self._preview_range = (a, b)
            self._preview_stale = False
            self.lbl_prev.setText(f"Vorschau {fmt(a)} – {fmt(b)} · Leertaste = Play, Tab = A/B")
            self._set_busy(False)
            self.statusBar().showMessage(f"Vorschau bereit ({b - a:.0f} s)")
            self._load_current(0)
            if self.player:
                self.player.play()

        self._run(job, done, "Berechne Vorschau…")

    def _ensure_player(self) -> QMediaPlayer | None:
        if self.player is None and not os.environ.get("KLANGRESTAURATOR_NO_AUDIO"):
            self.out = QAudioOutput(self)
            self.player = QMediaPlayer(self)
            self.player.setAudioOutput(self.out)
            self.player.mediaStatusChanged.connect(self._on_media_status)
            self.player.positionChanged.connect(self._on_play_pos)
            self.player.durationChanged.connect(lambda _: self._update_time())
            self.player.playbackStateChanged.connect(self._on_state)
        return self.player

    def _stop(self):
        if self.player:
            self.player.stop()

    def _load_current(self, pos_ms: int):
        if not self.files_ab or not self._ensure_player():
            return
        f = self.files_ab[0] if self.rb_a.isChecked() else self.files_ab[1]
        self.player.setSource(QUrl.fromLocalFile(str(f)))
        if pos_ms:
            QTimer.singleShot(50, lambda: self.player.setPosition(pos_ms))

    def _switch_ab(self, *_):
        if not self.files_ab or not self.player:
            return
        playing = self.player.playbackState() == QMediaPlayer.PlayingState
        pos = self.player.position()
        self._load_current(pos)
        if playing:
            self.player.play()

    def toggle_play(self):
        if not self.files_ab or not self._ensure_player():
            return
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def stop(self):
        """Stop and return to the start of the preview."""
        if self.player and self.files_ab:
            self.player.stop()
            self.player.setPosition(0)
            self._on_play_pos(0)

    def to_start(self):
        if self.player and self.files_ab:
            self.player.setPosition(0)

    def skip(self, delta_ms: int):
        if not self.player or not self.files_ab:
            return
        dur = self.player.duration()
        pos = max(0, self.player.position() + delta_ms)
        if dur > 0:
            pos = min(pos, dur - 1)
        self.player.setPosition(pos)

    def _set_loop(self, on: bool):
        self.prefs.setValue("loop", on)

    def _on_media_status(self, status):
        # Loop by hand: QMediaPlayer.setLoops() keeps repeating for a while after switching it off.
        if status == QMediaPlayer.EndOfMedia and self.btn_loop.isChecked():
            self.player.setPosition(0)
            self.player.play()

    def _on_state(self, state):
        playing = state == QMediaPlayer.PlayingState
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPause if playing else QStyle.SP_MediaPlay))

    def _update_time(self):
        if not self.player:
            return
        self.lbl_time.setText(f"{fmt(self.player.position() / 1000)} / {fmt(self.player.duration() / 1000)}")

    def _on_play_pos(self, ms: int):
        if self._preview_range:
            self.timeline.set_head(self._preview_range[0] + ms / 1000)
        self._update_time()

    # ------------------------------------------------------------------ export
    def export_dialog(self):
        if not self.project or not self.project.analysed:
            return
        default = default_output_name(self.project.source, self.cmb_preset.currentText())
        fn, _ = QFileDialog.getSaveFileName(self, "Exportieren als", str(default), "WAV 24 bit (*.wav)")
        if not fn:
            return
        st, proj = self.settings(), self.project
        out_sr = SR if self.cmb_rate.currentIndex() == 0 else int(proj.meta.get("sr_in", SR))
        self._stop()
        self._run(lambda progress, cancel: proj.export(st, fn, out_sr=out_sr, progress=progress, cancel=cancel),
                  lambda p: (self._set_busy(False), self.statusBar().showMessage(f"Gespeichert: {p}")),
                  "Exportiere…")

    def about(self):
        QMessageBox.about(self, "Klangrestaurator",
                          f"<b>Klangrestaurator {__version__}</b><p>Sprache: DeepFilterNet3 (KI) + optional WPE.<br>"
                          "Musik: stationäre spektrale Rauschminderung.<br>"
                          "Automatische Sprach-/Musik-Erkennung mit manueller Korrektur.</p>")

    def closeEvent(self, e):
        if self._job:
            self._job.cancel()
        worker.wait_all()
        super().closeEvent(e)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Klangrestaurator")
    app.setOrganizationName("Klangrestaurator")
    w = MainWindow()
    w.show()
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        w.load(Path(sys.argv[1]))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
