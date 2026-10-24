"""Main window."""
from __future__ import annotations

import os
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
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
    QMenu,
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

from .. import __version__, audio_io, project_file
from ..analysis import MIX, MUSIC
from ..pipeline import Project, default_output_name
from ..settings import MIX_VARIANTS, MUSIC_VARIANTS, PRESETS, SPEECH_VARIANTS, SR, Settings, variant_of
from . import worker
from .timeline import Timeline, fmt

LOUDNESS = [("−14 LUFS (Streaming, laut)", -14.0), ("−16 LUFS (Podcast, Web-Video)", -16.0),
            ("−18 LUFS (Lesung mit Musik)", -18.0), ("−23 LUFS (Rundfunk, EBU R128)", -23.0)]
AUDIO_EXT = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".m4a", ".ogg", ".w64"}
PREVIEW_S = 20.0
PREVIEW_KEEP = 8  # remembered previews for instant switching between settings
SKIP_MS = 5000  # transport back/forward step


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Klangrestaurator")
        self.resize(1100, 640)
        self.setAcceptDrops(True)
        self.prefs = QSettings("Klangrestaurator", "Klangrestaurator")
        self.project: Project | None = None
        self._project_path: Path | None = None   # the .klang file, once saved / opened
        self._dirty = False                       # unsaved decisions (sections, cut, settings …)
        self._pending_project: project_file.Loaded | None = None   # applied when its analysis is done
        self._job = None
        self._preview_range: tuple[float, float] | None = None
        self._preview_stale = True
        # remembered previews: key (settings, range, sections) → (A file, B file); each has its own files,
        # so switching back to a computed state is instant and nothing being played gets overwritten
        self._previews: OrderedDict[tuple, tuple[Path, Path]] = OrderedDict()
        self._slot_seq = 0
        self._clear_preview_files()
        self._job_is_preview = False
        self._refresh_pending = False
        self._next_range: tuple[float, float] | None = None  # preview range wanted by a new selection
        self._pending_load: tuple[int, bool] | None = None  # (position ms, play) once media is loaded
        self._pending_armed = False  # set once the new source started loading (see _on_media_status)
        self._in_set_source = False
        self._auto_timer = QTimer(self, singleShot=True, interval=400, timeout=self._auto_refresh)

        # --- players for A/B -------------------------------------------------
        self.player: QMediaPlayer | None = None  # created lazily (audio init can be slow)
        self.files_ab: tuple[Path, Path] | None = None
        # before the first preview the transport plays the original file itself (files_ab = (source,
        # None), range = whole file): useful for finding cut points and listening before computing
        self._orig_only = False

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
        self.timeline.tone_ref.connect(self._on_tone_ref)
        self.timeline.boundary_moved.connect(self._on_boundary)
        self.timeline.trim_changed.connect(self._on_trim)
        self.timeline.selection_changed.connect(self._on_selection)
        v.addWidget(self.timeline, 1)
        hint = QLabel("Klicken = Position · Ziehen = Bereich wählen · Grenze ziehen = Abschnitt anpassen · "
                      "In/Out-Marker ziehen oder I / O = Anfang / Ende · Rechtsklick = Korrigieren / Zuschneiden · "
                      "Mausrad = Zoom")
        hint.setStyleSheet("color: gray; font-size: 11px;")
        v.addWidget(hint)

        mid = QHBoxLayout()
        v.addLayout(mid)

        # settings, grouped along the signal path: sound restoration (left) | loudness + output (right)
        settings_box = QWidget()
        cols = QHBoxLayout(settings_box)
        cols.setContentsMargins(0, 0, 0, 0)
        left, right = QVBoxLayout(), QVBoxLayout()
        cols.addLayout(left, 3)
        cols.addLayout(right, 2)

        top = QFormLayout()
        self.cmb_preset = QComboBox()
        self.cmb_preset.addItems(list(PRESETS))
        self.cmb_preset.currentTextChanged.connect(self._apply_preset)
        top.addRow("Voreinstellung", self.cmb_preset)
        left.addLayout(top)

        # hidden music options that only the variants set (threshold, second pass, pause gate)
        self._music_extra = {k: getattr(Settings(), k) for k in ("music_nstd", "music_pass2", "music_gate")}
        self._syncing = False

        g_sp = QGroupBox("Klangverbesserung · Sprache")
        f = QFormLayout(g_sp)
        self.cmb_v_speech = self._variant_box(SPEECH_VARIANTS)
        f.addRow("Variante", self.cmb_v_speech)
        self.sl_speech, self.lbl_speech = self._slider(0, 40, "dB")
        f.addRow("Rauschreduktion", self._pair(self.sl_speech, self.lbl_speech))
        self.chk_wpe = QCheckBox("Zusätzliche Hallreduktion (langsamer, trockener)")
        self.chk_wpe.toggled.connect(self._mark_stale)
        f.addRow("", self.chk_wpe)
        self.sl_tone, self.lbl_tone = self._slider(0, 100, "%")
        self.sl_tone.setToolTip("Gleicht die Klangfarbe der Stimme an, wenn sich die Sprechposition ändert "
                                "(z. B. nah am Mikrofon der Beschallung: dröhnend und scharf). Ziel sind die "
                                "trockensten Stellen der Datei oder eine per Rechtsklick gewählte Klangreferenz.")
        f.addRow("Klang angleichen", self._pair(self.sl_tone, self.lbl_tone))
        self.sl_level, self.lbl_level = self._slider(0, 100, "%")
        self.sl_level.setToolTip("Gleicht die Lautstärke der Stimme über die Datei an (gemessen über je 6 s), "
                                 "z. B. nah am Mikrofon vs. normale Sitzposition. Die Musik bleibt unverändert.")
        f.addRow("Lautstärke angleichen", self._pair(self.sl_level, self.lbl_level))
        left.addWidget(g_sp)

        g_mu = QGroupBox("Klangverbesserung · Musik")
        f = QFormLayout(g_mu)
        self.cmb_v_music = self._variant_box(MUSIC_VARIANTS)
        f.addRow("Variante", self.cmb_v_music)
        self.sl_music, self.lbl_music = self._slider(0, 100, "%")
        f.addRow("Rauschreduktion", self._pair(self.sl_music, self.lbl_music))
        self.chk_hiss = QCheckBox("Hochton-Rauschen absenken (ab 8 kHz)")
        self.chk_hiss.setToolTip("Senkt bei Musik alles über 8 kHz sanft ab – dort liegt vor allem Rauschen, "
                                 "Instrumente wie das Cello haben dort kaum Anteile.")
        self.chk_hiss.toggled.connect(self._mark_stale)
        f.addRow("", self.chk_hiss)
        left.addWidget(g_mu)

        g_mix = QGroupBox("Klangverbesserung · Sprache + Musik")
        f = QFormLayout(g_mix)
        self.cmb_v_mix = self._variant_box(MIX_VARIANTS)
        f.addRow("Variante", self.cmb_v_mix)
        self.sl_mix, self.lbl_mix = self._slider(0, 100, "%")
        self.sl_mix.setToolTip("„Sprache + Musik“ = KI-Stimme + dieser Anteil des entrauschten Originals. "
                               "Weniger: trockener, weniger Rauschen, Musik leiser. "
                               "Mehr: mehr Musik, aber auch mehr Raumklang und Rauschen.")
        f.addRow("Musikanteil", self._pair(self.sl_mix, self.lbl_mix))
        left.addWidget(g_mix)
        left.addStretch(1)
        for wdg, sig in ((self.sl_speech, "valueChanged"), (self.chk_wpe, "toggled"), (self.sl_music, "valueChanged"),
                         (self.chk_hiss, "toggled"), (self.sl_mix, "valueChanged")):
            getattr(wdg, sig).connect(self._sync_variants)
        self.cmb_v_speech.activated.connect(lambda i: self._pick_variant(SPEECH_VARIANTS, i))
        self.cmb_v_music.activated.connect(lambda i: self._pick_variant(MUSIC_VARIANTS, i))
        self.cmb_v_mix.activated.connect(lambda i: self._pick_variant(MIX_VARIANTS, i))
        for box in (g_sp, g_mu, g_mix):  # same label width → sliders line up across the groups
            form = box.layout()
            for r in range(form.rowCount()):
                item = form.itemAt(r, QFormLayout.LabelRole)
                if item and item.widget():
                    item.widget().setMinimumWidth(150)

        # loudness: the group's own checkbox switches it on/off (children are disabled when off)
        self.chk_norm = QGroupBox("Lautheit")
        self.chk_norm.setCheckable(True)
        self.chk_norm.setToolTip("Bringt den Export auf eine Ziel-Lautheit (EBU R128 / ITU-R BS.1770), danach ein "
                                 "Limiter bei −1 dBFS. Aus: nur auf −1 dBFS Spitzenpegel normalisiert.")
        self.chk_norm.setChecked(self.prefs.value("normalize", True, type=bool))
        f = QFormLayout(self.chk_norm)
        self.cmb_loud = QComboBox()
        self.cmb_loud.addItems([label for label, _ in LOUDNESS])
        self.cmb_loud.setToolTip("Gilt für Export und Vorschau (dort als Schätzung).")
        self.cmb_loud.setCurrentIndex(min(int(self.prefs.value("loudness_idx", 1)), len(LOUDNESS) - 1))
        self.cmb_loud.currentIndexChanged.connect(lambda i: (self.prefs.setValue("loudness_idx", i),
                                                             self._mark_stale()))
        f.addRow("Ziel", self.cmb_loud)
        self.chk_ceiling = QCheckBox("Laute Musik begrenzen (höchstens 3 LU über dem Ziel)")
        self.chk_ceiling.setToolTip("Besteht eine Aufnahme vor allem aus leiser Sprache, hebt die Lautheits-Anpassung "
                                    "alles stark an – laute Musikstellen landen dann am Limiter und verzerren. "
                                    "Mit dieser Option werden nur diese Stellen sanft zurückgenommen.")
        self.chk_ceiling.setChecked(True)
        self.chk_ceiling.toggled.connect(self._mark_stale)
        f.addRow("", self.chk_ceiling)
        self.chk_norm.toggled.connect(self._toggle_normalize)
        self._toggle_normalize(self.chk_norm.isChecked(), refresh=False)
        right.addWidget(self.chk_norm)

        g_out = QGroupBox("Ausgabe")
        f = QFormLayout(g_out)
        self.cmb_rate = QComboBox()
        self.cmb_rate.addItems(["48 kHz (Video)", "Original-Samplerate"])
        f.addRow("WAV 24 bit", self.cmb_rate)
        right.addWidget(g_out)
        right.addStretch(1)
        mid.addWidget(settings_box, 3)

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
        m.addSeparator()
        a = QAction("Projekt speichern", self, shortcut=QKeySequence.Save, triggered=lambda: self.save_project())
        m.addAction(a)
        a = QAction("Projekt speichern unter…", self, shortcut=QKeySequence("Ctrl+Shift+S"),
                    triggered=lambda: self.save_project(ask=True))
        m.addAction(a)
        m.addSeparator()
        a = QAction("Exportieren…", self, shortcut=QKeySequence("Ctrl+E"), triggered=self.export_dialog)
        m.addAction(a)
        a = QAction("Erkennung neu berechnen", self, triggered=self.reclassify)
        m.addAction(a)
        for name, key, fn in (("Abspielen/Pause", Qt.Key_Space, self.toggle_play),
                              ("Stopp", Qt.Key_Period, self.stop),
                              ("Zurück", Qt.Key_Left, lambda: self.skip(-SKIP_MS)),
                              ("Vor", Qt.Key_Right, lambda: self.skip(SKIP_MS)),
                              ("Zum Anfang", Qt.Key_Home, self.to_start),
                              ("Schleife", Qt.Key_L, self.btn_loop.toggle),
                              ("Anfang hier setzen", Qt.Key_I, lambda: self._trim_at_head(start=True)),
                              ("Ende hier setzen", Qt.Key_O, lambda: self._trim_at_head(start=False))):
            self.addAction(QAction(name, self, shortcut=QKeySequence(key), triggered=fn))
        ab_toggle = QAction("A/B", self, shortcut=QKeySequence(Qt.Key_Tab),
                            triggered=lambda: self.rb_b.isEnabled()
                            and (self.rb_a if self.rb_b.isChecked() else self.rb_b).setChecked(True))
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
        self.btn_loop.setToolTip("Schleife ein/aus (L) · Pfeil: ganze Vorschau oder nur die Auswahl wiederholen")
        self.btn_loop.setCheckable(True)
        self.btn_loop.setChecked(self.prefs.value("loop", False, type=bool))
        self.btn_loop.toggled.connect(self._set_loop)
        self.btn_loop.setPopupMode(QToolButton.MenuButtonPopup)
        menu = QMenu(self.btn_loop)
        grp = QActionGroup(menu)
        self.act_loop_all = QAction("Ganze Vorschau wiederholen", menu, checkable=True)
        self.act_loop_sel = QAction("Nur Auswahl wiederholen", menu, checkable=True)
        for act in (self.act_loop_all, self.act_loop_sel):
            grp.addAction(act)
            menu.addAction(act)
        (self.act_loop_sel if self.prefs.value("loop_mode", "all") == "sel" else self.act_loop_all).setChecked(True)
        grp.triggered.connect(self._set_loop_mode)
        self.btn_loop.setMenu(menu)
        # selection loop: watch the position more often than positionChanged fires
        self._loop_timer = QTimer(self, interval=10, timeout=self._check_loop)
        self._loop_seek_until = 0.0
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

    # ------------------------------------------------------------------ variants
    @staticmethod
    def _variant_box(table) -> QComboBox:
        box = QComboBox()
        box.addItems([f"{letter} – {label}" for letter, label, _ in table] + ["Benutzerdefiniert"])
        box.setToolTip("Varianten aus den Hörvergleichen – gilt für das ganze Projekt. Die Regler darunter "
                       "zeigen, was die Variante einstellt; wer sie verändert, landet bei „Benutzerdefiniert“.")
        return box

    def _apply_fields(self, f: dict) -> None:
        """Put Settings fields into the widgets (only the ones present in ``f``)."""
        if "speech_reduction_db" in f:
            self.sl_speech.setValue(int(round(f["speech_reduction_db"])))
        if "speech_dereverb" in f:
            self.chk_wpe.setChecked(bool(f["speech_dereverb"]))
        if "music_reduction" in f:
            self.sl_music.setValue(int(round(f["music_reduction"] * 100)))
        if "music_hiss_db" in f:
            self.chk_hiss.setChecked(f["music_hiss_db"] < 0)
        for k in self._music_extra:
            if k in f:
                self._music_extra[k] = f[k]
        if "mix_music" in f:
            self.sl_mix.setValue(int(round(f["mix_music"] * 100)))

    def _pick_variant(self, table, i: int) -> None:
        if i >= len(table):       # "Benutzerdefiniert": nothing to apply
            return
        self._apply_fields(table[i][2])
        self._sync_variants()
        self._mark_stale()

    def _sync_variants(self, *_):
        """Show which variant the current settings are (or "Benutzerdefiniert")."""
        if self._syncing:
            return
        self._syncing = True
        st = self.settings()
        for box, table in ((self.cmb_v_speech, SPEECH_VARIANTS), (self.cmb_v_music, MUSIC_VARIANTS),
                           (self.cmb_v_mix, MIX_VARIANTS)):
            letter = variant_of(st, table)
            idx = next((i for i, (lt, _, _) in enumerate(table) if lt == letter), len(table))
            box.blockSignals(True)
            box.setCurrentIndex(idx)
            box.blockSignals(False)
        self._syncing = False

    def settings(self) -> Settings:
        base = PRESETS.get(self.cmb_preset.currentText(), Settings())
        return base.with_(speech_reduction_db=float(self.sl_speech.value()),
                          speech_dereverb=self.chk_wpe.isChecked(),
                          tone_match=self.sl_tone.value() / 100,
                          level_match=self.sl_level.value() / 100,
                          normalize=self.chk_norm.isChecked(),
                          loudness_target=LOUDNESS[self.cmb_loud.currentIndex()][1],
                          music_ceiling_lu=3.0 if self.chk_ceiling.isChecked() else None,
                          music_reduction=self.sl_music.value() / 100,
                          music_hiss_db=-6.0 if self.chk_hiss.isChecked() else 0.0,
                          **self._music_extra,
                          mix_music=self.sl_mix.value() / 100)

    def _apply_preset(self, name: str):
        st = PRESETS.get(name)
        if not st:
            return
        self._apply_fields({k: getattr(st, k) for k in ("speech_reduction_db", "speech_dereverb", "music_reduction",
                                                         "music_hiss_db", "mix_music", *self._music_extra)})
        self._sync_variants()
        self.sl_tone.setValue(int(round(st.tone_match * 100)))
        self.sl_level.setValue(int(round(st.level_match * 100)))
        self.prefs.setValue("preset", name)
        self._mark_stale()

    def _toggle_normalize(self, on: bool, refresh: bool = True):
        self.prefs.setValue("normalize", on)
        self.cmb_loud.setEnabled(on)
        if hasattr(self, "chk_ceiling"):
            self.chk_ceiling.setEnabled(on)
        if refresh:
            self._mark_stale()
            self._auto_timer.stop()   # a switch, not a slider: no need to wait for more changes
            QTimer.singleShot(20, self._auto_refresh)

    def _mark_stale(self, *_):
        self._preview_stale = True
        if self.project is not None and not self._syncing:
            self._set_dirty()
        if self._preview_range and not self._orig_only:
            self.lbl_prev.setText("Einstellungen geändert – Vorschau wird neu berechnet…")
            self._auto_timer.start()  # debounce: restarts while a slider is being dragged

    def _auto_refresh(self):
        """Recompute the preview (new settings and/or new selection), keeping playback going."""
        if not (self._preview_stale and self._preview_range and self.project) or self._orig_only:
            return
        if self._job:
            self._refresh_pending = True  # rerun when the current job has ended
            if self._job_is_preview:
                self._job.cancel()
            return
        rng, self._next_range = self._next_range or self._preview_range, None
        self.make_preview(rng, resume=True)

    def _set_busy(self, busy: bool):
        has = self.project is not None and self.project.analysed
        for wdg in (self.btn_preview, self.btn_export):
            wdg.setEnabled(has and not busy)
        self.btn_open.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        for b in self.transport:  # previews (and the original) can be listened to during jobs
            b.setEnabled(self.files_ab is not None and (not busy or self._job_is_preview or self._orig_only))
        self.rb_b.setEnabled(not self._orig_only)

    def _progress(self, stage: str, v: float):
        self.progress.setValue(int(v * 1000))
        self.progress.setMaximum(1000)
        self.progress.setFormat(f"{stage}  %p%")

    def _run(self, fn, on_done, label: str, preview: bool = False):
        self._job_is_preview = preview
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
        self._job_is_preview = False
        self.progress.reset()
        self.progress.setFormat("")
        if self._refresh_pending:
            self._refresh_pending = False
            QTimer.singleShot(0, self._auto_refresh)

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
            if p.suffix.lower() == project_file.SUFFIX:
                self.open_project(p)
                return
            if p.suffix.lower() in AUDIO_EXT:
                self.open_audio(p)
                return

    def open_dialog(self):
        start = self.prefs.value("last_dir", str(Path.home()))
        fn, _ = QFileDialog.getOpenFileName(
            self, "Audiodatei oder Projekt öffnen", start,
            "Audio oder Projekt (*.wav *.aif *.aiff *.flac *.mp3 *.m4a *.ogg *.w64 *.klang);;"
            "Klangrestaurator-Projekt (*.klang);;Audio (*.wav *.aif *.aiff *.flac *.mp3 *.m4a *.ogg *.w64)")
        if not fn:
            return
        if Path(fn).suffix.lower() == project_file.SUFFIX:
            self.open_project(Path(fn))
        else:
            self.open_audio(Path(fn))

    # ------------------------------------------------------------------ project files
    def open_audio(self, path: Path):
        if not self._confirm_discard():
            return
        self._pending_project = None
        self._project_path = None
        self.load(path)
        self._set_dirty(False)

    def open_project(self, path: Path):
        if not self._confirm_discard():
            return
        try:
            lp = project_file.load(path)
        except (OSError, ValueError, KeyError) as err:
            QMessageBox.critical(self, "Projekt öffnen", f"Das Projekt kann nicht geöffnet werden:\n{err}")
            return
        src = lp.source
        if src is None:   # audio moved: let the user find it
            QMessageBox.information(self, "Projekt öffnen", f"Die Audiodatei wurde nicht gefunden:\n{lp.source_abs}"
                                    "\n\nBitte die Datei suchen.")
            fn, _ = QFileDialog.getOpenFileName(self, "Audiodatei suchen", str(path.parent),
                                                "Audio (*.wav *.aif *.aiff *.flac *.mp3 *.m4a *.ogg *.w64)")
            if not fn:
                return
            src = Path(fn)
        self._pending_project = lp
        self._project_path = lp.path
        self._apply_settings(lp.settings, lp.gui)     # show the project's settings right away
        self.load(src)
        self._set_dirty(False)

    def _apply_project(self, lp: project_file.Loaded):
        """After the analysis: put the project's sections, cut and tone reference in place."""
        p = self.project
        if lp.segments:
            p.segments = lp.segments
            p.save_segments()
        try:
            p.set_trim(lp.trim)
        except ValueError:
            p.set_trim(None)
        p.set_tone_ref(lp.tone_ref)
        self.timeline.set_segments(p.segments)
        self.timeline.trim, self.timeline.ref = p.trim, p.tone_ref
        self.timeline.update()

    def save_project(self, ask: bool = False) -> bool:
        if not self.project:
            return False
        path = self._project_path
        if ask or path is None:
            default = path or self.project.source.with_suffix(project_file.SUFFIX)
            fn, _ = QFileDialog.getSaveFileName(self, "Projekt speichern", str(default),
                                                "Klangrestaurator-Projekt (*.klang)")
            if not fn:
                return False
            path = Path(fn)
        p = self.project
        gui = {"preset": self.cmb_preset.currentText(), "rate_idx": self.cmb_rate.currentIndex(),
               "loudness_idx": self.cmb_loud.currentIndex()}
        self._project_path = project_file.save(path, p.source, p.segments, p.trim, p.tone_ref, self.settings(), gui)
        self._set_dirty(False)
        self.statusBar().showMessage(f"Projekt gespeichert: {self._project_path}")
        return True

    def _apply_settings(self, st: Settings, gui: dict):
        """Put saved settings into all controls."""
        self._syncing = True
        self.cmb_preset.blockSignals(True)
        if gui.get("preset") in PRESETS:
            self.cmb_preset.setCurrentText(gui["preset"])
        self.cmb_preset.blockSignals(False)
        self._apply_fields({k: getattr(st, k) for k in ("speech_reduction_db", "speech_dereverb", "music_reduction",
                                                         "music_hiss_db", "mix_music", *self._music_extra)})
        self.sl_tone.setValue(int(round(st.tone_match * 100)))
        self.sl_level.setValue(int(round(st.level_match * 100)))
        targets = [t for _, t in LOUDNESS]
        if st.loudness_target is not None:
            self.cmb_loud.setCurrentIndex(min(range(len(targets)), key=lambda i: abs(targets[i] - st.loudness_target)))
        self.chk_norm.setChecked(st.normalize and st.loudness_target is not None)
        self.chk_ceiling.setChecked(st.music_ceiling_lu is not None)
        if "rate_idx" in gui:
            self.cmb_rate.setCurrentIndex(int(gui["rate_idx"]))
        self._syncing = False
        self._sync_variants()

    def _set_dirty(self, dirty: bool = True):
        self._dirty = dirty and self.project is not None
        name = (self._project_path.name if self._project_path
                else self.project.source.name if self.project else "")
        self.setWindowTitle("Klangrestaurator" + (f" – {name}" if name else "") + (" •" if self._dirty else ""))
        self.setWindowModified(self._dirty)

    def _confirm_discard(self) -> bool:
        """True if it is fine to drop the current project (saved, discarded, or nothing to lose)."""
        if not self._dirty or os.environ.get("KLANGRESTAURATOR_NO_PROMPT"):
            return True
        r = QMessageBox.question(self, "Ungespeicherte Änderungen",
                                 "Das Projekt hat ungespeicherte Änderungen. Jetzt speichern?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
        if r == QMessageBox.Save:
            return self.save_project()
        return r == QMessageBox.Discard

    def load(self, path: Path):
        self._stop()
        self.files_ab = None
        self._clear_preview_files()
        self.prefs.setValue("last_dir", str(path.parent))
        self.project = Project(path)
        self.lbl_file.setText(str(path))
        self._start_original(path)
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
        self.timeline.ref = p.tone_ref
        self.timeline.trim = p.trim
        if self._pending_project is not None:
            self._apply_project(self._pending_project)
            self._pending_project = None
            self._set_dirty(False)
        n_music = sum(1 for s in p.segments if s.kind == MUSIC)
        t_music = sum(s.duration for s in p.segments if s.kind == MUSIC)
        n_mix = sum(1 for s in p.segments if s.kind == MIX)
        t_mix = sum(s.duration for s in p.segments if s.kind == MIX)
        msg = f"{fmt(p.duration)} min · {n_music} Musikpassagen ({fmt(t_music)})"
        if n_mix:
            msg += f" · {n_mix}× Sprache + Musik ({fmt(t_mix)})"
        self.statusBar().showMessage(msg + " erkannt")
        self._set_busy(False)

    # ------------------------------------------------------------------ segments
    def _trim_at_head(self, start: bool):
        if not self.project or not self.project.analysed:
            return
        t = self.timeline.head
        a, b = self.project.trim or (0.0, self.project.duration)
        self._on_trim(*((t, b) if start else (a, t)))

    def _on_trim(self, a: float, b: float):
        if not self.project:
            return
        if a >= 0 and a <= 0.01 and b >= self.project.duration - 0.01:
            a = b = -1.0                                  # markers back at the edges: no trim
        try:
            self.project.set_trim(None if a < 0 else (a, b))
        except ValueError as err:
            self.statusBar().showMessage(f"Zuschnitt nicht möglich: {err}")
            return
        self.timeline.trim = self.project.trim
        self.timeline.update()
        self._set_dirty()
        tr = self.project.trim
        self.statusBar().showMessage(
            f"Export von {fmt(tr[0])} bis {fmt(tr[1])} ({fmt(tr[1] - tr[0])} min)" if tr else "Zuschnitt aufgehoben")

    def _on_boundary(self, old: float, new: float):
        if not self.project:
            return
        self.project.move_boundary(old, new)
        self._set_dirty()
        self.timeline.set_segments(self.project.segments)
        self.statusBar().showMessage(f"Grenze verschoben: {fmt(old)} → {fmt(new)}")
        self._mark_stale()

    def _on_tone_ref(self, a: float, b: float):
        if not self.project:
            return
        rng = None if a < 0 else (a, b)
        self.project.set_tone_ref(rng)
        self._set_dirty()
        self.timeline.ref = rng
        self.timeline.update()
        self.statusBar().showMessage("Klangreferenz: " + (f"{fmt(a)} – {fmt(b)}" if rng else "automatisch"))
        self._mark_stale()

    def _on_relabel(self, a: float, b: float, kind: str):
        if not self.project:
            return
        self.project.set_kind(a, b, kind)
        self._set_dirty()
        self.timeline.set_segments(self.project.segments)
        self._mark_stale()

    def _start_original(self, path: Path):
        """Make the transport play the original file (until a preview exists)."""
        try:
            dur = float(audio_io.probe(path)[2])
        except Exception:  # noqa: BLE001 – unreadable here: the analysis will report it
            return
        self._orig_only = True
        self.files_ab = (path, None)
        self._preview_range = (0.0, dur)
        self._preview_stale = False
        self.rb_a.setChecked(True)
        self.lbl_prev.setText("Original · Leertaste = Play · für B (restauriert) erst Vorschau berechnen")
        self._load_current(0)

    def _on_selection(self, a: float, b: float):
        if b > a:
            self.statusBar().showMessage(f"Auswahl {fmt(a)} – {fmt(b)} ({b - a:.1f} s)")
            self._loop_hint()
        # once a preview exists, a new selection (>= 1 s) becomes the new preview range
        if b - a >= 1 and self._preview_range and self.project and not self._orig_only:
            rng = (a, min(b, a + 120, self.project.duration))
            if rng != self._preview_range:
                self._next_range = rng
                self._preview_stale = True
                self.lbl_prev.setText(f"Auswahl geändert – Vorschau {fmt(rng[0])} – {fmt(rng[1])} wird berechnet…")
                self._auto_timer.start()

    def _on_seek(self, t: float):
        r = self._preview_range
        if self.player and self.files_ab and r and r[0] <= t < r[1]:
            self.player.setPosition(int((t - self._preview_range[0]) * 1000))

    # ------------------------------------------------------------------ preview
    def make_preview(self, rng: tuple[float, float] | None = None, resume: bool = False):
        """Compute a preview for ``rng`` (default: selection, else PREVIEW_S from the cursor).

        With ``resume`` the old preview keeps playing and the new one takes over at the same position.
        """
        if not self.project or self._job:
            return
        self._auto_timer.stop()
        if not resume:
            self._next_range = None  # an explicit request wins over a queued selection change
        if rng:
            a, b = rng
        else:
            a, b = self.timeline.sel
            if b - a < 1:
                a = self.timeline.head
                b = a + PREVIEW_S
            b = min(b, a + 120, self.project.duration)  # keep previews short, stay inside the file
        st, proj = self.settings(), self.project
        key = self._preview_key(st, a, b)
        if not resume:
            self._stop()

        def job(progress, cancel):
            progress("Vorschau berechnen", 0.2)
            r = proj.write_preview(st, a, b, cancel=cancel, slot=slot)
            progress("Vorschau berechnen", 1)
            return r

        def done(files, cached=False):
            if not cached:
                self._remember_preview(key, files)
            playing = bool(self.player) and self.player.playbackState() == QMediaPlayer.PlayingState
            pos = self.player.position() if (resume and self.player) else 0
            if resume and self._pending_load:  # previous swap still loading: carry its intent over
                pos, playing = self._pending_load[0], self._pending_load[1] or playing
            if resume and self._preview_range and self._preview_range != (a, b):
                # new range: keep the same moment in the recording if it is inside, else start at a
                t = self._preview_range[0] + pos / 1000
                pos = int((t - a) * 1000) if a <= t < b else 0
            if self._orig_only:            # first preview: from now on A/B of the preview range
                self._orig_only = False
                pos, playing = 0, playing or not resume
                self.rb_a.blockSignals(True)
                self.rb_b.setChecked(True)
                self.rb_a.blockSignals(False)
            self.files_ab = files
            self._preview_range = (a, b)
            self._preview_stale = False
            self.lbl_prev.setText(f"Vorschau {fmt(a)} – {fmt(b)} · Leertaste = Play, Tab = A/B")
            self._set_busy(False)
            msg = "Vorschau aktualisiert" if resume else f"Vorschau bereit ({b - a:.0f} s)"
            self.statusBar().showMessage(msg + (" · aus dem Zwischenspeicher" if cached else ""))
            self._load_current(pos, play=playing or not resume)

        hit = self._previews.get(key)
        if hit and all(f.exists() for f in hit):  # computed before: switch instantly
            self._previews.move_to_end(key)
            done(hit, cached=True)
            return
        self._slot_seq += 1
        slot = self._slot_seq

        self._run(job, done, "Aktualisiere Vorschau…" if resume else "Berechne Vorschau…", preview=True)

    # ------------------------------------------------------------------ preview memory
    def _preview_key(self, st, a: float, b: float) -> tuple:
        proj = self.project
        segs = tuple((s.start, s.end, s.kind, s.manual) for s in proj.segments)
        return (str(proj.source), st.effective(), round(a, 3), round(b, 3), segs, proj.tone_ref)

    def _remember_preview(self, key: tuple, files: tuple[Path, Path]) -> None:
        self._previews[key] = files
        self._previews.move_to_end(key)
        while len(self._previews) > PREVIEW_KEEP:
            _, old = self._previews.popitem(last=False)
            in_use = set(self.files_ab or ())
            for f in old:
                if f not in in_use:
                    f.unlink(missing_ok=True)

    def _clear_preview_files(self) -> None:
        """Forget all remembered previews and delete their files (also leftovers of earlier sessions)."""
        self._previews.clear()
        d = Path(tempfile.gettempdir()) / "klangrestaurator_preview"
        for f in d.glob("*.wav") if d.exists() else ():
            if f not in set(getattr(self, "files_ab", None) or ()):  # also called from __init__
                f.unlink(missing_ok=True)

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

    def _load_current(self, pos_ms: int, play: bool = False):
        """Load the A or B file; seek and (optionally) play once it is loaded, so nothing is heard early."""
        if not self.files_ab or not self._ensure_player():
            return
        f = self.files_ab[0] if (self.rb_a.isChecked() or self.files_ab[1] is None) else self.files_ab[1]
        self._pending_load = (pos_ms, play)
        # Status signals emitted inside setSource() are unreliable: it reports LoadedMedia for the old
        # file first, and when called during a load it reports LoadedMedia for the new one right away.
        # So decide by the status afterwards: still loading -> wait for LoadedMedia, else apply now.
        self._pending_armed = False
        self._in_set_source = True
        try:
            self.player.setSource(QUrl.fromLocalFile(str(f)))
        finally:
            self._in_set_source = False
        if self.player.mediaStatus() == QMediaPlayer.LoadingMedia:
            self._pending_armed = True
        elif self.player.mediaStatus() == QMediaPlayer.LoadedMedia:
            self._apply_pending_load()

    def _apply_pending_load(self):
        if self._pending_load is None:
            return
        pos, play = self._pending_load
        self._pending_load = None
        if pos:
            self.player.setPosition(pos)
        if play:
            self.player.play()

    def _switch_ab(self, *_):
        if not self.files_ab or not self.player or self.files_ab[1] is None:
            return
        playing = self.player.playbackState() == QMediaPlayer.PlayingState
        self._load_current(self.player.position(), play=playing)

    def toggle_play(self):
        if not self.files_ab or not self._ensure_player():
            return
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            r = self._loop_region()
            if r and not r[0] <= self.player.position() < r[1]:
                self._loop_seek(r[0])
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
        if on:
            self._loop_hint()

    def _set_loop_mode(self, act: QAction):
        self.prefs.setValue("loop_mode", "sel" if act is self.act_loop_sel else "all")
        self.btn_loop.setChecked(True)  # picking a mode means "loop like this"
        self._loop_hint()

    def _loop_region(self) -> tuple[int, int] | None:
        """Selection loop range in ms within the preview file, or None to loop the whole preview."""
        if not (self.btn_loop.isChecked() and self.act_loop_sel.isChecked() and self._preview_range):
            return None
        p0, p1 = self._preview_range
        a, b = max(self.timeline.sel[0], p0), min(self.timeline.sel[1], p1)
        if b - a < 0.2:
            return None
        return int((a - p0) * 1000), int((b - p0) * 1000)

    def _loop_hint(self):
        if not (self.btn_loop.isChecked() and self.act_loop_sel.isChecked()):
            return
        r = self._loop_region()
        if r:
            p0 = self._preview_range[0]
            self.statusBar().showMessage(f"Schleife über Auswahl {fmt(p0 + r[0] / 1000)} – {fmt(p0 + r[1] / 1000)}")
        else:
            self.statusBar().showMessage("Schleife: keine Auswahl innerhalb der Vorschau – "
                                         "ganze Vorschau wird wiederholt")

    def _loop_seek(self, ms: int):
        self.player.setPosition(ms)
        self._loop_seek_until = time.monotonic() + 0.15  # position() lags behind a seek for a moment

    def _check_loop(self):
        r = self._loop_region()
        if not r or not self.player or time.monotonic() < self._loop_seek_until:
            return
        if not r[0] <= self.player.position() < r[1]:
            self._loop_seek(r[0])

    def _on_media_status(self, status):
        if self._in_set_source:
            pass  # see _load_current
        elif status == QMediaPlayer.LoadingMedia and self._pending_load:
            self._pending_armed = True
        elif status == QMediaPlayer.LoadedMedia and self._pending_armed:
            self._pending_armed = False
            self._apply_pending_load()
        # Loop by hand: QMediaPlayer.setLoops() keeps repeating for a while after switching it off.
        if status == QMediaPlayer.EndOfMedia and self.btn_loop.isChecked():
            r = self._loop_region()
            self._loop_seek(r[0] if r else 0)
            self.player.play()

    def _on_state(self, state):
        playing = state == QMediaPlayer.PlayingState
        if playing:
            self._loop_timer.start()
        else:
            self._loop_timer.stop()
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
        if not self._confirm_discard():
            e.ignore()
            return
        if self._job:
            self._job.cancel()
        worker.wait_all()
        if self.player:
            self.player.stop()
            self.player.setSource(QUrl())
        self.files_ab = None
        self._clear_preview_files()
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
