"""Main window."""
from __future__ import annotations

import os
import sys
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
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
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import APP_NAME, __version__, audio_io, multiband, project_file
from ..analysis import MIX, MUSIC
from ..pipeline import Project, default_output_name
from ..settings import (
    MIX_VARIANTS,
    MUSIC_VARIANTS,
    PRESETS,
    SPEECH_VARIANTS,
    SR,
    Settings,
    preset_name,
    variant_of,
)
from . import style, worker
from .meter import LevelMeter
from .spectrum import N_FFT, Spectrum
from .timeline import Timeline, fmt

LOUDNESS = [("−14 LUFS (streaming, loud)", -14.0), ("−16 LUFS (podcast, web video)", -16.0),
            ("−18 LUFS (reading with music)", -18.0), ("−23 LUFS (broadcast, EBU R128)", -23.0)]
AUDIO_EXT = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".m4a", ".ogg", ".w64"}
PREVIEW_S = 20.0
PREVIEW_KEEP = 8  # remembered previews for instant switching between settings
SKIP_MS = 5000  # transport back/forward step


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
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
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # header: name, file, main actions
        header = QFrame()
        header.setObjectName("header")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(18, 10, 14, 10)
        hl.setSpacing(10)
        name = QLabel(APP_NAME)
        name.setObjectName("appname")
        self.lbl_file = QLabel("No file open")
        self.lbl_file.setObjectName("filename")
        self.lbl_file.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.btn_open = QPushButton("Open…")
        self.btn_open.clicked.connect(self.open_dialog)
        self.btn_save = QPushButton("Save")
        self.btn_save.setToolTip("Save project (⌘S): sections, trim, settings")
        self.btn_save.clicked.connect(lambda: self.save_project())
        self.btn_export = QPushButton("Export…")
        self.btn_export.setObjectName("primary")
        self.btn_export.clicked.connect(self.export_dialog)
        hl.addWidget(name)
        hl.addSpacing(8)
        hl.addWidget(self.lbl_file, 1)
        for b in (self.btn_open, self.btn_save, self.btn_export):
            hl.addWidget(b)
        outer.addWidget(header)

        content = QWidget()
        v = QVBoxLayout(content)
        v.setContentsMargins(14, 12, 14, 10)
        v.setSpacing(10)
        outer.addWidget(content, 1)

        self.timeline = Timeline()
        self.timeline.seek.connect(self._on_seek)
        self.timeline.relabel.connect(self._on_relabel)
        self.timeline.tone_ref.connect(self._on_tone_ref)
        self.timeline.boundary_moved.connect(self._on_boundary)
        self.timeline.trim_changed.connect(self._on_trim)
        self.timeline.selection_changed.connect(self._on_selection)
        tcard = QFrame()
        tcard.setObjectName("card")
        tl = QVBoxLayout(tcard)
        tl.setContentsMargins(8, 8, 8, 6)
        tl.setSpacing(4)
        tl.addWidget(self.timeline, 1)
        hint = QLabel("Click = position · Drag = select · Drag a boundary = adjust section · "
                      "Drag In/Out or press I / O = start / end · Right-click = correct / trim · "
                      "Pinch or ↕ = zoom, ↔ = scroll")
        hint.setObjectName("hint")
        tl.addWidget(hint)
        v.addWidget(tcard, 1)

        mid = QHBoxLayout()
        mid.setSpacing(12)
        v.addLayout(mid)

        # settings, grouped along the signal path: sound restoration (left) | loudness + output (right)
        settings_box = QWidget()
        cols = QHBoxLayout(settings_box)
        cols.setContentsMargins(0, 0, 0, 0)
        left, right = QVBoxLayout(), QVBoxLayout()
        cols.addLayout(left, 3)
        cols.addLayout(right, 2)

        top = QFormLayout()
        top.setContentsMargins(4, 4, 4, 0)
        self.cmb_preset = QComboBox()
        self.cmb_preset.addItems(list(PRESETS))
        self.cmb_preset.currentTextChanged.connect(self._apply_preset)
        top.addRow("Preset", self.cmb_preset)
        left.addLayout(top)

        # hidden music options that only the variants set (threshold, second pass, pause gate)
        self._music_extra = {k: getattr(Settings(), k) for k in ("music_nstd", "music_pass2", "music_gate")}
        self._syncing = False

        g_sp = QGroupBox("SPEECH")
        g_sp.setObjectName("speech")   # timeline colour accents (style.py)
        f = QFormLayout(g_sp)
        self.cmb_v_speech = self._variant_box(SPEECH_VARIANTS)
        f.addRow("Variant", self.cmb_v_speech)
        self.sl_speech, self.lbl_speech = self._slider(0, 40, "dB")
        f.addRow("Noise reduction", self._pair(self.sl_speech, self.lbl_speech))
        self.chk_wpe = QCheckBox("Extra dereverberation (slower, drier)")
        self.chk_wpe.toggled.connect(self._mark_stale)
        f.addRow("", self.chk_wpe)
        self.sl_tone, self.lbl_tone = self._slider(0, 100, "%")
        self.sl_tone.setToolTip("Evens out the voice's tone when the speaking position changes "
                                "(e.g. close to the PA microphone: boomy and harsh). Target: the driest parts "
                                "of the file, or a tone reference chosen by right-click.")
        f.addRow("Match tone", self._pair(self.sl_tone, self.lbl_tone))
        self.sl_level, self.lbl_level = self._slider(0, 100, "%")
        self.sl_level.setToolTip("Evens out the voice's level across the file (measured over 6 s), "
                                 "e.g. close to the microphone vs. normal position. Music is left alone.")
        f.addRow("Match level", self._pair(self.sl_level, self.lbl_level))
        left.addWidget(g_sp)

        g_mu = QGroupBox("MUSIC")
        g_mu.setObjectName("music")   # timeline colour accents (style.py)
        f = QFormLayout(g_mu)
        self.cmb_v_music = self._variant_box(MUSIC_VARIANTS)
        f.addRow("Variant", self.cmb_v_music)
        self.sl_music, self.lbl_music = self._slider(0, 100, "%")
        f.addRow("Noise reduction", self._pair(self.sl_music, self.lbl_music))
        self.chk_hiss = QCheckBox("Reduce treble hiss (above 8 kHz)")
        self.chk_hiss.setToolTip("Gently lowers everything above 8 kHz in music – mostly hiss lives there, "
                                 "instruments like the cello have little energy up there.")
        self.chk_hiss.toggled.connect(self._mark_stale)
        f.addRow("", self.chk_hiss)
        left.addWidget(g_mu)

        g_mix = QGroupBox("SPEECH + MUSIC")
        g_mix.setObjectName("mix")   # timeline colour accents (style.py)
        f = QFormLayout(g_mix)
        self.cmb_v_mix = self._variant_box(MIX_VARIANTS)
        f.addRow("Variant", self.cmb_v_mix)
        self.sl_mix, self.lbl_mix = self._slider(0, 100, "%")
        self.sl_mix.setToolTip("Speech + Music = AI voice + this share of the denoised original. "
                               "Less: drier, less noise, quieter music. "
                               "More: more music, but also more room sound and noise.")
        f.addRow("Music share", self._pair(self.sl_mix, self.lbl_mix))
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
        self.chk_norm = QGroupBox("LOUDNESS")
        self.chk_norm.setCheckable(True)
        self.chk_norm.setToolTip("Brings the export to a target loudness (EBU R128 / ITU-R BS.1770), followed by a "
                                 "limiter at −1 dBFS. Off: peak-normalised to −1 dBFS only.")
        self.chk_norm.setChecked(self.prefs.value("normalize", True, type=bool))
        f = QFormLayout(self.chk_norm)
        self.cmb_loud = QComboBox()
        self.cmb_loud.addItems([label for label, _ in LOUDNESS])
        self.cmb_loud.setToolTip("Applies to export and preview (estimated there).")
        self.cmb_loud.setCurrentIndex(min(int(self.prefs.value("loudness_idx", 1)), len(LOUDNESS) - 1))
        self.cmb_loud.currentIndexChanged.connect(lambda i: (self.prefs.setValue("loudness_idx", i),
                                                             self._mark_stale()))
        f.addRow("Target", self.cmb_loud)
        self.chk_ceiling = QCheckBox("Limit loud music (at most 3 LU above target)")
        self.chk_ceiling.setToolTip("If a recording is mostly quiet speech, loudness normalisation raises "
                                    "everything a lot – loud music then hits the limiter and distorts. "
                                    "This option gently turns down just those passages.")
        self.chk_ceiling.setChecked(True)
        self.chk_ceiling.toggled.connect(self._mark_stale)
        f.addRow("", self.chk_ceiling)
        self.chk_bc = QCheckBox("Broadcast compression (multiband)")
        self.chk_bc.setToolTip("Four-band compressor like Adobe Audition's Multiband Compressor, preset "
                               "“Broadcast”: brings speech closer to loud music and makes the sound denser "
                               "and more present. Runs before the loudness target is applied.")
        self.chk_bc.setChecked(self.prefs.value("broadcast", False, type=bool))
        self.chk_bc.toggled.connect(lambda on: (self.prefs.setValue("broadcast", on), self._mark_stale(),
                                                self._sync_spectrum()))
        f.addRow("", self.chk_bc)
        self.spectrum = Spectrum()
        f.addRow(self.spectrum)
        self._sync_spectrum()
        self.chk_norm.toggled.connect(self._toggle_normalize)
        self._toggle_normalize(self.chk_norm.isChecked(), refresh=False)
        right.addWidget(self.chk_norm)

        g_out = QGroupBox("OUTPUT")
        f = QFormLayout(g_out)
        self.cmb_rate = QComboBox()
        self.cmb_rate.addItems(["48 kHz (video)", "Original sample rate"])
        f.addRow("WAV 24 bit", self.cmb_rate)
        right.addWidget(g_out)
        right.addStretch(1)
        mid.addWidget(settings_box, 3)

        # preview / transport
        t = QGroupBox("LISTEN")
        tv = QVBoxLayout(t)
        tv.setSpacing(10)
        # first step of the listening workflow, so it sits on top and stands out
        self.btn_preview = QPushButton("  Compute preview")
        self.btn_preview.setObjectName("preview")
        self.btn_preview.setIcon(style.icon("headphones", style.C.accent_hi, 20))
        self.btn_preview.setIconSize(QSize(20, 20))
        self.btn_preview.setCursor(Qt.PointingHandCursor)
        self.btn_preview.setToolTip(f"Restores {int(PREVIEW_S)} s from the cursor, or the selection, "
                                    "so you can compare A (original) and B (restored).")
        self.btn_preview.clicked.connect(self.make_preview)
        tv.addWidget(self.btn_preview)
        self.lbl_prev = QLabel(f"{int(PREVIEW_S)} s from the cursor, or the selection")
        self.lbl_prev.setObjectName("muted")
        self.lbl_prev.setWordWrap(True)
        tv.addWidget(self.lbl_prev)
        tv.addSpacing(4)
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
        tv.addLayout(ab)
        tv.addLayout(self._transport())
        self.meter = LevelMeter()
        tv.addWidget(self.meter)
        tv.addStretch(1)
        mid.addWidget(t, 2)

        bottom = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setFormat("")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_job)
        bottom.addWidget(self.progress, 1)
        bottom.addWidget(self.btn_cancel)
        v.addLayout(bottom)

        # level meter: fed from the file being played, in step with the play position
        self._meter_src: tuple[Path, object] | None = None
        self._meter_timer = QTimer(self, interval=30, timeout=self._feed_meter)

        # menu
        m = self.menuBar().addMenu("File")
        a = QAction("Open…", self, shortcut=QKeySequence.Open, triggered=self.open_dialog)
        m.addAction(a)
        m.addSeparator()
        a = QAction("Save Project", self, shortcut=QKeySequence.Save, triggered=lambda: self.save_project())
        m.addAction(a)
        a = QAction("Save Project As…", self, shortcut=QKeySequence("Ctrl+Shift+S"),
                    triggered=lambda: self.save_project(ask=True))
        m.addAction(a)
        m.addSeparator()
        a = QAction("Export…", self, shortcut=QKeySequence("Ctrl+E"), triggered=self.export_dialog)
        m.addAction(a)
        a = QAction("Re-detect Speech/Music", self, triggered=self.reclassify)
        m.addAction(a)
        for name, key, fn in (("Play/Pause", Qt.Key_Space, self.toggle_play),
                              ("Stop", Qt.Key_Period, self.stop),
                              ("Back", Qt.Key_Left, lambda: self.skip(-SKIP_MS)),
                              ("Forward", Qt.Key_Right, lambda: self.skip(SKIP_MS)),
                              ("To start", Qt.Key_Home, self.to_start),
                              ("Loop", Qt.Key_L, self.btn_loop.toggle),
                              ("Set In here", Qt.Key_I, lambda: self._trim_at_head(start=True)),
                              ("Set Out here", Qt.Key_O, lambda: self._trim_at_head(start=False))):
            self.addAction(QAction(name, self, shortcut=QKeySequence(key), triggered=fn))
        ab_toggle = QAction("A/B", self, shortcut=QKeySequence(Qt.Key_Tab),
                            triggered=lambda: self.rb_b.isEnabled()
                            and (self.rb_a if self.rb_b.isChecked() else self.rb_b).setChecked(True))
        self.addAction(ab_toggle)
        hm = self.menuBar().addMenu("Help")
        hm.addAction(QAction(f"About {APP_NAME}", self, triggered=self.about))

        start_preset = preset_name(self.prefs.value("preset"))
        self._apply_preset(start_preset)
        self.cmb_preset.setCurrentText(start_preset)
        self._set_busy(False)
        self.statusBar().showMessage("Ready")

    # ------------------------------------------------------------------ helpers
    def _transport(self) -> QHBoxLayout:
        row = QHBoxLayout()
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

        row.setSpacing(6)
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
        # selection loop: watch the position more often than positionChanged fires
        self._loop_timer = QTimer(self, interval=10, timeout=self._check_loop)
        self._loop_seek_until = 0.0
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
        box.addItems([f"{letter} – {label}" for letter, label, _ in table] + ["Custom"])
        box.setToolTip("Variants from the listening tests – for the whole project. The controls below show "
                       "what a variant sets; changing them switches to “Custom”.")
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
                          broadcast=self.chk_bc.isChecked() and self.chk_norm.isChecked(),
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
        for name in ("chk_ceiling", "chk_bc"):
            if hasattr(self, name):
                getattr(self, name).setEnabled(on)
        self._sync_spectrum()
        if refresh:
            self._mark_stale()
            self._auto_timer.stop()   # a switch, not a slider: no need to wait for more changes
            QTimer.singleShot(20, self._auto_refresh)

    def _mark_stale(self, *_):
        self._preview_stale = True
        if self.project is not None and not self._syncing:
            self._set_dirty()
        if self._preview_range and not self._orig_only:
            self.lbl_prev.setText("Settings changed – updating preview…")
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
        job.cancelled.connect(lambda: (self.statusBar().showMessage("Cancelled"), self._set_busy(False)))
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
            self.statusBar().showMessage("Cancelling…")

    def _failed(self, msg: str):
        self._set_busy(False)
        self.progress.setFormat("Error")
        QMessageBox.critical(self, "Error", msg)

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
            self, "Open audio file or project", start,
            "Audio or project (*.wav *.aif *.aiff *.flac *.mp3 *.m4a *.ogg *.w64 *.klang);;"
            f"{APP_NAME} project (*.klang);;Audio (*.wav *.aif *.aiff *.flac *.mp3 *.m4a *.ogg *.w64)")
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
            QMessageBox.critical(self, "Open project", f"The project could not be opened:\n{err}")
            return
        src = lp.source
        if src is None:   # audio moved: let the user find it
            QMessageBox.information(self, "Open project", f"The audio file was not found:\n{lp.source_abs}"
                                    "\n\nPlease locate it.")
            fn, _ = QFileDialog.getOpenFileName(self, "Locate audio file", str(path.parent),
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
            fn, _ = QFileDialog.getSaveFileName(self, "Save project", str(default),
                                                f"{APP_NAME} project (*.klang)")
            if not fn:
                return False
            path = Path(fn)
        p = self.project
        gui = {"preset": self.cmb_preset.currentText(), "rate_idx": self.cmb_rate.currentIndex(),
               "loudness_idx": self.cmb_loud.currentIndex()}
        self._project_path = project_file.save(path, p.source, p.segments, p.trim, p.tone_ref, self.settings(), gui)
        self._set_dirty(False)
        self.statusBar().showMessage(f"Project saved: {self._project_path}")
        return True

    def _apply_settings(self, st: Settings, gui: dict):
        """Put saved settings into all controls."""
        self._syncing = True
        self.cmb_preset.blockSignals(True)
        if gui.get("preset"):
            self.cmb_preset.setCurrentText(preset_name(gui["preset"]))
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
        self.chk_bc.setChecked(st.broadcast)
        if "rate_idx" in gui:
            self.cmb_rate.setCurrentIndex(int(gui["rate_idx"]))
        self._syncing = False
        self._sync_variants()

    def _set_dirty(self, dirty: bool = True):
        self._dirty = dirty and self.project is not None
        name = (self._project_path.name if self._project_path
                else self.project.source.name if self.project else "")
        self.setWindowTitle(APP_NAME + (f" – {name}" if name else "") + (" •" if self._dirty else ""))
        self.setWindowModified(self._dirty)

    def _confirm_discard(self) -> bool:
        """True if it is fine to drop the current project (saved, discarded, or nothing to lose)."""
        if not self._dirty or os.environ.get("KLANGRESTAURATOR_NO_PROMPT"):
            return True
        r = QMessageBox.question(self, "Unsaved changes",
                                 "The project has unsaved changes. Save now?",
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
                  self._analysed, "Analysing… (the first time a file is opened this takes a few minutes)")

    def reclassify(self):
        if not self.project:
            return
        st, proj = self.settings(), self.project
        self._run(lambda progress, cancel: proj.prepare(st, progress=progress, cancel=cancel, reclassify=True),
                  self._analysed, "Detecting speech/music again…")

    def _analysed(self, _):
        p = self.project
        self.timeline.set_data(p.levels, p.segments, p.duration, wave=p.wave, audio=p.orig)
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
        msg = f"{fmt(p.duration)} min · {n_music} music sections ({fmt(t_music)})"
        if n_mix:
            msg += f" · {n_mix}× speech + music ({fmt(t_mix)})"
        self.statusBar().showMessage(msg + " detected")
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
            self.statusBar().showMessage(f"Cannot trim: {err}")
            return
        self.timeline.trim = self.project.trim
        self.timeline.update()
        self._set_dirty()
        tr = self.project.trim
        self.statusBar().showMessage(
            f"Export from {fmt(tr[0])} to {fmt(tr[1])} ({fmt(tr[1] - tr[0])} min)" if tr else "Trim removed")

    def _on_boundary(self, old: float, new: float):
        if not self.project:
            return
        self.project.move_boundary(old, new)
        self._set_dirty()
        self.timeline.set_segments(self.project.segments)
        self.statusBar().showMessage(f"Boundary moved: {fmt(old)} → {fmt(new)}")
        self._mark_stale()

    def _on_tone_ref(self, a: float, b: float):
        if not self.project:
            return
        rng = None if a < 0 else (a, b)
        self.project.set_tone_ref(rng)
        self._set_dirty()
        self.timeline.ref = rng
        self.timeline.update()
        self.statusBar().showMessage("Tone reference: " + (f"{fmt(a)} – {fmt(b)}" if rng else "automatic"))
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
        self.lbl_prev.setText("Original · space = play · compute a preview to hear B (restored)")
        self._load_current(0)

    def _on_selection(self, a: float, b: float):
        if b > a:
            self.statusBar().showMessage(f"Selection {fmt(a)} – {fmt(b)} ({b - a:.1f} s)")
            self._loop_hint()
        # once a preview exists, a new selection (>= 1 s) becomes the new preview range
        if b - a >= 1 and self._preview_range and self.project and not self._orig_only:
            rng = (a, min(b, a + 120, self.project.duration))
            if rng != self._preview_range:
                self._next_range = rng
                self._preview_stale = True
                self.lbl_prev.setText(f"Selection changed – computing preview {fmt(rng[0])} – {fmt(rng[1])}…")
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
            progress("Computing preview", 0.2)
            r = proj.write_preview(st, a, b, cancel=cancel, slot=slot)
            progress("Computing preview", 1)
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
            self.lbl_prev.setText(f"Preview {fmt(a)} – {fmt(b)} · space = play, Tab = A/B")
            self._set_busy(False)
            msg = "Preview updated" if resume else f"Preview ready ({b - a:.0f} s)"
            self.statusBar().showMessage(msg + (" · from memory" if cached else ""))
            self._load_current(pos, play=playing or not resume)

        hit = self._previews.get(key)
        if hit and all(f.exists() for f in hit):  # computed before: switch instantly
            self._previews.move_to_end(key)
            done(hit, cached=True)
            return
        self._slot_seq += 1
        slot = self._slot_seq

        self._run(job, done, "Updating preview…" if resume else "Computing preview…", preview=True)

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
            self._meter_file = Path(f)
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
            self.statusBar().showMessage(f"Looping selection {fmt(p0 + r[0] / 1000)} – {fmt(p0 + r[1] / 1000)}")
        else:
            self.statusBar().showMessage("Loop: no selection inside the preview – "
                                         "repeating the whole preview")

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
            self._meter_timer.start()
        else:
            self._loop_timer.stop()
            self._meter_timer.stop()
        self.btn_play.setIcon(style.icon("pause" if playing else "play", "#1b1406"))

    def _sync_spectrum(self):
        """Crossover lines in the spectrum: bright when the broadcast compressor is in use."""
        if hasattr(self, "spectrum"):
            self.spectrum.set_crossovers(multiband.BROADCAST.crossovers,
                                         self.chk_bc.isChecked() and self.chk_norm.isChecked())

    def _feed_meter(self):
        """Level meter (last 40 ms) and spectrum (last ~85 ms) before the play position, read from the
        file being played."""
        if not self.player or not getattr(self, "_meter_file", None):
            return
        pos = self.player.position() / 1000
        f = self._meter_file
        try:
            if self._meter_src is None or self._meter_src[0] != f:
                import soundfile as sf
                self._meter_src = (f, sf.SoundFile(str(f)))
            snd = self._meter_src[1]
            sr = snd.samplerate
            end = int(pos * sr)
            n = N_FFT
            snd.seek(max(0, end - n))
            x = snd.read(min(n, end) or n, dtype="float32", always_2d=True)
        except Exception:  # noqa: BLE001 – e.g. m4a/mp3 originals: use the decoded copy if there is one
            if not (self.project and self.project.orig is not None and self._orig_only):
                return
            sr, end = SR, int(pos * SR)
            x = np.asarray(self.project.orig[max(0, end - N_FFT):end], dtype=np.float32)
        if len(x) == 0:
            return
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        self.spectrum.set_audio(x[:, :2], sr)
        x = x[-int(0.04 * sr):]
        peak = np.abs(x[:, :2]).max(axis=0)
        rms = np.sqrt((x[:, :2] ** 2).mean(axis=0))
        self.meter.set_levels((float(peak[0]), float(peak[1])), (float(rms[0]), float(rms[1])))

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
        fn, _ = QFileDialog.getSaveFileName(self, "Export as", str(default), "WAV 24 bit (*.wav)")
        if not fn:
            return
        st, proj = self.settings(), self.project
        out_sr = SR if self.cmb_rate.currentIndex() == 0 else int(proj.meta.get("sr_in", SR))
        self._stop()
        self._run(lambda progress, cancel: proj.export(st, fn, out_sr=out_sr, progress=progress, cancel=cancel),
                  lambda p: (self._set_busy(False), self.statusBar().showMessage(f"Saved: {p}")),
                  "Exporting…")

    def about(self):
        QMessageBox.about(self, APP_NAME,
                          f"<b>{APP_NAME} {__version__}</b><p>Speech: DeepFilterNet3 (AI) + optional WPE.<br>"
                          "Music: stationary spectral noise reduction.<br>"
                          "Automatic speech/music detection with manual correction.</p>")

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
    style.apply(app)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    w = MainWindow()
    w.show()
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        w.load(Path(sys.argv[1]))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
