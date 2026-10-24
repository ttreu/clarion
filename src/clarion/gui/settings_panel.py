"""The settings side of the main window: preset, Speech / Music / Speech + Music, Loudness (with the live
spectrum) and Output. It turns widgets into a ``Settings`` object and back; the window only listens to
``changed`` (any setting) and ``switched`` (a switch that should refresh the preview at once)."""
from __future__ import annotations

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .. import multiband
from ..settings import MIX_VARIANTS, MUSIC_VARIANTS, PRESETS, SPEECH_VARIANTS, Settings, preset_name, variant_of
from .spectrum import Spectrum

LOUDNESS = [("−14 LUFS (streaming, loud)", -14.0), ("−16 LUFS (podcast, web video)", -16.0),
            ("−18 LUFS (reading with music)", -18.0), ("−23 LUFS (broadcast, EBU R128)", -23.0)]
# music options that only the variants set (threshold, second pass, pause gate): no widget of their own
MUSIC_EXTRA = ("music_nstd", "music_pass2", "music_gate")
# Settings fields that presets and variants put into the controls
SOUND_FIELDS = ("speech_reduction_db", "speech_dereverb", "music_reduction", "music_hiss_db", "mix_music",
                *MUSIC_EXTRA)


def _slider(lo: int, hi: int, unit: str) -> tuple[QSlider, QLabel]:
    s = QSlider(Qt.Horizontal)
    s.setRange(lo, hi)
    lab = QLabel()
    lab.setMinimumWidth(48)
    s.valueChanged.connect(lambda val: lab.setText(f"{val} {unit}"))
    return s, lab


def _pair(a: QWidget, b: QWidget) -> QWidget:
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.addWidget(a, 1)
    h.addWidget(b)
    return w


def _variant_box(table) -> QComboBox:
    box = QComboBox()
    box.addItems([f"{letter} – {label}" for letter, label, _ in table] + ["Custom"])
    box.setToolTip("Variants from the listening tests – for the whole project. The controls below show "
                   "what a variant sets; changing them switches to “Custom”.")
    return box


class SettingsPanel(QWidget):
    changed = Signal()        # any setting changed (sliders: many times while dragging)
    switched = Signal()       # loudness on/off: worth refreshing the preview right away

    def __init__(self, prefs: QSettings, parent=None):
        super().__init__(parent)
        self.prefs = prefs
        self._music_extra = {k: getattr(Settings(), k) for k in MUSIC_EXTRA}
        self._syncing = False      # True while the panel fills its own widgets: no "changed"
        cols = QHBoxLayout(self)
        cols.setContentsMargins(0, 0, 0, 0)
        left, right = QVBoxLayout(), QVBoxLayout()
        cols.addLayout(left, 3)
        cols.addLayout(right, 2)

        top = QFormLayout()
        top.setContentsMargins(4, 4, 4, 0)
        self.cmb_preset = QComboBox()
        self.cmb_preset.addItems(list(PRESETS))
        self.cmb_preset.currentTextChanged.connect(self.apply_preset)
        top.addRow("Preset", self.cmb_preset)
        left.addLayout(top)
        groups = (self._speech_group(), self._music_group(), self._mix_group())
        for g in groups:
            left.addWidget(g)
        left.addStretch(1)
        for box in groups:  # same label width → sliders line up across the groups
            form = box.layout()
            for r in range(form.rowCount()):
                item = form.itemAt(r, QFormLayout.LabelRole)
                if item and item.widget():
                    item.widget().setMinimumWidth(150)

        right.addWidget(self._loudness_group())
        right.addWidget(self._output_group())
        right.addStretch(1)

        for wdg in (self.sl_speech, self.sl_music, self.sl_mix, self.sl_tone, self.sl_level):
            wdg.valueChanged.connect(self._changed)
        for wdg in (self.chk_wpe, self.chk_hiss, self.chk_ceiling):
            wdg.toggled.connect(self._changed)
        for wdg, sig in ((self.sl_speech, "valueChanged"), (self.chk_wpe, "toggled"),
                         (self.sl_music, "valueChanged"), (self.chk_hiss, "toggled"), (self.sl_mix, "valueChanged")):
            getattr(wdg, sig).connect(self._sync_variants)
        self.cmb_v_speech.activated.connect(lambda i: self._pick_variant(SPEECH_VARIANTS, i))
        self.cmb_v_music.activated.connect(lambda i: self._pick_variant(MUSIC_VARIANTS, i))
        self.cmb_v_mix.activated.connect(lambda i: self._pick_variant(MIX_VARIANTS, i))

        start = preset_name(self.prefs.value("preset"))
        self.apply_preset(start)
        self.cmb_preset.setCurrentText(start)

    # ------------------------------------------------------------------ groups
    def _speech_group(self) -> QGroupBox:
        g = QGroupBox("SPEECH")
        g.setObjectName("speech")   # timeline colour accents (style.py)
        f = QFormLayout(g)
        self.cmb_v_speech = _variant_box(SPEECH_VARIANTS)
        f.addRow("Variant", self.cmb_v_speech)
        self.sl_speech, lab = _slider(0, 40, "dB")
        f.addRow("Noise reduction", _pair(self.sl_speech, lab))
        self.chk_wpe = QCheckBox("Extra dereverberation (slower, drier)")
        f.addRow("", self.chk_wpe)
        self.sl_tone, lab = _slider(0, 100, "%")
        self.sl_tone.setToolTip("Evens out the voice's tone when the speaking position changes "
                                "(e.g. close to the PA microphone: boomy and harsh). Target: the driest parts "
                                "of the file, or a tone reference chosen by right-click.")
        f.addRow("Match tone", _pair(self.sl_tone, lab))
        self.sl_level, lab = _slider(0, 100, "%")
        self.sl_level.setToolTip("Evens out the voice's level across the file (measured over 6 s), "
                                 "e.g. close to the microphone vs. normal position. Music is left alone.")
        f.addRow("Match level", _pair(self.sl_level, lab))
        return g

    def _music_group(self) -> QGroupBox:
        g = QGroupBox("MUSIC")
        g.setObjectName("music")
        f = QFormLayout(g)
        self.cmb_v_music = _variant_box(MUSIC_VARIANTS)
        f.addRow("Variant", self.cmb_v_music)
        self.sl_music, lab = _slider(0, 100, "%")
        f.addRow("Noise reduction", _pair(self.sl_music, lab))
        self.chk_hiss = QCheckBox("Reduce treble hiss (above 8 kHz)")
        self.chk_hiss.setToolTip("Gently lowers everything above 8 kHz in music – mostly hiss lives there, "
                                 "instruments like the cello have little energy up there.")
        f.addRow("", self.chk_hiss)
        return g

    def _mix_group(self) -> QGroupBox:
        g = QGroupBox("SPEECH + MUSIC")
        g.setObjectName("mix")
        f = QFormLayout(g)
        self.cmb_v_mix = _variant_box(MIX_VARIANTS)
        f.addRow("Variant", self.cmb_v_mix)
        self.sl_mix, lab = _slider(0, 100, "%")
        self.sl_mix.setToolTip("Speech + Music = AI voice + this share of the denoised original. "
                               "Less: drier, less noise, quieter music. "
                               "More: more music, but also more room sound and noise.")
        f.addRow("Music share", _pair(self.sl_mix, lab))
        return g

    def _loudness_group(self) -> QGroupBox:
        # the group's own checkbox switches loudness on/off (Qt disables the children when off)
        self.chk_norm = QGroupBox("LOUDNESS")
        self.chk_norm.setCheckable(True)
        self.chk_norm.setToolTip("Brings the export to a target loudness (EBU R128 / ITU-R BS.1770), followed by "
                                 "a limiter at −1 dBFS. Off: peak-normalised to −1 dBFS only.")
        self.chk_norm.setChecked(self.prefs.value("normalize", True, type=bool))
        f = QFormLayout(self.chk_norm)
        self.cmb_loud = QComboBox()
        self.cmb_loud.addItems([label for label, _ in LOUDNESS])
        self.cmb_loud.setToolTip("Applies to export and preview (estimated there).")
        self.cmb_loud.setCurrentIndex(min(int(self.prefs.value("loudness_idx", 1)), len(LOUDNESS) - 1))
        self.cmb_loud.currentIndexChanged.connect(lambda i: (self.prefs.setValue("loudness_idx", i), self._changed()))
        f.addRow("Target", self.cmb_loud)
        self.chk_ceiling = QCheckBox("Limit loud music (at most 3 LU above target)")
        self.chk_ceiling.setToolTip("If a recording is mostly quiet speech, loudness normalisation raises "
                                    "everything a lot – loud music then hits the limiter and distorts. "
                                    "This option gently turns down just those passages.")
        self.chk_ceiling.setChecked(True)
        f.addRow("", self.chk_ceiling)
        self.chk_bc = QCheckBox("Broadcast compression (multiband)")
        self.chk_bc.setToolTip("Four-band compressor like Adobe Audition's Multiband Compressor, preset "
                               "“Broadcast”: brings speech closer to loud music and makes the sound denser "
                               "and more present. Runs before the loudness target is applied.")
        self.chk_bc.setChecked(self.prefs.value("broadcast", False, type=bool))
        self.chk_bc.toggled.connect(lambda on: (self.prefs.setValue("broadcast", on), self._changed(),
                                                self._sync_spectrum()))
        f.addRow("", self.chk_bc)
        self.spectrum = Spectrum()
        f.addRow(self.spectrum)
        self._sync_spectrum()
        self.chk_norm.toggled.connect(self._toggle_normalize)
        return self.chk_norm

    def _output_group(self) -> QGroupBox:
        g = QGroupBox("OUTPUT")
        f = QFormLayout(g)
        self.cmb_rate = QComboBox()
        self.cmb_rate.addItems(["48 kHz (video)", "Original sample rate"])
        f.addRow("WAV 24 bit", self.cmb_rate)
        return g

    # ------------------------------------------------------------------ settings ↔ widgets
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

    @property
    def keep_source_rate(self) -> bool:
        return self.cmb_rate.currentIndex() == 1

    def gui_state(self) -> dict:
        """Choices that are not Settings fields, for the project file."""
        return {"preset": self.cmb_preset.currentText(), "rate_idx": self.cmb_rate.currentIndex(),
                "loudness_idx": self.cmb_loud.currentIndex()}

    def apply(self, st: Settings, gui: dict) -> None:
        """Put saved settings (and GUI choices) into all controls, without reporting a change."""
        self._syncing = True
        self.cmb_preset.blockSignals(True)
        if gui.get("preset"):
            self.cmb_preset.setCurrentText(preset_name(gui["preset"]))
        self.cmb_preset.blockSignals(False)
        self._apply_fields({k: getattr(st, k) for k in SOUND_FIELDS})
        self.sl_tone.setValue(int(round(st.tone_match * 100)))
        self.sl_level.setValue(int(round(st.level_match * 100)))
        if st.loudness_target is not None:
            targets = [t for _, t in LOUDNESS]
            self.cmb_loud.setCurrentIndex(min(range(len(targets)), key=lambda i: abs(targets[i] - st.loudness_target)))
        self.chk_norm.setChecked(st.normalize and st.loudness_target is not None)
        self.chk_ceiling.setChecked(st.music_ceiling_lu is not None)
        self.chk_bc.setChecked(st.broadcast)
        if "rate_idx" in gui:
            self.cmb_rate.setCurrentIndex(int(gui["rate_idx"]))
        self._syncing = False
        self._sync_variants()

    def apply_preset(self, name: str) -> None:
        st = PRESETS.get(name)
        if not st:
            return
        self._apply_fields({k: getattr(st, k) for k in SOUND_FIELDS})
        self._sync_variants()
        self.sl_tone.setValue(int(round(st.tone_match * 100)))
        self.sl_level.setValue(int(round(st.level_match * 100)))
        self.prefs.setValue("preset", name)
        self._changed()

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

    # ------------------------------------------------------------------ variants
    def _pick_variant(self, table, i: int) -> None:
        if i >= len(table):       # "Custom": nothing to apply
            return
        self._apply_fields(table[i][2])
        self._sync_variants()
        self._changed()

    def _sync_variants(self, *_) -> None:
        """Show which variant the current settings are (or "Custom")."""
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

    # ------------------------------------------------------------------ loudness
    def _toggle_normalize(self, on: bool) -> None:
        self.prefs.setValue("normalize", on)
        self._sync_spectrum()
        self._changed()
        self.switched.emit()

    def _sync_spectrum(self) -> None:
        """Crossover lines in the spectrum: bright when the broadcast compressor is in use."""
        self.spectrum.set_crossovers(multiband.BROADCAST.crossovers,
                                     self.chk_bc.isChecked() and self.chk_norm.isChecked())

    def _changed(self, *_) -> None:
        if not self._syncing:
            self.changed.emit()
