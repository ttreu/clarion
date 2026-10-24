"""Restoration settings and presets."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

SR = 48_000  # internal processing rate (DeepFilterNet requires 48 kHz)


@dataclass(frozen=True)
class Settings:
    # --- speech chain -------------------------------------------------------
    speech_reduction_db: float = 40.0
    """How far noise/reverb may be pushed down in speech passages.
    The AI output is blended with this amount of the original (attenuation limit).
    40 dB + WPE chosen by ear on 01.wav and 02.WAV over 20/25/30 dB (02: noise −27 → −34 dB below the
    voice); the price is a little more "breathing" of the noise floor with the syllables."""
    speech_dereverb: bool = True
    """Additional classic dereverberation (WPE) on speech. ~3.5× slower render, adds dryness."""
    speech_highpass_hz: float = 70.0
    speech_stereo_link: bool = True
    """Apply the AI's suppression with one gain for both channels (stops the voice drifting L/R)."""

    # --- speech + music ---------------------------------
    mix_music: float = 0.5
    """Speech + music = dry AI voice + this share of the gently denoised original (music chain).
    Lower: drier voice, less noise, quieter music. Higher: more music, more room sound and noise."""
    mix_reduction: float = 0.95
    """Noise reduction (0..1) for the music share in speech + music; stronger than in pure music."""
    mix_fade_s: float = 1.0
    """Crossfade at boundaries of speech + music sections."""
    mix_gap_db: float = 6.0
    """Detection: gaps between words this far above the noise floor count as music under speech."""
    mix_gap_share: float = 0.4
    """Detection: share of such gaps (within 8 s) needed to mark speech as speech + music."""
    mix_speech_share: float = 0.1
    """Detection: a music section where at least this share of frames carries speech becomes speech + music."""
    min_switch_s: float = 6.0
    """Automatic sections shorter than this are merged into a neighbour (no rapid switching)."""

    # --- tone matching ("Match tone") ---------------------------------
    tone_match: float = 0.7
    """0..1: how far the voice's timbre is pulled towards the reference (driest parts or a chosen range).
    Evens out changes of microphone position, e.g. close to the PA (boomy, harsh) vs. normal."""
    tone_max_db: float = 12.0
    """Largest EQ correction per band before ``tone_match`` is applied."""
    level_match: float = 0.7
    """0..1: how far the voice's loudness (6 s windows) is pulled to the file's median, e.g. speaker
    close to vs. away from the microphone. Music is not touched."""
    level_max_db: float = 10.0
    """Largest level correction before ``level_match`` is applied."""

    # --- music chain --------------------------------------------------------
    music_reduction: float = 0.85
    """0..1 – proportion of stationary noise removed in music passages (0.85 ≈ -10 dB between notes
    on 01.wav)."""
    music_highpass_hz: float = 30.0
    music_nstd: float = 1.5
    """NR threshold in noise standard deviations (higher catches noise slightly above the floor)."""
    music_pass2: float = 0.0
    """Strength of an optional second NR pass (0 = off)."""
    music_gate: bool = False
    """Lower the hiss between notes with a gentle expander (music sections only)."""
    music_hiss_db: float = -6.0
    """Shelf above 8 kHz for music (and the music share of speech + music); 0 = off. Chosen by ear on
    01.wav together with 85 % reduction over stronger or two-pass NR (no 'underwater' sound)."""

    # --- segmentation -------------------------------------------------------
    crossfade_s: float = 0.5
    """Crossfade music→speech, centred on the boundary."""
    onset_fade_s: float = 0.04
    """Crossfade speech→music. Short and ending exactly at the (5 ms precise) music onset, so the
    music chain's noise floor arrives with the music instead of swelling in the silence before."""
    min_segment_s: float = 3.0
    music_ratio_db: float = -18.0
    """A frame counts as 'music' if the speech AI removes more than this many dB."""
    active_level_db: float = -42.0
    """Frames quieter than this are treated as pauses (not used for classification)."""

    # --- loudness (after restoration) ----------------------------------------------
    normalize: bool = True
    """Loudness normalisation to ``loudness_target`` plus limiter. Off: only peak normalisation to
    ``peak_dbfs`` (the behaviour before). A mastering stage (EQ + multiband compression) and a voice
    compressor were tried and dropped: in a blind comparison the plain restoration sounded best."""
    loudness_target: float | None = -16.0
    """Integrated loudness of the export in LUFS (EBU R128 / ITU-R BS.1770), followed by a limiter.
    None: only normalise the peak to ``peak_dbfs`` as before."""
    music_ceiling_lu: float | None = 3.0
    """After loudness normalisation, music sections may be at most this many LU above the target
    (3 s short-term loudness); louder stretches are turned down smoothly. None = off. Without it a
    speech-heavy file was raised so much that loud music hit the limiter hard (ZOOM0002.WAV: 28:44)."""
    limiter_db: float = -1.0
    """Limiter ceiling (dBFS) after loudness normalisation."""

    # --- output ---------------------------------------------------------------
    peak_dbfs: float = -1.0

    def to_dict(self) -> dict:
        return asdict(self)

    def with_(self, **kw) -> Settings:
        return replace(self, **kw)

    def effective(self) -> Settings:
        """Settings as processing sees them (switched-off normalisation means peak mode)."""
        return self if self.normalize else replace(self, loudness_target=None)


# Variants from the listening tests, selectable per group for the whole project (GUI drop-downs).
# Each maps to Settings fields; the letters match the sample files the user compared.
SPEECH_VARIANTS: list[tuple[str, str, dict]] = [
    ("A", "20 dB", dict(speech_reduction_db=20.0, speech_dereverb=False)),
    ("B", "25 dB", dict(speech_reduction_db=25.0, speech_dereverb=False)),
    ("C", "30 dB", dict(speech_reduction_db=30.0, speech_dereverb=False)),
    ("D", "25 dB + dereverb", dict(speech_reduction_db=25.0, speech_dereverb=True)),
    ("E", "40 dB + dereverb", dict(speech_reduction_db=40.0, speech_dereverb=True)),
]
_M = dict(music_nstd=1.5, music_pass2=0.0, music_gate=False, music_hiss_db=0.0)
MUSIC_VARIANTS: list[tuple[str, str, dict]] = [
    ("A", "70 %", {**_M, "music_reduction": 0.70}),
    ("B", "95 %", {**_M, "music_reduction": 0.95}),
    ("C", "95 %, higher threshold", {**_M, "music_reduction": 0.95, "music_nstd": 2.0}),
    ("D", "85 % + treble −6 dB", {**_M, "music_reduction": 0.85, "music_hiss_db": -6.0}),
    ("E", "two passes 85 % + 60 %", {**_M, "music_reduction": 0.85, "music_pass2": 0.6}),
    ("F", "95 % + gap reduction", {**_M, "music_reduction": 0.95, "music_gate": True}),
]
MIX_VARIANTS: list[tuple[str, str, dict]] = [
    ("A", "music share 25 %", dict(mix_music=0.25)),
    ("B", "music share 50 %", dict(mix_music=0.5)),
    ("C", "music share 75 %", dict(mix_music=0.75)),
]


def variant_of(st: Settings, variants: list[tuple[str, str, dict]]) -> str | None:
    """Letter of the variant ``st`` matches exactly, or None (custom)."""
    for letter, _, fields in variants:
        if all(abs(getattr(st, k) - v) < 1e-6 if isinstance(v, float) else getattr(st, k) == v
               for k, v in fields.items()):
            return letter
    return None


def _from_variants(speech: str, music: str, mix: str, **extra) -> Settings:
    pick = lambda table, letter: next(f for lt, _, f in table if lt == letter)  # noqa: E731
    return Settings(**pick(SPEECH_VARIANTS, speech), **pick(MUSIC_VARIANTS, music), **pick(MIX_VARIANTS, mix),
                    **extra)


PRESETS: dict[str, Settings] = {
    "Natural": _from_variants("E", "D", "B"),
    "Strong": _from_variants("E", "C", "A"),
    "Gentle": _from_variants("A", "A", "C"),
}
DEFAULT_PRESET = "Natural"
# names used before the app was translated (stored in preferences and project files)
_OLD_PRESET_NAMES = {"Natürlich": "Natural", "Kräftig": "Strong", "Sanft": "Gentle"}


def preset_name(name: str | None) -> str:
    """Current preset name for ``name`` (also accepts the old German names)."""
    name = _OLD_PRESET_NAMES.get(name or "", name or "")
    return name if name in PRESETS else DEFAULT_PRESET
