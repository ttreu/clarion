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

    # --- speech + music ("Sprache + Musik") ---------------------------------
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

    # --- tone matching ("Klang angleichen") ---------------------------------
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


PRESETS: dict[str, Settings] = {
    "Natürlich": Settings(),
    "Kräftig": Settings(music_reduction=0.95),
    "Sanft": Settings(speech_reduction_db=20.0, speech_dereverb=False, music_reduction=0.5, music_hiss_db=-3.0),
}
