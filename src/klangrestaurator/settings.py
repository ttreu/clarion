"""Restoration settings and presets."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

SR = 48_000  # internal processing rate (DeepFilterNet requires 48 kHz)


@dataclass(frozen=True)
class Settings:
    # --- speech chain -------------------------------------------------------
    speech_reduction_db: float = 20.0
    """How far noise/reverb may be pushed down in speech passages.
    The AI output is blended with this amount of the original (attenuation limit)."""
    speech_dereverb: bool = False
    """Additional classic dereverberation (WPE) on speech. Slow, adds dryness."""
    speech_highpass_hz: float = 70.0

    # --- music chain --------------------------------------------------------
    music_reduction: float = 0.70
    """0..1 – proportion of stationary noise removed in music passages (0.7 ≈ -10 dB)."""
    music_highpass_hz: float = 30.0

    # --- segmentation -------------------------------------------------------
    crossfade_s: float = 0.5
    min_segment_s: float = 3.0
    music_ratio_db: float = -18.0
    """A frame counts as 'music' if the speech AI removes more than this many dB."""
    active_level_db: float = -42.0
    """Frames quieter than this are treated as pauses (not used for classification)."""

    # --- output ---------------------------------------------------------------
    peak_dbfs: float = -1.0

    def to_dict(self) -> dict:
        return asdict(self)

    def with_(self, **kw) -> Settings:
        return replace(self, **kw)


PRESETS: dict[str, Settings] = {
    "Natürlich": Settings(),
    "Kräftig": Settings(speech_reduction_db=30.0, speech_dereverb=True, music_reduction=0.85),
    "Sanft": Settings(speech_reduction_db=12.0, music_reduction=0.5),
}
