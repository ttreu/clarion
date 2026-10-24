"""Command line: `klangrestaurator-cli ZOOM0001.WAV --preset Natural`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analysis import KIND_LABEL, SPEECH
from .pipeline import Project, default_output_name
from .settings import DEFAULT_PRESET, PRESETS, SR, preset_name


def _bar(stage: str, v: float) -> None:
    n = int(v * 30)
    sys.stderr.write(f"\r{stage:<34} [{'#' * n}{'.' * (30 - n)}] {v * 100:5.1f}%")
    sys.stderr.flush()
    if v >= 1:
        sys.stderr.write("\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Restores readings and concerts: AI denoising for speech, "
                                             "gentle noise reduction for music.")
    ap.add_argument("input", type=Path, nargs="+")
    ap.add_argument("-o", "--output", type=Path, help="output file (only with a single input)")
    ap.add_argument("-p", "--preset", default=DEFAULT_PRESET, type=preset_name, choices=list(PRESETS))
    ap.add_argument("--speech-db", type=float, help="speech noise reduction in dB (overrides the preset)")
    ap.add_argument("--music", type=float, help="music noise reduction 0..1 (overrides the preset)")
    ap.add_argument("--hiss", type=float, help="music: cut above 8 kHz in dB, e.g. -6 (0 = off)")
    ap.add_argument("--mix-music", type=float,
                    help="speech + music: share of the denoised original 0..1 (default 0.5)")
    ap.add_argument("--tone", type=float, help="match the voice's tone 0..1 (default 0.7, 0 = off)")
    ap.add_argument("--level", type=float, help="match the voice's level 0..1 (default 0.7, 0 = off)")
    ap.add_argument("--no-normalize", action="store_true",
                    help="no loudness normalisation: peak-normalise to -1 dBFS instead of LUFS + limiter")
    ap.add_argument("--lufs", help="target loudness in LUFS (default -16) or 'peak' for peak normalisation to -1 dBFS")
    ap.add_argument("--dereverb", action=argparse.BooleanOptionalAction, help="extra dereverberation (WPE)")
    ap.add_argument("--keep-rate", action="store_true", help="Original-Samplerate statt 48 kHz ausgeben")
    ap.add_argument("--segments", action="store_true",
                    help="only analyse and list music / speech+music sections")
    a = ap.parse_args(argv)

    st = PRESETS[a.preset]
    if a.speech_db is not None:
        st = st.with_(speech_reduction_db=a.speech_db)
    if a.music is not None:
        st = st.with_(music_reduction=a.music)
    if a.hiss is not None:
        st = st.with_(music_hiss_db=-abs(a.hiss))
    if a.mix_music is not None:
        st = st.with_(mix_music=a.mix_music)
    if a.tone is not None:
        st = st.with_(tone_match=a.tone)
    if a.level is not None:
        st = st.with_(level_match=a.level)
    if a.no_normalize:
        st = st.with_(normalize=False)
    if a.lufs is not None:
        st = st.with_(loudness_target=None if a.lufs.lower() == "peak" else float(a.lufs))
    if a.dereverb is not None:
        st = st.with_(speech_dereverb=a.dereverb)

    for src in a.input:
        print(f"== {src}", file=sys.stderr)
        proj = Project(src)
        proj.prepare(st, progress=_bar)
        if a.segments:
            for s in proj.segments:
                if s.kind != SPEECH:
                    print(f"{KIND_LABEL[s.kind]} {int(s.start // 60)}:{s.start % 60:04.1f} – "
                          f"{int(s.end // 60)}:{s.end % 60:04.1f}")
            continue
        dest = a.output if (a.output and len(a.input) == 1) else default_output_name(src, a.preset)
        out_sr = proj.meta.get("sr_in", SR) if a.keep_rate else SR
        proj.export(st, dest, out_sr=out_sr, progress=_bar)
        print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
