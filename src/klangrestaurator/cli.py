"""Command line: `klangrestaurator-cli recording.wav --preset Natural`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analysis import KIND_LABEL, SPEECH
from .pipeline import Project, default_output_name
from .settings import DEFAULT_PRESET, PRESETS, SR, Settings, preset_name

# option → Settings field it overrides (value taken as given; None = option not used)
_DIRECT = {"speech_db": "speech_reduction_db", "music": "music_reduction", "mix_music": "mix_music",
           "tone": "tone_match", "level": "level_match", "dereverb": "speech_dereverb"}


def _lufs(text: str) -> float | None:
    """'-16' → -16.0, 'peak' → None (peak normalisation)."""
    if text.lower() == "peak":
        return None
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number like -16 or 'peak', got {text!r}") from None


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="klangrestaurator-cli",
                                 description="Cleans up field-recorder speech recordings and readings with "
                                             "accompanying music: AI denoising for the voice, gentle noise "
                                             "reduction for the music.")
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
    ap.add_argument("--broadcast", action="store_true",
                    help="multiband compression like Adobe Audition's 'Broadcast' preset "
                         "(before the loudness target)")
    ap.add_argument("--lufs", type=_lufs, default=argparse.SUPPRESS,   # attribute only exists when given
                    help="target loudness in LUFS (default -16) or 'peak' for peak normalisation to -1 dBFS")
    ap.add_argument("--dereverb", action=argparse.BooleanOptionalAction, help="extra dereverberation (WPE)")
    ap.add_argument("--keep-rate", action="store_true", help="write the original sample rate instead of 48 kHz")
    ap.add_argument("--segments", action="store_true",
                    help="only analyse and list music / speech+music sections")
    return ap


def settings_from_args(a: argparse.Namespace) -> Settings:
    """The preset with every option the user gave applied on top."""
    st = PRESETS[a.preset]
    changes = {field: getattr(a, opt) for opt, field in _DIRECT.items() if getattr(a, opt) is not None}
    if a.hiss is not None:
        changes["music_hiss_db"] = -abs(a.hiss)
    if a.no_normalize:
        changes["normalize"] = False
    if a.broadcast:
        changes["broadcast"] = True
    if hasattr(a, "lufs"):                 # given: a number, or None for 'peak'
        changes["loudness_target"] = a.lufs
    return st.with_(**changes)


def _bar(stage: str, v: float) -> None:
    n = int(v * 30)
    sys.stderr.write(f"\r{stage:<34} [{'#' * n}{'.' * (30 - n)}] {v * 100:5.1f}%")
    sys.stderr.flush()
    if v >= 1:
        sys.stderr.write("\n")


def _list_sections(proj: Project) -> None:
    for s in proj.segments:
        if s.kind != SPEECH:
            print(f"{KIND_LABEL[s.kind]} {int(s.start // 60)}:{s.start % 60:04.1f} – "
                  f"{int(s.end // 60)}:{s.end % 60:04.1f}")


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    missing = [str(p) for p in a.input if not p.exists()]
    if missing:
        ap.error("file not found: " + ", ".join(missing))
    if a.output and len(a.input) > 1:
        ap.error("-o/--output only works with a single input file")
    st = settings_from_args(a)

    for src in a.input:
        print(f"== {src}", file=sys.stderr)
        proj = Project(src)
        proj.prepare(st, progress=_bar)
        if a.segments:
            _list_sections(proj)
            continue
        dest = a.output or default_output_name(src, a.preset)
        out_sr = proj.meta.get("sr_in", SR) if a.keep_rate else SR
        proj.export(st, dest, out_sr=out_sr, progress=_bar)
        print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
