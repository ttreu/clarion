"""Command line: `klangrestaurator-cli ZOOM0001.WAV --preset Natürlich`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analysis import KIND_LABEL, SPEECH
from .pipeline import Project, default_output_name
from .settings import PRESETS, SR


def _bar(stage: str, v: float) -> None:
    n = int(v * 30)
    sys.stderr.write(f"\r{stage:<34} [{'#' * n}{'.' * (30 - n)}] {v * 100:5.1f}%")
    sys.stderr.flush()
    if v >= 1:
        sys.stderr.write("\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Restauriert Lesungen/Konzerte: KI-Entrauschung für Sprache, "
                                             "schonende Rauschminderung für Musik.")
    ap.add_argument("input", type=Path, nargs="+")
    ap.add_argument("-o", "--output", type=Path, help="Zieldatei (nur bei einer Eingabedatei)")
    ap.add_argument("-p", "--preset", default="Natürlich", choices=list(PRESETS))
    ap.add_argument("--speech-db", type=float, help="Rauschreduktion Sprache in dB (überschreibt Preset)")
    ap.add_argument("--music", type=float, help="Rauschreduktion Musik 0..1 (überschreibt Preset)")
    ap.add_argument("--mix-music", type=float,
                    help="„Sprache + Musik“: Anteil des entrauschten Originals 0..1 (Standard 0.5)")
    ap.add_argument("--dereverb", action=argparse.BooleanOptionalAction, help="Zusätzliche Hallreduktion (WPE)")
    ap.add_argument("--keep-rate", action="store_true", help="Original-Samplerate statt 48 kHz ausgeben")
    ap.add_argument("--segments", action="store_true",
                    help="Nur analysieren und Musik- / Sprache-+-Musik-Abschnitte auflisten")
    a = ap.parse_args(argv)

    st = PRESETS[a.preset]
    if a.speech_db is not None:
        st = st.with_(speech_reduction_db=a.speech_db)
    if a.music is not None:
        st = st.with_(music_reduction=a.music)
    if a.mix_music is not None:
        st = st.with_(mix_music=a.mix_music)
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
