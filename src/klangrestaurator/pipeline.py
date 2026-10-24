"""High-level project: analyse once (cached), then render any range with any settings."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

from . import analysis, audio_io, chains, dfn
from .analysis import MUSIC, Segment
from .settings import SR, Settings

Progress = Callable[[str, float], None] | None
Cancel = Callable[[], bool] | None

BLOCK_S = 60
PAD_S = 2


class Cancelled(Exception):
    pass


def cache_root() -> Path:
    if sys.platform == "darwin":
        d = Path.home() / "Library" / "Caches" / "Klangrestaurator"
    else:
        d = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "klangrestaurator"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _key(path: Path) -> str:
    st = path.stat()
    h = hashlib.sha1(f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:16]
    return f"{path.stem[:40]}-{h}"


class Project:
    def __init__(self, source: Path | str, workdir: Path | None = None):
        self.source = Path(source)
        self.dir = Path(workdir) if workdir else cache_root() / _key(self.source)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.orig: np.ndarray | None = None
        self.enh: np.ndarray | None = None
        self.levels: np.ndarray | None = None
        self.segments: list[Segment] = []
        self.meta: dict = {}

    # ------------------------------------------------------------------ analysis
    @property
    def duration(self) -> float:
        return 0.0 if self.orig is None else len(self.orig) / SR

    @property
    def analysed(self) -> bool:
        return (self.dir / "segments.json").exists() and (self.dir / "enh.npy").exists()

    def prepare(self, settings: Settings | None = None, progress: Progress = None, cancel: Cancel = None,
                reclassify: bool = False) -> None:
        """Decode, run the speech AI over the whole file and classify segments (cached)."""
        settings = settings or Settings()
        p = progress or (lambda *_: None)
        f_orig, f_enh, f_seg = self.dir / "orig.npy", self.dir / "enh.npy", self.dir / "segments.json"
        f_meta = self.dir / "meta.json"

        if not (f_orig.exists() and f_meta.exists()):
            p("Datei einlesen", 0)
            audio_io.load_to_memmap(self.source, f_orig, progress=lambda v: p("Datei einlesen", v))
            sr_in, ch, _ = audio_io.probe(self.source)
            orig = np.load(f_orig, mmap_mode="r")
            peak = max(float(np.abs(orig[s:s + BLOCK_S * SR]).max()) for s in range(0, len(orig), BLOCK_S * SR))
            f_meta.write_text(json.dumps({"source": str(self.source), "sr_in": sr_in, "channels": ch,
                                          "peak": peak}))
        self.meta = json.loads(f_meta.read_text())
        self.orig = np.load(f_orig, mmap_mode="r")
        self._check(cancel)

        if not f_enh.exists():
            p("KI-Entrauschung (DeepFilterNet)", 0)
            dfn.ensure_binary(progress=lambda v: p("KI-Modell herunterladen", v))
            tmp = self.dir / "enh.partial.npy"
            out = np.lib.format.open_memmap(tmp, "w+", np.float32, self.orig.shape)
            try:
                dfn.enhance(self.orig, out, progress=lambda v: p("KI-Entrauschung (DeepFilterNet)", v),
                            cancel=cancel)
            except InterruptedError:
                del out
                tmp.unlink(missing_ok=True)
                raise Cancelled() from None
            out.flush()
            del out
            tmp.rename(f_enh)
        self.enh = np.load(f_enh, mmap_mode="r")

        f_lev, f_elev = self.dir / "levels.npy", self.dir / "enh_levels.npy"
        if not f_lev.exists():
            np.save(f_lev, analysis.frame_levels(self.orig))
        if not f_elev.exists():
            np.save(f_elev, analysis.frame_levels(self.enh))
        self.levels = np.load(f_lev)

        if reclassify or not f_seg.exists():
            p("Sprache / Musik erkennen", 0.5)
            self.segments = analysis.classify(self.levels, np.load(f_elev), settings)
            self.save_segments()
        else:
            self.segments = analysis.load_segments(f_seg)
        p("Analyse fertig", 1)

    def save_segments(self) -> None:
        analysis.save_segments(self.segments, self.dir / "segments.json")

    def set_kind(self, start: float, end: float, kind: str) -> None:
        self.segments = analysis.set_kind(self.segments, start, end, kind)
        self.save_segments()

    @staticmethod
    def _check(cancel: Cancel):
        if cancel and cancel():
            raise Cancelled()

    # ------------------------------------------------------------------ rendering
    def _render_segments(self) -> list[analysis.Segment]:
        """Segments with automatic music starts snapped to the exact onset (cached per segment list)."""
        key = tuple((s.start, s.end, s.kind, s.manual) for s in self.segments)
        if getattr(self, "_snap_key", None) != key:
            self._snap_key = key
            self._snapped = analysis.snap_music_onsets(self.segments, self.orig)
        return self._snapped

    def render(self, settings: Settings, start_s: float = 0.0, end_s: float | None = None,
               progress: Progress = None, cancel: Cancel = None) -> np.ndarray:
        """Render [start_s, end_s) into a float32 array (memmap for long ranges)."""
        assert self.orig is not None and self.enh is not None, "prepare() zuerst aufrufen"
        p = progress or (lambda *_: None)
        N = len(self.orig)
        A = max(0, int(start_s * SR))
        B = N if end_s is None else min(N, int(end_s * SR))
        n = B - A
        if n > 10 * 60 * SR:
            out = np.lib.format.open_memmap(self.dir / "render.npy", "w+", np.float32, (n, 2))
        else:
            out = np.zeros((n, 2), np.float32)

        blk, pad = BLOCK_S * SR, PAD_S * SR
        n_blocks = max(1, -(-n // blk))
        segs = self._render_segments()
        music = [s for s in segs if s.kind == MUSIC and s.end * SR > A and s.start * SR < B]
        steps = n_blocks + len(music)
        step = 0
        for s in range(A, B, blk):
            self._check(cancel)
            e = min(B, s + blk)
            ps, pe = max(0, s - pad), min(N, e + pad)
            m = analysis.speech_mask(segs, e - s, settings.crossfade_s, start=s,
                                     onset_fade_s=settings.onset_fade_s)[:, None]
            o = np.asarray(self.orig[ps:pe])
            mu = chains.highpass(o, settings.music_highpass_hz)[s - ps:e - ps]
            if m.max() > 1e-3:
                sp = chains.speech_chain(o, np.asarray(self.enh[ps:pe]), settings.speech_reduction_db,
                                         settings.speech_highpass_hz, settings.speech_dereverb,
                                         settings.speech_stereo_link)[s - ps:e - ps]
                out[s - A:e - A] = sp * m + mu * (1 - m)
            else:
                out[s - A:e - A] = mu
            step += 1
            p("Restaurieren", step / steps)

        # music passages: add the spectral noise reduction on top of the plain high-passed signal
        for seg in music:
            self._check(cancel)
            a = max(A, int((seg.start - 1) * SR))
            b = min(B, int((seg.end + 1) * SR))
            if b - a < SR // 10:
                continue
            noise = chains.noise_profile(self.orig, self.levels, seg.start, seg.end)
            pa, pb = max(0, a - pad), min(N, b + pad)
            o = np.asarray(self.orig[pa:pb])
            x = chains.highpass(o, settings.music_highpass_hz)
            y = chains.music_chain(o, noise, settings.music_reduction, settings.music_highpass_hz)
            w = 1 - analysis.speech_mask(segs, b - a, settings.crossfade_s, start=a,
                                         onset_fade_s=settings.onset_fade_s)[:, None]
            out[a - A:b - A] += w * (y[a - pa:b - pa] - x[a - pa:b - pa])
            step += 1
            p("Restaurieren", step / steps)
        return out

    def output_gain(self, rendered: np.ndarray, settings: Settings) -> float:
        peak = 0.0
        for s in range(0, len(rendered), BLOCK_S * SR):
            peak = max(peak, float(np.abs(rendered[s:s + BLOCK_S * SR]).max()))
        return 10 ** (settings.peak_dbfs / 20) / max(peak, 1e-6)

    def preview_gain(self, settings: Settings) -> float:
        """Gain for short previews, based on the original's peak (no full render needed)."""
        return 10 ** (settings.peak_dbfs / 20) / max(self.meta.get("peak", 1.0), 1e-6)

    def export(self, settings: Settings, dest: Path | str, out_sr: int = SR, subtype: str = "PCM_24",
               start_s: float = 0.0, end_s: float | None = None, progress: Progress = None,
               cancel: Cancel = None) -> Path:
        p = progress or (lambda *_: None)
        y = self.render(settings, start_s, end_s, progress=lambda s, v: p(s, 0.9 * v), cancel=cancel)
        g = self.output_gain(y, settings)
        dest = Path(dest)
        audio_io.write_wav(y, SR, dest, gain=g, out_sr=out_sr, subtype=subtype,
                           progress=lambda v: p("Datei schreiben", 0.9 + 0.1 * min(1, v)))
        del y
        (self.dir / "render.npy").unlink(missing_ok=True)
        p("Fertig", 1)
        return dest

    def write_preview(self, settings: Settings, start_s: float, end_s: float,
                      cancel: Cancel = None, slot: int = 0) -> tuple[Path, Path]:
        """Render a short range and write (original, restored) WAVs for A/B listening.

        ``slot`` picks one of several file pairs, so a player can keep reading the previous preview
        while the next one is written.
        """
        g = self.preview_gain(settings)
        tmp = Path(tempfile.gettempdir()) / "klangrestaurator_preview"
        tmp.mkdir(exist_ok=True)
        a, b = int(start_s * SR), int(min(self.duration, end_s) * SR)
        tag = f"_{slot}" if slot else ""
        f_o, f_r = tmp / f"A_original{tag}.wav", tmp / f"B_restauriert{tag}.wav"
        audio_io.write_wav(self.orig, SR, f_o, start=a, stop=b, gain=g, subtype="PCM_16")
        y = self.render(settings, start_s, end_s, cancel=cancel)
        audio_io.write_wav(y, SR, f_r, gain=g, subtype="PCM_16")
        return f_o, f_r


def default_output_name(source: Path, preset_name: str = "") -> Path:
    tag = f"_{preset_name.lower()}" if preset_name else ""
    tag = tag.replace("ü", "ue").replace("ä", "ae").replace("ö", "oe")
    return source.with_name(f"{source.stem}_restauriert{tag}.wav")
