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

from . import analysis, audio_io, chains, dfn, tone
from .analysis import MIX, SPEECH, Segment
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
        self.voice_bands: np.ndarray | None = None   # third-octave bands of the AI voice (tone matching)
        self.enh_levels: np.ndarray | None = None
        self._curve_key = None
        self._curve: np.ndarray | None = None

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
        self.enh_levels = np.load(f_elev)

        f_vb = self.dir / "voice_bands.npy"
        if not f_vb.exists():
            p("Klang analysieren", 0.5)
            np.save(f_vb, self._compute_voice_bands(cancel))
        self.voice_bands = np.load(f_vb)

        if reclassify or not f_seg.exists():
            p("Sprache / Musik erkennen", 0.5)
            self.segments = analysis.classify(self.levels, np.load(f_elev), settings)
            self.save_segments()
        else:
            self.segments = analysis.load_segments(f_seg)
        p("Analyse fertig", 1)

    def _compute_voice_bands(self, cancel: Cancel = None) -> np.ndarray:
        """Band powers of the stereo-linked AI voice, blockwise over the whole file."""
        n, blk, pad = len(self.orig), BLOCK_S * SR, tone.N_FFT
        parts = []
        for s in range(0, n, blk):
            self._check(cancel)
            e = min(n, s + blk)
            a, b = (max(0, s - pad) // tone.HOP) * tone.HOP, min(n, e + pad)   # a on the frame grid
            v = chains.stereo_link(np.asarray(self.orig[a:b]), np.asarray(self.enh[a:b]))
            bp = tone.band_powers(v)                       # local frame j is global frame a/HOP + j
            f0, f1 = -(-s // tone.HOP), (e - 1) // tone.HOP + 1
            parts.append(bp[f0 - a // tone.HOP:f1 - a // tone.HOP])
        return np.concatenate(parts).astype(np.float32)

    # ------------------------------------------------------------------ tone matching
    @property
    def tone_ref(self) -> tuple[float, float] | None:
        """Range (s) whose voice timbre is the target of tone matching; None = automatic (driest parts)."""
        f = self.dir / "tone_ref.json"
        if f.exists():
            a, b = json.loads(f.read_text())
            return float(a), float(b)
        return None

    def set_tone_ref(self, rng: tuple[float, float] | None) -> None:
        f = self.dir / "tone_ref.json"
        if rng is None:
            f.unlink(missing_ok=True)
        else:
            f.write_text(json.dumps([float(rng[0]), float(rng[1])]))
        self._curve_key = None

    def tone_curve(self, settings: Settings) -> np.ndarray | None:
        """EQ curve (dB, frames × bands) for the voice, or None when tone matching is off."""
        if settings.tone_match <= 0 or self.voice_bands is None:
            return None
        key = (settings.tone_match, settings.tone_max_db, self.tone_ref)
        if key != self._curve_key:
            fr = np.arange(len(self.voice_bands)) * tone.HOP / SR        # frame times (s)
            dry = np.interp(fr * 4, np.arange(len(self.levels)) + 0.5, self.enh_levels - self.levels)
            self._curve = tone.correction_curve(self.voice_bands, dry, settings.tone_match,
                                                settings.tone_max_db, ref=self.tone_ref)
            self._curve_key = key
        return self._curve

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
        # consecutive music / speech+music sections share one noise-reduction pass (no double counting
        # at their common boundary)
        runs: list[list[Segment]] = []
        for sg in segs:
            if sg.kind == SPEECH:
                continue
            if runs and abs(runs[-1][-1].end - sg.start) < 1e-6:
                runs[-1].append(sg)
            else:
                runs.append([sg])
        runs = [r for r in runs if r[-1].end * SR > A and r[0].start * SR < B]
        steps = n_blocks + len(runs)
        step = 0
        curve = self.tone_curve(settings)
        for s in range(A, B, blk):
            self._check(cancel)
            e = min(B, s + blk)
            ps, pe = max(0, s - pad), min(N, e + pad)
            m = analysis.speech_mask(segs, e - s, settings.crossfade_s, start=s,
                                     onset_fade_s=settings.onset_fade_s, mix_fade_s=settings.mix_fade_s)[:, None]
            o = np.asarray(self.orig[ps:pe])
            mu = chains.highpass(o, settings.music_highpass_hz)[s - ps:e - ps]
            if m.max() > 1e-3:
                sp = chains.speech_chain(o, np.asarray(self.enh[ps:pe]), settings.speech_reduction_db,
                                         settings.speech_highpass_hz, settings.speech_dereverb,
                                         settings.speech_stereo_link)
                if curve is not None:
                    sp = tone.apply_eq(sp, curve, ps)
                sp = sp[s - ps:e - ps]
                out[s - A:e - A] = sp * m + mu * (1 - m)
            else:
                out[s - A:e - A] = mu
            step += 1
            p("Restaurieren", step / steps)

        # music: add the spectral noise reduction on top of the plain high-passed signal (M = music chain).
        # speech + music: dry AI voice + mix_music · M'  (like the speech chain, but with a large share of
        # the denoised original, so the music stays continuous under the voice; M' uses mix_reduction)
        for run in runs:
            self._check(cancel)
            r0, r1 = run[0].start, run[-1].end
            a = max(A, int((r0 - 1) * SR))
            b = min(B, int((r1 + 1) * SR))
            if b - a < SR // 10:
                continue
            noise = chains.noise_profile(self.orig, self.levels, r0, r1)
            pa, pb = max(0, a - pad), min(N, b + pad)
            o = np.asarray(self.orig[pa:pb])
            x = chains.highpass(o, settings.music_highpass_hz)[a - pa:b - pa]
            y = chains.music_chain(o, noise, settings.music_reduction, settings.music_highpass_hz)[a - pa:b - pa]
            ms = analysis.speech_mask(segs, b - a, settings.crossfade_s, start=a,
                                      onset_fade_s=settings.onset_fade_s, mix_fade_s=settings.mix_fade_s)[:, None]
            out[a - A:b - A] += (1 - ms) * (y - x)
            if any(sg.kind == MIX for sg in run):
                v = chains.highpass(chains.stereo_link(o, np.asarray(self.enh[pa:pb])),
                                    settings.speech_highpass_hz)
                if curve is not None:
                    v = tone.apply_eq(v, curve, pa)
                v = v[a - pa:b - pa]
                # the music share gets a stronger NR: noise is stationary, music isn't, so this costs
                # little music (02.WAV: noise −13 dB instead of −9 dB, cello −2.4 dB instead of −1.8 dB)
                ym = chains.music_chain(o, noise, settings.mix_reduction, settings.music_highpass_hz)
                if settings.speech_dereverb:  # "Zusätzliche Hallreduktion" also dries the room sound here
                    ym = chains.wpe_dereverb(ym)
                ym = ym[a - pa:b - pa]
                wm = analysis.kind_mask(segs, MIX, b - a, settings.mix_fade_s, start=a)[:, None]
                out[a - A:b - A] += wm * (v + settings.mix_music * ym - y)
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
