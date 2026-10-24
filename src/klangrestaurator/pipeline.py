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

from . import analysis, audio_io, chains, dfn, dynamics, tone
from .analysis import MIX, MUSIC, SPEECH, Segment
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
        self.wave: np.ndarray | None = None   # (peak, rms) per 5 ms, for the timeline
        self.segments: list[Segment] = []
        self.meta: dict = {}
        self.voice_bands: np.ndarray | None = None   # third-octave bands of the AI voice (tone matching)
        self.enh_levels: np.ndarray | None = None
        self._curve_key = None
        self._curve: np.ndarray | None = None
        self._loud_cache: dict = {}
        self._probe_cache: dict = {}

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
            p("Reading file", 0)
            audio_io.load_to_memmap(self.source, f_orig, progress=lambda v: p("Reading file", v))
            sr_in, ch, _ = audio_io.probe(self.source)
            orig = np.load(f_orig, mmap_mode="r")
            peak = max(float(np.abs(orig[s:s + BLOCK_S * SR]).max()) for s in range(0, len(orig), BLOCK_S * SR))
            f_meta.write_text(json.dumps({"source": str(self.source), "sr_in": sr_in, "channels": ch,
                                          "peak": peak}))
        self.meta = json.loads(f_meta.read_text())
        self.orig = np.load(f_orig, mmap_mode="r")
        self._check(cancel)

        if not f_enh.exists():
            p("AI denoising (DeepFilterNet)", 0)
            dfn.ensure_binary(progress=lambda v: p("Downloading AI model", v))
            tmp = self.dir / "enh.partial.npy"
            out = np.lib.format.open_memmap(tmp, "w+", np.float32, self.orig.shape)
            try:
                dfn.enhance(self.orig, out, progress=lambda v: p("AI denoising (DeepFilterNet)", v),
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
        f_wave = self.dir / "wave.npy"
        if not f_wave.exists():
            np.save(f_wave, analysis.waveform_overview(self.orig))
        self.wave = np.load(f_wave)

        f_vb = self.dir / "voice_bands.npy"
        if not f_vb.exists():
            p("Analysing tone", 0.5)
            np.save(f_vb, self._compute_voice_bands(cancel))
        self.voice_bands = np.load(f_vb)

        if reclassify or not f_seg.exists():
            p("Detecting speech / music", 0.5)
            self.segments = analysis.classify(self.levels, np.load(f_elev), settings)
            self.save_segments()
        else:
            self.segments = analysis.load_segments(f_seg)
        p("Analysis done", 1)

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

    # ------------------------------------------------------------------ trimming
    @property
    def trim(self) -> tuple[float, float] | None:
        """Exported range (s) set by the user (start/end cut), or None for the whole file."""
        f = self.dir / "trim.json"
        if f.exists():
            a, b = json.loads(f.read_text())
            return float(a), float(b)
        return None

    def set_trim(self, rng: tuple[float, float] | None) -> None:
        f = self.dir / "trim.json"
        if rng is None:
            f.unlink(missing_ok=True)
            return
        a, b = max(0.0, float(rng[0])), float(rng[1])
        if self.duration:
            b = min(b, self.duration)
        if b - a < 0.5:
            raise ValueError("the range must be at least 0.5 s long")
        f.write_text(json.dumps([a, b]))

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
        """Voice EQ (dB, frames × bands): tone matching plus level matching, or None when both are off."""
        if (settings.tone_match <= 0 and settings.level_match <= 0) or self.voice_bands is None:
            return None
        segs = tuple((s.start, s.end, s.kind) for s in self.segments)
        key = (settings.tone_match, settings.tone_max_db, self.tone_ref, settings.level_match,
               settings.level_max_db, segs)
        if key != self._curve_key:
            bp = self.voice_bands
            curve = np.zeros_like(bp)
            t = np.arange(len(bp)) * tone.HOP / SR
            voiced = np.zeros(len(bp), bool)        # only speech sections count as voice
            for sg in self.segments:
                if sg.kind in (SPEECH, MIX):
                    voiced |= (t >= sg.start) & (t < sg.end)
            if settings.tone_match > 0:
                fr = np.arange(len(bp)) * tone.HOP / SR        # frame times (s)
                dry = np.interp(fr * 4, np.arange(len(self.levels)) + 0.5, self.enh_levels - self.levels)
                curve = tone.correction_curve(bp, dry, settings.tone_match, settings.tone_max_db,
                                              ref=self.tone_ref, voiced=voiced)
            if settings.level_match > 0:
                curve = curve + tone.level_curve(bp, curve, settings.level_match, settings.level_max_db,
                                                 voiced=voiced)[:, None]
            self._curve, self._curve_key = curve, key
        return self._curve

    def save_segments(self) -> None:
        analysis.save_segments(self.segments, self.dir / "segments.json")

    def move_boundary(self, old: float, new: float) -> None:
        self.segments = analysis.move_boundary(self.segments, old, new)
        self.save_segments()

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
        settings = settings.effective()
        assert self.orig is not None and self.enh is not None, "call prepare() first"
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
            p("Restoring", step / steps)

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
            only_music = all(sg.kind == MUSIC for sg in run)   # the pause gate is for music sections only
            y = chains.music_chain(o, noise, settings.music_reduction, settings.music_highpass_hz,
                                   settings.music_hiss_db, settings.music_nstd, settings.music_pass2,
                                   settings.music_gate and only_music)[a - pa:b - pa]
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
                ym = chains.music_chain(o, noise, settings.mix_reduction, settings.music_highpass_hz,
                                        settings.music_hiss_db)
                # no WPE here: it treats sustained notes as reverb (02.WAV: −4 dB cello in the word gaps)
                ym = ym[a - pa:b - pa]
                wm = analysis.kind_mask(segs, MIX, b - a, settings.mix_fade_s, start=a)[:, None]
                out[a - A:b - A] += wm * (v + settings.mix_music * ym - y)
            step += 1
            p("Restoring", step / steps)
        return out

    def output_gain(self, rendered: np.ndarray, settings: Settings) -> float:
        peak = 0.0
        for s in range(0, len(rendered), BLOCK_S * SR):
            peak = max(peak, float(np.abs(rendered[s:s + BLOCK_S * SR]).max()))
        return 10 ** (settings.peak_dbfs / 20) / max(peak, 1e-6)

    def preview_gain(self, settings: Settings) -> float:
        """Gain for short previews, based on the original's peak (no full render needed)."""
        return 10 ** (settings.peak_dbfs / 20) / max(self.meta.get("peak", 1.0), 1e-6)

    @property
    def orig_loudness(self) -> float:
        """Integrated loudness (LUFS) of the whole original, measured once and kept in meta.json."""
        if "lufs" not in self.meta:
            self.meta["lufs"] = dynamics.loudness(self.orig)
            (self.dir / "meta.json").write_text(json.dumps(self.meta))
        return float(self.meta["lufs"])

    def _probes(self, settings: Settings, cancel: Cancel = None) -> tuple[np.ndarray, np.ndarray]:
        """(restored, original) audio of 8 probes à 2 s spread over the file (whole file if ≤ 20 s).

        Rendered before loudness normalisation; cached per restoration settings and sections.
        """
        st = settings.with_(loudness_target=None, normalize=False)
        key = (st, tuple((s.start, s.end, s.kind, s.manual) for s in self.segments), self.tone_ref)
        if key not in self._probe_cache:
            if self.duration <= 20:
                pr, po = self.render(st, cancel=cancel), np.asarray(self.orig)
            else:
                starts = np.linspace(0, self.duration - 2, 8)
                ys = [self.render(st, t, t + 2, cancel=cancel) for t in starts]
                pr = np.concatenate(ys)
                po = np.concatenate([np.asarray(self.orig[int(t * SR):int(t * SR) + len(y)])
                                     for t, y in zip(starts, ys, strict=True)])
            self._probe_cache = {key: (np.asarray(pr, np.float32), np.asarray(po, np.float32))}
        self._probe_key = key
        return self._probe_cache[key]

    def estimated_loudness(self, settings: Settings, cancel: Cancel = None) -> float:
        """Loudness (LUFS) the full render would have, without rendering it all.

        Short files are rendered completely; otherwise the change the restoration makes on the probes
        is added to the original's loudness (within ~0.5 dB on the test files; a single preview range
        can be off by 6 dB because speech loses more than music).
        """
        pr, po = self._probes(settings, cancel)
        if self.duration <= 20:
            return dynamics.loudness(pr)
        return self.orig_loudness + dynamics.loudness(pr) - dynamics.loudness(po)

    def _tame_music(self, y: np.ndarray, start_s: float, g: float, settings: Settings) -> bool:
        """Turn down music stretches whose short-term loudness after the gain ``g`` would exceed the
        target by more than ``music_ceiling_lu`` (in place). True if anything was changed."""
        hop = 0.1
        stl = dynamics.short_term_lufs(y, hop) + 20 * np.log10(max(g, 1e-12))
        if not len(stl):
            return False
        t = start_s + (np.arange(len(stl)) + 0.5) * hop
        music = np.zeros(len(stl), bool)
        for sg in self.segments:
            if sg.kind == MUSIC:
                music |= (t >= sg.start) & (t < sg.end)
        red = np.where(music, np.minimum(0.0, settings.loudness_target + settings.music_ceiling_lu - stl), 0.0)
        if red.min() > -0.1:
            return False
        import scipy.ndimage as nd
        red = nd.minimum_filter1d(red, 31)                         # start early, hold through the passage
        red = np.convolve(red, np.ones(21) / 21, "same")           # ~2 s soft ramps
        centres = (np.arange(len(red)) + 0.5) * hop * SR
        for s in range(0, len(y), BLOCK_S * SR):
            e = min(len(y), s + BLOCK_S * SR)
            gdb = np.interp(np.arange(s, e), centres, red)
            y[s:e] = np.asarray(y[s:e]) * (10 ** (gdb / 20)).astype(np.float32)[:, None]
        return True

    @staticmethod
    def _db_gain(target: float, measured: float, fallback: float = 1.0) -> float:
        return 10 ** ((target - measured) / 20) if np.isfinite(measured) else fallback

    def export(self, settings: Settings, dest: Path | str, out_sr: int = SR, subtype: str = "PCM_24",
               start_s: float = 0.0, end_s: float | None = None, progress: Progress = None,
               cancel: Cancel = None) -> Path:
        p = progress or (lambda *_: None)
        settings = settings.effective()
        if start_s == 0.0 and end_s is None and self.trim:   # the user's cut (Anfang/Ende)
            start_s, end_s = self.trim
        y = self.render(settings, start_s, end_s, progress=lambda s, v: p(s, 0.85 * v), cancel=cancel)
        if start_s > 0 or (end_s is not None and end_s < self.duration):
            _fade_edges(y, int(FADE_S * SR))           # no click where the recording was cut
        dest = Path(dest)
        if settings.loudness_target is None:   # peak normalisation only (as before)
            g, lim = self.output_gain(y, settings), None
        else:                                  # loudness normalisation (+ music ceiling) + limiter
            p("Measuring loudness", 0.86)
            g = self._db_gain(settings.loudness_target, dynamics.loudness(y), self.output_gain(y, settings))
            if settings.music_ceiling_lu is not None:
                # turn down music that would end up too loud; the gain stays as measured before, so
                # speech keeps its level (re-measuring would raise everything again and undo the cap
                # when music is a large part of the file)
                self._tame_music(y, start_s, g, settings)
            lim = dynamics.Limiter(settings.limiter_db)
        audio_io.write_wav(y, SR, dest, gain=g, out_sr=out_sr, subtype=subtype, limiter=lim,
                           progress=lambda v: p("Writing file", 0.9 + 0.1 * min(1, v)))
        del y
        (self.dir / "render.npy").unlink(missing_ok=True)
        p("Done", 1)
        return dest

    def write_preview(self, settings: Settings, start_s: float, end_s: float,
                      cancel: Cancel = None, slot: int = 0) -> tuple[Path, Path]:
        """Render a short range and write (original, restored) WAVs for A/B listening.

        ``slot`` picks one of several file pairs, so a player can keep reading the previous preview
        while the next one is written.
        """
        settings = settings.effective()
        tmp = Path(tempfile.gettempdir()) / "klangrestaurator_preview"
        tmp.mkdir(exist_ok=True)
        a, b = int(start_s * SR), int(min(self.duration, end_s) * SR)
        tag = f"_{slot}" if slot else ""
        f_o, f_r = tmp / f"A_original{tag}.wav", tmp / f"B_restored{tag}.wav"
        y = self.render(settings, start_s, end_s, cancel=cancel)
        if settings.loudness_target is None:
            # the export normalises the whole render's peak; estimate its gain from how much the
            # restoration changes the peak, on probes from the whole file (peaks change differently in
            # speech and music, a fixed gain from the original was up to 6 dB off)
            g_o = self.preview_gain(settings)
            pr, po = self._probes(settings, cancel)
            pk_o, pk_r = float(np.abs(po).max()), float(np.abs(pr).max())
            g = g_o * pk_o / pk_r if pk_o > 0 and pk_r > 0 else g_o
            lim_o = None
            lim_r = dynamics.Limiter(settings.peak_dbfs)   # estimate may overshoot locally: stay safe
        else:
            # same gain the export will use (estimated from probes); the original gets its own
            # normalisation so A and B are compared at the same loudness
            g_o = self._db_gain(settings.loudness_target, self.orig_loudness, self.preview_gain(settings))
            g = self._db_gain(settings.loudness_target, self.estimated_loudness(settings, cancel),
                              self.preview_gain(settings))
            if settings.music_ceiling_lu is not None:
                y = np.array(y, dtype=np.float32)
                self._tame_music(y, start_s, g, settings)
            lim_o, lim_r = dynamics.Limiter(settings.limiter_db), dynamics.Limiter(settings.limiter_db)
        audio_io.write_wav(self.orig, SR, f_o, start=a, stop=b, gain=g_o, subtype="PCM_16", limiter=lim_o)
        audio_io.write_wav(y, SR, f_r, gain=g, subtype="PCM_16", limiter=lim_r)
        return f_o, f_r


FADE_S = 0.02


def _fade_edges(y: np.ndarray, n: int) -> None:
    """Short raised-cosine fade in/out, in place (works on memmaps)."""
    n = min(n, len(y) // 2)
    if n < 2:
        return
    ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))).astype(np.float32)[:, None]
    y[:n] *= ramp
    y[len(y) - n:] *= ramp[::-1]


def default_output_name(source: Path, preset_name: str = "") -> Path:
    tag = f"_{preset_name.lower()}" if preset_name else ""
    return source.with_name(f"{source.stem}_restored{tag}.wav")
