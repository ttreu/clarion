"""Audio loading/saving. Everything is converted to 48 kHz float32 stereo for processing.

Large files are handled as .npy memmaps in a work directory so 1-hour recordings
fit into a few GB of RAM.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .settings import SR

BLOCK_S = 60


def _ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def probe(path: Path) -> tuple[int, int, float]:
    """Return (samplerate, channels, duration_s). Falls back to ffmpeg for mp3/m4a."""
    try:
        i = sf.info(str(path))
        return i.samplerate, i.channels, i.frames / i.samplerate
    except Exception:
        ff = _ffmpeg()
        if not ff:
            raise
        out = subprocess.run([ff.replace("ffmpeg", "ffprobe"), "-v", "error", "-select_streams", "a:0",
                              "-show_entries", "stream=sample_rate,channels:format=duration",
                              "-of", "default=nw=1", str(path)], capture_output=True, text=True).stdout
        d = dict(line.split("=", 1) for line in out.strip().splitlines() if "=" in line)
        return int(d["sample_rate"]), int(d["channels"]), float(d["duration"])


def _resample(x: np.ndarray, sr_in: int) -> np.ndarray:
    if sr_in == SR:
        return x
    g = np.gcd(SR, sr_in)
    return resample_poly(x, SR // g, sr_in // g, axis=0).astype(np.float32)


def load_to_memmap(path: Path, dest: Path, progress=None) -> np.memmap:
    """Decode `path` to a 48 kHz float32 stereo memmap at `dest` (.npy)."""
    path = Path(path)
    sr_in, ch, dur = probe(path)
    try:
        sf.info(str(path))
        native = True
    except Exception:
        native = False
    if not native:  # decode compressed formats via ffmpeg into a temp wav first
        ff = _ffmpeg()
        if not ff:
            raise RuntimeError("Für dieses Format wird ffmpeg benötigt (brew install ffmpeg).")
        tmp = dest.with_suffix(".decode.wav")
        subprocess.run([ff, "-v", "error", "-y", "-i", str(path), "-ar", str(SR), "-c:a", "pcm_f32le", str(tmp)],
                       check=True)
        path, sr_in = tmp, SR
    n_out = int(round(dur * SR))
    mm = np.lib.format.open_memmap(dest, "w+", np.float32, (n_out, 2))
    pos = 0
    blk = BLOCK_S * sr_in
    pad = sr_in  # 1 s context for the resampler
    with sf.SoundFile(str(path)) as f:
        total = f.frames
        for s in range(0, total, blk):
            a = max(0, s - pad)
            f.seek(a)
            x = f.read(min(total, s + blk + pad) - a, dtype="float32", always_2d=True)
            if x.shape[1] == 1:
                x = np.repeat(x, 2, axis=1)
            x = x[:, :2]
            y = _resample(x, sr_in)
            ratio = SR / sr_in
            y = y[int(round((s - a) * ratio)):int(round((s - a) * ratio)) + int(round(min(blk, total - s) * ratio))]
            y = y[: n_out - pos]
            mm[pos:pos + len(y)] = y
            pos += len(y)
            if progress:
                progress(min(1.0, (s + blk) / total))
    if pos < n_out:
        mm[pos:] = 0
    mm.flush()
    if not native:
        path.unlink(missing_ok=True)
    return mm


def write_wav(x: np.ndarray, sr: int, dest: Path, start: int = 0, stop: int | None = None,
              gain: float = 1.0, out_sr: int = SR, subtype: str = "PCM_24", progress=None) -> None:
    """Write (a slice of) a 48 kHz array to a WAV file in blocks, optionally resampling."""
    stop = len(x) if stop is None else stop
    blk = BLOCK_S * sr
    with sf.SoundFile(str(dest), "w", out_sr, x.shape[1], subtype=subtype) as f:
        for s in range(start, stop, blk):
            y = np.asarray(x[s:min(stop, s + blk)], dtype=np.float32) * gain
            if out_sr != sr:
                g = np.gcd(out_sr, sr)
                y = resample_poly(y, out_sr // g, sr // g, axis=0)
            f.write(np.clip(y, -1.0, 1.0))
            if progress:
                progress((s - start + blk) / max(1, stop - start))
