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
# files whose sound clarion takes from the first audio track (needs ffmpeg): camera and phone videos
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mts", ".m2ts", ".avi", ".mkv", ".mxf", ".3gp"}
AUDIO_EXT = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".m4a", ".ogg", ".w64", ".caf", *VIDEO_EXT}


def _ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def probe(path: Path) -> tuple[int, int, float]:
    """Return (samplerate, channels, duration_s). Falls back to ffmpeg for mp3/m4a and videos
    (first audio track; for videos the duration may be the container's, see ``load_to_memmap``)."""
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
    if not native:  # compressed formats and videos: ffmpeg decodes the first audio track to a temp wav
        ff = _ffmpeg()
        if not ff:
            raise RuntimeError("This format needs ffmpeg (brew install ffmpeg).")
        tmp = dest.with_suffix(".decode.wav")
        subprocess.run([ff, "-v", "error", "-y", "-i", str(path), "-map", "0:a:0", "-vn", "-ar", str(SR),
                        "-c:a", "pcm_f32le", str(tmp)], check=True)
        path, sr_in = tmp, SR
        dur = sf.info(str(tmp)).frames / SR    # the track itself: a video container can be longer
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


class StreamWriter:
    """Write 48 kHz audio arriving in blocks of any length, optionally resampling seamlessly.

    Resampling works on fixed chunks with ``ctx`` samples of context on both sides, so block
    borders leave no clicks (plain per-block resample_poly zero-pads every edge).
    """

    def __init__(self, dest: Path, channels: int, out_sr: int = SR, subtype: str = "PCM_24",
                 chunk_s: int = BLOCK_S, ctx: int = 4800):
        self.f = sf.SoundFile(str(dest), "w", out_sr, channels, subtype=subtype)
        g = np.gcd(out_sr, SR)
        self.up, self.down = out_sr // g, SR // g
        self.L, self.P = chunk_s * SR, ctx
        self.buf = np.zeros((0, channels), np.float32)
        self.prev = np.zeros((0, channels), np.float32)   # last ctx samples already written

    def _emit(self, chunk: np.ndarray, nxt: np.ndarray) -> None:
        if self.up == self.down:
            y = chunk
        else:
            seg = np.concatenate([self.prev, chunk, nxt])
            y = resample_poly(seg, self.up, self.down, axis=0)
            i0 = len(self.prev) * self.up // self.down
            y = y[i0:i0 + len(chunk) * self.up // self.down]
        self.f.write(np.clip(y, -1.0, 1.0))
        self.prev = chunk[-self.P:]

    def write(self, x: np.ndarray) -> None:
        self.buf = np.concatenate([self.buf, np.asarray(x, dtype=np.float32)])
        while len(self.buf) >= self.L + self.P:
            self._emit(self.buf[:self.L], self.buf[self.L:self.L + self.P])
            self.buf = self.buf[self.L:]

    def close(self) -> None:
        if len(self.buf):
            self._emit(self.buf, np.zeros((0, self.buf.shape[1]), np.float32))
        self.f.close()


def write_wav(x: np.ndarray, sr: int, dest: Path, start: int = 0, stop: int | None = None,
              gain: float = 1.0, out_sr: int = SR, subtype: str = "PCM_24", progress=None,
              limiter=None) -> None:
    """Write (a slice of) a 48 kHz array to a WAV file in blocks: gain → limiter → resampling."""
    stop = len(x) if stop is None else stop
    blk = BLOCK_S * sr
    w = StreamWriter(Path(dest), x.shape[1], out_sr, subtype)
    try:
        for s in range(start, stop, blk):
            y = np.asarray(x[s:min(stop, s + blk)], dtype=np.float32) * gain
            w.write(limiter.process(y) if limiter else y)
            if progress:
                progress((s - start + blk) / max(1, stop - start))
        if limiter:
            w.write(limiter.flush())
    finally:
        w.close()
