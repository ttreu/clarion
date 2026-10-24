"""DeepFilterNet3 speech enhancement via the official standalone `deep-filter` binary.

Using the Rust binary avoids a PyTorch dependency. It is downloaded once from the
GitHub release into the user's application-support folder. The binary loads the whole
input into memory, so audio is processed in chunks with overlap and crossfades.
"""
from __future__ import annotations

import os
import platform
import stat
import subprocess
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from .settings import SR

VERSION = "0.5.6"
URL = "https://github.com/Rikorose/DeepFilterNet/releases/download/v{v}/deep-filter-{v}-{t}"

CHUNK_S = 300      # 5 min per call keeps memory below ~1 GB
OVERLAP_S = 4


def _target() -> str:
    m = platform.machine().lower()
    if sys.platform == "darwin":
        return "aarch64-apple-darwin" if m in ("arm64", "aarch64") else "x86_64-apple-darwin"
    if sys.platform.startswith("linux"):
        return "aarch64-unknown-linux-gnu" if m in ("arm64", "aarch64") else "x86_64-unknown-linux-musl"
    if sys.platform == "win32":
        return "x86_64-pc-windows-msvc.exe"
    raise RuntimeError(f"Platform not supported: {sys.platform}/{m}")


def app_dir() -> Path:
    if sys.platform == "darwin":
        d = Path.home() / "Library" / "Application Support" / "Klangrestaurator"
    else:
        d = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "klangrestaurator"
    d.mkdir(parents=True, exist_ok=True)
    return d


def binary_path() -> Path:
    env = os.environ.get("KLANGRESTAURATOR_DEEPFILTER")
    if env:
        return Path(env)
    return app_dir() / "bin" / f"deep-filter-{VERSION}"


def ensure_binary(progress=None) -> Path:
    """Return the path to deep-filter, downloading it on first use."""
    p = binary_path()
    if p.exists():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    url = URL.format(v=VERSION, t=_target())
    tmp = p.with_suffix(".part")

    def hook(blocks, bs, total):
        if progress and total > 0:
            progress(min(1.0, blocks * bs / total))

    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    tmp.chmod(tmp.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    if sys.platform == "darwin":  # drop quarantine flag if present
        subprocess.run(["xattr", "-d", "com.apple.quarantine", str(tmp)], capture_output=True)
    tmp.rename(p)
    return p


def _run_chunk(binary: Path, audio: np.ndarray, workdir: Path, idx: int) -> np.ndarray:
    src = workdir / f"in_{idx:04d}.wav"
    outdir = workdir / f"out_{idx:04d}"
    outdir.mkdir(exist_ok=True)
    sf.write(src, audio, SR, subtype="FLOAT")
    # -D compensates the model delay so output is time-aligned with the input
    r = subprocess.run([str(binary), "-D", "-o", str(outdir), str(src)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"deep-filter failed: {r.stderr.strip()[-500:]}")
    y, _ = sf.read(outdir / src.name, dtype="float32", always_2d=True)
    src.unlink(missing_ok=True)
    (outdir / src.name).unlink(missing_ok=True)
    if len(y) < len(audio):  # binary may drop a few trailing samples
        y = np.concatenate([y, np.zeros((len(audio) - len(y), y.shape[1]), np.float32)])
    return y[: len(audio)]


def enhance(x: np.ndarray, out: np.ndarray, start: int = 0, stop: int | None = None,
            progress=None, cancel=None, workers: int | None = None) -> None:
    """Run DeepFilterNet on x[start:stop] (48 kHz, stereo) and write into out[start:stop]."""
    binary = ensure_binary()
    stop = len(x) if stop is None else stop
    C, O = CHUNK_S * SR, OVERLAP_S * SR
    starts = list(range(start, stop, C - O)) if stop - start > C else [start]
    starts = [s for i, s in enumerate(starts) if i == 0 or s < stop - O]
    workers = workers or max(1, min(4, (os.cpu_count() or 2) // 2))
    done = [0]
    fade = np.linspace(0, 1, O, dtype=np.float32)[:, None]

    with tempfile.TemporaryDirectory(prefix="klang_dfn_") as td:
        wd = Path(td)

        def job(i_s):
            i, s = i_s
            if cancel and cancel():
                raise InterruptedError
            e = min(stop, s + C)
            y = _run_chunk(binary, np.asarray(x[s:e]), wd, i)
            done[0] += 1
            if progress:
                progress(done[0] / len(starts))
            return i, s, e, y

        with ThreadPoolExecutor(workers) as ex:
            # map() yields in submission order, so chunks are written sequentially
            for i, s, e, y in ex.map(job, enumerate(starts)):
                _place(out, i, s, e, y, O, fade)


def _place(out, i, s, e, y, O, fade):
    if i == 0:
        out[s:e] = y
        return
    # crossfade the overlap with the previous chunk
    ov = min(O, e - s)
    out[s:s + ov] = out[s:s + ov] * (1 - fade[:ov]) + y[:ov] * fade[:ov]
    out[s + ov:e] = y[ov:]
