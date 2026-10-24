"""Loudness (ITU-R BS.1770 / EBU R128), true peak and a look-ahead limiter for the export.

A voice compressor and a mastering stage (EQ + 3-band compression) were built here and dropped:
at matched loudness they did not beat the plain restoration in a listening comparison.
"""
from __future__ import annotations

import numpy as np
import scipy.signal as ss

from .settings import SR

# --- loudness (ITU-R BS.1770-4) -----------------------------------------------------------------

# K-weighting at 48 kHz: high shelf (head) + high-pass (RLB), coefficients from the standard
_K1 = (np.array([1.53512485958697, -2.69169618940638, 1.19839281085285]),
       np.array([1.0, -1.69065929318241, 0.73248077421585]))
_K2 = (np.array([1.0, -2.0, 1.0]), np.array([1.0, -1.99004745483398, 0.99007225036621]))
_SOS_K = np.vstack([np.concatenate(_K1), np.concatenate(_K2)])


class LoudnessMeter:
    """Streaming integrated loudness (ITU-R BS.1770-4): feed blocks in order, read ``lufs``."""

    def __init__(self):
        self.step, self.win = int(0.1 * SR), int(0.4 * SR)   # 400 ms gating blocks, 75 % overlap
        self.zi = None
        self.powers: list[np.ndarray] = []
        self.carry = np.zeros((0,), np.float64)              # filtered power not yet in a full block

    def add(self, x: np.ndarray) -> None:
        y = np.asarray(x, dtype=np.float64)
        y = y[:, None] if y.ndim == 1 else y
        if len(y) == 0:
            return
        if self.zi is None:
            self.zi = np.zeros((_SOS_K.shape[0], 2, y.shape[1]))
        f, self.zi = ss.sosfilt(_SOS_K, y, axis=0, zi=self.zi)
        self.carry = np.concatenate([self.carry, (f ** 2).sum(axis=1)])   # channel weights 1.0 (L, R)
        m = (len(self.carry) - self.win) // self.step + 1 if len(self.carry) >= self.win else 0
        if m > 0:
            c = np.cumsum(np.concatenate([[0.0], self.carry]))
            idx = np.arange(m) * self.step
            self.powers.append((c[idx + self.win] - c[idx]) / self.win)
            self.carry = self.carry[m * self.step:]

    @property
    def lufs(self) -> float:
        if not self.powers:
            return float("-inf")
        z = np.concatenate(self.powers)
        z = z[-0.691 + 10 * np.log10(z + 1e-30) > -70]
        if len(z) == 0:
            return float("-inf")
        rel = -0.691 + 10 * np.log10(z.mean()) - 10
        z = z[-0.691 + 10 * np.log10(z + 1e-30) > rel]
        return float(-0.691 + 10 * np.log10(z.mean()))


def loudness(x: np.ndarray, block_s: int = 60) -> float:
    """Integrated loudness in LUFS of a 48 kHz (n, ch) signal; works blockwise on memmaps.

    -inf if everything is below the absolute gate (-70 LUFS).
    """
    m = LoudnessMeter()
    for s in range(0, len(x), block_s * SR):
        m.add(x[s:s + block_s * SR])
    return m.lufs


def short_term_lufs(x: np.ndarray, hop_s: float = 0.1, win_s: float = 3.0, block_s: int = 60) -> np.ndarray:
    """Short-term loudness (BS.1770 K-weighting, ``win_s`` window) every ``hop_s``; value i belongs to
    the time (i + 0.5) * hop_s. Blockwise, so it works on memmaps."""
    hop = int(hop_s * SR)
    zi, powers, carry = None, [], np.zeros(0)
    for s in range(0, len(x), block_s * SR):
        y = np.asarray(x[s:s + block_s * SR], dtype=np.float64)
        y = y[:, None] if y.ndim == 1 else y
        if zi is None:
            zi = np.zeros((_SOS_K.shape[0], 2, y.shape[1]))
        f, zi = ss.sosfilt(_SOS_K, y, axis=0, zi=zi)
        carry = np.concatenate([carry, (f ** 2).sum(axis=1)])
        m = len(carry) // hop
        if m:
            powers.append(carry[:m * hop].reshape(m, hop).mean(axis=1))
            carry = carry[m * hop:]
    if len(carry):
        powers.append(np.array([carry.mean()]))
    if not powers:
        return np.zeros(0)
    pw = np.concatenate(powers)
    k = max(1, int(round(win_s / hop_s)))
    avg = np.convolve(pw, np.ones(k) / k, "same")
    return -0.691 + 10 * np.log10(avg + 1e-12)


def true_peak_db(x: np.ndarray, block_s: int = 60) -> float:
    """Approximate true peak (4× oversampled) in dBTP, blockwise."""
    peak = 0.0
    for s in range(0, len(x), block_s * SR):
        y = np.asarray(x[max(0, s - 64):s + block_s * SR + 64], dtype=np.float32)
        peak = max(peak, float(np.abs(ss.resample_poly(y, 4, 1, axis=0)).max()) if len(y) else 0.0)
    return 20 * np.log10(max(peak, 1e-12))


# --- gain smoothing on a 1 ms grid ----------------------------------------------------------------

_HOP = SR // 1000  # 1 ms


def _frames_peak(x: np.ndarray) -> np.ndarray:
    """Per 1 ms frame: max |sample| over all channels."""
    n = len(x) // _HOP * _HOP
    a = np.abs(np.asarray(x[:n], dtype=np.float32)).reshape(-1, _HOP, x.shape[1]).max(axis=(1, 2))
    if n < len(x):
        a = np.append(a, np.abs(x[n:]).max())
    return a


def _to_samples(g: np.ndarray, n: int) -> np.ndarray:
    """Frame gains (centred on frames) → per-sample gains by linear interpolation."""
    centres = (np.arange(len(g)) + 0.5) * _HOP
    return np.interp(np.arange(n), centres, g).astype(np.float32)


# --- limiter ------------------------------------------------------------------------------------

class Limiter:
    """Look-ahead peak limiter on a 1 ms grid; keeps its state across consecutive blocks.

    Release 400 ms: the first version let go after 80 ms, which distorted loud low-pitched music
    audibly (ZOOM0002.WAV, cello at 28:44 – heard as "clipping").

    Feed blocks in order with ``process``; it delays the audio by the look-ahead, so call ``flush``
    at the end to get the remaining samples.
    """

    def __init__(self, ceiling_db: float = -1.0, lookahead_ms: int = 5, release_ms: float = 400.0):
        self.ceiling = 10 ** (ceiling_db / 20)
        self.la = lookahead_ms
        self.r = np.exp(-1.0 / release_ms)
        self.g = 1.0
        self._pend = np.zeros((0, 2), np.float32)     # audio waiting for its gain (look-ahead)
        self.min_gain = 1.0                           # statistics: strongest reduction so far
        self.reduced_frames = 0
        self.frames = 0

    def process(self, x: np.ndarray, final: bool = False) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        buf = np.concatenate([self._pend, x]) if len(self._pend) else x
        if len(buf) == 0:
            return buf
        need = np.minimum(1.0, self.ceiling / np.maximum(_frames_peak(buf), 1e-12))
        nf = len(need)
        # each frame must already be attenuated la frames ahead of a peak → running min forward
        req = np.array([need[i:i + self.la + 1].min() for i in range(nf)], np.float32)
        keep = nf if final else max(0, nf - self.la - 1)        # frames whose look-ahead is complete
        out_g = np.empty(keep, np.float32)
        g = self.g
        for i in range(keep):
            g = req[i] if req[i] < g else min(req[i], self.r * g + (1 - self.r))
            out_g[i] = g
        self.g = g
        if keep:
            self.min_gain = min(self.min_gain, float(out_g.min()))
            self.reduced_frames += int((out_g < 0.98).sum())     # > 0.2 dB reduction
            self.frames += keep
        n_out = len(buf) if final else keep * _HOP
        gains = np.minimum(_to_samples(out_g, n_out), np.repeat(out_g, _HOP)[:n_out]) if keep else \
            np.zeros(0, np.float32)
        y = buf[:n_out] * gains[:, None]
        self._pend = buf[n_out:]
        return np.clip(y, -self.ceiling, self.ceiling)

    def flush(self) -> np.ndarray:
        return self.process(np.zeros((0, self._pend.shape[1] if len(self._pend) else 2), np.float32),
                            final=True)


def limit(x: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    """Limit a whole (short) signal in one go."""
    lim = Limiter(ceiling_db)
    return np.concatenate([lim.process(x), lim.flush()])
