"""Four-band compressor modelled on Adobe Audition's Multiband Compressor ("Broadcast" preset).

Signal path (as Audition): Linkwitz-Riley crossovers (4th order, phase-compensated so the bands
sum flat) → per band: peak detector, hard-knee compressor, band gain → sum → the limiter stage
raises everything by (margin − threshold) and brick-wall limits at the margin.

Thresholds are absolute dBFS, so the result depends on the input level: the pipeline feeds this
stage at a fixed reference loudness (``REF_LUFS``) and normalises to the user's target after it.

Streaming: ``MultibandCompressor.process`` keeps filter and envelope state across blocks; the
limiter delays by its look-ahead, so call ``flush`` at the end.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, lfilter, sosfilt, sosfilt_zi

from .dynamics import Limiter
from .settings import SR

HOP = SR // 1000            # 1 ms detector grid
REF_LUFS = -17.2            # loudness of the clarion export the user ran through Audition (whole file)


@dataclass(frozen=True)
class Band:
    thresh_db: float
    ratio: float
    gain_db: float
    attack_ms: float = 1.0
    release_ms: float = 100.0


@dataclass(frozen=True)
class Preset:
    crossovers: tuple[float, float, float]
    bands: tuple[Band, Band, Band, Band]
    limiter_thresh_db: float
    margin_db: float
    limiter_attack_ms: float
    limiter_release_ms: float
    output_gain_db: float = 0.0


BROADCAST = Preset(
    crossovers=(137.0, 1147.0, 6910.0),
    bands=(Band(-20.2, 3.0, 1.0), Band(-17.6, 2.0, 0.0), Band(-24.8, 2.0, 0.0), Band(-25.0, 3.0, 3.0)),
    limiter_thresh_db=-10.0, margin_db=-0.1, limiter_attack_ms=4.0, limiter_release_ms=500.0,
)


class _Sos:
    """sosfilt with state kept across blocks (stereo)."""

    def __init__(self, sos: np.ndarray):
        self.sos = sos
        self.zi = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.zi is None:
            self.zi = np.repeat(sosfilt_zi(self.sos)[:, :, None], x.shape[1], axis=2) * 0.0
        y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y


def _lr4(f: float, kind: str) -> np.ndarray:
    s = butter(2, f, kind, fs=SR, output="sos")
    return np.concatenate([s, s])            # Butterworth² = Linkwitz-Riley 4


class _Splitter:
    """x → 4 bands that sum to an allpass-filtered x (flat magnitude)."""

    def __init__(self, xo):
        f1, f2, f3 = xo
        self.lp1, self.hp1 = _Sos(_lr4(f1, "low")), _Sos(_lr4(f1, "high"))
        self.lp2, self.hp2 = _Sos(_lr4(f2, "low")), _Sos(_lr4(f2, "high"))
        self.lp3, self.hp3 = _Sos(_lr4(f3, "low")), _Sos(_lr4(f3, "high"))
        # band 1 must get the same phase as the higher bands: allpass at f2 and f3 (= LP + HP)
        self.ap2 = (_Sos(_lr4(f2, "low")), _Sos(_lr4(f2, "high")))
        self.ap3 = (_Sos(_lr4(f3, "low")), _Sos(_lr4(f3, "high")))
        self.ap3b = (_Sos(_lr4(f3, "low")), _Sos(_lr4(f3, "high")))

    @staticmethod
    def _ap(pair, x):
        return pair[0](x) + pair[1](x)

    def __call__(self, x):
        b1, rest = self.lp1(x), self.hp1(x)
        b1 = self._ap(self.ap3, self._ap(self.ap2, b1))
        b2, rest = self.lp2(rest), self.hp2(rest)
        b2 = self._ap(self.ap3b, b2)
        b3, b4 = self.lp3(rest), self.hp3(rest)
        return b1, b2, b3, b4


class _BandComp:
    """Peak detector on a 1 ms grid, hard knee. The gain reduction follows rising targets with the
    attack time and decays with the release time (exponential). The release is vectorised:
    gr[i] = max(t[i], r·gr[i−1])  ⇔  gr[i] = r^i · cummax_j(t[j] · r^−j), in chunks so r^−j stays finite."""

    CHUNK = 10_000                     # ms
    DETECT = "peak"                    # level detector per 1 ms frame: "peak" or "rms"
    KNEE_DB = 0.0                      # soft-knee width (0 = hard knee)

    def __init__(self, band: Band):
        self.b = band
        self.r = np.exp(-1.0 / band.release_ms)
        self.a = np.exp(-1.0 / max(band.attack_ms, 1e-3))
        self.gr = 0.0                  # gain reduction at the end of the last call (dB)
        self.att = 0.0                 # attack filter state
        self.g_last = 10 ** (band.gain_db / 20)

    def gains(self, x: np.ndarray) -> np.ndarray:
        """Linear gain per sample for the band signal ``x`` (n, 2); blocks should be whole ms."""
        n = len(x)
        nf = -(-n // HOP)
        pad = np.zeros((nf * HOP - n, x.shape[1]))
        fr = np.concatenate([x, pad]).reshape(nf, HOP, -1)
        if self.DETECT == "rms":
            lvl = 10 * np.log10((fr ** 2).mean(axis=(1, 2)) * 2 + 1e-18)      # sine: rms·√2 = peak
        else:
            lvl = 20 * np.log10(np.maximum(np.abs(fr).max(axis=(1, 2)), 1e-9))
        over = lvl - self.b.thresh_db
        k = self.KNEE_DB
        if k > 0:
            over = np.where(over <= -k / 2, 0.0, np.where(over >= k / 2, over, (over + k / 2) ** 2 / (2 * k)))
        else:
            over = np.maximum(over, 0.0)
        target = over * (1.0 - 1.0 / self.b.ratio)
        target, zf = lfilter([1 - self.a], [1, -self.a], target, zi=[self.att * self.a])
        self.att = float(target[-1])
        gr = np.empty(nf)
        for s in range(0, nf, self.CHUNK):
            t = target[s:s + self.CHUNK]
            rk = self.r ** np.arange(1, len(t) + 1)
            run = np.maximum(np.maximum.accumulate(t / rk), self.gr)   # previous value decays too
            gr[s:s + len(t)] = run * rk
            self.gr = float(gr[s + len(t) - 1])
        g = 10 ** ((self.b.gain_db - gr) / 20)
        prev = np.concatenate([[self.g_last], g[:-1]])            # ramp from frame to frame (no zipper)
        ramp = (np.arange(HOP) + 1) / HOP
        self.g_last = float(g[-1])
        return (prev[:, None] + (g - prev)[:, None] * ramp[None, :]).ravel()[:n]


class MultibandCompressor:
    def __init__(self, preset: Preset = BROADCAST, input_gain_db: float = 0.0):
        self.p = preset
        self.split = _Splitter(preset.crossovers)
        self.comps = [_BandComp(b) for b in preset.bands]
        self.in_gain = 10 ** (input_gain_db / 20)
        self.makeup = 10 ** ((preset.margin_db - preset.limiter_thresh_db + preset.output_gain_db) / 20)
        self.lim = Limiter(preset.margin_db, lookahead_ms=int(round(preset.limiter_attack_ms)),
                           release_ms=preset.limiter_release_ms)

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float64) * self.in_gain
        y = np.zeros_like(x)
        for band, comp in zip(self.split(x), self.comps, strict=True):
            y += band * comp.gains(band)[:, None]
        return self.lim.process((y * self.makeup).astype(np.float32))

    def flush(self) -> np.ndarray:
        return self.lim.flush()


def compress_inplace(y: np.ndarray, input_gain_db: float = 0.0, preset: Preset = BROADCAST,
                     block_s: int = 60, cancel=None) -> None:
    """Compress ``y`` (array or memmap) in place, block by block. The output lags the input by the
    limiter's look-ahead, so each output lands before the next block is read."""
    mbc = MultibandCompressor(preset, input_gain_db)
    pos = 0
    for s in range(0, len(y), block_s * SR):
        if cancel is not None and cancel():
            from .pipeline import Cancelled
            raise Cancelled()
        out = mbc.process(np.asarray(y[s:s + block_s * SR]))
        y[pos:pos + len(out)] = out
        pos += len(out)
    out = mbc.flush()[:len(y) - pos]
    y[pos:pos + len(out)] = out


def compress(x: np.ndarray, preset: Preset = BROADCAST, input_gain_db: float = 0.0,
             block_s: int = 60) -> np.ndarray:
    """Whole signal in blocks (memmap friendly); output has the same length as the input."""
    mbc = MultibandCompressor(preset, input_gain_db)
    out = [mbc.process(np.asarray(x[s:s + block_s * SR])) for s in range(0, len(x), block_s * SR)]
    out.append(mbc.flush())
    return np.concatenate(out)[:len(x)]
