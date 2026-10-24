"""Tone and level matching for the voice ("Match tone", "Match level").

When a speaker moves (close to the PA microphone → the recorder hears mostly the loudspeaker: boomy
and harsh; normal position → direct voice), the voice's long-term spectrum changes a lot. This module
measures the AI voice's spectrum in third-octave bands over a sliding window and derives a slowly
varying EQ that pulls it towards a reference timbre: by default the driest stretches of the file
(where the AI removes least), or a range the user picks.

Everything works on the AI voice, so music under the voice does not disturb the measurement.
"""
from __future__ import annotations

import numpy as np
import scipy.signal as ss

from .settings import SR

N_FFT = 4096
HOP = 1024
CENTRES = np.array([80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500,
                    3150, 4000, 5000, 6300, 8000, 10000, 12500], float)
_EDGES = np.stack([CENTRES / 2 ** (1 / 6), CENTRES * 2 ** (1 / 6)], 1)
_FREQS = np.fft.rfftfreq(N_FFT, 1 / SR)


def _band_matrix() -> np.ndarray:
    """(bands, bins) averaging matrix."""
    m = np.zeros((len(CENTRES), len(_FREQS)), np.float32)
    for i, (a, b) in enumerate(_EDGES):
        sel = (_FREQS >= a) & (_FREQS < b)
        m[i, sel] = 1.0 / max(1, sel.sum())
    return m


def _interp_matrix() -> np.ndarray:
    """(bins, bands) linear interpolation in log frequency from band centres to FFT bins."""
    lf = np.log(np.maximum(_FREQS, 1.0))
    lc = np.log(CENTRES)
    w = np.zeros((len(_FREQS), len(CENTRES)), np.float32)
    for k, x in enumerate(lf):
        if x <= lc[0]:
            w[k, 0] = 1
        elif x >= lc[-1]:
            w[k, -1] = 1
        else:
            j = np.searchsorted(lc, x) - 1
            t = (x - lc[j]) / (lc[j + 1] - lc[j])
            w[k, j], w[k, j + 1] = 1 - t, t
    return w


_BANDS = _band_matrix()
_INTERP = _interp_matrix()


def band_powers(voice: np.ndarray, block_s: int = 60) -> np.ndarray:
    """Third-octave band power per STFT frame (HOP samples) of a mono or stereo voice signal.

    Frame i covers samples centred on i*HOP. Computed blockwise, so it works on long memmaps.
    """
    n = len(voice)
    n_frames = (n - 1) // HOP + 1            # frames whose centre lies inside the signal
    out = np.zeros((n_frames, len(CENTRES)), np.float32)
    blk = block_s * SR
    pad = N_FFT
    for s in range(0, n, blk):
        e = min(n, s + blk)
        a, b = (max(0, s - pad) // HOP) * HOP, min(n, e + pad)     # a on the frame grid
        x = np.asarray(voice[a:b], dtype=np.float32)
        x = x.mean(axis=1) if x.ndim == 2 else x
        _, _, Z = ss.stft(x, fs=SR, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP, boundary="zeros")
        P = (np.abs(Z) ** 2).astype(np.float32)               # (bins, frames); frame j centred on a + j*HOP
        f0 = -(-s // HOP)                                     # first frame index centred in [s, e)
        f1 = min(n_frames, (e - 1) // HOP + 1)
        j0 = f0 - a // HOP
        cols = np.arange(j0, j0 + (f1 - f0))
        cols = cols[cols < P.shape[1]]
        out[f0:f0 + len(cols)] = (_BANDS @ P[:, cols]).T
    return out


def _windowed_shape(bp: np.ndarray, active: np.ndarray, win_frames: int, min_active: int) -> np.ndarray:
    """Spectral shape (dB, level removed) of active frames within ±win/2 of every frame; NaN if too few."""
    w = active.astype(np.float32)
    num = np.cumsum(np.vstack([np.zeros((1, bp.shape[1]), np.float32), bp * w[:, None]]), axis=0)
    den = np.cumsum(np.concatenate([[0.0], w]))
    n = len(bp)
    i = np.arange(n)
    lo, hi = np.clip(i - win_frames // 2, 0, n), np.clip(i + win_frames // 2 + 1, 0, n)
    cnt = den[hi] - den[lo]
    mean = (num[hi] - num[lo]) / np.maximum(cnt, 1)[:, None]
    db = 10 * np.log10(mean + 1e-20)
    db -= 10 * np.log10(mean.sum(axis=1, keepdims=True) + 1e-20)   # shape only, independent of level
    db[cnt < min_active] = np.nan
    return db


def _fill_nan(x: np.ndarray) -> np.ndarray:
    """Fill NaN rows by linear interpolation in time (holding the ends)."""
    ok = ~np.isnan(x[:, 0])
    if not ok.any():
        return np.zeros_like(x)
    idx = np.arange(len(x))
    return np.stack([np.interp(idx, idx[ok], x[ok, b]) for b in range(x.shape[1])], 1)


def voice_active(lvl_db: np.ndarray, radius_s: float = 10.0, below_db: float = 15.0) -> np.ndarray:
    """Frames where the voice is clearly sounding, judged against its *local* loudness.

    A global threshold would drop most frames of a quiet section (speaker far from the microphone),
    exactly where tone and level matching are needed. Here a frame counts if it is within ``below_db``
    of the 90th percentile of the level within ±``radius_s``.
    """
    n = len(lvl_db)
    fps = SR / HOP
    r, step = int(radius_s * fps), max(1, int(0.5 * fps))
    pts = np.arange(0, n, step)
    ref = np.array([np.percentile(lvl_db[max(0, i - r):i + r + 1], 90) for i in pts])
    return lvl_db > np.interp(np.arange(n), pts, ref) - below_db


def correction_curve(bp: np.ndarray, dryness: np.ndarray, strength: float, max_db: float,
                     ref: tuple[float, float] | None = None, win_s: float = 6.0,
                     voiced: np.ndarray | None = None) -> np.ndarray:
    """EQ gain in dB per frame and band that pulls the voice towards the reference timbre.

    ``dryness`` (per frame, higher = drier, e.g. AI output minus original in dB) picks the automatic
    reference: the driest 30 % of 6 s windows. ``ref`` (seconds) overrides it. ``voiced`` (bool per
    frame) limits the measurement to speech sections: in music sections the AI output is only a faint
    residue, which must not count as voice.
    """
    tot = bp.sum(axis=1)
    lvl = 10 * np.log10(tot + 1e-20)
    if not np.isfinite(lvl).any():
        return np.zeros_like(bp)
    active = voice_active(lvl)
    if voiced is not None:
        active &= voiced[: len(active)]
    fps = SR / HOP
    win, min_act = int(win_s * fps), int(1.5 * fps)
    shape = _windowed_shape(bp, active, win, min_act)
    valid = ~np.isnan(shape[:, 0])
    if valid.sum() == 0:
        return np.zeros_like(bp)
    if ref is not None:
        a, b = int(ref[0] * fps), int(ref[1] * fps)
        sel = np.zeros(len(bp), bool)
        sel[max(0, a):max(0, b)] = True
        sel &= active
        if sel.sum() < min_act:
            ref = None
        else:
            m = bp[sel].mean(axis=0)
            target = 10 * np.log10(m + 1e-20) - 10 * np.log10(m.sum() + 1e-20)
    if ref is None:
        d = np.where(active, dryness[: len(bp)], np.nan)
        # windowed mean dryness over active frames
        dd = np.nan_to_num(d)
        c = np.cumsum(np.concatenate([[0.0], dd]))
        k = np.cumsum(np.concatenate([[0.0], active.astype(float)]))
        i = np.arange(len(bp))
        lo, hi = np.clip(i - win // 2, 0, len(bp)), np.clip(i + win // 2 + 1, 0, len(bp))
        wd = (c[hi] - c[lo]) / np.maximum(k[hi] - k[lo], 1)
        wd[~valid] = np.nan
        thr = np.nanpercentile(wd, 70)
        target = np.nanmedian(shape[wd >= thr], axis=0)
    corr = target[None, :] - _fill_nan(shape)
    corr = np.clip(corr, -max_db, max_db) * strength
    # smooth across neighbouring bands (no narrow peaks)
    k3 = np.array([0.25, 0.5, 0.25])
    corr = np.stack([np.convolve(np.pad(r, 1, mode="edge"), k3, "valid") for r in corr])
    return corr.astype(np.float32)


# rough loudness weighting per band (K-weighting-like: less bass, a little more presence)
_LOUD_W = 10 ** (np.interp(CENTRES, [80, 100, 125, 160, 200, 250, 1000, 1250, 1600, 2000, 2500, 16000],
                           [-12, -8, -5, -3, -1.5, 0, 0, 1, 2.5, 3.5, 4, 4]) / 10).astype(np.float32)


def level_curve(bp: np.ndarray, eq_db: np.ndarray | None, strength: float, max_db: float,
                win_s: float = 6.0, voiced: np.ndarray | None = None) -> np.ndarray:
    """Gain in dB per frame that evens out the voice's loudness (speaker closer / further away).

    Loudness = weighted band power of loud voice frames within a sliding window, measured after the
    tone EQ (``eq_db``) so both work together. Target: the median over the file. The gain is
    clipped to ±``max_db`` and scaled by ``strength``; pauses take the neighbouring values.
    ``voiced`` (bool per frame) limits the measurement to speech sections. Without it the faint AI
    residue in music sections pulled the target down: 01.wav's speech was lowered by 7–10 dB.
    """
    b = bp * (10 ** (eq_db / 10) if eq_db is not None else 1.0)
    lvl = 10 * np.log10((b * _LOUD_W).sum(axis=1) + 1e-20)
    if not np.isfinite(lvl).any() or len(lvl) == 0:
        return np.zeros(len(bp), np.float32)
    active = voice_active(lvl)
    if voiced is not None:
        active &= voiced[: len(active)]
    fps = SR / HOP
    win, min_act = int(win_s * fps), int(1.5 * fps)
    p = np.where(active, 10 ** (lvl / 10), 0.0)
    c = np.cumsum(np.concatenate([[0.0], p]))
    k = np.cumsum(np.concatenate([[0.0], active.astype(float)]))
    i = np.arange(len(lvl))
    lo, hi = np.clip(i - win // 2, 0, len(lvl)), np.clip(i + win // 2 + 1, 0, len(lvl))
    cnt = k[hi] - k[lo]
    loud = 10 * np.log10((c[hi] - c[lo]) / np.maximum(cnt, 1) + 1e-20)
    loud[cnt < min_act] = np.nan
    if np.isnan(loud).all():
        return np.zeros(len(bp), np.float32)
    target = np.nanmedian(loud)
    loud = _fill_nan(loud[:, None])[:, 0]
    return (np.clip(target - loud, -max_db, max_db) * strength).astype(np.float32)


def apply_eq(x: np.ndarray, curve_db: np.ndarray, start: int) -> np.ndarray:
    """Apply the time-varying band EQ ``curve_db`` (frames × bands, frame i at sample i*HOP) to x.

    ``start`` is the absolute sample position of x[0] (so curve frames line up).
    """
    x = np.asarray(x, dtype=np.float32)
    if len(x) < N_FFT or not np.any(curve_db):
        return x
    mono = x.ndim == 1
    if mono:  # always work on (samples, channels); a 1-D input would broadcast the gains into a huge array
        x = x[:, None]
    _, _, Z = ss.stft(x, fs=SR, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP, axis=0)  # (bins, ch, fr)
    n_fr = Z.shape[-1]
    # scipy pads N_FFT/2 on both sides, so frame j is centred on x[j*HOP], i.e. on start + j*HOP
    t = (start + np.arange(n_fr) * HOP) / HOP
    idx = np.clip(np.round(t).astype(int), 0, len(curve_db) - 1)
    g_bands = curve_db[idx]                                          # (frames, bands)
    g = 10 ** ((_INTERP @ g_bands.T) / 20)                           # (bins, frames)
    _, y = ss.istft(Z * g[:, None, :], fs=SR, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP,
                    time_axis=-1, freq_axis=0)
    y = np.asarray(y).T[: len(x)].astype(np.float32)     # istft gives (channels, samples)
    return y[:, 0] if mono else y
