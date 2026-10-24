"""The two processing chains: speech (AI denoise + optional WPE) and music (gentle spectral NR)."""
from __future__ import annotations

import numpy as np
import scipy.signal as ss

from .settings import SR


def highpass(x: np.ndarray, fc: float) -> np.ndarray:
    if fc <= 0:
        return np.asarray(x, dtype=np.float32)
    sos = ss.butter(2, fc, "hp", fs=SR, output="sos")
    return ss.sosfiltfilt(sos, np.asarray(x), axis=0).astype(np.float32)


# --- speech ----------------------------------------------------------------------

def speech_chain(orig: np.ndarray, enhanced: np.ndarray, reduction_db: float, hp_hz: float,
                 dereverb: bool) -> np.ndarray:
    """Blend AI output with a little original (keeps it natural), high-pass, optional WPE."""
    g = 10 ** (-reduction_db / 20)
    y = highpass(enhanced, hp_hz) + g * highpass(orig, hp_hz)
    if dereverb:
        y = wpe_dereverb(y)
    return y


def wpe_dereverb(y: np.ndarray, taps: int = 12, delay: int = 3, iterations: int = 3) -> np.ndarray:
    """Weighted prediction error dereverberation, processed in 10 s chunks with crossfades."""
    from nara_wpe.utils import istft, stft
    from nara_wpe.wpe import wpe

    def one(seg):
        Y = np.stack([stft(seg[:, c].astype(np.float64), size=1024, shift=256)
                      for c in range(seg.shape[1])]).transpose(2, 0, 1)
        Z = wpe(Y, taps=taps, delay=delay, iterations=iterations, statistics_mode="full").transpose(1, 2, 0)
        return np.stack([istft(Z[c], size=1024, shift=256)[:len(seg)] for c in range(seg.shape[1])], 1)

    L, O = 10 * SR, SR
    if len(y) < 2 * O:
        return y
    z = np.zeros_like(y, dtype=np.float64)
    w = np.zeros(len(y))
    for s in range(0, len(y), L - O):
        seg = y[s:s + L]
        if len(seg) < 2 * O:  # tail too short for WPE statistics
            z[s:s + len(seg)] += seg
            w[s:s + len(seg)] += 1
            continue
        f = np.ones(len(seg))
        if s > 0:
            f[:O] = np.linspace(0, 1, O)
        if s + L < len(y):
            f[-O:] = np.linspace(1, 0, O)
        z[s:s + len(seg)] += one(seg) * f[:, None]
        w[s:s + len(seg)] += f
    return (z / np.maximum(w, 1e-6)[:, None]).astype(np.float32)


# --- music -----------------------------------------------------------------------

def noise_profile(x: np.ndarray, levels_db: np.ndarray, t0: float, t1: float,
                  radius_s: float = 150, n_frames: int = 16) -> np.ndarray:
    """Collect the quietest 0.25 s frames within ±radius around [t0, t1] as a noise sample."""
    frame = SR // 4
    lo, hi = max(0, int((t0 - radius_s) * 4)), min(len(levels_db), int((t1 + radius_s) * 4))
    if hi <= lo:
        lo, hi = 0, len(levels_db)
    cand = np.arange(lo, hi)
    pick = np.sort(cand[np.argsort(levels_db[cand])[:n_frames]])
    return np.concatenate([np.asarray(x[i * frame:(i + 1) * frame]) for i in pick])


def music_chain(x: np.ndarray, noise: np.ndarray, reduction: float, hp_hz: float) -> np.ndarray:
    """Stationary spectral gating with a fine frequency resolution suited to instruments."""
    import noisereduce as nr

    xh = highpass(x, hp_hz)
    if reduction <= 0:
        return xh
    nh = highpass(noise, hp_hz)
    y = nr.reduce_noise(y=xh.T, sr=SR, y_noise=nh.T, stationary=True, prop_decrease=float(reduction),
                        n_fft=4096, n_std_thresh_stationary=1.5, freq_mask_smooth_hz=200,
                        time_mask_smooth_ms=100, n_jobs=1)
    return np.asarray(y, dtype=np.float32).T[: len(x)]
