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

def stereo_link(orig: np.ndarray, enhanced: np.ndarray, n_fft: int = 1024, max_gain: float = 1.5) -> np.ndarray:
    """Re-apply the AI's noise suppression with one gain shared by both channels.

    deep-filter treats left and right independently, so each channel gets its own time-varying
    attenuation and the voice drifts left/right once little original is mixed back. Here the gain per
    time-frequency bin is taken from the summed power of both AI channels and applied equally to the
    original L and R, which keeps the original stereo image.
    """
    x = np.asarray(orig, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] < 2 or len(x) < n_fft:
        return np.asarray(enhanced, dtype=np.float32)
    kw = dict(fs=SR, window="hann", nperseg=n_fft, noverlap=n_fft * 3 // 4, axis=0)
    _, _, X = ss.stft(x, **kw)                                    # (freq, ch, frames)
    _, _, E = ss.stft(np.asarray(enhanced, dtype=np.float32), **kw)
    px = (np.abs(X) ** 2).sum(axis=1, keepdims=True)
    pe = (np.abs(E) ** 2).sum(axis=1, keepdims=True)
    g = np.minimum(np.sqrt(pe / (px + 1e-12)), max_gain)
    _, y = ss.istft(X * g, fs=SR, window="hann", nperseg=n_fft, noverlap=n_fft * 3 // 4,
                    time_axis=-1, freq_axis=0)
    y = np.asarray(y).T if np.asarray(y).shape[0] == x.shape[1] else np.asarray(y)
    return y[: len(x)].astype(np.float32)


def speech_chain(orig: np.ndarray, enhanced: np.ndarray, reduction_db: float, hp_hz: float,
                 dereverb: bool, stereo_linked: bool = True) -> np.ndarray:
    """Blend AI output with a little original (keeps it natural), high-pass, optional WPE."""
    if stereo_linked:
        enhanced = stereo_link(orig, enhanced)
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


def hiss_cut(x: np.ndarray, db: float, f0: float = 8000.0) -> np.ndarray:
    """Lower everything above ``f0`` by ``db`` (≤ 0) with a gentle, zero-phase shelf.

    Instruments like the cello have little energy up there, the tape/mic hiss a lot.
    """
    if db >= 0:
        return np.asarray(x, dtype=np.float32)
    sos = ss.butter(2, f0, "hp", fs=SR, output="sos")
    hi = ss.sosfiltfilt(sos, np.asarray(x), axis=0)
    return (x - (1 - 10 ** (db / 20)) * hi).astype(np.float32)


def _gate_pauses(x: np.ndarray, below_db: float = 22.0, ratio: float = 2.0) -> np.ndarray:
    """Gentle downward expander: lowers what is more than ``below_db`` under the loud parts
    (the hiss between notes); smoothed over ~100 ms so note tails fade rather than chop."""
    f = SR // 200
    n = len(x) // f * f
    if n < 10 * f:
        return np.asarray(x, dtype=np.float32)
    lv = 10 * np.log10((np.asarray(x[:n]) ** 2).mean(axis=1).reshape(-1, f).mean(axis=1) + 1e-12)
    lv = ss.medfilt(lv, 5)
    ref = np.percentile(lv, 80) - below_db
    gdb = np.minimum(0.0, (lv - ref) * (ratio - 1))
    gdb = ss.filtfilt(np.ones(20) / 20, [1.0], gdb)
    g = 10 ** (np.interp(np.arange(len(x)), (np.arange(len(gdb)) + 0.5) * f, gdb) / 20)
    return (x * g[:, None]).astype(np.float32)


def music_chain(x: np.ndarray, noise: np.ndarray, reduction: float, hp_hz: float,
                hiss_db: float = 0.0, n_std: float = 1.5, second_pass: float = 0.0,
                gate: bool = False) -> np.ndarray:
    """Stationary spectral gating with a fine frequency resolution suited to instruments, then the
    optional hiss shelf. Variants from the listening tests (settings.MUSIC_VARIANTS): a higher
    threshold ``n_std``, a ``second_pass`` of NR, and a ``gate`` between notes.
    Chosen by ear on 01.wav: 85 % + −6 dB above 8 kHz."""
    import noisereduce as nr

    xh = highpass(x, hp_hz)
    if reduction <= 0:
        return hiss_cut(xh, hiss_db)
    nh = highpass(noise, hp_hz)

    def nr_pass(z, prop):
        y = nr.reduce_noise(y=z.T, sr=SR, y_noise=nh.T, stationary=True, prop_decrease=float(prop),
                            n_fft=4096, n_std_thresh_stationary=n_std, freq_mask_smooth_hz=200,
                            time_mask_smooth_ms=100, n_jobs=1)
        return np.asarray(y, dtype=np.float32).T[: len(x)]

    y = nr_pass(xh, reduction)
    if second_pass > 0:
        y = nr_pass(y, second_pass)
    if gate:
        y = _gate_pauses(y)
    return hiss_cut(y, hiss_db)
