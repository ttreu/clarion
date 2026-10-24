import numpy as np
import pytest
import soundfile as sf

from klangrestaurator.settings import SR


def synth_reading(sr: int = 44_100, seed: int = 0) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """30 s: bursty 'speech' 0–10 s, sustained 'cello' 10–20 s, 'speech' 20–30 s, plus hiss."""
    rng = np.random.default_rng(seed)
    t = np.arange(30 * sr) / sr
    x = np.zeros_like(t)
    syll = (np.sin(2 * np.pi * 4 * t) > 0.2).astype(float)  # 4 Hz syllable gating
    speech = rng.standard_normal(len(t)) * 0.08 * syll
    sp = (t < 10) | (t >= 20)
    x[sp] = speech[sp]
    mu = (t >= 10) & (t < 20)
    for k, a in ((1, 0.2), (2, 0.1), (3, 0.06), (4, 0.03)):
        x[mu] += a * np.sin(2 * np.pi * 110 * k * t[mu])
    x += rng.standard_normal(len(t)) * 0.003  # hiss ≈ -50 dBFS
    return np.stack([x, x * 0.9], 1).astype(np.float32), [(10.0, 20.0)]


def fake_enhance(x, out, start=0, stop=None, progress=None, cancel=None, workers=None):
    """Stand-in for DeepFilterNet: keeps 'speech', removes the harmonic part by ~40 dB."""
    stop = len(x) if stop is None else stop
    y = np.array(x[start:stop])
    t0 = np.arange(start, stop) / SR
    music = (t0 >= 10) & (t0 < 20)
    y[music] *= 0.01
    y[~music] *= 0.7
    out[start:stop] = y
    if progress:
        progress(1.0)


@pytest.fixture
def reading_file(tmp_path):
    x, music = synth_reading()
    p = tmp_path / "lesung.wav"
    sf.write(p, x, 44_100, subtype="PCM_16")
    return p, music


@pytest.fixture
def no_dfn(monkeypatch):
    from klangrestaurator import dfn
    monkeypatch.setattr(dfn, "enhance", fake_enhance)
    monkeypatch.setattr(dfn, "ensure_binary", lambda progress=None: None)
