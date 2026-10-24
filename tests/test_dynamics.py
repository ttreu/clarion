import numpy as np
import scipy.signal as ss
import soundfile as sf

from klangrestaurator import audio_io, dynamics
from klangrestaurator.settings import SR


def _sine(f, amp, seconds, ch=2):
    t = np.arange(int(seconds * SR)) / SR
    x = (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)
    return np.stack([x] * ch, 1)


def test_loudness_matches_bs1770_reference():
    # BS.1770: a 997 Hz sine at 0 dBFS in one channel reads -3.01 LUFS → at -20 dBFS: -23.01 LUFS
    x = _sine(997, 0.1, 10)
    x[:, 1] = 0
    assert abs(dynamics.loudness(x) - (-23.01)) < 0.1
    # blockwise measurement on long input gives the same value
    assert abs(dynamics.loudness(x, block_s=1) - dynamics.loudness(x)) < 0.01
    assert dynamics.loudness(np.zeros((SR, 2), np.float32)) == float("-inf")


def test_limiter_caps_peaks_and_leaves_quiet_audio_alone():
    rng = np.random.default_rng(0)
    loud = (rng.standard_normal((5 * SR, 2)) * 0.5).astype(np.float32)   # peaks around +6 dBFS
    y = dynamics.limit(loud, -1.0)
    assert y.shape == loud.shape
    assert np.abs(y).max() <= 10 ** (-1 / 20) + 1e-6
    quiet = loud * 0.05
    np.testing.assert_allclose(dynamics.limit(quiet, -1.0), quiet, atol=1e-6)
    # feeding blocks of odd sizes gives the same result as one go
    lim = dynamics.Limiter(-1.0)
    parts = [lim.process(loud[a:b]) for a, b in ((0, 12345), (12345, 100000), (100000, len(loud)))]
    parts.append(lim.flush())
    np.testing.assert_allclose(np.concatenate(parts), y, atol=1e-6)


def test_stream_writer_resamples_without_seams(tmp_path):
    x = _sine(1000, 0.5, 3)
    dest = tmp_path / "o.wav"
    w = audio_io.StreamWriter(dest, 2, out_sr=44100, chunk_s=1)
    for a in range(0, len(x), 17000):             # odd block sizes across the 1 s chunks
        w.write(x[a:a + 17000])
    w.close()
    y, sr = sf.read(dest, dtype="float32")
    ref = ss.resample_poly(x, 147, 160, axis=0)
    assert sr == 44100 and len(y) == len(ref)
    assert np.abs(y[2000:-2000] - ref[2000:-2000]).max() < 1e-3      # no clicks at chunk borders

