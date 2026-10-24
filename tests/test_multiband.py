"""Multiband compressor (Audition "Broadcast" model)."""
import numpy as np
import soundfile as sf

from clarion import dynamics
from clarion import multiband as mb
from clarion.pipeline import Project
from clarion.settings import PRESETS, SR


def _noise(seconds, level, seed=0):
    return (np.random.default_rng(seed).standard_normal((int(seconds * SR), 2)) * level).astype(np.float32)


def test_bands_sum_flat_below_threshold():
    x = _noise(4, 0.001)                                  # −60 dBFS: no band compresses
    y = mb.compress(x)
    makeup = 10 ** ((mb.BROADCAST.margin_db - mb.BROADCAST.limiter_thresh_db) / 20)
    sx = np.abs(np.fft.rfft(x[SR:, 0])) * makeup
    sy = np.abs(np.fft.rfft(y[SR:, 0]))
    f = np.fft.rfftfreq(len(x) - SR, 1 / SR)
    for lo, hi, gain_db in ((300, 900, 0.0), (1500, 5000, 0.0), (9000, 16000, 3.0)):
        k = (f > lo) & (f < hi)
        ratio = 20 * np.log10(np.sqrt((sy[k] ** 2).mean() / (sx[k] ** 2).mean()))
        assert abs(ratio - gain_db) < 0.6, (lo, hi, ratio)   # allpass crossovers: flat apart from band gains


def test_loud_parts_are_compressed_quiet_parts_lifted():
    quiet, loud = _noise(3, 0.003, 1), _noise(3, 0.3, 2)
    y = mb.compress(np.concatenate([quiet, loud]))
    lift_q = dynamics.loudness(y[SR:3 * SR]) - dynamics.loudness(quiet[SR:])
    lift_l = dynamics.loudness(y[4 * SR:]) - dynamics.loudness(loud[SR:])
    assert lift_q > 9 and lift_l < 2                       # +9.9 dB make-up, loud part held back
    assert np.abs(y).max() <= 10 ** (mb.BROADCAST.margin_db / 20) + 1e-4


def test_inplace_blocks_match_one_pass():
    x = _noise(7.3, 0.05, 3)
    ref = mb.compress(x, block_s=60)
    y = x.copy()
    mb.compress_inplace(y, block_s=2)
    np.testing.assert_allclose(y, ref, atol=2e-4)


def test_export_with_broadcast_hits_target(reading_file, no_dfn, tmp_path):
    path, _ = reading_file
    proj = Project(path)
    proj.prepare()
    st = PRESETS["Gentle"].with_(loudness_target=-16.0, broadcast=True)
    y, _ = sf.read(proj.export(st, tmp_path / "bc.wav"), dtype="float32")
    assert abs(dynamics.loudness(y) - (-16.0)) < 1.0
    assert np.abs(y).max() <= 10 ** (-1 / 20) + 1e-3
    _, b = proj.write_preview(st, 0, 30)
    yb, _ = sf.read(b, dtype="float32")
    assert abs(dynamics.loudness(yb) - (-16.0)) < 1.5
