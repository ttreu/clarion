import numpy as np
import scipy.signal as ss

from klangrestaurator import tone
from klangrestaurator.settings import SR


def _voice(seconds, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    gate = (np.sin(2 * np.pi * 3 * t) > -0.3).astype(np.float32)          # syllables with short gaps
    return (rng.standard_normal(len(t)) * 0.05 * gate).astype(np.float32)


def _shape(bp):
    m = bp.mean(axis=0)
    return 10 * np.log10(m + 1e-20) - 10 * np.log10(m.sum())


def test_apply_eq_identity_and_constant_gain():
    x = np.stack([_voice(3), _voice(3, 1)], 1)
    n_fr = len(x) // tone.HOP + 1
    zero = np.zeros((n_fr, len(tone.CENTRES)), np.float32)
    np.testing.assert_allclose(tone.apply_eq(x, zero, 0), x)
    six = zero + 6.0
    y = tone.apply_eq(x, six, 0)
    sl = slice(SR // 2, -SR // 2)
    assert abs(20 * np.log10(np.std(y[sl]) / np.std(x[sl])) - 6.0) < 0.2


def test_correction_pulls_coloured_part_to_reference():
    # 0–20 s: "loudspeaker" colour (boomy + harsh), 20–40 s: flat, drier voice = reference
    a, b = _voice(20, 2), _voice(20, 3)
    sos_low = ss.butter(2, [100, 300], "bp", fs=SR, output="sos")
    sos_hi = ss.butter(2, [2500, 5000], "bp", fs=SR, output="sos")
    coloured = a + 3 * ss.sosfilt(sos_low, a) + 3 * ss.sosfilt(sos_hi, a)
    v = np.concatenate([coloured, b]).astype(np.float32)
    bp = tone.band_powers(v)
    fps = SR / tone.HOP
    dryness = np.where(np.arange(len(bp)) / fps < 20, -8.0, -3.0)       # second half: AI removes less
    curve = tone.correction_curve(bp, dryness, strength=1.0, max_db=12.0)
    y = tone.apply_eq(v, curve, 0)
    bq = tone.band_powers(y)
    i0, i1, j0 = int(4 * fps), int(16 * fps), int(24 * fps)
    before = np.sqrt(np.mean((_shape(bp[i0:i1]) - _shape(bp[j0:])) ** 2))
    after = np.sqrt(np.mean((_shape(bq[i0:i1]) - _shape(bq[j0:])) ** 2))
    assert before > 3 and after < before / 2
    assert np.abs(curve[j0:]).mean() < 1.0                               # reference part left alone


def test_manual_reference_overrides_dryness():
    a, b = _voice(10, 4), _voice(10, 5)
    sos = ss.butter(2, 2000, "hp", fs=SR, output="sos")
    v = np.concatenate([a, ss.sosfilt(sos, b).astype(np.float32) * 3]).astype(np.float32)
    bp = tone.band_powers(v)
    dry = np.zeros(len(bp))
    auto = tone.correction_curve(bp, dry, 1.0, 12.0)
    manual = tone.correction_curve(bp, dry, 1.0, 12.0, ref=(12.0, 18.0))  # the bright second half
    fps = SR / tone.HOP
    first, second = slice(int(2 * fps), int(8 * fps)), slice(int(12 * fps), int(18 * fps))

    def dist_to_ref(curve):  # timbre distance of the EQ'd first half to the bright second half
        bq = tone.band_powers(tone.apply_eq(v, curve, 0))
        return np.sqrt(np.mean((_shape(bq[first]) - _shape(bp[second])) ** 2))

    assert dist_to_ref(manual) < dist_to_ref(auto) - 1


def test_apply_eq_mono_keeps_shape():
    x = _voice(2)
    n_fr = len(x) // tone.HOP + 1
    y = tone.apply_eq(x, np.full((n_fr, len(tone.CENTRES)), 3.0, np.float32), 0)
    assert y.shape == x.shape


def test_level_curve_evens_out_quiet_and_loud_parts():
    v = np.concatenate([_voice(15, 6) * 0.25, _voice(15, 7)]).astype(np.float32)   # 12 dB quieter first
    bp = tone.band_powers(v)
    g = tone.level_curve(bp, None, strength=1.0, max_db=10.0)
    y = tone.apply_eq(v, np.repeat(g[:, None], len(tone.CENTRES), 1), 0)
    q, l_ = slice(3 * SR, 12 * SR), slice(18 * SR, 27 * SR)
    before = 20 * np.log10(np.std(v[l_]) / np.std(v[q]))
    after = 20 * np.log10(np.std(y[l_]) / np.std(y[q]))
    assert before > 11 and abs(after) < 3
    assert np.all(np.abs(g) <= 10.0 + 1e-6)
    assert np.allclose(tone.level_curve(bp, None, 0.0, 10.0), 0)


def test_level_curve_ignores_music_sections():
    """In music sections the AI output is a faint residue; it must not drag the voice target down."""
    voice = _voice(10, 8)
    residue = _voice(20, 9) * 10 ** (-40 / 20)          # what the AI leaves of a cello
    v = np.concatenate([voice, residue]).astype(np.float32)
    bp = tone.band_powers(v)
    fps = SR / tone.HOP
    voiced = np.arange(len(bp)) / fps < 10
    sl = slice(int(2 * fps), int(8 * fps))
    g_bad = tone.level_curve(bp, None, 1.0, 10.0)
    g_ok = tone.level_curve(bp, None, 1.0, 10.0, voiced=voiced)
    assert g_bad[sl].mean() < -5                          # the old behaviour: speech pushed down
    assert abs(g_ok[sl].mean()) < 1                       # fixed: speech stays where it is
