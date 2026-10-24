import numpy as np

from clarion import chains
from clarion.settings import SR


def _balance_db(x, frame=SR // 20):
    n = len(x) // frame
    r = np.sqrt((x[: n * frame].reshape(n, frame, 2) ** 2).mean(1)) + 1e-12
    return 20 * np.log10(r[:, 0] / r[:, 1])


def test_stereo_link_keeps_image_when_channels_are_suppressed_differently():
    rng = np.random.default_rng(1)
    n = 4 * SR
    voice = rng.standard_normal(n).astype(np.float32) * 0.1
    x = np.stack([voice, voice * 0.8], 1)  # fixed image: L is 1.9 dB louder
    # fake AI that attenuates L and R with independent, wandering gains (what deep-filter does)
    t = np.arange(n) / SR
    gl = 0.5 + 0.4 * np.sin(2 * np.pi * 1.3 * t)
    gr = 0.5 + 0.4 * np.sin(2 * np.pi * 0.7 * t + 1)
    e = x * np.stack([gl, gr], 1).astype(np.float32)

    assert _balance_db(e).std() > 3  # the problem: image wanders
    y = chains.stereo_link(x, e)
    assert y.shape == x.shape
    b = _balance_db(y)[5:-5]
    assert abs(b.mean() - 20 * np.log10(1 / 0.8)) < 0.2
    assert b.std() < 0.2
    # overall suppression follows the AI
    assert abs(np.sqrt((y ** 2).sum()) / np.sqrt((e ** 2).sum()) - 1) < 0.15


def test_stereo_link_is_identity_for_mono_like_input():
    rng = np.random.default_rng(2)
    v = rng.standard_normal(SR).astype(np.float32) * 0.1
    x = np.stack([v, v], 1)
    e = x * 0.3
    y = chains.stereo_link(x, e)
    assert np.allclose(y[2000:-2000], e[2000:-2000], atol=1e-4)


def test_hiss_cut_lowers_only_the_top():
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((4 * SR, 2)) * 0.05).astype(np.float32)
    y = chains.hiss_cut(x, -6.0)
    f, px = __import__("scipy.signal", fromlist=["welch"]).welch(x[:, 0], SR, nperseg=4096)
    _, py = __import__("scipy.signal", fromlist=["welch"]).welch(y[:, 0], SR, nperseg=4096)
    d = 10 * np.log10(py / px)
    assert abs(d[(f > 14000) & (f < 20000)].mean() - (-6.0)) < 0.7     # top: about -6 dB
    assert abs(d[(f > 100) & (f < 2000)].mean()) < 0.2                  # body: untouched
    np.testing.assert_allclose(chains.hiss_cut(x, 0.0), x)
