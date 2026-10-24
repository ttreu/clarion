import os
import shutil

import numpy as np
import pytest
import soundfile as sf

from klangrestaurator import audio_io
from klangrestaurator.analysis import MUSIC
from klangrestaurator.pipeline import Project
from klangrestaurator.settings import PRESETS, SR


def test_load_resamples_to_48k_stereo(tmp_path):
    x = np.sin(2 * np.pi * 440 * np.arange(44_100 * 3) / 44_100).astype(np.float32) * 0.5
    sf.write(tmp_path / "mono.wav", x, 44_100)
    mm = audio_io.load_to_memmap(tmp_path / "mono.wav", tmp_path / "o.npy")
    assert mm.shape == (3 * SR, 2)
    assert abs(np.abs(mm[SR:2 * SR]).max() - 0.5) < 0.01


def test_end_to_end_with_fake_ai(reading_file, tmp_path, no_dfn):
    src, music = reading_file
    proj = Project(src, workdir=tmp_path / "work")
    proj.prepare(PRESETS["Natural"])
    found = [s for s in proj.segments if s.kind == MUSIC]
    assert len(found) == 1
    assert abs(found[0].start - 10) < 1.0 and abs(found[0].end - 20) < 1.5

    out = proj.export(PRESETS["Natural"], tmp_path / "out.wav")
    y, sr = sf.read(out)
    assert sr == SR and sf.info(out).subtype == "PCM_24"
    assert len(y) == 30 * SR
    assert np.isfinite(y).all()
    from klangrestaurator import dynamics
    assert abs(dynamics.loudness(y) - PRESETS["Natural"].loudness_target) < 1.0   # default: LUFS target
    assert 20 * np.log10(np.abs(y).max()) <= -1.0 + 1e-3
    out = proj.export(PRESETS["Natural"].with_(loudness_target=None), tmp_path / "peak.wav")
    y, _ = sf.read(out)
    assert abs(20 * np.log10(np.abs(y).max()) - (-1.0)) < 0.1                       # peak mode as before

    # cached: second prepare must not re-run the AI
    proj2 = Project(src, workdir=tmp_path / "work")
    proj2.prepare(PRESETS["Natural"])
    assert len(proj2.segments) == len(proj.segments)


def test_preview_and_keep_rate(reading_file, tmp_path, no_dfn):
    src, _ = reading_file
    proj = Project(src, workdir=tmp_path / "work")
    proj.prepare()
    a, b = proj.write_preview(PRESETS["Natural"], 8, 12)
    assert sf.info(a).duration == pytest.approx(4, abs=0.01)
    assert sf.info(b).duration == pytest.approx(4, abs=0.01)
    out = proj.export(PRESETS["Gentle"], tmp_path / "o44.wav", out_sr=44_100)
    assert sf.info(out).samplerate == 44_100


@pytest.mark.slow
@pytest.mark.skipif(not (os.environ.get("KLANGRESTAURATOR_DEEPFILTER") or shutil.which("curl")),
                    reason="needs deep-filter binary or network")
def test_real_deepfilternet(reading_file, tmp_path):
    src, _ = reading_file
    proj = Project(src, workdir=tmp_path / "work")
    proj.prepare()
    y = proj.render(PRESETS["Natural"], 0, 5)
    assert np.isfinite(y).all()


def test_mix_segment_is_voice_plus_share_of_music(reading_file, no_dfn):
    from klangrestaurator.analysis import MIX, SPEECH
    path, _ = reading_file
    proj = Project(path)
    proj.prepare()
    lvl = lambda x: 10 * np.log10((x ** 2).mean())  # noqa: E731
    st = PRESETS["Natural"].with_(tone_match=0.0, level_match=0.0)   # check the mix formula alone
    y_music = proj.render(st, 12, 18)          # 10–20 s is detected as music
    proj.set_kind(10, 20, SPEECH)              # as plain speech the AI wipes the music
    y_speech = proj.render(st, 12, 18)
    proj.set_kind(10, 20, MIX)
    same_nr = st.with_(mix_reduction=st.music_reduction)
    y_full = proj.render(same_nr.with_(mix_music=1.0), 12, 18)
    y_half = proj.render(same_nr.with_(mix_music=0.5), 12, 18)
    assert lvl(y_music) - lvl(y_speech) > 10
    assert abs(lvl(y_half) - (lvl(y_music) - 6)) < 1    # music share 50 % → music about 6 dB down
    assert lvl(y_full) >= lvl(y_music) - 0.1            # 100 %: full music (plus the dry voice)
    y_default = proj.render(st, 12, 18)                 # stronger NR on the music share costs a little
    assert lvl(y_half) - 3 < lvl(y_default) <= lvl(y_half) + 0.1


def test_adjacent_music_and_mix_are_not_denoised_twice(reading_file, no_dfn):
    from klangrestaurator.analysis import MIX
    path, _ = reading_file
    proj = Project(path)
    proj.prepare()
    st = PRESETS["Natural"].with_(mix_music=1.0)
    ref = proj.render(st, 14.5, 15.5)          # inside one music section
    proj.set_kind(15, 20, MIX)                 # music 10–15 | mix 15–20, boundary at 15 s
    y = proj.render(st.with_(mix_fade_s=0.01), 14.5, 14.9)  # just before the boundary: still music
    np.testing.assert_allclose(y, ref[: len(y)], atol=2e-3)


def test_export_hits_loudness_target_under_limiter(reading_file, no_dfn, tmp_path):
    from klangrestaurator import dynamics
    path, _ = reading_file
    proj = Project(path)
    proj.prepare()
    dest = tmp_path / "out.wav"
    proj.export(PRESETS["Natural"].with_(loudness_target=-16.0), dest)
    y, sr = sf.read(dest, dtype="float32")
    assert abs(dynamics.loudness(y) - (-16.0)) < 1.0
    assert np.abs(y).max() <= 10 ** (-1 / 20) + 1e-3
    a, b = proj.write_preview(PRESETS["Natural"].with_(loudness_target=-16.0), 0, 30)
    yb, _ = sf.read(b, dtype="float32")
    assert abs(dynamics.loudness(yb) - (-16.0)) < 1.5          # preview: estimate of the export level


def test_normalize_switch_off_restores_old_behaviour(reading_file, no_dfn, tmp_path):
    from klangrestaurator import dynamics
    path, _ = reading_file
    proj = Project(path)
    proj.prepare()
    on = PRESETS["Natural"]
    off = on.with_(normalize=False)
    assert off.effective().loudness_target is None and on.effective() is on
    y, _ = sf.read(proj.export(off, tmp_path / "off.wav"))
    assert abs(20 * np.log10(np.abs(y).max()) - (-1.0)) < 0.1          # peak normalisation only
    y, _ = sf.read(proj.export(on.with_(loudness_target=-14.0), tmp_path / "on.wav"))
    assert abs(dynamics.loudness(y) - (-14.0)) < 1.0
    assert np.abs(y).max() <= 10 ** (-1 / 20) + 1e-3


def test_trim_limits_export_with_soft_edges(reading_file, no_dfn, tmp_path):
    path, _ = reading_file
    proj = Project(path)
    proj.prepare()
    st = PRESETS["Gentle"].with_(normalize=False)     # fast (no WPE)
    proj.set_trim((5.0, 12.5))
    assert proj.trim == (5.0, 12.5)
    y, _ = sf.read(proj.export(st, tmp_path / "cut.wav"))
    assert len(y) == int(12.5 * SR) - int(5.0 * SR)
    assert np.abs(y[:5]).max() < 0.05 * np.abs(y).max()      # fades in, no hard start
    assert np.abs(y[-5:]).max() < 0.05 * np.abs(y).max()
    with pytest.raises(ValueError):
        proj.set_trim((3.0, 3.2))                               # too short
    proj.set_trim(None)
    assert proj.trim is None
    y, _ = sf.read(proj.export(st, tmp_path / "full.wav"))
    assert len(y) == 30 * SR


def test_loud_music_is_capped_above_the_loudness_target(tmp_path, no_dfn):
    """Mostly quiet speech + a short, much louder music part (like ZOOM0002.WAV): after normalisation
    the music must stay at most 3 LU above the target, the speech keeps its level."""
    from conftest import synth_reading

    from klangrestaurator import dynamics
    x, _ = synth_reading()                                     # speech 0–10, music 10–20, speech 20–30
    sr = 44100
    x[10 * sr:20 * sr] *= 4.0                                  # the music is much louder
    x = np.concatenate([x, np.tile(x[20 * sr:30 * sr], (5, 1))])   # + 50 s more speech: speech dominates
    path = tmp_path / "laut.wav"
    sf.write(path, x * 0.2, sr, subtype="PCM_16")
    proj = Project(path)
    proj.prepare()
    st = PRESETS["Gentle"].with_(loudness_target=-16.0)        # fast (no WPE)
    y, _ = sf.read(proj.export(st, tmp_path / "out.wav"), dtype="float32")
    y_off, _ = sf.read(proj.export(st.with_(music_ceiling_lu=None), tmp_path / "off.wav"), dtype="float32")
    mus = slice(int(12 * SR), int(18 * SR))
    on, off = dynamics.short_term_lufs(y[mus]).max(), dynamics.short_term_lufs(y_off[mus]).max()
    assert off > -16.0 + 3.0 + 2.0                             # without the cap the music is far too loud
    assert on <= -16.0 + 3.0 + 0.5                             # with it: at most target + 3 LU
    sp = slice(int(40 * SR), int(70 * SR))                     # speech keeps its level
    assert abs(dynamics.loudness(y[sp]) - dynamics.loudness(y_off[sp])) < 0.5
