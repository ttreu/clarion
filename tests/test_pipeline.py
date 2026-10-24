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
    proj.prepare(PRESETS["Natürlich"])
    found = [s for s in proj.segments if s.kind == MUSIC]
    assert len(found) == 1
    assert abs(found[0].start - 10) < 1.0 and abs(found[0].end - 20) < 1.5

    out = proj.export(PRESETS["Natürlich"], tmp_path / "out.wav")
    y, sr = sf.read(out)
    assert sr == SR and sf.info(out).subtype == "PCM_24"
    assert len(y) == 30 * SR
    assert np.isfinite(y).all()
    assert abs(20 * np.log10(np.abs(y).max()) - (-1.0)) < 0.1

    # cached: second prepare must not re-run the AI
    proj2 = Project(src, workdir=tmp_path / "work")
    proj2.prepare(PRESETS["Natürlich"])
    assert len(proj2.segments) == len(proj.segments)


def test_preview_and_keep_rate(reading_file, tmp_path, no_dfn):
    src, _ = reading_file
    proj = Project(src, workdir=tmp_path / "work")
    proj.prepare()
    a, b = proj.write_preview(PRESETS["Natürlich"], 8, 12)
    assert sf.info(a).duration == pytest.approx(4, abs=0.01)
    assert sf.info(b).duration == pytest.approx(4, abs=0.01)
    out = proj.export(PRESETS["Sanft"], tmp_path / "o44.wav", out_sr=44_100)
    assert sf.info(out).samplerate == 44_100


@pytest.mark.slow
@pytest.mark.skipif(not (os.environ.get("KLANGRESTAURATOR_DEEPFILTER") or shutil.which("curl")),
                    reason="needs deep-filter binary or network")
def test_real_deepfilternet(reading_file, tmp_path):
    src, _ = reading_file
    proj = Project(src, workdir=tmp_path / "work")
    proj.prepare()
    y = proj.render(PRESETS["Natürlich"], 0, 5)
    assert np.isfinite(y).all()


def test_mix_segment_is_voice_plus_share_of_music(reading_file, no_dfn):
    from klangrestaurator.analysis import MIX, SPEECH
    path, _ = reading_file
    proj = Project(path)
    proj.prepare()
    lvl = lambda x: 10 * np.log10((x ** 2).mean())  # noqa: E731
    st = PRESETS["Natürlich"].with_(tone_match=0.0, level_match=0.0)   # check the mix formula alone
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
    st = PRESETS["Natürlich"].with_(mix_music=1.0)
    ref = proj.render(st, 14.5, 15.5)          # inside one music section
    proj.set_kind(15, 20, MIX)                 # music 10–15 | mix 15–20, boundary at 15 s
    y = proj.render(st.with_(mix_fade_s=0.01), 14.5, 14.9)  # just before the boundary: still music
    np.testing.assert_allclose(y, ref[: len(y)], atol=2e-3)
