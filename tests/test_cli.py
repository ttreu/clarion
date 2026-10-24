"""Command line: options → Settings, input checks, and a full run with the fake AI."""
import re

import numpy as np
import pytest
import soundfile as sf

from klangrestaurator import cli
from klangrestaurator.settings import PRESETS


def _settings(*args):
    return cli.settings_from_args(cli.build_parser().parse_args(["x.wav", *args]))


def test_preset_alone_is_unchanged():
    assert _settings() == PRESETS["Natural"]
    assert _settings("-p", "Gentle") == PRESETS["Gentle"]
    assert _settings("-p", "Kräftig") == PRESETS["Strong"]          # old German names still work


def test_options_override_the_preset():
    st = _settings("--speech-db", "12", "--music", "0.5", "--mix-music", "0.3", "--tone", "0", "--level", "0.2",
                   "--hiss", "4", "--no-dereverb", "--broadcast", "--lufs", "-14")
    assert (st.speech_reduction_db, st.music_reduction, st.mix_music) == (12, 0.5, 0.3)
    assert (st.tone_match, st.level_match, st.music_hiss_db) == (0, 0.2, -4)   # hiss is always a cut
    assert not st.speech_dereverb and st.broadcast and st.loudness_target == -14


def test_loudness_options():
    assert _settings("--lufs", "peak").effective().loudness_target is None
    assert _settings("--no-normalize").effective().loudness_target is None
    assert _settings().loudness_target == PRESETS["Natural"].loudness_target   # not given: preset value


@pytest.mark.parametrize("args, message", [
    (["missing.wav"], "file not found"),
    (["{f}", "{f}", "-o", "out.wav"], "single input"),
    (["{f}", "--lufs", "loud"], "expected a number"),
])
def test_bad_input_gives_a_message_not_a_traceback(args, message, reading_file, capsys):
    f = str(reading_file[0])
    with pytest.raises(SystemExit) as e:
        cli.main([a.format(f=f) for a in args])
    assert e.value.code == 2 and message in capsys.readouterr().err


def test_full_run_exports_and_lists_sections(reading_file, no_dfn, tmp_path, capsys):
    src, _ = reading_file
    out = tmp_path / "out.wav"
    assert cli.main([str(src), "-o", str(out), "-p", "Gentle", "--keep-rate"]) == 0
    y, sr = sf.read(out)
    assert sr == 44_100 and abs(len(y) / sr - 30) < 0.05 and np.abs(y).max() > 0.01
    assert str(out) in capsys.readouterr().out
    assert cli.main([str(src), "--segments"]) == 0
    listed = capsys.readouterr().out
    m = re.match(r"Music (\d+):([\d.]+) – (\d+):([\d.]+)", listed)     # the cello part at 10–20 s
    assert m, listed
    start, end = int(m[1]) * 60 + float(m[2]), int(m[3]) * 60 + float(m[4])
    assert abs(start - 10) < 0.5 and abs(end - 20) < 0.5
