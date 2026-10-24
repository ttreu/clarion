import json

import pytest

from klangrestaurator import project_file
from klangrestaurator.analysis import MIX, SPEECH, Segment
from klangrestaurator.settings import PRESETS


def test_round_trip_and_moved_folder(tmp_path):
    audio = tmp_path / "rec" / "lesung.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"RIFF")                                  # existence is all that matters here
    segs = [Segment(0, 10, SPEECH), Segment(10, 20, MIX, manual=True)]
    st = PRESETS["Strong"].with_(tone_match=0.3, loudness_target=-18.0)
    saved = project_file.save(tmp_path / "rec" / "projekt", audio, segs, (1.5, 19.0), (12.0, 16.0), st,
                              {"preset": "Kräftig"})
    assert saved.suffix == ".klang"
    lp = project_file.load(saved)
    assert lp.source == audio.resolve()
    assert [(s.start, s.end, s.kind, s.manual) for s in lp.segments] == [(0, 10, SPEECH, False), (10, 20, MIX, True)]
    assert lp.trim == (1.5, 19.0) and lp.tone_ref == (12.0, 16.0)
    assert lp.settings == st and lp.gui["preset"] == "Kräftig"
    # the whole folder moved: the audio is found next to the project via the relative path
    moved = tmp_path / "moved"
    (tmp_path / "rec").rename(moved)
    lp = project_file.load(moved / "projekt.klang")
    assert lp.source == (moved / "lesung.wav").resolve()


def test_rejects_foreign_and_tolerates_unknown_settings(tmp_path):
    bad = tmp_path / "x.klang"
    bad.write_text(json.dumps({"format": "etwas anderes"}))
    with pytest.raises(ValueError):
        project_file.load(bad)
    ok = tmp_path / "y.klang"
    ok.write_text(json.dumps({"format": project_file.FORMAT, "version": 1, "source": "/nirgends.wav",
                              "settings": {"speech_reduction_db": 25.0, "gibt_es_nicht": 1}}))
    lp = project_file.load(ok)
    assert lp.settings.speech_reduction_db == 25.0 and lp.source is None


def test_old_german_preset_names_still_work():
    from klangrestaurator.settings import preset_name
    assert preset_name("Kräftig") == "Strong" and preset_name("Natürlich") == "Natural"
    assert preset_name("Gentle") == "Gentle" and preset_name(None) == "Natural" and preset_name("??") == "Natural"
