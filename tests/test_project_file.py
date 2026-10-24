import json

import pytest

from clarion import project_file
from clarion.analysis import MIX, SPEECH, Segment
from clarion.settings import PRESETS


def test_round_trip_and_moved_folder(tmp_path):
    audio = tmp_path / "rec" / "lesung.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"RIFF")                                  # existence is all that matters here
    segs = [Segment(0, 10, SPEECH), Segment(10, 20, MIX, manual=True)]
    st = PRESETS["Strong"].with_(tone_match=0.3, loudness_target=-18.0)
    saved = project_file.save(tmp_path / "rec" / "projekt", audio, segs, (1.5, 19.0), (12.0, 16.0), st,
                              {"preset": "Kräftig"})
    assert saved.suffix == ".clarion"
    lp = project_file.load(saved)
    assert lp.source == audio.resolve()
    assert [(s.start, s.end, s.kind, s.manual) for s in lp.segments] == [(0, 10, SPEECH, False), (10, 20, MIX, True)]
    assert lp.trim == (1.5, 19.0) and lp.tone_ref == (12.0, 16.0)
    assert lp.settings == st and lp.gui["preset"] == "Kräftig"
    # the whole folder moved: the audio is found next to the project via the relative path
    moved = tmp_path / "moved"
    (tmp_path / "rec").rename(moved)
    lp = project_file.load(moved / "projekt.clarion")
    assert lp.source == (moved / "lesung.wav").resolve()


def test_rejects_foreign_and_tolerates_unknown_settings(tmp_path):
    bad = tmp_path / "x.clarion"
    bad.write_text(json.dumps({"format": "etwas anderes"}))
    with pytest.raises(ValueError):
        project_file.load(bad)
    ok = tmp_path / "y.clarion"
    ok.write_text(json.dumps({"format": project_file.FORMAT, "version": 1, "source": "/nirgends.wav",
                              "settings": {"speech_reduction_db": 25.0, "gibt_es_nicht": 1}}))
    lp = project_file.load(ok)
    assert lp.settings.speech_reduction_db == 25.0 and lp.source is None


def test_projects_from_before_the_rename_still_open(tmp_path):
    """Version 0.1.0 saved '.klang' files with the format 'klangrestaurator-projekt'."""
    old = tmp_path / "reading.klang"
    old.write_text(json.dumps({"format": "klangrestaurator-projekt", "version": 1, "source": "/nowhere.wav",
                               "segments": [{"start": 0, "end": 5, "kind": "speech"}],
                               "settings": {"speech_reduction_db": 30.0}}))
    lp = project_file.load(old)
    assert lp.settings.speech_reduction_db == 30.0 and lp.segments[0].kind == SPEECH
    assert ".klang" in project_file.SUFFIXES
    resaved = project_file.save(old, tmp_path / "a.wav", lp.segments, None, None, lp.settings)
    assert resaved == tmp_path / "reading.clarion"                 # saving again uses the new suffix
    assert json.loads(resaved.read_text())["format"] == project_file.FORMAT


def test_old_german_preset_names_still_work():
    from clarion.settings import preset_name
    assert preset_name("Kräftig") == "Strong" and preset_name("Natürlich") == "Natural"
    assert preset_name("Gentle") == "Gentle" and preset_name(None) == "Natural" and preset_name("??") == "Natural"
