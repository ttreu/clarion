"""The rename Klangrestaurator → clarion: old data is taken over, old environment variables still work."""
import pytest

from clarion import compat


def test_old_folder_is_moved_once(tmp_path):
    old, new = tmp_path / "Klangrestaurator", tmp_path / "clarion"
    (old / "rec-1234").mkdir(parents=True)
    (old / "rec-1234" / "enh.npy").write_bytes(b"cached analysis")
    assert compat.migrate_dir(new, old) == new
    assert not old.exists() and (new / "rec-1234" / "enh.npy").read_bytes() == b"cached analysis"
    old.mkdir()                                            # an old folder appearing later is left alone
    assert compat.migrate_dir(new, old) == new and old.exists()


def test_nothing_to_migrate(tmp_path):
    assert compat.migrate_dir(tmp_path / "clarion", tmp_path / "Klangrestaurator") == tmp_path / "clarion"
    assert not (tmp_path / "Klangrestaurator").exists()


def test_env_prefers_new_name_and_falls_back_to_old(monkeypatch):
    monkeypatch.delenv("CLARION_DEEPFILTER", raising=False)
    monkeypatch.delenv("KLANGRESTAURATOR_DEEPFILTER", raising=False)
    assert compat.env("DEEPFILTER") is None
    monkeypatch.setenv("KLANGRESTAURATOR_DEEPFILTER", "/old/deep-filter")
    assert compat.env("DEEPFILTER") == "/old/deep-filter"
    monkeypatch.setenv("CLARION_DEEPFILTER", "/new/deep-filter")
    assert compat.env("DEEPFILTER") == "/new/deep-filter"



def test_settings_of_the_old_name_are_taken_over_once(tmp_path):
    QtCore = pytest.importorskip("PySide6.QtCore")
    from clarion.gui.app import take_over_settings

    def ini(name):
        return QtCore.QSettings(str(tmp_path / f"{name}.ini"), QtCore.QSettings.IniFormat)

    old = ini("old")
    old.setValue("broadcast", True)
    old.setValue("loop_mode", "sel")
    old.sync()
    assert take_over_settings(ini("new"), ini("old"))
    new = ini("new")
    assert new.value("loop_mode") == "sel" and new.value("broadcast", type=bool)
    new.setValue("loop_mode", "all")
    new.sync()
    assert not take_over_settings(ini("new"), ini("old"))             # only once: later changes stay
    assert ini("new").value("loop_mode") == "all"
