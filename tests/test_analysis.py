import numpy as np

from klangrestaurator.analysis import MUSIC, SPEECH, Segment, classify, set_kind, speech_mask
from klangrestaurator.settings import SR, Settings


def _levels(music_ranges, total_s=60, speech_db=-28, music_db=-20, removed_music=-40, removed_speech=-4):
    n = total_s * 4
    orig = np.full(n, speech_db, float)
    enh = orig + removed_speech
    for a, b in music_ranges:
        orig[int(a * 4):int(b * 4)] = music_db
        enh[int(a * 4):int(b * 4)] = music_db + removed_music
    # a few pauses inside speech
    for p in range(2, n, 23):
        if enh[p] > -40:
            orig[p] = enh[p] = -50
    return orig, enh


def test_classify_finds_music_block():
    orig, enh = _levels([(20, 35)])
    segs = classify(orig, enh, Settings())
    music = [s for s in segs if s.kind == MUSIC]
    assert len(music) == 1
    assert abs(music[0].start - 20) < 1.0 and abs(music[0].end - 35) < 1.5
    # gap-free coverage
    assert segs[0].start == 0 and abs(segs[-1].end - 60) < 1e-6
    assert all(abs(a.end - b.start) < 1e-6 for a, b in zip(segs, segs[1:], strict=False))


def test_short_blips_are_ignored():
    orig, enh = _levels([(30, 31)])  # 1 s < min_segment_s
    segs = classify(orig, enh, Settings())
    assert all(s.kind == SPEECH for s in segs)


def test_set_kind_splits_and_merges():
    segs = [Segment(0, 10, SPEECH), Segment(10, 20, MUSIC), Segment(20, 30, SPEECH)]
    out = set_kind(segs, 12, 15, SPEECH)
    assert [(s.start, s.end, s.kind) for s in out] == [
        (0, 10, SPEECH), (10, 12, MUSIC), (12, 15, SPEECH), (15, 20, MUSIC), (20, 30, SPEECH)]
    out = set_kind(out, 10, 20, SPEECH)
    assert [(s.start, s.end, s.kind) for s in out] == [(0, 30, SPEECH)]
    assert out[0].manual


def test_speech_mask_ramps():
    segs = [Segment(0, 2, SPEECH), Segment(2, 4, MUSIC)]
    m = speech_mask(segs, 4 * SR, crossfade_s=0.5)
    assert m[0] == 1 and m[-1] == 0
    assert 0.4 < m[2 * SR] < 0.6
    assert np.all(np.diff(m[int(1.5 * SR):int(2.5 * SR)]) <= 1e-6)  # monotone down
    # window starting mid-file gives the same values
    m2 = speech_mask(segs, SR, 0.5, start=int(1.5 * SR))
    np.testing.assert_allclose(m2, m[int(1.5 * SR):int(2.5 * SR)])
