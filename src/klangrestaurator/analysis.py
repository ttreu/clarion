"""Speech / music segmentation.

Idea: DeepFilterNet is trained to keep speech and remove everything else. In speech
passages it removes only a few dB; on a cello (or applause, crowd noise) it removes
30–50 dB. That ratio is a robust, model-free speech/music detector for recordings
where both rarely overlap (readings with music, lectures with interludes, …).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .settings import SR, Settings

FRAME = SR // 4  # 0.25 s analysis frames
SPEECH, MUSIC, MIX = "speech", "music", "mix"
# SPEECH → speech chain (AI); MUSIC → music chain (gentle NR); MIX → music chain + some dry AI voice
KIND_LABEL = {SPEECH: "Sprache", MUSIC: "Musik", MIX: "Sprache + Musik"}


@dataclass
class Segment:
    start: float  # seconds
    end: float
    kind: str     # "speech" | "music"
    manual: bool = False  # set by the user in the GUI

    @property
    def duration(self) -> float:
        return self.end - self.start


def frame_levels(x: np.ndarray, block_s: int = 60) -> np.ndarray:
    """Mono RMS level per 0.25 s frame in dBFS, computed blockwise (memmap friendly)."""
    nf = len(x) // FRAME
    out = np.empty(nf, np.float32)
    per = block_s * 4
    for b in range(0, nf, per):
        e = min(nf, b + per)
        blk = np.asarray(x[b * FRAME:e * FRAME], dtype=np.float32).mean(axis=1)
        out[b:e] = 10 * np.log10((blk.reshape(e - b, FRAME) ** 2).mean(axis=1) + 1e-12)
    return out


def _runs(flags: np.ndarray):
    out, s = [], 0
    for i in range(1, len(flags) + 1):
        if i == len(flags) or flags[i] != flags[s]:
            out.append((s, i, bool(flags[s])))
            s = i
    return out


def classify(orig_db: np.ndarray, enh_db: np.ndarray, st: Settings) -> list[Segment]:
    """Return a gap-free list of speech/music segments covering the whole file."""
    nf = min(len(orig_db), len(enh_db))
    orig_db, enh_db = orig_db[:nf], enh_db[:nf]
    if nf == 0:
        return []
    ratio = enh_db - orig_db
    active = orig_db > st.active_level_db
    vote = np.where(active, (ratio < st.music_ratio_db).astype(float), np.nan)

    # rolling mean of votes over 4 s, ignoring pauses
    k = 16
    v0 = np.nan_to_num(vote, nan=0.0)
    cnt = np.convolve(active.astype(float), np.ones(k), "same")
    sm = np.convolve(v0, np.ones(k), "same")
    score = np.where(cnt >= 3, sm / np.maximum(cnt, 1), np.nan)

    # pauses inherit the previous state
    music = np.zeros(nf, bool)
    last = False
    for i in range(nf):
        if not np.isnan(score[i]):
            last = score[i] > 0.5
        music[i] = last

    min_frames = int(st.min_segment_s * 4)
    for s, e, v in _runs(music):
        if e - s < min_frames:
            music[s:e] = not v

    segs = [[s / 4, e / 4, MUSIC if v else SPEECH] for s, e, v in _runs(music)]
    segs = _refine(segs, orig_db)
    out = _sanitize([(a, b) for a, b, k in segs if k == MUSIC], nf / 4)
    out = detect_mix(out, orig_db, enh_db, st)
    out = music_with_speech(out, orig_db, enh_db, st)
    return merge_short(out, st.min_switch_s)


def music_with_speech(segs: list[Segment], orig_db: np.ndarray, enh_db: np.ndarray,
                      st: Settings) -> list[Segment]:
    """Automatic music sections in which speech survives the AI are speech + music, not music.

    In pure music the AI removes 20–50 dB in every frame; where someone talks over the music, a fair
    share of frames keeps the AI output within a few dB of the original.
    """
    nf = min(len(orig_db), len(enh_db))
    out = []
    for sg in segs:
        if sg.kind == MUSIC and not sg.manual:
            a, b = int(round(sg.start * 4)), min(nf, int(round(sg.end * 4)))
            if b > a:
                talk = ((enh_db[a:b] - orig_db[a:b]) > -10) & (enh_db[a:b] > st.active_level_db)
                if talk.sum() >= 3 and talk.mean() >= st.mix_speech_share:
                    sg = Segment(sg.start, sg.end, MIX)
        out.append(sg)
    return _merge_equal(out)


def _merge_equal(segs: list[Segment]) -> list[Segment]:
    out: list[Segment] = []
    for s in segs:
        if out and out[-1].kind == s.kind and out[-1].manual == s.manual and abs(out[-1].end - s.start) < 1e-6:
            out[-1] = Segment(out[-1].start, s.end, s.kind, s.manual)
        else:
            out.append(Segment(s.start, s.end, s.kind, s.manual))
    return out


def merge_short(segs: list[Segment], min_s: float) -> list[Segment]:
    """Absorb automatic sections shorter than ``min_s`` into a neighbour, so methods don't flip quickly.

    Sections at the very start or end of the file are kept (they only cause a single switch).

    A short section goes to the neighbour that is speech + music if there is one (it keeps both voice
    and music), otherwise to the longer neighbour. Manual sections are never changed or absorbed into.
    """
    segs = _merge_equal(segs)
    stuck: set[tuple[float, float]] = set()  # short sections boxed in by manual ones
    while len(segs) > 1:
        # only sections with a neighbour on both sides cause back-and-forth switching
        cand = [i for i, s in enumerate(segs) if 0 < i < len(segs) - 1
                and not s.manual and s.duration < min_s and (s.start, s.end) not in stuck]
        if not cand:
            break
        i = min(cand, key=lambda k: segs[k].duration)
        nb = [j for j in (i - 1, i + 1) if 0 <= j < len(segs) and not segs[j].manual]
        if not nb:
            stuck.add((segs[i].start, segs[i].end))
            continue
        mixn = [j for j in nb if segs[j].kind == MIX]
        j = mixn[0] if mixn else max(nb, key=lambda k: segs[k].duration)
        segs[i] = Segment(segs[i].start, segs[i].end, segs[j].kind)
        segs = _merge_equal(segs)
    return segs


def _noise_floor(lev: np.ndarray, radius: int = 240, pct: float = 5) -> np.ndarray:
    """Local noise floor per frame: low percentile of the level within ±radius frames (±60 s)."""
    step = 20
    pts = np.arange(0, len(lev), step)
    vals = [np.percentile(lev[max(0, i - radius):i + radius + 1], pct) for i in pts]
    return np.interp(np.arange(len(lev)), pts, vals)


def detect_mix(segs: list[Segment], orig_db: np.ndarray, enh_db: np.ndarray, st: Settings) -> list[Segment]:
    """Mark stretches of automatic speech where music keeps playing under the voice as MIX.

    Looks at the gaps between words (frames the AI wipes by more than ``music_ratio_db``): in plain
    speech they are near the noise floor, with music under the voice they stay ``mix_gap_db`` above it.
    """
    nf = min(len(orig_db), len(enh_db))
    if nf == 0 or not segs:
        return segs
    o, e = orig_db[:nf], enh_db[:nf]
    gap = (e - o) < st.music_ratio_db
    musical = gap & (o > _noise_floor(o) + st.mix_gap_db)
    k = 32  # 8 s window
    min_frames = int(st.min_segment_s * 4)
    out: list[Segment] = []
    for sg in segs:
        a, b = int(round(sg.start * 4)), min(nf, int(round(sg.end * 4)))
        if sg.kind != SPEECH or sg.manual or b - a < min_frames:
            out.append(sg)
            continue
        # judge each speech segment on its own frames only (no carry-over from neighbouring music)
        g = gap[a:b]
        n_gap = np.convolve(g.astype(float), np.ones(k), "same")
        n_mus = np.convolve(musical[a:b].astype(float), np.ones(k), "same")
        # music *under* speech shows in many separate word gaps, not in one block (a short interlude)
        starts = g & ~np.concatenate([[False], g[:-1]])
        n_events = np.convolve(starts.astype(float), np.ones(k), "same")
        score = np.where((n_gap >= 4) & (n_events >= 3), n_mus / np.maximum(n_gap, 1), np.nan)
        f = np.zeros(b - a, bool)
        last = False
        for i in range(b - a):
            if not np.isnan(score[i]):
                last = score[i] >= st.mix_gap_share
            f[i] = last
        for s0, e0, v in _runs(f):
            if e0 - s0 < min_frames:
                f[s0:e0] = not v
        for s0, e0, v in _runs(f):
            t0 = sg.start if s0 == 0 else (a + s0) / 4
            t1 = sg.end if e0 == len(f) else (a + e0) / 4
            if t1 > t0:
                out.append(Segment(t0, t1, MIX if v else SPEECH))
    return out


def _refine(segs, lev):
    """Move music starts to the actual onset and extend ends over the decay."""
    nf = len(lev)
    for i, (s, e, k) in enumerate(segs):
        if k != MUSIC:
            continue
        a = int(s * 4)
        noise = np.percentile(lev[max(0, a - 240):min(nf, a + 240)], 5)
        on = next((j for j in range(max(0, a - 4), min(nf, a + 20)) if lev[j] > noise + 10), a)
        segs[i][0] = max(on / 4 - 0.3, segs[i - 1][0] + 0.5 if i > 0 else 0.0)
        j = int(e * 4)
        while j < min(nf, int(e * 4) + 12) and lev[j] > noise + 6:
            j += 1
        segs[i][1] = j / 4
    return segs


def _sanitize(music: list[tuple[float, float]], total: float) -> list[Segment]:
    """Merge overlapping music intervals and fill the gaps with speech."""
    music = sorted((max(0.0, a), min(total, b)) for a, b in music if b - a > 0.3)
    merged: list[list[float]] = []
    for a, b in music:
        if merged and a <= merged[-1][1] + 0.5:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out, t = [], 0.0
    for a, b in merged:
        if a > t:
            out.append(Segment(t, a, SPEECH))
        out.append(Segment(a, b, MUSIC))
        t = b
    if t < total:
        out.append(Segment(t, total, SPEECH))
    return out


def set_kind(segs: list[Segment], start: float, end: float, kind: str) -> list[Segment]:
    """Force [start, end) to `kind` (manual correction) and return a clean list."""
    total = segs[-1].end if segs else end
    pieces: list[Segment] = []
    for s in segs:
        if s.end <= start or s.start >= end:
            pieces.append(s)
            continue
        if s.start < start:
            pieces.append(Segment(s.start, start, s.kind, s.manual))
        if s.end > end:
            pieces.append(Segment(end, s.end, s.kind, s.manual))
    pieces.append(Segment(max(0.0, start), min(total, end), kind, True))
    pieces.sort(key=lambda s: s.start)
    out: list[Segment] = []
    for s in pieces:  # merge neighbours of equal kind
        if out and out[-1].kind == s.kind and abs(out[-1].end - s.start) < 1e-6:
            out[-1] = Segment(out[-1].start, s.end, s.kind, out[-1].manual or s.manual)
        else:
            out.append(s)
    return out


def fine_onset(x: np.ndarray, t0: float, search_s: tuple[float, float] = (-0.2, 1.0),
               hop_s: float = 0.005, rise_db: float = 12.0) -> float | None:
    """Locate a music onset near ``t0`` (s) with 5 ms resolution.

    The 0.25 s analysis frames only place a music start roughly (and deliberately early). This finds
    the first point in the search window where the level jumps ``rise_db`` above the local floor after
    at least 100 ms below it, and stays up for 30 ms. Returns None if there is no clear onset.
    """
    a = max(0, int((t0 + search_s[0]) * SR))
    b = min(len(x), int((t0 + search_s[1]) * SR))
    hop, win = int(hop_s * SR), int(2 * hop_s * SR)
    if b - a < 40 * hop:
        return None
    mono = np.asarray(x[a:b], dtype=np.float32)
    mono = mono.mean(axis=1) if mono.ndim == 2 else mono
    n = (len(mono) - win) // hop
    idx = np.arange(n)[:, None] * hop + np.arange(win)[None, :]
    env = 10 * np.log10((mono[idx] ** 2).mean(axis=1) + 1e-12)
    thr = np.percentile(env, 10) + rise_db
    quiet, hold = int(0.1 / hop_s), int(0.03 / hop_s)
    for i in range(quiet, n - hold):
        if env[i] >= thr and np.all(env[i - quiet:i] < thr) and np.all(env[i:i + hold] >= thr - 3):
            return (a + i * hop) / SR
    return None


def snap_music_onsets(segs: list[Segment], x: np.ndarray, pre_s: float = 0.025) -> list[Segment]:
    """Move automatic speech→music boundaries to the exact onset (just before it) found in ``x``.

    Manual boundaries are left alone. Returns a new list; the input is not modified.
    """
    out = [Segment(s.start, s.end, s.kind, s.manual) for s in segs]
    for i in range(1, len(out)):
        prev, seg = out[i - 1], out[i]
        if seg.kind != MUSIC or prev.kind != SPEECH or seg.manual or prev.manual:
            continue
        on = fine_onset(x, seg.start)
        if on is None:
            continue
        t = min(max(on - pre_s, prev.start + 0.5), seg.end - 0.5)
        if t > prev.start and t < seg.end:
            prev.end = seg.start = t
    return out


def speech_mask(segs: list[Segment], n: int, crossfade_s: float, start: int = 0,
                onset_fade_s: float | None = None, mix_fade_s: float | None = None) -> np.ndarray:
    """Per-sample weight (1 = speech chain, 0 = music chain) for samples [start, start+n).

    Music→speech boundaries get a raised-cosine ramp of ``crossfade_s`` centred on the boundary.
    With ``onset_fade_s``, speech→music boundaries instead get a short ramp that *ends* at the
    boundary: the music chain (with its higher noise floor) then comes in together with the music,
    which masks it, instead of the noise swelling audibly in the silence before.
    """
    m = np.zeros(n, np.float32)
    for s in segs:
        if s.kind != SPEECH:
            continue
        a, b = int(s.start * SR) - start, int(s.end * SR) - start
        a, b = max(0, a), min(n, b)
        if b > a:
            m[a:b] = 1.0
    # replace each hard switch by a raised-cosine ramp centred on the boundary
    def ramp(L):
        return (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, L))).astype(np.float32)

    for prev, nxt in zip(segs, segs[1:], strict=False):
        if (prev.kind == SPEECH) == (nxt.kind == SPEECH):
            continue
        c = int(nxt.start * SR) - start
        if prev.kind == SPEECH and nxt.kind == MUSIC and onset_fade_s is not None:
            L = int(onset_fade_s * SR)
            a = c - L
        else:
            L = int((mix_fade_s if MIX in (prev.kind, nxt.kind) and mix_fade_s else crossfade_s) * SR)
            a = c - L // 2
        b = a + L
        if L <= 1 or b <= 0 or a >= n:
            continue
        r = ramp(L) if nxt.kind == SPEECH else 1 - ramp(L)
        lo, hi = max(0, a), min(n, b)
        m[lo:hi] = r[lo - a:hi - a]
    return m


def move_boundary(segs: list[Segment], old: float, new: float, min_len: float = 0.3) -> list[Segment]:
    """Move the boundary at ``old`` to ``new`` (manual correction): both neighbours become manual.

    ``new`` is kept at least ``min_len`` inside the two neighbouring sections. Returns a new list.
    """
    out = [Segment(s.start, s.end, s.kind, s.manual) for s in segs]
    i = min(range(len(out) - 1), key=lambda k: abs(out[k].end - old), default=None)
    if i is None or abs(out[i].end - old) > 1e-3:
        return out
    left, right = out[i], out[i + 1]
    t = min(max(new, left.start + min_len), right.end - min_len)
    left.end = right.start = t
    left.manual = right.manual = True
    return out


def kind_mask(segs: list[Segment], kind: str, n: int, crossfade_s: float, start: int = 0) -> np.ndarray:
    """Per-sample weight of ``kind`` for samples [start, start+n), raised-cosine ramps centred on edges."""
    m = np.zeros(n, np.float32)
    L = int(crossfade_s * SR)
    for s in segs:
        if s.kind != kind:
            continue
        a, b = int(s.start * SR) - start, int(s.end * SR) - start
        lo, hi = max(0, a), min(n, b)
        if hi > lo:
            m[lo:hi] = 1.0
    if L > 1:
        ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, L))).astype(np.float32)
        for prev, nxt in zip(segs, segs[1:], strict=False):
            if (prev.kind == kind) == (nxt.kind == kind):
                continue
            a = int(nxt.start * SR) - start - L // 2
            lo, hi = max(0, a), min(n, a + L)
            if hi > lo:
                r = ramp if nxt.kind == kind else 1 - ramp
                m[lo:hi] = r[lo - a:hi - a]
    return m


def save_segments(segs: list[Segment], path: Path) -> None:
    Path(path).write_text(json.dumps([asdict(s) for s in segs], indent=1))


def load_segments(path: Path) -> list[Segment]:
    return [Segment(**d) for d in json.loads(Path(path).read_text())]
