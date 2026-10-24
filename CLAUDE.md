# CLAUDE.md – Klangrestaurator

Mac desktop app (Python + PySide6) for restoring recordings where **speech and music alternate**
(readings with cello, lectures with interludes). The user is German-speaking; UI text and user-facing docs are German,
code and comments are English.

## Commands

```bash
uv sync                                   # install (creates .venv)
uv run klangrestaurator                   # GUI
uv run klangrestaurator-cli FILE.wav      # CLI, see --help
uv run pytest -q                          # tests (fast ones use a fake AI; one test runs the real binary)
uv run pytest -q -m "not slow"            # offline
uvx ruff check src tests                  # lint
./scripts/build_app.sh                    # PyInstaller → dist/Klangrestaurator.app
```

Env vars: `KLANGRESTAURATOR_DEEPFILTER=/path/to/deep-filter` (use a local binary instead of downloading),
`KLANGRESTAURATOR_NO_AUDIO=1` (skip QtMultimedia, e.g. headless/CI). GUI can be driven headless with
`QT_QPA_PLATFORM=offscreen`.

## How it works (pipeline)

```
source file ──decode/resample──▶ orig.npy (48 kHz f32 stereo memmap)
                                   │
                                   ├─ deep-filter binary, 5-min chunks, 4 s crossfade ─▶ enh.npy
                                   │
levels (0.25 s dB) of orig & enh ─▶ classify(): ratio enh-orig < −18 dB on active frames
                                   ⇒ "music" (the speech AI wipes it) ⇒ segments.json
render(range, Settings):
   speech chain = HP(enh) + 10^(−dB/20)·HP(orig)  [+ WPE dereverb]
   music  chain = HP(orig) + (noisereduce(orig, local noise profile) − HP(orig))
   out = mask·speech + (1−mask)·music     mask = raised-cosine crossfade at segment boundaries
export: two-pass peak normalise to −1 dBFS, 24-bit WAV, optional resample to source rate
```

Key insight: DeepFilterNet removes 2–6 dB on speech but 30–50 dB on instruments/applause/crowd noise, so the
attenuation ratio *is* the speech/music detector — no extra classifier model needed.

Everything heavy works on memmaps in blocks of 60 s, so 1-hour files stay within a few GB RAM. The `deep-filter`
binary loads its whole input into memory, hence the chunking in `dfn.py`.

## Layout

- `src/klangrestaurator/settings.py` – `Settings` dataclass + `PRESETS` (Natürlich / Kräftig / Sanft)
- `audio_io.py` – decode to 48 kHz memmap (soundfile; ffmpeg fallback for m4a etc.), blockwise WAV writer
- `dfn.py` – downloads/locates DeepFilterNet3 Rust binary (GitHub release v0.5.6), chunked `enhance()`
- `analysis.py` – frame levels, `classify()`, onset/decay refinement, `set_kind()` for manual fixes, `speech_mask()`
- `chains.py` – high-pass, speech chain, WPE (nara_wpe, 10 s chunks), music chain (noisereduce, n_fft 4096)
- `pipeline.py` – `Project`: cache dir per source (`~/Library/Caches/Klangrestaurator/<stem>-<hash>`),
  `prepare()`, `render()`, `export()`, `write_preview()`
- `cli.py` – CLI; `gui/app.py` main window; `gui/timeline.py` level + segment view (click/drag/right-click/zoom);
  `gui/worker.py` QThread jobs with progress + cancel
- `tests/` – synthetic "reading" (bursty noise speech + harmonic cello) with a fake AI; one slow test with the real binary

## Conventions / gotchas

- Internal rate is always 48 kHz (DeepFilterNet requirement). Don't add resampling inside chains.
- Cache invalidation is by path+size+mtime. Changing the analysis algorithm → use "Erkennung neu berechnen"
  (reclassify) or delete the cache folder; `enh.npy` only depends on the source.
- Manually corrected segments have `manual=True`; reclassify overwrites them (by design for now).
- Long jobs must check `cancel()` between blocks and raise `Cancelled`.
- QtMultimedia is created lazily (it can segfault in headless Linux containers).
- The project sits in iCloud-synced ~/Documents. iCloud sets the macOS `hidden` flag on files it syncs, and Python
  then skips `.pth` files while Qt skips plugins. So the venv lives in `.venv.nosync/` (iCloud ignores `*.nosync`)
  with `.venv` as a symlink to it. If it was recreated as a plain dir: `rm -rf .venv && mkdir .venv.nosync &&
  ln -s .venv.nosync .venv && uv sync`.
- Origin: this replicates a manual restoration of two Zoom recordings (Ingeborg-Bachmann reading with cello)
  that was judged good; "Natürlich" is the preset that was chosen.

## Ideas / roadmap

- Third class "applause/crowd" (currently treated as music = gentle NR) with its own handling
- Speech-under-music overlap (cello under the voice): mix both chains instead of a hard choice
- Loudness normalisation (EBU R128) and gentle speech leveling as export option
- Batch window (several files, one preset), drag-out of finished files
- Code signing/notarisation for the .app; bundle the deep-filter binary instead of downloading
