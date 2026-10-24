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
   speech chain = HP(link(enh)) + 10^(−dB/20)·HP(orig)  [+ WPE dereverb]
                  link: one STFT gain per bin from both AI channels, applied to orig L+R
                  (deep-filter runs L/R independently → voice drifts L/R at high reduction)
   music  chain = HP(orig) + (noisereduce(orig, local noise profile) − HP(orig)), then hiss shelf −6 dB above
                  8 kHz (music_hiss_db; 85 % + shelf chosen by ear on 01.wav over 95 %, two-pass NR, gating)
   mix (speech + music) = HP(link(enh)) + mix_music·M'   M' = noisereduce(orig, prop 0.95) (no WPE: it
                  eats sustained notes, −4 dB cello on 02.WAV); like the speech chain with a large, denoised share of the original, so the music
                  stays continuous (no pumping). Noise, room sound and music share one knob (mix_music).
                  Earlier tries: recovering music from the AI residual (damaged the music) and music chain +
                  voice lift (too noisy and reverberant).
                  Consecutive music/mix sections share one NR pass (no double NR at their boundary).
   out = mask·speech + (1−mask)·music     mask = raised-cosine crossfade at segment boundaries:
         music→speech 0.5 s centred; speech→music 40 ms ending at the onset, which render() snaps
         to 5 ms precision (snap_music_onsets, auto boundaries only) so the music chain's higher
         noise floor arrives with the music instead of swelling in the silence before it
export: loudness normalise to loudness_target LUFS (BS.1770, blockwise) → look-ahead limiter at −1 dBFS
   (dynamics.Limiter, stateful across blocks) → 24-bit WAV via audio_io.StreamWriter (seamless resampling with
   context). Settings.normalize off (GUI group checkbox „Lautheit“, CLI --no-normalize) → old peak normalisation.
   Tried and dropped (no audible benefit in a blind listening comparison on 02.WAV, see git history): a voice
   compressor, and a mastering stage (EQ + 3-band compression; without a make-up gate it lifted pause noise by
   12–16 dB).
preview: the GUI remembers the last 8 previews (key: source, Settings.effective(), range, sections,
   tone ref), each with its own files, so switching back to a computed state is instant.
   Same gain as the export, from Project.estimated_loudness (8 probes × 2 s, cached per
   settings; a single preview range misjudges by up to 6 dB because speech loses more than music)
   In peak mode the preview gain is the original's peak gain × (orig peak / processed peak) on the probes.
```

Key insight: DeepFilterNet removes 2–6 dB on speech but 30–50 dB on instruments/applause/crowd noise, so the
attenuation ratio *is* the speech/music detector — no extra classifier model needed.

Everything heavy works on memmaps in blocks of 60 s, so 1-hour files stay within a few GB RAM. The `deep-filter`
binary loads its whole input into memory, hence the chunking in `dfn.py`.

## Layout

- `src/klangrestaurator/settings.py` – `Settings` dataclass + `PRESETS` (Natürlich / Kräftig / Sanft)
- `audio_io.py` – decode to 48 kHz memmap (soundfile; ffmpeg fallback for m4a etc.), blockwise WAV writer
- `dfn.py` – downloads/locates DeepFilterNet3 Rust binary (GitHub release v0.5.6), chunked `enhance()`
- `analysis.py` – frame levels, `classify()`: speech/music, then `detect_mix()` (speech whose word gaps stay
  ≥ 6 dB above the noise floor in ≥ 40 % of ≥ 3 separate gaps per 8 s → MIX), `music_with_speech()` (music
  with ≥ 10 % frames where speech survives the AI → MIX), `merge_short()` (inner sections < 6 s join a
  neighbour, preferring MIX: no rapid switching); onset/decay refinement, `snap_music_onsets()`, `set_kind()`
  for manual fixes, `speech_mask()` (SPEECH only; 1 s fades at MIX boundaries), `kind_mask()`
- `chains.py` – high-pass, `stereo_link`, speech chain, WPE (nara_wpe, 10 s chunks), music chain
  (noisereduce, n_fft 4096)
- `dynamics.py` – BS.1770 loudness (streaming meter), true peak, look-ahead limiter
- `tone.py` – tone matching for the voice: third-octave bands of the linked AI voice (cached as
  `voice_bands.npy`), 6 s sliding shape vs. a reference (driest 30 % of windows, or `tone_ref.json` set by
  right-click), clipped ±12 dB × `tone_match`; plus `level_curve` (weighted loudness of sounding voice frames,
  6 s windows, pulled to the file median, ±10 dB × `level_match`) added as a flat gain. Voice activity is
  judged locally (`voice_active`: within 15 dB of the 90th pct over ±10 s), not by a global threshold.
  Only frames inside SPEECH/MIX sections count (the faint AI residue in music lowered 01.wav's speech by 7–10 dB).
  Applied as one STFT EQ to the speech chain and the MIX voice part; music is untouched
- `pipeline.py` – `Project`: cache dir per source (`~/Library/Caches/Klangrestaurator/<stem>-<hash>`),
  `prepare()`, `render()`, `export()`, `write_preview()`
- `cli.py` – CLI; `gui/app.py` main window (settings grouped: Klangverbesserung Sprache / Musik / Sprache + Musik | Lautheit
  (checkable group: LUFS target) | Ausgabe); `gui/timeline.py` level + segment view (click/drag/right-click/zoom; dragging a section boundary
  ±5 px → `analysis.move_boundary`, both neighbours become manual);
  `gui/worker.py` QThread jobs with progress + cancel
- `tests/` – synthetic "reading" (bursty noise speech + harmonic cello) with a fake AI; one slow test with the real binary

## Conventions / gotchas

- Internal rate is always 48 kHz (DeepFilterNet requirement). Don't add resampling inside chains.
- Segment kinds: `speech`, `music`, `mix` (= "Sprache + Musik"). Labels in `analysis.KIND_LABEL`.
- Cache invalidation is by path+size+mtime. Changing the analysis algorithm → use "Erkennung neu berechnen"
  (reclassify) or delete the cache folder; `enh.npy` only depends on the source.
- Manually corrected segments have `manual=True`; reclassify overwrites them (by design for now).
- Long jobs must check `cancel()` between blocks and raise `Cancelled`.
- Tests run with the cache redirected to a temp dir (autouse fixture in `conftest.py`).
- STFT helpers take (samples, channels); a 1-D signal must be reshaped first, or gains broadcast into
  a (bins × bins × frames) array (this once ate 60 GB of RAM in `tone.apply_eq`).
- QtMultimedia is created lazily (it can segfault in headless Linux containers).
- The project sits in iCloud-synced ~/Documents. iCloud sets the macOS `hidden` flag on files it syncs, and Python
  then skips `.pth` files while Qt skips plugins. So the venv lives in `.venv.nosync/` (iCloud ignores `*.nosync`)
  with `.venv` as a symlink to it. If it was recreated as a plain dir: `rm -rf .venv && mkdir .venv.nosync &&
  ln -s .venv.nosync .venv && uv sync`.
- Origin: this replicates a manual restoration of two Zoom recordings (Ingeborg-Bachmann reading with cello)
  that was judged good; "Natürlich" is the preset that was chosen. Its speech setting was later raised
  from 20 dB to 40 dB + WPE by listening tests (old behaviour ≈ "Sanft").

## Ideas / roadmap

- Third class "applause/crowd" (currently treated as music = gentle NR) with its own handling
- Batch window (several files, one preset), drag-out of finished files
- Code signing/notarisation for the .app; bundle the deep-filter binary instead of downloading
