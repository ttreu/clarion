# CLAUDE.md – clarion

Mac desktop app **clarion** (always lowercase; Python + PySide6; package `clarion`, commands `clarion` / `clarion-cli`;
called "Klangrestaurator" before 0.1.1 – `compat.py` moves old data folders, `gui.app._prefs` copies old settings,
`project_file` still opens `.klang` projects) for restoring recordings where **speech and music alternate**
(readings with cello, lectures with interludes). The user is German-speaking (talk to them in German); the app's UI text is English,
code and comments are English.

## Commands

```bash
uv sync                                   # install (creates .venv)
uv run clarion                   # GUI
uv run clarion-cli FILE.wav      # CLI, see --help
uv run pytest -q                          # tests (fast ones use a fake AI; one test runs the real binary)
uv run pytest -q -m "not slow"            # offline
uvx ruff check src tests                  # lint
./scripts/build_app.sh                    # PyInstaller → dist/clarion-<version>-macos-<arch>.zip
```

Env vars: `CLARION_DEEPFILTER=/path/to/deep-filter` (use a local binary instead of downloading),
`CLARION_NO_AUDIO=1` (skip QtMultimedia, e.g. headless/CI). GUI can be driven headless with
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
                  8 kHz (music_hiss_db; 85 % + shelf chosen by ear on test file A over 95 %, two-pass NR, gating)
   mix (speech + music) = HP(link(enh)) + mix_music·M'   M' = noisereduce(orig, prop 0.95) (no WPE: it
                  eats sustained notes, −4 dB cello on test file B); like the speech chain with a large, denoised share of the original, so the music
                  stays continuous (no pumping). Noise, room sound and music share one knob (mix_music).
                  Earlier tries: recovering music from the AI residual (damaged the music) and music chain +
                  voice lift (too noisy and reverberant).
                  Consecutive music/mix sections share one NR pass (no double NR at their boundary).
   out = mask·speech + (1−mask)·music     mask = raised-cosine crossfade at segment boundaries:
         music→speech 0.5 s centred; speech→music 40 ms ending at the onset, which render() snaps
         to 5 ms precision (snap_music_onsets, auto boundaries only) so the music chain's higher
         noise floor arrives with the music instead of swelling in the silence before it
export: [Settings.broadcast: multiband.py, model of Adobe Audition's Multiband Compressor preset "Broadcast"
   (LR4 crossovers 137/1147/6910 Hz, peak detector on a 1 ms grid, hard knee, +9.9 dB make-up into a limiter at
   −0.1 dBFS); fed at multiband.REF_LUFS = −17.2 (the export the user ran through Audition) because its
   thresholds are absolute; matched Audition's output within 0.4 dB per 50 ms frame; chosen by ear]
   → loudness normalise to loudness_target LUFS (BS.1770, blockwise) → music ceiling (music_ceiling_lu: 3 s
   short-term loudness of MUSIC sections capped at target + 3 LU, smooth ~2 s ramps; gain is NOT re-measured
   afterwards, or the cap would be undone) → look-ahead limiter at −1 dBFS, release 400 ms
   (dynamics.Limiter, stateful across blocks) → 24-bit WAV via audio_io.StreamWriter (seamless resampling with
   context). Settings.normalize off (GUI group checkbox "Loudness", CLI --no-normalize) → old peak normalisation.
   Tried and dropped (no audible benefit in a blind listening comparison on test file B, see git history): a voice
   compressor, and a home-made mastering stage (EQ + 3-band compression; without a make-up gate it lifted pause noise by
   12–16 dB).
preview: the GUI remembers the last 8 previews (key: source, Settings.effective(), range, sections,
   tone ref), each with its own files, so switching back to a computed state is instant.
   Same gain as the export, from Project.estimated_loudness (8 probes × 2 s, cached per
   settings; a single preview range misjudges by up to 6 dB because speech loses more than music)
   In peak mode the preview gain is the original's peak gain × (orig peak / processed peak) on the probes.
   With broadcast on, the preview runs the multiband stage on the probes too (Project._broadcast_estimate).
```

Key insight: DeepFilterNet removes 2–6 dB on speech but 30–50 dB on instruments/applause/crowd noise, so the
attenuation ratio *is* the speech/music detector — no extra classifier model needed.

Everything heavy works on memmaps in blocks of 60 s, so 1-hour files stay within a few GB RAM. The `deep-filter`
binary loads its whole input into memory, hence the chunking in `dfn.py`.

## Layout

- `src/clarion/settings.py` – `Settings` dataclass + `PRESETS` (Natural / Strong / Gentle; old German names map via `preset_name`)
- `audio_io.py` – decode to 48 kHz memmap (soundfile; ffmpeg fallback for m4a etc. and videos: first audio
  track only, length taken from the decoded track so the export stays in sync with the picture;
  `AUDIO_EXT` / `VIDEO_EXT`), blockwise WAV writer
- `dfn.py` – downloads/locates DeepFilterNet3 Rust binary (GitHub release v0.5.6), chunked `enhance()`
- `analysis.py` – frame levels, `classify()`: speech/music, then `detect_mix()` (speech whose word gaps stay
  ≥ 6 dB above the noise floor in ≥ 40 % of ≥ 3 separate gaps per 8 s → MIX), `music_with_speech()` (music
  with ≥ 10 % frames where speech survives the AI → MIX), `merge_short()` (inner sections < 6 s join a
  neighbour, preferring MIX: no rapid switching); onset/decay refinement, `snap_music_onsets()`, `set_kind()`
  for manual fixes, `speech_mask()` (SPEECH only; 1 s fades at MIX boundaries), `kind_mask()`
- `chains.py` – high-pass, `stereo_link`, speech chain, WPE (nara_wpe, 10 s chunks), music chain
  (noisereduce, n_fft 4096)
- `dynamics.py` – BS.1770 loudness (streaming meter), true peak, look-ahead limiter
- `multiband.py` – four-band compressor ("Broadcast"), streaming (`MultibandCompressor`, `compress_inplace`);
  the release is vectorised (gr[i] = r^i · cummax(t[j]·r^−j) in 10 s chunks), ~1 s per minute of audio
- `tone.py` – tone matching for the voice: third-octave bands of the linked AI voice (cached as
  `voice_bands.npy`), 6 s sliding shape vs. a reference (driest 30 % of windows, or `tone_ref.json` set by
  right-click), clipped ±12 dB × `tone_match`; plus `level_curve` (weighted loudness of sounding voice frames,
  6 s windows, pulled to the file median, ±10 dB × `level_match`) added as a flat gain. Voice activity is
  judged locally (`voice_active`: within 15 dB of the 90th pct over ±10 s), not by a global threshold.
  Only frames inside SPEECH/MIX sections count (the faint AI residue in music lowered test file A's speech by 7–10 dB).
  Applied as one STFT EQ to the speech chain and the MIX voice part; music is untouched
- `pipeline.py` – `Project`: cache dir per source (`~/Library/Caches/clarion/<stem>-<hash>`),
  `prepare()`, `render()`, `export()`, `write_preview()`
- `project_file.py` – `.clarion` projects (JSON; old `.klang` files still open): source (absolute + relative), sections, trim, tone ref, all
  Settings, GUI choices; opening one re-runs/uses the cached analysis, then applies the user's decisions
- Icons: sources in `assets/icon/` (`make_icon.py` builds the mark and app icons; `clairon.icns` is the app
  bundle icon); runtime copies in `gui/assets/` (`clarion-icon.png` window icon, `clarion-mark.svg` header logo)
- `gui/style.py` – dark studio theme (colours in `C`, stylesheet, palette) + vector transport icons; `gui/meter.py`
  – stereo level meter fed from the file being played (`_feed_meter`, last 40 ms before the play position)
- `cli.py` – CLI: `build_parser()`, `settings_from_args()` (preset + overrides), `main()`
- `gui/app.py` – `MainWindow`: header, timeline, the two panels, menus/shortcuts; opening audio and `.clarion`
  projects, analysis jobs, previews (+ the memory of the last 8), export
- `gui/settings_panel.py` – `SettingsPanel`: Settings ↔ widgets (preset, Speech / Music / Speech + Music with
  variant drop-downs, Loudness with the live spectrum, Output); signals `changed` / `switched`
- `gui/listen_panel.py` – `ListenPanel`: preview button, A/B, transport + loop, level meter, and the
  `QMediaPlayer` (glitch-free loading, `play_original()` before the first preview, `show_preview()`)
- `gui/timeline.py` waveform + segment view (peak/RMS per 5 ms from `wave.npy` =
  `analysis.waveform_overview`, real samples when zoomed in, square-root scale, coloured by section, drawn into a
  cached pixmap; click/drag/right-click/zoom; dragging a section boundary
  ±5 px → `analysis.move_boundary`, both neighbours become manual); `gui/spectrum.py` live spectrum;
  `gui/worker.py` QThread jobs with progress + cancel
- `tests/` – synthetic "reading" (bursty noise speech + harmonic cello) with a fake AI; one slow test with the real binary

## Conventions / gotchas

- Internal rate is always 48 kHz (DeepFilterNet requirement). Don't add resampling inside chains.
- Segment kinds: `speech`, `music`, `mix` (= "Speech + Music"). Labels in `analysis.KIND_LABEL`.
- Variants (settings.SPEECH_/MUSIC_/MIX_VARIANTS, letters = the listening-test samples) are global per project;
  the GUI drop-downs set the sliders, any manual change shows "Custom"; presets are built from them.
- Timeline navigation: trackpad pinch = zoom, two-finger swipe sideways = pan, up/down = smooth zoom; the first
  movement locks the axis until the gesture ends. Mouse wheel = zoom step, Shift + wheel = pan.
- Trim (start/end): `Project.trim` (trim.json), set by dragging the In/Out markers (below the section band;
  in the band, boundaries win), right-click or I/O at the playhead; outside the range the waveform is grey; export uses it
  when no explicit range is given and fades 20 ms at cut edges. Analysis still covers the whole file.
- Before the first preview the transport plays the original source file (`_orig_only`, files_ab = (source,
  None), range = whole file); B is disabled and no automatic preview runs in that mode.
- Cache invalidation is by path+size+mtime. Changing the analysis algorithm → use "Re-detect Speech/Music"
  (reclassify) or delete the cache folder; `enh.npy` only depends on the source.
- Manually corrected segments have `manual=True`; reclassify overwrites them (by design for now).
- Long jobs must check `cancel()` between blocks and raise `Cancelled`.
- Tests run with the cache redirected to a temp dir (autouse fixture in `conftest.py`).
- STFT helpers take (samples, channels); a 1-D signal must be reshaped first, or gains broadcast into
  a (bins × bins × frames) array (this once ate 60 GB of RAM in `tone.apply_eq`).
- QtMultimedia is created lazily (it can segfault in headless Linux containers).
- Building the app (`scripts/build_app.sh`): builds in `$TMPDIR/clarion-build` because iCloud's file attributes
  make `codesign` fail inside the project; ad-hoc signs (needed to run on Apple Silicon), puts LICENSE and
  THIRD_PARTY_NOTICES.md in Resources, zips with `ditto`. `packaging/launch.py` must call
  `multiprocessing.freeze_support()` first (else a library's resource-tracker helper re-launches the GUI and
  hangs), and `clarion --cli …` runs the CLI inside the bundle (used to smoke-test a build).
- Minimum macOS of the app is 14 (numpy/scipy wheels); PySide6 is pinned < 6.12 because 6.12's bindings are
  built for macOS 26. Check with `otool -l … | grep minos` after dependency upgrades.
- The project sits in iCloud-synced ~/Documents. iCloud sets the macOS `hidden` flag on files it syncs, and Python
  then skips `.pth` files while Qt skips plugins. So the venv lives in `.venv.nosync/` (iCloud ignores `*.nosync`)
  with `.venv` as a symlink to it. If it was recreated as a plain dir: `rm -rf .venv && mkdir .venv.nosync &&
  ln -s .venv.nosync .venv && uv sync`.
- Origin: this replicates a manual restoration of two Zoom recordings (a reading with cello)
  that was judged good; "Natural" (formerly "Natürlich") is the preset that was chosen. Its speech setting was later raised
  from 20 dB to 40 dB + WPE by listening tests (old behaviour ≈ "Gentle").

## Ideas / roadmap

- Third class "applause/crowd" (currently treated as music = gentle NR) with its own handling
- Batch window (several files, one preset), drag-out of finished files
- Code signing/notarisation for the .app; bundle the deep-filter binary instead of downloading
