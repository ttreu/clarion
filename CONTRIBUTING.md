# Contributing to clarion

Thank you for your interest! Bug reports, ideas, listening feedback and code are all welcome.

## Reporting a problem or an idea

Open an [issue](../../issues) and describe:

- what you did, what you expected and what happened (an error message or a screenshot helps);
- your macOS version, Mac (Apple Silicon or Intel) and how you run clarion (app download or source);
- for sound problems: the kind of recording (recorder or camera, room, speech/music), the preset or
  settings, and where in the file it happens.

Please **do not attach recordings of other people without their consent.** A short snippet of your own,
or a description, is usually enough.

## Setting up for development

You need macOS (Linux works for most things), [uv](https://docs.astral.sh/uv/) and, for videos and M4A,
[ffmpeg](https://ffmpeg.org).

```bash
git clone https://github.com/<you>/clarion.git   # your fork
cd clarion
uv sync                    # creates .venv with all dependencies
uv run clarion             # start the app
uv run clarion-cli --help  # command line
```

The speech AI (`deep-filter`) is downloaded on first use. To use a local binary instead, set
`CLARION_DEEPFILTER=/path/to/deep-filter`.

## Tests and style

```bash
uv run pytest -q -m "not slow"   # fast tests, offline (a fake AI stands in for DeepFilterNet)
uv run pytest -q                 # all, including one test with the real deep-filter binary
uvx ruff check src tests         # lint (line length 120)
```

The GUI tests run without a screen (`QT_QPA_PLATFORM=offscreen`). Please add a test for new behaviour
and keep all tests green.

## Where things are

The developer notes in [CLAUDE.md](CLAUDE.md) explain the signal path, the file layout and the
conventions in detail. In short:

| Module | What it does |
|---|---|
| `analysis.py` | speech / music / speech + music detection, section editing |
| `chains.py`, `tone.py` | the processing for speech and music, tone and level matching |
| `dynamics.py`, `multiband.py` | loudness, limiter, broadcast compressor |
| `pipeline.py` | `Project`: cache, analysis, render, export, previews |
| `audio_io.py`, `dfn.py` | reading/writing audio and video sound, the DeepFilterNet binary |
| `gui/` | main window, settings and listen panels, timeline, spectrum, level meter |
| `cli.py` | command line |

A few rules that keep clarion working on hour-long recordings:

- Processing runs at 48 kHz internally; long work goes through memmaps in blocks of 60 s.
- Long jobs check `cancel()` between blocks and raise `Cancelled`.
- Code and comments are in English.

## Changes to the sound

Changes that alter how clarion sounds are judged by ear. Please describe what you compared (which
recordings, A/B at the same loudness) and, where possible, add numbers (noise floor, loudness, level of
the music) to your pull request. The existing defaults were chosen this way; the reasons are noted in the
code next to each setting.

## Pull requests

1. Fork the repository and create a branch.
2. Make your change with tests; run the tests and the linter.
3. Open a pull request that says what it changes and why.

By contributing you agree that your contribution is licensed under the [MIT License](LICENSE), like the
rest of clarion.

## Building the macOS app

```bash
./scripts/build_app.sh   # → dist/clarion-<version>-macos-<arch>.zip
```

The zip contains `clarion.app` for the architecture of the Mac that built it (arm64 or x86_64), ad-hoc
signed, with `LICENSE` and `THIRD_PARTY_NOTICES.md` inside. It is not notarized by Apple.
