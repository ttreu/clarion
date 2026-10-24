"""PyInstaller entry point. `clarion --cli …` runs the command line inside the app bundle, e.g.
`clarion.app/Contents/MacOS/clarion --cli recording.wav --preset Natural`."""
import multiprocessing
import sys

# Libraries start helper processes (multiprocessing's resource tracker). In a frozen app those re-run
# this executable; freeze_support() makes them do their job and exit instead of opening another window.
multiprocessing.freeze_support()

if len(sys.argv) > 1 and sys.argv[1] == "--cli":
    from klangrestaurator.cli import main as cli_main

    raise SystemExit(cli_main(sys.argv[2:]))

from klangrestaurator.gui.app import main  # noqa: E402

raise SystemExit(main())
