#!/bin/bash
# Baut dist/Klangrestaurator.app (unsigniert). Beim ersten Öffnen: Rechtsklick → Öffnen.
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --extra build
uv run pyinstaller --noconfirm --windowed \
  --name Klangrestaurator \
  --osx-bundle-identifier com.klangrestaurator.app \
  --collect-submodules noisereduce \
  --collect-submodules nara_wpe \
  --collect-data soundfile \
  --paths src \
  packaging/launch.py
echo "Fertig: dist/Klangrestaurator.app"
