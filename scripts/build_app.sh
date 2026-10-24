#!/bin/bash
# Builds dist/clarion.app (unsigned). First launch: right-click → Open.
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --extra build
uv run pyinstaller --noconfirm --windowed \
  --name clarion \
  --osx-bundle-identifier com.clarion.app \
  --collect-submodules noisereduce \
  --collect-submodules nara_wpe \
  --collect-data soundfile \
  --paths src \
  packaging/launch.py
echo "Done: dist/clarion.app"
