#!/bin/bash
# Doppelklick-Starter für macOS. Installiert beim ersten Mal uv und die Abhängigkeiten.
cd "$(dirname "$0")" || exit 1
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "Installiere uv (Python-Paketmanager)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh || { echo "uv-Installation fehlgeschlagen"; read -r; exit 1; }
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --quiet || { echo "Einrichtung fehlgeschlagen"; read -r; exit 1; }
exec uv run klangrestaurator "$@"
