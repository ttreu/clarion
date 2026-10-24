#!/bin/bash
# Double-click launcher for macOS. On first run it installs uv and the dependencies.
cd "$(dirname "$0")" || exit 1
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (Python package manager)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh || { echo "Installing uv failed"; read -r; exit 1; }
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --quiet || { echo "Setup failed"; read -r; exit 1; }
exec uv run klangrestaurator "$@"
