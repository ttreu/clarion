"""clarion was called "Klangrestaurator" before version 0.1.1. This module is the one place that still
knows the old names: data folders are moved over once, and old environment variables keep working.
(Old ``.klang`` project files are handled in ``project_file``, old settings in ``gui.app``.)
"""
from __future__ import annotations

import os
from pathlib import Path

OLD_NAME = "Klangrestaurator"


def env(name: str) -> str | None:
    """``CLARION_<name>``, or the old ``KLANGRESTAURATOR_<name>`` if only that one is set."""
    return os.environ.get(f"CLARION_{name}") or os.environ.get(f"KLANGRESTAURATOR_{name}")


def migrate_dir(new: Path, old: Path) -> Path:
    """Return ``new``; if it does not exist yet but ``old`` does, move ``old`` there first (same disk:
    instant, so a cache of many GB is kept instead of recomputed). If moving fails, keep using ``old``."""
    if not new.exists() and old.is_dir():
        try:
            new.parent.mkdir(parents=True, exist_ok=True)
            old.rename(new)
        except OSError:
            return old
    return new
