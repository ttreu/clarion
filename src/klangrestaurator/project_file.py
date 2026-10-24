"""Project files (.klang): source file, sections with corrections, cut, tone reference and settings.

The heavy analysis stays in the cache (keyed by the source file); a project file only holds what the
user decided, so it is small, human-readable JSON and survives a cleared cache.
"""
from __future__ import annotations

import json
import os
from dataclasses import fields
from pathlib import Path

from .analysis import Segment
from .settings import Settings

SUFFIX = ".klang"
FORMAT = "klangrestaurator-projekt"
VERSION = 1


def save(path: Path, source: Path, segments: list[Segment], trim: tuple[float, float] | None,
         tone_ref: tuple[float, float] | None, settings: Settings, gui: dict | None = None) -> Path:
    path = Path(path)
    if path.suffix != SUFFIX:
        path = path.with_suffix(SUFFIX)
    source = Path(source).resolve()
    try:
        rel = os.path.relpath(source, path.resolve().parent)
    except ValueError:            # different drive (Windows)
        rel = None
    data = {
        "format": FORMAT,
        "version": VERSION,
        "source": str(source),
        "source_relative": rel,
        "segments": [{"start": s.start, "end": s.end, "kind": s.kind, "manual": s.manual} for s in segments],
        "trim": list(trim) if trim else None,
        "tone_ref": list(tone_ref) if tone_ref else None,
        "settings": settings.to_dict(),
        "gui": gui or {},
    }
    tmp = path.with_suffix(SUFFIX + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    tmp.replace(path)             # never leave a half-written project behind
    return path


class Loaded:
    """Contents of a project file."""

    def __init__(self, data: dict, path: Path):
        self.path = path
        self.source_abs = Path(data["source"])
        rel = data.get("source_relative")
        self.source_rel = (path.parent / rel).resolve() if rel else None
        self.segments = [Segment(float(d["start"]), float(d["end"]), d["kind"], bool(d.get("manual", False)))
                         for d in data.get("segments", [])]
        self.trim = tuple(data["trim"]) if data.get("trim") else None
        self.tone_ref = tuple(data["tone_ref"]) if data.get("tone_ref") else None
        known = {f.name for f in fields(Settings)}
        self.settings = Settings(**{k: v for k, v in data.get("settings", {}).items() if k in known})
        self.gui = data.get("gui", {})

    @property
    def source(self) -> Path | None:
        """The audio file: the stored path, else next to the project as before (moved folder)."""
        for p in (self.source_abs, self.source_rel):
            if p is not None and p.exists():
                return p
        return None


def load(path: Path) -> Loaded:
    path = Path(path)
    data = json.loads(path.read_text())
    if data.get("format") != FORMAT:
        raise ValueError(f"{path.name} is not a project file of this app")
    if int(data.get("version", 0)) > VERSION:
        raise ValueError(f"{path.name} was written by a newer version")
    return Loaded(data, path)
