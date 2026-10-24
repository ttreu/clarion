"""Main window: header, timeline, settings panel, listen panel – and the work behind them (opening files
and projects, analysis, previews with their memory, export).

The two panels are in ``settings_panel.py`` (Settings ↔ widgets) and ``listen_panel.py`` (player, A/B,
transport, meters); this file connects them to the project.
"""
from __future__ import annotations

import os
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import APP_NAME, __version__, audio_io, project_file
from ..analysis import MIX, MUSIC
from ..pipeline import Project, default_output_name
from ..settings import SR, Settings
from . import style, worker
from .listen_panel import PREVIEW_S, SKIP_MS, ListenPanel
from .settings_panel import SettingsPanel
from .timeline import Timeline, fmt

AUDIO_EXT = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".m4a", ".ogg", ".w64"}
AUDIO_FILTER = "Audio (*.wav *.aif *.aiff *.flac *.mp3 *.m4a *.ogg *.w64)"
PREVIEW_KEEP = 8      # remembered previews for instant switching between settings
MAX_PREVIEW_S = 120   # longer selections are cut: previews must stay quick


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1100, 640)
        self.setAcceptDrops(True)
        self.prefs = QSettings("Klangrestaurator", "Klangrestaurator")
        self.project: Project | None = None
        self._project_path: Path | None = None   # the .klang file, once saved / opened
        self._dirty = False                       # unsaved decisions (sections, cut, settings …)
        self._pending_project: project_file.Loaded | None = None   # applied when its analysis is done
        self._job = None
        self._job_is_preview = False
        self._preview_stale = True
        self._refresh_pending = False
        self._next_range: tuple[float, float] | None = None  # preview range wanted by a new selection
        # remembered previews: key (settings, range, sections) → (A file, B file); each has its own files,
        # so switching back to a computed state is instant and nothing being played gets overwritten
        self._previews: OrderedDict[tuple, tuple[Path, Path]] = OrderedDict()
        self._slot_seq = 0
        self._auto_timer = QTimer(self, singleShot=True, interval=400, timeout=self._auto_refresh)

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._header())
        content = QWidget()
        v = QVBoxLayout(content)
        v.setContentsMargins(14, 12, 14, 10)
        v.setSpacing(10)
        outer.addWidget(content, 1)
        v.addWidget(self._timeline_card(), 1)

        self.panel = SettingsPanel(self.prefs)
        self.panel.changed.connect(self._mark_stale)
        self.panel.switched.connect(self._refresh_now)
        self.listen = ListenPanel(self.prefs)
        self.listen.preview_requested.connect(self.make_preview)
        self.listen.head.connect(self.timeline.set_head)
        self.listen.samples.connect(self.panel.spectrum.set_audio)
        self.listen.status.connect(self.statusBar().showMessage)
        self.listen.selection = lambda: self.timeline.sel
        self.listen.decoded_source = lambda: self.project.orig if self.project else None
        mid = QHBoxLayout()
        mid.setSpacing(12)
        mid.addWidget(self.panel, 3)
        mid.addWidget(self.listen, 2)
        v.addLayout(mid)

        bottom = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        self.progress.setFormat("")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_job)
        bottom.addWidget(self.progress, 1)
        bottom.addWidget(self.btn_cancel)
        v.addLayout(bottom)

        self._menus()
        self._clear_preview_files()
        self._set_busy(False)
        self.statusBar().showMessage("Ready")

    # ------------------------------------------------------------------ layout
    def _header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("header")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(18, 10, 14, 10)
        hl.setSpacing(10)
        name = QLabel(APP_NAME)
        name.setObjectName("appname")
        self.lbl_file = QLabel("No file open")
        self.lbl_file.setObjectName("filename")
        self.lbl_file.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.btn_open = QPushButton("Open…")
        self.btn_open.clicked.connect(self.open_dialog)
        self.btn_save = QPushButton("Save")
        self.btn_save.setToolTip("Save project (⌘S): sections, trim, settings")
        self.btn_save.clicked.connect(lambda: self.save_project())
        self.btn_export = QPushButton("Export…")
        self.btn_export.setObjectName("primary")
        self.btn_export.clicked.connect(self.export_dialog)
        hl.addWidget(name)
        hl.addSpacing(8)
        hl.addWidget(self.lbl_file, 1)
        for b in (self.btn_open, self.btn_save, self.btn_export):
            hl.addWidget(b)
        return header

    def _timeline_card(self) -> QFrame:
        self.timeline = Timeline()
        self.timeline.seek.connect(lambda t: self.listen.seek_to(t))
        self.timeline.relabel.connect(self._on_relabel)
        self.timeline.tone_ref.connect(self._on_tone_ref)
        self.timeline.boundary_moved.connect(self._on_boundary)
        self.timeline.trim_changed.connect(self._on_trim)
        self.timeline.selection_changed.connect(self._on_selection)
        card = QFrame()
        card.setObjectName("card")
        tl = QVBoxLayout(card)
        tl.setContentsMargins(8, 8, 8, 6)
        tl.setSpacing(4)
        tl.addWidget(self.timeline, 1)
        hint = QLabel("Click = position · Drag = select · Drag a boundary = adjust section · "
                      "Drag In/Out or press I / O = start / end · Right-click = correct / trim · "
                      "Pinch or ↕ = zoom, ↔ = scroll")
        hint.setObjectName("hint")
        tl.addWidget(hint)
        return card

    def _menus(self) -> None:
        m = self.menuBar().addMenu("File")
        m.addAction(QAction("Open…", self, shortcut=QKeySequence.Open, triggered=self.open_dialog))
        m.addSeparator()
        m.addAction(QAction("Save Project", self, shortcut=QKeySequence.Save, triggered=lambda: self.save_project()))
        m.addAction(QAction("Save Project As…", self, shortcut=QKeySequence("Ctrl+Shift+S"),
                            triggered=lambda: self.save_project(ask=True)))
        m.addSeparator()
        m.addAction(QAction("Export…", self, shortcut=QKeySequence("Ctrl+E"), triggered=self.export_dialog))
        m.addAction(QAction("Re-detect Speech/Music", self, triggered=self.reclassify))
        ls = self.listen
        for name, key, fn in (("Play/Pause", Qt.Key_Space, ls.toggle_play),
                              ("Stop", Qt.Key_Period, ls.stop),
                              ("Back", Qt.Key_Left, lambda: ls.skip(-SKIP_MS)),
                              ("Forward", Qt.Key_Right, lambda: ls.skip(SKIP_MS)),
                              ("To start", Qt.Key_Home, ls.to_start),
                              ("Loop", Qt.Key_L, ls.btn_loop.toggle),
                              ("A/B", Qt.Key_Tab, ls.toggle_ab),
                              ("Set In here", Qt.Key_I, lambda: self._trim_at_head(start=True)),
                              ("Set Out here", Qt.Key_O, lambda: self._trim_at_head(start=False))):
            self.addAction(QAction(name, self, shortcut=QKeySequence(key), triggered=fn))
        hm = self.menuBar().addMenu("Help")
        hm.addAction(QAction(f"About {APP_NAME}", self, triggered=self.about))
        hm.addAction(QAction("About Qt", self, triggered=lambda: QMessageBox.aboutQt(self)))

    # ------------------------------------------------------------------ settings → preview
    def settings(self) -> Settings:
        return self.panel.settings()

    def _mark_stale(self, *_):
        self._preview_stale = True
        if self.project is not None:
            self._set_dirty()
        if self.listen.range and not self.listen.orig_only:
            self.listen.lbl_prev.setText("Settings changed – updating preview…")
            self._auto_timer.start()  # debounce: restarts while a slider is being dragged

    def _refresh_now(self):
        """A switch, not a slider: no need to wait for more changes."""
        self._auto_timer.stop()
        QTimer.singleShot(20, self._auto_refresh)

    def _auto_refresh(self):
        """Recompute the preview (new settings and/or new selection), keeping playback going."""
        if not (self._preview_stale and self.listen.range and self.project) or self.listen.orig_only:
            return
        if self._job:
            self._refresh_pending = True  # rerun when the current job has ended
            if self._job_is_preview:
                self._job.cancel()
            return
        rng, self._next_range = self._next_range or self.listen.range, None
        self.make_preview(rng, resume=True)

    # ------------------------------------------------------------------ jobs
    def _set_busy(self, busy: bool):
        has = self.project is not None and self.project.analysed
        self.btn_export.setEnabled(has and not busy)
        self.btn_open.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        self.listen.set_busy(busy, has, self._job_is_preview)

    def _progress(self, stage: str, v: float):
        self.progress.setValue(int(v * 1000))
        self.progress.setMaximum(1000)
        self.progress.setFormat(f"{stage}  %p%")

    def _run(self, fn, on_done, label: str, preview: bool = False):
        self._job_is_preview = preview
        th, job = worker.start(self, fn)
        for sig in (job.done, job.failed, job.cancelled):
            sig.connect(self._clear_job)
        job.progress.connect(self._progress)
        job.done.connect(on_done)
        job.failed.connect(self._failed)
        job.cancelled.connect(lambda: (self.statusBar().showMessage("Cancelled"), self._set_busy(False)))
        self._job = job
        self._set_busy(True)
        self.statusBar().showMessage(label)
        th.start()

    def _clear_job(self, *_):
        self._job = None
        self._job_is_preview = False
        self.progress.reset()
        self.progress.setFormat("")
        if self._refresh_pending:
            self._refresh_pending = False
            QTimer.singleShot(0, self._auto_refresh)

    def cancel_job(self):
        if self._job:
            self._job.cancel()
            self.statusBar().showMessage("Cancelling…")

    def _failed(self, msg: str):
        self._set_busy(False)
        self.progress.setFormat("Error")
        QMessageBox.critical(self, "Error", msg)

    # ------------------------------------------------------------------ opening files
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            p = Path(u.toLocalFile())
            if p.suffix.lower() == project_file.SUFFIX:
                self.open_project(p)
                return
            if p.suffix.lower() in AUDIO_EXT:
                self.open_audio(p)
                return

    def open_dialog(self):
        start = self.prefs.value("last_dir", str(Path.home()))
        exts = AUDIO_FILTER[AUDIO_FILTER.index("(") + 1:-1]
        fn, _ = QFileDialog.getOpenFileName(
            self, "Open audio file or project", start,
            f"Audio or project ({exts} *.klang);;{APP_NAME} project (*.klang);;{AUDIO_FILTER}")
        if not fn:
            return
        if Path(fn).suffix.lower() == project_file.SUFFIX:
            self.open_project(Path(fn))
        else:
            self.open_audio(Path(fn))

    def open_audio(self, path: Path):
        if not self._confirm_discard():
            return
        self._pending_project = None
        self._project_path = None
        self.load(path)
        self._set_dirty(False)

    def load(self, path: Path):
        self.listen.forget()
        self._clear_preview_files()
        self.prefs.setValue("last_dir", str(path.parent))
        self.project = Project(path)
        self.lbl_file.setText(str(path))
        try:
            self.listen.play_original(path, float(audio_io.probe(path)[2]))
        except Exception:  # noqa: BLE001 – unreadable here: the analysis will report it
            pass
        st, proj = self.settings(), self.project
        self._run(lambda progress, cancel: proj.prepare(st, progress=progress, cancel=cancel),
                  self._analysed, "Analysing… (the first time a file is opened this takes a few minutes)")

    def reclassify(self):
        if not self.project:
            return
        st, proj = self.settings(), self.project
        self._run(lambda progress, cancel: proj.prepare(st, progress=progress, cancel=cancel, reclassify=True),
                  self._analysed, "Detecting speech/music again…")

    def _analysed(self, _):
        p = self.project
        self.timeline.set_data(p.levels, p.segments, p.duration, wave=p.wave, audio=p.orig)
        self.timeline.ref = p.tone_ref
        self.timeline.trim = p.trim
        if self._pending_project is not None:
            self._apply_project(self._pending_project)
            self._pending_project = None
            self._set_dirty(False)
        n_music = sum(1 for s in p.segments if s.kind == MUSIC)
        t_music = sum(s.duration for s in p.segments if s.kind == MUSIC)
        n_mix = sum(1 for s in p.segments if s.kind == MIX)
        t_mix = sum(s.duration for s in p.segments if s.kind == MIX)
        msg = f"{fmt(p.duration)} min · {n_music} music sections ({fmt(t_music)})"
        if n_mix:
            msg += f" · {n_mix}× speech + music ({fmt(t_mix)})"
        self.statusBar().showMessage(msg + " detected")
        self._set_busy(False)

    # ------------------------------------------------------------------ project files
    def open_project(self, path: Path):
        if not self._confirm_discard():
            return
        try:
            lp = project_file.load(path)
        except (OSError, ValueError, KeyError) as err:
            QMessageBox.critical(self, "Open project", f"The project could not be opened:\n{err}")
            return
        src = lp.source
        if src is None:   # audio moved: let the user find it
            QMessageBox.information(self, "Open project", f"The audio file was not found:\n{lp.source_abs}"
                                    "\n\nPlease locate it.")
            fn, _ = QFileDialog.getOpenFileName(self, "Locate audio file", str(path.parent), AUDIO_FILTER)
            if not fn:
                return
            src = Path(fn)
        self._pending_project = lp
        self._project_path = lp.path
        self.panel.apply(lp.settings, lp.gui)     # show the project's settings right away
        self.load(src)
        self._set_dirty(False)

    def _apply_project(self, lp: project_file.Loaded):
        """After the analysis: put the project's sections, cut and tone reference in place."""
        p = self.project
        if lp.segments:
            p.segments = lp.segments
            p.save_segments()
        try:
            p.set_trim(lp.trim)
        except ValueError:
            p.set_trim(None)
        p.set_tone_ref(lp.tone_ref)
        self.timeline.set_segments(p.segments)
        self.timeline.trim, self.timeline.ref = p.trim, p.tone_ref
        self.timeline.update()

    def save_project(self, ask: bool = False) -> bool:
        if not self.project:
            return False
        path = self._project_path
        if ask or path is None:
            default = path or self.project.source.with_suffix(project_file.SUFFIX)
            fn, _ = QFileDialog.getSaveFileName(self, "Save project", str(default), f"{APP_NAME} project (*.klang)")
            if not fn:
                return False
            path = Path(fn)
        p = self.project
        self._project_path = project_file.save(path, p.source, p.segments, p.trim, p.tone_ref, self.settings(),
                                               self.panel.gui_state())
        self._set_dirty(False)
        self.statusBar().showMessage(f"Project saved: {self._project_path}")
        return True

    def _set_dirty(self, dirty: bool = True):
        self._dirty = dirty and self.project is not None
        name = (self._project_path.name if self._project_path
                else self.project.source.name if self.project else "")
        self.setWindowTitle(APP_NAME + (f" – {name}" if name else "") + (" •" if self._dirty else ""))
        self.setWindowModified(self._dirty)

    def _confirm_discard(self) -> bool:
        """True if it is fine to drop the current project (saved, discarded, or nothing to lose)."""
        if not self._dirty or os.environ.get("KLANGRESTAURATOR_NO_PROMPT"):
            return True
        r = QMessageBox.question(self, "Unsaved changes", "The project has unsaved changes. Save now?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
        if r == QMessageBox.Save:
            return self.save_project()
        return r == QMessageBox.Discard

    # ------------------------------------------------------------------ timeline edits
    def _trim_at_head(self, start: bool):
        if not self.project or not self.project.analysed:
            return
        t = self.timeline.head
        a, b = self.project.trim or (0.0, self.project.duration)
        self._on_trim(*((t, b) if start else (a, t)))

    def _on_trim(self, a: float, b: float):
        if not self.project:
            return
        if 0 <= a <= 0.01 and b >= self.project.duration - 0.01:
            a = b = -1.0                                  # markers back at the edges: no trim
        try:
            self.project.set_trim(None if a < 0 else (a, b))
        except ValueError as err:
            self.statusBar().showMessage(f"Cannot trim: {err}")
            return
        self.timeline.trim = self.project.trim
        self.timeline.update()
        self._set_dirty()
        tr = self.project.trim
        self.statusBar().showMessage(
            f"Export from {fmt(tr[0])} to {fmt(tr[1])} ({fmt(tr[1] - tr[0])} min)" if tr else "Trim removed")

    def _on_boundary(self, old: float, new: float):
        if not self.project:
            return
        self.project.move_boundary(old, new)
        self.timeline.set_segments(self.project.segments)
        self.statusBar().showMessage(f"Boundary moved: {fmt(old)} → {fmt(new)}")
        self._mark_stale()

    def _on_tone_ref(self, a: float, b: float):
        if not self.project:
            return
        rng = None if a < 0 else (a, b)
        self.project.set_tone_ref(rng)
        self.timeline.ref = rng
        self.timeline.update()
        self.statusBar().showMessage("Tone reference: " + (f"{fmt(a)} – {fmt(b)}" if rng else "automatic"))
        self._mark_stale()

    def _on_relabel(self, a: float, b: float, kind: str):
        if not self.project:
            return
        self.project.set_kind(a, b, kind)
        self.timeline.set_segments(self.project.segments)
        self._mark_stale()

    def _on_selection(self, a: float, b: float):
        if b > a:
            self.statusBar().showMessage(f"Selection {fmt(a)} – {fmt(b)} ({b - a:.1f} s)")
            self.listen.loop_hint()
        # once a preview exists, a new selection (>= 1 s) becomes the new preview range
        if b - a >= 1 and self.listen.range and self.project and not self.listen.orig_only:
            rng = (a, min(b, a + MAX_PREVIEW_S, self.project.duration))
            if rng != self.listen.range:
                self._next_range = rng
                self._preview_stale = True
                self.listen.lbl_prev.setText(f"Selection changed – computing preview {fmt(rng[0])} – {fmt(rng[1])}…")
                self._auto_timer.start()

    # ------------------------------------------------------------------ preview
    def make_preview(self, rng: tuple[float, float] | None = None, resume: bool = False):
        """Compute a preview for ``rng`` (default: selection, else PREVIEW_S from the cursor).

        With ``resume`` the old preview keeps playing and the new one takes over at the same position.
        """
        if not self.project or self._job:
            return
        self._auto_timer.stop()
        if not resume:
            self._next_range = None  # an explicit request wins over a queued selection change
            self.listen.stop_playback()
        a, b = rng or self._default_preview_range()
        st, proj = self.settings(), self.project
        key = self._preview_key(st, a, b)

        def done(files, cached=False):
            if not cached:
                self._remember_preview(key, files)
            self._preview_stale = False
            self.listen.show_preview(files, (a, b), resume)
            self._set_busy(False)
            msg = "Preview updated" if resume else f"Preview ready ({b - a:.0f} s)"
            self.statusBar().showMessage(msg + (" · from memory" if cached else ""))

        hit = self._previews.get(key)
        if hit and all(f.exists() for f in hit):  # computed before: switch instantly
            self._previews.move_to_end(key)
            done(hit, cached=True)
            return
        self._slot_seq += 1
        slot = self._slot_seq

        def job(progress, cancel):
            progress("Computing preview", 0.2)
            r = proj.write_preview(st, a, b, cancel=cancel, slot=slot)
            progress("Computing preview", 1)
            return r

        self._run(job, done, "Updating preview…" if resume else "Computing preview…", preview=True)

    def _default_preview_range(self) -> tuple[float, float]:
        """The selection, else PREVIEW_S from the cursor; never longer than MAX_PREVIEW_S."""
        a, b = self.timeline.sel
        if b - a < 1:
            a = self.timeline.head
            b = a + PREVIEW_S
        return a, min(b, a + MAX_PREVIEW_S, self.project.duration)

    def _preview_key(self, st: Settings, a: float, b: float) -> tuple:
        proj = self.project
        segs = tuple((s.start, s.end, s.kind, s.manual) for s in proj.segments)
        return (str(proj.source), st.effective(), round(a, 3), round(b, 3), segs, proj.tone_ref)

    def _remember_preview(self, key: tuple, files: tuple[Path, Path]) -> None:
        self._previews[key] = files
        self._previews.move_to_end(key)
        while len(self._previews) > PREVIEW_KEEP:
            _, old = self._previews.popitem(last=False)
            in_use = set(self.listen.files_ab or ())
            for f in old:
                if f not in in_use:
                    f.unlink(missing_ok=True)

    def _clear_preview_files(self) -> None:
        """Forget all remembered previews and delete their files (also leftovers of earlier sessions)."""
        self._previews.clear()
        d = Path(tempfile.gettempdir()) / "klangrestaurator_preview"
        in_use = set(self.listen.files_ab or ()) if hasattr(self, "listen") else set()
        for f in d.glob("*.wav") if d.exists() else ():
            if f not in in_use:
                f.unlink(missing_ok=True)

    # ------------------------------------------------------------------ export
    def export_dialog(self):
        if not self.project or not self.project.analysed:
            return
        default = default_output_name(self.project.source, self.panel.cmb_preset.currentText())
        fn, _ = QFileDialog.getSaveFileName(self, "Export as", str(default), "WAV 24 bit (*.wav)")
        if not fn:
            return
        st, proj = self.settings(), self.project
        out_sr = int(proj.meta.get("sr_in", SR)) if self.panel.keep_source_rate else SR
        self.listen.stop_playback()
        self._run(lambda progress, cancel: proj.export(st, fn, out_sr=out_sr, progress=progress, cancel=cancel),
                  lambda p: (self._set_busy(False), self.statusBar().showMessage(f"Saved: {p}")),
                  "Exporting…")

    def about(self):
        QMessageBox.about(self, APP_NAME,
                          f"<b>{APP_NAME} {__version__}</b><p>Restores recordings where speech and music take "
                          "turns. © 2026 Thomas Treu, MIT License.</p>"
                          "<p>Built on: <a href='https://github.com/Rikorose/DeepFilterNet'>DeepFilterNet</a> "
                          "(Schröter et al.; MIT/Apache-2.0), "
                          "<a href='https://github.com/timsainb/noisereduce'>noisereduce</a> (Sainburg; MIT), "
                          "<a href='https://github.com/fgnt/nara_wpe'>nara_wpe</a> (Drude et al.; MIT), "
                          "Qt / PySide6 (LGPL-3.0), NumPy, SciPy, python-soundfile / libsndfile (LGPL-2.1).</p>"
                          "<p>Full list and licenses: THIRD_PARTY_NOTICES.md. "
                          "Adobe and Audition are trademarks of Adobe Inc.; not affiliated.</p>")

    def closeEvent(self, e):
        if not self._confirm_discard():
            e.ignore()
            return
        if self._job:
            self._job.cancel()
        worker.wait_all()
        self.listen.release()
        self._clear_preview_files()
        super().closeEvent(e)


def main() -> int:
    app = QApplication(sys.argv)
    style.apply(app)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    w = MainWindow()
    w.show()
    if len(sys.argv) > 1 and Path(sys.argv[1]).exists():
        w.load(Path(sys.argv[1]))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
