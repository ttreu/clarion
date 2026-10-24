"""Look of the app: a calm, dark "studio" theme plus vector icons for the transport.

Colours live in ``C``; the stylesheet and the custom widgets (timeline, level meter) read from it,
so the whole app changes together.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPalette, QPen, QPixmap
from PySide6.QtWidgets import QApplication


class C:
    bg = "#17181c"          # window
    card = "#212329"        # panels
    card_hi = "#2a2d34"     # inputs, hovered rows
    line = "#33363e"        # borders
    text = "#e7e8ec"
    muted = "#9097a3"
    faint = "#5c626d"
    accent = "#e8a33d"      # warm amber: primary actions, playhead, In/Out
    accent_hi = "#f3b453"
    speech = "#5b8def"
    music = "#e8a33d"
    mix = "#a77ce8"
    good = "#4cc38a"
    warn = "#e8c53d"
    bad = "#ec5f5f"
    wave = "#c9ccd4"


STYLESHEET = f"""
QWidget {{ color: {C.text}; font-size: 13px; }}
QMainWindow, QWidget#root {{ background: {C.bg}; }}
QToolTip {{ background: {C.card_hi}; color: {C.text}; border: 1px solid {C.line}; padding: 6px; }}

QFrame#header {{ background: {C.card}; border-bottom: 1px solid {C.line}; }}
QLabel#appname {{ font-size: 17px; font-weight: 600; letter-spacing: 0.3px; }}
QLabel#filename {{ color: {C.muted}; font-size: 13px; }}
QLabel#hint, QLabel#muted {{ color: {C.muted}; font-size: 11px; }}
QLabel#time {{ font-family: Menlo, monospace; font-size: 20px; color: {C.text}; }}
QLabel#section {{ color: {C.muted}; font-size: 11px; font-weight: 600; letter-spacing: 1px; }}

QFrame#card {{ background: {C.card}; border: 1px solid {C.line}; border-radius: 10px; }}

QGroupBox {{
    background: {C.card}; border: 1px solid {C.line}; border-radius: 10px;
    margin-top: 14px; padding: 12px 10px 8px 10px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left; left: 12px; padding: 0 4px;
    color: {C.muted}; font-size: 11px; font-weight: 600;
}}
QGroupBox::indicator {{ width: 14px; height: 14px; }}

QPushButton {{
    background: {C.card_hi}; border: 1px solid {C.line}; border-radius: 7px; padding: 6px 14px;
}}
QPushButton:hover {{ border-color: {C.faint}; }}
QPushButton:pressed {{ background: {C.line}; }}
QPushButton:disabled {{ color: {C.faint}; }}
QPushButton#primary {{ background: {C.accent}; border-color: {C.accent}; color: #1b1406; font-weight: 600; }}
QPushButton#primary:hover {{ background: {C.accent_hi}; }}
QPushButton#primary:disabled {{ background: {C.card_hi}; border-color: {C.line}; color: {C.faint}; }}

QPushButton#preview {{
    background: #3a3020; border: 1px solid {C.accent}; border-radius: 9px; padding: 10px 16px;
    color: {C.accent_hi}; font-size: 14px; font-weight: 600;
}}
QPushButton#preview:hover {{ background: #4a3b22; border-color: {C.accent_hi}; }}
QPushButton#preview:pressed {{ background: #2e2618; }}
QPushButton#preview:disabled {{ background: {C.card_hi}; border-color: {C.line}; color: {C.faint}; }}

QToolButton {{
    background: {C.card_hi}; border: 1px solid {C.line}; border-radius: 8px; padding: 6px;
}}
QToolButton:hover {{ border-color: {C.faint}; }}
QToolButton:checked {{ background: #3a3020; border-color: {C.accent}; }}
QToolButton:disabled {{ background: {C.card}; }}
QToolButton#play {{ background: {C.accent}; border-color: {C.accent}; border-radius: 20px; }}
QToolButton#play:hover {{ background: {C.accent_hi}; }}
QToolButton#play:disabled {{ background: {C.card_hi}; border-color: {C.line}; }}
QToolButton::menu-button {{ border: none; width: 14px; }}

QComboBox {{
    background: {C.card_hi}; border: 1px solid {C.line}; border-radius: 7px; padding: 4px 10px;
}}
QComboBox:hover {{ border-color: {C.faint}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {C.card_hi}; border: 1px solid {C.line}; selection-background-color: {C.accent};
    selection-color: #1b1406; outline: none;
}}

QSlider::groove:horizontal {{ height: 4px; background: {C.line}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {C.accent}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {C.text}; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{ background: #ffffff; }}
QSlider:disabled {{ opacity: 0.5; }}

QCheckBox, QRadioButton {{ spacing: 7px; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 15px; height: 15px; }}
QCheckBox::indicator {{ border: 1px solid {C.faint}; border-radius: 4px; background: {C.card_hi}; }}
QCheckBox::indicator:checked {{ background: {C.accent}; border-color: {C.accent}; }}
QRadioButton::indicator {{
    width: 14px; height: 14px; border: 1px solid {C.faint}; border-radius: 7px; background: {C.card_hi};
}}
QRadioButton::indicator:checked {{
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
                                stop:0 {C.accent}, stop:0.45 {C.accent}, stop:0.55 {C.card_hi}, stop:1 {C.card_hi});
    border: 1px solid {C.accent};
}}

QPushButton#seg {{ border-radius: 0; padding: 6px 12px; }}
QPushButton#segL {{ border-top-right-radius: 0; border-bottom-right-radius: 0; padding: 6px 12px; }}
QPushButton#segR {{ border-top-left-radius: 0; border-bottom-left-radius: 0; padding: 6px 12px; }}
QPushButton#segL:checked, QPushButton#segR:checked {{
    background: #3a3020; border-color: {C.accent}; color: {C.accent_hi}; font-weight: 600;
}}

QProgressBar {{
    background: {C.card_hi}; border: 1px solid {C.line}; border-radius: 6px; height: 14px;
    text-align: center; color: {C.muted}; font-size: 11px;
}}
QProgressBar::chunk {{ background: {C.accent}; border-radius: 5px; }}
QStatusBar {{ background: {C.card}; color: {C.muted}; border-top: 1px solid {C.line}; }}
QMenuBar {{ background: {C.card}; }}
QMenu {{ background: {C.card_hi}; border: 1px solid {C.line}; }}
QMenu::item:selected {{ background: {C.accent}; color: #1b1406; }}
QScrollBar {{ background: transparent; }}
"""

# the three section groups in the control panel carry their timeline colour: title, a stripe on the
# left, sliders and checkboxes (objectName "speech" / "music" / "mix")
for _name, _col in (("speech", C.speech), ("music", C.music), ("mix", C.mix)):
    STYLESHEET += f"""
QGroupBox#{_name} {{ border-left: 3px solid {_col}; }}
QGroupBox#{_name}::title {{ color: {_col}; }}
QGroupBox#{_name} QSlider::sub-page:horizontal {{ background: {_col}; }}
QGroupBox#{_name} QSlider::handle:horizontal:hover {{ background: {_col}; }}
QGroupBox#{_name} QCheckBox::indicator:checked {{ background: {_col}; border-color: {_col}; }}
QGroupBox#{_name} QComboBox:hover, QGroupBox#{_name} QComboBox:focus {{ border-color: {_col}; }}
QGroupBox#{_name} QComboBox QAbstractItemView {{ selection-background-color: {_col}; }}
"""


def apply(app: QApplication) -> None:
    """Dark palette (for widgets that paint from it) + stylesheet."""
    app.setStyle("Fusion")
    pal = QPalette()
    for role, col in ((QPalette.Window, C.bg), (QPalette.Base, "#131418"), (QPalette.AlternateBase, C.card),
                      (QPalette.Text, C.text), (QPalette.WindowText, C.text), (QPalette.Button, C.card_hi),
                      (QPalette.ButtonText, C.text), (QPalette.Highlight, C.accent),
                      (QPalette.HighlightedText, "#1b1406"), (QPalette.ToolTipBase, C.card_hi),
                      (QPalette.ToolTipText, C.text), (QPalette.PlaceholderText, C.faint)):
        pal.setColor(role, QColor(col))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(C.faint))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(C.faint))
    pal.setColor(QPalette.Disabled, QPalette.WindowText, QColor(C.faint))
    app.setPalette(pal)
    app.setStyleSheet(STYLESHEET)


# --- transport icons (vector, crisp at any size and on dark ground) ---------------------------

def icon(kind: str, color: str = C.text, size: int = 18) -> QIcon:
    pm = QPixmap(size * 2, size * 2)          # 2× for Retina
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.scale(2, 2)
    col = QColor(color)
    p.setBrush(col)
    p.setPen(Qt.NoPen)
    s = float(size)

    def tri(x0, x1, left=False):
        path = QPainterPath()
        if left:
            path.moveTo(x1, s * 0.2)
            path.lineTo(x0, s * 0.5)
            path.lineTo(x1, s * 0.8)
        else:
            path.moveTo(x0, s * 0.2)
            path.lineTo(x1, s * 0.5)
            path.lineTo(x0, s * 0.8)
        path.closeSubpath()
        p.drawPath(path)

    if kind == "play":
        tri(s * 0.3, s * 0.82)
    elif kind == "pause":
        p.drawRoundedRect(QRectF(s * 0.26, s * 0.2, s * 0.17, s * 0.6), 1.5, 1.5)
        p.drawRoundedRect(QRectF(s * 0.57, s * 0.2, s * 0.17, s * 0.6), 1.5, 1.5)
    elif kind == "stop":
        p.drawRoundedRect(QRectF(s * 0.25, s * 0.25, s * 0.5, s * 0.5), 2, 2)
    elif kind == "start":
        p.drawRect(QRectF(s * 0.2, s * 0.22, s * 0.1, s * 0.56))
        tri(s * 0.3, s * 0.8, left=True)
    elif kind == "back":
        tri(s * 0.1, s * 0.48, left=True)
        tri(s * 0.48, s * 0.86, left=True)
    elif kind == "fwd":
        tri(s * 0.14, s * 0.52)
        tri(s * 0.52, s * 0.9)
    elif kind == "headphones":
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(col, s * 0.1, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(QRectF(s * 0.16, s * 0.14, s * 0.68, s * 0.7), 0, 180 * 16)        # head band
        p.setPen(Qt.NoPen)
        p.setBrush(col)
        p.drawRoundedRect(QRectF(s * 0.1, s * 0.48, s * 0.22, s * 0.36), s * 0.08, s * 0.08)   # ear cups
        p.drawRoundedRect(QRectF(s * 0.68, s * 0.48, s * 0.22, s * 0.36), s * 0.08, s * 0.08)
    elif kind == "loop":
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(col, s * 0.1, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(QRectF(s * 0.2, s * 0.2, s * 0.6, s * 0.6), 60 * 16, 300 * 16)
        p.setPen(Qt.NoPen)
        p.setBrush(col)
        head = QPainterPath(QPointF(s * 0.62, s * 0.08))
        head.lineTo(s * 0.82, s * 0.26)
        head.lineTo(s * 0.56, s * 0.34)
        head.closeSubpath()
        p.drawPath(head)
    p.end()
    return QIcon(pm)
