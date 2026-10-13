"""Dark / light themes: a Fusion palette plus a small stylesheet."""
from __future__ import annotations

from PySide6.QtGui import QColor, QGuiApplication, QPalette
from PySide6.QtCore import Qt

THEMES = {
    "dark": dict(
        window="#16171a", panel="#1d1f23", panel2="#25282d", input="#2a2d33",
        border="#30343b", text="#e7e7e9", muted="#959ba4", accent="#e8a33d",
        accent_text="#17181a", hover="#2d3138", selection="#373c45",
        viewer="#0e0f11", groove="#3a3e46", handle="#eceef1", btn="#2d3137",
    ),
    "light": dict(
        window="#eceef2", panel="#fbfbfc", panel2="#f2f3f6", input="#ffffff",
        border="#d5d8de", text="#1c1e22", muted="#666c76", accent="#c47a12",
        accent_text="#ffffff", hover="#e6e9ee", selection="#dce1e8",
        viewer="#dfe2e7", groove="#cdd1d8", handle="#ffffff", btn="#ffffff",
    ),
}

QSS = """
QMainWindow, QDialog {{ background: {window}; }}
QToolBar {{ background: {panel}; border: none; border-bottom: 1px solid {border};
           spacing: 2px; padding: 4px 6px; }}
QToolBar::separator {{ background: {border}; width: 1px; margin: 4px 6px; }}
QToolButton {{ padding: 5px 10px; border-radius: 6px; border: 1px solid transparent; }}
QToolButton:hover {{ background: {hover}; }}
QToolButton:pressed {{ background: {selection}; }}
QToolButton:checked {{ background: {accent}; color: {accent_text}; }}
QToolButton:disabled {{ color: {muted}; }}
QToolButton#Primary {{ background: {accent}; color: {accent_text}; font-weight: 600; }}
QToolButton#Primary:disabled {{ background: {groove}; color: {muted}; }}
QMenuBar {{ background: {panel}; }}
QMenuBar::item:selected {{ background: {hover}; border-radius: 4px; }}
QMenu {{ background: {panel}; border: 1px solid {border}; padding: 4px; }}
QMenu::item {{ padding: 5px 22px 5px 22px; border-radius: 4px; }}
QMenu::item:selected {{ background: {accent}; color: {accent_text}; }}
QMenu::separator {{ height: 1px; background: {border}; margin: 4px 8px; }}

QWidget#Panel, QScrollArea#PanelScroll, QScrollArea#PanelScroll > QWidget > QWidget {{
    background: {panel}; }}
QListWidget#Filmstrip {{ background: {panel}; border: none; outline: 0; padding: 4px; }}
QListWidget#Filmstrip::item {{ border-radius: 6px; padding: 4px; color: {muted}; }}
QListWidget#Filmstrip::item:hover {{ background: {hover}; }}
QListWidget#Filmstrip::item:selected {{ background: {selection}; color: {text}; }}

QLabel#Muted {{ color: {muted}; }}
QLabel#PaneHeader {{ color: {muted}; font-weight: 600; font-size: 8pt; letter-spacing: 1px;
                     padding: 10px 12px 6px 12px; background: {panel}; }}

QFrame#SectionCard {{ background: {panel2}; border: 1px solid {border}; border-radius: 10px; }}
QWidget#SectionBody {{ background: transparent; }}
QToolButton#SectionReset {{ background: transparent; border: none; color: {muted};
                            padding: 2px 7px; font-size: 11pt; }}
QToolButton#SectionReset:hover {{ color: {accent}; background: {hover}; }}
QFrame#Footer {{ background: {panel}; border-top: 1px solid {border}; }}

QPushButton {{ background: {btn}; border: 1px solid {border}; border-radius: 6px;
               padding: 5px 10px; }}
QPushButton:hover {{ background: {hover}; }}
QPushButton:pressed {{ background: {selection}; }}
QPushButton:checked {{ background: {accent}; color: {accent_text}; border-color: {accent}; }}
QPushButton:disabled {{ color: {muted}; }}
QPushButton#Primary {{ background: {accent}; color: {accent_text}; border-color: {accent};
                       font-weight: 600; }}
QPushButton#Segment {{ border-radius: 0; padding: 6px 8px; }}
QPushButton#SegmentL {{ border-top-right-radius: 0; border-bottom-right-radius: 0; padding: 6px 8px; }}
QPushButton#SegmentR {{ border-top-left-radius: 0; border-bottom-left-radius: 0;
                        border-left: none; padding: 6px 8px; }}

QPushButton#SectionHeader {{ background: transparent; border: none; border-radius: 10px;
                             padding: 9px 12px; text-align: left; font-weight: 600;
                             font-size: 8pt; letter-spacing: 1px; color: {text}; }}
QPushButton#SectionHeader:hover {{ color: {accent}; background: transparent; }}
QPushButton#SectionHeader:checked {{ background: transparent; color: {text}; }}
QPushButton#SectionHeader:checked:hover {{ color: {accent}; }}
QLineEdit#Value {{ background: transparent; border: 1px solid transparent; border-radius: 4px;
                   color: {text}; padding: 1px 3px; }}
QLineEdit#Value:hover {{ border: 1px solid {border}; }}
QLineEdit#Value:focus {{ background: {input}; border: 1px solid {accent}; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {groove}; border-radius: 4px; min-height: 30px; margin: 2px; }}
QScrollBar::handle:vertical:hover {{ background: {muted}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {groove}; border-radius: 4px; min-width: 30px; margin: 2px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QSplitter::handle {{ background: {border}; }}
QStatusBar {{ background: {panel}; border-top: 1px solid {border}; color: {muted}; }}
QStatusBar::item {{ border: none; }}
QProgressBar {{ background: {input}; border: 1px solid {border}; border-radius: 5px;
                text-align: center; height: 14px; }}
QProgressBar::chunk {{ background: {accent}; border-radius: 4px; }}
QToolTip {{ background: {panel2}; color: {text}; border: 1px solid {border}; padding: 4px; }}
"""


def system_theme() -> str:
    try:
        scheme = QGuiApplication.styleHints().colorScheme()
        return "light" if scheme == Qt.ColorScheme.Light else "dark"
    except Exception:
        return "dark"


def apply_theme(app, name: str) -> dict:
    if name not in THEMES:
        name = system_theme()
    c = THEMES[name]
    q = {k: QColor(v) for k, v in c.items()}
    pal = QPalette()
    pal.setColor(QPalette.Window, q["window"])
    pal.setColor(QPalette.WindowText, q["text"])
    pal.setColor(QPalette.Base, q["input"])
    pal.setColor(QPalette.AlternateBase, q["panel2"])
    pal.setColor(QPalette.ToolTipBase, q["panel2"])
    pal.setColor(QPalette.ToolTipText, q["text"])
    pal.setColor(QPalette.Text, q["text"])
    pal.setColor(QPalette.PlaceholderText, q["muted"])
    pal.setColor(QPalette.Button, q["panel2"])
    pal.setColor(QPalette.ButtonText, q["text"])
    pal.setColor(QPalette.BrightText, QColor("#ff5555"))
    pal.setColor(QPalette.Highlight, q["accent"])
    pal.setColor(QPalette.HighlightedText, q["accent_text"])
    pal.setColor(QPalette.Link, q["accent"])
    pal.setColor(QPalette.Light, q["hover"])
    pal.setColor(QPalette.Midlight, q["border"])
    pal.setColor(QPalette.Mid, q["border"])
    pal.setColor(QPalette.Dark, q["window"])
    pal.setColor(QPalette.Shadow, QColor(0, 0, 0, 120))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, q["muted"])
    app.setPalette(pal)
    app.setStyleSheet(QSS.format(**c))
    return dict(c, name=name)
