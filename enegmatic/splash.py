"""Start-up splash screen and app icon.

Drawn in Century Gothic when it is installed (it ships with Windows / Office but
may not be redistributed), otherwise in the bundled Comfortaa font.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QEasingCurve, QPointF, QPropertyAnimation, QRectF, Qt
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QFontMetricsF, QGuiApplication, QIcon,
                           QLinearGradient, QPainter, QPainterPath, QPixmap)
from PySide6.QtWidgets import QSplashScreen

FONT_FILE = Path(__file__).parent / "assets" / "Comfortaa.ttf"
ACCENT = "#e8a33d"
# Pixel sizes, so the splash looks the same on Windows and macOS (point sizes differ).
TITLE_PX = 61
SMALL_PX = 11       # credit line and loading message share one size
PREFERRED = "Century Gothic"
# (title weight, small-text weight) per family; Comfortaa is drawn heavier to match
WEIGHTS = {PREFERRED: (400, 400), "Comfortaa": (700, 500)}


def load_brand_font() -> str:
    """Family for the splash and icon: Century Gothic if installed, else bundled Comfortaa."""
    if PREFERRED in QFontDatabase.families():
        return PREFERRED
    fid = QFontDatabase.addApplicationFont(str(FONT_FILE))
    families = QFontDatabase.applicationFontFamilies(fid) if fid >= 0 else []
    if families:
        return families[0]
    return "Comfortaa" if "Comfortaa" in QFontDatabase.families() else ""


def brand_font(family: str, size: float, weight: int = 700) -> QFont:
    f = QFont(family) if family else QFont()
    f.setPointSizeF(size)
    f.setWeight(QFont.Weight(weight))
    if family == "Comfortaa":
        try:  # variable font: pick the weight axis explicitly (Qt 6.7+)
            f.setVariableAxis(QFont.Tag("wght"), float(weight))
        except (AttributeError, TypeError):
            pass
    return f


def icon_pixmap(family: str, size: int, mac: bool = False) -> QPixmap:
    """The app icon (amber tile with an "e") drawn natively at one size.

    mac=True follows Apple's icon grid: the tile is 824/1024 of the canvas,
    centred, so it lines up with other apps in the Dock.
    """
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
    inset = size * 100 / 1024 if mac else 0.0
    tile = size - 2 * inset
    rect = QRectF(inset, inset, tile, tile)
    path = QPainterPath()
    path.addRoundedRect(rect, tile * 0.22, tile * 0.22)
    g = QLinearGradient(rect.topLeft(), rect.bottomRight())
    g.setColorAt(0, QColor("#f0b04e"))
    g.setColorAt(1, QColor("#c9771a"))
    p.fillPath(path, g)
    p.setPen(QColor("#17181a"))
    font = brand_font(family, 1, 700)
    font.setPixelSize(max(6, round(tile * 0.78)))
    p.setFont(font)
    glyph = QFontMetricsF(font).tightBoundingRect("e")   # centre the glyph itself
    p.drawText(QPointF(inset + (tile - glyph.width()) / 2 - glyph.left(),
                       inset + (tile - glyph.height()) / 2 - glyph.top()), "e")
    p.end()
    return pm


def app_icon(family: str) -> QIcon:
    """Window / Dock icon: the pre-rendered icon file for this platform (identical to the
    .exe / .app icon on every machine), or drawn live if the file is missing."""
    name = "enegmatic.icns" if sys.platform == "darwin" else "enegmatic.ico"
    path = FONT_FILE.parent / name
    if path.exists():
        icon = QIcon(str(path))
        if not icon.isNull():
            return icon
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(icon_pixmap(family, size, mac=sys.platform == "darwin"))
    return icon


class Splash(QSplashScreen):
    W, H = 560, 320

    def __init__(self, family: str, dark: bool = True):
        self.family = family
        self.dark = dark
        screen = QGuiApplication.primaryScreen()
        dpr = screen.devicePixelRatio() if screen else 1.0
        pm = QPixmap(int(self.W * dpr), int(self.H * dpr))
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.transparent)
        self._paint(pm)
        super().__init__(pm, Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._anim = None

    def _paint(self, pm: QPixmap):
        dark = self.dark
        bg0, bg1 = ("#1d1f24", "#111215") if dark else ("#ffffff", "#eceef2")
        text, muted = ("#eceef1", "#8d939c") if dark else ("#1c1e22", "#6a707a")
        hole = QColor(255, 255, 255, 18) if dark else QColor(0, 0, 0, 16)
        W, H = self.W, self.H

        p = QPainter(pm)
        p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        card = QPainterPath()
        card.addRoundedRect(QRectF(0.5, 0.5, W - 1, H - 1), 20, 20)
        g = QLinearGradient(0, 0, 0, H)
        g.setColorAt(0, QColor(bg0))
        g.setColorAt(1, QColor(bg1))
        p.fillPath(card, g)
        p.setPen(QColor(255, 255, 255, 22) if dark else QColor(0, 0, 0, 30))
        p.drawPath(card)

        # film-strip sprocket holes along the top and bottom edges
        p.setClipPath(card)
        p.setPen(Qt.NoPen)
        p.setBrush(hole)
        for x in range(18, W, 34):
            p.drawRoundedRect(QRectF(x, 16, 18, 12), 3, 3)
            p.drawRoundedRect(QRectF(x, H - 28, 18, 12), 3, 3)

        # title: accent "e" + "Negmatic", centred as one word
        title_w, small_w = WEIGHTS.get(self.family, (700, 500))
        title = brand_font(self.family, 12, title_w)
        title.setPixelSize(TITLE_PX)
        fm = QFontMetricsF(title)
        a, b = "e", "Negmatic"
        wa, wb = fm.horizontalAdvance(a), fm.horizontalAdvance(b)
        x = (W - wa - wb) / 2
        base_y = H / 2 + fm.ascent() / 2 - 14
        p.setFont(title)
        p.setPen(QColor(ACCENT))
        p.drawText(x, base_y, a)
        p.setPen(QColor(text))
        p.drawText(x + wa, base_y, b)

        p.setFont(self._small_font())
        p.setPen(QColor(muted))
        p.drawText(QRectF(0, base_y + 30, W, 24), Qt.AlignHCenter | Qt.AlignTop, "made by Rtx9 & Claude")
        p.end()

    def _small_font(self) -> QFont:
        f = brand_font(self.family, 8, WEIGHTS.get(self.family, (700, 500))[1])
        f.setPixelSize(SMALL_PX)
        f.setLetterSpacing(QFont.AbsoluteSpacing, 0.6)
        return f

    def drawContents(self, painter: QPainter):
        msg = self.message()
        if not msg:
            return
        painter.setFont(self._small_font())
        painter.setPen(QColor("#6f757e") if self.dark else QColor("#8a9099"))
        painter.drawText(QRectF(0, self.H - 66, self.W, 20), Qt.AlignHCenter | Qt.AlignVCenter, msg)

    def status(self, text: str):
        self.showMessage(text)
        QGuiApplication.processEvents()

    def fade_to(self, window, ms: int = 260):
        """Fade out and close once the main window is up."""
        self._anim = QPropertyAnimation(self, b"windowOpacity", self)
        self._anim.setDuration(ms)
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.0)
        self._anim.setEasingCurve(QEasingCurve.InOutQuad)
        self._anim.finished.connect(lambda: self.finish(window))
        self._anim.start()
