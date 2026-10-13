"""Custom widgets: parameter slider, image viewer with crop / picker tools, histogram."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEvent, QLocale, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QDoubleValidator, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QLineEdit, QPinchGesture, QPushButton,
                               QSizePolicy, QSlider,
                               QToolButton, QVBoxLayout, QWidget)


SLIDER_COLORS = {"groove": "#3a3e46", "fill": "#e8a33d", "handle": "#eceef1",
                 "border": "#30343b", "tick": "#959ba4"}


def set_slider_colors(colors: dict):
    SLIDER_COLORS.update(groove=colors["groove"], fill=colors["accent"], handle=colors["handle"],
                         border=colors["border"], tick=colors["muted"])


class FancySlider(QSlider):
    """Flat slider that fills from its origin (zero for +/- controls) and jumps to clicks.

    Ignores the mouse wheel unless focused, so scrolling the panel doesn't move it.
    """

    R = 7  # handle radius

    def __init__(self, origin: int = 0, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self._origin = origin
        self._hover = False
        self.setMinimumHeight(22)
        self.setFocusPolicy(Qt.StrongFocus)

    def _x(self, v: int) -> float:
        x0, x1 = self.R + 1, self.width() - self.R - 1
        span = max(1, self.maximum() - self.minimum())
        return x0 + (x1 - x0) * (v - self.minimum()) / span

    def _value_at(self, x: float) -> int:
        x0, x1 = self.R + 1, self.width() - self.R - 1
        f = min(max((x - x0) / max(1, x1 - x0), 0.0), 1.0)
        return round(self.minimum() + f * (self.maximum() - self.minimum()))

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = SLIDER_COLORS
        cy = self.height() / 2
        x0, x1 = self.R + 1, self.width() - self.R - 1
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(c["groove"]))
        p.drawRoundedRect(QRectF(x0, cy - 2, x1 - x0, 4), 2, 2)
        xo, xv = self._x(self._origin), self._x(self.value())
        if self.minimum() < self._origin < self.maximum():
            p.setBrush(QColor(c["tick"]))
            p.drawRect(QRectF(xo - 0.5, cy - 5, 1, 10))
        p.setBrush(QColor(c["fill"] if self.isEnabled() else c["groove"]))
        p.drawRoundedRect(QRectF(min(xo, xv), cy - 2, abs(xv - xo), 4), 2, 2)
        active = self._hover or self.isSliderDown() or self.hasFocus()
        p.setPen(QPen(QColor(c["fill"] if active and self.isEnabled() else c["border"]), 2 if active else 1))
        p.setBrush(QColor(c["handle"]))
        r = self.R + (1 if self.isSliderDown() else 0)
        p.drawEllipse(QPointF(xv, cy), r, r)

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return super().mousePressEvent(e)
        self.setFocus(Qt.MouseFocusReason)
        self.setSliderDown(True)
        self.setValue(self._value_at(e.position().x()))
        e.accept()

    def mouseMoveEvent(self, e):
        if self.isSliderDown():
            self.setValue(self._value_at(e.position().x()))
            e.accept()

    def mouseReleaseEvent(self, e):
        if self.isSliderDown():
            self.setSliderDown(False)
            self.update()
            e.accept()

    def enterEvent(self, e):
        self._hover = True
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._hover = False
        self.update()
        super().leaveEvent(e)

    def wheelEvent(self, e):
        if self.hasFocus():
            super().wheelEvent(e)
        else:
            e.ignore()


class CollapsibleSection(QFrame):
    """Card with a clickable header that shows / hides its body, plus an optional reset button."""

    toggled = Signal(bool)
    resetRequested = Signal()

    def __init__(self, title: str, expanded: bool = True, resettable: bool = True, parent=None):
        super().__init__(parent)
        self.setObjectName("SectionCard")
        self._title = title
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 6, 0)
        head.setSpacing(0)
        self.header = QPushButton()
        self.header.setObjectName("SectionHeader")
        self.header.setFlat(True)
        self.header.setCheckable(True)
        self.header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.header.setCursor(Qt.PointingHandCursor)
        self.header.toggled.connect(self.set_expanded)
        head.addWidget(self.header, 1)
        if resettable:
            self.reset_btn = QToolButton()
            self.reset_btn.setObjectName("SectionReset")
            self.reset_btn.setText("↺")
            self.reset_btn.setToolTip(f"Reset {title.lower()}")
            self.reset_btn.setCursor(Qt.PointingHandCursor)
            self.reset_btn.clicked.connect(self.resetRequested)
            head.addWidget(self.reset_btn)
        lay.addLayout(head)
        self.body = QWidget()
        self.body.setObjectName("SectionBody")
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(12, 2, 12, 12)
        self.body_layout.setSpacing(4)
        lay.addWidget(self.body)
        self.header.setChecked(expanded)
        self.set_expanded(expanded)

    def set_expanded(self, on: bool):
        self.header.blockSignals(True)
        self.header.setChecked(on)
        self.header.blockSignals(False)
        self.header.setText(("▾   " if on else "▸   ") + self._title.upper().replace("&", "&&"))
        self.body.setVisible(on)
        self.toggled.emit(on)

    def is_expanded(self) -> bool:
        return self.body.isVisible() or self.header.isChecked()

    def addWidget(self, w):
        self.body_layout.addWidget(w)

    def addLayout(self, layout):
        self.body_layout.addLayout(layout)


class ParamSlider(QWidget):
    """Label | slider | editable value. Double-click the label or slider to reset."""

    valueChanged = Signal(float)

    def __init__(self, label: str, minimum: float, maximum: float, step: float,
                 default: float, decimals: int = 0, tooltip: str = "", origin: float | None = None,
                 parent=None):
        super().__init__(parent)
        self._min, self._max, self._step, self._default = minimum, maximum, step, default
        self._dec = decimals
        if origin is None:
            origin = 0.0 if minimum < 0 < maximum else minimum
        self._value = default
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 1, 0, 1)
        lay.setSpacing(8)
        self.label = QLabel(label)
        self.label.setFixedWidth(84)
        self.slider = FancySlider(round((origin - minimum) / step))
        self.slider.setRange(0, round((maximum - minimum) / step))
        self.edit = QLineEdit()
        self.edit.setObjectName("Value")
        self.edit.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.edit.setFixedWidth(50)
        validator = QDoubleValidator(minimum, maximum, decimals, self.edit)
        validator.setNotation(QDoubleValidator.StandardNotation)
        validator.setLocale(QLocale.c())
        self.edit.setValidator(validator)
        lay.addWidget(self.label)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.edit)
        self.setToolTip((tooltip + "\n" if tooltip else "") + "Double-click to reset.")
        self.slider.valueChanged.connect(self._from_slider)
        self.edit.editingFinished.connect(self._from_edit)
        self.label.installEventFilter(self)
        self.slider.installEventFilter(self)
        self.setValue(default)

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.MouseButtonDblClick:
            self.reset()
            return True
        return super().eventFilter(obj, ev)

    def _to_pos(self, v: float) -> int:
        return round((v - self._min) / self._step)

    def _show(self, v: float):
        self.edit.setText(f"{v:+.{self._dec}f}" if self._min < 0 and v else f"{v:.{self._dec}f}")

    def _from_slider(self, pos: int):
        v = round(self._min + pos * self._step, 6)
        self._value = v
        self._show(v)
        self.valueChanged.emit(v)

    def _from_edit(self):
        try:
            v = float(self.edit.text().replace(",", "."))
        except ValueError:
            self._show(self._value)
            return
        v = min(max(v, self._min), self._max)
        self._show(v)
        if abs(v - self._value) < 1e-9:
            return
        self._value = v
        self.slider.blockSignals(True)
        self.slider.setValue(self._to_pos(v))
        self.slider.blockSignals(False)
        self.valueChanged.emit(v)

    def value(self) -> float:
        return float(self._value)

    def setValue(self, v: float):
        """Set without emitting."""
        v = min(max(float(v), self._min), self._max)
        self._value = v
        self.slider.blockSignals(True)
        self.slider.setValue(self._to_pos(v))
        self.slider.blockSignals(False)
        self._show(v)

    def reset(self):
        if abs(self.value() - self._default) > 1e-9:
            self.setValue(self._default)
            self.valueChanged.emit(self._default)

class Histogram(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(84)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._hist: np.ndarray | None = None
        self._mono = False
        self.bg = QColor("#25282d")
        self.grid = QColor("#30343b")

    def set_colors(self, bg: str, grid: str):
        self.bg, self.grid = QColor(bg), QColor(grid)
        self.update()

    def set_data(self, hist: np.ndarray | None, mono: bool = False):
        self._hist, self._mono = hist, mono
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = QPainterPath()
        path.addRoundedRect(r, 8, 8)
        p.fillPath(path, self.bg)
        p.setClipPath(path)
        p.setPen(QPen(self.grid, 1))
        for i in (1, 2, 3):
            x = r.left() + r.width() * i / 4
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
        if self._hist is None:
            return
        h = np.sqrt(self._hist[:1] if self._mono else self._hist)
        inner = h[:, 2:-2]
        top = float(np.percentile(inner, 99.5)) if inner.size else 1.0
        top = max(top, 1e-6) * 1.08
        colors = ([QColor(200, 200, 200, 150)] if self._mono else
                  [QColor(235, 70, 70, 110), QColor(70, 200, 90, 110), QColor(70, 130, 245, 110)])
        mode = QPainter.CompositionMode_Plus if self.bg.lightness() < 128 else QPainter.CompositionMode_Multiply
        if not self._mono:
            p.setCompositionMode(mode)
        w, hh = r.width(), r.height() - 4
        for ch, col in zip(h, colors):
            poly = QPolygonF()
            poly.append(QPointF(r.left(), r.bottom()))
            for i, v in enumerate(ch):
                poly.append(QPointF(r.left() + w * i / 255.0, r.bottom() - min(v / top, 1.0) * hh))
            poly.append(QPointF(r.right(), r.bottom()))
            if self.bg.lightness() >= 128 and not self._mono:
                col = QColor(col.red(), col.green(), col.blue(), 90)
            elif self.bg.lightness() >= 128:
                col = QColor(90, 90, 90, 130)
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            p.drawPolygon(poly)


class ImageView(QWidget):
    """Zoomable image display with crop editing and point picking.

    Zoom: click (toggle fit / 1:1), mouse wheel, Ctrl+wheel or touchpad pinch.
    Pan: drag, or two-finger scroll on a touchpad. While zoomed past the preview's
    own resolution it asks (viewportChanged) for a sharp full-resolution tile.
    """

    cropChanged = Signal(list)
    picked = Signal(float, float)
    viewportChanged = Signal(object)

    VIEW, CROP, PICK = range(3)
    HANDLE = 9
    MARGIN = 14
    MAX_PX = 4.0        # deepest zoom: screen pixels per full-resolution pixel

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setMinimumSize(320, 240)
        self._img = None
        self._message = "Open NEF / raw / TIFF negatives to begin\n\nFile ▸ Open, or drag files here"
        self._overlay = ""
        self._mode = self.VIEW
        self._crop = [0.0, 0.0, 1.0, 1.0]
        self._aspect: float | None = None  # pixel w/h of the crop
        self.frame_size = (3, 2)           # pixel size of the uncropped frame
        self.full_size: tuple[int, int] | None = None   # displayed image at full resolution (w, h)
        self._drag = None
        self._zoom = 1.0                   # 1 = fit to window
        self._center = [0.5, 0.5]          # normalised image point shown at the widget centre
        self._tile = None                  # (QImage, (x0, y0, x1, y1)) sharp full-resolution patch
        self._pan = None                   # (start pos, start centre, moved)
        self._badge = ""
        self._vp_timer = QTimer(self, singleShot=True, interval=120)
        self._vp_timer.timeout.connect(self._emit_viewport)
        self.grabGesture(Qt.PinchGesture)
        self.bg = QColor("#0e0f11")
        self.fg = QColor("#959ba4")

    # ---- state
    def set_colors(self, bg: str, fg: str):
        self.bg, self.fg = QColor(bg), QColor(fg)
        self.update()

    def set_image(self, qimg, reset_view: bool = False):
        self._img = qimg
        if reset_view or qimg is None:
            self._zoom, self._center, self._tile = 1.0, [0.5, 0.5], None
            self._update_cursor()
        self.update()
        self._vp_timer.start()

    def set_tile(self, qimg, rect):
        self._tile = (qimg, rect)
        self.update()

    def clear_tile(self):
        if self._tile is not None:
            self._tile = None
            self.update()

    def set_badge(self, text: str):
        self._badge = text
        self.update()

    def zoomed(self) -> bool:
        return self._zoom > 1.001

    # ---- zoom / pan
    def _logical_size(self) -> tuple[float, float]:
        if self.full_size:
            return float(self.full_size[0]), float(self.full_size[1])
        return float(self._img.width()), float(self._img.height())

    def _fit_scale(self) -> float:
        m = self.MARGIN
        aw, ah = max(1, self.width() - 2 * m), max(1, self.height() - 2 * m)
        iw, ih = self._logical_size()
        return min(aw / iw, ah / ih)

    def _clamped_center(self, w: float, h: float) -> tuple[float, float]:
        cx, cy = self._center
        hx, hy = self.width() / 2 / w, self.height() / 2 / h
        cx = 0.5 if hx >= 0.5 else min(max(cx, hx), 1 - hx)
        cy = 0.5 if hy >= 0.5 else min(max(cy, hy), 1 - hy)
        return cx, cy

    def zoom_percent(self) -> float:
        """Zoom relative to full-resolution pixels (100 = 1:1)."""
        if self._img is None:
            return 0.0
        return self._fit_scale() * self._zoom * self.devicePixelRatioF() * 100.0

    def _set_zoom(self, z: float, anchor=None):
        if self._img is None:
            return
        max_zoom = max(1.0, self.MAX_PX / (self._fit_scale() * self.devicePixelRatioF()))
        z = min(max(z, 1.0), max_zoom)
        if anchor is None:
            anchor = QPointF(self.width() / 2, self.height() / 2)
        r = self._image_rect()
        nx = (anchor.x() - r.left()) / max(r.width(), 1)
        ny = (anchor.y() - r.top()) / max(r.height(), 1)
        self._zoom = z
        if z <= 1.001:
            self._zoom, self._center = 1.0, [0.5, 0.5]
        else:
            iw, ih = self._logical_size()
            s = self._fit_scale() * z
            w, h = iw * s, ih * s
            self._center = [nx - (anchor.x() - self.width() / 2) / w,
                            ny - (anchor.y() - self.height() / 2) / h]
            self._center = list(self._clamped_center(w, h))
        self._update_cursor()
        self.update()
        self._vp_timer.start()

    def zoom_by(self, factor: float, anchor=None):
        self._set_zoom(self._zoom * factor, anchor)

    def fit(self):
        self._set_zoom(1.0)

    def toggle_zoom(self, anchor=None):
        if self.zoomed():
            self.fit()
        else:
            one_to_one = 1.0 / (self._fit_scale() * self.devicePixelRatioF())
            self._set_zoom(one_to_one if one_to_one >= 1.5 else 2.5, anchor)

    def pan_by(self, dx: float, dy: float):
        if not self.zoomed():
            return
        r = self._image_rect()
        cx, cy = self._clamped_center(r.width(), r.height())
        self._center = [cx - dx / r.width(), cy - dy / r.height()]
        self._center = list(self._clamped_center(r.width(), r.height()))
        self.update()
        self._vp_timer.start()

    def _update_cursor(self):
        if self._mode == self.PICK:
            self.setCursor(Qt.CrossCursor)
        elif self._mode == self.VIEW:
            self.setCursor(Qt.OpenHandCursor if self.zoomed() else Qt.ArrowCursor)

    def _emit_viewport(self):
        if self._img is None:
            return
        r = self._image_rect()
        dpr = self.devicePixelRatioF()
        rect = (max(0.0, -r.left() / r.width()), max(0.0, -r.top() / r.height()),
                min(1.0, (self.width() - r.left()) / r.width()),
                min(1.0, (self.height() - r.top()) / r.height()))
        self.viewportChanged.emit({
            "rect": rect,
            "display_w": r.width() * dpr,            # physical pixels across the whole image
            "need": r.width() * dpr > self._img.width() * 1.05,
            "zoomed": self.zoomed(),
        })

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._vp_timer.start()

    def set_message(self, text: str):
        self._message = text
        self.update()

    def set_overlay(self, text: str):
        self._overlay = text
        self.update()

    def set_mode(self, mode: int):
        self._mode = mode
        self._drag = None
        self.setCursor(Qt.ArrowCursor)
        self._update_cursor()
        self.update()

    def mode(self) -> int:
        return self._mode

    def set_crop(self, crop):
        self._crop = list(crop) if crop else [0.0, 0.0, 1.0, 1.0]
        self.update()

    def set_aspect(self, ratio: float | None):
        """Lock the crop to a pixel aspect ratio (w/h) and fit the current crop to it."""
        self._aspect = ratio
        if ratio and self._img is not None:
            self._crop = self._fit_aspect(self._crop, ratio)
            self.update()
            self.cropChanged.emit(list(self._crop))

    def _norm_ratio(self, ratio: float) -> float:
        fw, fh = self.frame_size
        return ratio * fh / max(fw, 1)

    def _fit_aspect(self, c, ratio):
        x0, y0, x1, y1 = c
        rn = self._norm_ratio(ratio)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = x1 - x0, y1 - y0
        if w / max(h, 1e-6) > rn:
            w = h * rn
        else:
            h = w / rn
        w, h = min(w, 1.0), min(h, 1.0)
        cx = min(max(cx, w / 2), 1 - w / 2)
        cy = min(max(cy, h / 2), 1 - h / 2)
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    # ---- geometry helpers
    def _image_rect(self) -> QRectF:
        if self._img is None:
            return QRectF()
        iw, ih = self._logical_size()
        s = self._fit_scale() * self._zoom
        w, h = iw * s, ih * s
        if self._zoom <= 1.001:
            return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)
        cx, cy = self._clamped_center(w, h)
        return QRectF(self.width() / 2 - cx * w, self.height() / 2 - cy * h, w, h)

    def _to_norm(self, pos) -> tuple[float, float]:
        r = self._image_rect()
        nx = (pos.x() - r.left()) / max(r.width(), 1)
        ny = (pos.y() - r.top()) / max(r.height(), 1)
        return min(max(nx, 0.0), 1.0), min(max(ny, 0.0), 1.0)

    def _crop_rect(self) -> QRectF:
        r = self._image_rect()
        x0, y0, x1, y1 = self._crop
        return QRectF(r.left() + x0 * r.width(), r.top() + y0 * r.height(),
                      (x1 - x0) * r.width(), (y1 - y0) * r.height())

    def _handles(self) -> dict:
        c = self._crop_rect()
        pts = {
            "tl": c.topLeft(), "tr": c.topRight(), "bl": c.bottomLeft(), "br": c.bottomRight(),
        }
        if not self._aspect:
            pts.update({
                "t": QPointF(c.center().x(), c.top()), "b": QPointF(c.center().x(), c.bottom()),
                "l": QPointF(c.left(), c.center().y()), "r": QPointF(c.right(), c.center().y()),
            })
        return pts

    def _hit(self, pos) -> str | None:
        for name, pt in self._handles().items():
            if abs(pos.x() - pt.x()) <= self.HANDLE + 2 and abs(pos.y() - pt.y()) <= self.HANDLE + 2:
                return name
        if self._crop_rect().contains(pos):
            return "move"
        return None

    # ---- painting
    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), self.bg)
        if self._img is None:
            p.setPen(self.fg)
            p.drawText(self.rect(), Qt.AlignCenter, self._message)
            return
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        r = self._image_rect()
        p.drawImage(r, self._img)
        if self._tile is not None:
            timg, (tx0, ty0, tx1, ty1) = self._tile
            p.drawImage(QRectF(r.left() + tx0 * r.width(), r.top() + ty0 * r.height(),
                               (tx1 - tx0) * r.width(), (ty1 - ty0) * r.height()), timg)
        if self._mode == self.CROP:
            p.setRenderHint(QPainter.Antialiasing)
            c = self._crop_rect()
            outside = QPainterPath()
            outside.addRect(r)
            inner = QPainterPath()
            inner.addRect(c)
            p.fillPath(outside.subtracted(inner), QColor(0, 0, 0, 150))
            p.setPen(QPen(QColor(255, 255, 255, 70), 1))
            for i in (1, 2):
                x = c.left() + c.width() * i / 3
                y = c.top() + c.height() * i / 3
                p.drawLine(QPointF(x, c.top()), QPointF(x, c.bottom()))
                p.drawLine(QPointF(c.left(), y), QPointF(c.right(), y))
            p.setPen(QPen(QColor(255, 255, 255, 230), 1.5))
            p.setBrush(Qt.NoBrush)
            p.drawRect(c)
            p.setBrush(QColor(255, 255, 255))
            p.setPen(QPen(QColor(0, 0, 0, 160), 1))
            hs = self.HANDLE
            for pt in self._handles().values():
                p.drawRect(QRectF(pt.x() - hs / 2, pt.y() - hs / 2, hs, hs))
        if self._overlay:
            p.setRenderHint(QPainter.Antialiasing)
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(self._overlay) + 24
            box = QRectF((self.width() - tw) / 2, 16, tw, fm.height() + 12)
            path = QPainterPath()
            path.addRoundedRect(box, box.height() / 2, box.height() / 2)
            p.fillPath(path, QColor(0, 0, 0, 170))
            p.setPen(QColor(255, 255, 255))
            p.drawText(box, Qt.AlignCenter, self._overlay)
        if self.zoomed() or self._badge:
            p.setRenderHint(QPainter.Antialiasing)
            text = f"{self.zoom_percent():.0f}%" if self.zoomed() else ""
            if self._badge:
                text = f"{text}  ·  {self._badge}" if text else self._badge
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(text) + 22
            box = QRectF(16, self.height() - fm.height() - 28, tw, fm.height() + 10)
            path = QPainterPath()
            path.addRoundedRect(box, box.height() / 2, box.height() / 2)
            p.fillPath(path, QColor(0, 0, 0, 160))
            p.setPen(QColor(255, 255, 255))
            p.drawText(box, Qt.AlignCenter, text)

    # ---- zoom / pan input
    def wheelEvent(self, e):
        if self._img is None:
            return
        ad, pd = e.angleDelta(), e.pixelDelta()
        if e.modifiers() & Qt.ControlModifier:
            # touchpad pinch arrives as Ctrl+wheel on Windows
            steps = ad.y() / 120.0 if ad.y() else pd.y() / 60.0
            self.zoom_by(1.25 ** steps, e.position())
        elif not pd.isNull() or ad.y() % 120 or ad.x():
            # touchpad two-finger scroll: pan
            if not self.zoomed():
                e.ignore()
                return
            dx, dy = (pd.x(), pd.y()) if not pd.isNull() else (ad.x() / 2.0, ad.y() / 2.0)
            self.pan_by(dx, dy)
        else:
            self.zoom_by(1.25 ** (ad.y() / 120.0), e.position())   # mouse wheel notch
        e.accept()

    def event(self, e):
        t = e.type()
        if t == QEvent.NativeGesture:
            g = e.gestureType()
            if g == Qt.ZoomNativeGesture:
                self.zoom_by(1.0 + e.value(), e.position())
                return True
            if g == Qt.SmartZoomNativeGesture:
                self.toggle_zoom(e.position())
                return True
        elif t == QEvent.Gesture:
            pinch = e.gesture(Qt.PinchGesture)
            if pinch is not None:
                if pinch.changeFlags() & QPinchGesture.ScaleFactorChanged:
                    pos = self.mapFromGlobal(pinch.centerPoint().toPoint())
                    self.zoom_by(pinch.scaleFactor(), QPointF(pos))
                e.accept()
                return True
        return super().event(e)

    # ---- interaction
    def mousePressEvent(self, e):
        if self._img is None:
            return
        pan_button = e.button() == Qt.MiddleButton or (e.button() == Qt.LeftButton and self._mode == self.VIEW)
        if pan_button:
            self._pan = (e.position(), e.button(), False)
            return
        if e.button() != Qt.LeftButton:
            return
        if self._mode == self.PICK:
            if self._image_rect().contains(e.position()):
                self.picked.emit(*self._to_norm(e.position()))
            return
        if self._mode != self.CROP:
            return
        hit = self._hit(e.position())
        nx, ny = self._to_norm(e.position())
        if hit is None:
            self._drag = ("new", (nx, ny), list(self._crop))
        else:
            self._drag = (hit, (nx, ny), list(self._crop))

    def mouseMoveEvent(self, e):
        if self._pan is not None:
            start, button, moved = self._pan
            d = e.position() - start
            if not moved and abs(d.x()) + abs(d.y()) > 4:
                moved = True
                if self.zoomed():
                    self.setCursor(Qt.ClosedHandCursor)
            if moved:
                self.pan_by(d.x(), d.y())
                start = e.position()
            self._pan = (start, button, moved)
            return
        if self._mode != self.CROP or self._img is None:
            return
        if self._drag is None:
            hit = self._hit(e.position())
            cursors = {"tl": Qt.SizeFDiagCursor, "br": Qt.SizeFDiagCursor,
                       "tr": Qt.SizeBDiagCursor, "bl": Qt.SizeBDiagCursor,
                       "t": Qt.SizeVerCursor, "b": Qt.SizeVerCursor,
                       "l": Qt.SizeHorCursor, "r": Qt.SizeHorCursor,
                       "move": Qt.SizeAllCursor}
            self.setCursor(cursors.get(hit, Qt.CrossCursor))
            return
        kind, (sx, sy), c0 = self._drag
        nx, ny = self._to_norm(e.position())
        if kind == "new" and abs(nx - sx) < 0.01 and abs(ny - sy) < 0.01:
            return
        x0, y0, x1, y1 = c0
        mn = 0.02
        if kind == "move":
            w, h = x1 - x0, y1 - y0
            dx = min(max(nx - sx, -x0), 1 - x1)
            dy = min(max(ny - sy, -y0), 1 - y1)
            self._crop = [x0 + dx, y0 + dy, x0 + dx + w, y0 + dy + h]
        elif kind in ("new", "tl", "tr", "bl", "br"):
            if kind == "new":
                ax, ay = sx, sy
            else:
                ax = x1 if "l" in kind else x0
                ay = y1 if "t" in kind else y0
            self._crop = self._corner_rect(ax, ay, nx, ny)
        else:
            if kind == "l":
                x0 = min(nx, x1 - mn)
            elif kind == "r":
                x1 = max(nx, x0 + mn)
            elif kind == "t":
                y0 = min(ny, y1 - mn)
            elif kind == "b":
                y1 = max(ny, y0 + mn)
            self._crop = [x0, y0, x1, y1]
        self.update()

    def _corner_rect(self, ax, ay, px, py):
        dx, dy = px - ax, py - ay
        sxn = 1 if dx >= 0 else -1
        syn = 1 if dy >= 0 else -1
        w = min(abs(dx), (1 - ax) if sxn > 0 else ax)
        h = min(abs(dy), (1 - ay) if syn > 0 else ay)
        w, h = max(w, 0.02), max(h, 0.02)
        if self._aspect:
            rn = self._norm_ratio(self._aspect)
            if w / h > rn:
                w = h * rn
            else:
                h = w / rn
        xa, xb = sorted((ax, ax + sxn * w))
        ya, yb = sorted((ay, ay + syn * h))
        return [max(xa, 0.0), max(ya, 0.0), min(xb, 1.0), min(yb, 1.0)]

    def mouseReleaseEvent(self, e):
        if self._pan is not None:
            _, button, moved = self._pan
            self._pan = None
            if not moved and button == Qt.LeftButton:
                self.toggle_zoom(e.position())
            self._update_cursor()
            return
        if self._mode == self.CROP and self._drag is not None:
            c0 = self._drag[2]
            self._drag = None
            x0, y0, x1, y1 = self._crop
            if x1 - x0 < 0.02 or y1 - y0 < 0.02 or self._crop == c0:
                return
            self.cropChanged.emit([round(v, 5) for v in self._crop])
