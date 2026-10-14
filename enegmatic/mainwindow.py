"""Main application window."""
from __future__ import annotations

import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import QObject, QRunnable, QSettings, QSize, Qt, QThreadPool, QTimer, QUrl, Signal
from PySide6.QtGui import (QAction, QActionGroup, QColor, QDesktopServices, QGuiApplication, QIcon,
                           QImage, QKeySequence, QPainter, QPixmap)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox,
                               QDialog, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QListView, QListWidget, QListWidgetItem, QMainWindow, QMenu,
                               QMessageBox, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
                               QSplitter, QToolBar, QVBoxLayout, QWidget)

from . import __version__, dust
from .autocrop import detect_frame
from .batch import ExportOptions
from .core import ASPECTS as CROP_ASPECTS
from .core import (DEFAULT_COPY_GROUPS, SETTING_GROUPS, Settings, detect_aspect, fit_aspect, oriented_ratio,
                   straighten_crop, detect_film_type, apply_crop, apply_geometry, histogram, negative_preview,
                   neutral_wb_gains, orient, render, resolve_analysis, sample, src_to_view, straighten,
                   to_uint8, transform_crop, view_scale, view_to_src)
from .export import ExportDialog, ExportRunner
from .fileio import (RAW_EXTS, SUPPORTED_EXTS, collect_files, load_negative_u16, load_proxy, load_sidecar,
                     save_sidecar)
from .theme import apply_theme, system_theme
from .widgets import CollapsibleSection, Histogram, ImageView, ParamSlider, glyph_icon, set_slider_colors

THUMB_SIZE = QSize(176, 118)

# field, label, min, max, step, decimals, tooltip
TONE_SLIDERS = [
    ("exposure", "Exposure", -3.0, 3.0, 0.05, 2, "Overall brightness in stops (EV)."),
    ("contrast", "Contrast", -100, 100, 1, 0, "S-curve contrast."),
    ("gamma", "Midtones", 0.3, 3.0, 0.01, 2, "Gamma: >1 brightens, <1 darkens the mid-tones."),
    ("highlights", "Highlights", -100, 100, 1, 0, "Brighten or recover the bright tones."),
    ("shadows", "Shadows", -100, 100, 1, 0, "Lift or deepen the dark tones."),
    ("black", "Blacks", -50, 50, 1, 0, "Black point. Positive = lifted / faded blacks, negative = deeper blacks."),
    ("white", "Whites", -50, 50, 1, 0, "White point. Positive = brighter whites (more clipping)."),
]
COLOR_SLIDERS = [
    ("temperature", "Temperature", -100, 100, 1, 0, "Cooler (−) / warmer (+)."),
    ("tint", "Tint", -100, 100, 1, 0, "Green (−) / magenta (+)."),
    ("saturation", "Saturation", -100, 100, 1, 0, "Colour intensity."),
]
DETAIL_SLIDERS = [
    ("sharpen", "Sharpen", 0, 100, 1, 0, "Unsharp-mask sharpening."),
]
FILM_SLIDERS = [
    ("clip_black", "Black clip %", 0.0, 2.0, 0.01, 2,
     "How many pixels auto levels may clip to black. Higher = punchier, lower = safer."),
    ("clip_white", "White clip %", 0.0, 2.0, 0.01, 2,
     "How many pixels auto levels may clip to white."),
]
DUST_SLIDERS = [
    ("dust_strength", "Sensitivity", 0, 100, 1, 0,
     "Higher also removes fainter and less certain specks; lower only obvious dust."),
]
GEOM_SLIDERS = [
    ("straighten", "Straighten", -45.0, 45.0, 0.05, 2,
     "Fine rotation in degrees. The picture turns about the crop centre and the crop shrinks\n"
     "just enough to keep empty corners out. In the crop tool, drag outside a corner to rotate."),
]

ASPECTS = [("Free", "")] + [(k, k) for k in CROP_ASPECTS]   # combo label, Settings.crop_aspect


def make_icon(pm: QPixmap) -> QIcon:
    """Icon that looks the same when selected (Qt tints selected icons otherwise)."""
    icon = QIcon()
    for mode in (QIcon.Normal, QIcon.Selected, QIcon.Active):
        icon.addPixmap(pm, mode)
    return icon


def np_to_qimage(img8) -> QImage:
    h, w = img8.shape[:2]
    return QImage(img8.data, w, h, 3 * w, QImage.Format_RGB888).copy()


# --------------------------------------------------------------------------- model

class ImageItem:
    def __init__(self, path: str, settings: Settings, fresh: bool):
        self.path = path
        self.name = Path(path).name
        self.folder = str(Path(path).parent)
        self.settings = settings
        self.fresh = fresh          # no saved settings yet -> auto-crop on first load
        self.thumb_neg = None
        self.thumb_qimage = None
        self.film_locked = False    # film type chosen by the user / pasted
        self.film_detected = False
        self.list_item: QListWidgetItem | None = None
        self.error: str | None = None


class ProxyData:
    """Preview-sized negative with a cached oriented / straightened copy."""

    def __init__(self, neg):
        self.neg = neg
        self.full_shape = None      # (h, w) of the full-resolution image
        self._key = None
        self._geom = None
        self.cleaner = dust.Cleaner()
        self.dust_specks: list = []  # auto-detected specks of the last geometry() call

    def geometry(self, s: Settings):
        key = (s.rotation % 360, bool(s.flip_h), round(float(s.straighten), 3), dust.dust_key(s))
        if key != self._key:
            neg, self.dust_specks = self.cleaner.get(self.neg, s)
            self._geom = straighten(orient(neg, s.rotation, s.flip_h), s.straighten)
            self._key = key
        return self._geom

    def frame_size(self, s: Settings) -> tuple[int, int]:
        h, w = self.neg.shape[:2]
        return (h, w) if (s.rotation // 90) % 2 else (w, h)


class ProxyCache:
    def __init__(self, capacity: int = 8):
        self._d: OrderedDict[str, ProxyData] = OrderedDict()
        self.capacity = capacity

    def get(self, path: str) -> ProxyData | None:
        p = self._d.get(path)
        if p is not None:
            self._d.move_to_end(path)
        return p

    def put(self, path: str, data: ProxyData, protect: str | None = None):
        self._d[path] = data
        self._d.move_to_end(path)
        while len(self._d) > self.capacity:
            victim = next((k for k in self._d if k != protect and k != path), None)
            if victim is None:
                break
            del self._d[victim]

    def clear(self):
        self._d.clear()


class WorkerSignals(QObject):
    loaded = Signal(str, object, object, object)        # path, proxy, thumb, info
    failed = Signal(str, str)
    rendered = Signal(int, str, object, object, object, bool)  # gen, path, qimage, analysis, hist, mono
    full_loaded = Signal(str, object)                   # path, full-resolution uint16 negative
    full_failed = Signal(str, str)
    tile = Signal(int, str, object, object, str)        # gen, path, qimage, rect, settings signature


class LoadJob(QRunnable):
    def __init__(self, path, signals, inset, orientation, detect_film):
        super().__init__()
        self.path, self.signals, self.inset, self.orientation = path, signals, inset, orientation
        self.detect_film = detect_film
        self.started = False

    def run(self):
        self.started = True
        try:
            proxy, thumb, full_shape = load_proxy(self.path)
            ac = film = None
            if self.inset is not None or self.detect_film:
                oriented = orient(proxy, *self.orientation)
                if self.inset is not None:
                    ac = detect_frame(oriented, self.inset)
                if self.detect_film:
                    if ac and ac[1]:
                        region = apply_crop(straighten(oriented, ac[0]), ac[1])
                    else:
                        region = apply_crop(oriented, [0.1, 0.1, 0.9, 0.9])
                    film = detect_film_type(region)
            info = {"autocrop": ac, "orientation": self.orientation, "film": film,
                    "full_shape": full_shape}
            self.signals.loaded.emit(self.path, proxy, thumb, info)
        except Exception as e:
            self.signals.failed.emit(self.path, f"{type(e).__name__}: {e}")


class FullLoadJob(QRunnable):
    """Decodes the full-resolution negative for zooming."""

    def __init__(self, path, signals):
        super().__init__()
        self.path, self.signals = path, signals

    def run(self):
        try:
            self.signals.full_loaded.emit(self.path, load_negative_u16(self.path))
        except Exception as e:
            self.signals.full_failed.emit(self.path, f"{type(e).__name__}: {e}")


class RenderJob(QRunnable):
    def __init__(self, gen, path, proxy, settings, full_view, negative, signals, latest, fast=False):
        super().__init__()
        self.gen, self.path, self.proxy, self.settings = gen, path, proxy, settings
        self.full_view, self.negative, self.signals, self.latest = full_view, negative, signals, latest
        self.fast = fast            # interactive drag: keep the current analysis

    def run(self):
        if self.latest() != self.gen:
            return
        try:
            s = self.settings
            full = self.proxy.geometry(s)
            cropped = apply_crop(full, s.crop)
            nch = 1 if s.film_type == "bw" else 3
            if self.fast and s.analysis and len(s.analysis.get("dmin", ())) == nch:
                a = s.analysis
            else:
                a = resolve_analysis(cropped, s)
            view = full if self.full_view else cropped
            img = negative_preview(view) if self.negative else render(view, s, a)
            img8 = to_uint8(img)
            mono = s.film_type == "bw" and not self.negative
            self.signals.rendered.emit(self.gen, self.path, np_to_qimage(img8), a,
                                       histogram(img8), mono)
        except Exception as e:
            self.signals.failed.emit(self.path, f"Render error: {type(e).__name__}: {e}")


class TileJob(QRunnable):
    """Renders the visible part of the full-resolution image for the zoomed view."""

    def __init__(self, gen, path, full, settings, full_view, rect, display_w, sig, signals, latest):
        super().__init__()
        self.gen, self.path, self.full, self.settings = gen, path, full, settings
        self.full_view, self.rect, self.display_w, self.sig = full_view, rect, display_w, sig
        self.signals, self.latest = signals, latest

    def run(self):
        if self.latest() != self.gen:
            return
        try:
            s = self.settings
            geom = self.full.geometry(s)
            cropped = apply_crop(geom, s.crop)
            base = geom if self.full_view else cropped
            h, w = base.shape[:2]
            x0, y0, x1, y1 = self.rect
            mx, my = (x1 - x0) * 0.08, (y1 - y0) * 0.08      # margin for small pans
            x0, y0, x1, y1 = max(0.0, x0 - mx), max(0.0, y0 - my), min(1.0, x1 + mx), min(1.0, y1 + my)
            px0, py0 = int(x0 * w), int(y0 * h)
            px1, py1 = max(px0 + 1, int(np.ceil(x1 * w))), max(py0 + 1, int(np.ceil(y1 * h)))
            a = resolve_analysis(cropped, s)
            img = render(base[py0:py1, px0:px1], s, a, ref_size=max(cropped.shape[:2]))
            want_w = int((px1 - px0) / w * self.display_w) + 2
            if want_w < img.shape[1]:
                img = cv2.resize(img, (want_w, max(1, round(img.shape[0] * want_w / img.shape[1]))),
                                 interpolation=cv2.INTER_AREA)
            if self.latest() != self.gen:
                return
            self.signals.tile.emit(self.gen, self.path, np_to_qimage(to_uint8(img)),
                                   (px0 / w, py0 / h, px1 / w, py1 / h), self.sig)
        except Exception as e:
            self.signals.full_failed.emit(self.path, f"Zoom render error: {type(e).__name__}: {e}")


class CopyDialog(QDialog):
    """Pick which groups of settings Copy takes (and therefore Paste applies)."""

    def __init__(self, parent, selected):
        super().__init__(parent)
        self.setWindowTitle("Copy settings")
        self.setMinimumWidth(320)
        v = QVBoxLayout(self)
        v.setSpacing(8)
        hint = QLabel("Copy takes these settings from the current frame;\nPaste applies only them.")
        hint.setObjectName("Muted")
        v.addWidget(hint)
        self.boxes = {}
        for key, (label, _) in SETTING_GROUPS.items():
            cb = QCheckBox(label)
            cb.setChecked(key in selected)
            if key == "film":
                cb.setToolTip("Film type, auto WB, clipping, picked base - and the locked analysis if "
                              "'Lock analysis' is on.")
            if key == "crop":
                cb.setToolTip("Usually left off: every frame sits differently in the holder.")
            self.boxes[key] = cb
            v.addWidget(cb)
        row = QHBoxLayout()
        b_all = QPushButton("All")
        b_all.clicked.connect(lambda: [cb.setChecked(True) for cb in self.boxes.values()])
        b_none = QPushButton("None")
        b_none.clicked.connect(lambda: [cb.setChecked(False) for cb in self.boxes.values()])
        row.addWidget(b_all)
        row.addWidget(b_none)
        row.addStretch(1)
        ok = QPushButton("Copy")
        ok.setObjectName("Primary")
        ok.setDefault(True)
        ok.clicked.connect(self.accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        row.addWidget(cancel)
        row.addWidget(ok)
        v.addSpacing(6)
        v.addLayout(row)

    def groups(self) -> list[str]:
        return [k for k, cb in self.boxes.items() if cb.isChecked()]


# --------------------------------------------------------------------------- window

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("eNegmatic")
        self.setAcceptDrops(True)
        self.resize(1480, 900)

        self.prefs = QSettings("eNegmatic", "eNegmatic")
        self.items: list[ImageItem] = []
        self.by_path: dict[str, ImageItem] = {}
        self.current: ImageItem | None = None
        self.cache = ProxyCache(8)
        self.signals = WorkerSignals()
        self.signals.loaded.connect(self._on_loaded)
        self.signals.failed.connect(self._on_failed)
        self.signals.rendered.connect(self._on_rendered)
        self.signals.full_loaded.connect(self._on_full_loaded)
        self.signals.full_failed.connect(self._on_full_failed)
        self.signals.tile.connect(self._on_tile)
        self._full: ProxyData | None = None
        self._full_path = self._full_loading = None
        self._full_job = None
        self._tile_gen = 0
        self._tile_sig = ""
        self._viewport = None
        self._render_full_size = (-1, None)
        self.load_pool = QThreadPool(self)
        self.load_pool.setMaxThreadCount(max(1, min(3, (os.cpu_count() or 2) // 2)))
        self.render_pool = QThreadPool(self)
        self.render_pool.setMaxThreadCount(1)
        # zoomed full-resolution tiles get their own worker, so the preview never waits
        # behind them (straightening while zoomed used to lag badly)
        self.tile_pool = QThreadPool(self)
        self.tile_pool.setMaxThreadCount(1)
        self._load_jobs: dict[str, LoadJob] = {}
        self._gen = 0
        self._clipboard: dict | None = None
        self._dirty_folders: set[str] = set()
        self._crop_mode = False
        self._pick: str | None = None
        self._compare = False
        self._heal = False
        self._fast_until = 0.0
        self._rot_base = None
        self._exporter: ExportRunner | None = None
        self._loaded = 0
        self.colors = {}

        self.export_options = ExportOptions.from_dict(self._pref_json("export"))
        self.default_film = self.prefs.value("default_film", "color")
        self.autocrop_on_load = self._pref_bool("autocrop_on_load", True)
        self.auto_film = self._pref_bool("auto_film", True)
        self.autocrop_inset = float(self.prefs.value("autocrop_inset", 1.0))
        self.save_sidecars = self._pref_bool("save_sidecars", True)
        self.brush_size = float(self.prefs.value("brush_size", 0.8))   # % of the frame's long edge
        groups = self._pref_json("copy_groups")
        self.copy_groups = [g for g in groups if g in SETTING_GROUPS] if isinstance(groups, list) else list(DEFAULT_COPY_GROUPS)

        self._render_timer = QTimer(self, singleShot=True, interval=15)
        self._render_timer.timeout.connect(self._do_render)
        self._settle_timer = QTimer(self, singleShot=True, interval=350)
        self._settle_timer.timeout.connect(lambda: self._schedule_render(0))
        self._save_timer = QTimer(self, singleShot=True, interval=1500)
        self._save_timer.timeout.connect(self._save_sidecars)

        self._build_actions()
        self._build_ui()
        self._build_menus()
        self._build_toolbar()
        self.set_theme(self.prefs.value("theme", "system"))
        try:
            QGuiApplication.styleHints().colorSchemeChanged.connect(self._on_system_scheme)
        except AttributeError:
            pass

        geo = self.prefs.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        split = self.prefs.value("splitter")
        if split is not None:
            self.splitter.restoreState(split)
        self._update_enabled()

    # ------------------------------------------------------------------ prefs helpers
    def _pref_bool(self, key, default):
        v = self.prefs.value(key, default)
        return v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes")

    def _pref_json(self, key):
        try:
            return json.loads(self.prefs.value(key, "{}"))
        except (TypeError, ValueError):
            return {}

    # ------------------------------------------------------------------ actions / menus
    def _act(self, text, slot, shortcut=None, tip=None, checkable=False):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        if tip:
            a.setToolTip(f"{tip} ({shortcut})" if shortcut else tip)
            a.setStatusTip(tip)
        a.setCheckable(checkable)
        a.triggered.connect(slot)
        return a

    def _build_actions(self):
        A = self._act
        self.act_open = A("Open…", self._open_files, "Ctrl+O", "Open negatives (replaces the current list)")
        self.act_add = A("Add Files…", lambda: self._open_files(add=True), "Ctrl+Shift+A", "Add negatives to the list")
        self.act_folder = A("Open Folder…", self._open_folder, "Ctrl+Shift+O", "Open every negative in a folder")
        self.act_prev = A("Previous Frame", lambda: self._step(-1), None, "Previous frame (←)")
        self.act_next = A("Next Frame", lambda: self._step(1), None, "Next frame (→)")
        self.act_rot_l = A("Rotate Left", lambda: self._rotate(90), "[", "Rotate 90° left ([)")
        self.act_rot_r = A("Rotate Right", lambda: self._rotate(-90), "]", "Rotate 90° right (])")
        self.act_flip = A("⇋ Flip", self._flip, "Shift+F", "Mirror horizontally (emulsion side)")
        self.act_crop = A("Crop", self._toggle_crop, "C", "Crop tool – drag on the image", checkable=True)
        self.act_autocrop = A("Auto Crop", self._auto_crop, "A", "Detect the frame edge, straighten and crop")
        self.act_compare = A("Negative", self._toggle_compare, "B", "Show the original negative", checkable=True)
        self.act_heal = A("Heal", self._toggle_heal, "H",
                          "Healing brush – paint over dust and hairs; right-click undoes", checkable=True)
        self.act_theme = A("☀", self._toggle_theme, "Ctrl+T", "Toggle light / dark theme")
        self.act_export = A("Export…", lambda: self._export("current"), "Ctrl+E", "Export the current image")
        self.act_export_all = A("Export All…", lambda: self._export("all"), "Ctrl+Shift+E", "Export every image")
        self.act_copy = A("Copy Settings", lambda: self._copy_settings(), "Ctrl+Shift+C",
                          "Copy this frame's settings (the groups chosen with ⋯)")
        self.act_copy_choose = A("Copy Settings…", lambda: self._copy_settings(choose=True), "Ctrl+Alt+C",
                                 "Choose which settings to copy")
        self.act_paste = A("Paste Settings", lambda: self._paste(self._selected_items()), "Ctrl+Shift+V",
                           "Paste adjustments to the selected frames")
        self.act_paste_all = A("Paste Settings to All", lambda: self._paste(list(self.items)), None,
                               "Paste adjustments to every frame")
        self.act_reset = A("Reset Adjustments", self._reset_adjustments, "Ctrl+R",
                           "Reset tone / colour (keeps crop & rotation)")
        self.act_quit = A("Quit", self.close, "Ctrl+Q")
        self.act_zoom_toggle = A("Zoom 1:1 / Fit", lambda: self.view.toggle_zoom(), "Z",
                                 "Toggle between fit and 100% (or click the image)")
        self.act_zoom_in = A("Zoom In", lambda: self.view.zoom_by(1.25), "Ctrl+=", "Zoom in")
        self.act_zoom_out = A("Zoom Out", lambda: self.view.zoom_by(0.8), "Ctrl+-", "Zoom out")
        self.act_zoom_fit = A("Fit to Window", lambda: self.view.fit(), "Ctrl+0", "Fit the image to the window")
        for a in (self.act_prev, self.act_next, self.act_rot_l, self.act_rot_r, self.act_flip,
                  self.act_crop, self.act_autocrop, self.act_compare, self.act_heal, self.act_theme,
                  self.act_export, self.act_export_all, self.act_copy, self.act_copy_choose, self.act_paste,
                  self.act_paste_all, self.act_reset, self.act_open, self.act_add,
                  self.act_zoom_toggle, self.act_zoom_in, self.act_zoom_out, self.act_zoom_fit,
                  self.act_folder, self.act_quit):
            self.addAction(a)

    def _build_menus(self):
        mb = self.menuBar()
        m = mb.addMenu("&File")
        for a in (self.act_open, self.act_add, self.act_folder):
            m.addAction(a)
        m.addSeparator()
        m.addAction(self.act_export)
        m.addAction("Export Selected…", lambda: self._export("selected"))
        m.addAction(self.act_export_all)
        m.addSeparator()
        m.addAction(self.act_quit)

        m = mb.addMenu("&Edit")
        for a in (self.act_copy, self.act_copy_choose, self.act_paste, self.act_paste_all):
            m.addAction(a)
        m.addSeparator()
        m.addAction(self.act_reset)
        m.addSeparator()
        for a in (self.act_rot_l, self.act_rot_r, self.act_flip, self.act_crop, self.act_autocrop):
            m.addAction(a)
        m.addAction("Reset Crop && Straighten", self._reset_crop)
        m.addSeparator()
        m.addAction(self.act_heal)
        m.addAction("Clear Dust Spots", self._clear_spots)

        m = mb.addMenu("&View")
        m.addAction(self.act_compare)
        m.addSeparator()
        for a in (self.act_zoom_toggle, self.act_zoom_in, self.act_zoom_out, self.act_zoom_fit):
            m.addAction(a)
        m.addSeparator()
        grp = QActionGroup(self)
        self.theme_actions = {}
        for key, label in (("system", "System Theme"), ("dark", "Dark Theme"), ("light", "Light Theme")):
            a = QAction(label, self, checkable=True)
            a.triggered.connect(lambda _=False, k=key: self.set_theme(k))
            grp.addAction(a)
            m.addAction(a)
            self.theme_actions[key] = a
        m.addAction(self.act_theme)

        m = mb.addMenu("&Options")
        a = QAction("Auto-detect Film Type (Colour / B&&W)", self, checkable=True)
        a.setChecked(self.auto_film)
        a.toggled.connect(lambda v: self._set_pref("auto_film", v))
        m.addAction(a)
        a = QAction("Auto-crop New Images", self, checkable=True)
        a.setChecked(self.autocrop_on_load)
        a.toggled.connect(lambda v: self._set_pref("autocrop_on_load", v))
        m.addAction(a)
        insets = m.addMenu("Auto-crop Inset")
        igrp = QActionGroup(self)
        for v in (0.0, 0.5, 1.0, 1.5, 2.5, 4.0):
            ia = QAction(f"{v:g} %", self, checkable=True)
            ia.setChecked(abs(v - self.autocrop_inset) < 1e-6)
            ia.triggered.connect(lambda _=False, x=v: self._set_pref("autocrop_inset", x))
            igrp.addAction(ia)
            insets.addAction(ia)
        a = QAction("Save Edits Next to Images (.enegmatic.json)", self, checkable=True)
        a.setChecked(self.save_sidecars)
        a.toggled.connect(lambda v: self._set_pref("save_sidecars", v))
        m.addAction(a)

        m = mb.addMenu("&Help")
        m.addAction("Shortcuts && Tips", self._show_help)
        m.addAction("About", lambda: QMessageBox.about(
            self, "eNegmatic",
            f"<b>eNegmatic {__version__}</b><br>Colour & B&W negative conversion.<br>"
            "Raw decoding by LibRaw (rawpy)."))

    def _set_pref(self, key, value):
        setattr(self, key, value)
        self.prefs.setValue(key, value)

    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QSize(20, 20))
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(Qt.TopToolBarArea, tb)
        tb.addAction(self.act_open)
        tb.addAction(self.act_folder)
        tb.addSeparator()
        tb.addAction(self.act_prev)
        tb.addAction(self.act_next)
        tb.addSeparator()
        tb.addAction(self.act_rot_l)
        tb.addAction(self.act_rot_r)
        for a in (self.act_prev, self.act_next, self.act_rot_l, self.act_rot_r):
            tb.widgetForAction(a).setToolButtonStyle(Qt.ToolButtonIconOnly)
        tb.addAction(self.act_flip)
        tb.addSeparator()
        tb.addAction(self.act_crop)
        tb.addAction(self.act_autocrop)
        tb.addAction(self.act_heal)
        tb.addAction(self.act_compare)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)
        tb.addAction(self.act_theme)
        tb.addSeparator()
        tb.addAction(self.act_export)
        tb.addAction(self.act_export_all)
        tb.widgetForAction(self.act_export_all).setObjectName("Primary")

    # ------------------------------------------------------------------ layout
    def _add_section(self, layout, key, title, expanded=True) -> CollapsibleSection:
        sec = CollapsibleSection(title, self._pref_bool(f"section/{key}", expanded))
        sec.toggled.connect(lambda on, k=key: self.prefs.setValue(f"section/{k}", on))
        sec.resetRequested.connect(lambda k=key: self._reset_section(k))
        layout.addWidget(sec)
        self.sections[key] = sec
        return sec

    def _action_button(self, text, action) -> QPushButton:
        """Push button mirroring an action (trigger, tooltip and enabled state)."""
        b = QPushButton(text)
        b.setToolTip(action.toolTip())
        b.clicked.connect(action.trigger)
        b.setEnabled(action.isEnabled())
        action.changed.connect(lambda: b.setEnabled(action.isEnabled()))
        return b

    def _sliders(self, layout, specs):
        defaults = Settings()
        for field, label, lo, hi, step, dec, tip in specs:
            origin = 1.0 if field == "gamma" else None
            sl = ParamSlider(label, lo, hi, step, getattr(defaults, field), dec, tip, origin)
            sl.valueChanged.connect(lambda v, f=field: self._set_field(f, v))
            self.sliders[field] = sl
            layout.addWidget(sl)

    def _build_ui(self):
        self.sliders: dict[str, ParamSlider] = {}

        # filmstrip
        self.strip = QListWidget()
        self.strip.setObjectName("Filmstrip")
        self.strip.setViewMode(QListView.IconMode)
        self.strip.setFlow(QListView.TopToBottom)
        self.strip.setWrapping(False)
        self.strip.setMovement(QListView.Static)
        self.strip.setResizeMode(QListView.Adjust)
        self.strip.setIconSize(THUMB_SIZE)
        self.strip.setSpacing(3)
        self.strip.setUniformItemSizes(True)
        self.strip.setTextElideMode(Qt.ElideMiddle)
        self.strip.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.strip.setMinimumWidth(150)
        self.strip.currentItemChanged.connect(self._on_current_changed)
        self.strip.setContextMenuPolicy(Qt.CustomContextMenu)
        self.strip.customContextMenuRequested.connect(self._strip_menu)

        # viewer
        self.view = ImageView()
        self.view.cropChanged.connect(self._on_crop_changed)
        self.view.picked.connect(self._on_picked)
        self.view.healPainted.connect(self._on_heal_painted)
        self.view.rotateDragged.connect(self._on_rotate_drag)
        self.view.healErase.connect(self._on_heal_erase)
        self.view.brushStep.connect(lambda d: self._set_brush(self.brush_size * (1.2 ** d)))
        self.view.viewportChanged.connect(self._on_viewport)

        # adjustments panel
        panel = QWidget()
        panel.setObjectName("Panel")
        v = QVBoxLayout(panel)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(10)

        hist_card = QFrame()
        hist_card.setObjectName("SectionCard")
        hv = QVBoxLayout(hist_card)
        hv.setContentsMargins(10, 10, 10, 8)
        hv.setSpacing(6)
        self.hist = Histogram()
        hv.addWidget(self.hist)
        self.info = QLabel(" ")
        self.info.setObjectName("Muted")
        self.info.setWordWrap(True)
        hv.addWidget(self.info)
        v.addWidget(hist_card)

        seg = QHBoxLayout()
        seg.setSpacing(0)
        self.btn_color = QPushButton("Colour negative")
        self.btn_color.setObjectName("SegmentL")
        self.btn_bw = QPushButton("B&&W negative")
        self.btn_bw.setObjectName("SegmentR")
        self.film_group = QButtonGroup(self)
        for b in (self.btn_color, self.btn_bw):
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            self.film_group.addButton(b)
            seg.addWidget(b)
        self.btn_color.clicked.connect(lambda: self._set_film("color"))
        self.btn_bw.clicked.connect(lambda: self._set_film("bw"))
        v.addLayout(seg)

        self.sections: dict[str, CollapsibleSection] = {}

        # geometry (first: it's usually the first thing done to a frame)
        sec = self._add_section(v, "geometry", "Crop & rotate")
        grid = QGridLayout()
        grid.setSpacing(6)
        b_l = QPushButton("⟲ Left")
        b_l.setToolTip("Rotate 90° left ([)")
        b_l.clicked.connect(lambda: self._rotate(90))
        b_r = QPushButton("Right ⟳")
        b_r.setToolTip("Rotate 90° right (])")
        b_r.clicked.connect(lambda: self._rotate(-90))
        b_f = QPushButton("⇋ Flip")
        b_f.setToolTip("Mirror horizontally (Shift+F)")
        b_f.clicked.connect(self._flip)
        for i, b in enumerate((b_l, b_r, b_f)):
            grid.addWidget(b, 0, i)
        b_auto = QPushButton("Auto crop")
        b_auto.setToolTip("Detect the frame edge and its format, straighten and crop (A)")
        b_auto.clicked.connect(self._auto_crop)
        b_reset = QPushButton("Reset crop")
        b_reset.setToolTip("Remove crop and straightening.")
        b_reset.clicked.connect(self._reset_crop)
        grid.addWidget(b_auto, 1, 0, 1, 3)
        grid.addWidget(b_reset, 2, 0, 1, 3)
        sec.addLayout(grid)
        asp = QHBoxLayout()
        asp.setSpacing(6)
        self.btn_crop = QPushButton("Crop tool")
        self.btn_crop.setCheckable(True)
        self.btn_crop.setToolTip("Drag to crop, drag outside a corner to rotate (C)")
        self.btn_crop.toggled.connect(self._toggle_crop)
        self.aspect = QComboBox()
        self.aspect.setToolTip("Crop format. Auto crop picks it from the frame (3:2 if unsure).")
        for name, key in ASPECTS:
            self.aspect.addItem(name, key)
        self.aspect.currentIndexChanged.connect(self._on_aspect)
        self.btn_swap = QPushButton("⇅")
        self.btn_swap.setToolTip("Swap portrait / landscape")
        self.btn_swap.setFixedWidth(36)
        self.btn_swap.clicked.connect(self._swap_aspect)
        asp.addWidget(self.btn_crop, 1)
        asp.addWidget(self.aspect, 1)
        asp.addWidget(self.btn_swap)
        sec.addLayout(asp)
        self._sliders(sec, GEOM_SLIDERS)


        # inversion
        sec = self._add_section(v, "inversion", "Inversion")
        self.cb_auto_wb = QCheckBox("Auto white balance")
        self.cb_auto_wb.setToolTip("Grey-balance the mid-tones after removing the orange mask.")
        self.cb_auto_wb.toggled.connect(lambda on: self._set_field("auto_wb", on))
        sec.addWidget(self.cb_auto_wb)
        self.cb_auto_bright = QCheckBox("Auto brightness")
        self.cb_auto_bright.setToolTip("Brighten dim conversions like a gentle exposure push: highlights come\n"
                                       "up toward white, blacks stay black and night frames stay dark.")
        self.cb_auto_bright.toggled.connect(lambda on: self._set_field("auto_brightness", on))
        sec.addWidget(self.cb_auto_bright)
        lock_row = QHBoxLayout()
        self.cb_lock = QCheckBox("Lock analysis (roll)")
        self.cb_lock.setToolTip(
            "Freeze this frame's film base & levels.\nThen Copy → Paste to all to convert the whole "
            "roll identically,\nwhich keeps colour and density consistent between frames.")
        self.cb_lock.toggled.connect(lambda on: self._set_field("lock_analysis", on))
        self.btn_reanalyze = QPushButton("Re-analyse")
        self.btn_reanalyze.setToolTip("Recompute base & levels from the current crop.")
        self.btn_reanalyze.clicked.connect(self._reanalyze)
        lock_row.addWidget(self.cb_lock, 1)
        lock_row.addWidget(self.btn_reanalyze)
        sec.addLayout(lock_row)
        base_row = QHBoxLayout()
        lab = QLabel("Film base")
        lab.setFixedWidth(84)
        self.base_swatch = QLabel()
        self.base_swatch.setFixedSize(22, 22)
        self.base_label = QLabel("Auto")
        self.base_label.setObjectName("Muted")
        self.btn_pick_base = QPushButton("Pick")
        self.btn_pick_base.setCheckable(True)
        self.btn_pick_base.setToolTip("Click an unexposed area of film (between frames / the rebate).")
        self.btn_pick_base.toggled.connect(lambda on: self._start_pick("base", on))
        self.btn_clear_base = QPushButton("Auto")
        self.btn_clear_base.setToolTip("Estimate the base automatically from the image.")
        self.btn_clear_base.clicked.connect(lambda: self._set_field("base_rgb", None, reload_ui=True))
        base_row.addWidget(lab)
        base_row.addWidget(self.base_swatch)
        base_row.addWidget(self.base_label, 1)
        base_row.addWidget(self.btn_pick_base)
        base_row.addWidget(self.btn_clear_base)
        sec.addLayout(base_row)
        self._sliders(sec, FILM_SLIDERS)

        # tone
        sec = self._add_section(v, "tone", "Tone")
        self._sliders(sec, TONE_SLIDERS)

        # colour
        sec = self._add_section(v, "color", "Colour")
        self._sliders(sec, COLOR_SLIDERS)
        wb_row = QHBoxLayout()
        self.btn_pick_wb = QPushButton("Pick neutral")
        self.btn_pick_wb.setCheckable(True)
        self.btn_pick_wb.setToolTip("Click something that should be neutral grey or white.")
        self.btn_pick_wb.toggled.connect(lambda on: self._start_pick("wb", on))
        btn_reset_wb = QPushButton("Reset WB")
        btn_reset_wb.clicked.connect(self._reset_wb)
        wb_row.addWidget(self.btn_pick_wb)
        wb_row.addWidget(btn_reset_wb)
        sec.addLayout(wb_row)
        self.color_box = sec

        # detail
        sec = self._add_section(v, "detail", "Detail", expanded=False)
        self._sliders(sec, DETAIL_SLIDERS)

        # dust
        sec = self._add_section(v, "dust", "Dust & hairs (beta)")
        self.cb_dust = QCheckBox("Auto remove dust (beta)")
        self.cb_dust.setToolTip(
            "Finds dust specks and hairs (white in the positive) on smooth areas such as sky\n"
            "and walls, and fills them in from their surroundings, grain included.\n"
            "Turn on the healing brush to see what it removes; right-click a speck to keep it.")
        self.cb_dust.toggled.connect(lambda on: self._set_field("dust_auto", on))
        sec.addWidget(self.cb_dust)
        self._sliders(sec, DUST_SLIDERS)
        heal_row = QHBoxLayout()
        self.btn_heal = QPushButton("Healing brush")
        self.btn_heal.setCheckable(True)
        self.btn_heal.setToolTip("Paint over dust, hairs or scratches (H).\nRight-click a spot to undo it, "
                                 "Shift+wheel for brush size, Ctrl+Z undoes the last spot.")
        self.btn_heal.toggled.connect(self._toggle_heal)
        b_clear = QPushButton("Clear spots")
        b_clear.setToolTip("Remove every healing-brush spot and kept speck on this frame.")
        b_clear.clicked.connect(self._clear_spots)
        heal_row.addWidget(self.btn_heal, 1)
        heal_row.addWidget(b_clear)
        sec.addLayout(heal_row)
        self.brush_slider = ParamSlider("Brush size", 0.2, 5.0, 0.1, 0.8, 1,
                                        "Healing brush radius in % of the frame's long edge (Shift+wheel).")
        self.brush_slider.setValue(self.brush_size)
        self.brush_slider.valueChanged.connect(self._set_brush)
        sec.addWidget(self.brush_slider)
        self.dust_info = QLabel("")
        self.dust_info.setObjectName("Muted")
        self.dust_info.setWordWrap(True)
        sec.addWidget(self.dust_info)

        v.addStretch(1)

        self.panel = panel
        scroll = QScrollArea()
        scroll.setObjectName("PanelScroll")
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # footer: settings transfer, always visible
        footer = QFrame()
        footer.setObjectName("Footer")
        fg = QGridLayout(footer)
        fg.setContentsMargins(12, 10, 12, 12)
        fg.setHorizontalSpacing(6)
        fg.setVerticalSpacing(6)
        copy_row = QHBoxLayout()
        copy_row.setSpacing(0)
        b_copy = self._action_button("Copy", self.act_copy)
        b_copy.setObjectName("SegmentL")
        b_copy_opts = self._action_button("⋯", self.act_copy_choose)
        b_copy_opts.setObjectName("SegmentR")
        b_copy_opts.setFixedWidth(34)
        copy_row.addWidget(b_copy, 1)
        copy_row.addWidget(b_copy_opts)
        fg.addLayout(copy_row, 0, 0)
        fg.addWidget(self._action_button("Paste", self.act_paste), 0, 1)
        fg.addWidget(self._action_button("Paste to all", self.act_paste_all), 1, 0)
        fg.addWidget(self._action_button("Reset all", self.act_reset), 1, 1)
        fg.setColumnStretch(0, 1)
        fg.setColumnStretch(1, 1)

        right = QWidget()
        right.setObjectName("Panel")
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(0)
        rv.addWidget(scroll, 1)
        rv.addWidget(footer)
        right.setMinimumWidth(340)

        left = QWidget()
        left.setObjectName("Panel")
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(0)
        self.strip_header = QLabel("FILMSTRIP")
        self.strip_header.setObjectName("PaneHeader")
        lv.addWidget(self.strip_header)
        lv.addWidget(self.strip, 1)

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(left)
        self.splitter.addWidget(self.view)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setSizes([214, 920, 350])
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(1)
        self.setCentralWidget(self.splitter)

        sb = self.statusBar()
        self.status = QLabel("Ready")
        sb.addWidget(self.status, 1)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(220)
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.hide()
        self.btn_cancel.clicked.connect(self._cancel_export)
        sb.addPermanentWidget(self.progress)
        sb.addPermanentWidget(self.btn_cancel)

    # ------------------------------------------------------------------ theme
    def set_theme(self, name: str):
        self.theme_pref = name if name in ("system", "dark", "light") else "system"
        self.prefs.setValue("theme", self.theme_pref)
        resolved = system_theme() if self.theme_pref == "system" else self.theme_pref
        self.colors = apply_theme(QApplication.instance(), resolved)
        c = self.colors
        self.view.set_colors(c["viewer"], c["muted"], c.get("accent"))
        for a, kind in ((self.act_prev, "prev"), (self.act_next, "next"),
                        (self.act_rot_l, "rot_l"), (self.act_rot_r, "rot_r")):
            a.setIcon(glyph_icon(kind, c["text"], 20))
        self.hist.set_colors(c["input"], c["border"])
        set_slider_colors(c)
        for sl in list(self.sliders.values()) + ([self.brush_slider] if hasattr(self, "brush_slider") else []):
            sl.slider.update()
        dark = c["name"] == "dark"
        self.act_theme.setText("☀" if dark else "☾")
        self.act_theme.setToolTip(f"Switch to {'light' if dark else 'dark'} theme (Ctrl+T)")
        if hasattr(self, "theme_actions"):
            self.theme_actions[self.theme_pref].setChecked(True)
        self._placeholder = self._make_placeholder()
        for it in self.items:
            if it.thumb_qimage is None and it.list_item is not None:
                it.list_item.setIcon(self._placeholder)
        self._update_base_swatch()

    def _on_system_scheme(self, *_):
        if self.theme_pref == "system":
            self.set_theme("system")

    def _toggle_theme(self):
        self.set_theme("light" if self.colors.get("name") == "dark" else "dark")

    def _make_placeholder(self) -> QIcon:
        pm = QPixmap(THUMB_SIZE)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor(self.colors.get("panel2", "#25282d")))
        p.setPen(QColor(self.colors.get("border", "#30343b")))
        p.drawRoundedRect(pm.rect().adjusted(1, 1, -1, -1), 6, 6)
        p.setPen(QColor(self.colors.get("muted", "#959ba4")))
        p.drawText(pm.rect(), Qt.AlignCenter, "loading…")
        p.end()
        return make_icon(pm)

    # ------------------------------------------------------------------ opening
    def _open_files(self, add=False):
        exts = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTS))
        raws = " ".join(f"*{e}" for e in sorted(RAW_EXTS))
        files, _ = QFileDialog.getOpenFileNames(
            self, "Open negatives", self.prefs.value("last_dir", ""),
            f"Negatives ({exts});;Nikon NEF (*.nef *.nrw);;Raw files ({raws});;"
            "Images (*.tif *.tiff *.png *.jpg *.jpeg);;All files (*)")
        if files:
            self.open_paths(files, add=add)

    def _open_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Open folder of negatives",
                                             self.prefs.value("last_dir", ""))
        if d:
            self.open_paths([d])

    def open_paths(self, paths, add=False):
        files = collect_files(paths)
        if not files:
            self._set_status("No supported files found.")
            return
        if not add:
            self._close_session()
        sidecars: dict[str, dict] = {}
        new_items = []
        for f in files:
            if f in self.by_path:
                continue
            folder = str(Path(f).parent)
            if folder not in sidecars:
                sidecars[folder] = load_sidecar(folder) if self.save_sidecars else {}
            saved = sidecars[folder].get(Path(f).name)
            settings = Settings.from_dict(saved) if saved else Settings(film_type=self.default_film)
            item = ImageItem(f, settings, fresh=saved is None)
            li = QListWidgetItem(self._placeholder, item.name)
            li.setData(Qt.UserRole, f)
            li.setToolTip(f)
            li.setSizeHint(QSize(THUMB_SIZE.width() + 14, THUMB_SIZE.height() + 30))
            self.strip.addItem(li)
            item.list_item = li
            self.items.append(item)
            self.by_path[f] = item
            new_items.append(item)
        self.prefs.setValue("last_dir", str(Path(files[0]).parent))
        if self.current is None and self.items:
            self.strip.setCurrentRow(self.items.index(new_items[0]) if new_items else 0)
        for it in new_items:
            self._request_load(it, 0)
        self._set_status(f"{len(self.items)} image(s) — building previews…")
        self._update_enabled()

    def _close_session(self):
        self._save_sidecars()
        self.load_pool.clear()
        self._load_jobs.clear()
        self.cache.clear()
        self._exit_tools()
        self.strip.blockSignals(True)
        self.strip.clear()
        self.strip.blockSignals(False)
        self.items, self.by_path, self.current = [], {}, None
        self._loaded = 0
        self.view.set_image(None)
        self.view.set_overlay("")
        self.hist.set_data(None)
        self._load_settings_to_ui()

    def _remove_selected(self):
        sel = self._selected_items()
        if not sel:
            return
        self._save_sidecars()
        for it in sel:
            self.items.remove(it)
            self.by_path.pop(it.path, None)
            self.strip.takeItem(self.strip.row(it.list_item))
        if not self.items:
            self._close_session()
        self._update_enabled()

    # ------------------------------------------------------------------ loading
    def _request_load(self, item: ImageItem, priority: int):
        job = self._load_jobs.get(item.path)
        if job is not None:
            if priority <= 0 or job.started or not self.load_pool.tryTake(job):
                return
        s = item.settings
        want_ac = self.autocrop_on_load and item.fresh and s.crop is None
        want_film = self.auto_film and item.fresh and not item.film_locked
        job = LoadJob(item.path, self.signals, self.autocrop_inset / 100.0 if want_ac else None,
                      (s.rotation, s.flip_h), want_film)
        self._load_jobs[item.path] = job
        self.load_pool.start(job, priority)

    def _on_loaded(self, path, proxy, thumb, info):
        self._load_jobs.pop(path, None)
        item = self.by_path.get(path)
        if item is None:
            return
        data = ProxyData(proxy)
        data.full_shape = info["full_shape"]
        self.cache.put(path, data, protect=self.current.path if self.current else None)
        item.thumb_neg = thumb
        s = item.settings
        ac, orientation = info["autocrop"], info["orientation"]
        if info["film"] and item.fresh and not item.film_locked:
            s.film_type = info["film"]
            item.film_detected = True
            self._touch(item)
        if item.fresh and s.crop is None and self.autocrop_on_load:
            if ac is None or (s.rotation, s.flip_h) != tuple(orientation):
                # rotated / pasted while loading: detect again in the current orientation
                ac = detect_frame(orient(proxy, s.rotation, s.flip_h), self.autocrop_inset / 100.0)
            angle, crop = ac
            if crop:
                self._apply_autocrop(s, data, angle, crop)
                self._touch(item)
        item.fresh = False
        self._loaded += 1
        if self._loaded >= len(self.items):
            self._set_status(f"{len(self.items)} image(s) ready")
        else:
            self._set_status(f"Building previews… {self._loaded}/{len(self.items)}")
        self._update_thumb(item)
        if item is self.current:
            self._load_settings_to_ui()
            self._schedule_render(0)

    def _on_failed(self, path, msg):
        self._load_jobs.pop(path, None)
        item = self.by_path.get(path)
        if item is not None and item is self.current:
            self.view.set_overlay("")
            self.view.set_image(None)
            self.view.set_message(f"Could not open {item.name}\n\n{msg}")
        if item is not None:
            item.error = msg
            if self._loaded < len(self.items):
                self._loaded += 1
        self._set_status(f"{Path(path).name}: {msg}")

    # ------------------------------------------------------------------ current image
    def _on_current_changed(self, cur, _prev):
        item = self.by_path.get(cur.data(Qt.UserRole)) if cur is not None else None
        if item is self.current:
            return
        self._exit_tools()
        self.current = item
        self._full = None                    # free the previous full-resolution image
        self._full_path = None
        self._tile_gen += 1
        self.view.set_badge("")
        self.view.full_size = None
        self._load_settings_to_ui()
        self._update_enabled()
        if item is None:
            self.view.set_image(None, reset_view=True)
            self.setWindowTitle("eNegmatic")
            return
        self.setWindowTitle(f"{item.name} — eNegmatic")
        self.hist.set_data(None)
        if item.error:
            self.view.set_image(None)
            self.view.set_message(f"Could not open {item.name}\n\n{item.error}")
            return
        self.view.set_image(item.thumb_qimage, reset_view=True)
        if self.cache.get(item.path) is not None:
            self.view.set_overlay("")
            self._schedule_render(0)
        else:
            self.view.set_overlay("Loading…")
            self._request_load(item, 10)

    def _step(self, d):
        if not self.items:
            return
        row = self.strip.currentRow() + d
        if 0 <= row < self.strip.count():
            self.strip.setCurrentRow(row)

    def _selected_items(self) -> list[ImageItem]:
        rows = sorted(self.strip.row(li) for li in self.strip.selectedItems())
        items = [self.items[r] for r in rows if 0 <= r < len(self.items)]
        if not items and self.current:
            items = [self.current]
        return items

    def _load_settings_to_ui(self):
        s = self.current.settings if self.current else Settings()
        for field, sl in self.sliders.items():
            sl.setValue(getattr(s, field))
        for w, on in ((self.btn_color, s.film_type != "bw"), (self.btn_bw, s.film_type == "bw")):
            w.blockSignals(True)
            w.setChecked(on)
            w.blockSignals(False)
        for cb, on in ((self.cb_auto_wb, s.auto_wb), (self.cb_auto_bright, s.auto_brightness),
                       (self.cb_lock, s.lock_analysis), (self.cb_dust, s.dust_auto)):
            cb.blockSignals(True)
            cb.setChecked(bool(on))
            cb.blockSignals(False)
        self.color_box.setVisible(s.film_type != "bw")
        self.cb_auto_wb.setEnabled(s.film_type != "bw")
        self.view.set_crop(s.crop)
        self.aspect.blockSignals(True)
        self.aspect.setCurrentIndex(max(0, self.aspect.findData(s.crop_aspect or "")))
        self.aspect.blockSignals(False)
        self._update_frame_size()
        self._update_base_swatch()
        self._update_dust_ui()
        kind = "B&W negative" if s.film_type == "bw" else "Colour negative"
        if self.current and self.current.film_detected:
            kind += " (auto-detected)"
        self.info.setText(f"{self.current.name} · {kind}" if self.current else " ")

    def _update_frame_size(self):
        if not self.current:
            return
        p = self.cache.get(self.current.path)
        if p is not None:
            self.view.frame_size = p.frame_size(self.current.settings)

    def _update_base_swatch(self):
        if not hasattr(self, "base_swatch"):
            return
        s = self.current.settings if self.current else None
        border = self.colors.get("border", "#30343b")
        if s and s.base_rgb:
            r, g, b = (int(min(max(v, 0.0), 1.0) ** (1 / 2.2) * 255) for v in s.base_rgb)
            self.base_swatch.setStyleSheet(f"background: rgb({r},{g},{b}); border: 1px solid {border};"
                                           "border-radius: 4px;")
            self.base_label.setText("Picked")
        else:
            self.base_swatch.setStyleSheet(f"background: transparent; border: 1px dashed {border};"
                                           "border-radius: 4px;")
            self.base_label.setText("Auto")

    def _update_enabled(self):
        has = self.current is not None
        for a in (self.act_rot_l, self.act_rot_r, self.act_flip, self.act_crop, self.act_autocrop,
                  self.act_compare, self.act_export, self.act_copy, self.act_copy_choose, self.act_reset):
            a.setEnabled(has)
        self.act_paste.setEnabled(has and self._clipboard is not None)
        self.act_paste_all.setEnabled(bool(self.items) and self._clipboard is not None)
        busy = self._exporter is not None
        self.act_export.setEnabled(has and not busy)
        self.act_export_all.setEnabled(bool(self.items) and not busy)
        self.act_prev.setEnabled(len(self.items) > 1)
        self.act_next.setEnabled(len(self.items) > 1)
        self.panel.setEnabled(has)
        self.strip_header.setText(f"FILMSTRIP  ·  {len(self.items)}" if self.items else "FILMSTRIP")

    # ------------------------------------------------------------------ editing
    def _touch(self, item: ImageItem):
        if self.save_sidecars:
            self._dirty_folders.add(item.folder)
            self._save_timer.start()

    def _set_field(self, name, value, reload_ui=False):
        if not self.current:
            return
        if name == "straighten":
            self.current.fresh = False
            self._straighten_to(float(value))
            return
        setattr(self.current.settings, name, value)
        self._touch(self.current)
        if reload_ui:
            self._load_settings_to_ui()
        self._schedule_render()

    def _straighten_to(self, angle, base=None):
        """Set the straighten angle, turning the picture about the crop centre."""
        item = self.current
        s = item.settings
        proxy = self.cache.get(item.path)
        a0, c0 = base if base is not None else (s.straighten, s.crop)
        if proxy is not None:
            s.crop = straighten_crop(c0, proxy.frame_size(s), a0, angle)
        s.straighten = angle
        self.view.set_crop(s.crop)
        self._touch(item)
        self._interactive()
        self._schedule_render()

    def _interactive(self):
        """Mark a geometry drag: renders reuse the analysis until it settles."""
        self._fast_until = time.monotonic() + 0.3
        self._settle_timer.start()

    def _on_rotate_drag(self, delta, done):
        if not self.current:
            return
        s = self.current.settings
        if self._rot_base is None:
            self._rot_base = (s.straighten, s.crop)
        a0 = self._rot_base[0]
        angle = round(min(max(a0 - delta, -45.0), 45.0), 2)    # clockwise drag turns it clockwise
        self.current.fresh = False
        self._straighten_to(angle, self._rot_base)
        self.sliders["straighten"].setValue(angle)
        if done:
            self._rot_base = None

    def _set_film(self, kind):
        if self.current:
            self.current.film_locked = True
            self.current.film_detected = False
        self.default_film = kind
        self.prefs.setValue("default_film", kind)
        self._set_field("film_type", kind, reload_ui=True)

    def _reanalyze(self):
        self._set_field("analysis", None)

    def _reset_wb(self):
        if not self.current:
            return
        s = self.current.settings
        s.wb_gains, s.temperature, s.tint = [1.0, 1.0, 1.0], 0.0, 0.0
        self._touch(self.current)
        self._load_settings_to_ui()
        self._schedule_render()

    def _reset_section(self, key):
        if not self.current:
            return
        if key == "geometry":
            self._reset_crop()
            return
        group = {"inversion": "film", "tone": "tone", "color": "color", "detail": "detail",
                 "dust": "dust"}[key]
        defaults = Settings()
        s = self.current.settings
        for f in SETTING_GROUPS[group][1]:
            if f != "film_type":
                setattr(s, f, getattr(defaults, f))
        if key == "dust":
            s.dust_spots, s.dust_keep = [], []
        self._touch(self.current)
        self._load_settings_to_ui()
        self._schedule_render()

    def _reset_adjustments(self):
        if not self.current:
            return
        old = self.current.settings
        self.current.settings = Settings(film_type=old.film_type, rotation=old.rotation,
                                         flip_h=old.flip_h, straighten=old.straighten, crop=old.crop,
                                         dust_spots=list(old.dust_spots), dust_keep=list(old.dust_keep))
        self._touch(self.current)
        self._load_settings_to_ui()
        self._schedule_render()

    def _rotate(self, delta):
        if not self.current:
            return
        s = self.current.settings
        new_rot = (s.rotation + delta) % 360
        s.crop = transform_crop(s.crop, s.rotation, s.flip_h, new_rot, s.flip_h)
        s.rotation = new_rot
        self._touch(self.current)
        self._load_settings_to_ui()
        self._schedule_render()

    def _flip(self):
        if not self.current:
            return
        s = self.current.settings
        s.crop = transform_crop(s.crop, s.rotation, s.flip_h, s.rotation, not s.flip_h)
        s.flip_h = not s.flip_h
        s.straighten = -s.straighten
        self._touch(self.current)
        self._load_settings_to_ui()
        self._schedule_render()

    def _auto_crop(self):
        if not self.current:
            return
        proxy = self.cache.get(self.current.path)
        if proxy is None:
            self._set_status("Still loading — try again in a moment.")
            return
        s = self.current.settings
        angle, crop = detect_frame(orient(proxy.neg, s.rotation, s.flip_h), self.autocrop_inset / 100.0)
        if crop is None:
            self._set_status("No frame border found — showing the full image.")
            s.straighten, s.crop = 0.0, None
        else:
            self._apply_autocrop(s, proxy, angle, crop)
            self._set_status(f"Frame detected: {s.crop_aspect} (straightened {angle:+.2f}°).")
        self.current.fresh = False
        self._touch(self.current)
        self._load_settings_to_ui()
        self._schedule_render()

    @staticmethod
    def _apply_autocrop(s: Settings, proxy: ProxyData, angle, crop):
        """Use a detected frame: recognise its format (3:2 if unsure) and crop to it exactly."""
        wh = proxy.frame_size(s)
        s.crop_aspect = detect_aspect(crop, wh)
        s.straighten = angle
        s.crop = fit_aspect(crop, wh, oriented_ratio(s.crop_aspect, crop, wh))

    def _reset_crop(self):
        if not self.current:
            return
        s = self.current.settings
        s.crop, s.straighten = None, 0.0
        self.current.fresh = False
        self._touch(self.current)
        self._load_settings_to_ui()
        self._schedule_render()

    def _on_crop_changed(self, crop):
        if not self.current:
            return
        full = crop[0] <= 1e-4 and crop[1] <= 1e-4 and crop[2] >= 1 - 1e-4 and crop[3] >= 1 - 1e-4
        self.current.settings.crop = None if full else crop
        self.current.fresh = False
        self._touch(self.current)
        self._schedule_render()

    def _aspect_for_view(self) -> float | None:
        """Pixel w/h the crop tool locks to: the frame's format, matching the crop's orientation."""
        if not self.current:
            return None
        s = self.current.settings
        proxy = self.cache.get(self.current.path)
        if not s.crop_aspect or proxy is None:
            return None
        return oriented_ratio(s.crop_aspect, s.crop, proxy.frame_size(s))

    def _on_aspect(self, _=None):
        if not self.current:
            return
        self.current.settings.crop_aspect = self.aspect.currentData() or ""
        self._touch(self.current)
        if self._crop_mode:
            self.view.set_aspect(self._aspect_for_view())
        else:
            self._toggle_crop(True)

    def _swap_aspect(self):
        r = self._aspect_for_view()
        if not r:
            return
        if not self._crop_mode:
            self._toggle_crop(True)
        self.view.set_aspect(1.0 / r)

    # ------------------------------------------------------------------ tools
    def _sync_tool_buttons(self):
        for w, on in ((self.btn_crop, self._crop_mode), (self.act_crop, self._crop_mode),
                      (self.btn_pick_base, self._pick == "base"), (self.btn_pick_wb, self._pick == "wb"),
                      (self.act_compare, self._compare), (self.btn_heal, self._heal), (self.act_heal, self._heal)):
            w.blockSignals(True)
            w.setChecked(on)
            w.blockSignals(False)

    def _apply_view_mode(self):
        if self._pick:
            self.view.set_mode(ImageView.PICK)
            self.view.set_overlay("Click the unexposed film border (base)" if self._pick == "base"
                                  else "Click something that should be neutral grey")
        elif self._crop_mode:
            self.view.set_mode(ImageView.CROP)
            self.view.set_overlay("Drag to crop · drag outside a corner to rotate · Enter to apply")
        elif self._heal:
            self.view.set_mode(ImageView.HEAL)
            self.view.set_overlay("Paint over dust · right-click undoes · Shift+wheel brush size")
        else:
            self.view.set_mode(ImageView.VIEW)
            self.view.set_overlay("Original negative" if self._compare else "")
        if self.current:
            self.view.set_crop(self.current.settings.crop)
        self._sync_tool_buttons()
        self._update_heal_overlay()
        self._schedule_render(0)

    def _toggle_crop(self, on=None):
        on = (not self._crop_mode) if on is None or not isinstance(on, bool) else on
        if not self.current:
            on = False
        self._crop_mode = on
        if on:
            self._pick = None
            self._compare = False
            self._heal = False
        self._apply_view_mode()
        # the frame's format stays chosen between visits to the crop tool
        self.view.set_aspect(self._aspect_for_view() if on else None)

    def _toggle_compare(self, on=None):
        self._compare = (not self._compare) if not isinstance(on, bool) else on
        if self._compare:
            self._crop_mode = False
            self._pick = None
            self._heal = False
        self._apply_view_mode()

    def _toggle_heal(self, on=None):
        on = (not self._heal) if not isinstance(on, bool) else on
        self._heal = bool(on and self.current)
        if self._heal:
            self._crop_mode, self._pick, self._compare = False, None, False
        self._apply_view_mode()

    def _start_pick(self, kind, on):
        if on and self.current:
            self._pick = kind
            self._crop_mode = False
            self._compare = False
            self._heal = False
        elif self._pick == kind:
            self._pick = None
        self._apply_view_mode()

    def _exit_tools(self):
        if self._crop_mode or self._pick or self._compare or self._heal:
            self._crop_mode, self._pick, self._compare, self._heal = False, None, False, False
            self.view.set_aspect(None)
            self._apply_view_mode()

    def _on_picked(self, nx, ny):
        item = self.current
        proxy = self.cache.get(item.path) if item else None
        if proxy is None or not self._pick:
            return
        s = item.settings
        full = apply_geometry(proxy.neg, s, crop=False)
        if self._pick == "base":
            s.base_rgb = [float(v) for v in sample(full, nx, ny, radius=4)]
            self._set_status("Film base set.")
        else:
            cropped = apply_crop(full, s.crop)
            a = resolve_analysis(cropped, s)
            s.analysis = a
            s.wb_gains = neutral_wb_gains(sample(cropped, nx, ny, radius=3), s, a)
            s.temperature = s.tint = 0.0
            self._set_status("White balance set from picked point.")
        self._pick = None
        self._touch(item)
        self._load_settings_to_ui()
        self._apply_view_mode()

    # ------------------------------------------------------------------ dust / healing brush
    def _set_brush(self, size):
        self.brush_size = min(max(float(size), 0.2), 5.0)
        self.prefs.setValue("brush_size", self.brush_size)
        self.brush_slider.setValue(self.brush_size)
        self._update_heal_overlay()

    def _update_heal_overlay(self):
        item = self.current
        proxy = self.cache.get(item.path) if item else None
        if not self._heal or proxy is None:
            self.view.set_spots([], [])
            return
        s = item.settings
        shape = proxy.neg.shape
        k = view_scale(shape, s)
        self.view.set_brush(self.brush_size / 100.0 / k)

        def to_view(spots):
            if not spots:
                return []
            pts = src_to_view([sp[:2] for sp in spots], shape, s)
            return [(float(x), float(y), float(sp[2]) / k) for (x, y), sp in zip(pts, spots)]

        auto = proxy.dust_specks if s.dust_auto else []
        self.view.set_spots(to_view(s.dust_spots), to_view(auto))

    def _update_dust_ui(self):
        s = self.current.settings if self.current else Settings()
        proxy = self.cache.get(self.current.path) if self.current else None
        parts = []
        if s.dust_auto and proxy is not None:
            n = len(proxy.dust_specks)
            parts.append(f"{n} speck{'s' if n != 1 else ''} found in the preview (full-resolution "
                         "export also catches finer ones)")
        if s.dust_spots:
            parts.append(f"{len(s.dust_spots)} brush spot{'s' if len(s.dust_spots) != 1 else ''}")
        if s.dust_keep:
            parts.append(f"{len(s.dust_keep)} kept")
        self.dust_info.setText(" · ".join(parts))
        self.dust_info.setVisible(bool(parts))

    def _on_heal_painted(self, points):
        item = self.current
        proxy = self.cache.get(item.path) if item else None
        if proxy is None or not points:
            return
        s = item.settings
        r = round(self.brush_size / 100.0, 5)
        src = view_to_src(points, proxy.neg.shape, s)
        new = [[round(float(x), 5), round(float(y), 5), r] for x, y in src if 0 <= x <= 1 and 0 <= y <= 1]
        if not new:
            return
        s.dust_spots = list(s.dust_spots) + new
        self._touch(item)
        self._update_dust_ui()
        self._update_heal_overlay()
        self._schedule_render()

    def _on_heal_erase(self, nx, ny):
        item = self.current
        proxy = self.cache.get(item.path) if item else None
        if proxy is None:
            return
        s = item.settings
        h, w = proxy.neg.shape[:2]
        L = max(h, w)
        (x, y), = view_to_src([(nx, ny)], proxy.neg.shape, s)

        def dist(px, py):
            return ((px - x) * w) ** 2 + ((py - y) * h) ** 2

        tol = 4.0 / L
        hits = [i for i, sp in enumerate(s.dust_spots) if dist(sp[0], sp[1]) ** 0.5 / L <= sp[2] + tol]
        if hits:
            i = min(hits, key=lambda j: dist(*s.dust_spots[j][:2]))
            s.dust_spots = [sp for j, sp in enumerate(s.dust_spots) if j != i]
            self._set_status("Brush spot removed.")
        else:
            kept = [i for i, p in enumerate(s.dust_keep) if dist(p[0], p[1]) ** 0.5 / L <= 0.006]
            specks = [sp for sp in (proxy.dust_specks if s.dust_auto else [])
                      if dist(sp[0], sp[1]) ** 0.5 / L <= sp[2] * 1.6 + 0.004]
            if specks:
                sp = min(specks, key=lambda q: dist(q[0], q[1]))
                s.dust_keep = list(s.dust_keep) + [[round(sp[0], 5), round(sp[1], 5)]]
                self._set_status("Speck kept: auto removal will leave it alone (right-click again to undo).")
            elif kept:
                s.dust_keep = [p for j, p in enumerate(s.dust_keep) if j not in kept]
                self._set_status("Auto removal may remove that speck again.")
            else:
                self._set_status("Nothing to undo there.")
                return
        self._touch(item)
        self._update_dust_ui()
        self._update_heal_overlay()
        self._schedule_render()

    def _undo_spot(self):
        if self.current and self.current.settings.dust_spots:
            s = self.current.settings
            s.dust_spots = list(s.dust_spots)[:-1]
            self._touch(self.current)
            self._update_dust_ui()
            self._update_heal_overlay()
            self._schedule_render()
            self._set_status("Last brush spot undone.")

    def _clear_spots(self):
        if not self.current:
            return
        s = self.current.settings
        if s.dust_spots or s.dust_keep:
            s.dust_spots, s.dust_keep = [], []
            self._touch(self.current)
            self._update_dust_ui()
            self._update_heal_overlay()
            self._schedule_render()
            self._set_status("Dust spots cleared.")

    # ------------------------------------------------------------------ copy / paste
    def _copy_settings(self, choose=False):
        if not self.current:
            return
        if choose or not self.copy_groups:
            dlg = CopyDialog(self, self.copy_groups or list(DEFAULT_COPY_GROUPS))
            if dlg.exec() != QDialog.Accepted or not dlg.groups():
                return
            self.copy_groups = dlg.groups()
            self.prefs.setValue("copy_groups", json.dumps(self.copy_groups))
        s = self.current.settings
        wanted = {f for g in self.copy_groups for f in SETTING_GROUPS[g][1]}
        clip = {k: v for k, v in s.to_dict().items() if k in wanted}
        if not s.lock_analysis:
            clip.pop("analysis", None)
        self._clipboard = clip
        self._update_enabled()
        names = ", ".join(SETTING_GROUPS[g][0].lower() for g in self.copy_groups)
        self._set_status(f"Copied {names}" + (" (with locked analysis)." if "analysis" in clip else "."))

    def _paste(self, items):
        if not self._clipboard or not items:
            return
        clip = self._clipboard
        for it in items:
            old = it.settings
            new = Settings.from_dict({**old.to_dict(), **clip}).copy()   # own copies of lists
            if "film_type" in clip:
                it.film_locked, it.film_detected = True, False
            if "crop" in clip:
                it.fresh = False                          # don't let a pending auto crop override it
            elif (old.rotation, old.flip_h) != (new.rotation, new.flip_h):
                new.crop = transform_crop(old.crop, old.rotation, old.flip_h, new.rotation, new.flip_h)
                if old.flip_h != new.flip_h:
                    new.straighten = -old.straighten
            it.settings = new
            self._touch(it)
            if it is not self.current:
                self._update_thumb(it)
        if self.current in items:
            self._load_settings_to_ui()
            self._schedule_render()
        self._set_status(f"Settings pasted to {len(items)} image(s).")

    # ------------------------------------------------------------------ rendering
    def _schedule_render(self, delay: int = 12):
        self._gen += 1
        self._render_timer.start(delay)

    def _full_view(self) -> bool:
        return self._crop_mode or self._pick == "base"

    def _display_full_size(self, proxy: ProxyData, s: Settings, full_view: bool):
        """Full-resolution pixel size of what the viewer shows (defines 100% zoom)."""
        if proxy.full_shape is None:
            return None
        fh, fw = proxy.full_shape
        if (s.rotation // 90) % 2:
            fw, fh = fh, fw
        if s.crop and not full_view:
            x0, y0, x1, y1 = s.crop
            fw, fh = fw * (x1 - x0), fh * (y1 - y0)
        return (max(1, round(fw)), max(1, round(fh)))

    def _view_sig(self, s: Settings, full_view: bool) -> str:
        return json.dumps([s.to_dict(), full_view, self._compare], sort_keys=True)

    def _do_render(self):
        item = self.current
        if item is None:
            return
        proxy = self.cache.get(item.path)
        if proxy is None:
            self._request_load(item, 10)
            return
        self._update_frame_size()
        full_view = self._full_view()
        self._render_full_size = (self._gen, self._display_full_size(proxy, item.settings, full_view))
        job = RenderJob(self._gen, item.path, proxy, item.settings.copy(), full_view,
                        self._compare, self.signals, lambda: self._gen,
                        fast=time.monotonic() < self._fast_until)
        self.render_pool.start(job)

    def _on_rendered(self, gen, path, qimg, analysis, hist, mono):
        item = self.current
        if gen != self._gen or item is None or path != item.path:
            return
        if not self._compare and analysis != item.settings.analysis:
            item.settings.analysis = analysis
            self._touch(item)
        if self._render_full_size[0] == gen:
            self.view.full_size = self._render_full_size[1]
        if self._tile_sig != self._view_sig(item.settings, self._full_view()):
            self.view.clear_tile()            # stale: settings or view changed
        self.view.set_image(qimg)
        self._update_heal_overlay()
        self._update_dust_ui()
        if not self._pick and not self._crop_mode and not self._heal:
            self.view.set_overlay("Original negative" if self._compare else "")
        self.hist.set_data(hist, mono)
        if not (self._compare or self._crop_mode or self._pick == "base"):
            item.thumb_qimage = qimg.scaled(THUMB_SIZE * 2, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            item.list_item.setIcon(make_icon(QPixmap.fromImage(item.thumb_qimage)))

    # ------------------------------------------------------------------ zoom (full resolution)
    def _on_viewport(self, vp):
        self._viewport = vp
        item = self.current
        if item is None or item.error or self._compare or not vp["need"]:
            self.view.set_badge("")
            if not vp["zoomed"]:
                self.view.clear_tile()
            return
        if self._full is None or self._full_path != item.path:
            if self._full_loading != item.path:
                self._full_loading = item.path
                self._full_job = FullLoadJob(item.path, self.signals)
                self.load_pool.start(self._full_job, 20)
            self.view.set_badge("loading full resolution…")
            return
        self._tile_gen += 1
        full_view = self._full_view()
        job = TileJob(self._tile_gen, item.path, self._full, item.settings.copy(), full_view,
                      vp["rect"], vp["display_w"], self._view_sig(item.settings, full_view),
                      self.signals, lambda: self._tile_gen)
        self.tile_pool.start(job)

    def _on_full_loaded(self, path, arr):
        if self._full_loading == path:
            self._full_loading = None
        if self.current is None or path != self.current.path:
            return
        self._full = ProxyData(arr)
        self._full_path = path
        self.view.set_badge("")
        if self._viewport:
            self._on_viewport(self._viewport)

    def _on_full_failed(self, path, msg):
        if self._full_loading == path:
            self._full_loading = None
        if self.current is not None and path == self.current.path:
            self.view.set_badge("full resolution unavailable")
            self._set_status(msg)

    def _on_tile(self, gen, path, qimg, rect, sig):
        item = self.current
        if gen != self._tile_gen or item is None or path != item.path:
            return
        if sig != self._view_sig(item.settings, self._full_view()):
            return
        self._tile_sig = sig
        self.view.set_tile(qimg, rect)

    def _update_thumb(self, item: ImageItem):
        if item.thumb_neg is None or item.list_item is None:
            return
        s = item.settings
        try:
            g = apply_geometry(item.thumb_neg, s)
            img8 = to_uint8(render(g, s, resolve_analysis(g, s)))
        except Exception:
            return
        item.thumb_qimage = np_to_qimage(img8)
        item.list_item.setIcon(make_icon(QPixmap.fromImage(item.thumb_qimage)))

    # ------------------------------------------------------------------ persistence
    def _save_sidecars(self):
        if not self.save_sidecars:
            self._dirty_folders.clear()
            return
        for folder in list(self._dirty_folders):
            data = {it.name: it.settings.to_dict() for it in self.items if it.folder == folder}
            try:
                save_sidecar(folder, data)
            except OSError as e:
                self._set_status(f"Could not save edits in {folder}: {e}")
        self._dirty_folders.clear()

    # ------------------------------------------------------------------ export
    def _export(self, scope):
        if not self.items or self._exporter is not None:
            return
        selected = self._selected_items()
        counts = {"current": 1 if self.current else 0, "selected": len(selected), "all": len(self.items)}
        if counts[scope] == 0:
            scope = "all"
        dlg = ExportDialog(self, self.export_options, counts, scope)
        if dlg.exec() != QDialog.Accepted:
            return
        opts, scope = dlg.result_options()
        self.export_options = opts
        self.prefs.setValue("export", json.dumps(opts.to_dict()))
        targets = {"current": [self.current] if self.current else [],
                   "selected": selected, "all": list(self.items)}[scope]
        targets = [t for t in targets if t is not None and not t.error]
        if not targets:
            return
        self._save_sidecars()
        jobs = [(it.path, it.settings.to_dict()) for it in targets]
        self._exporter = ExportRunner(jobs, opts, self)
        self._exporter.progress.connect(self._on_export_progress)
        self._exporter.done.connect(self._on_export_done)
        self._exporter.finished.connect(self._exporter.deleteLater)
        self.progress.setRange(0, len(jobs))
        self.progress.setValue(0)
        self.progress.show()
        self.btn_cancel.show()
        self._set_status(f"Exporting {len(jobs)} image(s)…")
        self._update_enabled()
        self._exporter.start()

    def _cancel_export(self):
        if self._exporter is not None:
            self._exporter.cancel()
            self._set_status("Cancelling… (finishing images in progress)")

    def _on_export_progress(self, done, total, name):
        self.progress.setValue(done)
        self._set_status(f"Exporting… {done}/{total}  {name}")

    def _on_export_done(self, results):
        self.progress.hide()
        self.btn_cancel.hide()
        self._exporter = None
        self._update_enabled()
        ok = [r for r in results if r[1]]
        errs = [r for r in results if r[2]]
        self._set_status(f"Exported {len(ok)} image(s)" + (f", {len(errs)} failed." if errs else "."))
        box = QMessageBox(self)
        box.setWindowTitle("Export finished")
        box.setIcon(QMessageBox.Warning if errs else QMessageBox.Information)
        text = f"Exported {len(ok)} image(s)."
        if errs:
            text += f"\n{len(errs)} failed:\n" + "\n".join(
                f"• {Path(s).name}: {e}" for s, _, e in errs[:8])
        box.setText(text)
        open_btn = box.addButton("Open Folder", QMessageBox.AcceptRole) if ok else None
        box.addButton("Close", QMessageBox.RejectRole)
        box.exec()
        if ok and box.clickedButton() is open_btn:
            folder = str(Path(ok[0][1]).parent)
            if sys.platform == "win32":
                os.startfile(folder)
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(folder))

    # ------------------------------------------------------------------ misc UI
    def _strip_menu(self, pos):
        if not self.items:
            return
        m = QMenu(self)
        m.addAction(self.act_copy)
        m.addAction(self.act_paste)
        m.addAction(self.act_paste_all)
        m.addSeparator()
        m.addAction("Export Selected…", lambda: self._export("selected"))
        m.addSeparator()
        m.addAction("Remove from List", self._remove_selected)
        m.exec(self.strip.mapToGlobal(pos))

    def _set_status(self, text):
        self.status.setText(text)

    def _show_help(self):
        QMessageBox.information(self, "Shortcuts & tips", (
            "<b>Navigation</b><br>← / → previous / next frame (click the image first)<br><br>"
            "<b>Tools</b><br>C crop tool (drag outside a corner to rotate) · A auto crop · "
            "B show negative · [ / ] rotate · Shift+F flip · Esc leave tool · Enter apply crop<br>"
            "H healing brush: paint over dust · right-click undoes a spot (or keeps an auto-found "
            "speck) · Shift+wheel brush size · Ctrl+Z undo last spot · middle-drag to pan<br><br>"
            "<b>Settings</b><br>Ctrl+Shift+C copy · Ctrl+Shift+V paste to selected · Ctrl+R reset<br>"
            "Double-click any slider (or its label) to reset it.<br><br>"
            "<b>Tips</b><br>• For best colour, Pick the film base on the clear border between frames.<br>"
            "• For a consistent roll: get one frame right, tick <i>Lock analysis</i>, Copy, then "
            "<i>Paste to all</i>.<br>"
            "• Edits are saved automatically next to the images in <i>.enegmatic.json</i>."))

    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key_Escape:
            self._exit_tools()
        elif k in (Qt.Key_Return, Qt.Key_Enter) and self._crop_mode:
            self._toggle_crop(False)
        elif e.matches(QKeySequence.Undo):
            self._undo_spot()
        elif k == Qt.Key_Left:
            self._step(-1)
        elif k == Qt.Key_Right:
            self._step(1)
        else:
            super().keyPressEvent(e)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.open_paths(paths, add=True)

    def closeEvent(self, e):
        if self._exporter is not None:
            r = QMessageBox.question(self, "Export running",
                                     "An export is still running. Cancel it and quit?")
            if r != QMessageBox.Yes:
                e.ignore()
                return
            self._exporter.cancel()
            self._exporter.wait()
        self._save_sidecars()
        self.prefs.setValue("geometry", self.saveGeometry())
        self.prefs.setValue("splitter", self.splitter.saveState())
        self.load_pool.clear()
        self.render_pool.clear()
        self.tile_pool.clear()
        self.load_pool.waitForDone(3000)
        self.render_pool.waitForDone(3000)
        self.tile_pool.waitForDone(3000)
        super().closeEvent(e)
