"""Export dialog and background export runner."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QRadioButton, QSpinBox, QVBoxLayout, QWidget)

from .batch import ExportOptions, export_one
from .colorspace import SPACES

SCOPES = ("current", "selected", "all")


class ExportDialog(QDialog):
    def __init__(self, parent, options: ExportOptions, counts: dict, scope: str):
        super().__init__(parent)
        self.setWindowTitle("Export positives")
        self.setMinimumWidth(460)
        o = options
        root = QVBoxLayout(self)
        root.setSpacing(12)

        # scope
        scope_row = QHBoxLayout()
        self._scope = QButtonGroup(self)
        labels = {"current": "Current image", "selected": f"Selected ({counts['selected']})",
                  "all": f"All ({counts['all']})"}
        for i, key in enumerate(SCOPES):
            rb = QRadioButton(labels[key])
            rb.setEnabled(counts[key] > 0)
            self._scope.addButton(rb, i)
            scope_row.addWidget(rb)
            if key == scope:
                rb.setChecked(True)
        scope_row.addStretch(1)
        root.addLayout(scope_row)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(8)

        self.fmt = QComboBox()
        for key, label in (("jpeg", "JPEG"), ("tiff", "TIFF"), ("png", "PNG")):
            self.fmt.addItem(label, key)
        self.fmt.setCurrentIndex(max(0, self.fmt.findData(o.fmt)))
        form.addRow("Format", self.fmt)

        self.bits = QComboBox()
        self.bits.addItem("8-bit", 8)
        self.bits.addItem("16-bit", 16)
        self.bits.setCurrentIndex(1 if o.bits == 16 else 0)
        form.addRow("Bit depth", self.bits)

        self.space = QComboBox()
        for key, sp in SPACES.items():
            self.space.addItem(sp.label, key)
        self.space.setCurrentIndex(max(0, self.space.findData(o.color_space)))
        self.space.setToolTip("Colour space of the exported files. Each file gets a matching\n"
                              "embedded ICC profile, so colour-managed apps show it correctly.")
        self.space_hint = QLabel()
        self.space_hint.setObjectName("Muted")
        self.space_hint.setWordWrap(True)
        space_box = QWidget()
        sv = QVBoxLayout(space_box)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(3)
        sv.addWidget(self.space)
        sv.addWidget(self.space_hint)
        form.addRow("Colour space", space_box)

        self.quality = QSpinBox()
        self.quality.setRange(50, 100)
        self.quality.setValue(int(o.quality))
        form.addRow("JPEG quality", self.quality)

        self.long_edge = QSpinBox()
        self.long_edge.setRange(0, 20000)
        self.long_edge.setSingleStep(100)
        self.long_edge.setSpecialValueText("Full resolution")
        self.long_edge.setSuffix(" px")
        self.long_edge.setValue(int(o.long_edge))
        form.addRow("Long edge", self.long_edge)

        self.subfolder_cb = QCheckBox("Next to the originals, in subfolder")
        self.subfolder_cb.setChecked(o.use_subfolder)
        self.subfolder = QLineEdit(o.subfolder)
        self.subfolder.setMaximumWidth(140)
        row = QHBoxLayout()
        row.addWidget(self.subfolder_cb)
        row.addWidget(self.subfolder)
        row.addStretch(1)
        w = QWidget()
        w.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        form.addRow("Destination", w)

        self.out_dir = QLineEdit(o.out_dir)
        self.out_dir.setPlaceholderText("Choose a folder…")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        row2 = QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        row2.addWidget(self.out_dir, 1)
        row2.addWidget(browse)
        self._dir_row = QWidget()
        self._dir_row.setLayout(row2)
        form.addRow("Folder", self._dir_row)

        self.suffix = QLineEdit(o.suffix)
        self.suffix.setPlaceholderText("e.g. _pos")
        form.addRow("Filename suffix", self.suffix)

        self.overwrite = QCheckBox("Overwrite existing files")
        self.overwrite.setChecked(o.overwrite)
        form.addRow("", self.overwrite)

        self.gray = QCheckBox("Save B&&W film as single-channel grayscale")
        self.gray.setChecked(o.bw_grayscale)
        form.addRow("", self.gray)

        self.workers = QSpinBox()
        self.workers.setRange(1, max(1, os.cpu_count() or 1))
        self.workers.setValue(min(int(o.workers), self.workers.maximum()))
        self.workers.setToolTip("Images processed in parallel. Each needs ~1–2 GB RAM for large raws.")
        form.addRow("Parallel jobs", self.workers)

        root.addLayout(form)
        hint = QLabel("Full-resolution files are decoded and processed with the same settings as the preview.")
        hint.setObjectName("Muted")
        hint.setWordWrap(True)
        root.addWidget(hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Export")
        buttons.button(QDialogButtonBox.Ok).setObjectName("Primary")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.fmt.currentIndexChanged.connect(self._sync)
        self.subfolder_cb.toggled.connect(self._sync)
        self.space.currentIndexChanged.connect(self._sync)
        self.bits.currentIndexChanged.connect(self._sync)
        self._sync()

    def _sync(self):
        is_jpeg = self.fmt.currentData() == "jpeg"
        self.quality.setEnabled(is_jpeg)
        self.bits.setEnabled(not is_jpeg)
        if is_jpeg:
            self.bits.setCurrentIndex(0)
        key = self.space.currentData()
        hint = SPACES[key].hint
        if key == "prophoto" and (is_jpeg or self.bits.currentData() == 8):
            hint += "  ⚠ 8-bit is too coarse for ProPhoto and can cause banding."
        self.space_hint.setText(hint)
        sub = self.subfolder_cb.isChecked()
        self.subfolder.setEnabled(sub)
        self._dir_row.setEnabled(not sub)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Export folder", self.out_dir.text())
        if d:
            self.out_dir.setText(d)

    def _accept(self):
        if not self.subfolder_cb.isChecked() and not self.out_dir.text().strip():
            self._browse()
            if not self.out_dir.text().strip():
                return
        self.accept()

    def result_options(self) -> tuple[ExportOptions, str]:
        o = ExportOptions(
            fmt=self.fmt.currentData(), bits=int(self.bits.currentData()),
            quality=self.quality.value(), long_edge=self.long_edge.value(),
            use_subfolder=self.subfolder_cb.isChecked(),
            subfolder=self.subfolder.text().strip() or "positives",
            out_dir=self.out_dir.text().strip(), suffix=self.suffix.text(),
            overwrite=self.overwrite.isChecked(), bw_grayscale=self.gray.isChecked(),
            color_space=self.space.currentData(), workers=self.workers.value(),
        )
        return o, SCOPES[max(0, self._scope.checkedId())]


class ExportRunner(QThread):
    progress = Signal(int, int, str)          # done, total, last file name
    done = Signal(list)                       # [(src, out, err), ...]

    def __init__(self, jobs: list[tuple[str, dict]], options: ExportOptions, parent=None):
        super().__init__(parent)
        self.jobs = jobs
        self.options = options
        self._cancel = False
        self._pool = None

    def cancel(self):
        self._cancel = True
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)

    def run(self):
        results = []
        total = len(self.jobs)
        opts = self.options.to_dict()
        workers = max(1, min(int(self.options.workers), total))
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                self._pool = pool
                futures = [pool.submit(export_one, src, s, opts) for src, s in self.jobs]
                for fut in as_completed(futures):
                    if fut.cancelled():
                        continue
                    try:
                        res = fut.result()
                    except Exception as e:
                        res = ("?", None, f"{type(e).__name__}: {e}")
                    results.append(res)
                    self.progress.emit(len(results), total, os.path.basename(res[0]))
                    if self._cancel:
                        break
        except Exception as e:
            results.append(("", None, f"Export failed: {e}"))
        self._pool = None
        self.done.emit(results)
