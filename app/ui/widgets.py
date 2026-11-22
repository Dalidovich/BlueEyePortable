from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from PIL import Image, ImageOps
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QResizeEvent, QStandardItemModel
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QPushButton, QSizePolicy, QTableWidget, QTableWidgetItem, QWidget)

from settings import INPUT_DIR, PRECISIONS, from_portable_path
from ui.device_state import BF16, DeviceState

PREVIEW_MAX_SIDE = 1600
PREVIEW_WAIT_MILLISECONDS = 5000

DirectoryChooser = Callable[[str, Path], Path | None]


class DeviceSelector(QWidget):
    def __init__(self, state: DeviceState, parent: QWidget | None = None):
        super().__init__(parent)
        self._state = state
        self.device_combo = QComboBox()
        self.device_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.precision_combo = QComboBox()
        for precision in PRECISIONS:
            self.precision_combo.addItem(precision, precision)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Device:"))
        layout.addWidget(self.device_combo)
        layout.addSpacing(12)
        layout.addWidget(QLabel("Precision:"))
        layout.addWidget(self.precision_combo)
        layout.addStretch(1)
        self.device_combo.activated.connect(self._on_device_activated)
        self.precision_combo.activated.connect(self._on_precision_activated)
        state.changed.connect(self.sync_from_state)
        self.sync_from_state()

    def device_values(self) -> list[str]:
        return [self.device_combo.itemData(i) for i in range(self.device_combo.count())]

    def precision_enabled(self, precision: str) -> bool:
        index = self.precision_combo.findData(precision)
        item = self._precision_model().item(index)
        return item is not None and item.isEnabled()

    @Slot()
    def sync_from_state(self) -> None:
        state = self._state
        self.device_combo.blockSignals(True)
        self.precision_combo.blockSignals(True)
        try:
            self.device_combo.clear()
            for value, label in state.device_choices():
                self.device_combo.addItem(label, value)
            index = self.device_combo.findData(state.preference)
            self.device_combo.setCurrentIndex(max(0, index))
            bf16_item = self._precision_model().item(self.precision_combo.findData(BF16))
            supported = state.bf16_supported()
            bf16_item.setEnabled(supported)
            bf16_item.setToolTip("" if supported else state.bf16_unsupported_reason())
            self.precision_combo.setCurrentIndex(max(0, self.precision_combo.findData(state.precision)))
        finally:
            self.device_combo.blockSignals(False)
            self.precision_combo.blockSignals(False)

    def _precision_model(self) -> QStandardItemModel:
        return cast(QStandardItemModel, self.precision_combo.model())

    @Slot(int)
    def _on_device_activated(self, index: int) -> None:
        self._state.set_preference(self.device_combo.itemData(index))

    @Slot(int)
    def _on_precision_activated(self, index: int) -> None:
        self._state.set_precision(self.precision_combo.itemData(index))


class ElidedLabel(QLabel):
    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self._full_text = ""
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(40)
        self.set_full_text(text)

    def full_text(self) -> str:
        return self._full_text

    def set_full_text(self, text: str) -> None:
        self._full_text = text
        self.setToolTip(text)
        self._apply_elision()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._apply_elision()

    def _apply_elision(self) -> None:
        width = max(0, self.contentsRect().width())
        self.setText(self.fontMetrics().elidedText(self._full_text, Qt.TextElideMode.ElideMiddle, width))


class FolderField(QWidget):
    def __init__(self, title: str, chooser: DirectoryChooser, parent: QWidget | None = None):
        super().__init__(parent)
        self._title = title
        self._chooser = chooser
        self.edit = QLineEdit()
        self.browse_button = QPushButton("Browse...")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.browse_button)
        self.browse_button.clicked.connect(self._browse)

    def text(self) -> str:
        return self.edit.text().strip()

    def set_path(self, path: Path) -> None:
        self.edit.setText(str(path))

    def path(self) -> Path | None:
        text = self.text()
        return from_portable_path(text) if text else None

    @Slot()
    def _browse(self) -> None:
        current = self.path()
        start = current if current is not None and current.is_dir() else INPUT_DIR
        chosen = self._chooser(self._title, start)
        if chosen is not None:
            self.set_path(chosen)


@dataclass
class PreviewPixels:
    data: bytes
    width: int
    height: int


def load_preview_pixels(path: Path, max_side: int = PREVIEW_MAX_SIDE) -> PreviewPixels:
    with Image.open(path) as source:
        source.draft("RGB", (max_side, max_side))
        image = ImageOps.exif_transpose(source)
    image.thumbnail((max_side, max_side))
    rgb = image.convert("RGB")
    return PreviewPixels(rgb.tobytes(), rgb.width, rgb.height)


class _PreviewSignals(QObject):
    loaded = Signal(int, object)


class _PreviewTask(QRunnable):
    def __init__(self, request_id: int, path: Path, signals: _PreviewSignals):
        super().__init__()
        self._request_id = request_id
        self._path = path
        self._signals = signals

    def run(self) -> None:
        try:
            outcome: object = load_preview_pixels(self._path)
        except Exception as error:
            outcome = f"Cannot display image:\n{type(error).__name__}: {error}"
        self._signals.loaded.emit(self._request_id, outcome)


class ImagePreview(QLabel):
    image_loaded = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setWordWrap(True)
        self.setStyleSheet("QLabel { background: #202020; color: #e0e0e0; }")
        self._pixmap: QPixmap | None = None
        self._request_id = 0
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._signals = _PreviewSignals(self)
        self._signals.loaded.connect(self._on_loaded)
        self.show_message("No image selected")

    def has_image(self) -> bool:
        return self._pixmap is not None

    def show_message(self, text: str) -> None:
        self._request_id += 1
        self._pixmap = None
        self.setPixmap(QPixmap())
        self.setText(text)

    def show_image(self, path: Path) -> None:
        self._request_id += 1
        self._pixmap = None
        self.setPixmap(QPixmap())
        self.setText("Loading...")
        self._pool.clear()
        self._pool.start(_PreviewTask(self._request_id, path, self._signals))

    def shutdown(self) -> None:
        self._request_id += 1
        self._pool.clear()
        self._pool.waitForDone(PREVIEW_WAIT_MILLISECONDS)

    @Slot(int, object)
    def _on_loaded(self, request_id: int, outcome: object) -> None:
        if request_id != self._request_id:
            return
        if isinstance(outcome, PreviewPixels):
            image = QImage(outcome.data, outcome.width, outcome.height, outcome.width * 3,
                           QImage.Format.Format_RGB888).copy()
            self._pixmap = QPixmap.fromImage(image)
            self.setText("")
            self._apply_pixmap()
        else:
            self._pixmap = None
            self.setPixmap(QPixmap())
            self.setText(str(outcome))
        self.image_loaded.emit()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._apply_pixmap()

    def _apply_pixmap(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        self.setPixmap(self._pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation))


def make_static_table(headers: list[str]) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.verticalHeader().setVisible(False)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)
    return table


def fill_static_table(table: QTableWidget, rows: list[list[str]], numeric_columns: set[int] | None = None) -> None:
    numeric_columns = numeric_columns or set()
    table.setRowCount(len(rows))
    for row_index, row in enumerate(rows):
        for column_index, text in enumerate(row):
            item = QTableWidgetItem(text)
            if column_index in numeric_columns:
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(row_index, column_index, item)
