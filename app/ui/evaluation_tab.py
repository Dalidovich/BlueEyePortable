from __future__ import annotations

from PySide6.QtCore import QItemSelectionModel, QModelIndex, Qt, Slot
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView, QLabel, QProgressBar,
                               QPushButton, QSplitter, QTableView, QVBoxLayout, QWidget)

from evaluation import LABELS, compute_metrics, export_metrics
from settings import INPUT_DIR, OUTPUT_DIR, PRECISIONS, from_portable_path
from ui.context import AppContext
from ui.evaluation_model import FILTERS, EvaluationTableModel
from ui.metrics_panel import MetricsPanel
from ui.widgets import ImagePreview
from ui.workers import LABEL_IMPORT_JOB, CountUpdate, LabelImportJob

NOTE_TEXT = "Quality metrics only - independent of benchmark speed."
IMPORT_TOOLTIP = ("Choose a folder whose subfolders are named safe, suggestive and explicit. Every image inside "
                  "such a subfolder gets that folder name as its ground-truth label. Manual labels are never "
                  "overwritten.")
LABEL_KEYS = (("safe", "Safe (1)", "1"), ("suggestive", "Suggestive (2)", "2"), ("explicit", "Explicit (3)", "3"))
CLEAR_KEY = "0"


class EvaluationTab(QWidget):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None):
        super().__init__(parent)
        self.ctx = ctx
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(PRECISIONS)
        self.precision_combo.setCurrentText(ctx.device_state.precision)
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(FILTERS)
        self.model = EvaluationTableModel(self)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setMinimumSectionSize(70)
        self.label_buttons: dict[str, QPushButton] = {label: QPushButton(title) for label, title, _key in LABEL_KEYS}
        self.clear_button = QPushButton("Clear (0)")
        self.import_button = QPushButton("Import labels from subfolder names...")
        self.import_button.setToolTip(IMPORT_TOOLTIP)
        self.import_progress = QProgressBar()
        self.import_progress.setVisible(False)
        self.count_label = QLabel()
        self.preview = ImagePreview()
        self.metrics_panel = MetricsPanel(self.export_metrics)
        self._build_layout()
        for label, _title, key in LABEL_KEYS:
            self.label_buttons[label].clicked.connect(lambda _checked=False, value=label: self.label_current(value))
            self._add_shortcut(key, lambda value=label: self.label_current(value))
        self.clear_button.clicked.connect(lambda: self.label_current(None))
        self._add_shortcut(CLEAR_KEY, lambda: self.label_current(None))
        self.import_button.clicked.connect(self.import_folder_labels)
        self.precision_combo.currentIndexChanged.connect(self.refresh)
        self.filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        self.table.selectionModel().currentRowChanged.connect(self._on_current_row_changed)
        ctx.events.results_changed.connect(self.refresh)
        self.refresh()

    def _add_shortcut(self, key: str, handler) -> None:
        shortcut = QShortcut(QKeySequence(key), self)
        shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut.activated.connect(handler)

    def _build_layout(self) -> None:
        selectors = QHBoxLayout()
        selectors.addWidget(QLabel("Results precision:"))
        selectors.addWidget(self.precision_combo)
        selectors.addSpacing(12)
        selectors.addWidget(QLabel("Show:"))
        selectors.addWidget(self.filter_combo)
        selectors.addStretch(1)
        selectors.addWidget(self.count_label)
        label_row = QHBoxLayout()
        for button in self.label_buttons.values():
            label_row.addWidget(button)
        label_row.addWidget(self.clear_button)
        label_row.addStretch(1)
        import_row = QHBoxLayout()
        import_row.addWidget(self.import_button)
        import_row.addWidget(self.import_progress, 1)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addLayout(selectors)
        left_layout.addWidget(self.table, 1)
        left_layout.addLayout(label_row)
        left_layout.addLayout(import_row)
        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.preview)
        right.addWidget(self.metrics_panel)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        note = QLabel(NOTE_TEXT)
        note.setStyleSheet("QLabel { font-style: italic; }")
        layout = QVBoxLayout(self)
        layout.addWidget(note)
        layout.addWidget(splitter, 1)

    @property
    def precision(self) -> str:
        return self.precision_combo.currentText()

    def set_busy(self, kind: str | None) -> None:
        self.import_button.setEnabled(kind is None)
        self.import_progress.setVisible(kind == LABEL_IMPORT_JOB)

    def on_shown(self) -> None:
        self.refresh_metrics()

    @Slot()
    def refresh(self) -> None:
        rows = self.ctx.db.query_results(precision=self.precision, status="ok", order_by="file_name",
                                         descending=False)
        labels = {file_hash: row["label"] for file_hash, row in self.ctx.db.get_labels().items()}
        self.model.set_data(rows, labels)
        self._after_reset()
        self.refresh_metrics()

    @Slot()
    def _on_filter_changed(self) -> None:
        self.model.set_filter(self.filter_combo.currentText())
        self._after_reset()

    def _after_reset(self) -> None:
        self.count_label.setText(f"{self.model.rowCount()} images")
        if self.model.rowCount():
            self.select_row(0)
        else:
            self.preview.show_message("No image selected")

    def refresh_metrics(self) -> None:
        pairs = self.ctx.db.labeled_pairs(self.precision)
        self.metrics_panel.show_metrics(compute_metrics((p["true_label"], p["predicted_label"]) for p in pairs))

    def select_row(self, row: int) -> None:
        index = self.model.index(row, 0)
        self.table.selectionModel().setCurrentIndex(
            index, QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows)
        self.table.scrollTo(index)

    def current_row(self) -> int:
        index = self.table.currentIndex()
        return index.row() if index.isValid() else -1

    @Slot(QModelIndex, QModelIndex)
    def _on_current_row_changed(self, current: QModelIndex, _previous: QModelIndex) -> None:
        if not current.isValid():
            self.preview.show_message("No image selected")
            return
        path = from_portable_path(self.model.row_at(current.row())["path"])
        if not path.exists():
            self.preview.show_message(f"File not found:\n{path}")
            return
        self.preview.show_image(path)

    def label_current(self, label: str | None) -> None:
        row_index = self.current_row()
        if row_index < 0:
            return
        row = self.model.row_at(row_index)
        file_hash = row.get("file_hash")
        if not file_hash:
            self.ctx.dialogs.warning("Label", "This result has no file hash and cannot be labeled.")
            return
        if label is None:
            self.ctx.db.remove_label(file_hash)
        elif label in LABELS:
            self.ctx.db.set_label(file_hash, label, row.get("path"), "manual")
        else:
            return
        self.model.set_ground_truth(file_hash, label)
        self.refresh_metrics()
        if row_index + 1 < self.model.rowCount():
            self.select_row(row_index + 1)

    @Slot()
    def import_folder_labels(self) -> None:
        root = self.ctx.dialogs.choose_directory("Choose the folder with safe / suggestive / explicit subfolders",
                                                 INPUT_DIR)
        if root is None:
            return
        self.import_progress.setRange(0, 0)
        if not self.ctx.jobs.start(LabelImportJob(self.ctx.db_path, root), on_progress=self._on_import_progress,
                                   on_success=self._on_import_finished, on_failure=self._on_import_failed):
            return
        self.ctx.status(f"Importing folder labels from {root}...")

    def _on_import_progress(self, update: object) -> None:
        if isinstance(update, CountUpdate):
            self.import_progress.setRange(0, max(1, update.total))
            self.import_progress.setValue(update.done)

    def _on_import_finished(self, imported: object) -> None:
        self.refresh()
        message = f"Imported {imported} folder label(s). Manual labels were kept."
        self.ctx.status(message)
        self.ctx.dialogs.information("Import labels", message)

    def _on_import_failed(self, _error: BaseException) -> None:
        self.refresh()

    def export_metrics(self, fmt: str) -> None:
        default = OUTPUT_DIR / f"metrics_{self.precision}.{fmt}"
        file_filter = "CSV files (*.csv)" if fmt == "csv" else "JSON files (*.json)"
        target = self.ctx.dialogs.choose_save_path(f"Export metrics ({fmt.upper()})", default, file_filter)
        if target is None:
            return
        try:
            export_metrics(self.metrics_panel.metrics, target, fmt)
        except OSError as error:
            self.ctx.dialogs.error("Export failed", f"Cannot write {target}:\n{error}")
            return
        self.ctx.status(f"Metrics exported to {target}")

    def shutdown(self) -> None:
        self.preview.shutdown()
