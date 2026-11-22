from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from PySide6.QtCore import QModelIndex, Qt, QTimer, Slot
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                               QPushButton, QTableView, QVBoxLayout, QWidget)

from database import export_results_csv
from settings import RESULTS_CSV_PATH, from_portable_path
from ui.context import AppContext
from ui.results_model import ResultsProxyModel, ResultsTableModel

logger = logging.getLogger("blueeye.app")

CATEGORY_ALL = "All"
CATEGORY_ERRORS = "errors"
CATEGORIES = (CATEGORY_ALL, "safe", "suggestive", "explicit", CATEGORY_ERRORS)
PRECISION_FILTERS = ("All", "fp32", "bf16")
TEXT_FILTER_DELAY_MILLISECONDS = 250


class ResultsTab(QWidget):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None):
        super().__init__(parent)
        self.ctx = ctx
        self.model = ResultsTableModel(self)
        self.proxy = ResultsProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setWordWrap(False)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(22)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.category_combo = QComboBox()
        self.category_combo.addItems(CATEGORIES)
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(PRECISION_FILTERS)
        self.text_filter = QLineEdit()
        self.text_filter.setPlaceholderText("Filter by file name")
        self.text_filter.setClearButtonEnabled(True)
        self.refresh_button = QPushButton("Refresh")
        self.export_button = QPushButton("Export CSV...")
        self.open_file_button = QPushButton("Open file")
        self.open_folder_button = QPushButton("Open containing folder")
        self.count_label = QLabel()
        self._text_timer = QTimer(self)
        self._text_timer.setSingleShot(True)
        self._text_timer.setInterval(TEXT_FILTER_DELAY_MILLISECONDS)
        self._text_timer.timeout.connect(self._apply_text_filter)
        self._build_layout()
        self.category_combo.currentIndexChanged.connect(self.reload)
        self.precision_combo.currentIndexChanged.connect(self.reload)
        self.text_filter.textChanged.connect(self._text_timer.start)
        self.refresh_button.clicked.connect(self.reload)
        self.export_button.clicked.connect(self.export_csv)
        self.open_file_button.clicked.connect(self.open_selected_file)
        self.open_folder_button.clicked.connect(self.open_selected_folder)
        self.table.doubleClicked.connect(self._on_double_clicked)
        ctx.events.results_changed.connect(self.reload)
        self.reload()
        self.table.resizeColumnsToContents()

    def _build_layout(self) -> None:
        filters = QHBoxLayout()
        filters.addWidget(QLabel("Category:"))
        filters.addWidget(self.category_combo)
        filters.addSpacing(12)
        filters.addWidget(QLabel("Precision:"))
        filters.addWidget(self.precision_combo)
        filters.addSpacing(12)
        filters.addWidget(self.text_filter, 1)
        filters.addWidget(self.refresh_button)
        actions = QHBoxLayout()
        actions.addWidget(self.export_button)
        actions.addWidget(self.open_file_button)
        actions.addWidget(self.open_folder_button)
        actions.addStretch(1)
        actions.addWidget(self.count_label)
        layout = QVBoxLayout(self)
        layout.addLayout(filters)
        layout.addWidget(self.table, 1)
        layout.addLayout(actions)

    def set_busy(self, kind: str | None) -> None:
        pass

    def query_filters(self) -> dict:
        category = self.category_combo.currentText()
        precision = self.precision_combo.currentText()
        filters: dict = {"precision": None if precision == "All" else precision}
        if category == CATEGORY_ERRORS:
            filters["status"] = "error"
        elif category != CATEGORY_ALL:
            filters["label"] = category
            filters["status"] = "ok"
        return filters

    @Slot()
    def reload(self) -> None:
        rows = self.ctx.db.query_results(**self.query_filters())
        self.model.set_rows(rows)
        self._update_count()

    @Slot()
    def _apply_text_filter(self) -> None:
        self.proxy.setFilterFixedString(self.text_filter.text().strip())
        self._update_count()

    def _update_count(self) -> None:
        total = self.model.rowCount()
        shown = self.proxy.rowCount()
        self.count_label.setText(f"{total} results" if shown == total else f"{shown} of {total} results shown")

    def selected_row(self) -> dict | None:
        indexes = self.table.selectionModel().selectedRows()
        if not indexes:
            current = self.table.currentIndex()
            if not current.isValid():
                return None
            indexes = [current]
        source = self.proxy.mapToSource(indexes[0])
        return self.model.row_at(source.row()) if source.isValid() else None

    def _selected_existing_path(self) -> Path | None:
        row = self.selected_row()
        if row is None:
            self.ctx.dialogs.information("Results", "Select a result row first.")
            return None
        path = from_portable_path(row["path"])
        if not path.exists():
            self.ctx.dialogs.warning("Results", f"File no longer exists:\n{path}")
            return None
        return path

    @Slot()
    def open_selected_file(self) -> None:
        path = self._selected_existing_path()
        if path is None:
            return
        try:
            os.startfile(str(path))
        except OSError as error:
            self.ctx.dialogs.error("Open file", f"Cannot open {path}:\n{error}")

    @Slot()
    def open_selected_folder(self) -> None:
        path = self._selected_existing_path()
        if path is None:
            return
        try:
            subprocess.Popen(f'explorer /select,"{path}"')
        except OSError as error:
            self.ctx.dialogs.error("Open folder", f"Cannot open the folder of {path}:\n{error}")

    @Slot(QModelIndex)
    def _on_double_clicked(self, _index: QModelIndex) -> None:
        self.open_selected_file()

    @Slot()
    def export_csv(self) -> None:
        target = self.ctx.dialogs.choose_save_path("Export results", RESULTS_CSV_PATH, "CSV files (*.csv)")
        if target is None:
            return
        try:
            count = export_results_csv(self.ctx.db, target, **self.query_filters())
        except OSError as error:
            self.ctx.dialogs.error("Export failed", f"Cannot write {target}:\n{error}")
            return
        message = f"Exported {count} rows to {target}"
        if self.text_filter.text().strip():
            message += "\n\nThe file name filter is not applied to the export; category and precision filters are."
        self.ctx.status(f"Exported {count} rows to {target}")
        self.ctx.dialogs.information("Export CSV", message)
