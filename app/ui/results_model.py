from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import (QAbstractTableModel, QModelIndex, QPersistentModelIndex, QSortFilterProxyModel, Qt)

from ui.formatting import format_float, yes_no

SORT_ROLE = Qt.ItemDataRole.UserRole + 1
PROBABILITY_DIGITS = 4
SECONDS_DIGITS = 3

ModelIndex = QModelIndex | QPersistentModelIndex


def text_key(name: str) -> Callable[[dict], object]:
    return lambda row: (row.get(name) or "").lower()


def number_key(name: str) -> Callable[[dict], object]:
    return lambda row: row.get(name)


def number_text(name: str, digits: int) -> Callable[[dict], str]:
    return lambda row: format_float(row.get(name), digits)


def device_text(row: dict) -> str:
    gpu_name, device = row.get("gpu_name"), row.get("device") or ""
    return f"{gpu_name} ({device})" if gpu_name else device


@dataclass(frozen=True)
class ResultColumn:
    title: str
    display: Callable[[dict], str]
    sort_key: Callable[[dict], object]
    numeric: bool = False


RESULT_COLUMNS: tuple[ResultColumn, ...] = (
    ResultColumn("File name", lambda row: row.get("file_name") or "", text_key("file_name")),
    ResultColumn("Predicted class", lambda row: row.get("label") or "", text_key("label")),
    ResultColumn("Confidence", number_text("confidence", PROBABILITY_DIGITS), number_key("confidence"), True),
    ResultColumn("Safe", number_text("p_safe", PROBABILITY_DIGITS), number_key("p_safe"), True),
    ResultColumn("Suggestive", number_text("p_suggestive", PROBABILITY_DIGITS), number_key("p_suggestive"), True),
    ResultColumn("Explicit", number_text("p_explicit", PROBABILITY_DIGITS), number_key("p_explicit"), True),
    ResultColumn("Time (s)", number_text("seconds", SECONDS_DIGITS), number_key("seconds"), True),
    ResultColumn("Status", lambda row: row.get("status") or "", text_key("status")),
    ResultColumn("Error", lambda row: row.get("error") or "", text_key("error")),
    ResultColumn("Precision", lambda row: row.get("precision") or "", text_key("precision")),
    ResultColumn("Device/GPU", device_text, lambda row: device_text(row).lower()),
    ResultColumn("Reused", lambda row: yes_no(bool(row.get("reused"))), number_key("reused")),
    ResultColumn("Path", lambda row: row.get("path") or "", text_key("path")),
)
TOOLTIP_COLUMNS = frozenset({"Error", "Path", "File name"})


def null_safe(key: Callable[[dict], object]) -> Callable[[dict], tuple[bool, object]]:
    def wrapped(row: dict) -> tuple[bool, object]:
        value = key(row)
        return (True, 0) if value is None else (False, value)
    return wrapped


class ResultsTableModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[dict] = []
        self._sort_column = -1
        self._sort_order = Qt.SortOrder.AscendingOrder

    def set_rows(self, rows: list[dict]) -> None:
        self.beginResetModel()
        self._rows = list(rows)
        self._apply_sort()
        self.endResetModel()

    def row_at(self, row: int) -> dict:
        return self._rows[row]

    def rows(self) -> list[dict]:
        return list(self._rows)

    def rowCount(self, parent: ModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: ModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(RESULT_COLUMNS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return RESULT_COLUMNS[section].title
        return None

    def data(self, index: ModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        column = RESULT_COLUMNS[index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            return column.display(row)
        if role == SORT_ROLE:
            return column.sort_key(row)
        if role == Qt.ItemDataRole.ToolTipRole and column.title in TOOLTIP_COLUMNS:
            return column.display(row) or None
        if role == Qt.ItemDataRole.TextAlignmentRole and column.numeric:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    def sort(self, column: int, order: Qt.SortOrder = Qt.SortOrder.AscendingOrder) -> None:
        self.beginResetModel()
        self._sort_column = column
        self._sort_order = order
        self._apply_sort()
        self.endResetModel()

    def _apply_sort(self) -> None:
        if not 0 <= self._sort_column < len(RESULT_COLUMNS):
            return
        key = null_safe(RESULT_COLUMNS[self._sort_column].sort_key)
        self._rows.sort(key=key, reverse=self._sort_order == Qt.SortOrder.DescendingOrder)


class ResultsProxyModel(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSortRole(SORT_ROLE)
        self.setFilterKeyColumn(0)
        self.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

    def sort(self, column: int, order: Qt.SortOrder = Qt.SortOrder.AscendingOrder) -> None:
        source = self.sourceModel()
        if column >= 0 and source is not None:
            source.sort(column, order)
