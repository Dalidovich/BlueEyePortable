from __future__ import annotations

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPersistentModelIndex, Qt

from ui.formatting import format_float

FILTER_ALL = "All"
FILTER_UNLABELED = "Unlabeled"
FILTER_LABELED = "Labeled"
FILTER_MISMATCHES = "Mismatches"
FILTERS = (FILTER_ALL, FILTER_UNLABELED, FILTER_LABELED, FILTER_MISMATCHES)
HEADERS = ("File name", "Predicted", "Confidence", "Ground truth")
GROUND_TRUTH_COLUMN = 3
CONFIDENCE_DIGITS = 4

ModelIndex = QModelIndex | QPersistentModelIndex


class EvaluationTableModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._all_rows: list[dict] = []
        self._rows: list[dict] = []
        self._labels: dict[str, str] = {}
        self._filter = FILTER_ALL

    def set_data(self, rows: list[dict], labels: dict[str, str]) -> None:
        self._all_rows = list(rows)
        self._labels = dict(labels)
        self._refilter()

    def set_filter(self, mode: str) -> None:
        self._filter = mode
        self._refilter()

    def ground_truth(self, row: dict) -> str | None:
        file_hash = row.get("file_hash")
        return self._labels.get(file_hash) if file_hash else None

    def set_ground_truth(self, file_hash: str, label: str | None) -> None:
        if label is None:
            self._labels.pop(file_hash, None)
        else:
            self._labels[file_hash] = label
        if self._rows:
            self.dataChanged.emit(self.index(0, GROUND_TRUTH_COLUMN),
                                  self.index(len(self._rows) - 1, GROUND_TRUTH_COLUMN))

    def row_at(self, row: int) -> dict:
        return self._rows[row]

    def rowCount(self, parent: ModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: ModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(HEADERS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return HEADERS[section]
        return None

    def data(self, index: ModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        column = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            if column == 0:
                return row.get("file_name") or ""
            if column == 1:
                return row.get("label") or ""
            if column == 2:
                return format_float(row.get("confidence"), CONFIDENCE_DIGITS)
            return self.ground_truth(row) or ""
        if role == Qt.ItemDataRole.ToolTipRole and column == 0:
            return row.get("path")
        if role == Qt.ItemDataRole.TextAlignmentRole and column == 2:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return None

    def _accepts(self, row: dict) -> bool:
        truth = self.ground_truth(row)
        if self._filter == FILTER_UNLABELED:
            return truth is None
        if self._filter == FILTER_LABELED:
            return truth is not None
        if self._filter == FILTER_MISMATCHES:
            return truth is not None and truth != row.get("label")
        return True

    def _refilter(self) -> None:
        self.beginResetModel()
        self._rows = [row for row in self._all_rows if self._accepts(row)]
        self.endResetModel()
