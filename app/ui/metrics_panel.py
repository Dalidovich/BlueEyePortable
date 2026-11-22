from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QFormLayout, QHBoxLayout, QHeaderView, QLabel, QPushButton, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from evaluation import LABELS, EvaluationMetrics, compute_metrics
from ui.widgets import fill_static_table, make_static_table

CLASS_HEADERS = ["Class", "Precision", "Recall", "F1", "Support", "Per-class accuracy"]
METRIC_DIGITS = 4


def format_metric(value: float) -> str:
    return f"{value:.{METRIC_DIGITS}f}"


def fit_height_to_rows(table: QTableWidget) -> None:
    table.resizeRowsToContents()
    rows_height = sum(table.rowHeight(row) for row in range(table.rowCount()))
    header_height = table.horizontalHeader().sizeHint().height()
    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    table.setFixedHeight(header_height + rows_height + 2 * table.frameWidth())


class MetricsPanel(QWidget):
    def __init__(self, export: Callable[[str], None], parent: QWidget | None = None):
        super().__init__(parent)
        self.metrics: EvaluationMetrics = compute_metrics([])
        self.labeled_label = QLabel()
        self.accuracy_label = QLabel()
        self.macro_label = QLabel()
        self.weighted_label = QLabel()
        self.class_table = make_static_table(CLASS_HEADERS)
        self.confusion_table = QTableWidget(len(LABELS), len(LABELS))
        self.confusion_table.setHorizontalHeaderLabels([f"Predicted {label}" for label in LABELS])
        self.confusion_table.setVerticalHeaderLabels([f"True {label}" for label in LABELS])
        self.confusion_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.confusion_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.export_csv_button = QPushButton("Export metrics CSV...")
        self.export_json_button = QPushButton("Export metrics JSON...")
        self.export_csv_button.clicked.connect(lambda: export("csv"))
        self.export_json_button.clicked.connect(lambda: export("json"))
        summary = QFormLayout()
        summary.addRow("Labeled images:", self.labeled_label)
        summary.addRow("Accuracy:", self.accuracy_label)
        summary.addRow("Macro precision / recall / F1:", self.macro_label)
        summary.addRow("Weighted F1:", self.weighted_label)
        buttons = QHBoxLayout()
        buttons.addWidget(self.export_csv_button)
        buttons.addWidget(self.export_json_button)
        buttons.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(summary)
        layout.addWidget(self.class_table)
        layout.addWidget(QLabel("Confusion matrix (rows = ground truth, columns = predicted):"))
        layout.addWidget(self.confusion_table)
        layout.addLayout(buttons)
        layout.addStretch(1)
        self.class_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.show_metrics(self.metrics)

    def show_metrics(self, metrics: EvaluationMetrics) -> None:
        self.metrics = metrics
        self.labeled_label.setText(str(metrics.total))
        self.accuracy_label.setText(f"{format_metric(metrics.accuracy)} ({metrics.correct} / {metrics.total})")
        self.macro_label.setText(" / ".join(format_metric(value) for value in
                                            (metrics.macro_precision, metrics.macro_recall, metrics.macro_f1)))
        self.weighted_label.setText(format_metric(metrics.weighted_f1))
        rows = [[c.label, format_metric(c.precision), format_metric(c.recall), format_metric(c.f1), str(c.support),
                 format_metric(c.accuracy)] for c in metrics.per_class]
        fill_static_table(self.class_table, rows, numeric_columns={1, 2, 3, 4, 5})
        for row_index, row in enumerate(metrics.confusion):
            for column_index, count in enumerate(row):
                item = QTableWidgetItem(str(count))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.confusion_table.setItem(row_index, column_index, item)
        fit_height_to_rows(self.class_table)
        fit_height_to_rows(self.confusion_table)
        has_data = metrics.total > 0
        self.export_csv_button.setEnabled(has_data)
        self.export_json_button.setEnabled(has_data)
