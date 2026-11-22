from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QProgressBar,
                               QPushButton, QSpinBox, QVBoxLayout, QWidget)

from benchmark import BenchmarkResult, export_benchmarks, parse_batch_sizes
from settings import OUTPUT_DIR, from_portable_path
from ui.context import AppContext
from ui.device_state import BF16
from ui.formatting import format_mib
from ui.widgets import DeviceSelector, FolderField, fill_static_table, make_static_table
from ui.workers import BENCHMARK_JOB, BenchmarkJob, BenchmarkUpdate, StatusUpdate

RESULT_HEADERS = ["GPU", "Device", "Precision", "Batch", "Effective batch", "Images", "Total s", "Images/s",
                  "ms/image", "Model images/s", "Model ms/image", "Peak GPU memory (MiB)"]
NUMERIC_COLUMNS = {3, 4, 5, 6, 7, 8, 9, 10, 11}
MIN_IMAGE_LIMIT = 1
MAX_IMAGE_LIMIT = 100000
NOTE_TEXT = "Speed only. Classification quality is measured in the Evaluation tab."


def benchmark_row_cells(row: dict) -> list[str]:
    return [row.get("gpu_name") or "CPU", row.get("device") or "", row.get("precision") or "",
            str(row.get("batch_size")), str(row.get("effective_batch_size")), str(row.get("images")),
            f"{row.get('total_seconds') or 0.0:.2f}", f"{row.get('images_per_second') or 0.0:.2f}",
            f"{row.get('ms_per_image') or 0.0:.1f}", f"{row.get('model_images_per_second') or 0.0:.2f}",
            f"{row.get('model_ms_per_image') or 0.0:.1f}", format_mib(row.get("peak_memory_bytes"))]


def group_sessions(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    sessions: dict[str, list[dict]] = {}
    for row in rows:
        sessions.setdefault(row["session_id"], []).append(row)
    return [(session_id, sorted(items, key=lambda item: item["id"])) for session_id, items in sessions.items()]


def session_title(rows: list[dict]) -> str:
    first = rows[0]
    target = first.get("gpu_name") or first.get("device") or "CPU"
    sizes = ",".join(str(row.get("batch_size")) for row in rows)
    return f"{first.get('created_at')}  {target}  {first.get('precision')}  batches {sizes}"


class BenchmarkTab(QWidget):
    def __init__(self, ctx: AppContext, default_folder: Callable[[], str], parent: QWidget | None = None):
        super().__init__(parent)
        self.ctx = ctx
        self._default_folder = default_folder
        settings = ctx.settings
        self.folder_field = FolderField("Select the benchmark folder",
                                        lambda title, start: self.ctx.dialogs.choose_directory(title, start))
        self.folder_field.edit.setText(default_folder())
        self.folder_field.edit.setPlaceholderText("Same as the Scan folder")
        self.batch_sizes_edit = QLineEdit(settings.benchmark_batch_sizes)
        self.batch_sizes_edit.setToolTip("Comma-separated positive batch sizes, for example 1,2,4,8")
        self.limit_spin = QSpinBox()
        self.limit_spin.setRange(MIN_IMAGE_LIMIT, MAX_IMAGE_LIMIT)
        self.limit_spin.setValue(min(MAX_IMAGE_LIMIT, max(MIN_IMAGE_LIMIT, settings.benchmark_image_limit)))
        self.device_selector = DeviceSelector(ctx.device_state)
        self.run_button = QPushButton("Run")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.progress_label = QLabel("Idle")
        self.progress_bar = QProgressBar()
        self.results_table = make_static_table(RESULT_HEADERS)
        self.session_combo = QComboBox()
        self.session_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.export_csv_button = QPushButton("Export CSV...")
        self.export_json_button = QPushButton("Export JSON...")
        self.displayed_rows: list[dict] = []
        self._build_layout()
        self.run_button.clicked.connect(self.run_benchmark)
        self.stop_button.clicked.connect(self.stop_benchmark)
        self.session_combo.currentIndexChanged.connect(self._on_session_selected)
        self.export_csv_button.clicked.connect(lambda: self.export_session("csv"))
        self.export_json_button.clicked.connect(lambda: self.export_session("json"))
        self.reload_history()

    def _build_layout(self) -> None:
        form = QFormLayout()
        form.addRow("Folder:", self.folder_field)
        form.addRow("Batch sizes:", self.batch_sizes_edit)
        form.addRow("Image limit:", self.limit_spin)
        form.addRow("Inference:", self.device_selector)
        settings_group = QGroupBox("Benchmark settings")
        settings_group.setLayout(form)
        buttons = QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.stop_button)
        buttons.addStretch(1)
        note = QLabel(NOTE_TEXT)
        note.setStyleSheet("QLabel { font-style: italic; }")
        history = QHBoxLayout()
        history.addWidget(QLabel("Session:"))
        history.addWidget(self.session_combo, 1)
        history.addWidget(self.export_csv_button)
        history.addWidget(self.export_json_button)
        layout = QVBoxLayout(self)
        layout.addWidget(settings_group)
        layout.addWidget(note)
        layout.addLayout(buttons)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)
        layout.addLayout(history)
        layout.addWidget(self.results_table, 1)

    def store_settings(self) -> None:
        settings = self.ctx.settings
        try:
            parse_batch_sizes(self.batch_sizes_edit.text())
        except ValueError:
            pass
        else:
            settings.benchmark_batch_sizes = self.batch_sizes_edit.text().strip()
        settings.benchmark_image_limit = self.limit_spin.value()

    def set_busy(self, kind: str | None) -> None:
        idle = kind is None
        for widget in (self.run_button, self.folder_field, self.batch_sizes_edit, self.limit_spin,
                       self.device_selector):
            widget.setEnabled(idle)
        self.stop_button.setEnabled(kind == BENCHMARK_JOB)

    def _resolve_folder(self) -> Path | None:
        text = self.folder_field.text() or self._default_folder().strip()
        return from_portable_path(text) if text else None

    @Slot()
    def run_benchmark(self) -> None:
        folder = self._resolve_folder()
        if folder is None or not folder.is_dir():
            self.ctx.dialogs.warning("Benchmark", "Choose an existing folder with images first.")
            return
        try:
            batch_sizes = parse_batch_sizes(self.batch_sizes_edit.text())
        except ValueError as error:
            self.ctx.dialogs.warning("Benchmark", f"Invalid batch sizes: {error}")
            return
        state = self.ctx.device_state
        if state.precision == BF16 and not state.bf16_supported():
            self.ctx.dialogs.warning("Benchmark", f"{state.bf16_unsupported_reason()}. Select fp32.")
            return
        self.store_settings()
        self.ctx.save_settings()
        device = state.resolved_device()
        job = BenchmarkJob(self.ctx.engine_manager, self.ctx.db_path, folder, batch_sizes, self.limit_spin.value(),
                           device, state.label_for(device), state.gpu_name_for(device), state.precision)
        job.retry = self.run_benchmark
        if not self.ctx.jobs.start(job, on_progress=self._on_progress, on_success=self._on_finished,
                                   on_failure=self._on_failed):
            return
        self.progress_label.setText(f"Starting benchmark on {job.device_label} ({job.precision})...")
        self.progress_bar.setRange(0, 0)

    @Slot()
    def stop_benchmark(self) -> None:
        if self.ctx.jobs.current_kind != BENCHMARK_JOB:
            return
        self.ctx.jobs.request_stop()
        self.stop_button.setEnabled(False)
        self.progress_label.setText("Stopping... finishing the current batch")

    def _on_progress(self, update: object) -> None:
        if isinstance(update, StatusUpdate):
            self.progress_label.setText(update.message)
            self.progress_bar.setRange(0, 0)
        elif isinstance(update, BenchmarkUpdate):
            self.progress_label.setText(f"{update.stage.capitalize()}: {update.done} / {update.total} images")
            self.progress_bar.setRange(0, max(1, update.total))
            self.progress_bar.setValue(min(update.done, update.total))

    def _on_finished(self, results: object) -> None:
        self.progress_bar.setRange(0, 1)
        if not isinstance(results, list) or not results:
            self.progress_bar.setValue(0)
            self.progress_label.setText("Benchmark stopped before any batch size finished.")
            self.ctx.status("Benchmark stopped; no results.")
            return
        self.progress_bar.setValue(1)
        first: BenchmarkResult = results[0]
        self.progress_label.setText(f"Benchmark finished: {len(results)} batch size(s) measured.")
        self.ctx.status(f"Benchmark finished on {first.device} ({first.precision}).")
        self.reload_history(select_session=first.session_id)
        if not self.displayed_rows:
            self._show_rows([asdict(result) for result in results])

    def _on_failed(self, error: BaseException) -> None:
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"Benchmark failed: {type(error).__name__}: {error}")
        self.reload_history()

    def reload_history(self, select_session: str | None = None) -> None:
        sessions = group_sessions(self.ctx.db.list_benchmarks())
        current = select_session or self.session_combo.currentData()
        self.session_combo.blockSignals(True)
        try:
            self.session_combo.clear()
            for session_id, rows in sessions:
                self.session_combo.addItem(session_title(rows), session_id)
            index = self.session_combo.findData(current) if current else 0
            self.session_combo.setCurrentIndex(max(0, index) if self.session_combo.count() else -1)
        finally:
            self.session_combo.blockSignals(False)
        self._on_session_selected(self.session_combo.currentIndex())

    @Slot(int)
    def _on_session_selected(self, index: int) -> None:
        session_id = self.session_combo.itemData(index) if index >= 0 else None
        rows = self.ctx.db.list_benchmarks(session_id) if session_id else []
        self._show_rows(rows)

    def _show_rows(self, rows: list[dict]) -> None:
        self.displayed_rows = rows
        fill_static_table(self.results_table, [benchmark_row_cells(row) for row in rows], NUMERIC_COLUMNS)
        has_rows = bool(rows)
        self.export_csv_button.setEnabled(has_rows)
        self.export_json_button.setEnabled(has_rows)

    def export_session(self, fmt: str) -> None:
        if not self.displayed_rows:
            return
        session = str(self.displayed_rows[0].get("session_id", "session"))[:8]
        default = OUTPUT_DIR / f"benchmark_{session}.{fmt}"
        file_filter = "CSV files (*.csv)" if fmt == "csv" else "JSON files (*.json)"
        target = self.ctx.dialogs.choose_save_path(f"Export benchmark ({fmt.upper()})", default, file_filter)
        if target is None:
            return
        try:
            export_benchmarks(self.displayed_rows, target, fmt)
        except OSError as error:
            self.ctx.dialogs.error("Export failed", f"Cannot write {target}:\n{error}")
            return
        self.ctx.status(f"Exported {len(self.displayed_rows)} benchmark rows to {target}")
