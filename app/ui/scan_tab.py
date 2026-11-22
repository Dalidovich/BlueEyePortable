from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import QElapsedTimer, QTimer, Slot
from PySide6.QtWidgets import (QCheckBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QListWidget, QProgressBar, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from scanner import ScanProgress, ScanSummary
from settings import to_portable_path
from ui.context import AppContext
from ui.device_state import BF16
from ui.formatting import format_duration
from ui.widgets import DeviceSelector, ElidedLabel, FolderField
from ui.workers import SCAN_JOB, ScanJob, ScanUpdate, StatusUpdate

MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 64
MAX_LOG_ENTRIES = 1000
MAX_LOGGED_FILE_ERRORS = 200
ELAPSED_REFRESH_MILLISECONDS = 500
PHASE_TITLES = {"discovering": "Discovering files", "checking": "Checking files",
                "inference": "Classifying", "done": "Done"}
COUNTER_KEYS = ("processed", "reused", "skipped", "errors")


class ScanTab(QWidget):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None):
        super().__init__(parent)
        self.ctx = ctx
        settings = ctx.settings
        self.folder_field = FolderField("Select the input folder",
                                        lambda title, start: self.ctx.dialogs.choose_directory(title, start))
        self.folder_field.set_path(settings.input_path)
        self.recursive_check = QCheckBox("Include subfolders")
        self.recursive_check.setChecked(settings.recursive)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(MIN_BATCH_SIZE, MAX_BATCH_SIZE)
        self.batch_spin.setValue(min(MAX_BATCH_SIZE, max(MIN_BATCH_SIZE, settings.batch_size)))
        self.device_selector = DeviceSelector(ctx.device_state)
        self.force_check = QCheckBox("Reprocess files that already have results")
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.progress_bar = QProgressBar()
        self.phase_label = QLabel("Idle")
        self.current_file_label = ElidedLabel()
        self.counter_labels = {key: QLabel("0") for key in COUNTER_KEYS}
        self.rate_label = QLabel("-")
        self.elapsed_label = QLabel(format_duration(0))
        self.log_list = QListWidget()
        self.log_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self._elapsed = QElapsedTimer()
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(ELAPSED_REFRESH_MILLISECONDS)
        self._elapsed_timer.timeout.connect(self._refresh_elapsed)
        self._last_effective_batch: int | None = None
        self._build_layout()
        self.start_button.clicked.connect(self.start_scan)
        self.stop_button.clicked.connect(self.stop_scan)
        self.stop_button.setEnabled(False)

    def _build_layout(self) -> None:
        options = QFormLayout()
        options.addRow("Input folder:", self.folder_field)
        batch_row = QHBoxLayout()
        batch_row.addWidget(self.batch_spin)
        batch_row.addSpacing(16)
        batch_row.addWidget(self.recursive_check)
        batch_row.addSpacing(16)
        batch_row.addWidget(self.force_check)
        batch_row.addStretch(1)
        options.addRow("Batch size:", batch_row)
        options.addRow("Inference:", self.device_selector)
        settings_group = QGroupBox("Scan settings")
        settings_group.setLayout(options)

        buttons = QHBoxLayout()
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addStretch(1)

        stats = QGridLayout()
        stats.addWidget(QLabel("Phase:"), 0, 0)
        stats.addWidget(self.phase_label, 0, 1, 1, 7)
        stats.addWidget(QLabel("Current file:"), 1, 0)
        stats.addWidget(self.current_file_label, 1, 1, 1, 7)
        for column, key in enumerate(COUNTER_KEYS):
            stats.addWidget(QLabel(f"{key.capitalize()}:"), 2, column * 2)
            stats.addWidget(self.counter_labels[key], 2, column * 2 + 1)
        stats.addWidget(QLabel("Images per second:"), 3, 0)
        stats.addWidget(self.rate_label, 3, 1)
        stats.addWidget(QLabel("Elapsed:"), 3, 2)
        stats.addWidget(self.elapsed_label, 3, 3)
        stats.setColumnStretch(7, 1)
        progress_group = QGroupBox("Progress")
        progress_box = QVBoxLayout(progress_group)
        progress_box.addWidget(self.progress_bar)
        progress_box.addLayout(stats)

        layout = QVBoxLayout(self)
        layout.addWidget(settings_group)
        layout.addLayout(buttons)
        layout.addWidget(progress_group)
        layout.addWidget(QLabel("Log:"))
        layout.addWidget(self.log_list, 1)

    def folder_text(self) -> str:
        return self.folder_field.text()

    def store_settings(self) -> None:
        settings = self.ctx.settings
        folder = self.folder_field.path()
        if folder is not None:
            settings.input_folder = to_portable_path(folder)
        settings.recursive = self.recursive_check.isChecked()
        settings.batch_size = self.batch_spin.value()

    def set_busy(self, kind: str | None) -> None:
        idle = kind is None
        for widget in (self.start_button, self.folder_field, self.recursive_check, self.batch_spin,
                       self.device_selector, self.force_check):
            widget.setEnabled(idle)
        self.stop_button.setEnabled(kind == SCAN_JOB)

    def log(self, message: str) -> None:
        self.log_list.addItem(f"{datetime.now():%H:%M:%S}  {message}")
        while self.log_list.count() > MAX_LOG_ENTRIES:
            self.log_list.takeItem(0)
        self.log_list.scrollToBottom()

    @Slot()
    def start_scan(self) -> None:
        folder = self.folder_field.path()
        if folder is None or not folder.is_dir():
            self.ctx.dialogs.warning("Scan", "Choose an existing input folder first.")
            return
        state = self.ctx.device_state
        if state.precision == BF16 and not state.bf16_supported():
            self.ctx.dialogs.warning("Scan", f"{state.bf16_unsupported_reason()}. Select fp32.")
            return
        self.store_settings()
        self.ctx.save_settings()
        device = state.resolved_device()
        job = ScanJob(self.ctx.engine_manager, self.ctx.db_path, folder, self.recursive_check.isChecked(),
                      self.batch_spin.value(), self.force_check.isChecked(), device, state.label_for(device),
                      state.gpu_name_for(device), state.precision)
        job.retry = self.start_scan
        if not self.ctx.jobs.start(job, on_progress=self._on_progress, on_success=self._on_finished,
                                   on_failure=self._on_failed):
            return
        self._reset_progress()
        self.log(f"Scan started: {folder} (recursive={job.recursive}, batch={job.batch_size}, "
                 f"force={job.force}) on {job.device_label}, {job.precision}")

    @Slot()
    def stop_scan(self) -> None:
        if self.ctx.jobs.current_kind != SCAN_JOB:
            return
        self.ctx.jobs.request_stop()
        self.stop_button.setEnabled(False)
        self.phase_label.setText("Stopping... finishing the current batch")
        self.log("Stop requested; finishing the current batch...")

    def _reset_progress(self) -> None:
        self._last_effective_batch = None
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)
        self.phase_label.setText("Starting...")
        self.current_file_label.set_full_text("")
        for label in self.counter_labels.values():
            label.setText("0")
        self.rate_label.setText("-")
        self._elapsed.start()
        self._refresh_elapsed()
        self._elapsed_timer.start()

    @Slot()
    def _refresh_elapsed(self) -> None:
        if self._elapsed.isValid():
            self.elapsed_label.setText(format_duration(self._elapsed.elapsed() / 1000.0))

    def _on_progress(self, update: object) -> None:
        if isinstance(update, StatusUpdate):
            self.phase_label.setText(update.message)
            self.log(update.message)
            return
        if not isinstance(update, ScanUpdate):
            return
        progress = update.progress
        self._show_progress(progress)
        effective = update.effective_batch_size
        if self._last_effective_batch is not None and effective < self._last_effective_batch:
            self.log(f"Out of GPU memory: batch size reduced to {effective}")
        self._last_effective_batch = effective

    def _show_progress(self, progress: ScanProgress) -> None:
        self.phase_label.setText(PHASE_TITLES.get(progress.phase, progress.phase))
        total = max(progress.total, 0)
        if progress.phase == "discovering":
            self.progress_bar.setRange(0, 0)
        elif progress.phase == "checking":
            self.progress_bar.setRange(0, max(total, 1))
            self.progress_bar.setValue(min(progress.done, total))
        else:
            finished = progress.processed + progress.reused + progress.skipped + progress.errors
            self.progress_bar.setRange(0, max(total, 1))
            self.progress_bar.setValue(total if progress.phase == "done" else min(finished, total))
        self.current_file_label.set_full_text(progress.current_file)
        self._show_counters(progress.processed, progress.reused, progress.skipped, progress.errors)
        if progress.images_per_second is not None:
            self.rate_label.setText(f"{progress.images_per_second:.2f}")

    def _show_counters(self, processed: int, reused: int, skipped: int, errors: int) -> None:
        for key, value in zip(COUNTER_KEYS, (processed, reused, skipped, errors)):
            self.counter_labels[key].setText(str(value))

    def _finish_timers(self) -> None:
        self._elapsed_timer.stop()
        self._refresh_elapsed()

    def _on_finished(self, summary: object) -> None:
        self._finish_timers()
        self.progress_bar.setRange(0, 1)
        if not isinstance(summary, ScanSummary):
            self.progress_bar.setValue(0)
            self.phase_label.setText("Cancelled")
            self.log("Scan cancelled before it started.")
            self.ctx.status("Scan cancelled.")
            return
        self.progress_bar.setValue(1 if summary.status == "completed" else 0)
        self._show_counters(summary.processed, summary.reused, summary.skipped, summary.errors)
        self.current_file_label.set_full_text("")
        if summary.elapsed_seconds > 0 and summary.processed:
            self.rate_label.setText(f"{summary.processed / summary.elapsed_seconds:.2f}")
        self.phase_label.setText("Completed" if summary.status == "completed" else summary.status.capitalize())
        self._log_file_errors(summary)
        message = (f"Scan {summary.status}: found {summary.total_found}, processed {summary.processed}, "
                   f"reused {summary.reused}, skipped {summary.skipped}, errors {summary.errors} "
                   f"in {format_duration(summary.elapsed_seconds)}")
        self.log(message)
        self.ctx.status(message)
        self.ctx.events.results_changed.emit()

    def _log_file_errors(self, summary: ScanSummary) -> None:
        if not summary.errors:
            return
        rows = [row for row in self.ctx.db.query_results(status="error", order_by="file_name", descending=False)
                if row["run_id"] == summary.run_id]
        for row in rows[:MAX_LOGGED_FILE_ERRORS]:
            self.log(f"Error: {row['path']}: {row['error']}")
        if len(rows) > MAX_LOGGED_FILE_ERRORS:
            self.log(f"... and {len(rows) - MAX_LOGGED_FILE_ERRORS} more errors (see the Results tab)")

    def _on_failed(self, error: BaseException) -> None:
        self._finish_timers()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.phase_label.setText("Failed")
        self.log(f"Scan failed: {type(error).__name__}: {error}")
        self.ctx.status("Scan failed.")
        self.ctx.events.results_changed.emit()
