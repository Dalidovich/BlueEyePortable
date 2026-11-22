from __future__ import annotations

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget)

from gpu import DeviceReport
from model_manager import ModelStatus, check_model, read_manifest
from ui.context import AppContext
from ui.formatting import yes_no
from ui.widgets import fill_static_table, make_static_table
from ui.workers import DEVICE_DETECT_JOB, MODEL_VERIFY_JOB, DeviceDetectJob, ModelVerifyJob

GPU_HEADERS = ["Index", "Name", "VRAM (GiB)", "Compute capability", "Compatible", "bf16"]
STAT_KEYS = (("total", "Total processed"), ("safe", "Safe"), ("suggestive", "Suggestive"),
             ("explicit", "Explicit"), ("errors", "Errors"))


class DashboardTab(QWidget):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None):
        super().__init__(parent)
        self.ctx = ctx
        layout = QVBoxLayout(self)
        layout.addWidget(self._build_model_group())
        layout.addWidget(self._build_environment_group(), 1)
        layout.addWidget(self._build_statistics_group())
        self.verify_button.clicked.connect(self.verify_model)
        self.redetect_button.clicked.connect(self.redetect_devices)
        ctx.device_state.changed.connect(self._on_device_state_changed)
        ctx.events.results_changed.connect(self.refresh_statistics)
        ctx.events.model_status_changed.connect(self.show_model_status)
        self.show_model_status(check_model(full_hash=False))
        self.refresh_environment()
        self.refresh_statistics()

    def _build_model_group(self) -> QGroupBox:
        group = QGroupBox("Model")
        form = QFormLayout()
        self.model_state_label = QLabel()
        self.model_state_label.setWordWrap(True)
        self.model_revision_label = QLabel()
        self.model_verified_label = QLabel()
        form.addRow("Status:", self.model_state_label)
        form.addRow("Revision:", self.model_revision_label)
        form.addRow("Verified against Hugging Face:", self.model_verified_label)
        self.verify_button = QPushButton("Verify model files (full checksum)")
        buttons = QHBoxLayout()
        buttons.addWidget(self.verify_button)
        buttons.addStretch(1)
        box = QVBoxLayout(group)
        box.addLayout(form)
        box.addLayout(buttons)
        return group

    def _build_environment_group(self) -> QGroupBox:
        group = QGroupBox("Environment")
        form = QFormLayout()
        self.torch_label = QLabel()
        self.cuda_runtime_label = QLabel()
        self.cuda_available_label = QLabel()
        self.cuda_available_label.setWordWrap(True)
        self.inference_device_label = QLabel()
        form.addRow("PyTorch version:", self.torch_label)
        form.addRow("CUDA runtime:", self.cuda_runtime_label)
        form.addRow("CUDA available:", self.cuda_available_label)
        form.addRow("Inference device:", self.inference_device_label)
        self.gpu_table = make_static_table(GPU_HEADERS)
        self.redetect_button = QPushButton("Re-detect devices")
        buttons = QHBoxLayout()
        buttons.addWidget(self.redetect_button)
        buttons.addStretch(1)
        box = QVBoxLayout(group)
        box.addLayout(form)
        box.addWidget(QLabel("Detected GPUs:"))
        box.addWidget(self.gpu_table, 1)
        box.addLayout(buttons)
        return group

    def _build_statistics_group(self) -> QGroupBox:
        group = QGroupBox("Statistics")
        self.statistics_title = QLabel()
        form = QFormLayout()
        self.stat_labels: dict[str, QLabel] = {}
        for key, title in STAT_KEYS:
            label = QLabel("0")
            self.stat_labels[key] = label
            form.addRow(f"{title}:", label)
        box = QVBoxLayout(group)
        box.addWidget(self.statistics_title)
        box.addLayout(form)
        return group

    def set_busy(self, kind: str | None) -> None:
        idle = kind is None
        self.verify_button.setEnabled(idle)
        self.redetect_button.setEnabled(idle)
        self.verify_button.setText("Verifying model files..." if kind == MODEL_VERIFY_JOB
                                   else "Verify model files (full checksum)")
        self.redetect_button.setText("Detecting devices..." if kind == DEVICE_DETECT_JOB else "Re-detect devices")

    def on_shown(self) -> None:
        self.refresh_statistics()

    @Slot(object)
    def show_model_status(self, status: ModelStatus) -> None:
        manifest = read_manifest() or {}
        self.model_state_label.setText(("Present - " if status.present else "Not ready - ") + status.message)
        self.model_revision_label.setText(status.revision or manifest.get("revision") or "unknown")
        self.model_verified_label.setText(yes_no(bool(manifest.get("verified_against_hub"))))

    @Slot()
    def refresh_environment(self) -> None:
        state = self.ctx.device_state
        report: DeviceReport = state.report
        self.torch_label.setText(report.torch_version)
        self.cuda_runtime_label.setText(report.cuda_runtime or "none")
        available = yes_no(report.cuda_available)
        self.cuda_available_label.setText(f"{available} ({report.cuda_error})" if report.cuda_error else available)
        self.inference_device_label.setText(f"{state.resolved_label()} - {state.precision}")
        rows = [[str(gpu.index), gpu.name, f"{gpu.vram_gib:.1f}", gpu.capability_label,
                 "yes" if gpu.compatible else "no (not supported by this PyTorch build)",
                 yes_no(gpu.bf16_supported)] for gpu in report.gpus]
        fill_static_table(self.gpu_table, rows, numeric_columns={0, 2})

    @Slot()
    def refresh_statistics(self) -> None:
        precision = self.ctx.device_state.precision
        summary = self.ctx.db.summary(precision)
        self.statistics_title.setText(f"Stored results for precision {precision}:")
        for key, _title in STAT_KEYS:
            self.stat_labels[key].setText(str(summary.get(key, 0)))

    @Slot()
    def verify_model(self) -> None:
        self.model_state_label.setText("Verifying checksums of the model files...")
        if not self.ctx.jobs.start(ModelVerifyJob(), on_success=self._on_model_verified,
                                   on_failure=self._on_model_verify_failed):
            self.show_model_status(check_model(full_hash=False))

    def _on_model_verified(self, status: object) -> None:
        if not isinstance(status, ModelStatus):
            return
        self.ctx.events.model_status_changed.emit(status)
        if status.present:
            self.ctx.status("Model files verified: all checksums match.")
        else:
            self.ctx.dialogs.warning("Model verification failed", status.message)

    def _on_model_verify_failed(self, _error: BaseException) -> None:
        self.show_model_status(check_model(full_hash=False))

    @Slot()
    def redetect_devices(self) -> None:
        self.ctx.jobs.start(DeviceDetectJob(), on_success=self._on_devices_detected)

    def _on_devices_detected(self, report: object) -> None:
        if isinstance(report, DeviceReport):
            self.ctx.device_state.set_report(report)
            self.ctx.status(f"Devices re-detected: {len(report.gpus)} GPU(s), "
                            f"{len(report.compatible_gpus)} compatible.")

    @Slot()
    def _on_device_state_changed(self) -> None:
        self.refresh_environment()
        self.refresh_statistics()
