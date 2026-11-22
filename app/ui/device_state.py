from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal

from gpu import DeviceReport, bf16_available, choose_device, device_label
from settings import PRECISIONS, Settings

logger = logging.getLogger("blueeye.app")

AUTO_DEVICE = "auto"
CPU_DEVICE = "cpu"
FP32 = "fp32"
BF16 = "bf16"


class DeviceState(QObject):
    changed = Signal()
    notice = Signal(str)

    def __init__(self, settings: Settings, report: DeviceReport, parent: QObject | None = None):
        super().__init__(parent)
        self._settings = settings
        self._report = report
        self._preference = settings.device or AUTO_DEVICE
        self._precision = settings.precision if settings.precision in PRECISIONS else FP32

    @property
    def report(self) -> DeviceReport:
        return self._report

    @property
    def preference(self) -> str:
        return self._preference

    @property
    def precision(self) -> str:
        return self._precision

    def resolved_device(self) -> str:
        return choose_device(self._report, self._preference)

    def label_for(self, device: str) -> str:
        return device_label(self._report, device)

    def resolved_label(self) -> str:
        return self.label_for(self.resolved_device())

    def gpu_name_for(self, device: str) -> str | None:
        gpu = self._report.gpu_for(device)
        return gpu.name if gpu is not None else None

    def device_choices(self) -> list[tuple[str, str]]:
        auto_label = self.label_for(choose_device(self._report, AUTO_DEVICE))
        choices = [(AUTO_DEVICE, f"Auto ({auto_label})")]
        choices.extend((gpu.device, f"{gpu.name} ({gpu.device})") for gpu in self._report.compatible_gpus)
        choices.append((CPU_DEVICE, "CPU"))
        return choices

    def bf16_supported(self, device: str | None = None) -> bool:
        return bf16_available(self._report, device or self.resolved_device())

    def bf16_unsupported_reason(self) -> str:
        return f"bfloat16 is not supported by {self.resolved_label()}"

    def normalize(self) -> None:
        preference_changed = self._validate_preference()
        precision_changed = self._validate_precision()
        if preference_changed or precision_changed:
            self._persist()
        self.changed.emit()

    def set_preference(self, preference: str) -> None:
        if preference == self._preference:
            return
        self._preference = preference
        self._validate_preference()
        self._validate_precision()
        self._persist()
        self.changed.emit()

    def set_precision(self, precision: str) -> None:
        if precision not in PRECISIONS:
            raise ValueError(f"Unsupported precision {precision!r}")
        if precision == BF16 and not self.bf16_supported():
            self.notice.emit(f"{self.bf16_unsupported_reason()}; keeping {self._precision}.")
            self.changed.emit()
            return
        if precision == self._precision:
            return
        self._precision = precision
        self._persist()
        self.changed.emit()

    def set_report(self, report: DeviceReport) -> None:
        self._report = report
        self._validate_preference()
        self._validate_precision()
        self._persist()
        self.changed.emit()

    def describe(self) -> str:
        return f"Device: {self.resolved_label()} | Precision: {self._precision}"

    def _validate_preference(self) -> bool:
        if self._preference in (AUTO_DEVICE, CPU_DEVICE):
            return False
        gpu = self._report.gpu_for(self._preference)
        if gpu is not None and gpu.compatible:
            return False
        self.notice.emit(f"Saved device {self._preference} is not available; using automatic selection.")
        self._preference = AUTO_DEVICE
        return True

    def _validate_precision(self) -> bool:
        if self._precision != BF16 or self.bf16_supported():
            return False
        self.notice.emit(f"{self.bf16_unsupported_reason()}; precision switched to fp32.")
        self._precision = FP32
        return True

    def _persist(self) -> None:
        self._settings.device = self._preference
        self._settings.precision = self._precision
        try:
            self._settings.save()
        except OSError:
            logger.exception("Cannot save settings")
