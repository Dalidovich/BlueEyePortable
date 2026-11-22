from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

CUDA_ERROR_MARKERS = ("cuda", "cudnn", "cublas", "no kernel image", "driver")


def describe_error(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def is_cuda_error(error: BaseException) -> bool:
    is_runtime_like = isinstance(error, RuntimeError) or type(error).__module__.startswith("torch")
    text = describe_error(error).lower()
    return is_runtime_like and any(marker in text for marker in CUDA_ERROR_MARKERS)


class DialogService:
    def __init__(self, parent: QWidget):
        self.parent = parent

    def error(self, title: str, text: str) -> None:
        QMessageBox.critical(self.parent, title, text)

    def warning(self, title: str, text: str) -> None:
        QMessageBox.warning(self.parent, title, text)

    def information(self, title: str, text: str) -> None:
        QMessageBox.information(self.parent, title, text)

    def confirm(self, title: str, text: str) -> bool:
        answer = QMessageBox.question(self.parent, title, text,
                                      QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                      QMessageBox.StandardButton.No)
        return answer == QMessageBox.StandardButton.Yes

    def offer_cpu_retry(self, title: str, text: str) -> bool:
        box = QMessageBox(QMessageBox.Icon.Critical, title, text, QMessageBox.StandardButton.NoButton, self.parent)
        retry_button = box.addButton("Switch to CPU and retry", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Close)
        box.exec()
        return box.clickedButton() is retry_button

    def choose_directory(self, title: str, start: Path) -> Path | None:
        chosen = QFileDialog.getExistingDirectory(self.parent, title, str(start))
        return Path(chosen) if chosen else None

    def choose_save_path(self, title: str, default: Path, file_filter: str) -> Path | None:
        chosen, _selected_filter = QFileDialog.getSaveFileName(self.parent, title, str(default), file_filter)
        return Path(chosen) if chosen else None
