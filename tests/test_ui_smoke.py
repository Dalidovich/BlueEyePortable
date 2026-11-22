from __future__ import annotations

import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import shutil
import tempfile
import time
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest import mock

import settings
from settings import isolate_huggingface_environment

isolate_huggingface_environment()

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QApplication

from gpu import bf16_available
from model_manager import check_model
from tests.fakes import write_image
from ui.device_state import BF16, CPU_DEVICE, FP32
from ui.main_window import MainWindow

EXPECTED_TABS = ["Dashboard", "Scan", "Results", "Benchmark", "Evaluation"]
SCAN_TIMEOUT_SECONDS = 600.0
POLL_MILLISECONDS = 50


class RecordingDialogs:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def _record(self, kind: str, title: str, text: str) -> None:
        self.calls.append((kind, title, text))

    def error(self, title: str, text: str) -> None:
        self._record("error", title, text)

    def warning(self, title: str, text: str) -> None:
        self._record("warning", title, text)

    def information(self, title: str, text: str) -> None:
        self._record("information", title, text)

    def confirm(self, title: str, text: str) -> bool:
        self._record("confirm", title, text)
        return True

    def offer_cpu_retry(self, title: str, text: str) -> bool:
        self._record("offer_cpu_retry", title, text)
        return False

    def choose_directory(self, title: str, start: Path) -> Path | None:
        return None

    def choose_save_path(self, title: str, default: Path, file_filter: str) -> Path | None:
        return None


def application() -> QApplication:
    instance = QApplication.instance()
    return instance if isinstance(instance, QApplication) else QApplication([])


def spin_until(predicate: Callable[[], bool], timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    loop = QEventLoop()
    timer = QTimer()
    timer.setInterval(POLL_MILLISECONDS)

    def check() -> None:
        if predicate() or time.monotonic() >= deadline:
            loop.quit()

    timer.timeout.connect(check)
    timer.start()
    if not predicate():
        loop.exec()
    timer.stop()
    return predicate()


class UiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = application()

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="blueeye_ui_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        settings_patch = mock.patch.object(settings, "SETTINGS_PATH", self.tmp / "settings.json")
        settings_patch.start()
        self.addCleanup(settings_patch.stop)
        self.images = self.tmp / "images"
        for seed in range(3):
            write_image(self.images / f"synthetic_{seed}.png", seed + 1, size=64)
        self.window = MainWindow(db_path=self.tmp / "results.db")
        self.dialogs = RecordingDialogs()
        self.window.context.dialogs = self.dialogs
        self.addCleanup(self._close_window)

    def _close_window(self) -> None:
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def test_window_structure(self) -> None:
        window = self.window
        self.assertEqual(window.windowTitle(), "BlueEye Portable Tester")
        self.assertEqual([window.tabs.tabText(i) for i in range(window.tabs.count())], EXPECTED_TABS)
        self.assertIn(CPU_DEVICE, window.scan_tab.device_selector.device_values())
        self.assertIn(CPU_DEVICE, window.benchmark_tab.device_selector.device_values())
        self.assertIn("Precision:", window.status_label.text())

    def test_bf16_item_matches_backend_support(self) -> None:
        state = self.window.context.device_state
        selector = self.window.scan_tab.device_selector
        for preference in selector.device_values():
            state.set_preference(preference)
            expected = bf16_available(state.report, state.resolved_device())
            self.assertEqual(selector.precision_enabled(BF16), expected, preference)
            self.assertTrue(selector.precision_enabled(FP32))
            if not expected:
                self.assertEqual(state.precision, FP32)

    def test_unsupported_bf16_falls_back_to_fp32(self) -> None:
        state = self.window.context.device_state
        state.set_preference(CPU_DEVICE)
        state.set_precision(BF16)
        expected = BF16 if bf16_available(state.report, CPU_DEVICE) else FP32
        self.assertEqual(state.precision, expected)

    def test_metrics_panel_handles_empty_data(self) -> None:
        tab = self.window.evaluation_tab
        tab.refresh()
        self.assertEqual(tab.model.rowCount(), 0)
        self.assertEqual(tab.metrics_panel.metrics.total, 0)
        self.assertEqual(tab.metrics_panel.labeled_label.text(), "0")
        self.assertFalse(tab.metrics_panel.export_csv_button.isEnabled())
        tab.label_current("safe")
        self.assertEqual(tab.metrics_panel.metrics.total, 0)

    def test_results_model_sorts_numerically(self) -> None:
        tab = self.window.results_tab
        tab.model.set_rows([
            {"file_name": "a.png", "label": "safe", "confidence": 0.9, "status": "ok", "path": "a.png"},
            {"file_name": "b.png", "label": "explicit", "confidence": 0.15, "status": "ok", "path": "b.png"},
            {"file_name": "c.png", "label": None, "confidence": None, "status": "error", "path": "c.png"},
            {"file_name": "d.png", "label": "suggestive", "confidence": 0.5, "status": "ok", "path": "d.png"},
        ])
        tab.table.sortByColumn(2, Qt.SortOrder.AscendingOrder)
        names = [tab.proxy.index(row, 0).data() for row in range(tab.proxy.rowCount())]
        self.assertEqual(names, ["b.png", "d.png", "a.png", "c.png"])

    @unittest.skipUnless(check_model(full_hash=False).present, "model files are not installed")
    def test_end_to_end_scan_on_cpu(self) -> None:
        window = self.window
        state = window.context.device_state
        state.set_preference(CPU_DEVICE)
        state.set_precision(FP32)
        window.scan_tab.folder_field.set_path(self.images)
        window.scan_tab.batch_spin.setValue(2)
        window.scan_tab.start_scan()
        self.assertTrue(window.context.jobs.is_running())
        self.assertFalse(window.scan_tab.start_button.isEnabled())
        self.assertFalse(window.benchmark_tab.run_button.isEnabled())
        self.assertTrue(window.scan_tab.stop_button.isEnabled())
        finished = spin_until(lambda: not window.context.jobs.is_running(), SCAN_TIMEOUT_SECONDS)
        self.assertTrue(finished, "scan did not finish in time")
        self.assertEqual(self.dialogs.calls, [])
        self.assertTrue(window.scan_tab.start_button.isEnabled())

        self.assertEqual(window.results_tab.model.rowCount(), 3)
        rows = window.results_tab.model.rows()
        self.assertTrue(all(row["status"] == "ok" and row["device"] == "cpu" for row in rows))
        self.assertEqual(window.dashboard_tab.stat_labels["total"].text(), "3")
        counted = sum(int(window.dashboard_tab.stat_labels[key].text()) for key in ("safe", "suggestive", "explicit"))
        self.assertEqual(counted, 3)

        tab = window.evaluation_tab
        tab.refresh()
        self.assertEqual(tab.model.rowCount(), 3)
        tab.select_row(0)
        tab.label_current("safe")
        self.assertEqual(tab.metrics_panel.metrics.total, 1)
        self.assertEqual(tab.current_row(), 1)
        self.assertEqual(window.context.db.labeled_pairs(FP32)[0]["true_label"], "safe")
        spin_until(lambda: tab.preview.has_image(), 10.0)
        self.assertTrue(tab.preview.has_image())


if __name__ == "__main__":
    unittest.main()
