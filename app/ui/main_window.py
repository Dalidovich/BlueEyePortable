from __future__ import annotations

import logging
import time
from pathlib import Path

from PySide6.QtCore import QEventLoop, QTimer, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QTabWidget

from classifier import ModelNotReadyError
from database import Database
from gpu import DeviceReport, detect_devices
from model_manager import ModelStatus, check_model
from scanner import InsufficientDiskSpaceError
from settings import DB_PATH, Settings
from ui.benchmark_tab import BenchmarkTab
from ui.context import AppContext, AppEvents
from ui.dashboard_tab import DashboardTab
from ui.device_state import CPU_DEVICE, DeviceState
from ui.dialogs import DialogService, describe_error, is_cuda_error
from ui.engine_manager import EngineManager
from ui.evaluation_tab import EvaluationTab
from ui.job_controller import JobController
from ui.results_tab import ResultsTab
from ui.scan_tab import ScanTab
from ui.workers import EngineJob, Job, NoImagesFoundError

logger = logging.getLogger("blueeye.app")

WINDOW_TITLE = "BlueEye Portable Tester"
STATUS_MESSAGE_MILLISECONDS = 10000
CLOSE_WAIT_SECONDS = 300.0
CLOSE_WAIT_SLICE_MILLISECONDS = 50


class MainWindow(QMainWindow):
    def __init__(self, db_path: Path = DB_PATH, device_report: DeviceReport | None = None):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1320, 860)
        db_path = Path(db_path)
        settings = Settings.load()
        self.engine_manager = EngineManager()
        jobs = JobController(self.engine_manager, self)
        report = device_report if device_report is not None else detect_devices()
        device_state = DeviceState(settings, report, self)
        self.context = AppContext(db_path=db_path, db=Database(db_path), settings=settings,
                                  device_state=device_state, jobs=jobs, engine_manager=self.engine_manager,
                                  dialogs=DialogService(self), events=AppEvents(self))
        self._model_status: ModelStatus = check_model(full_hash=False)
        self._closing = False
        self._closed = False
        self.status_label = QLabel()
        self.statusBar().addPermanentWidget(self.status_label)
        device_state.notice.connect(self.show_status_message)
        self.context.events.status_message.connect(self.show_status_message)
        self.context.events.model_status_changed.connect(self._on_model_status_changed)
        device_state.changed.connect(self._update_status_label)
        device_state.normalize()

        self.dashboard_tab = DashboardTab(self.context)
        self.scan_tab = ScanTab(self.context)
        self.results_tab = ResultsTab(self.context)
        self.benchmark_tab = BenchmarkTab(self.context, self.scan_tab.folder_text)
        self.evaluation_tab = EvaluationTab(self.context)
        self.tabs = QTabWidget()
        self.tabs.addTab(self.dashboard_tab, "Dashboard")
        self.tabs.addTab(self.scan_tab, "Scan")
        self.tabs.addTab(self.results_tab, "Results")
        self.tabs.addTab(self.benchmark_tab, "Benchmark")
        self.tabs.addTab(self.evaluation_tab, "Evaluation")
        self.setCentralWidget(self.tabs)

        jobs.busy_changed.connect(self._on_busy_changed)
        jobs.job_failed.connect(self._on_job_failed)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self._update_status_label()
        self._on_busy_changed(None)
        if not self._model_status.present:
            QTimer.singleShot(0, lambda: self.show_status_message(
                f"Model not ready: {self._model_status.message}"))

    @property
    def all_tabs(self) -> tuple:
        return (self.dashboard_tab, self.scan_tab, self.results_tab, self.benchmark_tab, self.evaluation_tab)

    @Slot(str)
    def show_status_message(self, message: str) -> None:
        self.statusBar().showMessage(message, STATUS_MESSAGE_MILLISECONDS)

    @Slot()
    def _update_status_label(self) -> None:
        model_text = "ready" if self._model_status.present else "not ready"
        self.status_label.setText(f"{self.context.device_state.describe()} | Model: {model_text}")

    @Slot(object)
    def _on_model_status_changed(self, status: ModelStatus) -> None:
        self._model_status = status
        self._update_status_label()

    @Slot(object)
    def _on_busy_changed(self, kind: object) -> None:
        for tab in self.all_tabs:
            tab.set_busy(kind if isinstance(kind, str) else None)

    @Slot(int)
    def _on_tab_changed(self, index: int) -> None:
        widget = self.tabs.widget(index)
        if widget is self.dashboard_tab or widget is self.evaluation_tab:
            widget.on_shown()

    @Slot(object, object)
    def _on_job_failed(self, job: Job, error: BaseException) -> None:
        if self._closing:
            return
        dialogs = self.context.dialogs
        details = describe_error(error)
        if isinstance(error, ModelNotReadyError):
            dialogs.error("Model not ready", f"{error}\n\nRun Setup.bat to download and verify the model files, "
                                             "then start the application again.")
        elif isinstance(error, InsufficientDiskSpaceError):
            dialogs.error("Not enough disk space", f"{error}\n\nFree some disk space and try again.")
        elif isinstance(job, EngineJob) and job.device != CPU_DEVICE and is_cuda_error(error):
            text = (f"{job.title} failed on {job.device_label}:\n\n{details}\n\n"
                    "The GPU could not run the model. You can switch to the CPU and retry.")
            if dialogs.offer_cpu_retry("GPU error", text):
                self.context.device_state.set_preference(CPU_DEVICE)
                if job.retry is not None:
                    QTimer.singleShot(0, job.retry)
        elif isinstance(error, (FileNotFoundError, NotADirectoryError, NoImagesFoundError)):
            dialogs.warning(job.title, str(error))
        else:
            dialogs.error(f"{job.title} failed", details)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._closed:
            event.accept()
            return
        if self._closing:
            event.ignore()
            return
        jobs = self.context.jobs
        if jobs.is_running():
            title = jobs.current_job.title if jobs.current_job is not None else "A job"
            if not self.context.dialogs.confirm("Job in progress",
                                                f"{title} is still running. Stop it and exit?"):
                event.ignore()
                return
            self._closing = True
            self._wait_for_running_job()
        self._closing = True
        self._shutdown()
        self._closed = True
        event.accept()

    def _wait_for_running_job(self) -> None:
        jobs = self.context.jobs
        jobs.request_stop()
        self.tabs.setEnabled(False)
        self.statusBar().showMessage("Stopping... waiting for the current batch to finish")
        deadline = time.monotonic() + CLOSE_WAIT_SECONDS
        while jobs.is_running() and time.monotonic() < deadline:
            jobs.wait(CLOSE_WAIT_SLICE_MILLISECONDS)
            QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, CLOSE_WAIT_SLICE_MILLISECONDS)
        if jobs.is_running():
            logger.warning("The running job did not stop within %.0f s; closing anyway", CLOSE_WAIT_SECONDS)

    def _shutdown(self) -> None:
        self.scan_tab.store_settings()
        self.benchmark_tab.store_settings()
        self.context.save_settings()
        self.evaluation_tab.shutdown()
        self.context.db.close()
        if not self.context.jobs.is_running():
            self.engine_manager.close()
