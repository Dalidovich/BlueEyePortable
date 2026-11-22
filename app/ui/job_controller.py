from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot

from ui.dialogs import is_cuda_error
from ui.engine_manager import EngineManager
from ui.workers import Job

ProgressCallback = Callable[[object], None]
SuccessCallback = Callable[[object], None]
FailureCallback = Callable[[BaseException], None]


class JobController(QObject):
    busy_changed = Signal(object)
    job_failed = Signal(object, object)

    def __init__(self, engine_manager: EngineManager, parent: QObject | None = None):
        super().__init__(parent)
        self._engine_manager = engine_manager
        self._job: Job | None = None
        self._thread: QThread | None = None
        self._on_progress: ProgressCallback | None = None
        self._on_success: SuccessCallback | None = None
        self._on_failure: FailureCallback | None = None
        self._result: object = None
        self._error: BaseException | None = None

    @property
    def current_job(self) -> Job | None:
        return self._job

    @property
    def current_kind(self) -> str | None:
        return self._job.kind if self._job is not None else None

    def is_running(self) -> bool:
        return self._job is not None

    def start(self, job: Job, on_progress: ProgressCallback | None = None,
              on_success: SuccessCallback | None = None, on_failure: FailureCallback | None = None) -> bool:
        if self._job is not None:
            return False
        thread = QThread()
        thread.setObjectName(f"job-{job.kind}")
        job.moveToThread(thread)
        self._job = job
        self._thread = thread
        self._on_progress = on_progress
        self._on_success = on_success
        self._on_failure = on_failure
        self._result = None
        self._error = None
        thread.started.connect(job.run)
        job.progress.connect(self._handle_progress, Qt.ConnectionType.QueuedConnection)
        job.succeeded.connect(self._handle_success, Qt.ConnectionType.QueuedConnection)
        job.failed.connect(self._handle_failure, Qt.ConnectionType.QueuedConnection)
        job.completed.connect(thread.quit)
        thread.finished.connect(self._handle_thread_finished, Qt.ConnectionType.QueuedConnection)
        self.busy_changed.emit(job.kind)
        thread.start()
        return True

    def request_stop(self) -> None:
        if self._job is not None:
            self._job.request_stop()

    def wait(self, milliseconds: int) -> bool:
        if self._thread is None:
            return True
        return self._thread.wait(milliseconds)

    @Slot(object)
    def _handle_progress(self, update: object) -> None:
        if self._job is not None and self._on_progress is not None:
            self._on_progress(update)

    @Slot(object)
    def _handle_success(self, result: object) -> None:
        self._result = result

    @Slot(object)
    def _handle_failure(self, error: object) -> None:
        self._error = error if isinstance(error, BaseException) else RuntimeError(str(error))

    @Slot()
    def _handle_thread_finished(self) -> None:
        job, thread = self._job, self._thread
        if job is None or thread is None:
            return
        thread.wait()
        on_success, on_failure = self._on_success, self._on_failure
        result, error = self._result, self._error
        self._job = None
        self._thread = None
        self._on_progress = None
        self._on_success = None
        self._on_failure = None
        if error is not None and job.uses_engine and is_cuda_error(error):
            self._engine_manager.discard()
        self.busy_changed.emit(None)
        if error is not None:
            if on_failure is not None:
                on_failure(error)
            self.job_failed.emit(job, error)
        elif on_success is not None:
            on_success(result)
