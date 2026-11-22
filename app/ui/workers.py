from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from benchmark import BenchmarkResult, run_benchmark
from classifier import BlueEyeEngine
from database import Database
from evaluation import import_folder_labels
from gpu import DeviceReport, detect_devices
from model_manager import ModelStatus, check_model
from scanner import ScanProgress, Scanner, ScanSummary, discover_images
from ui.engine_manager import EngineManager

logger = logging.getLogger("blueeye.app")

PROGRESS_INTERVAL_SECONDS = 0.1

SCAN_JOB = "scan"
BENCHMARK_JOB = "benchmark"
LABEL_IMPORT_JOB = "label_import"
MODEL_VERIFY_JOB = "model_verify"
DEVICE_DETECT_JOB = "device_detect"


class NoImagesFoundError(Exception):
    pass


@dataclass
class StatusUpdate:
    message: str


@dataclass
class ScanUpdate:
    progress: ScanProgress
    effective_batch_size: int


@dataclass
class BenchmarkUpdate:
    stage: str
    done: int
    total: int


@dataclass
class CountUpdate:
    done: int
    total: int


class ProgressThrottle:
    def __init__(self, interval: float = PROGRESS_INTERVAL_SECONDS):
        self._interval = interval
        self._last_time = 0.0
        self._last_key: object = None

    def ready(self, key: object = None, force: bool = False) -> bool:
        now = time.monotonic()
        if force or key != self._last_key or now - self._last_time >= self._interval:
            self._last_time = now
            self._last_key = key
            return True
        return False


class Job(QObject):
    progress = Signal(object)
    succeeded = Signal(object)
    failed = Signal(object)
    completed = Signal()

    kind = ""

    def __init__(self, title: str):
        super().__init__()
        self.title = title
        self.uses_engine = False
        self.retry: Callable[[], None] | None = None
        self._stop_event = threading.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    @Slot()
    def run(self) -> None:
        try:
            result = self.execute()
        except Exception as error:
            logger.exception("%s failed", self.title)
            self.failed.emit(error)
        else:
            self.succeeded.emit(result)
        finally:
            self.completed.emit()

    def execute(self) -> object:
        raise NotImplementedError


class EngineJob(Job):
    def __init__(self, title: str, engine_manager: EngineManager, device: str, device_label: str,
                 precision: str):
        super().__init__(title)
        self.uses_engine = True
        self.engine_manager = engine_manager
        self.device = device
        self.device_label = device_label
        self.precision = precision

    def acquire_engine(self) -> BlueEyeEngine | None:
        if not self.engine_manager.is_loaded(self.device, self.precision):
            self.progress.emit(StatusUpdate(f"Loading model on {self.device_label} ({self.precision})..."))
        engine = self.engine_manager.acquire(self.device, self.precision)
        return None if self.stop_requested() else engine


class ScanJob(EngineJob):
    kind = SCAN_JOB

    def __init__(self, engine_manager: EngineManager, db_path: Path, folder: Path, recursive: bool,
                 batch_size: int, force: bool, device: str, device_label: str, gpu_name: str | None,
                 precision: str):
        super().__init__("Scan", engine_manager, device, device_label, precision)
        self.db_path = db_path
        self.folder = folder
        self.recursive = recursive
        self.batch_size = batch_size
        self.force = force
        self.gpu_name = gpu_name

    def execute(self) -> ScanSummary | None:
        engine = self.acquire_engine()
        if engine is None:
            return None
        revision = check_model(full_hash=False).revision
        throttle = ProgressThrottle()

        def report(progress: ScanProgress) -> None:
            if throttle.ready(progress.phase, force=progress.phase == "done"):
                self.progress.emit(ScanUpdate(progress, engine.effective_batch_size(self.batch_size)))

        with Database(self.db_path) as db:
            scanner = Scanner(db, engine, self.gpu_name, revision)
            return scanner.run(self.folder, self.recursive, self.batch_size, force=self.force,
                               progress=report, should_stop=self.stop_requested)


class BenchmarkJob(EngineJob):
    kind = BENCHMARK_JOB

    def __init__(self, engine_manager: EngineManager, db_path: Path, folder: Path, batch_sizes: list[int],
                 image_limit: int, device: str, device_label: str, gpu_name: str | None, precision: str):
        super().__init__("Benchmark", engine_manager, device, device_label, precision)
        self.db_path = db_path
        self.folder = folder
        self.batch_sizes = batch_sizes
        self.image_limit = image_limit
        self.gpu_name = gpu_name

    def execute(self) -> list[BenchmarkResult] | None:
        paths = discover_images(self.folder, recursive=True)
        if not paths:
            raise NoImagesFoundError(f"No supported images found in {self.folder}")
        engine = self.acquire_engine()
        if engine is None:
            return None
        throttle = ProgressThrottle()

        def report(stage: str, done: int, total: int) -> None:
            if throttle.ready(stage, force=done >= total):
                self.progress.emit(BenchmarkUpdate(stage, done, total))

        with Database(self.db_path) as db:
            return run_benchmark(engine, paths, self.batch_sizes, self.gpu_name, image_limit=self.image_limit,
                                 progress=report, should_stop=self.stop_requested, db=db)


class LabelImportJob(Job):
    kind = LABEL_IMPORT_JOB

    def __init__(self, db_path: Path, root: Path):
        super().__init__("Label import")
        self.db_path = db_path
        self.root = root

    def execute(self) -> int:
        throttle = ProgressThrottle()

        def report(done: int, total: int) -> None:
            if throttle.ready(force=done >= total):
                self.progress.emit(CountUpdate(done, total))

        with Database(self.db_path) as db:
            return import_folder_labels(db, self.root, recursive=True, progress=report)


class ModelVerifyJob(Job):
    kind = MODEL_VERIFY_JOB

    def __init__(self) -> None:
        super().__init__("Model verification")

    def execute(self) -> ModelStatus:
        return check_model(full_hash=True)


class DeviceDetectJob(Job):
    kind = DEVICE_DETECT_JOB

    def __init__(self) -> None:
        super().__init__("Device detection")

    def execute(self) -> DeviceReport:
        return detect_devices()
