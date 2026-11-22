from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from classifier import SUPPORTED_EXTENSIONS, ImageResult, InferenceStopped
from database import Database, ResultRow
from settings import DATA_DIR, OUTPUT_DIR, to_portable_path

logger = logging.getLogger("blueeye.inference")

HASH_CHUNK_BYTES = 1024 * 1024
MIN_FREE_BYTES = 200 * 1024 * 1024
REUSE_FLUSH_SIZE = 100
EmitFn = Callable[..., None]


class InsufficientDiskSpaceError(Exception):
    pass


@dataclass
class ScanProgress:
    phase: str
    done: int
    total: int
    current_file: str
    processed: int
    reused: int
    skipped: int
    errors: int
    images_per_second: float | None


@dataclass
class ScanSummary:
    run_id: int
    status: str
    total_found: int
    processed: int
    reused: int
    skipped: int
    errors: int
    elapsed_seconds: float
    message: str | None


@dataclass
class _QueueEntry:
    path: Path
    portable: str
    file_hash: str
    file_size: int
    file_mtime: float
    followers: list["_QueueEntry"] = field(default_factory=list)


def discover_images(folder: Path, recursive: bool) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder does not exist or is not a directory: {folder}")
    found: list[Path] = []

    def log_walk_error(error: OSError) -> None:
        logger.warning("Cannot read directory %s: %s", error.filename, error)

    if recursive:
        for current, _dirs, names in os.walk(folder, onerror=log_walk_error):
            for name in names:
                if Path(name).suffix.lower() in SUPPORTED_EXTENSIONS:
                    found.append(Path(current) / name)
    else:
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    if Path(entry.name).suffix.lower() in SUPPORTED_EXTENSIONS and entry.is_file():
                        found.append(Path(entry.path))
        except OSError as e:
            logger.warning("Cannot read directory %s: %s", folder, e)
    return sorted(found, key=lambda p: str(p).lower())


def sha256_file(path: Path, chunk: int = HASH_CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _existing_ancestor(path: Path) -> Path:
    current = path
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def check_disk_space(min_free_bytes: int = MIN_FREE_BYTES) -> None:
    for directory in (DATA_DIR, OUTPUT_DIR):
        free = shutil.disk_usage(_existing_ancestor(directory)).free
        if free < min_free_bytes:
            raise InsufficientDiskSpaceError(
                f"Not enough free disk space for {directory}: {free / 2**30:.2f} GiB free, "
                f"{min_free_bytes / 2**30:.2f} GiB required")


class Scanner:
    def __init__(self, db: Database, engine, gpu_name: str | None, model_revision: str | None):
        self.db = db
        self.engine = engine
        self.gpu_name = gpu_name
        self.model_revision = model_revision

    def run(self, folder: Path, recursive: bool, batch_size: int, force: bool = False,
            progress: Callable[[ScanProgress], None] | None = None,
            should_stop: Callable[[], bool] | None = None) -> ScanSummary:
        stop_requested = should_stop or (lambda: False)
        check_disk_space()
        folder = Path(folder)
        paths = discover_images(folder, recursive)
        precision = self.engine.precision
        run_id = self.db.start_run(to_portable_path(folder), recursive, self.engine.device, self.gpu_name,
                                   precision, batch_size)
        started = time.perf_counter()
        counters = {"processed": 0, "reused": 0, "skipped": 0, "errors": 0}
        total = len(paths)
        self.db.update_run(run_id, total_found=total)
        logger.info("Run %d started: folder=%s recursive=%s files=%d device=%s precision=%s batch=%d force=%s",
                    run_id, folder, recursive, total, self.engine.device, precision, batch_size, force)

        def emit(phase: str, done: int, current: str, rate: float | None = None) -> None:
            if progress is not None:
                progress(ScanProgress(phase, done, total, current, counters["processed"], counters["reused"],
                                      counters["skipped"], counters["errors"], rate))

        pending_rows: list[ResultRow] = []

        def flush_pending() -> None:
            if pending_rows:
                self.db.save_results(pending_rows)
                pending_rows.clear()
            self.db.update_run(run_id, **counters)

        status = "completed"
        message: str | None = None
        try:
            emit("discovering", 0, str(folder))
            queue = self._check_files(paths, precision, force, run_id, counters, pending_rows, flush_pending,
                                      stop_requested, emit)
            flush_pending()
            self._infer(queue, batch_size, run_id, counters, emit, stop_requested)
            emit("done", total, "")
        except InferenceStopped:
            flush_pending()
            status = "stopped"
            message = "Stopped by user"
        except Exception as e:
            flush_pending()
            self.db.finish_run(run_id, "failed", f"{type(e).__name__}: {e}")
            logger.exception("Run %d failed", run_id)
            raise
        self.db.update_run(run_id, **counters)
        self.db.finish_run(run_id, status, message)
        elapsed = time.perf_counter() - started
        logger.info("Run %d %s: found=%d processed=%d reused=%d skipped=%d errors=%d in %.1f s",
                    run_id, status, total, counters["processed"], counters["reused"], counters["skipped"],
                    counters["errors"], elapsed)
        return ScanSummary(run_id, status, total, counters["processed"], counters["reused"],
                           counters["skipped"], counters["errors"], elapsed, message)

    def _base_row(self, path: Path, portable: str, status: str, run_id: int) -> ResultRow:
        return ResultRow(path=portable, file_name=path.name, status=status, precision=self.engine.precision,
                         device=self.engine.device, gpu_name=self.gpu_name,
                         model_revision=self.model_revision, run_id=run_id)

    def _error_row(self, path: Path, portable: str, run_id: int, message: str, file_hash: str | None = None,
                   file_size: int | None = None, file_mtime: float | None = None) -> ResultRow:
        row = self._base_row(path, portable, "error", run_id)
        row.error = message
        row.file_hash = file_hash
        row.file_size = file_size
        row.file_mtime = file_mtime
        return row

    def _reused_row(self, source: dict | ResultRow, path: Path, portable: str, run_id: int, file_hash: str,
                    file_size: int, file_mtime: float) -> ResultRow:
        get = source.get if isinstance(source, dict) else lambda key: getattr(source, key)
        row = self._base_row(path, portable, "ok", run_id)
        row.file_hash = file_hash
        row.file_size = file_size
        row.file_mtime = file_mtime
        row.label = get("label")
        row.p_safe = get("p_safe")
        row.p_suggestive = get("p_suggestive")
        row.p_explicit = get("p_explicit")
        row.confidence = get("confidence")
        row.model_revision = get("model_revision")
        row.device = get("device")
        row.gpu_name = get("gpu_name")
        row.seconds = 0.0
        row.reused = 1
        return row

    def _check_files(self, paths: list[Path], precision: str, force: bool, run_id: int, counters: dict[str, int],
                     pending_rows: list[ResultRow], flush_pending: Callable[[], None],
                     stop_requested: Callable[[], bool], emit: EmitFn) -> list[_QueueEntry]:
        queue: list[_QueueEntry] = []
        by_hash: dict[str, _QueueEntry] = {}
        for index, path in enumerate(paths):
            if stop_requested():
                raise InferenceStopped()
            emit("checking", index, str(path))
            portable = to_portable_path(path)
            try:
                stat = path.stat()
                existing = self.db.find_result(portable, precision)
                unchanged_stat = (existing is not None and existing["status"] == "ok"
                                  and existing["file_size"] == stat.st_size
                                  and existing["file_mtime"] == stat.st_mtime)
                if not force and unchanged_stat:
                    counters["skipped"] += 1
                    continue
                file_hash = sha256_file(path)
            except OSError as e:
                message = f"{type(e).__name__}: {e}"
                logger.warning("Cannot read %s: %s", path, message)
                pending_rows.append(self._error_row(path, portable, run_id, message))
                counters["errors"] += 1
                self._flush_if_full(pending_rows, flush_pending)
                continue
            if not force and existing is not None and existing["status"] == "ok" \
                    and existing["file_hash"] == file_hash:
                self.db.touch_result(portable, precision, stat.st_mtime)
                counters["skipped"] += 1
                continue
            if not force:
                donor = self.db.find_ok_result_by_hash(file_hash, precision)
                if donor is not None:
                    pending_rows.append(self._reused_row(donor, path, portable, run_id, file_hash,
                                                         stat.st_size, stat.st_mtime))
                    counters["reused"] += 1
                    self._flush_if_full(pending_rows, flush_pending)
                    continue
            entry = _QueueEntry(path, portable, file_hash, stat.st_size, stat.st_mtime)
            leader = by_hash.get(file_hash)
            if leader is not None:
                leader.followers.append(entry)
                continue
            by_hash[file_hash] = entry
            queue.append(entry)
        return queue

    @staticmethod
    def _flush_if_full(pending_rows: list[ResultRow], flush_pending: Callable[[], None]) -> None:
        if len(pending_rows) >= REUSE_FLUSH_SIZE:
            flush_pending()

    def _result_row(self, entry: _QueueEntry, result: ImageResult, run_id: int) -> ResultRow:
        if not result.ok:
            logger.warning("Cannot classify %s: %s", entry.path, result.error)
            return self._error_row(entry.path, entry.portable, run_id, result.error or "Unknown error",
                                   entry.file_hash, entry.file_size, entry.file_mtime)
        probabilities = result.probabilities or {}
        row = self._base_row(entry.path, entry.portable, "ok", run_id)
        row.file_hash = entry.file_hash
        row.file_size = entry.file_size
        row.file_mtime = entry.file_mtime
        row.label = result.label
        row.p_safe = probabilities.get("safe")
        row.p_suggestive = probabilities.get("suggestive")
        row.p_explicit = probabilities.get("explicit")
        row.confidence = result.confidence
        row.seconds = result.seconds
        return row

    def _infer(self, queue: list[_QueueEntry], batch_size: int, run_id: int,
               counters: dict[str, int], emit: EmitFn, stop_requested: Callable[[], bool]) -> None:
        if not queue:
            return
        entries = {entry.path: entry for entry in queue}
        inference_started = time.perf_counter()
        done = 0
        for report in self.engine.classify([entry.path for entry in queue], batch_size, stop_requested):
            rows: list[ResultRow] = []
            batch_processed = batch_reused = batch_errors = 0
            last_file = ""
            for result in report.results:
                entry = entries[Path(result.path)]
                last_file = str(entry.path)
                row = self._result_row(entry, result, run_id)
                rows.append(row)
                if result.ok:
                    batch_processed += 1
                else:
                    batch_errors += 1
                for follower in entry.followers:
                    if result.ok:
                        rows.append(self._reused_row(row, follower.path, follower.portable, run_id,
                                                     follower.file_hash, follower.file_size, follower.file_mtime))
                        batch_reused += 1
                    else:
                        rows.append(self._error_row(follower.path, follower.portable, run_id,
                                                    result.error or "Unknown error", follower.file_hash,
                                                    follower.file_size, follower.file_mtime))
                        batch_errors += 1
            self.db.save_results(rows)
            counters["processed"] += batch_processed
            counters["reused"] += batch_reused
            counters["errors"] += batch_errors
            self.db.update_run(run_id, **counters)
            done += len(report.results)
            elapsed = time.perf_counter() - inference_started
            rate = counters["processed"] / elapsed if elapsed > 0 else None
            emit("inference", done, last_file, rate)
