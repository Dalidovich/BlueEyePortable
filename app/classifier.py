from __future__ import annotations

import logging
import os
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

from model_manager import check_model
from modeling import BlueEyeClassifier, normalize, preprocess_uint8
from settings import MODEL_DIR, PRECISIONS

logger = logging.getLogger("blueeye.inference")

SUPPORTED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})
AUTOCAST_DTYPES: dict[str, torch.dtype | None] = {"fp32": None, "bf16": torch.bfloat16}
MAX_PREPROCESS_WORKERS = 8


class ModelNotReadyError(Exception):
    pass


class InferenceStopped(Exception):
    pass


@dataclass
class ImageResult:
    path: Path
    label: str | None
    probabilities: dict[str, float] | None
    seconds: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def confidence(self) -> float | None:
        return max(self.probabilities.values()) if self.probabilities else None


@dataclass
class BatchReport:
    results: list[ImageResult]
    preprocess_seconds: float
    inference_seconds: float
    batch_size: int


@dataclass
class _Prepared:
    path: Path
    array: np.ndarray | None
    seconds: float
    error: str | None


def is_out_of_memory(error: BaseException) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def default_worker_count() -> int:
    return max(1, min(MAX_PREPROCESS_WORKERS, (os.cpu_count() or 2) - 1))


class BlueEyeEngine:
    def __init__(self, model: BlueEyeClassifier, device: str, precision: str):
        self.model = model
        self.device = device
        self.precision = precision
        self.autocast_dtype = AUTOCAST_DTYPES[precision]
        self.labels: list[str] = model.labels
        self.max_batch_size: int | None = None

    @classmethod
    def load(cls, device: str, precision: str = "fp32", model_dir: Path = MODEL_DIR) -> "BlueEyeEngine":
        if precision not in PRECISIONS:
            raise ValueError(f"Unsupported precision {precision!r}; expected one of {PRECISIONS}")
        status = check_model(full_hash=False)
        if not status.present:
            raise ModelNotReadyError(status.message)
        started = time.perf_counter()
        model = BlueEyeClassifier.from_pretrained(str(model_dir), device=device)
        logger.info("Model loaded on %s (%s) in %.1f s", device, precision, time.perf_counter() - started)
        return cls(model, device, precision)

    def close(self) -> None:
        self.model.to("cpu")
        del self.model
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

    def preprocess(self, path: Path) -> np.ndarray:
        return preprocess_uint8(str(path), self.model.config)

    def _prepare(self, path: Path) -> _Prepared:
        started = time.perf_counter()
        try:
            array = self.preprocess(path)
            return _Prepared(path, array, time.perf_counter() - started, None)
        except Exception as e:
            return _Prepared(path, None, time.perf_counter() - started, f"{type(e).__name__}: {e}")

    def _run_model(self, arrays: list[np.ndarray]) -> np.ndarray:
        x = normalize(torch.from_numpy(np.stack(arrays)).to(self.device), self.model.config)
        probabilities = self.model.predict_proba(x, autocast_dtype=self.autocast_dtype)
        return probabilities.cpu().numpy()

    def predict(self, arrays: list[np.ndarray]) -> np.ndarray:
        try:
            return self._run_model(arrays)
        except Exception as e:
            if not is_out_of_memory(e) or not self.device.startswith("cuda"):
                raise
        torch.cuda.empty_cache()
        if len(arrays) == 1:
            raise RuntimeError("CUDA out of memory even with batch size 1")
        half = len(arrays) // 2
        self.max_batch_size = half if self.max_batch_size is None else min(self.max_batch_size, half)
        logger.warning("CUDA out of memory with batch size %d; retrying as batches of %d", len(arrays), half)
        return np.concatenate([self.predict(arrays[:half]), self.predict(arrays[half:])])

    def effective_batch_size(self, requested: int) -> int:
        if self.max_batch_size is None:
            return max(1, requested)
        return max(1, min(requested, self.max_batch_size))

    def _results_for(self, prepared: list[_Prepared], probabilities: np.ndarray | None,
                     inference_seconds: float) -> list[ImageResult]:
        valid = [item for item in prepared if item.error is None]
        share = inference_seconds / len(valid) if valid else 0.0
        results: list[ImageResult] = []
        rows = iter(probabilities if probabilities is not None else [])
        for item in prepared:
            if item.error is not None:
                results.append(ImageResult(item.path, None, None, item.seconds, item.error))
                continue
            row = next(rows)
            probs = {label: float(row[k]) for k, label in enumerate(self.labels)}
            label = self.labels[int(np.argmax(row))]
            results.append(ImageResult(item.path, label, probs, item.seconds + share))
        return results

    def classify(self, paths: Iterable[Path], batch_size: int,
                 should_stop: Callable[[], bool] | None = None,
                 workers: int | None = None) -> Iterator[BatchReport]:
        should_stop = should_stop or (lambda: False)
        pending: deque[Future[_Prepared]] = deque()
        source = iter(paths)
        with ThreadPoolExecutor(max_workers=workers or default_worker_count(),
                                thread_name_prefix="preprocess") as pool:
            def fill(target: int) -> None:
                while len(pending) < target:
                    path = next(source, None)
                    if path is None:
                        return
                    pending.append(pool.submit(self._prepare, Path(path)))

            try:
                while True:
                    size = self.effective_batch_size(batch_size)
                    fill(size * 2)
                    if not pending:
                        return
                    if should_stop():
                        raise InferenceStopped()
                    wait_started = time.perf_counter()
                    prepared = [pending.popleft().result() for _ in range(min(size, len(pending)))]
                    preprocess_wait = time.perf_counter() - wait_started
                    fill(size * 2)
                    arrays = [item.array for item in prepared if item.array is not None]
                    probabilities = None
                    inference_seconds = 0.0
                    if arrays:
                        started = time.perf_counter()
                        probabilities = self.predict(arrays)
                        inference_seconds = time.perf_counter() - started
                    for item in prepared:
                        if item.error is not None:
                            logger.warning("Cannot read %s: %s", item.path, item.error)
                    yield BatchReport(self._results_for(prepared, probabilities, inference_seconds),
                                      preprocess_wait, inference_seconds, len(prepared))
            finally:
                for future in pending:
                    future.cancel()

    def reset_peak_memory(self) -> None:
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)

    def peak_memory_bytes(self) -> int | None:
        if self.device.startswith("cuda"):
            return int(torch.cuda.max_memory_allocated(self.device))
        return None

    def synchronize(self) -> None:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
