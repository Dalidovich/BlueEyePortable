from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from PIL import Image

from classifier import BatchReport, ImageResult, InferenceStopped

LABELS = ("safe", "suggestive", "explicit")


def probabilities_for(data: bytes) -> dict[str, float]:
    digest = hashlib.sha256(data).digest()
    raw = [digest[0] + 1, digest[1] + 1, digest[2] + 1]
    total = sum(raw)
    return {label: value / total for label, value in zip(LABELS, raw)}


def write_image(path: Path, seed: int, size: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (size, size), (seed % 256, (seed * 7) % 256, (seed * 13) % 256))
    image.putpixel((0, 0), (seed % 251, 1, 2))
    image.save(path)


def write_corrupt(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"this is not an image")


class FakeEngine:
    def __init__(self, precision: str = "fp32", device: str = "cpu"):
        self.precision = precision
        self.device = device
        self.labels = list(LABELS)
        self.max_batch_size: int | None = None
        self.classified_paths: list[Path] = []
        self.stop_after_batches: int | None = None
        self.batches_yielded = 0

    def effective_batch_size(self, requested: int) -> int:
        if self.max_batch_size is None:
            return max(1, requested)
        return max(1, min(requested, self.max_batch_size))

    def reset_peak_memory(self) -> None:
        pass

    def peak_memory_bytes(self) -> int | None:
        return None

    def synchronize(self) -> None:
        pass

    def classify(self, paths: Iterable[Path], batch_size: int,
                 should_stop: Callable[[], bool] | None = None) -> Iterator[BatchReport]:
        items = list(paths)
        for start in range(0, len(items), batch_size):
            if should_stop is not None and should_stop():
                raise InferenceStopped()
            chunk = items[start:start + batch_size]
            results = [self._classify_one(Path(p)) for p in chunk]
            self.classified_paths.extend(Path(p) for p in chunk)
            self.batches_yielded += 1
            yield BatchReport(results, 0.001, 0.002, len(chunk))

    def _classify_one(self, path: Path) -> ImageResult:
        try:
            with Image.open(path) as image:
                image.load()
            probabilities = probabilities_for(path.read_bytes())
        except Exception as e:
            return ImageResult(path, None, None, 0.001, f"{type(e).__name__}: {e}")
        label = max(probabilities, key=probabilities.get)
        return ImageResult(path, label, probabilities, 0.01)
