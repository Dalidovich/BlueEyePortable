from __future__ import annotations

import csv
import json
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path

from classifier import InferenceStopped
from database import Database, utc_now_iso

MIN_WARMUP_IMAGES = 2


@dataclass
class BenchmarkResult:
    session_id: str
    created_at: str
    gpu_name: str | None
    device: str
    precision: str
    batch_size: int
    effective_batch_size: int
    images: int
    total_seconds: float
    images_per_second: float
    ms_per_image: float
    model_images_per_second: float
    model_ms_per_image: float
    peak_memory_bytes: int | None
    torch_version: str
    cuda_runtime: str | None
    warmup_images: int


def parse_batch_sizes(text: str) -> list[int]:
    sizes: list[int] = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            raise ValueError(f"Empty batch size in {text!r}")
        try:
            value = int(token)
        except ValueError:
            raise ValueError(f"Invalid batch size {token!r}") from None
        if value <= 0:
            raise ValueError(f"Batch size must be positive, got {value}")
        if value not in sizes:
            sizes.append(value)
    return sizes


def _per_second(count: int, seconds: float) -> float:
    return count / seconds if seconds > 0 else 0.0


def _milliseconds_per_item(count: int, seconds: float) -> float:
    return seconds * 1000.0 / count if count > 0 else 0.0


def _consume(engine, paths: list[Path], batch_size: int, should_stop: Callable[[], bool] | None) -> None:
    for _report in engine.classify(paths, batch_size, should_stop):
        pass


def run_benchmark(engine, paths: list[Path], batch_sizes: list[int], gpu_name: str | None,
                  image_limit: int = 64, warmup_images: int | None = None,
                  progress: Callable[[str, int, int], None] | None = None,
                  should_stop: Callable[[], bool] | None = None,
                  db: Database | None = None) -> list[BenchmarkResult]:
    import torch

    subset = [Path(p) for p in paths][:max(1, image_limit)]
    session_id = uuid.uuid4().hex
    results: list[BenchmarkResult] = []
    total_images = len(subset)
    try:
        for batch_size in batch_sizes:
            engine.max_batch_size = None
            warmup_count = min(total_images, warmup_images if warmup_images is not None
                               else max(MIN_WARMUP_IMAGES, batch_size))
            if progress is not None:
                progress(f"batch {batch_size}: warmup", 0, total_images)
            _consume(engine, subset[:warmup_count], batch_size, should_stop)
            engine.synchronize()
            engine.reset_peak_memory()

            images = 0
            model_seconds = 0.0
            done = 0
            started = time.perf_counter()
            for report in engine.classify(subset, batch_size, should_stop):
                images += sum(1 for r in report.results if r.ok)
                model_seconds += report.inference_seconds
                done += len(report.results)
                if progress is not None:
                    progress(f"batch {batch_size}: timed run", done, total_images)
            engine.synchronize()
            total_seconds = time.perf_counter() - started

            result = BenchmarkResult(
                session_id=session_id,
                created_at=utc_now_iso(),
                gpu_name=gpu_name,
                device=engine.device,
                precision=engine.precision,
                batch_size=batch_size,
                effective_batch_size=engine.effective_batch_size(batch_size),
                images=images,
                total_seconds=total_seconds,
                images_per_second=_per_second(images, total_seconds),
                ms_per_image=_milliseconds_per_item(images, total_seconds),
                model_images_per_second=_per_second(images, model_seconds),
                model_ms_per_image=_milliseconds_per_item(images, model_seconds),
                peak_memory_bytes=engine.peak_memory_bytes(),
                torch_version=torch.__version__,
                cuda_runtime=torch.version.cuda,
                warmup_images=warmup_count,
            )
            results.append(result)
            if db is not None:
                db.save_benchmark(asdict(result))
    except InferenceStopped:
        pass
    return results


def _as_dicts(results: list[BenchmarkResult] | list[dict]) -> list[dict]:
    return [asdict(r) if isinstance(r, BenchmarkResult) else dict(r) for r in results]


def export_benchmarks(results: list[BenchmarkResult] | list[dict], path: Path, fmt: str) -> None:
    if fmt not in ("csv", "json"):
        raise ValueError(f"Unsupported export format {fmt!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _as_dicts(results)
    temp_path = path.with_name(path.name + ".tmp")
    try:
        if fmt == "csv":
            columns = [f.name for f in fields(BenchmarkResult)]
            with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
        else:
            payload = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "results": rows}
            temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
