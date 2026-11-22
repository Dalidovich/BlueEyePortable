from __future__ import annotations

import csv
import json
import os
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from database import Database
from scanner import discover_images, sha256_file
from settings import to_portable_path

LABELS = ("safe", "suggestive", "explicit")


@dataclass
class ClassMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    support: int
    accuracy: float


@dataclass
class EvaluationMetrics:
    total: int
    correct: int
    accuracy: float
    per_class: list[ClassMetrics]
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    confusion: list[list[int]]


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def per_class_accuracy(correct_for_class: int, support: int) -> float:
    return safe_divide(correct_for_class, support)


def compute_metrics(pairs: Iterable[tuple[str, str]]) -> EvaluationMetrics:
    size = len(LABELS)
    index = {label: i for i, label in enumerate(LABELS)}
    confusion = [[0] * size for _ in range(size)]
    for true_label, predicted_label in pairs:
        if true_label in index and predicted_label in index:
            confusion[index[true_label]][index[predicted_label]] += 1
    total = sum(sum(row) for row in confusion)
    correct = sum(confusion[i][i] for i in range(size))
    per_class: list[ClassMetrics] = []
    for i, label in enumerate(LABELS):
        support = sum(confusion[i])
        predicted = sum(confusion[row][i] for row in range(size))
        precision = safe_divide(confusion[i][i], predicted)
        recall = safe_divide(confusion[i][i], support)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        per_class.append(ClassMetrics(label, precision, recall, f1, support,
                                      per_class_accuracy(confusion[i][i], support)))
    return EvaluationMetrics(
        total=total,
        correct=correct,
        accuracy=safe_divide(correct, total),
        per_class=per_class,
        macro_precision=safe_divide(sum(c.precision for c in per_class), size),
        macro_recall=safe_divide(sum(c.recall for c in per_class), size),
        macro_f1=safe_divide(sum(c.f1 for c in per_class), size),
        weighted_f1=safe_divide(sum(c.f1 * c.support for c in per_class), total),
        confusion=confusion,
    )


def folder_label_for(path: Path, root: Path) -> str | None:
    root = Path(root).resolve()
    try:
        relative_parent = Path(path).resolve().parent.relative_to(root)
    except ValueError:
        return None
    for name in reversed(relative_parent.parts):
        if name.lower() in LABELS:
            return name.lower()
    return None


def import_folder_labels(db: Database, root: Path, recursive: bool = True,
                         progress: Callable[[int, int], None] | None = None) -> int:
    images = discover_images(root, recursive)
    imported = 0
    for index, path in enumerate(images):
        label = folder_label_for(path, root)
        if label is not None:
            try:
                file_hash = sha256_file(path)
            except OSError:
                file_hash = None
            if file_hash is not None:
                db.set_label(file_hash, label, to_portable_path(path), source="folder")
                imported += 1
        if progress is not None:
            progress(index + 1, len(images))
    return imported


def export_metrics(metrics: EvaluationMetrics, path: Path, fmt: str) -> None:
    if fmt not in ("csv", "json"):
        raise ValueError(f"Unsupported export format {fmt!r}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    try:
        if fmt == "json":
            temp_path.write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")
        else:
            with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["label", "precision", "recall", "f1", "support", "accuracy"])
                for c in metrics.per_class:
                    writer.writerow([c.label, c.precision, c.recall, c.f1, c.support, c.accuracy])
                writer.writerow(["macro", metrics.macro_precision, metrics.macro_recall, metrics.macro_f1,
                                 metrics.total, metrics.accuracy])
                writer.writerow(["weighted_f1", "", "", metrics.weighted_f1, metrics.total, ""])
                writer.writerow([])
                writer.writerow(["true\\predicted", *LABELS])
                for label, row in zip(LABELS, metrics.confusion):
                    writer.writerow([label, *row])
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
