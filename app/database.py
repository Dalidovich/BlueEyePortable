from __future__ import annotations

import csv
import os
import sqlite3
from dataclasses import astuple, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

from settings import DB_PATH, RESULTS_CSV_PATH

SCHEMA_VERSION = 1
RUN_STATUSES = ("running", "completed", "stopped", "failed", "interrupted")
LABEL_VALUES = ("safe", "suggestive", "explicit")
RUN_COUNTER_COLUMNS = ("total_found", "processed", "reused", "skipped", "errors")
RESULT_ORDER_COLUMNS = ("processed_at", "label", "confidence", "file_name", "seconds")
BENCHMARK_COLUMNS = (
    "session_id", "created_at", "gpu_name", "device", "precision", "batch_size", "effective_batch_size",
    "images", "total_seconds", "images_per_second", "ms_per_image", "model_images_per_second",
    "model_ms_per_image", "peak_memory_bytes", "torch_version", "cuda_runtime", "warmup_images",
)
CSV_COLUMNS = (
    "path", "file_name", "predicted_class", "safe_probability", "suggestive_probability",
    "explicit_probability", "confidence", "processing_seconds", "status", "error", "precision",
    "device", "gpu_name", "file_hash", "processed_at",
)
CSV_SOURCE_COLUMNS = (
    "path", "file_name", "label", "p_safe", "p_suggestive", "p_explicit", "confidence", "seconds",
    "status", "error", "precision", "device", "gpu_name", "file_hash", "processed_at",
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    started_at TEXT,
    finished_at TEXT,
    status TEXT CHECK(status IN ('running','completed','stopped','failed','interrupted')),
    input_folder TEXT,
    recursive INTEGER,
    device TEXT,
    gpu_name TEXT,
    precision TEXT,
    batch_size INTEGER,
    total_found INTEGER DEFAULT 0,
    processed INTEGER DEFAULT 0,
    reused INTEGER DEFAULT 0,
    skipped INTEGER DEFAULT 0,
    errors INTEGER DEFAULT 0,
    message TEXT
);
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_hash TEXT,
    file_size INTEGER,
    file_mtime REAL,
    label TEXT,
    p_safe REAL,
    p_suggestive REAL,
    p_explicit REAL,
    confidence REAL,
    seconds REAL,
    status TEXT NOT NULL CHECK(status IN ('ok','error')),
    error TEXT,
    precision TEXT NOT NULL,
    device TEXT,
    gpu_name TEXT,
    model_revision TEXT,
    reused INTEGER NOT NULL DEFAULT 0,
    run_id INTEGER REFERENCES runs(id),
    processed_at TEXT NOT NULL,
    UNIQUE(path, precision)
);
CREATE INDEX IF NOT EXISTS idx_results_file_hash ON results(file_hash);
CREATE INDEX IF NOT EXISTS idx_results_label ON results(label);
CREATE TABLE IF NOT EXISTS labels (
    file_hash TEXT PRIMARY KEY,
    label TEXT NOT NULL CHECK(label IN ('safe','suggestive','explicit')),
    path TEXT,
    source TEXT NOT NULL CHECK(source IN ('manual','folder')),
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS benchmarks (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    gpu_name TEXT,
    device TEXT,
    precision TEXT,
    batch_size INTEGER,
    effective_batch_size INTEGER,
    images INTEGER,
    total_seconds REAL,
    images_per_second REAL,
    ms_per_image REAL,
    model_images_per_second REAL,
    model_ms_per_image REAL,
    peak_memory_bytes INTEGER,
    torch_version TEXT,
    cuda_runtime TEXT,
    warmup_images INTEGER
);
"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ResultRow:
    path: str
    file_name: str
    status: str
    precision: str
    file_hash: str | None = None
    file_size: int | None = None
    file_mtime: float | None = None
    label: str | None = None
    p_safe: float | None = None
    p_suggestive: float | None = None
    p_explicit: float | None = None
    confidence: float | None = None
    seconds: float | None = None
    error: str | None = None
    device: str | None = None
    gpu_name: str | None = None
    model_revision: str | None = None
    reused: int = 0
    run_id: int | None = None
    processed_at: str = ""

    def with_timestamp(self) -> "ResultRow":
        if self.processed_at:
            return self
        self.processed_at = utc_now_iso()
        return self


RESULT_COLUMNS = tuple(f.name for f in fields(ResultRow))
UPSERT_SQL = (
    f"INSERT INTO results ({', '.join(RESULT_COLUMNS)}) VALUES ({', '.join('?' for _ in RESULT_COLUMNS)}) "
    "ON CONFLICT(path, precision) DO UPDATE SET "
    + ", ".join(f"{name}=excluded.{name}" for name in RESULT_COLUMNS if name not in ("path", "precision"))
)


class Database:
    def __init__(self, path: Path = DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._ensure_schema()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    def _ensure_schema(self) -> None:
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if version >= SCHEMA_VERSION:
            return
        with self.conn:
            self.conn.executescript(SCHEMA_SQL)
            self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def _fetch_all(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict]:
        return [dict(row) for row in self.conn.execute(sql, params).fetchall()]

    def _fetch_one(self, sql: str, params: tuple[Any, ...] = ()) -> dict | None:
        row = self.conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def mark_interrupted_runs(self) -> int:
        with self.conn:
            cursor = self.conn.execute(
                "UPDATE runs SET status='interrupted', finished_at=? WHERE status='running'", (utc_now_iso(),))
        return cursor.rowcount

    def start_run(self, input_folder: str, recursive: bool, device: str | None, gpu_name: str | None,
                  precision: str, batch_size: int) -> int:
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO runs (started_at, status, input_folder, recursive, device, gpu_name, precision, "
                "batch_size) VALUES (?, 'running', ?, ?, ?, ?, ?, ?)",
                (utc_now_iso(), input_folder, int(recursive), device, gpu_name, precision, batch_size))
        return int(cursor.lastrowid)

    def update_run(self, run_id: int, **counters: int) -> None:
        unknown = set(counters) - set(RUN_COUNTER_COLUMNS)
        if unknown:
            raise ValueError(f"Unknown run counters: {', '.join(sorted(unknown))}")
        if not counters:
            return
        assignments = ", ".join(f"{name}=?" for name in counters)
        with self.conn:
            self.conn.execute(f"UPDATE runs SET {assignments} WHERE id=?", (*counters.values(), run_id))

    def finish_run(self, run_id: int, status: str, message: str | None = None) -> None:
        if status not in RUN_STATUSES or status == "running":
            raise ValueError(f"Invalid final run status: {status!r}")
        with self.conn:
            self.conn.execute("UPDATE runs SET status=?, finished_at=?, message=? WHERE id=?",
                              (status, utc_now_iso(), message, run_id))

    def get_run(self, run_id: int) -> dict | None:
        return self._fetch_one("SELECT * FROM runs WHERE id=?", (run_id,))

    def list_runs(self, limit: int = 50) -> list[dict]:
        return self._fetch_all("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,))

    def find_result(self, path_portable: str, precision: str) -> dict | None:
        return self._fetch_one("SELECT * FROM results WHERE path=? AND precision=?", (path_portable, precision))

    def find_ok_result_by_hash(self, file_hash: str, precision: str) -> dict | None:
        return self._fetch_one(
            "SELECT * FROM results WHERE file_hash=? AND precision=? AND status='ok' ORDER BY id LIMIT 1",
            (file_hash, precision))

    def save_results(self, rows: list[ResultRow]) -> None:
        if not rows:
            return
        with self.conn:
            self.conn.executemany(UPSERT_SQL, [astuple(row.with_timestamp()) for row in rows])

    def touch_result(self, path_portable: str, precision: str, file_mtime: float) -> None:
        with self.conn:
            self.conn.execute("UPDATE results SET file_mtime=? WHERE path=? AND precision=?",
                              (file_mtime, path_portable, precision))

    def query_results(self, label: str | None = None, precision: str | None = None, status: str | None = None,
                      order_by: str = "processed_at", descending: bool = True,
                      limit: int | None = None) -> list[dict]:
        if order_by not in RESULT_ORDER_COLUMNS:
            raise ValueError(f"Unsupported order_by {order_by!r}; expected one of {RESULT_ORDER_COLUMNS}")
        conditions: list[str] = []
        params: list[Any] = []
        for column, value in (("label", label), ("precision", precision), ("status", status)):
            if value is not None:
                conditions.append(f"{column}=?")
                params.append(value)
        sql = "SELECT * FROM results"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += f" ORDER BY {order_by} {'DESC' if descending else 'ASC'}, id {'DESC' if descending else 'ASC'}"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return self._fetch_all(sql, tuple(params))

    def summary(self, precision: str | None = None) -> dict:
        where = " WHERE precision=?" if precision is not None else ""
        params = (precision,) if precision is not None else ()
        row = self.conn.execute(
            "SELECT COUNT(*) AS total, "
            "COALESCE(SUM(status='ok'), 0) AS ok, "
            "COALESCE(SUM(status='ok' AND label='safe'), 0) AS safe, "
            "COALESCE(SUM(status='ok' AND label='suggestive'), 0) AS suggestive, "
            "COALESCE(SUM(status='ok' AND label='explicit'), 0) AS explicit, "
            "COALESCE(SUM(status='error'), 0) AS errors "
            f"FROM results{where}", params).fetchone()
        return {key: int(row[key]) for key in row.keys()}

    def set_label(self, file_hash: str, label: str, path_portable: str | None, source: str = "manual") -> None:
        if label not in LABEL_VALUES:
            raise ValueError(f"Invalid label {label!r}")
        if source not in ("manual", "folder"):
            raise ValueError(f"Invalid label source {source!r}")
        with self.conn:
            self.conn.execute(
                "INSERT INTO labels (file_hash, label, path, source, updated_at) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(file_hash) DO UPDATE SET label=excluded.label, path=excluded.path, "
                "source=excluded.source, updated_at=excluded.updated_at "
                "WHERE ?='manual' OR labels.source='folder'",
                (file_hash, label, path_portable, source, utc_now_iso(), source))

    def remove_label(self, file_hash: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM labels WHERE file_hash=?", (file_hash,))

    def get_labels(self) -> dict[str, dict]:
        return {row["file_hash"]: row for row in self._fetch_all("SELECT * FROM labels")}

    def labeled_pairs(self, precision: str) -> list[dict]:
        return self._fetch_all(
            "SELECT l.file_hash AS file_hash, r.path AS path, l.label AS true_label, "
            "r.label AS predicted_label, r.confidence AS confidence "
            "FROM labels l JOIN results r ON r.file_hash=l.file_hash AND r.status='ok' AND r.precision=? "
            "WHERE r.id = (SELECT MIN(r2.id) FROM results r2 WHERE r2.file_hash=l.file_hash "
            "AND r2.status='ok' AND r2.precision=?) ORDER BY l.file_hash",
            (precision, precision))

    def save_benchmark(self, row: dict) -> None:
        values = tuple(row.get(column) for column in BENCHMARK_COLUMNS)
        placeholders = ", ".join("?" for _ in BENCHMARK_COLUMNS)
        with self.conn:
            self.conn.execute(
                f"INSERT INTO benchmarks ({', '.join(BENCHMARK_COLUMNS)}) VALUES ({placeholders})", values)

    def list_benchmarks(self, session_id: str | None = None) -> list[dict]:
        if session_id is None:
            return self._fetch_all("SELECT * FROM benchmarks ORDER BY id DESC")
        return self._fetch_all("SELECT * FROM benchmarks WHERE session_id=? ORDER BY id", (session_id,))

    def export_results_csv(self, path: Path = RESULTS_CSV_PATH, **query_filters: Any) -> int:
        return export_results_csv(self, path, **query_filters)


def export_results_csv(db: Database, path: Path = RESULTS_CSV_PATH, **query_filters: Any) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = db.query_results(**query_filters)
    temp_path = path.with_name(path.name + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_COLUMNS)
            for row in rows:
                writer.writerow(["" if row[column] is None else row[column] for column in CSV_SOURCE_COLUMNS])
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    return len(rows)
