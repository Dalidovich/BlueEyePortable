from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
MODEL_DIR = ROOT_DIR / "model"
RUNTIME_DIR = ROOT_DIR / "runtime"
HF_HOME_DIR = RUNTIME_DIR / "hf_home"
INPUT_DIR = ROOT_DIR / "input"
OUTPUT_DIR = ROOT_DIR / "output"
LOG_DIR = OUTPUT_DIR / "logs"
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "results.db"
SETTINGS_PATH = DATA_DIR / "settings.json"
RESULTS_CSV_PATH = OUTPUT_DIR / "results.csv"

PRECISIONS = ("fp32", "bf16")
DEFAULT_BATCH_SIZE = 4


def isolate_huggingface_environment() -> None:
    os.environ["HF_HOME"] = str(HF_HOME_DIR)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ.pop("HF_HUB_CACHE", None)
    os.environ.pop("HUGGINGFACE_HUB_CACHE", None)
    os.environ.pop("TRANSFORMERS_CACHE", None)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def ensure_directories() -> None:
    for directory in (MODEL_DIR, INPUT_DIR, OUTPUT_DIR, LOG_DIR, DATA_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def to_portable_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(resolved)


def from_portable_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


@dataclass
class Settings:
    input_folder: str = "input"
    recursive: bool = True
    batch_size: int = DEFAULT_BATCH_SIZE
    precision: str = "fp32"
    device: str = "auto"
    benchmark_batch_sizes: str = "1,2,4,8"
    benchmark_image_limit: int = 64

    @classmethod
    def load(cls) -> "Settings":
        if not SETTINGS_PATH.exists():
            return cls()
        try:
            raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        known = {f.name for f in fields(cls)}
        settings = cls(**{k: v for k, v in raw.items() if k in known})
        if settings.precision not in PRECISIONS:
            settings.precision = "fp32"
        settings.batch_size = max(1, int(settings.batch_size))
        return settings

    def save(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = SETTINGS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        os.replace(tmp, SETTINGS_PATH)

    @property
    def input_path(self) -> Path:
        return from_portable_path(self.input_folder)
