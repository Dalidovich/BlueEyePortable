from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from classifier import BlueEyeEngine

logger = logging.getLogger("blueeye.app")

EngineLoader = Callable[[str, str], BlueEyeEngine]


def load_engine(device: str, precision: str) -> BlueEyeEngine:
    return BlueEyeEngine.load(device, precision)


class EngineManager:
    def __init__(self, loader: EngineLoader = load_engine):
        self._loader = loader
        self._engine: BlueEyeEngine | None = None
        self._key: tuple[str, str] | None = None
        self._lock = threading.Lock()

    def is_loaded(self, device: str, precision: str) -> bool:
        with self._lock:
            return self._engine is not None and self._key == (device, precision)

    def acquire(self, device: str, precision: str) -> BlueEyeEngine:
        key = (device, precision)
        with self._lock:
            if self._engine is not None and self._key == key:
                return self._engine
            self._close_locked()
            engine = self._loader(device, precision)
            self._engine = engine
            self._key = key
            return engine

    def discard(self) -> None:
        with self._lock:
            self._close_locked()

    def close(self) -> None:
        self.discard()

    def _close_locked(self) -> None:
        engine = self._engine
        self._engine = None
        self._key = None
        if engine is None:
            return
        try:
            engine.close()
        except Exception:
            logger.exception("Failed to release the inference engine")
