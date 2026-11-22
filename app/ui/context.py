from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from database import Database
from settings import Settings
from ui.device_state import DeviceState
from ui.dialogs import DialogService
from ui.engine_manager import EngineManager
from ui.job_controller import JobController

logger = logging.getLogger("blueeye.app")


class AppEvents(QObject):
    results_changed = Signal()
    status_message = Signal(str)
    model_status_changed = Signal(object)


@dataclass
class AppContext:
    db_path: Path
    db: Database
    settings: Settings
    device_state: DeviceState
    jobs: JobController
    engine_manager: EngineManager
    dialogs: DialogService
    events: AppEvents

    def save_settings(self) -> None:
        try:
            self.settings.save()
        except OSError:
            logger.exception("Cannot save settings")

    def status(self, message: str) -> None:
        self.events.status_message.emit(message)
