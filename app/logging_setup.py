from __future__ import annotations

import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from types import TracebackType

from settings import LOG_DIR

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
MAX_LOG_BYTES = 5 * 1024 * 1024
LOG_BACKUPS = 3
ROOT_LOGGER_NAME = "blueeye"
INFERENCE_LOGGER_NAME = "blueeye.inference"
APP_LOGGER_NAME = "blueeye.app"

_state = {"configured": False, "console": False, "hooks": False}


def _file_handler(name: str, level: int) -> RotatingFileHandler:
    handler = RotatingFileHandler(LOG_DIR / name, maxBytes=MAX_LOG_BYTES, backupCount=LOG_BACKUPS,
                                  encoding="utf-8", delay=True)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    return handler


def _install_excepthooks() -> None:
    if _state["hooks"]:
        return
    _state["hooks"] = True
    app_logger = logging.getLogger(APP_LOGGER_NAME)
    previous_sys_hook = sys.excepthook
    previous_thread_hook = threading.excepthook

    def log_uncaught(exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None) -> None:
        app_logger.critical("Uncaught exception", exc_info=(exc_type, exc, tb))

    def sys_hook(exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None) -> None:
        log_uncaught(exc_type, exc, tb)
        previous_sys_hook(exc_type, exc, tb)

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is not SystemExit:
            log_uncaught(args.exc_type, args.exc_value, args.exc_traceback)
        previous_thread_hook(args)

    sys.excepthook = sys_hook
    threading.excepthook = thread_hook


def configure_logging(console: bool = False) -> None:
    root = logging.getLogger(ROOT_LOGGER_NAME)
    if not _state["configured"]:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        root.setLevel(logging.INFO)
        root.addHandler(_file_handler("app.log", logging.INFO))
        root.addHandler(_file_handler("errors.log", logging.WARNING))
        logging.getLogger(INFERENCE_LOGGER_NAME).addHandler(_file_handler("inference.log", logging.INFO))
        _state["configured"] = True
        _install_excepthooks()
    if console and not _state["console"]:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(handler)
        _state["console"] = True
