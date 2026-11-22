from __future__ import annotations

import logging
import sys

from settings import DB_PATH, ensure_directories, isolate_huggingface_environment

logger = logging.getLogger("blueeye.app")


def show_fatal_error(message: str) -> None:
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
    except Exception:
        return
    if QApplication.instance() is not None:
        QMessageBox.critical(None, "BlueEye Portable Tester", message)


def main(argv: list[str] | None = None) -> int:
    isolate_huggingface_environment()
    try:
        ensure_directories()
        from logging_setup import configure_logging

        configure_logging()
    except Exception as error:
        print(f"Startup failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    application = None
    try:
        from PySide6.QtWidgets import QApplication

        application = QApplication.instance() or QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
        application.setApplicationName("BlueEye Portable Tester")
        from database import Database

        with Database(DB_PATH) as db:
            interrupted = db.mark_interrupted_runs()
        if interrupted:
            logger.warning("Marked %d unfinished run(s) as interrupted", interrupted)
        from ui.main_window import MainWindow

        window = MainWindow(DB_PATH)
        window.show()
        logger.info("Application started")
        return int(application.exec())
    except Exception as error:
        logger.critical("Fatal startup error", exc_info=True)
        if application is not None:
            show_fatal_error(f"The application could not start:\n\n{type(error).__name__}: {error}\n\n"
                             "See output\\logs\\errors.log for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
