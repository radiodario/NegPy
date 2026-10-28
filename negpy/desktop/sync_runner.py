"""Runs the device sync beside the desktop app: a worker thread that pulls and pushes
edits on a timer, writes the open roll's record and the proxies a phone renders from, and
tells the controller when the open frame's edit changed under it."""

from __future__ import annotations

import os
from typing import Any, Optional

from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot

from negpy.domain.models import WorkspaceConfig
from negpy.kernel.system.logging import get_logger
from negpy.services.assets import rolls
from negpy.services.sync.service import SYNC_FOLDER_KEY, SYNC_PROXIES_KEY, SyncReport, SyncService, default_sync_folder
from negpy.services.sync.store import SyncStore

logger = get_logger(__name__)

SYNC_ENABLED_KEY = "sync_enabled"
SYNC_INTERVAL_MS = 60_000
FIRST_SYNC_DELAY_MS = 5_000


def sync_settings(repo: Any) -> tuple[bool, str, bool]:
    """(enabled, folder, proxies) as Preferences stores them; the folder has a default."""
    folder = str(repo.get_global_setting(SYNC_FOLDER_KEY, "") or "") or default_sync_folder()
    return bool(repo.get_global_setting(SYNC_ENABLED_KEY, False)), folder, bool(repo.get_global_setting(SYNC_PROXIES_KEY, True))


class SyncWorker(QObject):
    """One sync pass per `run` call, on its own thread. Decodes with a CPU processor of its own."""

    finished = pyqtSignal(object)

    def __init__(self, repo: Any):
        super().__init__()
        self.repo = repo
        self._processor = None

    @pyqtSlot(str, bool, list, object)
    def run(self, folder: str, proxies: bool, frames: list[dict], roll: Optional[dict]) -> None:
        report = SyncReport()
        try:
            service = SyncService(self.repo, SyncStore(folder))
            report = service.sync()
            if roll is not None:
                service.write_roll(roll["id"], roll["name"], frames)
            if proxies:
                service.ensure_proxies(frames, self._linear_source, report)
        except Exception as exc:
            logger.warning("Sync pass failed: %s", exc)
            report.errors.append(str(exc))
        self.finished.emit(report)

    def _linear_source(self, path: str, file_hash: str, flat: dict) -> Any:
        if self._processor is None:
            from negpy.services.rendering.image_processor import ImageProcessor

            self._processor = ImageProcessor(use_gpu=False)
        params = WorkspaceConfig.from_flat_dict(dict(flat)) if flat else WorkspaceConfig()
        return self._processor.linear_source(path, params, file_hash)


class SyncRunner(QObject):
    """Owns the worker thread and the timer. `controller` supplies the open files and takes
    the reload when a pulled edit is the open frame's."""

    def __init__(self, controller: Any):
        super().__init__()
        self.controller = controller
        self.repo = controller.session.repo
        self._thread = QThread()
        self._worker = SyncWorker(self.repo)
        self._worker.moveToThread(self._thread)
        self._worker.finished.connect(self._on_finished)
        self._busy = False
        self._timer = QTimer(self)
        self._timer.setInterval(SYNC_INTERVAL_MS)
        self._timer.timeout.connect(self.sync_now)

    def start(self) -> None:
        self._thread.start()
        self._timer.start()
        QTimer.singleShot(FIRST_SYNC_DELAY_MS, self.sync_now)

    def stop(self) -> None:
        self._timer.stop()
        self._thread.quit()
        self._thread.wait(5_000)

    def _open_frames(self) -> list[dict]:
        """The open files as the worker needs them: hash, path and the saved edit."""
        frames = []
        for asset in list(self.controller.state.uploaded_files):
            file_hash, path = asset.get("hash"), asset.get("path")
            if not file_hash or not path:
                continue
            config = self.repo.load_file_settings(file_hash)
            frames.append({"hash": file_hash, "path": path, "config": config.to_dict() if config is not None else None})
        return frames

    def _open_roll(self) -> Optional[dict]:
        roll_id = self.controller.state.active_roll_id
        if not roll_id:
            return None
        record = rolls.roll_for_id(self.repo, roll_id)
        if record is None:
            return None
        return {"id": roll_id, "name": str(record.get("name") or os.path.basename(record.get("folder_path") or "") or roll_id)}

    def sync_now(self) -> None:
        enabled, folder, proxies = sync_settings(self.repo)
        if not enabled or self._busy:
            return
        self._busy = True
        frames, roll = self._open_frames(), self._open_roll()
        QTimer.singleShot(0, lambda: self._invoke(folder, proxies, frames, roll))

    def _invoke(self, folder: str, proxies: bool, frames: list[dict], roll: Optional[dict]) -> None:
        from PyQt6.QtCore import QMetaObject, Q_ARG, Qt

        QMetaObject.invokeMethod(
            self._worker,
            "run",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, folder),
            Q_ARG(bool, proxies),
            Q_ARG(list, frames),
            Q_ARG(object, roll),
        )

    def _on_finished(self, report: SyncReport) -> None:
        self._busy = False
        if report.errors:
            logger.warning("Sync: %d error(s): %s", len(report.errors), report.errors[0])
        current = self.controller.state.current_file_hash
        if current and current in report.pulled:
            config = self.repo.load_file_settings(current)
            if config is not None:
                self.controller.apply_config(config)
