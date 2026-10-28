"""Moves edits between the database and the sync folder, newest side wins.

`push` writes every row saved after its record; `pull` reads every record saved after its
row. A pulled row keeps the record's time, so the next push does not echo it. The service
holds no state between calls: the times on both sides are the state.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import numpy as np

from negpy.kernel.system.logging import get_logger
from negpy.services.sync.store import EditRecord, SyncStore, device_name

logger = get_logger(__name__)

SYNC_FOLDER_KEY = "sync_folder"
SYNC_PROXIES_KEY = "sync_proxies"

# The proxy source: the frame's baked, scene-linear buffer for (path, hash, flat config).
ProxySource = Callable[[str, str, dict[str, Any]], Optional[np.ndarray]]


def default_sync_folder() -> str:
    """iCloud Drive's NegPy folder on a Mac; a NegPy folder in the home directory elsewhere."""
    home = os.path.expanduser("~")
    icloud = os.path.join(home, "Library", "Mobile Documents", "com~apple~CloudDocs")
    if os.path.isdir(icloud):
        return os.path.join(icloud, "NegPy")
    return os.path.join(home, "NegPy Sync")


@dataclass
class SyncReport:
    pushed: list[str] = field(default_factory=list)
    pulled: list[str] = field(default_factory=list)
    proxies: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.pushed or self.pulled or self.proxies)


class SyncService:
    def __init__(self, repo: Any, store: SyncStore):
        self.repo = repo
        self.store = store

    def push(self, report: Optional[SyncReport] = None) -> SyncReport:
        """Writes every edit the database saved after the folder's record of it."""
        report = report or SyncReport()
        marks = self.repo.load_file_marks()
        for file_hash, settings_json, file_path, modified_at in self.repo.iter_file_settings():
            if modified_at is None:
                continue
            existing = self.store.read_edit(file_hash)
            if existing is not None and existing.modified_at >= modified_at:
                continue
            try:
                import json

                record = EditRecord(
                    hash=file_hash,
                    file_name=os.path.basename(file_path),
                    modified_at=modified_at,
                    settings=json.loads(settings_json),
                    mark=marks.get(file_hash),
                    device=device_name(),
                )
                self.store.write_edit(record)
                report.pushed.append(file_hash)
            except (OSError, ValueError) as exc:
                report.errors.append(f"{file_hash}: {exc}")
        return report

    def pull(self, report: Optional[SyncReport] = None) -> SyncReport:
        """Reads every record saved after the database's row of it. Returns the hashes changed."""
        report = report or SyncReport()
        for record in self.store.iter_edits():
            local = self.repo.file_settings_modified_at(record.hash)
            if local is not None and local >= record.modified_at:
                continue
            try:
                import json

                path = self.repo.path_for_file_hash(record.hash) or ""
                self.repo.save_file_settings_json(record.hash, json.dumps(record.settings), path, record.modified_at)
                self.repo.save_file_mark(record.hash, record.mark, path)
                report.pulled.append(record.hash)
            except (OSError, ValueError) as exc:
                report.errors.append(f"{record.hash}: {exc}")
        return report

    def sync(self) -> SyncReport:
        """Pull, then push: a record newer than its row lands first, so the push never overwrites it."""
        report = self.pull()
        return self.push(report)

    def ensure_proxies(self, frames: Iterable[dict[str, Any]], source: ProxySource, report: Optional[SyncReport] = None) -> SyncReport:
        """Writes a proxy for every frame (a dict with `hash`, `path` and `config`) that has none."""
        report = report or SyncReport()
        for frame in frames:
            file_hash = frame["hash"]
            if self.store.has_proxy(file_hash):
                continue
            try:
                linear = source(frame["path"], file_hash, frame.get("config") or {})
            except Exception as exc:  # a decode failure of one frame must not stop the others
                report.errors.append(f"{file_hash}: {exc}")
                continue
            if linear is None:
                continue
            self.store.write_proxy(file_hash, linear)
            report.proxies.append(file_hash)
        return report

    def write_roll(self, roll_id: str, name: str, frames: Iterable[dict[str, Any]]) -> None:
        self.store.write_roll(roll_id, name, [{"hash": f["hash"], "file_name": os.path.basename(f["path"])} for f in frames])
