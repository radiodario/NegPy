"""The sync folder: one record per frame, so a synced drive moves rows, never a database file.

The folder lives in iCloud Drive (or any synced drive). A record is a JSON file named by the
frame's content hash, so every device writes the same name for the same scan and the newest
`modified_at` wins. A proxy beside it is the frame's baked, scene-linear source at a bounded
size, as a 16-bit PNG, which a phone renders without the scanner's RAW decoder.
"""

from __future__ import annotations

import glob
import json
import os
import platform
import re
import time
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import numpy as np

from negpy.kernel.system.logging import get_logger
from negpy.kernel.image.logic import _to_uint16_jit
from negpy.services.export.encoders import encode_png

logger = get_logger(__name__)

RECORD_VERSION = 1
EDITS_DIR = "edits"
PROXIES_DIR = "proxies"
ROLLS_DIR = "rolls"
PROXY_LONG_EDGE = 2048
# A sync service names the conflict copy it makes when two devices write one file at once.
_CONFLICT_SUFFIX = re.compile(r"[ _](\d+|conflicted copy.*)$")


@dataclass(frozen=True)
class EditRecord:
    hash: str
    file_name: str
    modified_at: float
    settings: dict[str, Any]
    mark: Optional[str] = None
    device: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "version": RECORD_VERSION,
            "hash": self.hash,
            "file_name": self.file_name,
            "modified_at": self.modified_at,
            "device": self.device,
            "mark": self.mark,
            "settings": self.settings,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "EditRecord":
        return cls(
            hash=str(data["hash"]),
            file_name=str(data.get("file_name", "")),
            modified_at=float(data.get("modified_at", 0.0)),
            settings=dict(data.get("settings", {})),
            mark=data.get("mark"),
            device=str(data.get("device", "")),
        )


def device_name() -> str:
    return platform.node() or "unknown"


def _hash_stem(file_hash: str) -> str:
    """A record file name: the hash, with the roll fork and half-frame suffixes made file-safe."""
    return file_hash.replace("#", "+").replace(":", "-").replace("/", "-")


def _write_atomic(path: str, data: bytes) -> None:
    tmp = f"{path}.part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


class SyncStore:
    """The folder's layout and record I/O. Nothing here reads the database."""

    def __init__(self, folder: str):
        self.folder = folder

    @property
    def edits_dir(self) -> str:
        return os.path.join(self.folder, EDITS_DIR)

    @property
    def proxies_dir(self) -> str:
        return os.path.join(self.folder, PROXIES_DIR)

    @property
    def rolls_dir(self) -> str:
        return os.path.join(self.folder, ROLLS_DIR)

    def ensure(self) -> None:
        for d in (self.edits_dir, self.proxies_dir, self.rolls_dir):
            os.makedirs(d, exist_ok=True)

    def record_path(self, file_hash: str) -> str:
        return os.path.join(self.edits_dir, f"{_hash_stem(file_hash)}.json")

    def proxy_path(self, file_hash: str) -> str:
        return os.path.join(self.proxies_dir, f"{_hash_stem(file_hash)}.png")

    def roll_path(self, roll_id: str) -> str:
        return os.path.join(self.rolls_dir, f"{roll_id}.json")

    def write_edit(self, record: EditRecord) -> str:
        self.ensure()
        path = self.record_path(record.hash)
        _write_atomic(path, json.dumps(record.to_json(), sort_keys=True).encode("utf-8"))
        return path

    def _record_copies(self, file_hash: str) -> list[str]:
        """The record and any conflict copy of it, which a sync service names with a suffix."""
        stem = _hash_stem(file_hash)
        copies = glob.glob(os.path.join(glob.escape(self.edits_dir), glob.escape(stem) + "*.json"))
        return [c for c in copies if os.path.basename(c)[: len(stem)] == stem]

    def read_edit(self, file_hash: str) -> Optional[EditRecord]:
        """The newest record for the hash across its conflict copies; None when there is none."""
        newest: Optional[EditRecord] = None
        for path in self._record_copies(file_hash):
            record = _read_record(path)
            if record is not None and (newest is None or record.modified_at > newest.modified_at):
                newest = record
        return newest

    def drop_conflict_copies(self, file_hash: str) -> int:
        """Removes every conflict copy whose content the record already holds."""
        canonical = self.record_path(file_hash)
        newest = self.read_edit(file_hash)
        if newest is None:
            return 0
        if not os.path.exists(canonical) or (_read_record(canonical) or newest).modified_at < newest.modified_at:
            self.write_edit(newest)
        removed = 0
        for path in self._record_copies(file_hash):
            if path != canonical:
                os.remove(path)
                removed += 1
        return removed

    def iter_edits(self) -> Iterator[EditRecord]:
        """Every record in the folder, newest copy per hash."""
        if not os.path.isdir(self.edits_dir):
            return
        seen: set[str] = set()
        for path in sorted(os.listdir(self.edits_dir)):
            if not path.endswith(".json"):
                continue
            record = _read_record(os.path.join(self.edits_dir, path))
            if record is None or record.hash in seen:
                continue
            seen.add(record.hash)
            newest = self.read_edit(record.hash)
            if newest is not None:
                yield newest

    def has_proxy(self, file_hash: str) -> bool:
        return os.path.exists(self.proxy_path(file_hash))

    def write_proxy(self, file_hash: str, linear: np.ndarray, long_edge: int = PROXY_LONG_EDGE) -> str:
        """Writes the frame's scene-linear source, scaled so its long edge is at most `long_edge`,
        as a 16-bit PNG. Values above 1 (a flat-field gain) clip: the proxy is a preview source."""
        import negpy_core

        self.ensure()
        img = np.ascontiguousarray(linear, dtype=np.float32)
        h, w = img.shape[:2]
        scale = min(1.0, long_edge / float(max(h, w)))
        if scale < 1.0:
            img = negpy_core.resize_area(img, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))))
        png = encode_png(_to_uint16_jit(img))
        path = self.proxy_path(file_hash)
        _write_atomic(path, png)
        return path

    def write_roll(self, roll_id: str, name: str, frames: list[dict[str, Any]], modified_at: Optional[float] = None) -> str:
        """A roll as the phone lists it: its name and its frames, each a hash and a file name."""
        self.ensure()
        doc = {
            "version": RECORD_VERSION,
            "id": roll_id,
            "name": name,
            "modified_at": time.time() if modified_at is None else modified_at,
            "device": device_name(),
            "frames": frames,
        }
        path = self.roll_path(roll_id)
        _write_atomic(path, json.dumps(doc, sort_keys=True).encode("utf-8"))
        return path

    def read_rolls(self) -> list[dict[str, Any]]:
        if not os.path.isdir(self.rolls_dir):
            return []
        rolls = []
        for name in sorted(os.listdir(self.rolls_dir)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(self.rolls_dir, name), "r", encoding="utf-8") as f:
                    rolls.append(json.load(f))
            except (OSError, ValueError) as exc:
                logger.warning("Unreadable roll record %s: %s", name, exc)
        return rolls


def _read_record(path: str) -> Optional[EditRecord]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return EditRecord.from_json(json.load(f))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Unreadable sync record %s: %s", path, exc)
        return None
