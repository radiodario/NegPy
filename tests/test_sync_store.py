import json
import os
import time

import numpy as np

from negpy.domain.models import WorkspaceConfig
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.services.sync.service import SyncService
from negpy.services.sync.store import EditRecord, SyncStore


def _repo(tmp_path, name):
    repo = StorageRepository(str(tmp_path / f"{name}_edits.db"), str(tmp_path / f"{name}_settings.db"))
    repo.initialize()
    return repo


def test_a_saved_edit_carries_its_time(tmp_path):
    repo = _repo(tmp_path, "a")
    before = time.time()
    repo.save_file_settings("h1", WorkspaceConfig(), file_path="/scans/one.tif")
    assert repo.file_settings_modified_at("h1") >= before
    assert repo.file_settings_modified_at("missing") is None


def test_records_round_trip_and_the_newest_copy_wins(tmp_path):
    store = SyncStore(str(tmp_path / "sync"))
    old = EditRecord("h1", "one.tif", 10.0, {"density": 1.0})
    store.write_edit(old)
    # A conflict copy, as a sync service names one.
    copy = os.path.join(store.edits_dir, "h1 2.json")
    with open(copy, "w", encoding="utf-8") as f:
        json.dump(EditRecord("h1", "one.tif", 20.0, {"density": 1.4}).to_json(), f)
    assert store.read_edit("h1").settings["density"] == 1.4
    assert store.drop_conflict_copies("h1") == 1
    assert not os.path.exists(copy)
    assert store.read_edit("h1").modified_at == 20.0
    assert [r.hash for r in store.iter_edits()] == ["h1"]


def test_fork_and_half_frame_hashes_name_files_safely(tmp_path):
    store = SyncStore(str(tmp_path / "sync"))
    path = store.record_path("abc#roll:12")
    assert "#" not in os.path.basename(path) and ":" not in os.path.basename(path)
    store.write_edit(EditRecord("abc#roll:12", "one.tif", 1.0, {}))
    assert store.read_edit("abc#roll:12").hash == "abc#roll:12"


def test_push_then_pull_moves_the_newer_side(tmp_path):
    mac = _repo(tmp_path, "mac")
    phone = _repo(tmp_path, "phone")
    store = SyncStore(str(tmp_path / "sync"))
    mac.save_file_settings("h1", WorkspaceConfig(), file_path="/scans/one.tif")
    mac.save_file_mark("h1", "keeper", "/scans/one.tif")
    report = SyncService(mac, store).sync()
    assert report.pushed == ["h1"] and report.pulled == []

    report = SyncService(phone, store).sync()
    assert report.pulled == ["h1"] and report.pushed == []
    assert phone.load_file_settings("h1") is not None
    assert phone.load_file_marks()["h1"] == "keeper"
    # The pulled row keeps the record's time, so nothing echoes back.
    assert SyncService(phone, store).sync().changed is False

    # The phone edits later; the Mac pulls it.
    cfg = phone.load_file_settings("h1")
    from dataclasses import replace

    phone.save_file_settings("h1", replace(cfg, exposure=replace(cfg.exposure, density=1.5)), file_path="/scans/one.tif")
    assert SyncService(phone, store).sync().pushed == ["h1"]
    assert SyncService(mac, store).sync().pulled == ["h1"]
    assert mac.load_file_settings("h1").exposure.density == 1.5


def test_a_row_older_than_the_column_is_never_pushed(tmp_path):
    repo = _repo(tmp_path, "old")
    with repo._connect(repo.edits_db_path) as conn:
        conn.execute("INSERT INTO file_settings (file_hash, settings_json, file_path) VALUES ('h0', '{}', '/x.tif')")
    assert SyncService(repo, SyncStore(str(tmp_path / "sync"))).push().pushed == []


def test_proxies_are_bounded_16_bit_linear_png(tmp_path):
    store = SyncStore(str(tmp_path / "sync"))
    linear = np.linspace(0.0, 1.0, 64 * 96 * 3, dtype=np.float32).reshape(64, 96, 3)
    path = store.write_proxy("h1", linear, long_edge=48)
    import imagecodecs

    back = imagecodecs.png_decode(open(path, "rb").read())
    assert back.dtype == np.uint16 and back.shape == (32, 48, 3)
    assert store.has_proxy("h1")
    service = SyncService(_repo(tmp_path, "p"), store)
    calls = []

    def source(p, h, cfg):
        calls.append(h)
        return linear

    report = service.ensure_proxies([{"hash": "h1", "path": "/a.tif"}, {"hash": "h2", "path": "/b.tif"}], source)
    assert calls == ["h2"] and report.proxies == ["h2"]
