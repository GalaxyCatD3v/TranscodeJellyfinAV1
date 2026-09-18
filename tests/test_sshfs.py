"""
Unit and integration tests for SSHFS connection management, retry mechanisms,
and Z: drive manual staging fallback.
"""

from pathlib import Path
import shutil
import subprocess
from unittest.mock import MagicMock, patch
import pytest

from av1_migrator.config import AppConfig, SSHFSConfig, ManualStagingConfig
from av1_migrator.db import MigrationDB
from av1_migrator.engine import MigrationEngine
from av1_migrator.models import MediaFile
from av1_migrator.sshfs import SSHFSManager, ManualStagingDB, ManualStagingManager


def test_sshfs_manager_mount_and_unmount(monkeypatch, tmp_path):
    cfg = SSHFSConfig(mount_drive="U:", host="192.168.1.180", user="root", password="Trent101$$")
    mgr = SSHFSManager(cfg)

    # Mock subprocess.run and mounting state
    called_cmds = []
    mount_state = {"mounted": True}

    def fake_run(cmd, capture_output=True, text=True, timeout=10):
        called_cmds.append(cmd)
        res = MagicMock()
        res.returncode = 0
        res.stdout = "The command completed successfully."
        res.stderr = ""
        if "/delete" in cmd:
            mount_state["mounted"] = False
        elif "net" in cmd and "use" in cmd:
            mount_state["mounted"] = True
        return res

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(mgr, "is_mounted", lambda: mount_state["mounted"])

    assert mgr.is_mounted() is True
    assert mgr.unmount() is True
    assert any("net" in cmd and "/delete" in cmd for cmd in called_cmds)

    called_cmds.clear()
    assert mgr.mount() is True
    assert any("net" in cmd and "use" in cmd for cmd in called_cmds)


def test_sshfs_manager_reconnect_with_warnings(monkeypatch):
    cfg = SSHFSConfig(reconnect_delay=0.01)
    mgr = SSHFSManager(cfg)

    unmount_calls = []
    mount_calls = []
    verify_calls = []

    monkeypatch.setattr(mgr, "unmount", lambda force=False: unmount_calls.append(force) or True)
    monkeypatch.setattr(mgr, "mount", lambda: mount_calls.append(True) or True)
    monkeypatch.setattr(mgr, "verify_connection", lambda timeout=10: verify_calls.append(True) or True)

    res = mgr.reconnect()
    assert res is True
    assert len(unmount_calls) == 1
    assert unmount_calls[0] is True
    assert len(mount_calls) == 1
    assert len(verify_calls) == 1


def test_manual_staging_db(tmp_path):
    db_file = tmp_path / "test_manual.db"
    staging_db = ManualStagingDB(db_file)

    src = Path("/media/movies/Avatar (2009).mkv")
    dst = Path("U:/srv/storage/Movies/Avatar (2009).mkv")
    staged = Path("Z:/JellyfinManualUpload/Avatar (2009).mkv")

    staging_db.record_manual_staged(
        source_path=src,
        destination_path=dst,
        staged_path=staged,
        size_bytes=5 * 1024**3,
        notes="Failed 5 SSHFS upload attempts",
    )

    records = staging_db.get_staged_files()
    assert len(records) == 1
    assert records[0]["size_bytes"] == 5 * 1024**3
    assert records[0]["status"] == "ready_for_manual_transfer"
    assert "Failed 5 SSHFS" in records[0]["notes"]


def test_manual_staging_manager_capacity_and_stage(tmp_path):
    stage_dir = tmp_path / "Z_staging"
    db_file = tmp_path / "manual_upload.db"

    cfg = ManualStagingConfig(
        staging_dir=str(stage_dir),
        max_size="700GB",
        db_path=str(db_file),
    )
    mgr = ManualStagingManager(cfg)

    # Create dummy output file to stage
    test_file = tmp_path / "encoded_movie.mkv"
    test_file.write_bytes(b"0" * 1024 * 1024)

    assert mgr.has_capacity_for(test_file.stat().st_size) is True

    dest_remote = Path("U:/srv/storage/Movies/encoded_movie.mkv")
    staged_path = mgr.stage_file(
        source_path=Path("U:/raw/movie.mkv"),
        output_file=test_file,
        destination_path=dest_remote,
        reason="Test manual upload fallback",
    )

    assert staged_path.exists()
    assert staged_path.parent == stage_dir
    assert not test_file.exists()  # Was moved/cleaned

    # Verify database entry
    staged_records = mgr.db.get_staged_files()
    assert len(staged_records) == 1
    assert staged_records[0]["staged_path"] == str(staged_path)


def test_migration_engine_upload_failure_5_retries_and_z_staging(tmp_path, monkeypatch):
    # Setup directories
    f_drive_db = tmp_path / "migration.db"
    z_staging = tmp_path / "z_staging"
    z_db = tmp_path / "z_manual.db"
    f_transcode = tmp_path / "f_transcode"

    config = AppConfig()
    config.database.path = str(f_drive_db)
    config.storage.local_staging_dir = str(f_transcode)
    config.storage.minimum_free_space = "1KB"
    config.storage.safety_margin = "1KB"
    config.manual_staging.staging_dir = str(z_staging)
    config.manual_staging.db_path = str(z_db)
    config.sshfs.max_retries = 5
    config.sshfs.reconnect_delay = 0.01
    config.processing.pre_encode_check = False

    db = MigrationDB(f_drive_db)
    engine = MigrationEngine(config=config, db=db, no_ui=True)

    # Create a source file and media file
    src_file = tmp_path / "test_movie.mkv"
    src_file.write_bytes(b"0" * 20000)

    mf = MediaFile(
        source=src_file,
        output_path=tmp_path / "remote_u" / "test_movie.mkv",
        temp_output_path=tmp_path / "remote_u" / "test_movie.mkv.encoding.mkv",
        size=src_file.stat().st_size,
        status="pending",
    )
    db.upsert_file(source_path=mf.source, source_size=mf.size, source_mtime=123456.0)

    # Mock SSHFSManager reconnect
    reconnect_counts = []
    monkeypatch.setattr(engine.sshfs_manager, "reconnect", lambda: reconnect_counts.append(1) or True)

    class MockEncoder:
        def __init__(self, media_file, config, on_progress=None, encoder_type="gpu", input_path=None, output_path=None):
            self.media_file = media_file
            self.output_path = output_path or media_file.temp_output_path

        def run(self):
            # Produce valid smaller encoded output
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_bytes(b"1" * 8000)
            return True, "Encode finished"

        def abort(self, reason=""):
            pass

    # Mock copy2 to fail during remote promotion
    upload_attempts = []
    orig_copy2 = shutil.copy2

    def fake_copy2(src, dst):
        if "remote_u" in str(dst) or "test_movie" in str(dst) and not str(dst).startswith(str(f_transcode)):
            upload_attempts.append(1)
            raise OSError("WinError 1450: Insufficient system resources or SSHFS pipe broken")
        return orig_copy2(src, dst)

    monkeypatch.setattr("shutil.copy2", fake_copy2)

    with patch("av1_migrator.engine.FFmpegEncoder", MockEncoder), \
         patch("av1_migrator.engine.validate_converted_file", return_value=(True, "OK", {})), \
         patch("av1_migrator.engine.check_storage_safety", return_value=(True, "OK", MagicMock(free_bytes=10**12))):

        pbar_mock = MagicMock()
        engine._process_media_file(
            media_file=mf,
            worker_pbar=pbar_mock,
            worker_type="GPU Worker 1",
        )

    # Verify status in DB on F: drive
    db_row = db.get_file(mf.source)
    assert db_row["status"] == "ready_for_manual_transfer"
    assert "ready for manual transfer" in db_row["error"]

    # Verify 5 upload attempts were made and SSHFS was reconnected
    assert len(upload_attempts) == 5
    assert len(reconnect_counts) >= 4

    # Verify staged on Z: drive and recorded in Z: manual DB
    z_records = engine.manual_staging_mgr.db.get_staged_files()
    assert len(z_records) == 1
    assert "Failed 5 SSHFS" in z_records[0]["notes"]
