"""
Integration tests for Galaxy AV1 Migrator engine.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from av1_migrator.config import AppConfig
from av1_migrator.db import MigrationDB
from av1_migrator.engine import MigrationEngine
from av1_migrator.models import AudioStream, MediaFile, StorageStats, VideoStream


@pytest.fixture
def test_env(tmp_path):
    media_dir = tmp_path / "Movies"
    media_dir.mkdir()

    # Create dummy media files
    m1 = media_dir / "Movie Large (2020).mkv"
    m1.write_bytes(b"0" * 50000)

    m2 = media_dir / "Movie Small (2021).mkv"
    m2.write_bytes(b"0" * 10000)

    db_path = tmp_path / "test_migration.db"
    db = MigrationDB(db_path)

    config = AppConfig()
    config.media_roots = [str(media_dir)]
    config.database.path = str(db_path)
    config.storage.minimum_free_space = "10KB"
    config.storage.safety_margin = "1KB"
    config.processing.delete_original = True

    return {
        "media_dir": media_dir,
        "m1": m1,
        "m2": m2,
        "db": db,
        "config": config,
    }


def test_engine_dry_run(test_env):
    config = test_env["config"]
    db = test_env["db"]

    def mock_probe(p, cfg):
        sz = p.stat().st_size
        return MediaFile(
            source=p,
            size=sz,
            video_codec="hevc",
            duration=100.0,
            hdr=False,
            output_path=p.parent / f"{p.stem} [AV1 1080p SDR CQ28].mkv",
            temp_output_path=p.parent / f"{p.stem} [AV1 1080p SDR CQ28].mkv.encoding.mkv",
            selected_video=VideoStream(index=0, codec_name="hevc", width=3840, height=2160),
            selected_audio=[AudioStream(index=1, codec_name="truehd", language="eng")],
            status="pending",
        )

    with patch("av1_migrator.engine.probe_and_populate_media_file", side_effect=mock_probe):
        engine = MigrationEngine(config=config, db=db, dry_run=True, no_ui=True)
        res = engine.execute()
        assert res["status"] == "dry_run_complete"
        assert res["queue_count"] == 2
        # Largest first order verified by default
        assert engine.queue[0].source == test_env["m1"]
        assert engine.queue[1].source == test_env["m2"]


def test_engine_dry_run_smallest_first(test_env):
    config = test_env["config"]
    config.processing.sort = "smallest_first"
    db = test_env["db"]

    def mock_probe(p, cfg):
        sz = p.stat().st_size
        return MediaFile(
            source=p,
            size=sz,
            video_codec="hevc",
            duration=100.0,
            hdr=False,
            output_path=p.parent / f"{p.stem} [AV1 1080p SDR CQ28].mkv",
            temp_output_path=p.parent / f"{p.stem} [AV1 1080p SDR CQ28].mkv.encoding.mkv",
            selected_video=VideoStream(index=0, codec_name="hevc", width=3840, height=2160),
            selected_audio=[AudioStream(index=1, codec_name="truehd", language="eng")],
            status="pending",
        )

    with patch("av1_migrator.engine.probe_and_populate_media_file", side_effect=mock_probe):
        engine = MigrationEngine(config=config, db=db, dry_run=True, no_ui=True)
        res = engine.execute()
        assert res["status"] == "dry_run_complete"
        assert res["queue_count"] == 2
        # Smallest first order verified when configured
        assert engine.queue[0].source == test_env["m2"]
        assert engine.queue[1].source == test_env["m1"]


def test_engine_full_workflow_success(test_env):
    config = test_env["config"]
    db = test_env["db"]
    m1 = test_env["m1"]

    def mock_probe(p, cfg):
        sz = p.stat().st_size
        return MediaFile(
            source=p,
            size=sz,
            video_codec="hevc",
            duration=100.0,
            hdr=False,
            output_path=p.parent / f"{p.stem} [AV1 1080p SDR CQ28].mkv",
            temp_output_path=p.parent / f"{p.stem} [AV1 1080p SDR CQ28].mkv.encoding.mkv",
            selected_video=VideoStream(index=0, codec_name="hevc", width=3840, height=2160),
            selected_audio=[AudioStream(index=1, codec_name="truehd", language="eng")],
            status="pending",
        )

    class MockEncoder:
        def __init__(self, media_file, config, on_progress=None):
            self.media_file = media_file

        def run(self):
            # Create the temp file
            self.media_file.temp_output_path.write_bytes(b"encoded av1 bytes")
            return True, "Encode completed successfully"

        def abort(self, reason=""):
            pass

    with patch("av1_migrator.engine.probe_and_populate_media_file", side_effect=mock_probe), \
         patch("av1_migrator.engine.FFmpegEncoder", MockEncoder), \
         patch("av1_migrator.engine.validate_converted_file", return_value=(True, "OK", {})), \
         patch("av1_migrator.engine.check_storage_safety", return_value=(True, "OK", StorageStats())):

        engine = MigrationEngine(config=config, db=db, limit=1, no_ui=True)
        res = engine.execute()

        assert res["completed"] == 1
        assert res["failed"] == 0

        # Original m1 should have been deleted (delete_original=True)
        assert not m1.exists()
        # Final promoted output should exist
        final_out = m1.parent / f"{m1.stem} [AV1 1080p SDR CQ28].mkv"
        assert final_out.exists()

        # Check DB record
        db_rec = db.get_file(m1)
        assert db_rec is not None
        assert db_rec["status"] == "completed"


def test_engine_startup_recovery(test_env):
    config = test_env["config"]
    db = test_env["db"]
    m1 = test_env["m1"]

    # Simulate an interrupted encode
    temp_m1 = m1.parent / f"{m1.stem} [AV1 1080p SDR CQ28].mkv.encoding.mkv"
    temp_m1.write_bytes(b"corrupt partial encode")

    db.upsert_file(
        source_path=m1,
        source_size=m1.stat().st_size,
        source_mtime=m1.stat().st_mtime,
        output_path=m1.parent / f"{m1.stem} [AV1 1080p SDR CQ28].mkv",
        status="encoding",
    )

    engine = MigrationEngine(config=config, db=db, no_ui=True)
    actions = engine.run_startup_recovery()
    assert len(actions) == 1
    assert actions[0]["action"] == "reset_to_pending_for_retry"
    assert actions[0]["temp_cleaned"] is True
    assert not temp_m1.exists()

    db_rec = db.get_file(m1)
    assert db_rec["status"] == "pending"


def test_engine_bloated_encode_preserves_smaller_original(test_env):
    config = test_env["config"]
    db = test_env["db"]
    m2 = test_env["m2"]
    original_size = m2.stat().st_size  # 10000 bytes

    def mock_probe(p, cfg):
        sz = p.stat().st_size
        return MediaFile(
            source=p,
            size=sz,
            video_codec="hevc",
            duration=100.0,
            hdr=False,
            output_path=p.parent / f"{p.stem} [AV1 1080p SDR CQ28].mkv",
            temp_output_path=p.parent / f"{p.stem} [AV1 1080p SDR CQ28].mkv.encoding.mkv",
            selected_video=VideoStream(index=0, codec_name="hevc", width=3840, height=2160),
            selected_audio=[AudioStream(index=1, codec_name="truehd", language="eng")],
            status="pending",
        )

    class MockBloatedEncoder:
        def __init__(self, media_file, config, on_progress=None):
            self.media_file = media_file

        def run(self):
            # Write a larger bloated file (25000 bytes > 10000 bytes source)
            self.media_file.temp_output_path.write_bytes(b"X" * 25000)
            return True, "Encode completed"

        def abort(self, reason=""):
            pass

    with patch("av1_migrator.engine.probe_and_populate_media_file", side_effect=mock_probe), \
         patch("av1_migrator.engine.FFmpegEncoder", MockBloatedEncoder), \
         patch("av1_migrator.engine.validate_converted_file", return_value=(True, "OK", {})), \
         patch("av1_migrator.engine.check_storage_safety", return_value=(True, "OK", StorageStats())):

        engine = MigrationEngine(config=config, db=db, single_file=m2, no_ui=True)
        res = engine.execute()

        assert res["completed"] == 0
        assert res["skipped"] == 1

        # Original source MUST be preserved!
        assert m2.exists()
        assert m2.stat().st_size == original_size

        # Bloated temp file and final output should NOT exist
        final_out = m2.parent / f"{m2.stem} [AV1 1080p SDR CQ28].mkv"
        temp_out = m2.parent / f"{m2.stem} [AV1 1080p SDR CQ28].mkv.encoding.mkv"
        assert not final_out.exists()
        assert not temp_out.exists()

        # Database record should reflect skipped due to bloat
        db_rec = db.get_file(m2)
        assert db_rec is not None
        assert db_rec["status"] == "skipped"
        assert "bloated" in db_rec["skip_reason"].lower()


def test_engine_existing_bloated_output_removed_and_original_preserved(test_env):
    config = test_env["config"]
    db = test_env["db"]
    m2 = test_env["m2"]

    # Pre-create a bloated existing output on disk (30000 bytes > 10000 bytes)
    final_out = m2.parent / f"{m2.stem} [AV1 1080p SDR CQ28].mkv"
    final_out.write_bytes(b"Y" * 30000)

    def mock_probe(p, cfg):
        if not p.is_file():
            return MediaFile(source=p, status="failed", error_reason="File not found")
        sz = p.stat().st_size
        return MediaFile(
            source=p,
            size=sz,
            video_codec="hevc",
            duration=100.0,
            hdr=False,
            output_path=final_out,
            temp_output_path=m2.parent / f"{m2.stem} [AV1 1080p SDR CQ28].mkv.encoding.mkv",
            selected_video=VideoStream(index=0, codec_name="hevc", width=3840, height=2160),
            selected_audio=[AudioStream(index=1, codec_name="truehd", language="eng")],
            status="pending",
        )

    with patch("av1_migrator.engine.probe_and_populate_media_file", side_effect=mock_probe), \
         patch("av1_migrator.engine.validate_converted_file", return_value=(True, "OK", {})):

        engine = MigrationEngine(config=config, db=db, single_file=m2, no_ui=True)
        res = engine.execute()

        # Bloated output was removed
        assert not final_out.exists()
        # Original smaller file is kept
        assert m2.exists()

        db_rec = db.get_file(m2)
        assert db_rec is not None
        assert db_rec["status"] == "skipped"
        assert "bloated" in db_rec["skip_reason"].lower()


def test_engine_pre_encode_check_skips_bloat(test_env):
    config = test_env["config"]
    config.processing.pre_encode_check = True
    db = test_env["db"]
    m1 = test_env["m1"]

    def mock_probe(p, cfg):
        sz = p.stat().st_size
        return MediaFile(
            source=p,
            size=sz,
            video_codec="hevc",
            duration=100.0,
            hdr=False,
            output_path=p.parent / f"{p.stem} [AV1 1080p SDR CQ28].mkv",
            temp_output_path=p.parent / f"{p.stem} [AV1 1080p SDR CQ28].mkv.encoding.mkv",
            selected_video=VideoStream(index=0, codec_name="hevc", width=3840, height=2160),
            selected_audio=[AudioStream(index=1, codec_name="truehd", language="eng")],
            status="pending",
        )

    from av1_migrator.optimizer import OptimizationResult
    bloat_opt_result = OptimizationResult(
        should_encode=False,
        estimated_size_bytes=60000,
        estimated_savings_bytes=-10000,
        estimated_savings_percent=-20.0,
        reason="Sample test predicted bloat: estimated output (60.00 KB) >= original (50.00 KB)",
        method="sample_test",
    )

    with patch("av1_migrator.engine.probe_and_populate_media_file", side_effect=mock_probe), \
         patch("av1_migrator.engine.evaluate_pre_encode_optimization", return_value=bloat_opt_result), \
         patch("av1_migrator.engine.FFmpegEncoder") as mock_encoder, \
         patch("av1_migrator.engine.check_storage_safety", return_value=(True, "OK", StorageStats())):

        engine = MigrationEngine(config=config, db=db, single_file=m1, no_ui=True)
        res = engine.execute()

        assert res["completed"] == 0
        assert res["skipped"] == 1
        # Full encoder was NOT called!
        assert mock_encoder.call_count == 0

        # DB record updated to skipped
        db_rec = db.get_file(m1)
        assert db_rec is not None
        assert db_rec["status"] == "skipped"
        assert "predicted bloat" in db_rec["skip_reason"].lower()


def test_engine_scan_cache_within_24h_skips_remote_walk(test_env):
    from datetime import datetime, timedelta
    config = test_env["config"]
    db = test_env["db"]
    m1 = test_env["m1"]
    m2 = test_env["m2"]

    # Seed DB with last scan 2 hours ago (< 24h) and pending records
    db.set_metadata("last_scan_time", (datetime.now() - timedelta(hours=2)).isoformat())
    db.upsert_file(m1, source_size=50000, source_mtime=1.0, status="pending")
    db.upsert_file(m2, source_size=10000, source_mtime=2.0, status="pending")

    def mock_probe(p, cfg):
        sz = p.stat().st_size
        return MediaFile(
            source=p,
            size=sz,
            video_codec="hevc",
            duration=100.0,
            hdr=False,
            output_path=p.parent / f"{p.stem} [AV1 1080p SDR CQ28].mkv",
            temp_output_path=p.parent / f"{p.stem} [AV1 1080p SDR CQ28].mkv.encoding.mkv",
            selected_video=VideoStream(index=0, codec_name="hevc", width=3840, height=2160),
            selected_audio=[AudioStream(index=1, codec_name="truehd", language="eng")],
            status="pending",
        )

    with patch("av1_migrator.engine.scan_all_roots") as mock_scan_roots, \
         patch("av1_migrator.engine.probe_and_populate_media_file", side_effect=mock_probe):

        engine = MigrationEngine(config=config, db=db, dry_run=True, no_ui=True)
        res = engine.execute()

        # scan_all_roots must NOT have been called because last scan was 2h ago (< 24h)
        assert mock_scan_roots.call_count == 0
        assert res["status"] == "dry_run_complete"
        assert res["queue_count"] == 2
        # Verify sort order
        assert engine.queue[0].source == m1
        assert engine.queue[1].source == m2


def test_engine_force_scan_overrides_24h_cache(test_env):
    from datetime import datetime, timedelta
    config = test_env["config"]
    db = test_env["db"]
    m1 = test_env["m1"]
    m2 = test_env["m2"]

    # Seed DB with last scan 1 hour ago
    db.set_metadata("last_scan_time", (datetime.now() - timedelta(hours=1)).isoformat())

    def mock_probe(p, cfg):
        sz = p.stat().st_size
        return MediaFile(
            source=p,
            size=sz,
            video_codec="hevc",
            duration=100.0,
            hdr=False,
            output_path=p.parent / f"{p.stem} [AV1 1080p SDR CQ28].mkv",
            temp_output_path=p.parent / f"{p.stem} [AV1 1080p SDR CQ28].mkv.encoding.mkv",
            selected_video=VideoStream(index=0, codec_name="hevc", width=3840, height=2160),
            selected_audio=[AudioStream(index=1, codec_name="truehd", language="eng")],
            status="pending",
        )

    from av1_migrator.models import ScanStats
    with patch("av1_migrator.engine.scan_all_roots", return_value=([m1, m2], ScanStats(files_discovered=2))) as mock_scan_roots, \
         patch("av1_migrator.engine.probe_and_populate_media_file", side_effect=mock_probe):

        # With force_scan=True, scan_all_roots must be invoked
        engine = MigrationEngine(config=config, db=db, dry_run=True, force_scan=True, no_ui=True)
        res = engine.execute()

        assert mock_scan_roots.call_count == 1
        assert res["queue_count"] == 2
