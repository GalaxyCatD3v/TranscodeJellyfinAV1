"""
Unit tests for SQLite database operations.
"""

from pathlib import Path
import pytest
from av1_migrator.db import MigrationDB


def test_db_upsert_and_summary(tmp_path):
    db_file = tmp_path / "test_migration.db"
    db = MigrationDB(db_file)

    source = tmp_path / "movie.mkv"
    db.upsert_file(
        source_path=source,
        source_size=5000000,
        source_mtime=1000.0,
        output_path=tmp_path / "movie [AV1 1080p SDR CQ28].mkv",
        status="pending",
        source_codec="h264",
        duration=120.0,
        hdr=False,
    )

    row = db.get_file(source)
    assert row is not None
    assert row["status"] == "pending"
    assert row["source_size"] == 5000000

    # Update to completed
    db.update_status(
        source,
        status="completed",
        source_bytes=5000000,
        output_bytes=2000000,
    )

    summary = db.get_summary_stats()
    assert summary["total_files"] == 1
    assert summary["completed_count"] == 1
    assert summary["completed_source_bytes"] == 5000000
    assert summary["completed_output_bytes"] == 2000000


def test_db_cleanup_interrupted_tasks(tmp_path):
    db_file = tmp_path / "test_migration.db"
    db = MigrationDB(db_file)

    src1 = tmp_path / "interrupted_movie1.mkv"
    src1.write_bytes(b"original source 1")
    out1 = tmp_path / "interrupted_movie1 [AV1 1080p SDR CQ28].mkv"
    temp1 = tmp_path / "interrupted_movie1 [AV1 1080p SDR CQ28].mkv.encoding.mkv"
    temp1.write_bytes(b"partial temp file")

    db.upsert_file(
        source_path=src1,
        source_size=1000,
        source_mtime=500.0,
        output_path=out1,
        status="encoding",
    )

    interrupted = db.get_interrupted_files()
    assert len(interrupted) == 1
    assert interrupted[0]["source_path"] == str(src1)

    # Run cleanup
    actions = db.cleanup_interrupted_tasks()
    assert len(actions) == 1
    assert actions[0]["action"] == "reset_to_pending_for_retry"
    assert actions[0]["temp_cleaned"] is True
    # Verify temp file was removed
    assert not temp1.exists()
    # Verify DB status was reset to pending
    row = db.get_file(src1)
    assert row["status"] == "pending"


def test_db_cleanup_interrupted_tasks_bloated_output(tmp_path):
    db_file = tmp_path / "test_migration_bloat.db"
    db = MigrationDB(db_file)

    src = tmp_path / "interrupted_movie.mkv"
    src.write_bytes(b"0" * 1000)  # Source is 1000 bytes
    out = tmp_path / "interrupted_movie [AV1 1080p SDR CQ28].mkv"
    out.write_bytes(b"0" * 3000)  # Output is bloated (3000 bytes > 1000 bytes)

    db.upsert_file(
        source_path=src,
        source_size=1000,
        source_mtime=500.0,
        output_path=out,
        status="validating",
    )

    def mock_validator(out_path, src_path):
        return True, "Valid AV1", {}

    actions = db.cleanup_interrupted_tasks(
        validator_fn=mock_validator,
        delete_original=True,
        keep_smaller=True,
    )

    assert len(actions) == 1
    assert actions[0]["action"] == "skipped_bloated_output_kept_smaller_original"

    # Bloated output removed, smaller original preserved
    assert not out.exists()
    assert src.exists()
    assert src.stat().st_size == 1000

    row = db.get_file(src)
    assert row["status"] == "skipped"
    assert "bloated" in row["skip_reason"].lower()


def test_db_path_normalization_and_output_path_saving(tmp_path):
    db_file = tmp_path / "test_norm_db.db"
    db = MigrationDB(db_file)

    src_str = "movies/action/../sci-fi/movie (2020).mkv"
    out_str = "movies/sci-fi/movie (2020) [AV1 1080p SDR CQ28].mkv"

    db.upsert_file(
        source_path=src_str,
        source_size=123456,
        source_mtime=555.5,
        output_path=out_str,
        status="pending",
    )

    # Lookup with Path object
    row = db.get_file(Path(src_str))
    assert row is not None
    assert row["output_path"] is not None
    assert "movie (2020) [AV1 1080p SDR CQ28].mkv" in row["output_path"]

    # Status transitions preserve/update output_path
    db.update_status(src_str, status="encoding", output_path=out_str)
    row = db.get_file(src_str)
    assert row["status"] == "encoding"
    assert row["output_path"] is not None

    db.update_status(src_str, status="completed", output_path=out_str, source_bytes=123456, output_bytes=50000)
    row = db.get_file(src_str)
    assert row["status"] == "completed"
    assert row["output_path"] is not None
    assert row["output_bytes"] == 50000


def test_db_metadata_and_pending_queries(tmp_path):
    db_file = tmp_path / "test_meta.db"
    db = MigrationDB(db_file)

    # Metadata get/set
    assert db.get_metadata("last_scan_time") is None
    db.set_metadata("last_scan_time", "2026-09-09T20:00:00")
    assert db.get_metadata("last_scan_time") == "2026-09-09T20:00:00"

    # Upsert pending and completed files
    p1 = tmp_path / "Pending1.mkv"
    p2 = tmp_path / "Pending2.mkv"
    c1 = tmp_path / "Completed1.mkv"

    db.upsert_file(p1, source_size=2000, source_mtime=1.0, status="pending")
    db.upsert_file(p2, source_size=4000, source_mtime=2.0, status="pending")
    db.upsert_file(c1, source_size=1000, source_mtime=3.0, status="completed")

    pending = db.get_pending_files()
    assert len(pending) == 2
    assert {r["source_path"] for r in pending} == {str(p1), str(p2)}
