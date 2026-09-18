"""
Unit tests for concurrent CPU/GPU multi-worker encoding and local NVMe staging.
"""

from pathlib import Path
import shutil
from unittest.mock import MagicMock, patch
import pytest

from av1_migrator.config import AppConfig
from av1_migrator.db import MigrationDB
from av1_migrator.engine import MigrationEngine
from av1_migrator.models import AudioStream, MediaFile, VideoStream
from av1_migrator.progress import create_worker_progressbar, create_overall_progressbar


def test_progress_bar_worker_positions():
    overall = create_overall_progressbar(10, position=0)
    cpu_bar = create_worker_progressbar("CPU", position=1)
    gpu_bar = create_worker_progressbar("GPU", position=2)

    assert overall.position == 0
    assert cpu_bar.position == 1
    assert gpu_bar.position == 2

    cpu_bar.set_description("CPU [Test.mkv]")
    assert "CPU [Test.mkv]" in cpu_bar.pbar.desc

    gpu_bar.reset(total=100.0, desc="GPU [Movie.mkv]")
    assert "GPU [Movie.mkv]" in gpu_bar.pbar.desc

    overall.finish()
    cpu_bar.finish()
    gpu_bar.finish()


def test_dual_worker_execution(tmp_path):
    media_dir = tmp_path / "Media"
    media_dir.mkdir()
    staging_dir = tmp_path / "Staging"
    staging_dir.mkdir()

    f1 = media_dir / "Movie1.mkv"
    f1.write_bytes(b"1" * 20000)
    f2 = media_dir / "Movie2.mkv"
    f2.write_bytes(b"2" * 5000)

    db_path = tmp_path / "test.db"
    db = MigrationDB(db_path)

    config = AppConfig()
    config.media_roots = [str(media_dir)]
    config.storage.minimum_free_space = "1KB"
    config.storage.safety_margin = "100B"
    config.storage.local_staging_dir = str(staging_dir)
    config.storage.enable_local_staging = True
    config.storage.local_min_free_space = "1KB"
    config.processing.enable_gpu_encoding = True
    config.processing.gpu_workers = 1
    config.processing.enable_cpu_encoding = True
    config.processing.pre_encode_check = False
    config.processing.delete_original = True

    workers_invoked = []

    class MockWorkerEncoder:
        def __init__(self, media_file, config, on_progress=None, encoder_type="gpu", input_path=None, output_path=None, **kwargs):
            self.media_file = media_file
            self.encoder_type = encoder_type
            self.output_path = output_path or media_file.temp_output_path

        def run(self):
            workers_invoked.append(self.encoder_type)
            # Write converted output
            self.output_path.write_bytes(b"converted av1 bytes" * 10)
            return True, "Success"

        def abort(self, reason=""):
            pass

    def mock_probe(p, cfg):
        sz = p.stat().st_size
        return MediaFile(
            source=p,
            size=sz,
            video_codec="hevc",
            duration=60.0,
            hdr=False,
            output_path=p.parent / f"{p.stem} [AV1 1080p SDR CQ28].mkv",
            temp_output_path=p.parent / f"{p.stem} [AV1 1080p SDR CQ28].mkv.encoding.mkv",
            selected_video=VideoStream(index=0, codec_name="hevc", width=3840, height=2160),
            selected_audio=[AudioStream(index=1, codec_name="truehd", language="eng")],
            status="pending",
        )

    with patch("av1_migrator.engine.probe_and_populate_media_file", side_effect=mock_probe), \
         patch("av1_migrator.engine.FFmpegEncoder", MockWorkerEncoder), \
         patch("av1_migrator.engine.validate_converted_file", return_value=(True, "OK", {})), \
         patch("av1_migrator.engine.is_av1_nvenc_available", return_value=True), \
         patch("av1_migrator.engine.check_storage_safety", return_value=(True, "OK", MagicMock(free_bytes=10**12, minimum_free_bytes=10**9))):

        engine = MigrationEngine(config=config, db=db, no_ui=True)
        res = engine.execute()

        assert res["completed"] == 2
        assert res["failed"] == 0
        assert "gpu" in workers_invoked
        assert "cpu" in workers_invoked

        # Verify originals deleted and new files exist
        assert not f1.exists()
        assert not f2.exists()
        assert (media_dir / f"{f1.stem} [AV1 1080p SDR CQ28].mkv").exists()
        assert (media_dir / f"{f2.stem} [AV1 1080p SDR CQ28].mkv").exists()


def test_multi_gpu_worker_execution(tmp_path):
    media_dir = tmp_path / "Media"
    media_dir.mkdir()
    staging_dir = tmp_path / "Staging"
    staging_dir.mkdir()

    f1 = media_dir / "Movie1.mkv"
    f1.write_bytes(b"1" * 30000)
    f2 = media_dir / "Movie2.mkv"
    f2.write_bytes(b"2" * 20000)
    f3 = media_dir / "Movie3.mkv"
    f3.write_bytes(b"3" * 10000)

    db_path = tmp_path / "test.db"
    db = MigrationDB(db_path)

    config = AppConfig()
    config.media_roots = [str(media_dir)]
    config.storage.minimum_free_space = "1KB"
    config.storage.safety_margin = "100B"
    config.storage.local_staging_dir = str(staging_dir)
    config.storage.enable_local_staging = True
    config.storage.local_min_free_space = "1KB"
    config.processing.enable_gpu_encoding = True
    config.processing.gpu_workers = 2
    config.processing.enable_cpu_encoding = False
    config.processing.pre_encode_check = False
    config.processing.delete_original = True

    workers_invoked = []

    class MockWorkerEncoder:
        def __init__(self, media_file, config, on_progress=None, encoder_type="gpu", input_path=None, output_path=None, **kwargs):
            self.media_file = media_file
            self.encoder_type = encoder_type
            self.output_path = output_path or media_file.temp_output_path

        def run(self):
            workers_invoked.append(self.encoder_type)
            self.output_path.write_bytes(b"converted av1 bytes" * 10)
            return True, "Success"

        def abort(self, reason=""):
            pass

    def mock_probe(p, cfg):
        sz = p.stat().st_size
        return MediaFile(
            source=p,
            size=sz,
            video_codec="hevc",
            duration=60.0,
            hdr=False,
            output_path=p.parent / f"{p.stem} [AV1 1080p SDR CQ28].mkv",
            temp_output_path=p.parent / f"{p.stem} [AV1 1080p SDR CQ28].mkv.encoding.mkv",
            selected_video=VideoStream(index=0, codec_name="hevc", width=3840, height=2160),
            selected_audio=[AudioStream(index=1, codec_name="truehd", language="eng")],
            status="pending",
        )

    with patch("av1_migrator.engine.probe_and_populate_media_file", side_effect=mock_probe), \
         patch("av1_migrator.engine.FFmpegEncoder", MockWorkerEncoder), \
         patch("av1_migrator.engine.validate_converted_file", return_value=(True, "OK", {})), \
         patch("av1_migrator.engine.is_av1_nvenc_available", return_value=True), \
         patch("av1_migrator.engine.check_storage_safety", return_value=(True, "OK", MagicMock(free_bytes=10**12, minimum_free_bytes=10**9))):

        engine = MigrationEngine(config=config, db=db, no_ui=True)
        res = engine.execute()

        assert res["completed"] == 3
        assert res["failed"] == 0
        assert all(w == "gpu" for w in workers_invoked)
        assert len(workers_invoked) == 3


def test_single_gpu_worker_execution(tmp_path):
    media_dir = tmp_path / "Media"
    media_dir.mkdir()
    staging_dir = tmp_path / "Staging"
    staging_dir.mkdir()

    f1 = media_dir / "Movie1.mkv"
    f1.write_bytes(b"1" * 15000)

    db_path = tmp_path / "test_single_gpu.db"
    db = MigrationDB(db_path)

    config = AppConfig()
    config.media_roots = [str(media_dir)]
    config.storage.minimum_free_space = "1KB"
    config.storage.safety_margin = "100B"
    config.storage.local_staging_dir = str(staging_dir)
    config.storage.enable_local_staging = True
    config.storage.local_min_free_space = "1KB"
    config.processing.enable_gpu_encoding = True
    config.processing.enable_gpu1 = True
    config.processing.enable_gpu2 = False
    config.processing.gpu_workers = 1
    config.processing.enable_cpu_encoding = False
    config.processing.pre_encode_check = False
    config.processing.delete_original = True

    workers_invoked = []

    class MockWorkerEncoder:
        def __init__(self, media_file, config, on_progress=None, encoder_type="gpu", input_path=None, output_path=None, **kwargs):
            self.media_file = media_file
            self.encoder_type = encoder_type
            self.output_path = output_path or media_file.temp_output_path

        def run(self):
            workers_invoked.append(self.encoder_type)
            self.output_path.write_bytes(b"av1 data" * 10)
            return True, "Success"

        def abort(self, reason=""):
            pass

    def mock_probe(p, cfg):
        sz = p.stat().st_size
        return MediaFile(
            source=p,
            size=sz,
            video_codec="hevc",
            duration=60.0,
            hdr=False,
            output_path=p.parent / f"{p.stem} [AV1 1080p SDR CQ28].mkv",
            temp_output_path=p.parent / f"{p.stem} [AV1 1080p SDR CQ28].mkv.encoding.mkv",
            selected_video=VideoStream(index=0, codec_name="hevc", width=1920, height=1080),
            selected_audio=[AudioStream(index=1, codec_name="aac", language="eng")],
            status="pending",
        )

    with patch("av1_migrator.engine.probe_and_populate_media_file", side_effect=mock_probe), \
         patch("av1_migrator.engine.FFmpegEncoder", MockWorkerEncoder), \
         patch("av1_migrator.engine.validate_converted_file", return_value=(True, "OK", {})), \
         patch("av1_migrator.engine.is_av1_nvenc_available", return_value=True), \
         patch("av1_migrator.engine.check_storage_safety", return_value=(True, "OK", MagicMock(free_bytes=10**12, minimum_free_bytes=10**9))):

        engine = MigrationEngine(config=config, db=db, no_ui=True)
        res = engine.execute()

        assert res["completed"] == 1
        assert res["failed"] == 0
        assert workers_invoked == ["gpu"]

        # Check that DB recorded worker, progress, and destination
        rec = db.get_file(f1)
        assert rec["status"] == "completed"
        assert rec["progress"] == 100.0
        assert rec["worker"] == "GPU"
        assert rec["target_codec"] == "av1"
        assert rec["resolution"] == "1920x1080"
        assert rec["output_path"] is not None


def test_sshfs_single_stream_serialization(tmp_path):
    import time
    media_dir = tmp_path / "Media"
    media_dir.mkdir()
    staging_dir = tmp_path / "Staging"
    staging_dir.mkdir()

    f1 = media_dir / "Movie1.mkv"
    f1.write_bytes(b"1" * 10000)
    f2 = media_dir / "Movie2.mkv"
    f2.write_bytes(b"2" * 10000)

    db_path = tmp_path / "test_sshfs_lock.db"
    db = MigrationDB(db_path)

    config = AppConfig()
    config.media_roots = [str(media_dir)]
    config.storage.minimum_free_space = "1KB"
    config.storage.safety_margin = "100B"
    config.storage.local_staging_dir = str(staging_dir)
    config.storage.enable_local_staging = True
    config.storage.local_min_free_space = "1KB"
    config.processing.enable_gpu_encoding = True
    config.processing.gpu_workers = 2
    config.processing.enable_cpu_encoding = False
    config.processing.pre_encode_check = False
    config.processing.delete_original = False

    concurrent_sshfs_streams = 0
    max_concurrent_sshfs_streams = 0

    original_copy2 = shutil.copy2

    def monitored_copy2(src, dst, *args, **kwargs):
        nonlocal concurrent_sshfs_streams, max_concurrent_sshfs_streams
        concurrent_sshfs_streams += 1
        if concurrent_sshfs_streams > max_concurrent_sshfs_streams:
            max_concurrent_sshfs_streams = concurrent_sshfs_streams
        time.sleep(0.05)
        res = original_copy2(src, dst, *args, **kwargs)
        concurrent_sshfs_streams -= 1
        return res

    class MockWorkerEncoder:
        def __init__(self, media_file, config, on_progress=None, encoder_type="gpu", input_path=None, output_path=None, **kwargs):
            self.media_file = media_file
            self.encoder_type = encoder_type
            self.output_path = output_path or media_file.temp_output_path

        def run(self):
            time.sleep(0.02)
            self.output_path.write_bytes(b"av1 output bytes" * 10)
            return True, "Success"

        def abort(self, reason=""):
            pass

    def mock_probe(p, cfg):
        sz = p.stat().st_size
        return MediaFile(
            source=p,
            size=sz,
            video_codec="hevc",
            duration=60.0,
            hdr=False,
            output_path=p.parent / f"{p.stem} [AV1 1080p SDR CQ28].mkv",
            temp_output_path=p.parent / f"{p.stem} [AV1 1080p SDR CQ28].mkv.encoding.mkv",
            selected_video=VideoStream(index=0, codec_name="hevc", width=1920, height=1080),
            selected_audio=[AudioStream(index=1, codec_name="aac", language="eng")],
            status="pending",
        )

    with patch("av1_migrator.engine.probe_and_populate_media_file", side_effect=mock_probe), \
         patch("av1_migrator.engine.FFmpegEncoder", MockWorkerEncoder), \
         patch("av1_migrator.engine.validate_converted_file", return_value=(True, "OK", {})), \
         patch("av1_migrator.engine.is_av1_nvenc_available", return_value=True), \
         patch("shutil.copy2", side_effect=monitored_copy2), \
         patch("av1_migrator.engine.check_storage_safety", return_value=(True, "OK", MagicMock(free_bytes=10**12, minimum_free_bytes=10**9))):

        engine = MigrationEngine(config=config, db=db, no_ui=True)
        res = engine.execute()

        assert res["completed"] == 2
        assert res["failed"] == 0
        # Verify that strictly one SSHFS copy/stream was in flight at any time
        assert max_concurrent_sshfs_streams == 1
