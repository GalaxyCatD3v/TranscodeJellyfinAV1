"""
Tests for the pre-encode optimization and sample estimation module.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from av1_migrator.config import AppConfig
from av1_migrator.models import AudioStream, MediaFile, SubtitleStream, VideoStream
from av1_migrator.optimizer import (
    build_sample_ffmpeg_command,
    evaluate_pre_encode_optimization,
    OptimizationResult,
    run_pre_encode_sample_test,
)


@pytest.fixture
def sample_media_file(tmp_path):
    p = tmp_path / "Movie.2021.1080p.mkv"
    p.write_bytes(b"0" * 100_000_000)  # 100 MB
    return MediaFile(
        source=p,
        size=100_000_000,
        video_codec="hevc",
        duration=600.0,
        hdr=False,
        output_path=tmp_path / "Movie.2021.1080p [AV1 1080p SDR CQ28].mkv",
        temp_output_path=tmp_path / "Movie.2021.1080p [AV1 1080p SDR CQ28].mkv.encoding.mkv",
        selected_video=VideoStream(index=0, codec_name="hevc", width=1920, height=1080),
        selected_audio=[AudioStream(index=1, codec_name="aac", language="eng")],
        status="pending",
    )


def test_build_sample_ffmpeg_command(sample_media_file):
    config = AppConfig()
    out_path = Path("/tmp/sample_out.mkv")
    cmd = build_sample_ffmpeg_command(
        sample_media_file,
        config,
        sample_output_path=out_path,
        seek_offset=30.0,
        duration=25.0,
        use_cuda_scale=True,
    )

    assert "-ss" in cmd
    assert "30.00" in cmd
    assert "-t" in cmd
    assert "25.00" in cmd
    assert "-c:v" in cmd
    assert "av1_nvenc" in cmd
    assert str(out_path) in cmd


def test_evaluate_pre_encode_optimization_disabled(sample_media_file):
    config = AppConfig()
    config.processing.pre_encode_check = False
    res = evaluate_pre_encode_optimization(sample_media_file, config)
    assert res.should_encode is True
    assert res.method == "disabled"


def test_run_pre_encode_sample_test_short_clip(tmp_path):
    config = AppConfig()
    p = tmp_path / "Short.mkv"
    p.write_bytes(b"0" * 1000)
    mf = MediaFile(
        source=p,
        size=1000,
        video_codec="h264",
        duration=20.0,  # Less than sample_duration * 1.5
        status="pending",
    )
    res = run_pre_encode_sample_test(mf, config)
    assert res.should_encode is True
    assert res.method == "short_clip"


def test_run_pre_encode_sample_test_predicted_savings(sample_media_file):
    config = AppConfig()
    config.processing.sample_duration = 30.0
    config.processing.min_savings_percent = 10.0

    # 30s sample produces 1.5 MB -> 600s will be 30 MB (saving 70 MB / 70%)
    mock_stat = MagicMock()
    mock_stat.st_size = 1_500_000

    with patch("subprocess.run") as mock_run, \
         patch("pathlib.Path.is_file", return_value=True), \
         patch("pathlib.Path.stat", return_value=mock_stat):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_run.return_value = mock_proc

        res = run_pre_encode_sample_test(sample_media_file, config)
        assert res.should_encode is True
        assert res.estimated_size_bytes == 30_000_000
        assert res.estimated_savings_percent == 70.0


def test_run_pre_encode_sample_test_predicted_bloat(sample_media_file):
    config = AppConfig()
    config.processing.sample_duration = 30.0
    config.processing.keep_smaller = True

    # 30s sample produces 6 MB -> 600s will be 120 MB (> original 100 MB)
    mock_stat = MagicMock()
    mock_stat.st_size = 6_000_000

    with patch("subprocess.run") as mock_run, \
         patch("pathlib.Path.is_file", return_value=True), \
         patch("pathlib.Path.stat", return_value=mock_stat):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_run.return_value = mock_proc

        res = run_pre_encode_sample_test(sample_media_file, config)
        assert res.should_encode is False
        assert res.estimated_size_bytes == 120_000_000
        assert "predicted bloat" in res.reason.lower()


def test_run_pre_encode_sample_test_insufficient_savings(sample_media_file):
    config = AppConfig()
    config.processing.sample_duration = 30.0
    config.processing.min_savings_percent = 20.0

    # 30s sample produces 4.5 MB -> 600s will be 90 MB (10% savings, but required 20%)
    mock_stat = MagicMock()
    mock_stat.st_size = 4_500_000

    with patch("subprocess.run") as mock_run, \
         patch("pathlib.Path.is_file", return_value=True), \
         patch("pathlib.Path.stat", return_value=mock_stat):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_run.return_value = mock_proc

        res = run_pre_encode_sample_test(sample_media_file, config)
        assert res.should_encode is False
        assert "insufficient savings" in res.reason.lower()
