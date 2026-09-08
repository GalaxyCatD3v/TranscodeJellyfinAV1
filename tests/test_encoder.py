"""
Unit tests for scanner and encoder.
"""

from pathlib import Path
import pytest
from av1_migrator.config import AppConfig
from av1_migrator.encoder import build_ffmpeg_command, parse_out_time_to_seconds
from av1_migrator.models import AudioStream, EncodeProgress, MediaFile, ScanStats, SubtitleStream, VideoStream
from av1_migrator.scanner import is_encoding_temp_file, scan_media_root


def test_scanner_ignores_encoding_temp_files(tmp_path):
    root = tmp_path / "movies"
    root.mkdir()

    valid_movie = root / "Movie.mkv"
    valid_movie.write_bytes(b"123")

    temp_movie = root / "Movie [AV1 1080p HDR10 CQ28].mkv.encoding.mkv"
    temp_movie.write_bytes(b"456")

    other_ext = root / "cover.jpg"
    other_ext.write_bytes(b"789")

    stats = ScanStats()
    valid_exts = {".mkv", ".mp4"}
    discovered = scan_media_root(root, valid_exts, stats)

    assert len(discovered) == 1
    assert discovered[0] == valid_movie
    assert is_encoding_temp_file(temp_movie) is True


def test_encoder_build_command():
    config = AppConfig()
    mf = MediaFile(
        source=Path("/media/movie.mkv"),
        selected_video=VideoStream(index=0, codec_name="hevc", width=3840, height=2160),
        selected_audio=[AudioStream(index=1, codec_name="truehd", language="eng")],
        selected_subtitles=[SubtitleStream(index=2, codec_name="subrip", language="eng")],
        temp_output_path=Path("/media/movie.encoding.mkv"),
        hdr=True,
    )

    cmd = build_ffmpeg_command(mf, config, use_cuda_scale=True)
    assert "-c:v" in cmd
    assert "av1_nvenc" in cmd
    assert "-preset" in cmd
    assert "p5" in cmd
    assert "-cq" in cmd
    assert "28" in cmd
    assert "-pix_fmt" in cmd
    assert "p010le" in cmd
    assert "-color_primaries" in cmd
    assert "bt2020" in cmd
    assert "hwupload_cuda,scale_cuda=1920:1080:interp_algo=lanczos" in cmd


def test_parse_out_time():
    assert parse_out_time_to_seconds("01:30:15.500") == 5415.5
    assert parse_out_time_to_seconds("00:00:10.000") == 10.0


def test_encode_progress_eta_str():
    ep = EncodeProgress()
    assert ep.eta_str == "--:--:--"

    ep.eta_seconds = 75.0
    assert ep.eta_str == "00:01:15"

    ep.eta_seconds = 3675.0
    assert ep.eta_str == "01:01:15"
