"""
Unit tests for output validation.
"""

from pathlib import Path
from unittest.mock import patch
import pytest
from av1_migrator.config import AppConfig
from av1_migrator.models import MediaFile
from av1_migrator.validator import validate_converted_file


def test_validator_valid_av1(tmp_path):
    out_file = tmp_path / "test [AV1 1080p SDR CQ28].mkv"
    out_file.write_bytes(b"dummy mkv content")

    mock_ffprobe_data = {
        "format": {"duration": "120.5", "size": "1000000"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "av1",
                "width": 1920,
                "height": 1080,
                "color_transfer": "bt709",
                "color_primaries": "bt709",
                "color_space": "bt709",
                "disposition": {"attached_pic": 0},
            },
            {
                "codec_type": "audio",
                "codec_name": "aac",
                "disposition": {},
            },
        ],
    }

    mf = MediaFile(
        source=tmp_path / "test.mkv",
        duration=120.0,
        hdr=False,
    )

    config = AppConfig()

    with patch("av1_migrator.validator.run_ffprobe_json", return_value=mock_ffprobe_data):
        is_valid, msg, _ = validate_converted_file(out_file, mf, config)
        assert is_valid is True
        assert "verified" in msg.lower()


def test_validator_wrong_codec(tmp_path):
    out_file = tmp_path / "test [AV1 1080p SDR CQ28].mkv"
    out_file.write_bytes(b"dummy content")

    mock_ffprobe_data = {
        "format": {"duration": "120.0"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "disposition": {"attached_pic": 0},
            },
            {"codec_type": "audio", "codec_name": "aac"},
        ],
    }

    mf = MediaFile(source=tmp_path / "test.mkv", duration=120.0, hdr=False)
    config = AppConfig()

    with patch("av1_migrator.validator.run_ffprobe_json", return_value=mock_ffprobe_data):
        is_valid, msg, _ = validate_converted_file(out_file, mf, config)
        assert is_valid is False
        assert "expected 'av1'" in msg
