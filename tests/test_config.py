"""
Unit tests for configuration and byte utilities.
"""

from pathlib import Path
import pytest
from av1_migrator.config import format_bytes, load_config, parse_size_to_bytes


def test_parse_size_to_bytes():
    assert parse_size_to_bytes("1TB") == 1024**4
    assert parse_size_to_bytes("10GB") == 10 * (1024**3)
    assert parse_size_to_bytes("500MB") == 500 * (1024**2)
    assert parse_size_to_bytes("1024B") == 1024
    assert parse_size_to_bytes(1048576) == 1048576


def test_format_bytes():
    assert format_bytes(1024**4) == "1.00 TB"
    assert format_bytes(10 * (1024**3)) == "10.00 GB"
    assert format_bytes(500 * (1024**2)) == "500.00 MB"
    assert format_bytes(100) == "100 B"


def test_load_default_config(tmp_path):
    cfg_file = tmp_path / "test_config.yaml"
    cfg_file.write_text("""
media_roots:
  - "/test/movies"
output:
  video_codec: "av1_nvenc"
  cq: 26
storage:
  minimum_free_space: "2TB"
""", encoding="utf-8")

    config = load_config(cfg_file)
    assert config.media_roots == ["/test/movies"]
    assert config.output.video_codec == "av1_nvenc"
    assert config.output.cq == 26
    assert config.storage.minimum_free_space == "2TB"
    assert config.storage.minimum_free_space_bytes == 2 * (1024**4)
    assert config.processing.keep_smaller is True


def test_load_processing_keep_smaller(tmp_path):
    cfg_file = tmp_path / "test_keep_smaller.yaml"
    cfg_file.write_text("""
processing:
  keep_smaller: false
  pre_encode_check: false
  sample_duration: 15.0
  min_savings_percent: 5.0
""", encoding="utf-8")
    config = load_config(cfg_file)
    assert config.processing.keep_smaller is False
    assert config.processing.pre_encode_check is False
    assert config.processing.sample_duration == 15.0
    assert config.processing.min_savings_percent == 5.0
