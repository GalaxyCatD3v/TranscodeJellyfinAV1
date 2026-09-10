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
  enable_gpu1: false
  enable_gpu2: true
  scan_cache_hours: 12.0
""", encoding="utf-8")
    config = load_config(cfg_file)
    assert config.processing.keep_smaller is False
    assert config.processing.pre_encode_check is False
    assert config.processing.sample_duration == 15.0
    assert config.processing.min_savings_percent == 5.0
    assert config.processing.enable_gpu1 is False
    assert config.processing.enable_gpu2 is True
    assert config.processing.scan_cache_hours == 12.0


def test_cli_worker_and_sort_flags(monkeypatch):
    from av1_migrator.cli import parse_args

    monkeypatch.setattr("sys.argv", ["prog", "--biggest-first", "--gpu1", "--no-gpu2", "--no-cpu", "--force-scan", "--scan-cache-hours", "48"])
    args = parse_args()
    assert args.sort == "largest_first"
    assert args.enable_gpu1 is True
    assert args.enable_gpu2 is False
    assert args.enable_cpu is False
    assert args.force_scan is True
    assert args.scan_cache_hours == 48.0

    monkeypatch.setattr("sys.argv", ["prog", "--smallest-first", "--workers", "cpu,gpu1"])
    args2 = parse_args()
    assert args2.sort == "smallest_first"
    assert args2.workers == "cpu,gpu1"
