"""
Unit tests for storage safety checks.
"""

from pathlib import Path
from unittest.mock import patch
import pytest
from av1_migrator.models import StorageStats
from av1_migrator.storage import check_storage_safety


def test_storage_safety_sufficient():
    # Free space = 1.2 TB, original = 100 GB, margin = 10 GB, min = 1 TB
    # Required = 1 TB + 100 GB + 10 GB = 1.11 TB <= 1.2 TB -> SAFE
    mock_stats = StorageStats(
        free_bytes=int(1.2 * 1024**4),
        total_bytes=int(2.0 * 1024**4),
        used_bytes=int(0.8 * 1024**4),
    )

    with patch("av1_migrator.storage.get_path_disk_usage", return_value=mock_stats):
        is_safe, msg, stats = check_storage_safety(
            path=Path("/dummy"),
            original_file_size=100 * 1024**3,
            minimum_free_space_bytes=1024**4,
            safety_margin_bytes=10 * 1024**3,
        )
        assert is_safe is True
        assert "passed" in msg.lower()


def test_storage_safety_insufficient():
    # Free space = 1.08 TB, original = 100 GB, margin = 10 GB, min = 1 TB
    # Required = 1.11 TB > 1.08 TB -> UNSAFE
    mock_stats = StorageStats(
        free_bytes=int(1.08 * 1024**4),
        total_bytes=int(2.0 * 1024**4),
        used_bytes=int(0.92 * 1024**4),
    )

    with patch("av1_migrator.storage.get_path_disk_usage", return_value=mock_stats):
        is_safe, msg, stats = check_storage_safety(
            path=Path("/dummy"),
            original_file_size=100 * 1024**3,
            minimum_free_space_bytes=1024**4,
            safety_margin_bytes=10 * 1024**3,
        )
        assert is_safe is False
        assert "failed" in msg.lower()


def test_get_path_disk_usage_winerror_1450_recovery():
    """Verify that get_path_disk_usage recovers from WinError 1450 using fallback stats."""
    from av1_migrator.storage import get_path_disk_usage

    fallback = StorageStats(
        free_bytes=1000,
        total_bytes=2000,
        used_bytes=1000,
    )

    with patch("shutil.disk_usage", side_effect=OSError(1450, "Insufficient system resources")):
        stats = get_path_disk_usage(
            "Y:\\srv\\storage\\Movies\\Movie (2020)\\Movie.mkv",
            fallback_stats=fallback,
        )
        assert stats.free_bytes == 1000
        assert stats.total_bytes == 2000


def test_storage_monitor_thread_winerror_1450_resilience():
    """Verify that StorageMonitorThread handles transient WinError 1450 without aborting."""
    from av1_migrator.storage import StorageMonitorThread
    import time

    emergency_called = []
    monitor = StorageMonitorThread(
        watch_path="Y:\\srv\\storage\\Movies\\Movie (2020)\\Movie.mkv",
        minimum_free_space_bytes=100,
        poll_interval=0.05,
        on_emergency_stop=lambda s: emergency_called.append(s),
    )

    with patch("shutil.disk_usage", side_effect=OSError(1450, "Insufficient system resources")):
        monitor.start()
        time.sleep(0.15)
        monitor.stop()

    assert len(emergency_called) == 0
    assert monitor.is_emergency_triggered is False
