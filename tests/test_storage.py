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
