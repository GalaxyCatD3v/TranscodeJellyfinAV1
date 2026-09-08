"""
Unit tests for cross-platform filename sanitization, filepath normalization, and binary resolution.
"""

import os
from pathlib import Path
import pytest
from av1_migrator.utils import (
    find_binary_executable,
    format_seconds,
    normalize_filepath,
    sanitize_filename,
    sanitize_stem,
)


def test_sanitize_filename_windows_illegal_chars():
    # Windows prohibits < > : " / \ | ? *
    raw = 'Star Wars: Episode IV - A New Hope? <Director\'s "Cut"> | 4K* [AV1].mkv'
    sanitized = sanitize_filename(raw)
    
    assert ":" not in sanitized
    assert "?" not in sanitized
    assert "<" not in sanitized
    assert ">" not in sanitized
    assert '"' not in sanitized
    assert "|" not in sanitized
    assert "*" not in sanitized
    # Colon converted to ' - '
    assert "Star Wars - Episode IV" in sanitized


def test_sanitize_filename_dots_and_spaces():
    # Trailing dots and spaces are invalid on Windows NTFS
    raw = "What If...?.mkv"
    sanitized = sanitize_filename(raw)
    assert not sanitized.endswith(" ")
    assert not sanitized.endswith("..")
    assert sanitized.endswith(".mkv")


def test_sanitize_stem_removes_av1_tags():
    stem = "Movie Title (2022) [AV1 1080p HDR10 CQ28]"
    clean = sanitize_stem(stem)
    assert clean == "Movie Title (2022)"


def test_sanitize_filename_reserved_windows_names():
    assert sanitize_filename("CON.mkv") == "_CON.mkv"
    assert sanitize_filename("nul.mkv") == "_nul.mkv"
    assert sanitize_filename("aux.mp4") == "_aux.mp4"
    assert sanitize_filename("com1.mkv") == "_com1.mkv"


def test_sanitize_filename_length_capping():
    long_name = "A" * 300 + ".mkv"
    sanitized = sanitize_filename(long_name, max_length=100)
    assert len(sanitized.encode("utf-8")) <= 100
    assert sanitized.endswith(".mkv")


def test_normalize_filepath():
    p = Path("movies/action/../comedy/file.mkv")
    norm = normalize_filepath(p)
    expected = os.path.normpath(str(p))
    assert norm == expected
    assert normalize_filepath("") == ""


def test_find_binary_executable():
    # Should resolve python or standard system executable
    py = find_binary_executable("python")
    assert py is not None
    assert find_binary_executable("non_existent_binary_xyz_123") is None


def test_format_seconds():
    assert format_seconds(None) == "--:--:--"
    assert format_seconds(-5) == "--:--:--"
    assert format_seconds(0) == "00:00:00"
    assert format_seconds(65) == "00:01:05"
    assert format_seconds(3665) == "01:01:05"
    assert format_seconds(90061) == "1d 01h 01m"
