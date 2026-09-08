"""
Unit tests for tqdm progress bar utilities and factories.
"""

import io
from tqdm import tqdm
import pytest
from av1_migrator.progress import (
    create_encode_progressbar,
    create_overall_progressbar,
    create_probe_progressbar,
    create_scan_progressbar,
    MigrationProgressBar,
)


def test_create_scan_progressbar():
    out_buf = io.StringIO()
    bar = create_scan_progressbar("Scanning Roots", fd=out_buf)
    assert isinstance(bar, MigrationProgressBar)
    assert isinstance(bar.pbar, tqdm)
    bar.start()
    bar.update(5, dirs="5 dirs", files="20 files", current="Movie.mkv")
    bar.finish()
    output = out_buf.getvalue()
    assert "Scanning Roots" in output


def test_create_probe_progressbar():
    out_buf = io.StringIO()
    bar = create_probe_progressbar(10, "Probing Files", fd=out_buf)
    assert isinstance(bar, MigrationProgressBar)
    assert isinstance(bar.pbar, tqdm)
    bar.start()
    bar.update(5, file="TestMovie.mkv")
    bar.update(10, file="Done.mkv")
    bar.finish()
    output = out_buf.getvalue()
    assert "Probing Files" in output


def test_create_encode_progressbar():
    out_buf = io.StringIO()
    bar = create_encode_progressbar("Encoding AV1", fd=out_buf)
    assert isinstance(bar, MigrationProgressBar)
    assert isinstance(bar.pbar, tqdm)
    bar.start()
    bar.update(50.0, postfix_str="in: 1.50 GB -> out: 500.00 MB | 2.50x (60.0 fps) | ETA 00:05:00")
    bar.finish()
    output = out_buf.getvalue()
    assert "Encoding AV1" in output
    assert "in: 1.50 GB -> out: 500.00 MB" in output


def test_create_overall_progressbar():
    out_buf = io.StringIO()
    bar = create_overall_progressbar(100, "Migration Total", fd=out_buf)
    assert isinstance(bar, MigrationProgressBar)
    assert isinstance(bar.pbar, tqdm)
    bar.start()
    bar.update(50, saved="20.5 GB saved")
    bar.finish()
    output = out_buf.getvalue()
    assert "Migration Total" in output


def test_migration_progressbar_context_manager():
    out_buf = io.StringIO()
    raw_bar = create_probe_progressbar(5, "Context Probing", fd=out_buf)
    with raw_bar as mpb:
        mpb.update(2, file="File1.mkv")
        mpb.update(5, file="File2.mkv")
    output = out_buf.getvalue()
    assert "Context Probing" in output
