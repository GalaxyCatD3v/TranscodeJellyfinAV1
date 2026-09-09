"""
Unit tests for CUDA scaling interpolation benchmarking module.
"""

from unittest.mock import MagicMock, patch
import pytest

from av1_migrator.benchmark import run_interpolation_benchmark
from av1_migrator.config import AppConfig


def test_run_interpolation_benchmark_mocked():
    config = AppConfig()

    class MockSubprocessResult:
        returncode = 0
        stdout = ""
        stderr = "frame=  300 fps= 85.5 q=28.0 Lsize=N/A time=00:00:05.00 bitrate=N/A speed=1.42x"

    with patch("av1_migrator.benchmark.is_av1_nvenc_available", return_value=True), \
         patch("av1_migrator.benchmark.is_scale_cuda_available", return_value=True), \
         patch("subprocess.run", return_value=MockSubprocessResult()):

        results = run_interpolation_benchmark(
            source_path=None,
            config=config,
            duration=5.0,
            algorithms=["bicubic", "bilinear"],
        )

        assert "bicubic" in results
        assert "bilinear" in results
        assert results["bicubic"]["frames"] == 300
        assert results["bicubic"]["fps"] == 85.5
        assert results["bicubic"]["speed"] == 1.42


def test_run_interpolation_benchmark_unavailable_encoder():
    config = AppConfig()
    with patch("av1_migrator.benchmark.is_av1_nvenc_available", return_value=False):
        res = run_interpolation_benchmark(config=config)
        assert "error" in res
        assert "not available" in res["error"]
