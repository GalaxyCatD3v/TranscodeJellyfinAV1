"""
Hardware CUDA scaling and interpolation algorithm benchmarking module.
Compares throughput (FPS, speed, GPU utilization) of scale_cuda interpolation algorithms
(bicubic, bilinear, lanczos) on NVIDIA RTX 4070 Ti.
"""

from pathlib import Path
import re
import subprocess
import time
from typing import Any, Dict, List, Optional

from av1_migrator.config import AppConfig, format_bytes, load_config
from av1_migrator.encoder import is_av1_nvenc_available, is_scale_cuda_available
from av1_migrator.gpu import fetch_gpu_stats
from av1_migrator.logger import get_logger
from av1_migrator.progress import MigrationProgressBar
from av1_migrator.utils import find_binary_executable


def run_interpolation_benchmark(
    source_path: Optional[str | Path] = None,
    config: Optional[AppConfig] = None,
    duration: float = 10.0,
    algorithms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Benchmarks CUDA scaling interpolation algorithms (bicubic, bilinear, lanczos)
    on the current NVIDIA GPU (e.g. RTX 4070 Ti).
    Prints a formatted performance comparison table and returns the benchmark metrics.
    """
    config = config or load_config()
    algorithms = algorithms or ["bicubic", "bilinear", "lanczos"]
    logger = get_logger()
    exe = find_binary_executable(config.ffmpeg.executable) or config.ffmpeg.executable

    if not is_av1_nvenc_available(exe):
        msg = "NVIDIA av1_nvenc encoder is not available on this system."
        logger.error(msg)
        MigrationProgressBar.write(f"Benchmark Error: {msg}")
        return {"error": msg}

    if not is_scale_cuda_available(exe):
        msg = "FFmpeg scale_cuda filter is not available on this system."
        logger.error(msg)
        MigrationProgressBar.write(f"Benchmark Error: {msg}")
        return {"error": msg}

    # Resolve input source
    input_desc = ""
    is_synthetic = False
    resolved_src: Optional[Path] = None

    if source_path:
        p = Path(source_path).resolve()
        if p.is_file():
            resolved_src = p
            input_desc = f"{p.name} ({format_bytes(p.stat().st_size)})"

    if resolved_src is None:
        # Check media roots for a sample video
        for root_str in config.media_roots:
            root_p = Path(root_str)
            if root_p.exists():
                for ext in config.extensions:
                    found = list(root_p.glob(f"*{ext}"))
                    if found:
                        resolved_src = found[0]
                        input_desc = f"{resolved_src.name} ({format_bytes(resolved_src.stat().st_size)})"
                        break
            if resolved_src is not None:
                break

    if resolved_src is None:
        is_synthetic = True
        input_desc = f"Synthetic 4K 60fps clip ({duration:.0f}s, 3840x2160)"

    target_w = config.output.width
    target_h = config.output.height
    preset = config.output.preset
    cq = config.output.cq
    pix_fmt = config.output.pixel_format

    MigrationProgressBar.write("=" * 80)
    MigrationProgressBar.write("CUDA Scaling Interpolation Benchmark (NVIDIA RTX 4070 Ti)")
    MigrationProgressBar.write(f"Source: {input_desc}")
    MigrationProgressBar.write(f"Target: {target_w}x{target_h} AV1 NVENC (Preset: {preset.upper()}, CQ: {cq}, PixFmt: {pix_fmt})")
    MigrationProgressBar.write(f"Sample Duration: {duration:.1f} seconds")
    MigrationProgressBar.write("=" * 80)

    results: Dict[str, Dict[str, Any]] = {}

    for algo in algorithms:
        MigrationProgressBar.write(f"Running benchmark with interp_algo={algo}...")
        if is_synthetic:
            vf_filter = f"format=yuv420p,hwupload_cuda,scale_cuda={target_w}:{target_h}:interp_algo={algo}:format=p010le"
        else:
            vf_filter = f"hwupload_cuda,scale_cuda={target_w}:{target_h}:interp_algo={algo}:format={pix_fmt}"

        cmd = [
            exe,
            "-y",
            "-nostdin",
            "-hide_banner",
        ]

        if is_synthetic:
            cmd.extend([
                "-f", "lavfi",
                "-i", f"testsrc=duration={duration:.1f}:size=3840x2160:rate=60",
            ])
        else:
            cmd.extend([
                "-ss", "30.0",
                "-t", f"{duration:.1f}",
                "-i", str(resolved_src),
            ])

        cmd.extend([
            "-map", "0:v:0",
            "-vf", vf_filter,
            "-c:v", config.output.video_codec,
            "-preset", preset,
            "-cq", str(cq),
            "-f", "null",
            "-",
        ])

        start_time = time.time()
        gpu_stats_before = fetch_gpu_stats()

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120.0,
                check=False,
            )
        except Exception as e:
            logger.error(f"Error benchmarking {algo}: {e}")
            results[algo] = {"error": str(e)}
            continue

        elapsed = time.time() - start_time
        gpu_stats_after = fetch_gpu_stats()

        if proc.returncode != 0:
            err_line = (proc.stderr or proc.stdout).splitlines()[-1] if (proc.stderr or proc.stdout) else "Unknown error"
            logger.error(f"Benchmark run failed for {algo}: {err_line}")
            results[algo] = {"error": err_line}
            continue

        # Parse stderr for frame count, fps, and speed
        # e.g.: frame=  600 fps=142.3 q=28.0 Lsize=N/A time=00:00:10.00 bitrate=N/A speed=2.37x
        stderr_txt = proc.stderr or ""
        frames = 0
        fps = 0.0
        speed_factor = 0.0

        m_frame = re.findall(r"frame=\s*(\d+)", stderr_txt)
        if m_frame:
            frames = int(m_frame[-1])

        m_fps = re.findall(r"fps=\s*([\d\.]+)", stderr_txt)
        if m_fps:
            fps = float(m_fps[-1])
        elif frames > 0 and elapsed > 0:
            fps = frames / elapsed

        m_speed = re.findall(r"speed=\s*([\d\.]+)x", stderr_txt)
        if m_speed:
            speed_factor = float(m_speed[-1])
        elif elapsed > 0:
            speed_factor = duration / elapsed

        gpu_util = gpu_stats_after.gpu_util if gpu_stats_after.gpu_util is not None else (gpu_stats_before.gpu_util or 0.0)
        enc_util = gpu_stats_after.enc_util if gpu_stats_after.enc_util is not None else (gpu_stats_before.enc_util or 0.0)
        temp_c = gpu_stats_after.temperature_c or gpu_stats_before.temperature_c or 0.0

        results[algo] = {
            "elapsed_seconds": elapsed,
            "frames": frames,
            "fps": fps,
            "speed": speed_factor,
            "gpu_util": gpu_util,
            "enc_util": enc_util,
            "temperature_c": temp_c,
        }

    # Print summary table
    MigrationProgressBar.write("")
    MigrationProgressBar.write("=" * 80)
    MigrationProgressBar.write(f"{'Algorithm':<14} {'FPS':<10} {'Speed':<10} {'Time (s)':<12} {'GPU / NVENC':<14} {'vs Lanczos':<12}")
    MigrationProgressBar.write("-" * 80)

    lanczos_fps = results.get("lanczos", {}).get("fps", 0.0)

    for algo in algorithms:
        res = results.get(algo, {})
        if "error" in res:
            MigrationProgressBar.write(f"{algo:<14} ERROR: {res['error']}")
            continue

        fps_val = res.get("fps", 0.0)
        spd_val = res.get("speed", 0.0)
        el_val = res.get("elapsed_seconds", 0.0)
        gpu_str = f"{res.get('gpu_util', 0):.0f}% / {res.get('enc_util', 0):.0f}%"

        if lanczos_fps > 0 and algo != "lanczos":
            diff_pct = ((fps_val - lanczos_fps) / lanczos_fps) * 100.0
            diff_str = f"{diff_pct:+.1f}%"
        elif algo == "lanczos":
            diff_str = "baseline"
        else:
            diff_str = "--"

        MigrationProgressBar.write(f"{algo:<14} {fps_val:<10.1f} {spd_val:<9.2f}x {el_val:<11.2f}s {gpu_str:<14} {diff_str:<12}")

    MigrationProgressBar.write("=" * 80)
    MigrationProgressBar.write("Conclusion:")
    MigrationProgressBar.write("  - 'bicubic': Optimal balance of sharpness and high GPU throughput.")
    MigrationProgressBar.write("  - 'bilinear': Highest raw throughput with minimal GPU load.")
    MigrationProgressBar.write("  - 'lanczos': High-quality 8-tap scaling with higher computational cost.")
    MigrationProgressBar.write("=" * 80)

    return results
