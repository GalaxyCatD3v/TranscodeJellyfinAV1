"""
Pre-encode optimization and estimation module for Galaxy AV1 Migrator.
Performs fast analytical heuristics and short sample transcodes to predict whether
a transcode will achieve positive space savings before performing a full encode.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Optional, Tuple

from av1_migrator.config import AppConfig, format_bytes
from av1_migrator.encoder import check_ffmpeg_feature, is_scale_cuda_available
from av1_migrator.logger import get_logger
from av1_migrator.models import MediaFile
from av1_migrator.utils import find_binary_executable


@dataclass
class OptimizationResult:
    """Result of pre-encode optimization and space savings analysis."""
    should_encode: bool
    estimated_size_bytes: int
    estimated_savings_bytes: int
    estimated_savings_percent: float
    reason: str
    method: str  # "sample_test", "analytical_heuristic", "disabled", "short_clip", "fallback"


def build_sample_ffmpeg_command(
    media_file: MediaFile,
    config: AppConfig,
    sample_output_path: Path,
    seek_offset: float = 30.0,
    duration: float = 30.0,
    use_cuda_scale: bool = True,
) -> list[str]:
    """
    Builds FFmpeg command for encoding a short representative test sample.
    """
    exe = find_binary_executable(config.ffmpeg.executable) or config.ffmpeg.executable
    cmd = [
        exe,
        "-y",
        "-nostdin",
        "-hide_banner",
        "-ss", f"{seek_offset:.2f}",
        "-t", f"{duration:.2f}",
        "-i", str(media_file.source),
    ]

    # Map Video
    if media_file.selected_video:
        cmd.extend(["-map", f"0:{media_file.selected_video.index}"])
    else:
        cmd.extend(["-map", "0:v:0"])

    # Map Audio
    if media_file.selected_audio:
        for a in media_file.selected_audio:
            cmd.extend(["-map", f"0:{a.index}"])
    else:
        cmd.extend(["-map", "0:a"])

    # Map Subtitles
    if media_file.selected_subtitles:
        for s in media_file.selected_subtitles:
            cmd.extend(["-map", f"0:{s.index}"])

    # Video Filter: CUDA vs Software Scaling
    target_w = config.output.width
    target_h = config.output.height
    interp = getattr(config.output, "cuda_interp_algo", "bicubic")
    if use_cuda_scale:
        vf_filter = f"hwupload_cuda,scale_cuda={target_w}:{target_h}:interp_algo={interp}"
    else:
        vf_filter = f"scale={target_w}:{target_h}:flags={interp}"

    cmd.extend(["-vf", vf_filter])

    # Video Codec settings
    cmd.extend([
        "-c:v", config.output.video_codec,
        "-preset", config.output.preset,
        "-cq", str(config.output.cq),
        "-pix_fmt", config.output.pixel_format,
    ])

    # Color metadata
    if media_file.hdr:
        cmd.extend([
            "-color_primaries", "bt2020",
            "-color_trc", "smpte2084",
            "-colorspace", "bt2020nc",
        ])
    else:
        cmd.extend([
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
        ])

    # Audio handling
    if config.audio.mode == "copy":
        cmd.extend(["-c:a", "copy"])
    else:
        cmd.extend(["-c:a", config.audio.mode, "-b:a", config.audio.bitrate])

    # Subtitle handling
    cmd.extend(["-c:s", config.subtitles.mode])

    # Metadata & Chapters
    cmd.extend([
        "-map_metadata", "0",
        "-map_chapters", "0",
    ])

    cmd.append(str(sample_output_path))
    return cmd


def run_pre_encode_sample_test(
    media_file: MediaFile,
    config: AppConfig,
) -> OptimizationResult:
    """
    Encodes a short snippet of the video to accurately predict output file size and savings.
    """
    logger = get_logger()
    total_dur = media_file.duration or 0.0
    sample_dur = config.processing.sample_duration

    # If the clip is very short, sampling is unnecessary; proceed to full encode
    if total_dur <= 0 or total_dur < (sample_dur * 1.5):
        return OptimizationResult(
            should_encode=True,
            estimated_size_bytes=int(media_file.size * 0.5),
            estimated_savings_bytes=int(media_file.size * 0.5),
            estimated_savings_percent=50.0,
            reason="Clip too short for sampling; proceeding to full encode",
            method="short_clip",
        )

    # Calculate seek offset (15% into the video or 30s min, capped at total_dur - sample_dur)
    seek_offset = min(total_dur * 0.15, max(10.0, total_dur - sample_dur - 5.0))
    actual_sample_dur = min(sample_dur, max(5.0, total_dur - seek_offset))

    # Use local temp directory for fast I/O
    temp_dir = Path(tempfile.gettempdir())
    sample_file = temp_dir / f"av1_sample_{os.getpid()}_{int(time.time() * 1000)}.mkv"

    use_cuda = config.output.prefer_cuda_scale and is_scale_cuda_available(config.ffmpeg.executable)
    cmd = build_sample_ffmpeg_command(
        media_file,
        config,
        sample_file,
        seek_offset=seek_offset,
        duration=actual_sample_dur,
        use_cuda_scale=use_cuda,
    )

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=45.0,
            check=False,
        )

        # Fallback to software Lanczos scale if CUDA scaling fails
        if proc.returncode != 0 and use_cuda:
            logger.debug(f"CUDA scaling failed in sample test for {media_file.source.name}, retrying software scale")
            cmd_sw = build_sample_ffmpeg_command(
                media_file,
                config,
                sample_file,
                seek_offset=seek_offset,
                duration=actual_sample_dur,
                use_cuda_scale=False,
            )
            proc = subprocess.run(
                cmd_sw,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=45.0,
                check=False,
            )

        if proc.returncode != 0 or not sample_file.is_file():
            logger.warning(f"Sample transcode test failed for {media_file.source.name}: {proc.stderr[-200:] if proc.stderr else 'unknown error'}")
            return OptimizationResult(
                should_encode=True,
                estimated_size_bytes=int(media_file.size * 0.5),
                estimated_savings_bytes=int(media_file.size * 0.5),
                estimated_savings_percent=50.0,
                reason="Sample test failed, defaulting to full encode",
                method="fallback",
            )

        sample_size = sample_file.stat().st_size
        if sample_size <= 0:
            return OptimizationResult(
                should_encode=True,
                estimated_size_bytes=int(media_file.size * 0.5),
                estimated_savings_bytes=int(media_file.size * 0.5),
                estimated_savings_percent=50.0,
                reason="Sample output was 0 bytes, defaulting to full encode",
                method="fallback",
            )

        # Extrapolate full file size
        estimated_total_bytes = int((sample_size / actual_sample_dur) * total_dur)
        estimated_savings = media_file.size - estimated_total_bytes
        savings_pct = (estimated_savings / media_file.size) * 100.0 if media_file.size > 0 else 0.0

        min_savings_req = config.processing.min_savings_percent

        if config.processing.keep_smaller and estimated_total_bytes >= media_file.size:
            reason = (
                f"Sample test predicted bloat: estimated output ({format_bytes(estimated_total_bytes)}) "
                f">= original ({format_bytes(media_file.size)}) [{savings_pct:+.1f}% space change]"
            )
            return OptimizationResult(
                should_encode=False,
                estimated_size_bytes=estimated_total_bytes,
                estimated_savings_bytes=estimated_savings,
                estimated_savings_percent=savings_pct,
                reason=reason,
                method="sample_test",
            )
        elif savings_pct < min_savings_req:
            reason = (
                f"Sample test predicted insufficient savings: {savings_pct:.1f}% "
                f"< required {min_savings_req:.1f}% ({format_bytes(estimated_total_bytes)} vs {format_bytes(media_file.size)})"
            )
            return OptimizationResult(
                should_encode=False,
                estimated_size_bytes=estimated_total_bytes,
                estimated_savings_bytes=estimated_savings,
                estimated_savings_percent=savings_pct,
                reason=reason,
                method="sample_test",
            )
        else:
            reason = (
                f"Sample test predicts savings of {format_bytes(estimated_savings)} "
                f"({savings_pct:.1f}% reduction, estimated size: {format_bytes(estimated_total_bytes)})"
            )
            return OptimizationResult(
                should_encode=True,
                estimated_size_bytes=estimated_total_bytes,
                estimated_savings_bytes=estimated_savings,
                estimated_savings_percent=savings_pct,
                reason=reason,
                method="sample_test",
            )

    except subprocess.TimeoutExpired:
        logger.warning(f"Sample test timed out for {media_file.source.name}")
        return OptimizationResult(
            should_encode=True,
            estimated_size_bytes=int(media_file.size * 0.5),
            estimated_savings_bytes=int(media_file.size * 0.5),
            estimated_savings_percent=50.0,
            reason="Sample test timed out, defaulting to full encode",
            method="fallback",
        )
    except Exception as e:
        logger.warning(f"Error during sample test for {media_file.source.name}: {e}")
        return OptimizationResult(
            should_encode=True,
            estimated_size_bytes=int(media_file.size * 0.5),
            estimated_savings_bytes=int(media_file.size * 0.5),
            estimated_savings_percent=50.0,
            reason=f"Sample test error ({e}), defaulting to full encode",
            method="fallback",
        )
    finally:
        if sample_file.exists():
            try:
                sample_file.unlink()
            except Exception:
                pass


def evaluate_pre_encode_optimization(
    media_file: MediaFile,
    config: AppConfig,
) -> OptimizationResult:
    """
    Main evaluation entry point. Checks whether pre-encode analysis is enabled,
    and runs sample test to decide whether to proceed with encoding.
    """
    if not config.processing.pre_encode_check:
        return OptimizationResult(
            should_encode=True,
            estimated_size_bytes=int(media_file.size * 0.4),
            estimated_savings_bytes=int(media_file.size * 0.6),
            estimated_savings_percent=60.0,
            reason="Pre-encode check disabled",
            method="disabled",
        )

    return run_pre_encode_sample_test(media_file, config)
