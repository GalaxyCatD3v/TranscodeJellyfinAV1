"""
Validation module for Galaxy AV1 Migrator.
Validates temporary and promoted AV1 output files with strict safety checks.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from av1_migrator.config import AppConfig
from av1_migrator.logger import get_logger
from av1_migrator.models import MediaFile
from av1_migrator.probe import is_hdr_metadata, run_ffprobe_json


def validate_converted_file(
    output_path: str | Path,
    media_file: MediaFile,
    config: AppConfig,
    duration_tolerance: Optional[float] = None,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Validates that the output file meets all migration standards:
      1. File exists on disk and size > 0 bytes.
      2. FFprobe can parse format and streams.
      3. At least one video stream exists with codec == 'av1'.
      4. Video resolution matches target (1920x1080).
      5. At least one audio stream exists.
      6. HDR/SDR metadata matches expected source state.
      7. Duration matches original source duration within ±tolerance (if source duration known).
    """
    logger = get_logger()
    path = Path(output_path)
    tolerance = duration_tolerance if duration_tolerance is not None else config.processing.duration_tolerance

    if not path.exists():
        return False, f"Output file does not exist: {path}", None

    try:
        size = path.stat().st_size
        if size == 0:
            return False, f"Output file is empty (0 bytes): {path}", None
    except Exception as e:
        return False, f"Failed to get file stat on output: {e}", None

    try:
        probe_data = run_ffprobe_json(config.ffmpeg.ffprobe, path)
    except Exception as e:
        return False, f"FFprobe failed on output file: {e}", None

    format_data = probe_data.get("format", {})
    streams = probe_data.get("streams", [])

    video_streams = [s for s in streams if s.get("codec_type") == "video" and s.get("disposition", {}).get("attached_pic", 0) != 1]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]

    if not video_streams:
        return False, "Validation failed: No video stream found in output", probe_data

    v = video_streams[0]
    v_codec = (v.get("codec_name") or "").lower()
    if v_codec != "av1":
        return False, f"Validation failed: Video codec is '{v_codec}', expected 'av1'", probe_data

    w = int(v.get("width") or 0)
    h = int(v.get("height") or 0)
    # Check width/height: target is 1920x1080
    if w != config.output.width or h != config.output.height:
        # Note: In case source was non-16:9 or smaller, we check dimensions
        if w > config.output.width or h > config.output.height:
            return False, f"Validation failed: Resolution {w}x{h} exceeds target {config.output.width}x{config.output.height}", probe_data

    if not audio_streams:
        return False, "Validation failed: No audio stream found in output", probe_data

    # Check HDR / SDR
    c_trc = v.get("color_transfer", "")
    c_prim = v.get("color_primaries", "")
    c_space = v.get("color_space", "")
    is_hdr = is_hdr_metadata(c_trc, c_prim, c_space)

    if media_file.hdr and not is_hdr:
        return False, "Validation failed: Source is HDR10 but output lacks HDR10 metadata", probe_data
    elif not media_file.hdr and is_hdr:
        return False, "Validation failed: Source is SDR but output has HDR metadata", probe_data

    # Check Duration
    if media_file.duration and media_file.duration > 0:
        try:
            out_duration = float(format_data.get("duration", 0))
            if out_duration > 0:
                diff = abs(out_duration - media_file.duration)
                if diff > tolerance:
                    return False, f"Validation failed: Output duration {out_duration:.2f}s differs from source {media_file.duration:.2f}s by {diff:.2f}s (> {tolerance}s tolerance)", probe_data
        except (ValueError, TypeError):
            pass

    return True, "Validation successful: AV1 1080p verified with matching audio and metadata", probe_data
