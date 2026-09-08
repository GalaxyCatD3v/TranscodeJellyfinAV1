"""
FFprobe inspection, stream selection, HDR determination, and naming module.
"""

import json
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from av1_migrator.config import AppConfig
from av1_migrator.logger import get_logger
from av1_migrator.models import AudioStream, MediaFile, SubtitleStream, VideoStream
from av1_migrator.utils import find_binary_executable, sanitize_stem


AUDIO_CODEC_RANKS = {
    "truehd": 60,
    "dts-hd ma": 50,
    "dtshdma": 50,
    "dts-hd": 45,
    "eac3": 40,
    "e-ac-3": 40,
    "dts": 30,
    "ac3": 20,
    "ac-3": 20,
    "flac": 15,
    "opus": 12,
    "aac": 10,
    "mp3": 5,
    "vorbis": 5,
}

COMMENTARY_KEYWORDS = [
    "commentary",
    "director",
    "directors",
    "audio description",
    "description",
    "descriptive",
    "narration",
    "dvs",
]


def run_ffprobe_json(ffprobe_path: str, file_path: str | Path) -> Dict[str, Any]:
    """Runs ffprobe on the target file and returns the parsed JSON dict."""
    exe = find_binary_executable(ffprobe_path) or ffprobe_path
    cmd = [
        exe,
        "-v", "error",
        "-show_format",
        "-show_streams",
        "-of", "json",
        str(file_path),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60.0,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffprobe returned code {proc.returncode}: {proc.stderr.strip()}")
        return json.loads(proc.stdout)
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"ffprobe timed out on {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse ffprobe JSON output: {e}")


def is_hdr_metadata(color_transfer: str, color_primaries: str, color_space: str) -> bool:
    """
    Checks whether color metadata indicates HDR10.
    Standard HDR10 uses SMPTE ST 2084 (PQ) transfer and BT.2020 primaries/space.
    """
    transfer = (color_transfer or "").lower()
    primaries = (color_primaries or "").lower()
    space = (color_space or "").lower()

    if "smpte2084" in transfer or "arib-std-b67" in transfer:
        return True
    if "bt2020" in primaries:
        return True
    if space in ("bt2020nc", "bt2020_ncl", "bt2020c", "bt2020_cl"):
        return True
    return False


def is_english_language(lang_tag: Optional[str], accepted_languages: List[str]) -> bool:
    if not lang_tag:
        return False
    clean = lang_tag.strip().lower()
    for acc in accepted_languages:
        if clean == acc.lower() or clean.startswith(f"{acc.lower()}-"):
            return True
    return False


def is_commentary_stream(title: str, disposition: Dict[str, Any]) -> bool:
    if disposition.get("commentary", 0) == 1:
        return True
    if disposition.get("descriptions", 0) == 1 or disposition.get("visual_impaired", 0) == 1:
        return True
    t = title.lower()
    return any(kw in t for kw in COMMENTARY_KEYWORDS)


def select_video_stream(
    video_streams: List[VideoStream],
    prefer_native_1080p: bool = False,
) -> Optional[VideoStream]:
    """
    Selects the main video stream.
    Filters out attached picture / cover art streams.
    Priority:
      1. If prefer_native_1080p and native 1080p stream exists, choose it.
      2. Highest resolution (width * height).
      3. Default disposition flag.
      4. First index.
    """
    candidates = [v for v in video_streams if not v.is_attached_pic]
    if not candidates:
        return None

    if prefer_native_1080p:
        native_1080 = [v for v in candidates if v.width == 1920 or v.height == 1080]
        if native_1080:
            native_1080.sort(key=lambda v: (v.is_default, v.duration or 0), reverse=True)
            return native_1080[0]

    # Sort candidates by: (resolution, default flag, duration, -index)
    def sort_key(v: VideoStream):
        res = (v.width or 0) * (v.height or 0)
        return (res, 1 if v.is_default else 0, v.duration or 0, -v.index)

    candidates.sort(key=sort_key, reverse=True)
    return candidates[0]


def select_audio_streams(
    audio_streams: List[AudioStream],
    accepted_languages: List[str],
    exclude_commentary: bool = True,
) -> List[AudioStream]:
    """
    Selects English audio streams, prioritizing non-commentary and sorting by quality.
    If only commentary tracks exist, retains them.
    """
    eng_streams = [
        a for a in audio_streams
        if is_english_language(a.language, accepted_languages)
    ]
    if not eng_streams:
        return []

    if exclude_commentary:
        non_comm = [a for a in eng_streams if not a.is_commentary]
        if non_comm:
            eng_streams = non_comm

    def audio_quality_key(a: AudioStream):
        codec = a.codec_name.lower()
        rank = AUDIO_CODEC_RANKS.get(codec, 0)
        channels = a.channels or 2
        is_def = 1 if a.is_default else 0
        bit_rate = a.bit_rate or 0
        return (rank, channels, is_def, bit_rate)

    eng_streams.sort(key=audio_quality_key, reverse=True)
    return eng_streams


def select_subtitle_streams(
    subtitle_streams: List[SubtitleStream],
    accepted_languages: List[str],
) -> List[SubtitleStream]:
    """
    Selects all English subtitle streams.
    """
    return [
        s for s in subtitle_streams
        if is_english_language(s.language, accepted_languages)
    ]


def generate_output_paths(source_path: Path, is_hdr: bool, cq: int, container: str = "mkv") -> Tuple[Path, Path]:
    """
    Generates the target final output path and temporary `.encoding.mkv` path.
    Sanitizes the filename to ensure compatibility across both Windows and Linux.
    Strips existing tag to avoid duplication.
    Example:
      "Movie (2020) [AV1 1080p HDR10 CQ28].mkv"
      "Movie (2020) [AV1 1080p HDR10 CQ28].mkv.encoding.mkv"
    """
    tag_hdr = "HDR10" if is_hdr else "SDR"
    tag_str = f"[AV1 1080p {tag_hdr} CQ{cq}]"

    # Sanitize stem for cross-platform compatibility (Linux and Windows)
    clean_stem = sanitize_stem(source_path.stem)
    
    out_filename = f"{clean_stem} {tag_str}.{container.lstrip('.')}"
    final_output = source_path.parent / out_filename
    temp_output = source_path.parent / f"{out_filename}.encoding.{container.lstrip('.')}"
    return final_output, temp_output


def probe_and_populate_media_file(
    source_path: str | Path,
    config: AppConfig,
) -> MediaFile:
    """
    Probes the file using ffprobe and constructs a populated MediaFile instance.
    Determines stream selections and target paths.
    """
    path = Path(source_path)
    try:
        stat = path.stat()
        size = stat.st_size
        mtime = stat.st_mtime
    except Exception as e:
        mf = MediaFile(source=path, status="failed", error_reason=f"Failed to stat file: {e}")
        return mf

    media_file = MediaFile(source=path, size=size, mtime=mtime)

    try:
        data = run_ffprobe_json(config.ffmpeg.ffprobe, path)
    except Exception as e:
        media_file.status = "failed"
        media_file.error_reason = f"FFprobe error: {e}"
        return media_file

    format_data = data.get("format", {})
    streams_data = data.get("streams", [])

    try:
        duration_val = float(format_data.get("duration", 0))
        media_file.duration = duration_val if duration_val > 0 else None
    except (ValueError, TypeError):
        media_file.duration = None

    try:
        bit_rate_val = int(format_data.get("bit_rate", 0))
        media_file.bit_rate = bit_rate_val if bit_rate_val > 0 else None
    except (ValueError, TypeError):
        media_file.bit_rate = None

    for s in streams_data:
        codec_type = s.get("codec_type")
        disposition = s.get("disposition", {})
        tags = s.get("tags", {})
        # Normalize tag keys to lowercase
        norm_tags = {k.lower(): str(v) for k, v in tags.items()}
        lang = norm_tags.get("language") or norm_tags.get("lang") or "und"
        title = norm_tags.get("title", "")
        idx = int(s.get("index", 0))

        if codec_type == "video":
            c_name = s.get("codec_name", "")
            c_long = s.get("codec_long_name", "")
            w = int(s.get("width") or 0)
            h = int(s.get("height") or 0)
            pix = s.get("pix_fmt", "")
            prof = s.get("profile", "")
            c_space = s.get("color_space", "")
            c_trc = s.get("color_transfer", "")
            c_prim = s.get("color_primaries", "")
            is_hdr = is_hdr_metadata(c_trc, c_prim, c_space)
            is_def = disposition.get("default", 0) == 1
            is_attached = disposition.get("attached_pic", 0) == 1 or c_name in ("mjpeg", "png", "bmp") and (w == 0 or h == 0 or s.get("duration") == "0")

            try:
                st_dur = float(s.get("duration", 0))
            except (ValueError, TypeError):
                st_dur = None

            try:
                st_br = int(s.get("bit_rate", 0))
            except (ValueError, TypeError):
                st_br = None

            v_stream = VideoStream(
                index=idx,
                codec_name=c_name,
                codec_long_name=c_long,
                width=w,
                height=h,
                pix_fmt=pix,
                profile=prof,
                bit_rate=st_br,
                duration=st_dur,
                color_space=c_space,
                color_transfer=c_trc,
                color_primaries=c_prim,
                is_hdr=is_hdr,
                is_default=is_def,
                is_attached_pic=is_attached,
            )
            media_file.video_streams.append(v_stream)

        elif codec_type == "audio":
            c_name = s.get("codec_name", "")
            ch = int(s.get("channels") or 2)
            ch_lay = s.get("channel_layout", "")
            is_def = disposition.get("default", 0) == 1
            is_comm = is_commentary_stream(title, disposition)

            try:
                st_br = int(s.get("bit_rate", 0))
            except (ValueError, TypeError):
                st_br = None

            a_stream = AudioStream(
                index=idx,
                codec_name=c_name,
                language=lang,
                title=title,
                channels=ch,
                channel_layout=ch_lay,
                bit_rate=st_br,
                is_default=is_def,
                is_commentary=is_comm,
            )
            media_file.audio_streams.append(a_stream)

        elif codec_type == "subtitle":
            c_name = s.get("codec_name", "")
            is_def = disposition.get("default", 0) == 1
            is_forced = disposition.get("forced", 0) == 1
            is_hi = disposition.get("hearing_impaired", 0) == 1

            sub_stream = SubtitleStream(
                index=idx,
                codec_name=c_name,
                language=lang,
                title=title,
                is_default=is_def,
                is_forced=is_forced,
                is_hearing_impaired=is_hi,
            )
            media_file.subtitle_streams.append(sub_stream)

    # Select main video stream
    main_v = select_video_stream(
        media_file.video_streams,
        prefer_native_1080p=config.video_selection.prefer_native_1080p,
    )
    if not main_v:
        media_file.status = "skipped"
        media_file.skip_reason = "No valid video stream found"
        return media_file

    media_file.selected_video = main_v
    media_file.video_codec = main_v.codec_name
    media_file.width = main_v.width
    media_file.height = main_v.height
    media_file.hdr = main_v.is_hdr
    if not media_file.duration and main_v.duration:
        media_file.duration = main_v.duration

    # Select English audio streams
    selected_audio = select_audio_streams(
        media_file.audio_streams,
        accepted_languages=config.audio.languages,
        exclude_commentary=config.audio.exclude_commentary,
    )
    if not selected_audio:
        media_file.status = "skipped"
        media_file.skip_reason = "No English audio track found"
        return media_file

    media_file.selected_audio = selected_audio

    # Select English subtitle streams
    selected_subs = select_subtitle_streams(
        media_file.subtitle_streams,
        accepted_languages=config.subtitles.languages,
    )
    media_file.selected_subtitles = selected_subs

    # Check if already AV1 1080p
    final_out, temp_out = generate_output_paths(
        path,
        is_hdr=media_file.hdr,
        cq=config.output.cq,
        container=config.output.container,
    )
    media_file.output_path = final_out
    media_file.temp_output_path = temp_out

    # If the source itself is already AV1 at 1080p and has the target container
    if (
        main_v.codec_name.lower() == "av1"
        and main_v.width <= config.output.width
        and main_v.height <= config.output.height
        and path.suffix.lower() == f".{config.output.container.lstrip('.')}"
    ):
        media_file.status = "completed"
        media_file.skip_reason = "Source is already AV1 1080p"
        return media_file

    media_file.status = "pending"
    return media_file
