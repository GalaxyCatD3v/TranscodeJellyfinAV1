"""
Data models for Galaxy AV1 Migrator.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from av1_migrator.utils import format_seconds


@dataclass
class VideoStream:
    index: int
    codec_name: str
    codec_long_name: str = ""
    width: int = 0
    height: int = 0
    pix_fmt: str = ""
    profile: str = ""
    bit_rate: Optional[int] = None
    duration: Optional[float] = None
    color_space: str = ""
    color_transfer: str = ""
    color_primaries: str = ""
    is_hdr: bool = False
    is_default: bool = False
    is_attached_pic: bool = False


@dataclass
class AudioStream:
    index: int
    codec_name: str
    language: str = "und"
    title: str = ""
    channels: int = 2
    channel_layout: str = ""
    bit_rate: Optional[int] = None
    is_default: bool = False
    is_commentary: bool = False


@dataclass
class SubtitleStream:
    index: int
    codec_name: str
    language: str = "und"
    title: str = ""
    is_default: bool = False
    is_forced: bool = False
    is_hearing_impaired: bool = False


@dataclass
class MediaFile:
    source: Path
    size: int = 0
    mtime: float = 0.0
    duration: Optional[float] = None
    video_codec: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    hdr: bool = False
    bit_rate: Optional[int] = None
    
    video_streams: List[VideoStream] = field(default_factory=list)
    audio_streams: List[AudioStream] = field(default_factory=list)
    subtitle_streams: List[SubtitleStream] = field(default_factory=list)
    
    selected_video: Optional[VideoStream] = None
    selected_audio: List[AudioStream] = field(default_factory=list)
    selected_subtitles: List[SubtitleStream] = field(default_factory=list)
    
    output_path: Optional[Path] = None
    temp_output_path: Optional[Path] = None
    
    status: str = "pending"  # pending, encoding, validating, completed, skipped, failed, aborted
    skip_reason: Optional[str] = None
    error_reason: Optional[str] = None
    output_size: Optional[int] = None
    encoded_duration: Optional[float] = None


@dataclass
class GPUStats:
    name: str = "N/A"
    gpu_util: Optional[float] = None
    enc_util: Optional[float] = None
    mem_util: Optional[float] = None
    vram_used_mb: Optional[float] = None
    vram_total_mb: Optional[float] = None
    temperature_c: Optional[float] = None
    power_w: Optional[float] = None
    available: bool = False


@dataclass
class EncodeProgress:
    frame: int = 0
    fps: float = 0.0
    q: float = 0.0
    bitrate_kbs: float = 0.0
    total_size_bytes: int = 0
    out_time_s: float = 0.0
    speed: float = 0.0
    percent: float = 0.0
    eta_seconds: Optional[float] = None

    @property
    def eta_str(self) -> str:
        return format_seconds(self.eta_seconds)


@dataclass
class StorageStats:
    free_bytes: int = 0
    total_bytes: int = 0
    used_bytes: int = 0
    minimum_free_bytes: int = 0
    safety_margin_bytes: int = 0
    
    @property
    def buffer_bytes(self) -> int:
        return max(0, self.free_bytes - self.minimum_free_bytes)


@dataclass
class ScanStats:
    directories_scanned: int = 0
    files_discovered: int = 0
    files_skipped: int = 0
    access_errors: int = 0
    probe_errors: int = 0
    already_converted: int = 0
    eligible_files: int = 0
