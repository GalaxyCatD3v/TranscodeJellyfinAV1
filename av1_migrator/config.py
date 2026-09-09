"""
Galaxy AV1 Migrator - Configuration Module
Handles YAML configuration loading, default settings, parsing, and CLI overrides.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional
import yaml


def parse_size_to_bytes(size_str: str | int | float) -> int:
    """
    Parse a human-readable size string (e.g. '1TB', '10GB', '500MB', '1000000') into bytes.
    """
    if isinstance(size_str, (int, float)):
        return int(size_str)
    
    s = str(size_str).strip().upper()
    if not s:
        return 0
    
    multipliers = {
        "B": 1,
        "K": 1024,
        "KB": 1024,
        "KIB": 1024,
        "M": 1024**2,
        "MB": 1024**2,
        "MIB": 1024**2,
        "G": 1024**3,
        "GB": 1024**3,
        "GIB": 1024**3,
        "T": 1024**4,
        "TB": 1024**4,
        "TIB": 1024**4,
        "P": 1024**5,
        "PB": 1024**5,
        "PIB": 1024**5,
    }
    
    # Check suffixes in descending order of length
    for suffix in sorted(multipliers.keys(), key=len, reverse=True):
        if s.endswith(suffix):
            num_part = s[: -len(suffix)].strip()
            try:
                return int(float(num_part) * multipliers[suffix])
            except ValueError:
                pass
                
    try:
        return int(float(s))
    except ValueError:
        raise ValueError(f"Invalid size specification: '{size_str}'")


def format_bytes(bytes_val: int | float) -> str:
    """Format bytes into a human-readable string (e.g. 1.43 TB, 82.5 GB)."""
    if bytes_val < 0:
        return f"-{format_bytes(-bytes_val)}"
    
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    val = float(bytes_val)
    idx = 0
    while val >= 1024.0 and idx < len(units) - 1:
        val /= 1024.0
        idx += 1
    
    if idx == 0:
        return f"{int(val)} B"
    return f"{val:.2f} {units[idx]}"


@dataclass
class OutputConfig:
    container: str = "mkv"
    video_codec: str = "av1_nvenc"
    preset: str = "p5"
    cq: int = 28
    width: int = 1920
    height: int = 1080
    prefer_cuda_scale: bool = True
    cuda_interp_algo: str = "bicubic"  # "bicubic", "bilinear", or "lanczos"
    pixel_format: str = "p010le"
    cpu_video_codec: str = "auto"  # "auto", "libsvtav1", or "libaom-av1"
    cpu_preset: str = "6"
    cpu_crf: int = 28


@dataclass
class AudioConfig:
    languages: List[str] = field(default_factory=lambda: ["eng", "en", "english"])
    mode: str = "copy"  # "copy" or "aac"
    bitrate: str = "256k"
    exclude_commentary: bool = True


@dataclass
class SubtitlesConfig:
    languages: List[str] = field(default_factory=lambda: ["eng", "en", "english"])
    mode: str = "copy"


@dataclass
class StorageConfig:
    minimum_free_space: str = "1TB"
    safety_margin: str = "10GB"
    poll_interval: float = 0.5
    local_staging_dir: str = "F:\\JellyfinTranscode"
    enable_local_staging: bool = True
    local_min_free_space: str = "20GB"

    @property
    def minimum_free_space_bytes(self) -> int:
        return parse_size_to_bytes(self.minimum_free_space)

    @property
    def safety_margin_bytes(self) -> int:
        return parse_size_to_bytes(self.safety_margin)

    @property
    def local_min_free_space_bytes(self) -> int:
        return parse_size_to_bytes(self.local_min_free_space)


@dataclass
class ProcessingConfig:
    sort: str = "largest_first"  # "largest_first" or "smallest_first"
    delete_original: bool = True
    validate_output: bool = True
    keep_smaller: bool = True
    pre_encode_check: bool = True  # Toggleable pre-encode space optimization & bloat prediction check
    sample_duration: float = 30.0  # Duration in seconds of sample clip for space optimization test
    min_savings_percent: float = 0.0  # Minimum % space savings required to proceed (0.0 = must not bloat)
    enable_gpu_encoding: bool = True
    gpu_workers: int = 2  # Number of concurrent GPU workers (default: 2 to saturate dual NVENC engines)
    enable_cpu_encoding: bool = True
    cpu_max_file_size: Optional[str] = None
    resume: bool = True
    network_retries: int = 3
    network_retry_delay: float = 30.0
    duration_tolerance: float = 2.0  # seconds


@dataclass
class VideoSelectionConfig:
    prefer_native_1080p: bool = False


@dataclass
class FFmpegConfig:
    executable: str = "ffmpeg.exe"
    ffprobe: str = "ffprobe.exe"


@dataclass
class UIConfig:
    refresh_rate: float = 1.0


@dataclass
class DatabaseConfig:
    path: str = "migration.db"


@dataclass
class LoggingConfig:
    directory: str = "logs"
    filename: str = "av1-migrator.log"
    max_bytes: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5


@dataclass
class AppConfig:
    media_roots: List[str] = field(default_factory=lambda: ["Y:\\srv\\storage\\Movies", "Y:\\media\\TVShows"])
    extensions: List[str] = field(default_factory=lambda: [".mkv", ".mp4", ".m4v", ".m2ts", ".ts", ".avi", ".mov"])
    output: OutputConfig = field(default_factory=OutputConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    subtitles: SubtitlesConfig = field(default_factory=SubtitlesConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    video_selection: VideoSelectionConfig = field(default_factory=VideoSelectionConfig)
    ffmpeg: FFmpegConfig = field(default_factory=FFmpegConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(config_path: Optional[str | Path] = None) -> AppConfig:
    """
    Loads config from a YAML file if specified or found, otherwise returns default AppConfig.
    """
    config = AppConfig()
    
    target_path = None
    if config_path:
        target_path = Path(config_path)
    elif Path("config.yaml").is_file():
        target_path = Path("config.yaml")
    elif Path("config.yml").is_file():
        target_path = Path("config.yml")
        
    if target_path and target_path.is_file():
        with open(target_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
            
        if "media_roots" in raw_data:
            config.media_roots = [str(r) for r in raw_data["media_roots"]]
        if "extensions" in raw_data:
            config.extensions = [ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in raw_data["extensions"]]
            
        if "output" in raw_data and isinstance(raw_data["output"], dict):
            out_data = raw_data["output"]
            config.output = OutputConfig(
                container=out_data.get("container", config.output.container),
                video_codec=out_data.get("video_codec", config.output.video_codec),
                preset=out_data.get("preset", config.output.preset),
                cq=int(out_data.get("cq", config.output.cq)),
                width=int(out_data.get("width", config.output.width)),
                height=int(out_data.get("height", config.output.height)),
                prefer_cuda_scale=bool(out_data.get("prefer_cuda_scale", config.output.prefer_cuda_scale)),
                cuda_interp_algo=str(out_data.get("cuda_interp_algo", config.output.cuda_interp_algo)),
                pixel_format=out_data.get("pixel_format", config.output.pixel_format),
                cpu_video_codec=str(out_data.get("cpu_video_codec", config.output.cpu_video_codec)),
                cpu_preset=str(out_data.get("cpu_preset", config.output.cpu_preset)),
                cpu_crf=int(out_data.get("cpu_crf", config.output.cpu_crf)),
            )
            
        if "audio" in raw_data and isinstance(raw_data["audio"], dict):
            aud_data = raw_data["audio"]
            config.audio = AudioConfig(
                languages=[str(l).lower() for l in aud_data.get("languages", config.audio.languages)],
                mode=aud_data.get("mode", config.audio.mode),
                bitrate=aud_data.get("bitrate", config.audio.bitrate),
                exclude_commentary=bool(aud_data.get("exclude_commentary", config.audio.exclude_commentary)),
            )
            
        if "subtitles" in raw_data and isinstance(raw_data["subtitles"], dict):
            sub_data = raw_data["subtitles"]
            config.subtitles = SubtitlesConfig(
                languages=[str(l).lower() for l in sub_data.get("languages", config.subtitles.languages)],
                mode=sub_data.get("mode", config.subtitles.mode),
            )
            
        if "storage" in raw_data and isinstance(raw_data["storage"], dict):
            st_data = raw_data["storage"]
            config.storage = StorageConfig(
                minimum_free_space=str(st_data.get("minimum_free_space", config.storage.minimum_free_space)),
                safety_margin=str(st_data.get("safety_margin", config.storage.safety_margin)),
                poll_interval=float(st_data.get("poll_interval", config.storage.poll_interval)),
                local_staging_dir=str(st_data.get("local_staging_dir", config.storage.local_staging_dir)),
                enable_local_staging=bool(st_data.get("enable_local_staging", config.storage.enable_local_staging)),
                local_min_free_space=str(st_data.get("local_min_free_space", config.storage.local_min_free_space)),
            )
            
        if "processing" in raw_data and isinstance(raw_data["processing"], dict):
            pr_data = raw_data["processing"]
            config.processing = ProcessingConfig(
                sort=pr_data.get("sort", config.processing.sort),
                delete_original=bool(pr_data.get("delete_original", config.processing.delete_original)),
                validate_output=bool(pr_data.get("validate_output", config.processing.validate_output)),
                keep_smaller=bool(pr_data.get("keep_smaller", config.processing.keep_smaller)),
                pre_encode_check=bool(pr_data.get("pre_encode_check", config.processing.pre_encode_check)),
                sample_duration=float(pr_data.get("sample_duration", config.processing.sample_duration)),
                min_savings_percent=float(pr_data.get("min_savings_percent", config.processing.min_savings_percent)),
                enable_gpu_encoding=bool(pr_data.get("enable_gpu_encoding", config.processing.enable_gpu_encoding)),
                gpu_workers=int(pr_data.get("gpu_workers", config.processing.gpu_workers)),
                enable_cpu_encoding=bool(pr_data.get("enable_cpu_encoding", config.processing.enable_cpu_encoding)),
                cpu_max_file_size=pr_data.get("cpu_max_file_size", config.processing.cpu_max_file_size),
                resume=bool(pr_data.get("resume", config.processing.resume)),
                network_retries=int(pr_data.get("network_retries", config.processing.network_retries)),
                network_retry_delay=float(pr_data.get("network_retry_delay", config.processing.network_retry_delay)),
                duration_tolerance=float(pr_data.get("duration_tolerance", config.processing.duration_tolerance)),
            )
            
        if "video_selection" in raw_data and isinstance(raw_data["video_selection"], dict):
            vs_data = raw_data["video_selection"]
            config.video_selection = VideoSelectionConfig(
                prefer_native_1080p=bool(vs_data.get("prefer_native_1080p", config.video_selection.prefer_native_1080p))
            )
            
        if "ffmpeg" in raw_data and isinstance(raw_data["ffmpeg"], dict):
            ff_data = raw_data["ffmpeg"]
            config.ffmpeg = FFmpegConfig(
                executable=ff_data.get("executable", config.ffmpeg.executable),
                ffprobe=ff_data.get("ffprobe", config.ffmpeg.ffprobe),
            )
            
        if "ui" in raw_data and isinstance(raw_data["ui"], dict):
            ui_data = raw_data["ui"]
            config.ui = UIConfig(
                refresh_rate=float(ui_data.get("refresh_rate", config.ui.refresh_rate))
            )
            
        if "database" in raw_data and isinstance(raw_data["database"], dict):
            db_data = raw_data["database"]
            config.database = DatabaseConfig(
                path=db_data.get("path", config.database.path)
            )
            
        if "logging" in raw_data and isinstance(raw_data["logging"], dict):
            lg_data = raw_data["logging"]
            config.logging = LoggingConfig(
                directory=lg_data.get("directory", config.logging.directory),
                filename=lg_data.get("filename", config.logging.filename),
                max_bytes=int(lg_data.get("max_bytes", config.logging.max_bytes)),
                backup_count=int(lg_data.get("backup_count", config.logging.backup_count)),
            )

    return config
