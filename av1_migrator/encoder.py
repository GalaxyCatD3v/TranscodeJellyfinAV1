"""
FFmpeg encoding engine for Galaxy AV1 Migrator.
Constructs safe argument lists, executes av1_nvenc, parses real-time progress, and handles emergency aborts.
"""

from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple
from av1_migrator.config import AppConfig
from av1_migrator.logger import get_logger
from av1_migrator.models import EncodeProgress, MediaFile
from av1_migrator.utils import find_binary_executable


def check_ffmpeg_feature(executable: str, feature_flag: str) -> bool:
    """Checks if ffmpeg supports a specific encoder, filter, or hardware acceleration."""
    try:
        proc = subprocess.run(
            [executable, "-hide_banner", feature_flag],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5.0,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


def is_av1_nvenc_available(ffmpeg_path: str = "ffmpeg.exe") -> bool:
    """Checks if av1_nvenc encoder is compiled into ffmpeg."""
    exe = find_binary_executable(ffmpeg_path) or ffmpeg_path
    try:
        proc = subprocess.run(
            [exe, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5.0,
            check=False,
        )
        return "av1_nvenc" in proc.stdout
    except Exception:
        return False


def is_scale_cuda_available(ffmpeg_path: str = "ffmpeg.exe") -> bool:
    """Checks if scale_cuda filter is available in ffmpeg."""
    exe = find_binary_executable(ffmpeg_path) or ffmpeg_path
    try:
        proc = subprocess.run(
            [exe, "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5.0,
            check=False,
        )
        return "scale_cuda" in proc.stdout
    except Exception:
        return False


def build_ffmpeg_command(
    media_file: MediaFile,
    config: AppConfig,
    use_cuda_scale: bool = True,
) -> List[str]:
    """
    Constructs the list of arguments for FFmpeg.
    NEVER use shell=True. Filenames and paths are passed verbatim as list elements.
    """
    exe = find_binary_executable(config.ffmpeg.executable) or config.ffmpeg.executable
    cmd = [
        exe,
        "-y",
        "-nostdin",
        "-hide_banner",
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

    # Video Filter: CUDA vs Software Lanczos
    target_w = config.output.width
    target_h = config.output.height
    
    if use_cuda_scale:
        vf_filter = f"hwupload_cuda,scale_cuda={target_w}:{target_h}:interp_algo=lanczos"
    else:
        vf_filter = f"scale={target_w}:{target_h}:flags=lanczos"

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

    # Progress reporting
    cmd.extend([
        "-progress", "pipe:1",
        "-stats_period", "0.5",
    ])

    # Temporary output target
    temp_target = media_file.temp_output_path or (media_file.source.parent / f"{media_file.source.stem}.encoding.mkv")
    cmd.append(str(temp_target))

    return cmd


def parse_out_time_to_seconds(time_str: str) -> float:
    """Parses FFmpeg out_time (HH:MM:SS.micro) into seconds."""
    try:
        parts = time_str.strip().split(":")
        if len(parts) == 3:
            h = float(parts[0])
            m = float(parts[1])
            s = float(parts[2])
            return h * 3600 + m * 60 + s
    except Exception:
        pass
    return 0.0


class FFmpegEncoder:
    """
    Manages the lifecycle of an FFmpeg encoding job.
    """
    def __init__(
        self,
        media_file: MediaFile,
        config: AppConfig,
        on_progress: Optional[Callable[[EncodeProgress], None]] = None,
    ):
        self.media_file = media_file
        self.config = config
        self.on_progress = on_progress

        self.process: Optional[subprocess.Popen] = None
        self.progress = EncodeProgress()
        self.is_aborted = False
        self.abort_reason: Optional[str] = None
        self.return_code: Optional[int] = None
        self._lock = threading.Lock()

    def run(self) -> Tuple[bool, str]:
        """
        Executes FFmpeg. First tries CUDA scaling (if enabled/available), falls back to software scale if needed.
        """
        logger = get_logger()
        temp_file = self.media_file.temp_output_path
        if not temp_file:
            return False, "No temporary output path specified"

        # If a leftover temp file already exists, clean it up before encoding
        if temp_file.exists():
            try:
                temp_file.unlink()
                logger.info(f"Removed stale temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to remove stale temporary file {temp_file}: {e}")

        # Check CUDA scaling capability
        use_cuda = self.config.output.prefer_cuda_scale and is_scale_cuda_available(self.config.ffmpeg.executable)

        cmd = build_ffmpeg_command(self.media_file, self.config, use_cuda_scale=use_cuda)
        logger.info(f"Starting FFmpeg encode for {self.media_file.source.name} (CUDA scale: {use_cuda})")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")

        success, msg = self._execute_process(cmd)

        # If failed and was using CUDA scaling, attempt fallback to software scaling
        if not success and not self.is_aborted and use_cuda:
            logger.warning("CUDA scaling failed, retrying with software Lanczos scaling fallback...")
            # Clean partial temp output
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass

            fallback_cmd = build_ffmpeg_command(self.media_file, self.config, use_cuda_scale=False)
            logger.info(f"Retrying FFmpeg encode for {self.media_file.source.name} with software scale")
            success, msg = self._execute_process(fallback_cmd)

        if not success:
            # Clean up temporary output on failure
            if temp_file.exists():
                try:
                    temp_file.unlink()
                    logger.info(f"Cleaned up temporary file after failure: {temp_file}")
                except Exception as e:
                    logger.error(f"Failed to delete temp file {temp_file}: {e}")

        return success, msg

    def _execute_process(self, cmd: List[str]) -> Tuple[bool, str]:
        logger = get_logger()
        total_duration = self.media_file.duration or 0.0
        start_time = time.time()

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                universal_newlines=True,
            )
        except Exception as e:
            return False, f"Failed to spawn FFmpeg process: {e}"

        stderr_lines: List[str] = []

        def read_stderr():
            if self.process and self.process.stderr:
                for line in self.process.stderr:
                    stderr_lines.append(line)
                    if len(stderr_lines) > 50:
                        stderr_lines.pop(0)

        err_thread = threading.Thread(target=read_stderr, daemon=True)
        err_thread.start()

        # Parse stdout (progress pipe)
        try:
            if self.process.stdout:
                for line in self.process.stdout:
                    if self.is_aborted:
                        break
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    if "=" in line:
                        key, _, val = line.partition("=")
                        key = key.strip()
                        val = val.strip()

                        if key == "frame":
                            try:
                                self.progress.frame = int(val)
                            except ValueError:
                                pass
                        elif key == "fps":
                            try:
                                self.progress.fps = float(val)
                            except ValueError:
                                pass
                        elif key == "bitrate":
                            # e.g. "8200.5kbits/s" or "N/A"
                            m = re.search(r"([\d\.]+)", val)
                            if m:
                                try:
                                    self.progress.bitrate_kbs = float(m.group(1))
                                except ValueError:
                                    pass
                        elif key == "total_size":
                            try:
                                self.progress.total_size_bytes = int(val)
                            except ValueError:
                                pass
                        elif key == "out_time":
                            secs = parse_out_time_to_seconds(val)
                            self.progress.out_time_s = secs
                            if total_duration > 0:
                                self.progress.percent = min(100.0, (secs / total_duration) * 100.0)
                        elif key == "speed":
                            # e.g. "2.31x"
                            m = re.search(r"([\d\.]+)", val)
                            if m:
                                try:
                                    speed = float(m.group(1))
                                    self.progress.speed = speed
                                    if speed > 0 and total_duration > self.progress.out_time_s:
                                        rem_sec = (total_duration - self.progress.out_time_s) / speed
                                        self.progress.eta_seconds = rem_sec
                                except ValueError:
                                    pass
                        elif key == "progress":
                            if self.on_progress:
                                try:
                                    self.on_progress(self.progress)
                                except Exception as cb_err:
                                    logger.debug(f"Progress callback exception: {cb_err}")

            self.process.wait()
            self.return_code = self.process.returncode

            if self.is_aborted:
                return False, f"Aborted: {self.abort_reason}"

            if self.return_code == 0:
                self.progress.percent = 100.0
                if self.on_progress:
                    try:
                        self.on_progress(self.progress)
                    except Exception as cb_err:
                        logger.debug(f"Progress callback exception: {cb_err}")
                return True, "Encode completed successfully"
            else:
                err_msg = "".join(stderr_lines[-10:]).strip()
                return False, f"FFmpeg exited with code {self.return_code}: {err_msg}"

        except Exception as e:
            self.abort(f"Process read error: {e}")
            return False, f"Exception during encode: {e}"

    def abort(self, reason: str = "User cancelled or safety stop") -> None:
        """Immediately terminates the FFmpeg process."""
        self.is_aborted = True
        self.abort_reason = reason
        logger = get_logger()
        logger.warning(f"Aborting FFmpeg encode: {reason}")

        with self._lock:
            if self.process and self.process.poll() is None:
                try:
                    self.process.terminate()
                    # Wait up to 3 seconds before killing
                    for _ in range(30):
                        if self.process.poll() is not None:
                            break
                        time.sleep(0.1)
                    if self.process.poll() is None:
                        self.process.kill()
                except Exception as e:
                    logger.error(f"Error while terminating FFmpeg: {e}")
