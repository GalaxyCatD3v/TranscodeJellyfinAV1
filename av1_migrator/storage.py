"""
Storage safety and disk space monitoring for Galaxy AV1 Migrator.
Provides pre-flight space checks and a high-frequency independent background thread monitor.
"""

from pathlib import Path
import shutil
import threading
import time
from typing import Callable, Optional
from av1_migrator.config import format_bytes
from av1_migrator.logger import get_logger
from av1_migrator.models import StorageStats


def get_path_disk_usage(
    path: str | Path,
    fallback_stats: Optional[StorageStats] = None,
) -> StorageStats:
    """
    Get disk usage for the given path with robust error tolerance for SSHFS/network mounts.
    Queries the volume/anchor root to avoid expensive, failing stat calls on deep remote files.
    """
    logger = get_logger()
    try:
        p = Path(path)
        # On Windows / UNC, use anchor (e.g. 'Y:\\' or '\\\\server\\share\\') to query volume directly
        if p.anchor and (p.drive or str(path).startswith("\\\\")):
            target = p.anchor
        else:
            # On POSIX or relative paths, use parent if path looks like a file
            target = str(p.parent) if p.suffix else str(p)
    except Exception:
        target = str(path)

    try:
        total, used, free = shutil.disk_usage(target)
        return StorageStats(
            free_bytes=free,
            total_bytes=total,
            used_bytes=used,
        )
    except (OSError, PermissionError, ValueError) as e:
        # Fallback attempt with anchor if different
        try:
            anchor = Path(path).anchor
            if anchor and anchor != target:
                total, used, free = shutil.disk_usage(anchor)
                return StorageStats(
                    free_bytes=free,
                    total_bytes=total,
                    used_bytes=used,
                )
        except Exception:
            pass

        # If fallback_stats provided and valid, return that
        if fallback_stats is not None and fallback_stats.total_bytes > 0:
            return fallback_stats

        logger.debug(f"Could not get disk usage for {path} (target: {target}): {e}")
        return StorageStats(free_bytes=0, total_bytes=0, used_bytes=0)


def check_storage_safety(
    path: str | Path,
    original_file_size: int,
    minimum_free_space_bytes: int,
    safety_margin_bytes: int,
) -> tuple[bool, str, StorageStats]:
    """
    Checks if it is safe to start encoding.
    Formula: required_free = original_file_size + safety_margin + minimum_free_space
    """
    stats = get_path_disk_usage(path)
    stats.minimum_free_bytes = minimum_free_space_bytes
    stats.safety_margin_bytes = safety_margin_bytes

    required_free = original_file_size + safety_margin_bytes + minimum_free_space_bytes
    is_safe = stats.free_bytes >= required_free

    if is_safe:
        msg = (
            f"Storage check passed: Free {format_bytes(stats.free_bytes)} >= "
            f"Required {format_bytes(required_free)} "
            f"(Min floor: {format_bytes(minimum_free_space_bytes)} + Original: {format_bytes(original_file_size)} + Margin: {format_bytes(safety_margin_bytes)})"
        )
    else:
        msg = (
            f"STORAGE SAFETY CHECK FAILED: Free {format_bytes(stats.free_bytes)} < "
            f"Required {format_bytes(required_free)} "
            f"(Min floor: {format_bytes(minimum_free_space_bytes)} + Original: {format_bytes(original_file_size)} + Margin: {format_bytes(safety_margin_bytes)})"
        )

    return is_safe, msg, stats


class StorageMonitorThread:
    """
    Independent background monitor thread that checks free space every poll_interval seconds.
    If free space falls below minimum_free_space_bytes, immediately invokes on_emergency_stop callback.
    Resilient against transient SSHFS / network I/O errors (e.g. WinError 1450).
    """

    def __init__(
        self,
        watch_path: str | Path,
        minimum_free_space_bytes: int,
        poll_interval: float = 0.5,
        on_emergency_stop: Optional[Callable[[StorageStats], None]] = None,
    ):
        self.watch_path = str(watch_path)
        self.minimum_free_space_bytes = minimum_free_space_bytes
        self.poll_interval = poll_interval
        self.on_emergency_stop = on_emergency_stop

        # Determine target path once for efficiency
        try:
            p = Path(watch_path)
            if p.anchor and (p.drive or str(watch_path).startswith("\\\\")):
                self._disk_target = p.anchor
            else:
                self._disk_target = str(p.parent) if p.suffix else str(p)
        except Exception:
            self._disk_target = str(watch_path)

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_stats = StorageStats()
        self.is_emergency_triggered = False
        self._consecutive_errors = 0

    def start(self) -> None:
        self._stop_event.clear()
        self.is_emergency_triggered = False
        self._consecutive_errors = 0
        self._thread = threading.Thread(target=self._run, name="StorageMonitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        logger = get_logger()
        while not self._stop_event.is_set():
            try:
                stats = get_path_disk_usage(self._disk_target, fallback_stats=self.last_stats)
                stats.minimum_free_bytes = self.minimum_free_space_bytes

                if stats.total_bytes > 0:
                    self.last_stats = stats
                    self._consecutive_errors = 0

                    if stats.free_bytes < self.minimum_free_space_bytes:
                        self.is_emergency_triggered = True
                        logger.critical(
                            f"!!! STORAGE SAFETY STOP TRIGGERED !!! "
                            f"Free space: {format_bytes(stats.free_bytes)} < Minimum: {format_bytes(self.minimum_free_space_bytes)}"
                        )
                        if self.on_emergency_stop:
                            self.on_emergency_stop(stats)
                        break
            except Exception as e:
                self._consecutive_errors += 1
                if self._consecutive_errors in (5, 20, 60):
                    logger.warning(f"Storage monitor transient error ({e}) for {self._disk_target}")
                else:
                    logger.debug(f"Storage monitor error: {e}")

            time.sleep(self.poll_interval)
