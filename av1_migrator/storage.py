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


def get_path_disk_usage(path: str | Path) -> StorageStats:
    """
    Get disk usage for the given path.
    Finds the closest existing parent if the path itself does not exist yet.
    """
    p = Path(path).resolve()
    while not p.exists() and p.parent != p:
        p = p.parent

    try:
        total, used, free = shutil.disk_usage(str(p))
    except Exception as e:
        get_logger().warning(f"Could not get disk usage for {p}: {e}")
        # Return fallback zeros
        total, used, free = 0, 0, 0

    return StorageStats(
        free_bytes=free,
        total_bytes=total,
        used_bytes=used,
    )


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
        
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_stats = StorageStats()
        self.is_emergency_triggered = False

    def start(self) -> None:
        self._stop_event.clear()
        self.is_emergency_triggered = False
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
                stats = get_path_disk_usage(self.watch_path)
                stats.minimum_free_bytes = self.minimum_free_space_bytes
                self.last_stats = stats

                if stats.total_bytes > 0 and stats.free_bytes < self.minimum_free_space_bytes:
                    self.is_emergency_triggered = True
                    logger.critical(
                        f"!!! STORAGE SAFETY STOP TRIGGERED !!! "
                        f"Free space: {format_bytes(stats.free_bytes)} < Minimum: {format_bytes(self.minimum_free_space_bytes)}"
                    )
                    if self.on_emergency_stop:
                        self.on_emergency_stop(stats)
                    break
            except Exception as e:
                logger.error(f"Error in StorageMonitorThread: {e}")

            time.sleep(self.poll_interval)
