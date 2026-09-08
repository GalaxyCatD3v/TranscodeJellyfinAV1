"""
Fault-tolerant filesystem scanner for SSHFS / network storage.
"""

import os
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Set, Tuple
from av1_migrator.config import AppConfig
from av1_migrator.logger import get_logger
from av1_migrator.models import ScanStats


def is_encoding_temp_file(path: Path) -> bool:
    """Checks if the file is a temporary encoding artifact (e.g., ends in .encoding.mkv)."""
    name = path.name.lower()
    return ".encoding." in name or name.endswith(".encoding.mkv")


def scan_media_root(
    root_path: str | Path,
    valid_extensions: Set[str],
    stats: ScanStats,
    on_file_found: Optional[Callable[[Path], None]] = None,
    on_progress: Optional[Callable[[int, int, Path], None]] = None,
) -> List[Path]:
    """
    Recursively scans a media root directory with error tolerance for SSHFS/network mounts.
    Never lets a single inaccessible folder or file stop the scan.
    """
    logger = get_logger()
    root = Path(root_path)
    discovered: List[Path] = []

    if not root.exists():
        logger.warning(f"Media root does not exist or is currently inaccessible: {root}")
        stats.access_errors += 1
        return discovered

    stack = [root]

    while stack:
        current_dir = stack.pop()
        stats.directories_scanned += 1

        if on_progress:
            on_progress(stats.directories_scanned, stats.files_discovered, current_dir)

        try:
            # os.scandir is faster and more reliable than Path.iterdir
            with os.scandir(current_dir) as it:
                entries = list(it)
        except (PermissionError, OSError, FileNotFoundError) as e:
            logger.warning(f"Could not access directory {current_dir}: {e}")
            stats.access_errors += 1
            continue

        for entry in entries:
            try:
                # Handle broken symlinks and disappearing files
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    stats.files_discovered += 1
                    file_path = Path(entry.path)
                    
                    if is_encoding_temp_file(file_path):
                        stats.files_skipped += 1
                        continue

                    ext = file_path.suffix.lower()
                    if ext in valid_extensions:
                        discovered.append(file_path)
                        if on_file_found:
                            on_file_found(file_path)
                    else:
                        stats.files_skipped += 1
            except (PermissionError, OSError, FileNotFoundError) as e:
                logger.warning(f"Error accessing entry {entry.name} in {current_dir}: {e}")
                stats.access_errors += 1
                continue

    return discovered


def scan_all_roots(
    config: AppConfig,
    stats: Optional[ScanStats] = None,
    on_progress: Optional[Callable[[int, int, Path], None]] = None,
) -> Tuple[List[Path], ScanStats]:
    """
    Scans all configured media roots and aggregates stats.
    """
    if stats is None:
        stats = ScanStats()

    valid_exts = {ext.lower() for ext in config.extensions}
    all_files: List[Path] = []

    for root_str in config.media_roots:
        found = scan_media_root(root_str, valid_exts, stats, on_progress=on_progress)
        all_files.extend(found)

    return all_files, stats
