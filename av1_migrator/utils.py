"""
Cross-platform path, filename sanitization, and executable resolution utilities.
Ensures file names and file paths are valid and compatible across both Windows and Linux.
"""

import os
from pathlib import Path
import re
import shutil
import unicodedata
from datetime import timedelta
from typing import Optional, Tuple

# Windows prohibited characters: < > : " / \ | ? * and ASCII control chars (0-31, 127-159)
# Linux prohibited characters: / and \0
INVALID_CHARS_REGEX = re.compile(r'[\x00-\x1f\x7f-\x9f<>:"/\\|?*]')
CONSECUTIVE_SPACES_REGEX = re.compile(r'\s+')
CONSECUTIVE_DASHES_REGEX = re.compile(r'-{2,}')

# Windows reserved device names
WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
}


def format_seconds(secs: Optional[float]) -> str:
    """Formats a duration in seconds into HH:MM:SS or Dd HHh MMm format."""
    if secs is None or secs < 0 or secs > 365 * 24 * 3600:
        return "--:--:--"
    total_sec = int(secs)
    total_hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    seconds = total_sec % 60
    if total_hours >= 24:
        days = total_hours // 24
        rem_hours = total_hours % 24
        return f"{days}d {rem_hours:02d}h {minutes:02d}m"
    return f"{total_hours:02d}:{minutes:02d}:{seconds:02d}"


def normalize_filepath(path: str | Path) -> str:
    """
    Normalizes a filesystem path for consistent cross-platform representation and DB storage.
    """
    if not path:
        return ""
    p_str = str(path)
    # Normalize path separators according to current platform
    return os.path.normpath(p_str)


def sanitize_filename(name: str, max_length: int = 255) -> str:
    """
    Sanitizes a filename to ensure it is valid on both Windows and Linux/SSHFS/Samba filesystems.
    - Normalizes Unicode characters (NFC form).
    - Converts ':' to ' - ' to preserve media subtitle formatting (e.g. "Movie: Subtitle" -> "Movie - Subtitle").
    - Converts '/' and '\\' to '-'.
    - Converts '"' to single quote "'".
    - Strips '?', '*', '<', '>', '|', and ASCII control characters.
    - Normalizes multiple spaces and dashes.
    - Strips leading and trailing spaces and dots (prohibited by Windows NTFS).
    - Prevents Windows reserved device names (CON, NUL, AUX, etc.).
    - Caps filename length to max_length (default 255).
    """
    if not name:
        return "unnamed"

    # Unicode normalization
    clean = unicodedata.normalize("NFC", name)

    # Intelligent character replacements for common movie/TV patterns
    clean = clean.replace(":", " - ")
    clean = clean.replace("/", "-")
    clean = clean.replace("\\", "-")
    clean = clean.replace('"', "'")

    # Remove all remaining invalid characters
    clean = INVALID_CHARS_REGEX.sub("", clean)

    # Normalize whitespace and dashes
    clean = CONSECUTIVE_SPACES_REGEX.sub(" ", clean)
    clean = CONSECUTIVE_DASHES_REGEX.sub("-", clean)
    clean = clean.strip(" .-\t\r\n")

    if not clean:
        clean = "unnamed"

    # Check for Windows reserved names (e.g. CON, NUL)
    base_name = clean.split(".")[0].upper()
    if base_name in WINDOWS_RESERVED_NAMES:
        clean = f"_{clean}"

    # Truncate to max_length while preserving extension if present
    if len(clean.encode("utf-8")) > max_length:
        parts = clean.rsplit(".", 1)
        if len(parts) == 2 and len(parts[1]) < 10:
            ext = "." + parts[1]
            stem = parts[0]
            max_stem_bytes = max_length - len(ext.encode("utf-8"))
            # Truncate stem by UTF-8 bytes safely
            encoded = stem.encode("utf-8")[:max_stem_bytes]
            clean = encoded.decode("utf-8", "ignore").rstrip(" .-") + ext
        else:
            encoded = clean.encode("utf-8")[:max_length]
            clean = encoded.decode("utf-8", "ignore").rstrip(" .-")

    return clean


def sanitize_stem(stem: str) -> str:
    """
    Sanitizes a file stem (name without extension) and removes any existing [AV1 ...] migration tags.
    """
    # Remove existing [AV1 ...] tags
    clean = re.sub(r"\s*\[AV1[^\]]*\]", "", stem).strip()
    return sanitize_filename(clean)


def find_binary_executable(name: str) -> Optional[str]:
    """
    Resolves binary executable path in a cross-platform manner.
    - Works whether on Linux (where ffmpeg has no extension) or Windows (where ffmpeg has .exe).
    - Tolerates config specifying 'ffmpeg.exe' on Linux or 'ffmpeg' on Windows.
    """
    if not name:
        return None

    # 1. Direct which lookup
    found = shutil.which(name)
    if found:
        return found

    # 2. If name ends with .exe and running on Linux/non-Windows, check without .exe
    if name.lower().endswith(".exe"):
        base = name[:-4]
        found = shutil.which(base)
        if found:
            return found

    # 3. If name does not end with .exe and running on Windows, check with .exe
    if not name.lower().endswith(".exe"):
        exe_name = name + ".exe"
        found = shutil.which(exe_name)
        if found:
            return found

    return None
