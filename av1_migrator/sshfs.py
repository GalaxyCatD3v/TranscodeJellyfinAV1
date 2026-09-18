"""
SSHFS connection management and manual transfer staging for Galaxy AV1 Migrator.
Manages Windows SSHFS mounts (e.g. U: -> root@192.168.1.180), handles connection drops/hangs,
graceful restarts, retry loops, and 700GB manual transfer staging to Z: drive.
"""

from datetime import datetime
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional

from av1_migrator.config import SSHFSConfig, ManualStagingConfig, format_bytes, parse_size_to_bytes
from av1_migrator.logger import get_logger
from av1_migrator.utils import normalize_filepath


class ManualStagingDB:
    """
    SQLite database on Z: drive tracking files staged for manual upload.
    Stores source path, remote target destination, staged path on Z:, file size, and status.
    """

    def __init__(self, db_path: str | Path = "Z:\\JellyfinManualUpload\\manual_upload.db"):
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._local = threading.local()
        p = Path(self.db_path)
        try:
            if p.parent and not p.parent.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            try:
                sqlite3.connect(self.db_path, timeout=1.0).close()
            except Exception:
                self.db_path = "manual_upload.db"
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            try:
                conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
            except Exception:
                self.db_path = "manual_upload.db"
                conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            self._local.conn = conn
        return self._local.conn

    def init_db(self) -> None:
        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS manual_uploads (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_path TEXT NOT NULL,
                        destination_path TEXT NOT NULL,
                        staged_path TEXT NOT NULL,
                        size_bytes INTEGER NOT NULL,
                        status TEXT DEFAULT 'ready_for_manual_transfer',
                        created_at TEXT NOT NULL,
                        notes TEXT
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_staged_status ON manual_uploads(status);")

    def record_manual_staged(
        self,
        source_path: str | Path,
        destination_path: str | Path,
        staged_path: str | Path,
        size_bytes: int,
        notes: Optional[str] = None,
    ) -> None:
        norm_src = normalize_filepath(source_path)
        norm_dst = normalize_filepath(destination_path)
        norm_staged = normalize_filepath(staged_path)
        now_str = datetime.now().isoformat()
        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.execute("""
                    INSERT INTO manual_uploads (
                        source_path, destination_path, staged_path, size_bytes,
                        status, created_at, notes
                    ) VALUES (?, ?, ?, ?, 'ready_for_manual_transfer', ?, ?)
                """, (norm_src, norm_dst, norm_staged, size_bytes, now_str, notes))

    def get_staged_files(self) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM manual_uploads ORDER BY id DESC")
        return [dict(r) for r in cur.fetchall()]


class ManualStagingManager:
    """
    Manages staging files on Z: drive up to a configured limit (700 GB).
    Ensures transcoded media folders exist, checks capacity, and persists upload metadata.
    """

    def __init__(self, config: Optional[ManualStagingConfig] = None):
        self.config = config or ManualStagingConfig()
        self.staging_dir = Path(self.config.staging_dir)
        self.db = ManualStagingDB(self.config.db_path)
        self.ensure_staging_dir()

    def ensure_staging_dir(self) -> Path:
        logger = get_logger()
        try:
            self.staging_dir.mkdir(parents=True, exist_ok=True)
            return self.staging_dir
        except Exception as e:
            logger.warning(f"Could not create Z: drive staging directory {self.staging_dir}: {e}. Falling back to local folder.")
            fallback = Path("JellyfinManualUpload")
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback

    def get_current_staged_bytes(self) -> int:
        target_dir = self.ensure_staging_dir()
        total_bytes = 0
        try:
            for root, _, files in os.walk(target_dir):
                for f in files:
                    if f.endswith(".db") or f.endswith(".db-wal") or f.endswith(".db-shm"):
                        continue
                    try:
                        total_bytes += (Path(root) / f).stat().st_size
                    except Exception:
                        pass
        except Exception:
            pass
        return total_bytes

    def has_capacity_for(self, file_size: int) -> bool:
        max_limit = self.config.max_size_bytes
        current_used = self.get_current_staged_bytes()
        if (current_used + file_size) > max_limit:
            return False

        target_dir = self.ensure_staging_dir()
        try:
            _, _, free_bytes = shutil.disk_usage(str(target_dir))
            return free_bytes >= file_size
        except Exception:
            return True

    def stage_file(
        self,
        source_path: Path,
        output_file: Path,
        destination_path: Path,
        reason: str = "SSHFS upload failed after max retries",
    ) -> Path:
        logger = get_logger()
        target_dir = self.ensure_staging_dir()
        out_size = output_file.stat().st_size if output_file.exists() else 0

        if not self.has_capacity_for(out_size):
            logger.warning(
                f"Z: drive manual staging limit (700GB) exceeded: "
                f"current {format_bytes(self.get_current_staged_bytes())} + file {format_bytes(out_size)} > {self.config.max_size}"
            )

        # Preserve file name on Z: drive
        dest_filename = output_file.name
        # Remove intermediate .encoding tag if present
        if dest_filename.endswith(".encoding.mkv"):
            dest_filename = dest_filename[:-len(".encoding.mkv")]
            if not dest_filename.endswith(".mkv"):
                dest_filename += ".mkv"
        target_path = target_dir / dest_filename

        # Copy or move to Z: staging
        logger.info(f"Staging {output_file.name} to Z: drive for manual transfer -> {target_path}")
        try:
            shutil.copy2(output_file, target_path)
            output_file.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Failed copying {output_file} to Z: drive ({target_path}): {e}")
            target_path = output_file  # Keep local copy if Z: copy failed

        # Record in Z: drive database
        try:
            self.db.record_manual_staged(
                source_path=source_path,
                destination_path=destination_path,
                staged_path=target_path,
                size_bytes=out_size,
                notes=reason,
            )
        except Exception as e:
            logger.error(f"Failed to record manual staging in Z: DB: {e}")

        return target_path


class SSHFSManager:
    """
    Manages the SSHFS connection on Windows (mounting drive letter e.g. U: to remote host).
    Handles detecting drops/hangs, graceful disconnections, process cleanups,
    and automated reconnect/verification.
    """

    def __init__(self, config: Optional[SSHFSConfig] = None):
        self.config = config or SSHFSConfig()
        self.mount_drive = self.config.mount_drive.rstrip("\\").rstrip("/")
        if not self.mount_drive.endswith(":"):
            self.mount_drive = f"{self.mount_drive}:"
        self.mount_path = Path(f"{self.mount_drive}\\")
        self._lock = threading.Lock()

    def is_mounted(self) -> bool:
        """Check if the configured mount drive exists and responds to directory access."""
        try:
            if not self.mount_path.exists():
                return False
            # Try listing root to verify it is active and not hung
            _ = os.listdir(str(self.mount_path))
            return True
        except Exception:
            return False

    def verify_connection(self, timeout: float = 10.0) -> bool:
        """Verify the SSHFS connection is alive and responsive."""
        logger = get_logger()
        if not self.is_mounted():
            return False
        try:
            # Test directory readability
            entries = os.listdir(str(self.mount_path))
            logger.debug(f"SSHFS verified on {self.mount_drive} ({len(entries)} root entries found).")
            return True
        except Exception as e:
            logger.warning(f"SSHFS connection verification check failed for {self.mount_drive}: {e}")
            return False

    def mount(self) -> bool:
        """
        Mounts the SSHFS drive letter using net use or sshfs-win.
        Connects mount drive to remote (e.g. U: -> root@192.168.1.180).
        """
        logger = get_logger()
        with self._lock:
            if self.is_mounted():
                logger.info(f"SSHFS drive {self.mount_drive} is already mounted and accessible.")
                return True

            logger.info(
                f"Mounting SSHFS drive {self.mount_drive} to {self.config.user}@{self.config.host}..."
            )

            # 1. First attempt: net use with \\sshfs.r\ (root path)
            unc_path = f"\\\\sshfs.r\\{self.config.user}@{self.config.host}"
            cmd = [
                "net", "use", self.mount_drive, unc_path,
                self.config.password,
                f"/user:{self.config.user}",
                "/persistent:no",
            ]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                if res.returncode == 0 and self.is_mounted():
                    logger.info(f"SSHFS drive {self.mount_drive} successfully mounted via {unc_path}.")
                    return True
                else:
                    logger.warning(
                        f"Initial mount attempt ({unc_path}) returned code {res.returncode}: {res.stderr.strip() or res.stdout.strip()}"
                    )
            except Exception as e:
                logger.warning(f"Mount execution exception for {unc_path}: {e}")

            # 2. Second attempt: net use with standard \\sshfs\
            unc_path_alt = f"\\\\sshfs\\{self.config.user}@{self.config.host}"
            cmd_alt = [
                "net", "use", self.mount_drive, unc_path_alt,
                self.config.password,
                f"/user:{self.config.user}",
                "/persistent:no",
            ]
            try:
                res = subprocess.run(cmd_alt, capture_output=True, text=True, timeout=15)
                if res.returncode == 0 and self.is_mounted():
                    logger.info(f"SSHFS drive {self.mount_drive} successfully mounted via {unc_path_alt}.")
                    return True
                else:
                    logger.warning(
                        f"Secondary mount attempt ({unc_path_alt}) returned code {res.returncode}: {res.stderr.strip() or res.stdout.strip()}"
                    )
            except Exception as e:
                logger.warning(f"Secondary mount execution exception: {e}")

            # 3. Third attempt: Check if SSHFS-Win binary exists directly
            sshfs_win_exe = Path(r"C:\Program Files\SSHFS-Win\bin\sshfs-win.exe")
            if sshfs_win_exe.exists():
                try:
                    svc_target = f"\\sshfs.r\\{self.config.user}@{self.config.host}"
                    cmd_bin = [str(sshfs_win_exe), "svc", svc_target, self.mount_drive]
                    proc = subprocess.Popen(cmd_bin, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    time.sleep(2.0)
                    if self.is_mounted():
                        logger.info(f"SSHFS drive {self.mount_drive} mounted via direct sshfs-win.exe service.")
                        return True
                except Exception as e:
                    logger.warning(f"Direct sshfs-win.exe mount attempt failed: {e}")

            # Verify final state
            if self.is_mounted():
                return True
            logger.error(f"Failed to mount SSHFS drive {self.mount_drive}.")
            return False

    def unmount(self, force: bool = False) -> bool:
        """
        Gracefully unmounts the SSHFS drive.
        If graceful disconnect fails or hangs, emits warnings and optionally forces cleanup.
        """
        logger = get_logger()
        with self._lock:
            logger.info(f"Unmounting SSHFS drive {self.mount_drive} gracefully...")
            cmd = ["net", "use", self.mount_drive, "/delete", "/y"]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if res.returncode == 0:
                    logger.info(f"SSHFS drive {self.mount_drive} successfully unmounted.")
                    return True
                else:
                    logger.warning(
                        f"Graceful unmount of {self.mount_drive} reported warning (code {res.returncode}): {res.stderr.strip() or res.stdout.strip()}"
                    )
            except subprocess.TimeoutExpired:
                logger.warning(f"SSHFS unmount timed out after 10s for {self.mount_drive}. Connection hung!")
            except Exception as e:
                logger.warning(f"Exception during SSHFS unmount of {self.mount_drive}: {e}")

            # If force requested or connection didn't close cleanly
            if force or not self.is_mounted():
                try:
                    # Terminate any hung sshfs processes if needed
                    logger.warning("Spitting warning: Cleaning up hung SSHFS processes...")
                    subprocess.run(["taskkill", "/f", "/im", "sshfs.exe"], capture_output=True, timeout=5)
                    subprocess.run(["taskkill", "/f", "/im", "sshfs-win.exe"], capture_output=True, timeout=5)
                except Exception as e:
                    logger.debug(f"Process cleanup notice: {e}")

            return not self.is_mounted()

    def reconnect(self) -> bool:
        """
        Gracefully restarts the SSHFS connection:
        1. Closes existing connection with warnings if not graceful.
        2. Waits for reconnect delay.
        3. Reconnects and verifies the mount.
        """
        logger = get_logger()
        logger.warning(f"SSHFS connection drop/hang detected. Initiating graceful restart of {self.mount_drive}...")
        self.unmount(force=True)
        time.sleep(self.config.reconnect_delay)
        success = self.mount()
        if success and self.verify_connection():
            logger.info(f"SSHFS connection to {self.mount_drive} successfully restored and verified.")
            return True
        else:
            logger.warning(f"SSHFS reconnect verification failed for {self.mount_drive}!")
            return False
