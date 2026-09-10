"""
Database module for Galaxy AV1 Migrator using SQLite.
Stores media file statuses, resume data, and migration metrics.
"""

from datetime import datetime
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, List, Optional
from av1_migrator.utils import normalize_filepath


class MigrationDB:
    def __init__(self, db_path: str | Path = "migration.db"):
        self.db_path = str(db_path)
        self._local = threading.local()
        self._lock = threading.Lock()
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            self._local.conn = conn
        return self._local.conn

    def init_db(self) -> None:
        """Create the necessary database tables and indices if they do not exist."""
        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS media_files (
                        source_path TEXT PRIMARY KEY,
                        source_size INTEGER NOT NULL,
                        source_mtime REAL NOT NULL,
                        output_path TEXT,
                        status TEXT NOT NULL DEFAULT 'pending',
                        started_at TEXT,
                        completed_at TEXT,
                        source_bytes INTEGER,
                        output_bytes INTEGER,
                        duration REAL,
                        source_codec TEXT,
                        hdr INTEGER DEFAULT 0,
                        error TEXT,
                        skip_reason TEXT
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS app_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON media_files(status);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_size ON media_files(source_size);")

    def get_file(self, source_path: str | Path) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        cur = conn.cursor()
        norm_src = normalize_filepath(source_path)
        cur.execute("SELECT * FROM media_files WHERE source_path = ?", (norm_src,))
        row = cur.fetchone()
        return dict(row) if row else None

    def upsert_file(
        self,
        source_path: str | Path,
        source_size: int,
        source_mtime: float,
        output_path: Optional[str | Path] = None,
        status: str = "pending",
        source_codec: Optional[str] = None,
        duration: Optional[float] = None,
        hdr: bool = False,
        skip_reason: Optional[str] = None,
        error: Optional[str] = None,
        source_bytes: Optional[int] = None,
        output_bytes: Optional[int] = None,
    ) -> None:
        norm_src = normalize_filepath(source_path)
        norm_out = normalize_filepath(output_path) if output_path else None
        with self._lock:
            conn = self._get_connection()
            with conn:
                # Check if exists
                cur = conn.cursor()
                cur.execute("SELECT source_size, source_mtime, status FROM media_files WHERE source_path = ?", (norm_src,))
                existing = cur.fetchone()
                
                if existing:
                    # If file modified since, reset to pending
                    if existing["source_size"] != source_size or abs(existing["source_mtime"] - source_mtime) > 1.0:
                        conn.execute("""
                            UPDATE media_files
                            SET source_size = ?, source_mtime = ?, output_path = ?, status = 'pending',
                                source_codec = ?, duration = ?, hdr = ?, error = NULL, skip_reason = NULL,
                                started_at = NULL, completed_at = NULL, source_bytes = NULL, output_bytes = NULL
                            WHERE source_path = ?
                        """, (
                            source_size, source_mtime, norm_out,
                            source_codec, duration, 1 if hdr else 0, norm_src
                        ))
                    else:
                        # Update metadata without overwriting completed status if already completed
                        conn.execute("""
                            UPDATE media_files
                            SET output_path = COALESCE(?, output_path),
                                source_codec = COALESCE(?, source_codec),
                                duration = COALESCE(?, duration),
                                hdr = ?,
                                skip_reason = COALESCE(?, skip_reason),
                                error = COALESCE(?, error),
                                source_bytes = COALESCE(?, source_bytes),
                                output_bytes = COALESCE(?, output_bytes)
                            WHERE source_path = ?
                        """, (
                            norm_out,
                            source_codec, duration, 1 if hdr else 0,
                            skip_reason, error, source_bytes, output_bytes, norm_src
                        ))
                else:
                    conn.execute("""
                        INSERT INTO media_files (
                            source_path, source_size, source_mtime, output_path, status,
                            source_codec, duration, hdr, skip_reason, error, source_bytes, output_bytes
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        norm_src, source_size, source_mtime,
                        norm_out, status,
                        source_codec, duration, 1 if hdr else 0, skip_reason, error, source_bytes, output_bytes
                    ))

    def update_status(
        self,
        source_path: str | Path,
        status: str,
        output_path: Optional[str | Path] = None,
        error: Optional[str] = None,
        skip_reason: Optional[str] = None,
        source_bytes: Optional[int] = None,
        output_bytes: Optional[int] = None,
        started_at: Optional[str] = None,
        completed_at: Optional[str] = None,
    ) -> None:
        norm_src = normalize_filepath(source_path)
        norm_out = normalize_filepath(output_path) if output_path is not None else None
        with self._lock:
            conn = self._get_connection()
            with conn:
                now_str = datetime.now().isoformat()
                updates = ["status = ?"]
                params: List[Any] = [status]
                
                if norm_out is not None:
                    updates.append("output_path = ?")
                    params.append(norm_out)
                if error is not None:
                    updates.append("error = ?")
                    params.append(error)
                if skip_reason is not None:
                    updates.append("skip_reason = ?")
                    params.append(skip_reason)
                if source_bytes is not None:
                    updates.append("source_bytes = ?")
                    params.append(source_bytes)
                if output_bytes is not None:
                    updates.append("output_bytes = ?")
                    params.append(output_bytes)
                if started_at is not None:
                    updates.append("started_at = ?")
                    params.append(started_at)
                elif status == "encoding":
                    updates.append("started_at = ?")
                    params.append(now_str)
                if completed_at is not None:
                    updates.append("completed_at = ?")
                    params.append(completed_at)
                elif status in ("completed", "failed", "skipped", "aborted"):
                    updates.append("completed_at = ?")
                    params.append(now_str)
                
                params.append(norm_src)
                query = f"UPDATE media_files SET {', '.join(updates)} WHERE source_path = ?"
                conn.execute(query, params)

    def get_summary_stats(self) -> Dict[str, Any]:
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT
                COUNT(*) as total_files,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed_count,
                SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending_count,
                SUM(CASE WHEN status = 'encoding' THEN 1 ELSE 0 END) as encoding_count,
                SUM(CASE WHEN status = 'skipped' THEN 1 ELSE 0 END) as skipped_count,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_count,
                SUM(CASE WHEN status = 'aborted' THEN 1 ELSE 0 END) as aborted_count,
                SUM(CASE WHEN status = 'completed' THEN COALESCE(source_bytes, source_size) ELSE 0 END) as completed_source_bytes,
                SUM(CASE WHEN status = 'completed' THEN COALESCE(output_bytes, 0) ELSE 0 END) as completed_output_bytes,
                SUM(CASE WHEN status IN ('pending', 'encoding', 'aborted') THEN source_size ELSE 0 END) as remaining_bytes,
                SUM(source_size) as total_source_bytes
            FROM media_files
        """)
        row = cur.fetchone()
        res = dict(row) if row else {}
        # Clean nulls
        for k, v in res.items():
            if v is None:
                res[k] = 0
        return res

    def get_failed_files(self) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT source_path, error, skip_reason FROM media_files WHERE status = 'failed'")
        return [dict(r) for r in cur.fetchall()]

    def get_interrupted_files(self) -> List[Dict[str, Any]]:
        """Returns all files that were left in 'encoding' or 'validating' state when the app stopped."""
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT source_path, source_size, source_mtime, output_path, status, started_at
            FROM media_files
            WHERE status IN ('encoding', 'validating')
        """)
        return [dict(r) for r in cur.fetchall()]

    def cleanup_interrupted_tasks(
        self,
        validator_fn=None,
        delete_original: bool = False,
        keep_smaller: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Scans DB for interrupted tasks (status in 'encoding' or 'validating').
        - Removes stale temporary .encoding.mkv files.
        - If final output exists and passes validation:
          - If keep_smaller is True and output is bloated (larger than or equal to source),
            removes bloated output, keeps original source, and marks as skipped.
          - Otherwise marks as completed and safely deletes original if configured.
        - If original source exists, resets status to 'pending' so it can be cleanly retried.
        - If original source is missing, marks as 'failed'.
        """
        interrupted = self.get_interrupted_files()
        actions = []

        for row in interrupted:
            src_str = row["source_path"]
            src_path = Path(src_str)
            out_str = row["output_path"]
            out_path = Path(out_str) if out_str else None
            status_before = row["status"]

            temp_path = None
            if out_path:
                temp_path = Path(str(out_path) + ".encoding.mkv")
            else:
                temp_path = Path(str(src_path) + ".encoding.mkv")

            temp_cleaned = False
            if temp_path and temp_path.is_file():
                try:
                    temp_path.unlink()
                    temp_cleaned = True
                except Exception:
                    pass

            # Check if final output already exists and is valid
            final_valid = False
            if out_path and out_path.is_file() and validator_fn:
                try:
                    is_valid, _, _ = validator_fn(out_path, src_path)
                    if is_valid:
                        final_valid = True
                except Exception:
                    final_valid = False

            if final_valid and out_path:
                out_sz = out_path.stat().st_size
                src_sz = row["source_size"]
                if keep_smaller and out_sz >= src_sz and src_path.is_file():
                    # Bloated output: prioritize space by keeping smaller original
                    try:
                        out_path.unlink()
                    except Exception:
                        pass
                    self.update_status(
                        src_path,
                        status="skipped",
                        skip_reason=f"Interrupted output was bloated ({src_sz} B source vs {out_sz} B output)",
                        source_bytes=src_sz,
                        output_bytes=src_sz,
                    )
                    actions.append({
                        "source_path": src_str,
                        "action": "skipped_bloated_output_kept_smaller_original",
                        "status_before": status_before,
                        "temp_cleaned": temp_cleaned,
                    })
                else:
                    self.update_status(
                        src_path,
                        status="completed",
                        source_bytes=src_sz,
                        output_bytes=out_sz,
                    )
                    if delete_original and src_path.is_file() and src_path != out_path:
                        try:
                            src_path.unlink()
                        except Exception:
                            pass
                    actions.append({
                        "source_path": src_str,
                        "action": "marked_completed_existing_valid_output",
                        "status_before": status_before,
                        "temp_cleaned": temp_cleaned,
                    })
            elif src_path.is_file():
                # Reset to pending for retry
                self.update_status(
                    src_path,
                    status="pending",
                    error=None,
                    started_at=None,
                    completed_at=None,
                )
                actions.append({
                    "source_path": src_str,
                    "action": "reset_to_pending_for_retry",
                    "status_before": status_before,
                    "temp_cleaned": temp_cleaned,
                })
            else:
                # Source file missing
                self.update_status(
                    src_path,
                    status="failed",
                    error="Original source file missing after interrupted session",
                )
                actions.append({
                    "source_path": src_str,
                    "action": "marked_failed_source_missing",
                    "status_before": status_before,
                    "temp_cleaned": temp_cleaned,
                })

        return actions

    def reset_failed(self) -> int:
        with self._lock:
            conn = self._get_connection()
            with conn:
                cur = conn.cursor()
                cur.execute("UPDATE media_files SET status = 'pending', error = NULL WHERE status IN ('failed', 'aborted')")
                return cur.rowcount

    def reset_file(self, source_path: str | Path) -> bool:
        norm_src = normalize_filepath(source_path)
        with self._lock:
            conn = self._get_connection()
            with conn:
                cur = conn.cursor()
                cur.execute("UPDATE media_files SET status = 'pending', error = NULL, skip_reason = NULL, started_at = NULL, completed_at = NULL WHERE source_path = ?", (norm_src,))
                return cur.rowcount > 0

    def set_metadata(self, key: str, value: str) -> None:
        """Sets a key-value pair in app_metadata."""
        with self._lock:
            conn = self._get_connection()
            with conn:
                conn.execute("INSERT OR REPLACE INTO app_metadata (key, value) VALUES (?, ?)", (str(key), str(value)))

    def get_metadata(self, key: str) -> Optional[str]:
        """Gets a value from app_metadata by key."""
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT value FROM app_metadata WHERE key = ?", (str(key),))
        row = cur.fetchone()
        return row["value"] if row else None

    def get_pending_files(self) -> List[Dict[str, Any]]:
        """Retrieves all media files currently in pending status."""
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM media_files WHERE status = 'pending'")
        return [dict(r) for r in cur.fetchall()]

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None
