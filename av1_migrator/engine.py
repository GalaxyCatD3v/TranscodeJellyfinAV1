"""
Workflow orchestration engine for Galaxy AV1 Migrator.
Enforces strict safety priorities, SQLite state persistence, live progress, and smallest-first processing.
"""

import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from av1_migrator.config import AppConfig, format_bytes
from av1_migrator.dashboard import MigrationDashboard
from av1_migrator.db import MigrationDB
from av1_migrator.encoder import FFmpegEncoder, is_av1_nvenc_available, is_scale_cuda_available
from av1_migrator.gpu import GPUMonitorThread, is_nvidia_smi_available
from av1_migrator.logger import get_logger
from av1_migrator.models import EncodeProgress, GPUStats, MediaFile, ScanStats, StorageStats
from av1_migrator.optimizer import evaluate_pre_encode_optimization
from av1_migrator.probe import probe_and_populate_media_file
from av1_migrator.progress import (
    create_encode_progressbar,
    create_overall_progressbar,
    create_probe_progressbar,
    create_scan_progressbar,
    MigrationProgressBar,
)
from av1_migrator.scanner import is_encoding_temp_file, scan_all_roots
from av1_migrator.storage import check_storage_safety, get_path_disk_usage, StorageMonitorThread
from av1_migrator.validator import validate_converted_file
from av1_migrator.utils import find_binary_executable, normalize_filepath
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)


class MigrationEngine:
    def __init__(
        self,
        config: AppConfig,
        db: Optional[MigrationDB] = None,
        dashboard: Optional[MigrationDashboard] = None,
        dry_run: bool = False,
        limit: Optional[int] = None,
        single_file: Optional[str] = None,
        retry_failed: bool = False,
        no_delete: bool = False,
        no_ui: bool = False,
    ):
        self.config = config
        self.db = db or MigrationDB(config.database.path)
        self.dashboard = dashboard or MigrationDashboard()
        self.dry_run = dry_run
        self.limit = limit
        self.single_file = single_file
        self.retry_failed = retry_failed
        self.no_delete = no_delete
        self.no_ui = no_ui

        self.logger = get_logger()
        self.gpu_monitor = GPUMonitorThread(poll_interval=1.0)
        
        self.is_running = False
        self.is_paused = False
        self.stop_requested = False
        self.emergency_stop_triggered = False
        self.emergency_stop_msg = ""
        
        self.active_encoder: Optional[FFmpegEncoder] = None
        self.active_media_file: Optional[MediaFile] = None
        self.active_storage_monitor: Optional[StorageMonitorThread] = None

        self.total_source_bytes_processed = 0
        self.total_output_bytes_created = 0
        self.completed_count = 0
        self.skipped_count = 0
        self.failed_count = 0
        self.queue: List[MediaFile] = []
        self.total_files_count = 0

        self.start_time = 0.0
        self.lock = threading.Lock()

    def run_preflight_checks(self) -> Tuple[bool, List[str]]:
        """
        Executes pre-flight checks:
        Python version, FFmpeg, FFprobe, NVIDIA GPU, AV1 NVENC, CUDA scaling, media roots, storage floor.
        """
        checks: List[str] = []
        all_ok = True

        # Python version
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        if sys.version_info >= (3, 10):
            checks.append(f"✓ Python ({py_ver})")
        else:
            checks.append(f"✗ Python ({py_ver} < 3.10 required)")
            all_ok = False

        # FFmpeg executable
        ff_exe = find_binary_executable(self.config.ffmpeg.executable)
        if ff_exe:
            checks.append(f"✓ FFmpeg ({ff_exe})")
        else:
            checks.append(f"✗ FFmpeg not found ({self.config.ffmpeg.executable})")
            all_ok = False

        # FFprobe executable
        probe_exe = find_binary_executable(self.config.ffmpeg.ffprobe)
        if probe_exe:
            checks.append(f"✓ FFprobe ({probe_exe})")
        else:
            checks.append(f"✗ FFprobe not found ({self.config.ffmpeg.ffprobe})")
            all_ok = False

        # NVIDIA GPU
        if is_nvidia_smi_available():
            checks.append("✓ NVIDIA GPU (nvidia-smi detected)")
        else:
            checks.append("⚠ NVIDIA GPU (nvidia-smi not found in PATH)")

        # AV1 NVENC
        if is_av1_nvenc_available(self.config.ffmpeg.executable):
            checks.append("✓ AV1 NVENC encoder available")
        else:
            checks.append("✗ AV1 NVENC encoder not found in FFmpeg")
            all_ok = False

        # CUDA scaling
        if is_scale_cuda_available(self.config.ffmpeg.executable):
            checks.append("✓ CUDA scaling (scale_cuda)")
        else:
            checks.append("⚠ CUDA scaling not found (will use software Lanczos fallback)")

        # Media roots check
        for r in self.config.media_roots:
            p = Path(r)
            if p.exists():
                checks.append(f"✓ Media root: {r}")
            else:
                checks.append(f"⚠ Media root currently inaccessible: {r}")

        # Storage safety check
        for r in self.config.media_roots:
            p = Path(r)
            if p.exists():
                st = get_path_disk_usage(p)
                min_bytes = self.config.storage.minimum_free_space_bytes
                if st.free_bytes >= min_bytes:
                    checks.append(f"✓ Storage safety ({r}): Free {format_bytes(st.free_bytes)} >= Min {format_bytes(min_bytes)}")
                else:
                    checks.append(f"✗ Storage safety ({r}): Free {format_bytes(st.free_bytes)} < Min {format_bytes(min_bytes)}")
                    all_ok = False
                break

        return all_ok, checks

    def setup_signal_handlers(self) -> None:
        def handle_sigint(signum, frame):
            self.logger.warning("Received interrupt signal (Ctrl+C). Initiating safe stop...")
            self.stop_safely(reason="Ctrl+C interrupted")

        signal.signal(signal.SIGINT, handle_sigint)
        signal.signal(signal.SIGTERM, handle_sigint)

    def stop_safely(self, reason: str = "User requested stop") -> None:
        """
        Graceful stop:
        1. Signals active FFmpeg to terminate safely.
        2. Waits for termination.
        3. Deletes temporary .encoding.mkv.
        4. Preserves original file.
        5. Updates database.
        """
        self.stop_requested = True
        self.logger.info(f"Stopping migration safely: {reason}")
        
        with self.lock:
            if self.active_encoder:
                self.active_encoder.abort(reason=reason)

            if self.active_media_file:
                # Mark as aborted in DB
                self.db.update_status(
                    self.active_media_file.source,
                    status="aborted",
                    output_path=self.active_media_file.output_path,
                    error=reason,
                )
                temp_file = self.active_media_file.temp_output_path
                if temp_file and temp_file.exists():
                    try:
                        temp_file.unlink()
                        self.logger.info(f"Removed temporary file: {temp_file}")
                    except Exception as e:
                        self.logger.error(f"Failed to remove temp file {temp_file}: {e}")

    def trigger_emergency_stop(self, stats: StorageStats) -> None:
        """
        Continuous storage monitor callback when free space falls below minimum floor (1 TB).
        """
        self.emergency_stop_triggered = True
        self.emergency_stop_msg = (
            f"Free space: {format_bytes(stats.free_bytes)} < Minimum required: {format_bytes(stats.minimum_free_bytes)}"
        )
        self.logger.critical(f"EMERGENCY STOP TRIGGERED: {self.emergency_stop_msg}")
        self.stop_safely(reason="EMERGENCY STOP: Storage space below 1 TB")

    def run_startup_recovery(self) -> List[Dict[str, Any]]:
        """
        Inspects database for any tasks that were previously interrupted (status in encoding/validating).
        Cleans up stale temporary files and resets items to pending so they retry cleanly.
        """
        self.logger.info("Checking database for uncompleted/interrupted migration tasks from previous runs...")
        
        def validator_adapter(out_path: Path, src_path: Path):
            dummy_mf = MediaFile(source=src_path, output_path=out_path)
            return validate_converted_file(out_path, dummy_mf, self.config)

        actions = self.db.cleanup_interrupted_tasks(
            validator_fn=validator_adapter,
            delete_original=self.config.processing.delete_original and not self.no_delete,
            keep_smaller=self.config.processing.keep_smaller,
        )
        
        if actions:
            self.logger.warning(f"Startup recovery: Found and resolved {len(actions)} interrupted tasks.")
            for act in actions:
                self.logger.info(
                    f"  - {act['source_path']}: {act['action']} "
                    f"(previous status: {act['status_before']}, temp cleaned: {act['temp_cleaned']})"
                )
                if self.no_ui or not sys.stdout.isatty():
                    self.dashboard.console.print(f"[bold yellow]Startup Recovery:[/bold yellow] {Path(act['source_path']).name} -> {act['action']}")
        else:
            self.logger.info("Startup recovery: Database is clean, no interrupted tasks found.")
            
        return actions

    def scan_and_prepare_queue(self) -> Tuple[List[MediaFile], ScanStats]:
        """
        Scans media roots, probes files, updates SQLite DB, and returns sorted queue (smallest first).
        """
        self.logger.info("Starting media discovery and queue preparation...")
        scan_stats = ScanStats()

        if self.retry_failed:
            reset_cnt = self.db.reset_failed()
            self.logger.info(f"Reset {reset_cnt} failed/aborted files back to pending")

        if self.single_file:
            target_path = Path(self.single_file).resolve()
            if not target_path.is_file():
                raise FileNotFoundError(f"Specified single file not found: {target_path}")
            candidate_paths = [target_path]
            scan_stats.files_discovered = 1
        else:
            scan_prog = create_scan_progressbar(message="Scanning Media Roots")
            scan_prog.start()

            def on_scan_progress(dirs_cnt, files_cnt, cur_p):
                try:
                    scan_prog.update(
                        dirs_cnt,
                        postfix_str=f"{dirs_cnt} dirs | {files_cnt} files | {cur_p.name[:25]}",
                    )
                except Exception:
                    pass

            try:
                candidate_paths, scan_stats = scan_all_roots(self.config, scan_stats, on_progress=on_scan_progress)
            finally:
                try:
                    scan_prog.finish()
                except Exception:
                    pass

        is_largest_first = self.config.processing.sort in ("largest_first", "biggest_first", "desc", "largest")
        sort_order_label = "largest files first" if is_largest_first else "smallest files first"
        self.logger.info(f"Discovered {len(candidate_paths)} candidate files across media roots. Sorting by size ({sort_order_label})...")

        # Gather stats and sort candidates by size
        candidates_with_stats: List[Tuple[Path, int, float]] = []
        for p in candidate_paths:
            try:
                st = p.stat()
                candidates_with_stats.append((p, st.st_size, st.st_mtime))
            except Exception as e:
                self.logger.warning(f"Could not stat {p}: {e}")
                scan_stats.access_errors += 1

        if is_largest_first:
            candidates_with_stats.sort(key=lambda x: x[1], reverse=True)
        else:
            candidates_with_stats.sort(key=lambda x: x[1])

        media_files_to_process: List[MediaFile] = []
        total_candidates = len(candidates_with_stats)

        probe_prog = create_probe_progressbar(total_candidates, message="Probing Media Files")
        probe_prog.start()

        def do_probing_loop():
            for idx, (p, file_size, file_mtime) in enumerate(candidates_with_stats, 1):
                if not p.is_file():
                    continue

                try:
                    probe_prog.update(idx, postfix_str=p.name[:35])
                except Exception:
                    pass

                # If limit is reached, stop probing further
                if self.limit and len(media_files_to_process) >= self.limit:
                    break

                # Check DB record
                db_row = self.db.get_file(p)

                # If already marked completed in DB and file unchanged
                if (
                    db_row
                    and db_row["status"] == "completed"
                    and db_row["source_size"] == file_size
                    and abs(db_row["source_mtime"] - file_mtime) <= 1.0
                    and not self.single_file
                ):
                    scan_stats.already_converted += 1
                    self.completed_count += 1
                    self.total_source_bytes_processed += db_row["source_bytes"] or file_size
                    self.total_output_bytes_created += db_row["output_bytes"] or file_size
                    continue

                # If already marked skipped in DB and file unchanged (and not retrying failed)
                if (
                    db_row
                    and db_row["status"] == "skipped"
                    and db_row["source_size"] == file_size
                    and abs(db_row["source_mtime"] - file_mtime) <= 1.0
                    and not self.single_file
                ):
                    scan_stats.files_skipped += 1
                    self.skipped_count += 1
                    continue

                # Probe media file
                mf = probe_and_populate_media_file(p, self.config)

                if mf.status == "failed":
                    scan_stats.probe_errors += 1
                    self.failed_count += 1
                    self.db.upsert_file(
                        source_path=p,
                        source_size=file_size,
                        source_mtime=file_mtime,
                        status="failed",
                        error=mf.error_reason,
                    )
                    continue

                if mf.status == "skipped":
                    scan_stats.files_skipped += 1
                    self.skipped_count += 1
                    self.db.upsert_file(
                        source_path=p,
                        source_size=file_size,
                        source_mtime=file_mtime,
                        status="skipped",
                        skip_reason=mf.skip_reason,
                        source_codec=mf.video_codec,
                        duration=mf.duration,
                        hdr=mf.hdr,
                    )
                    continue

                # If source is already AV1 1080p
                if mf.skip_reason == "Source is already AV1 1080p":
                    scan_stats.already_converted += 1
                    self.completed_count += 1
                    self.db.upsert_file(
                        source_path=p,
                        source_size=file_size,
                        source_mtime=file_mtime,
                        output_path=mf.source,
                        status="completed",
                        skip_reason="Source is already AV1 1080p",
                        source_codec=mf.video_codec,
                        duration=mf.duration,
                        hdr=mf.hdr,
                        source_bytes=file_size,
                        output_bytes=file_size,
                    )
                    continue

                # Check if final output already exists on disk
                if mf.output_path and mf.output_path.is_file():
                    self.logger.info(f"Final output already exists on disk for {p.name}. Validating existing output...")
                    is_valid, v_msg, _ = validate_converted_file(mf.output_path, mf, self.config)
                    if is_valid:
                        out_sz = mf.output_path.stat().st_size
                        if self.config.processing.keep_smaller and out_sz >= file_size:
                            self.logger.warning(
                                f"Existing output for {p.name} is larger ({format_bytes(out_sz)}) than or equal to original ({format_bytes(file_size)}). "
                                f"Prioritizing space: keeping smaller original and removing bloated output."
                            )
                            try:
                                mf.output_path.unlink()
                            except Exception as e:
                                self.logger.error(f"Could not remove bloated output file {mf.output_path}: {e}")

                            scan_stats.files_skipped += 1
                            self.skipped_count += 1
                            self.db.upsert_file(
                                source_path=p,
                                source_size=file_size,
                                source_mtime=file_mtime,
                                output_path=mf.output_path,
                                status="skipped",
                                skip_reason=f"Existing output was bloated ({format_bytes(file_size)} vs {format_bytes(out_sz)})",
                                source_codec=mf.video_codec,
                                duration=mf.duration,
                                hdr=mf.hdr,
                                source_bytes=file_size,
                                output_bytes=file_size,
                            )
                            continue

                        self.logger.info(f"Existing output is valid AV1 1080p for {p.name}. Marking as completed.")
                        scan_stats.already_converted += 1
                        self.completed_count += 1
                        self.total_source_bytes_processed += file_size
                        self.total_output_bytes_created += out_sz
                        
                        # Delete original if configured
                        if self.config.processing.delete_original and not self.no_delete and p.exists() and p != mf.output_path:
                            try:
                                p.unlink()
                                self.logger.info(f"Deleted original file after validating existing output: {p}")
                            except Exception as e:
                                self.logger.warning(f"Could not delete original file {p}: {e}")

                        self.db.upsert_file(
                            source_path=p,
                            source_size=file_size,
                            source_mtime=file_mtime,
                            output_path=mf.output_path,
                            status="completed",
                            source_codec=mf.video_codec,
                            duration=mf.duration,
                            hdr=mf.hdr,
                            source_bytes=file_size,
                            output_bytes=out_sz,
                        )
                        continue
                    else:
                        self.logger.warning(f"Existing output for {p.name} was corrupt or invalid: {v_msg}. Removing and scheduling re-encode.")
                        try:
                            mf.output_path.unlink()
                        except Exception as e:
                            self.logger.error(f"Could not delete invalid output file {mf.output_path}: {e}")

                # Register pending file in DB and add to queue
                self.db.upsert_file(
                    source_path=p,
                    source_size=file_size,
                    source_mtime=file_mtime,
                    output_path=mf.output_path,
                    status="pending",
                    source_codec=mf.video_codec,
                    duration=mf.duration,
                    hdr=mf.hdr,
                )
                media_files_to_process.append(mf)
                scan_stats.eligible_files += 1

        try:
            do_probing_loop()
        finally:
            try:
                probe_prog.finish()
            except Exception:
                pass

        # Sort queue
        if is_largest_first:
            media_files_to_process.sort(key=lambda m: m.size, reverse=True)
        else:
            media_files_to_process.sort(key=lambda m: m.size)

        # Apply limit if requested
        if self.limit and self.limit > 0:
            media_files_to_process = media_files_to_process[: self.limit]

        self.queue = media_files_to_process
        self.total_files_count = len(candidate_paths)
        return self.queue, scan_stats

    def execute(self) -> Dict[str, Any]:
        """
        Main execution loop.
        """
        self.is_running = True
        self.start_time = time.time()
        self.setup_signal_handlers()

        # 1. Pre-flight checks
        ok, check_lines = self.run_preflight_checks()
        self.logger.info("=== Pre-flight System Checks ===")
        for line in check_lines:
            self.logger.info(line)
            if self.no_ui:
                self.dashboard.console.print(line)

        if not ok and not self.dry_run:
            self.logger.error("Pre-flight checks failed. Aborting execution.")
            return {"status": "error", "message": "Pre-flight checks failed"}

        # 2. Startup database check and interrupted task recovery
        self.run_startup_recovery()

        # 3. Scan and build queue
        queue, scan_stats = self.scan_and_prepare_queue()

        total_source_bytes_in_queue = sum(m.size for m in queue)
        estimated_output_bytes = int(total_source_bytes_in_queue * 0.35)  # ~65% estimated AV1 reduction
        estimated_savings_bytes = total_source_bytes_in_queue - estimated_output_bytes

        # 3. Dry-run mode
        if self.dry_run:
            self.logger.info("=== DRY RUN SUMMARY ===")
            summary = (
                f"\nDRY RUN SUMMARY:\n"
                f"  Eligible files to convert: {len(queue)}\n"
                f"  Already AV1/Completed:     {scan_stats.already_converted}\n"
                f"  Skipped (No Eng/etc):      {scan_stats.files_skipped}\n"
                f"  Probe Errors:              {scan_stats.probe_errors}\n"
                f"  Total source size:         {format_bytes(total_source_bytes_in_queue)}\n"
                f"  Estimated output size:     ~{format_bytes(estimated_output_bytes)}\n"
                f"  Estimated savings:         ~{format_bytes(estimated_savings_bytes)}\n"
            )
            self.logger.info(summary)
            if self.no_ui or not sys.stdout.isatty():
                self.dashboard.console.print(summary)
            return {
                "status": "dry_run_complete",
                "queue_count": len(queue),
                "total_source_bytes": total_source_bytes_in_queue,
                "estimated_savings_bytes": estimated_savings_bytes,
            }

        if not queue:
            self.logger.info("No eligible files to transcode in queue. All files up to date!")
            if self.no_ui or not sys.stdout.isatty():
                self.dashboard.console.print("All files up to date! Nothing to migrate.")
            return {"status": "completed", "converted_count": 0}

        # 4. Start GPU monitor
        self.gpu_monitor.start()

        def get_gpu_status_str() -> str:
            g = self.gpu_monitor.stats
            parts = []
            if g.gpu_util is not None:
                parts.append(f"GPU: {g.gpu_util:.0f}%")
            if g.enc_util is not None:
                parts.append(f"NVENC: {g.enc_util:.0f}%")
            if g.temperature_c is not None:
                parts.append(f"{int(g.temperature_c)}°C")
            return " | ".join(parts) if parts else ""

        overall_prog = create_overall_progressbar(len(queue), message="Overall Migration")
        overall_prog.start()

        def update_ui(current_progress: Optional[EncodeProgress] = None, status_msg: str = "Encoding"):
            pass

        # 5. Process Queue
        try:
            while self.queue and not self.stop_requested:
                media_file = self.queue.pop(0)
                self.active_media_file = media_file

                # Handle pause
                while self.is_paused and not self.stop_requested:
                    time.sleep(0.5)

                if self.stop_requested:
                    break

                # Pre-encode storage check
                is_safe, s_msg, s_stats = check_storage_safety(
                    path=media_file.source,
                    original_file_size=media_file.size,
                    minimum_free_space_bytes=self.config.storage.minimum_free_space_bytes,
                    safety_margin_bytes=self.config.storage.safety_margin_bytes,
                )

                if not is_safe:
                    self.logger.critical(f"Storage check failed before starting {media_file.source.name}: {s_msg}")
                    self.trigger_emergency_stop(s_stats)
                    break

                # Pre-encode optimization check: test snippet to predict bloat and savings before doing full encode
                if self.config.processing.pre_encode_check:
                    opt_res = evaluate_pre_encode_optimization(media_file, self.config)
                    if not opt_res.should_encode:
                        self.logger.warning(
                            f"Pre-encode check: Skipping {media_file.source.name}. {opt_res.reason}"
                        )
                        self.skipped_count += 1
                        self.db.update_status(
                            media_file.source,
                            status="skipped",
                            output_path=media_file.output_path,
                            skip_reason=opt_res.reason,
                            source_bytes=media_file.size,
                            output_bytes=media_file.size,
                        )
                        continue
                    else:
                        self.logger.info(f"Pre-encode check passed for {media_file.source.name}: {opt_res.reason}")

                # Start continuous background storage monitor
                self.active_storage_monitor = StorageMonitorThread(
                    watch_path=media_file.source,
                    minimum_free_space_bytes=self.config.storage.minimum_free_space_bytes,
                    poll_interval=self.config.storage.poll_interval,
                    on_emergency_stop=self.trigger_emergency_stop,
                )
                self.active_storage_monitor.start()

                # Update DB to encoding
                self.db.update_status(media_file.source, status="encoding", output_path=media_file.output_path)

                # Progress bar for encoding
                encode_prog = create_encode_progressbar(f"Encoding [{media_file.source.name[:25]}]")
                encode_prog.start()

                # Progress callback
                def on_progress(p: EncodeProgress):
                    gpu_str = get_gpu_status_str()
                    eta_val = getattr(p, "eta_str", "--:--:--")
                    in_sz = format_bytes(media_file.size)
                    out_sz = format_bytes(p.total_size_bytes)
                    parts = [
                        f"in: {in_sz} -> out: {out_sz}",
                        f"{p.speed:.2f}x ({p.fps:.1f} fps)",
                        f"ETA {eta_val}",
                    ]
                    if gpu_str:
                        parts.append(gpu_str)
                    try:
                        encode_prog.update(p.percent, postfix_str=" | ".join(parts))
                    except Exception:
                        pass

                # Create and execute encoder
                self.active_encoder = FFmpegEncoder(
                    media_file=media_file,
                    config=self.config,
                    on_progress=on_progress,
                )

                encode_success, encode_msg = self.active_encoder.run()
                try:
                    encode_prog.finish()
                except Exception:
                    pass
                self.active_storage_monitor.stop()

                if self.emergency_stop_triggered:
                    self.logger.critical("Emergency stop triggered. Aborting pipeline.")
                    break

                if self.stop_requested:
                    self.logger.info("Stop requested by user. Aborting pipeline.")
                    break

                if not encode_success:
                    self.logger.error(f"Encoding failed for {media_file.source.name}: {encode_msg}")
                    self.failed_count += 1
                    self.db.update_status(
                        media_file.source,
                        status="failed",
                        output_path=media_file.output_path,
                        error=encode_msg,
                    )
                    continue

                # 6. Validate temporary output
                update_ui(status_msg=f"Validating temp output: {media_file.source.name}")
                self.db.update_status(media_file.source, status="validating", output_path=media_file.output_path)

                temp_path = media_file.temp_output_path
                is_valid, val_msg, _ = validate_converted_file(temp_path, media_file, self.config)

                if not is_valid:
                    self.logger.error(f"Validation failed for temp output {temp_path}: {val_msg}")
                    self.failed_count += 1
                    if temp_path and temp_path.exists():
                        try:
                            temp_path.unlink()
                        except Exception:
                            pass
                    self.db.update_status(
                        media_file.source,
                        status="failed",
                        output_path=media_file.output_path,
                        error=f"Validation failed: {val_msg}",
                    )
                    continue

                # Check if encoding bloated and prioritize saving space
                temp_size = temp_path.stat().st_size
                if self.config.processing.keep_smaller and temp_size >= media_file.size:
                    self.logger.warning(
                        f"Encoding bloated for {media_file.source.name}: output ({format_bytes(temp_size)}) "
                        f"is larger than or equal to original ({format_bytes(media_file.size)}). "
                        f"Prioritizing space: keeping smaller original and removing bloated output."
                    )
                    if temp_path and temp_path.exists():
                        try:
                            temp_path.unlink()
                            self.logger.info(f"Removed bloated temporary output: {temp_path}")
                        except Exception as e:
                            self.logger.error(f"Failed to remove bloated temp file {temp_path}: {e}")

                    self.skipped_count += 1
                    self.db.update_status(
                        media_file.source,
                        status="skipped",
                        output_path=media_file.output_path,
                        skip_reason=(
                            f"Encoding bloated: original is smaller "
                            f"({format_bytes(media_file.size)} vs {format_bytes(temp_size)})"
                        ),
                        source_bytes=media_file.size,
                        output_bytes=media_file.size,
                    )
                    update_ui(status_msg=f"Skipped bloated {media_file.source.name}")
                    continue

                # 7. Atomic promotion (rename .encoding.mkv -> final .mkv)
                final_path = media_file.output_path
                update_ui(status_msg=f"Promoting {media_file.source.name}")
                try:
                    os.replace(str(temp_path), str(final_path))
                    self.logger.info(f"Promoted {temp_path.name} to {final_path.name}")
                except Exception as e:
                    self.logger.error(f"Failed to promote output file {temp_path} -> {final_path}: {e}")
                    self.failed_count += 1
                    self.db.update_status(
                        media_file.source,
                        status="failed",
                        output_path=media_file.output_path,
                        error=f"Atomic promotion failed: {e}",
                    )
                    continue

                # 8. Re-validate promoted final output
                is_final_valid, fval_msg, _ = validate_converted_file(final_path, media_file, self.config)
                if not is_final_valid:
                    self.logger.critical(f"Final validation failed on promoted file {final_path}: {fval_msg}")
                    self.failed_count += 1
                    self.db.update_status(
                        media_file.source,
                        status="failed",
                        output_path=media_file.output_path,
                        error=f"Post-promotion validation failed: {fval_msg}",
                    )
                    continue

                # 9. Safely delete original file if configured
                out_size = final_path.stat().st_size
                if self.config.processing.delete_original and not self.no_delete:
                    if media_file.source.exists() and media_file.source != final_path:
                        try:
                            media_file.source.unlink()
                            self.logger.info(f"Safely deleted original file: {media_file.source}")
                        except Exception as e:
                            self.logger.error(f"Failed to delete original file {media_file.source}: {e}")

                # 10. Mark Completed in DB & update stats
                self.completed_count += 1
                self.total_source_bytes_processed += media_file.size
                self.total_output_bytes_created += out_size
                
                self.db.update_status(
                    media_file.source,
                    status="completed",
                    output_path=final_path,
                    source_bytes=media_file.size,
                    output_bytes=out_size,
                )
                self.logger.info(
                    f"Successfully migrated {media_file.source.name} -> {final_path.name} "
                    f"({format_bytes(media_file.size)} -> {format_bytes(out_size)}, "
                    f"saved {format_bytes(media_file.size - out_size)})"
                )

                saved_bytes = max(0, self.total_source_bytes_processed - self.total_output_bytes_created)
                parts = [f"saved: {format_bytes(saved_bytes)}"]
                gpu_str = get_gpu_status_str()
                if gpu_str:
                    parts.append(gpu_str)
                try:
                    overall_prog.update(self.completed_count, postfix_str=" | ".join(parts))
                except Exception:
                    pass

                update_ui(status_msg=f"Completed {media_file.source.name}")

        finally:
            try:
                overall_prog.finish()
            except Exception:
                pass
            self.gpu_monitor.stop()
            if self.active_storage_monitor:
                self.active_storage_monitor.stop()
            self.dashboard.stop()

        # Final Summary
        saved_bytes = max(0, self.total_source_bytes_processed - self.total_output_bytes_created)
        pct = (saved_bytes / self.total_source_bytes_processed * 100.0) if self.total_source_bytes_processed > 0 else 0.0

        final_summary = {
            "completed": self.completed_count,
            "failed": self.failed_count,
            "skipped": self.skipped_count,
            "source_processed": self.total_source_bytes_processed,
            "output_created": self.total_output_bytes_created,
            "saved_bytes": saved_bytes,
            "reduction_percent": pct,
        }

        self.logger.info("=== Migration Summary ===")
        self.logger.info(f"Completed: {self.completed_count}")
        self.logger.info(f"Failed:    {self.failed_count}")
        self.logger.info(f"Skipped:   {self.skipped_count}")
        self.logger.info(f"Source:    {format_bytes(self.total_source_bytes_processed)}")
        self.logger.info(f"Output:    {format_bytes(self.total_output_bytes_created)}")
        self.logger.info(f"Saved:     {format_bytes(saved_bytes)} ({pct:.1f}%)")

        failed_files = self.db.get_failed_files()
        if failed_files:
            self.logger.warning(f"=== FAILED FILES ({len(failed_files)}) ===")
            for ff in failed_files:
                self.logger.warning(f"  - {ff['source_path']}: {ff['error'] or ff['skip_reason']}")

        return final_summary
