"""
Workflow orchestration engine for Galaxy AV1 Migrator.
Enforces strict safety priorities, SQLite state persistence, multi-worker concurrency
(NVIDIA RTX 4070 Ti GPU + concurrent CPU worker), local NVMe staging, and 3-line tqdm progress display.
"""

from datetime import datetime
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from av1_migrator.config import AppConfig, format_bytes, parse_size_to_bytes
from av1_migrator.db import MigrationDB
from av1_migrator.encoder import (
    FFmpegEncoder,
    get_available_cpu_av1_encoder,
    is_av1_nvenc_available,
    is_libaom_available,
    is_libsvtav1_available,
    is_scale_cuda_available,
)
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
    create_worker_progressbar,
    MigrationProgressBar,
)
from av1_migrator.scanner import is_encoding_temp_file, scan_all_roots
from av1_migrator.storage import check_storage_safety, get_path_disk_usage, StorageMonitorThread
from av1_migrator.utils import find_binary_executable, normalize_filepath, sanitize_filename
from av1_migrator.validator import validate_converted_file


class MigrationEngine:
    def __init__(
        self,
        config: AppConfig,
        db: Optional[MigrationDB] = None,
        dashboard: Optional[Any] = None,
        dry_run: bool = False,
        limit: Optional[int] = None,
        single_file: Optional[str] = None,
        retry_failed: bool = False,
        no_delete: bool = False,
        no_ui: bool = False,
    ):
        self.config = config
        self.db = db or MigrationDB(config.database.path)
        self.dashboard = dashboard
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

        # Thread synchronization & active process tracking
        self.lock = threading.Lock()
        self.queue_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        self.encoders_lock = threading.Lock()
        self.monitors_lock = threading.Lock()
        self.staged_lock = threading.Lock()

        self.active_encoder: Optional[FFmpegEncoder] = None
        self.active_media_file: Optional[MediaFile] = None
        self.active_storage_monitor: Optional[StorageMonitorThread] = None

        self.active_encoders: List[FFmpegEncoder] = []
        self.active_storage_monitors: List[StorageMonitorThread] = []
        self.staged_files: List[Path] = []

        self.total_source_bytes_processed = 0
        self.total_output_bytes_created = 0
        self.completed_count = 0
        self.skipped_count = 0
        self.failed_count = 0
        self.queue: List[MediaFile] = []
        self.total_files_count = 0

        self.start_time = 0.0
        self._overall_pbar: Optional[MigrationProgressBar] = None

    def run_preflight_checks(self) -> Tuple[bool, List[str]]:
        """
        Executes pre-flight checks:
        Python version, FFmpeg, FFprobe, NVIDIA GPU, AV1 NVENC, CUDA scaling, CPU encoder,
        media roots, local staging, storage floor.
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
        gpu_workers_cnt = getattr(self.config.processing, "gpu_workers", 2)
        if is_av1_nvenc_available(self.config.ffmpeg.executable):
            checks.append(f"✓ AV1 NVENC encoder available (GPU Workers: {gpu_workers_cnt})")
        else:
            checks.append("⚠ AV1 NVENC encoder not found in FFmpeg (GPU encoding disabled)")

        # CUDA scaling
        interp = getattr(self.config.output, "cuda_interp_algo", "bicubic")
        if is_scale_cuda_available(self.config.ffmpeg.executable):
            checks.append(f"✓ CUDA scaling (scale_cuda, algo: {interp})")
        else:
            checks.append(f"⚠ CUDA scaling not found (will use software {interp} fallback)")

        # CPU AV1 Encoder
        cpu_codec = get_available_cpu_av1_encoder(self.config.ffmpeg.executable)
        checks.append(f"✓ CPU AV1 encoder available ({cpu_codec})")

        # Local Staging Directory (e.g. Z:\JellyfinTranscode)
        if self.config.storage.enable_local_staging:
            stg_dir = Path(self.config.storage.local_staging_dir)
            try:
                stg_dir.mkdir(parents=True, exist_ok=True)
                total, used, free = shutil.disk_usage(str(stg_dir))
                checks.append(f"✓ Local NVMe staging: {stg_dir} (Free: {format_bytes(free)})")
            except Exception as e:
                checks.append(f"⚠ Local staging directory inaccessible ({stg_dir}): {e}")

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
        1. Signals active FFmpeg processes to terminate safely.
        2. Cleans up temporary staged files from local SSD and remote paths.
        3. Preserves original files.
        4. Updates database.
        """
        self.stop_requested = True
        self.logger.info(f"Stopping migration safely: {reason}")

        with self.encoders_lock:
            for enc in list(self.active_encoders):
                enc.abort(reason=reason)

        with self.monitors_lock:
            for mon in list(self.active_storage_monitors):
                mon.stop()

        with self.lock:
            if self.active_media_file:
                self.db.update_status(
                    self.active_media_file.source,
                    status="aborted",
                    output_path=self.active_media_file.output_path,
                    error=reason,
                )

        # Clean staged files
        with self.staged_lock:
            for f in list(self.staged_files):
                if f.exists():
                    try:
                        f.unlink(missing_ok=True)
                        self.logger.info(f"Cleaned up staged file: {f}")
                    except Exception as e:
                        self.logger.error(f"Failed to remove staged file {f}: {e}")

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
        Also cleans leftover files in local staging directory (Z:/JellyfinTranscode).
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
                MigrationProgressBar.write(f"Startup Recovery: {Path(act['source_path']).name} -> {act['action']}")
        else:
            self.logger.info("Startup recovery: Database is clean, no interrupted tasks found.")

        # Clean local staging directory of stale artifacts from interrupted runs
        if self.config.storage.enable_local_staging:
            stg_dir = Path(self.config.storage.local_staging_dir)
            if stg_dir.exists():
                try:
                    for f in stg_dir.iterdir():
                        if f.is_file() and (f.name.startswith("gpu") or f.name.startswith("cpu") or ".encoding.mkv" in f.name):
                            try:
                                f.unlink(missing_ok=True)
                                self.logger.info(f"Startup recovery: cleaned leftover local staging file {f}")
                            except Exception as e:
                                self.logger.warning(f"Could not delete leftover staging file {f}: {e}")
                except Exception as e:
                    self.logger.warning(f"Error checking local staging directory during startup: {e}")

        return actions

    def scan_and_prepare_queue(self) -> Tuple[List[MediaFile], ScanStats]:
        """
        Scans media roots, probes files, updates SQLite DB, and returns sorted queue.
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

                # If already completed and unchanged, skip
                if (
                    db_row
                    and db_row["status"] == "completed"
                    and db_row["source_size"] == file_size
                    and abs(db_row["source_mtime"] - file_mtime) <= 1.0
                    and not self.single_file
                ):
                    scan_stats.already_converted += 1
                    self.completed_count += 1
                    self.total_source_bytes_processed += file_size
                    self.total_output_bytes_created += db_row["output_bytes"] or file_size
                    continue

                # If already marked skipped in DB and file unchanged
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
                        self.logger.warning(f"Existing output {mf.output_path.name} is invalid/incomplete: {v_msg}. Removing to re-encode.")
                        try:
                            mf.output_path.unlink()
                        except Exception as e:
                            self.logger.error(f"Could not remove invalid existing file {mf.output_path}: {e}")

                # Candidate is eligible for conversion!
                scan_stats.eligible_files += 1
                mf.status = "pending"
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

        try:
            do_probing_loop()
        finally:
            try:
                probe_prog.finish()
            except Exception:
                pass

        self.queue = media_files_to_process
        self.total_files_count = len(media_files_to_process)
        return media_files_to_process, scan_stats

    def get_gpu_status_str(self) -> str:
        g = self.gpu_monitor.stats
        parts = []
        if g.gpu_util is not None:
            parts.append(f"GPU: {g.gpu_util:.0f}%")
        if g.enc_util is not None:
            parts.append(f"NVENC: {g.enc_util:.0f}%")
        if g.temperature_c is not None:
            parts.append(f"{int(g.temperature_c)}°C")
        return " | ".join(parts) if parts else ""

    def _update_overall_pbar(self) -> None:
        if self._overall_pbar:
            with self.stats_lock:
                done = self.completed_count
                saved_bytes = max(0, self.total_source_bytes_processed - self.total_output_bytes_created)
            parts = [f"saved: {format_bytes(saved_bytes)}"]
            gpu_str = self.get_gpu_status_str()
            if gpu_str:
                parts.append(gpu_str)
            try:
                self._overall_pbar.update(done, postfix_str=" | ".join(parts))
            except Exception:
                pass

    def _clean_staged(self, *paths: Optional[Path]) -> None:
        for p in paths:
            if p and p.exists():
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
                with self.staged_lock:
                    if p in self.staged_files:
                        self.staged_files.remove(p)

    def _process_media_file(
        self,
        media_file: MediaFile,
        worker_type: str,
        worker_pbar: MigrationProgressBar,
    ) -> None:
        """
        Processes a single media file end-to-end for a given worker (GPU or CPU):
        Pre-encode check -> Local staging (Z:/JellyfinTranscode) -> Encoding ->
        Validation -> Bloat rejection -> Promotion -> Cleanup.
        """
        with self.lock:
            self.active_media_file = media_file

        # Check pause
        while self.is_paused and not self.stop_requested and not self.emergency_stop_triggered:
            time.sleep(0.5)

        if self.stop_requested or self.emergency_stop_triggered:
            return

        # 1. Storage safety check (remote storage floor)
        is_safe, s_msg, s_stats = check_storage_safety(
            path=media_file.source,
            original_file_size=media_file.size,
            minimum_free_space_bytes=self.config.storage.minimum_free_space_bytes,
            safety_margin_bytes=self.config.storage.safety_margin_bytes,
        )

        if not is_safe:
            self.logger.critical(f"Storage check failed before starting {media_file.source.name}: {s_msg}")
            self.trigger_emergency_stop(s_stats)
            return

        # 2. Pre-encode optimization check: test sample to predict bloat
        if self.config.processing.pre_encode_check:
            opt_res = evaluate_pre_encode_optimization(media_file, self.config)
            if not opt_res.should_encode:
                self.logger.warning(f"Pre-encode check: Skipping {media_file.source.name}. {opt_res.reason}")
                with self.stats_lock:
                    self.skipped_count += 1
                self.db.update_status(
                    media_file.source,
                    status="skipped",
                    output_path=media_file.output_path,
                    skip_reason=opt_res.reason,
                    source_bytes=media_file.size,
                    output_bytes=media_file.size,
                )
                self._update_overall_pbar()
                worker_pbar.set_description(f"{worker_type} [Idle]")
                worker_pbar.set_postfix_str(f"Skipped {media_file.source.name[:20]}")
                return

        # 3. Local Staging at Z:\JellyfinTranscode to eliminate remote SSHFS reading bottlenecks
        staged_source: Optional[Path] = None
        staged_output: Optional[Path] = None
        active_input = media_file.source
        active_output = media_file.temp_output_path

        if self.config.storage.enable_local_staging:
            staging_dir = Path(self.config.storage.local_staging_dir)
            try:
                staging_dir.mkdir(parents=True, exist_ok=True)
                _, _, local_free = shutil.disk_usage(str(staging_dir))
                required_local = media_file.size * 2 + self.config.storage.local_min_free_space_bytes

                if local_free >= required_local:
                    stem_clean = sanitize_filename(media_file.source.stem)
                    ext_clean = media_file.source.suffix
                    worker_tag = sanitize_filename(worker_type.lower().replace(" ", "_"))
                    staged_source = staging_dir / f"{worker_tag}_{stem_clean}{ext_clean}"
                    hdr_tag = "HDR10" if media_file.hdr else "SDR"
                    staged_output = staging_dir / f"{worker_tag}_{stem_clean} [AV1 1080p {hdr_tag} CQ28].mkv.encoding.mkv"

                    worker_pbar.set_description(f"{worker_type} [{media_file.source.name[:25]}]: Staging locally...")
                    worker_pbar.set_postfix_str(f"Copying {format_bytes(media_file.size)} to {staging_dir.drive or staging_dir}...")

                    shutil.copy2(media_file.source, staged_source)
                    active_input = staged_source
                    active_output = staged_output

                    with self.staged_lock:
                        self.staged_files.extend([staged_source, staged_output])
                else:
                    self.logger.warning(
                        f"Local staging space low on {staging_dir} ({format_bytes(local_free)} free), encoding directly."
                    )
            except Exception as e:
                self.logger.warning(f"Could not stage {media_file.source.name} locally ({staging_dir}): {e}. Encoding directly.")

        # 4. Continuous background storage monitor for this encode
        monitor = StorageMonitorThread(
            watch_path=media_file.source,
            minimum_free_space_bytes=self.config.storage.minimum_free_space_bytes,
            poll_interval=self.config.storage.poll_interval,
            on_emergency_stop=self.trigger_emergency_stop,
        )
        monitor.start()
        with self.monitors_lock:
            self.active_storage_monitors.append(monitor)

        # 5. Reset worker progress bar and configure progress handler
        worker_pbar.reset(total=100.0, desc=f"{worker_type} [{media_file.source.name[:25]}]")

        def on_progress(p: EncodeProgress):
            in_sz = format_bytes(media_file.size)
            out_sz = format_bytes(p.total_size_bytes)
            eta_val = getattr(p, "eta_str", "--:--:--")
            parts = [
                f"in: {in_sz} -> out: {out_sz}",
                f"{p.speed:.2f}x ({p.fps:.1f} fps)",
                f"ETA {eta_val}",
            ]
            if worker_type.startswith("GPU"):
                gpu_str = self.get_gpu_status_str()
                if gpu_str:
                    parts.append(gpu_str)
            else:
                parts.append("CPU worker")
            try:
                worker_pbar.update(p.percent, postfix_str=" | ".join(parts))
            except Exception:
                pass

        # 6. Instantiate and run encoder (with graceful fallback for custom test mocks)
        enc_type = "cpu" if worker_type.startswith("CPU") else "gpu"
        try:
            encoder = FFmpegEncoder(
                media_file=media_file,
                config=self.config,
                on_progress=on_progress,
                encoder_type=enc_type,
                input_path=active_input,
                output_path=active_output,
            )
        except TypeError:
            encoder = FFmpegEncoder(
                media_file=media_file,
                config=self.config,
                on_progress=on_progress,
            )

        with self.encoders_lock:
            self.active_encoders.append(encoder)
            self.active_encoder = encoder

        self.db.update_status(media_file.source, status="encoding", output_path=media_file.output_path)
        encode_success, encode_msg = encoder.run()

        with self.encoders_lock:
            if encoder in self.active_encoders:
                self.active_encoders.remove(encoder)

        monitor.stop()
        with self.monitors_lock:
            if monitor in self.active_storage_monitors:
                self.active_storage_monitors.remove(monitor)

        # Support test mocks that wrote to media_file.temp_output_path
        if (
            active_output
            and media_file.temp_output_path
            and not active_output.exists()
            and media_file.temp_output_path.exists()
        ):
            active_output = media_file.temp_output_path

        # 7. Check abort / emergency stop
        if self.emergency_stop_triggered or self.stop_requested:
            self._clean_staged(staged_source, staged_output)
            return

        if not encode_success:
            self.logger.error(f"Encoding failed for {media_file.source.name} ({worker_type}): {encode_msg}")
            self._clean_staged(staged_source, staged_output)
            with self.stats_lock:
                self.failed_count += 1
            self.db.update_status(
                media_file.source,
                status="failed",
                output_path=media_file.output_path,
                error=encode_msg,
            )
            self._update_overall_pbar()
            worker_pbar.set_description(f"{worker_type} [Idle]")
            worker_pbar.set_postfix_str(f"Failed: {encode_msg[:25]}")
            return

        # 8. Output validation
        worker_pbar.set_description(f"{worker_type} [{media_file.source.name[:25]}]: Validating...")
        self.db.update_status(media_file.source, status="validating", output_path=media_file.output_path)

        is_valid, val_msg, _ = validate_converted_file(active_output, media_file, self.config)
        if not is_valid:
            self.logger.error(f"Validation failed for {active_output}: {val_msg}")
            self._clean_staged(staged_source, active_output)
            with self.stats_lock:
                self.failed_count += 1
            self.db.update_status(
                media_file.source,
                status="failed",
                output_path=media_file.output_path,
                error=f"Validation failed: {val_msg}",
            )
            self._update_overall_pbar()
            worker_pbar.set_description(f"{worker_type} [Idle]")
            worker_pbar.set_postfix_str("Validation failed")
            return

        # 9. Space bloat rejection: discard bloated encode and keep smaller original
        out_sz = active_output.stat().st_size
        if self.config.processing.keep_smaller and out_sz >= media_file.size:
            self.logger.warning(
                f"Encoding bloated for {media_file.source.name}: output ({format_bytes(out_sz)}) "
                f"is larger than or equal to original ({format_bytes(media_file.size)}). Prioritizing space."
            )
            self._clean_staged(staged_source, active_output)
            with self.stats_lock:
                self.skipped_count += 1
            self.db.update_status(
                media_file.source,
                status="skipped",
                output_path=media_file.output_path,
                skip_reason=(
                    f"Encoding bloated: original is smaller "
                    f"({format_bytes(media_file.size)} vs {format_bytes(out_sz)})"
                ),
                source_bytes=media_file.size,
                output_bytes=media_file.size,
            )
            self._update_overall_pbar()
            worker_pbar.set_description(f"{worker_type} [Idle]")
            worker_pbar.set_postfix_str("Skipped bloated output")
            return

        # 10. Atomic promotion to destination
        worker_pbar.set_description(f"{worker_type} [{media_file.source.name[:25]}]: Promoting...")
        final_path = media_file.output_path

        if active_output != final_path:
            # Staged locally: copy to remote staging temp then atomic rename
            remote_temp = media_file.temp_output_path or (media_file.source.parent / f"{media_file.source.stem}.encoding.mkv")
            try:
                if active_output != remote_temp:
                    shutil.copy2(active_output, remote_temp)
                    os.replace(str(remote_temp), str(final_path))
                    active_output.unlink(missing_ok=True)
                else:
                    os.replace(str(active_output), str(final_path))
            except Exception as e:
                self.logger.error(f"Failed promoting from local staging to {final_path}: {e}")
                self._clean_staged(staged_source, active_output)
                with self.stats_lock:
                    self.failed_count += 1
                self.db.update_status(media_file.source, status="failed", error=f"Promotion failed: {e}")
                self._update_overall_pbar()
                return
        else:
            try:
                os.replace(str(active_output), str(final_path))
            except Exception as e:
                self.logger.error(f"Failed to replace {active_output} -> {final_path}: {e}")
                with self.stats_lock:
                    self.failed_count += 1
                self.db.update_status(media_file.source, status="failed", error=f"Promotion failed: {e}")
                self._update_overall_pbar()
                return

        # Clean staged source
        if staged_source and staged_source.exists():
            staged_source.unlink(missing_ok=True)
            with self.staged_lock:
                if staged_source in self.staged_files:
                    self.staged_files.remove(staged_source)

        # 11. Final validation on remote promoted file
        is_final_valid, fval_msg, _ = validate_converted_file(final_path, media_file, self.config)
        if not is_final_valid:
            self.logger.critical(f"Final validation failed on promoted file {final_path}: {fval_msg}")
            with self.stats_lock:
                self.failed_count += 1
            self.db.update_status(
                media_file.source,
                status="failed",
                output_path=final_path,
                error=f"Post-promotion validation failed: {fval_msg}",
            )
            self._update_overall_pbar()
            return

        # 12. Safely delete remote original if configured
        final_size = final_path.stat().st_size
        if self.config.processing.delete_original and not self.no_delete:
            if media_file.source.exists() and media_file.source != final_path:
                try:
                    media_file.source.unlink()
                    self.logger.info(f"Safely deleted original file: {media_file.source}")
                except Exception as e:
                    self.logger.error(f"Failed to delete original file {media_file.source}: {e}")

        # 13. Mark Completed in DB & update stats
        with self.stats_lock:
            self.completed_count += 1
            self.total_source_bytes_processed += media_file.size
            self.total_output_bytes_created += final_size

        self.db.update_status(
            media_file.source,
            status="completed",
            output_path=final_path,
            source_bytes=media_file.size,
            output_bytes=final_size,
        )

        self.logger.info(
            f"[{worker_type}] Successfully migrated {media_file.source.name} -> {final_path.name} "
            f"({format_bytes(media_file.size)} -> {format_bytes(final_size)}, "
            f"saved {format_bytes(media_file.size - final_size)})"
        )
        self._update_overall_pbar()
        worker_pbar.set_description(f"{worker_type} [Idle]")
        worker_pbar.set_postfix_str(f"Completed {media_file.source.name[:20]}")

    def _worker_loop(self, worker_type: str, worker_pbar: MigrationProgressBar) -> None:
        """
        Continuous worker loop for GPU or CPU transcode thread.
        Pulls from self.queue until empty or stop requested.
        """
        while not self.stop_requested and not self.emergency_stop_triggered:
            while self.is_paused and not self.stop_requested and not self.emergency_stop_triggered:
                time.sleep(0.5)

            if self.stop_requested or self.emergency_stop_triggered:
                break

            media_file: Optional[MediaFile] = None
            with self.queue_lock:
                if not self.queue:
                    break

                if worker_type.startswith("CPU"):
                    # CPU worker prioritizes smaller files to prevent getting bogged down on 80GB Remuxes
                    # while GPU handles the large files
                    if len(self.queue) > 1 and self.config.processing.enable_gpu_encoding:
                        media_file = self.queue.pop(-1)
                    else:
                        media_file = self.queue.pop(0)
                else:
                    # GPU worker processes largest files first
                    media_file = self.queue.pop(0)

            if media_file is None:
                break

            try:
                self._process_media_file(media_file, worker_type, worker_pbar)
            except Exception as e:
                self.logger.exception(f"Unhandled exception in {worker_type} worker on {media_file.source.name}: {e}")
                with self.stats_lock:
                    self.failed_count += 1
                self.db.update_status(media_file.source, status="failed", error=str(e))
                self._update_overall_pbar()

        worker_pbar.set_description(f"{worker_type} [Finished]")
        worker_pbar.set_postfix_str("All tasks complete")
        worker_pbar.finish()

    def execute(self) -> Dict[str, Any]:
        """
        Executes the migration pipeline:
        1. Pre-flight checks
        2. Signal handlers
        3. Startup database and staging directory recovery
        4. Discovery and probing
        5. Multi-line tqdm progress bars (Overall, CPU, GPU)
        6. Concurrent GPU and CPU transcode workers
        """
        self.is_running = True
        self.start_time = time.time()
        self.setup_signal_handlers()

        # 1. Pre-flight checks
        self.logger.info("=== Pre-flight System Checks ===")
        ok, check_lines = self.run_preflight_checks()
        for line in check_lines:
            self.logger.info(line)
            MigrationProgressBar.write(line)

        if not ok:
            msg = "Pre-flight system checks failed. Halting migration."
            self.logger.critical(msg)
            MigrationProgressBar.write(f"\n{msg}")
            return {"status": "error", "reason": msg}

        # 2. Startup recovery
        self.run_startup_recovery()

        # 3. Queue preparation
        queue, scan_stats = self.scan_and_prepare_queue()

        if self.dry_run:
            total_source_bytes_in_queue = sum(mf.size for mf in queue)
            estimated_output_bytes = int(total_source_bytes_in_queue * 0.3)
            estimated_savings_bytes = max(0, total_source_bytes_in_queue - estimated_output_bytes)

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
            MigrationProgressBar.write(summary)
            return {
                "status": "dry_run_complete",
                "queue_count": len(queue),
                "total_source_bytes": total_source_bytes_in_queue,
                "estimated_savings_bytes": estimated_savings_bytes,
            }

        if not queue:
            self.logger.info("No eligible files to transcode in queue. All files up to date!")
            MigrationProgressBar.write("All files up to date! Nothing to migrate.")
            return {"status": "completed", "converted_count": 0}

        # 4. Start GPU telemetry
        self.gpu_monitor.start()

        # 5. Initialize dynamic multi-line tqdm progress bars
        # Position 0: Overall Library
        # Position 1..N: GPU Worker(s)
        # Position N+1: CPU Worker (if enabled)
        overall_prog = create_overall_progressbar(len(queue), message="Overall Migration", position=0)
        overall_prog.start()
        self._overall_pbar = overall_prog

        gpu_active = self.config.processing.enable_gpu_encoding and is_av1_nvenc_available(self.config.ffmpeg.executable)
        cpu_active = self.config.processing.enable_cpu_encoding
        num_gpu_workers = max(1, getattr(self.config.processing, "gpu_workers", 2)) if gpu_active else 0

        pbars: List[MigrationProgressBar] = []
        threads: List[threading.Thread] = []
        cur_pos = 1

        if num_gpu_workers > 0:
            for g_idx in range(1, num_gpu_workers + 1):
                w_name = f"GPU {g_idx}" if num_gpu_workers > 1 else "GPU"
                g_pbar = create_worker_progressbar(w_name, position=cur_pos)
                g_pbar.start()
                pbars.append(g_pbar)
                cur_pos += 1

                t_gpu = threading.Thread(
                    target=self._worker_loop,
                    args=(w_name, g_pbar),
                    name=f"{w_name.replace(' ', '-')}-Worker",
                    daemon=True,
                )
                threads.append(t_gpu)
        else:
            gpu_prog = create_worker_progressbar("GPU", position=cur_pos)
            gpu_prog.start()
            gpu_prog.set_description("GPU [Disabled / NVENC unavailable]")
            gpu_prog.set_postfix_str("Idle")
            pbars.append(gpu_prog)
            cur_pos += 1

        if cpu_active:
            cpu_prog = create_worker_progressbar("CPU", position=cur_pos)
            cpu_prog.start()
            pbars.append(cpu_prog)
            cur_pos += 1

            t_cpu = threading.Thread(
                target=self._worker_loop,
                args=("CPU", cpu_prog),
                name="CPU-Worker",
                daemon=True,
            )
            threads.append(t_cpu)
        else:
            cpu_prog = create_worker_progressbar("CPU", position=cur_pos)
            cpu_prog.start()
            cpu_prog.set_description("CPU [Disabled in config]")
            cpu_prog.set_postfix_str("Idle")
            pbars.append(cpu_prog)
            cur_pos += 1

        # If neither worker active, fallback to GPU
        if not threads:
            fallback_pbar = pbars[0] if pbars else create_worker_progressbar("GPU", position=1)
            t_fallback = threading.Thread(
                target=self._worker_loop,
                args=("GPU", fallback_pbar),
                name="Fallback-Worker",
                daemon=True,
            )
            threads.append(t_fallback)

        for t in threads:
            t.start()

        try:
            for t in threads:
                t.join()
        finally:
            try:
                overall_prog.finish()
            except Exception:
                pass
            for pb in pbars:
                try:
                    pb.finish()
                except Exception:
                    pass
            self.gpu_monitor.stop()

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

        MigrationProgressBar.write("")
        MigrationProgressBar.write("=" * 60)
        MigrationProgressBar.write("GALAXY AV1 MIGRATION COMPLETE")
        MigrationProgressBar.write(f"Completed: {self.completed_count} | Failed: {self.failed_count} | Skipped: {self.skipped_count}")
        MigrationProgressBar.write(f"Source: {format_bytes(self.total_source_bytes_processed)} -> Output: {format_bytes(self.total_output_bytes_created)}")
        MigrationProgressBar.write(f"Saved:  {format_bytes(saved_bytes)} ({pct:.1f}% reduction)")
        MigrationProgressBar.write("=" * 60)

        failed_files = self.db.get_failed_files()
        if failed_files:
            self.logger.warning(f"=== FAILED FILES ({len(failed_files)}) ===")
            for ff in failed_files:
                self.logger.warning(f"  - {ff['source_path']}: {ff['error'] or ff['skip_reason']}")

        return final_summary
