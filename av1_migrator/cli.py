"""
Command Line Interface for Galaxy AV1 Migrator.
"""

import argparse
from pathlib import Path
import sys

from av1_migrator.benchmark import run_interpolation_benchmark
from av1_migrator.config import load_config
from av1_migrator.db import MigrationDB
from av1_migrator.engine import MigrationEngine
from av1_migrator.logger import setup_logger
from av1_migrator.progress import MigrationProgressBar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Galaxy AV1 Migrator",
        description="Safely migrate Jellyfin media library to AV1 1080p using NVIDIA NVENC over SSHFS.",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and calculate estimated savings without executing any encodes",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Limit number of media files to transcode in this run",
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Process a single specific media file",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Reset previously failed or aborted files in the database back to pending",
    )
    parser.add_argument(
        "--no-delete",
        action="store_true",
        help="Do not delete original source files after successful conversion and validation",
    )
    parser.add_argument(
        "--sort",
        type=str,
        choices=["largest_first", "smallest_first", "biggest_first"],
        default=None,
        help="Sort queue by file size ('largest_first' or 'smallest_first')",
    )
    parser.add_argument(
        "--biggest-first", "--largest-first",
        dest="sort",
        action="store_const",
        const="largest_first",
        help="Sort queue to process largest files first",
    )
    parser.add_argument(
        "--smallest-first",
        dest="sort",
        action="store_const",
        const="smallest_first",
        help="Sort queue to process smallest files first",
    )
    parser.add_argument(
        "--pre-check",
        dest="pre_encode_check",
        action="store_true",
        default=None,
        help="Enable pre-encode sample testing to predict space savings and prevent bloated transcodes",
    )
    parser.add_argument(
        "--no-pre-check",
        dest="pre_encode_check",
        action="store_false",
        help="Disable pre-encode sample testing",
    )
    parser.add_argument(
        "--sample-duration",
        type=float,
        default=None,
        help="Duration in seconds of sample clip for pre-encode testing (default: 30.0)",
    )
    parser.add_argument(
        "--min-savings",
        type=float,
        default=None,
        help="Minimum percentage savings required to proceed with encoding (default: 0.0)",
    )
    parser.add_argument(
        "--reset-file",
        type=str,
        default=None,
        help="Reset a specific file record in the migration database",
    )
    parser.add_argument(
        "--benchmark",
        nargs="?",
        const="",
        default=None,
        help="Benchmark CUDA scaling interpolation algorithms (bicubic, bilinear, lanczos) on a test clip or file",
    )
    parser.add_argument(
        "--interp-algo",
        type=str,
        choices=["bicubic", "bilinear", "lanczos"],
        default=None,
        help="Hardware CUDA scaling interpolation algorithm (default: bicubic)",
    )
    parser.add_argument(
        "--cpu",
        dest="enable_cpu",
        action="store_true",
        default=None,
        help="Enable concurrent CPU AV1 encoding worker",
    )
    parser.add_argument(
        "--no-cpu",
        dest="enable_cpu",
        action="store_false",
        help="Disable concurrent CPU AV1 encoding worker",
    )
    parser.add_argument(
        "--gpu1",
        dest="enable_gpu1",
        action="store_true",
        default=None,
        help="Enable GPU 1 NVENC encoding worker",
    )
    parser.add_argument(
        "--no-gpu1",
        dest="enable_gpu1",
        action="store_false",
        help="Disable GPU 1 NVENC encoding worker",
    )
    parser.add_argument(
        "--gpu2",
        dest="enable_gpu2",
        action="store_true",
        default=None,
        help="Enable GPU 2 NVENC encoding worker",
    )
    parser.add_argument(
        "--no-gpu2",
        dest="enable_gpu2",
        action="store_false",
        help="Disable GPU 2 NVENC encoding worker",
    )
    parser.add_argument(
        "--workers",
        type=str,
        default=None,
        help="Select active worker types as comma-separated list (e.g. 'cpu,gpu1,gpu2', 'gpu1,gpu2', 'gpu1', 'cpu')",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable all GPU encoding workers",
    )
    parser.add_argument(
        "--gpu-workers",
        type=int,
        default=None,
        help="Number of concurrent GPU NVENC workers (default: 2)",
    )
    parser.add_argument(
        "--force-scan", "--rescan",
        action="store_true",
        help="Force full remote filesystem scan even if last scan was < 24 hours ago",
    )
    parser.add_argument(
        "--scan-cache-hours",
        type=float,
        default=None,
        help="Maximum hours to cache remote scan results before performing fresh scan (default: 24.0)",
    )
    parser.add_argument(
        "--staging-dir",
        type=str,
        default=None,
        help="Path to local fast NVMe staging directory (default: F:\\JellyfinTranscode)",
    )
    parser.add_argument(
        "--no-staging",
        action="store_true",
        help="Disable local staging (encode directly over remote storage)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable detailed debug logging to file and console",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Disable interactive Rich live dashboard (logs only)",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run system pre-flight checks and exit",
    )
    return parser.parse_args()


def main() -> int:
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    if args.sort:
        config.processing.sort = args.sort
    if args.pre_encode_check is not None:
        config.processing.pre_encode_check = args.pre_encode_check
    if args.sample_duration is not None:
        config.processing.sample_duration = args.sample_duration
    if args.min_savings is not None:
        config.processing.min_savings_percent = args.min_savings
    if args.interp_algo:
        config.output.cuda_interp_algo = args.interp_algo
    if args.scan_cache_hours is not None:
        config.processing.scan_cache_hours = max(0.0, args.scan_cache_hours)

    # Worker configuration
    if args.workers is not None:
        tokens = [t.strip().lower() for t in args.workers.split(",") if t.strip()]
        config.processing.enable_cpu_encoding = "cpu" in tokens or "all" in tokens
        config.processing.enable_gpu1 = "gpu1" in tokens or "gpu" in tokens or "all" in tokens
        config.processing.enable_gpu2 = "gpu2" in tokens or "all" in tokens
        config.processing.enable_gpu_encoding = config.processing.enable_gpu1 or config.processing.enable_gpu2
    else:
        if args.enable_cpu is not None:
            config.processing.enable_cpu_encoding = args.enable_cpu
        if args.enable_gpu1 is not None:
            config.processing.enable_gpu1 = args.enable_gpu1
        if args.enable_gpu2 is not None:
            config.processing.enable_gpu2 = args.enable_gpu2
        if args.no_gpu:
            config.processing.enable_gpu_encoding = False
            config.processing.enable_gpu1 = False
            config.processing.enable_gpu2 = False
        else:
            config.processing.enable_gpu_encoding = config.processing.enable_gpu1 or config.processing.enable_gpu2

    if args.gpu_workers is not None:
        config.processing.gpu_workers = max(1, args.gpu_workers)
    if args.staging_dir:
        config.storage.local_staging_dir = args.staging_dir
    if args.no_staging:
        config.storage.enable_local_staging = False

    # Setup logger
    logger = setup_logger(
        log_dir=config.logging.directory,
        filename=config.logging.filename,
        max_bytes=config.logging.max_bytes,
        backup_count=config.logging.backup_count,
        verbose=args.verbose,
        enable_console=args.no_ui or not sys.stdout.isatty(),
    )

    # Handle --benchmark if requested
    if args.benchmark is not None:
        bench_src = args.benchmark if args.benchmark else args.file
        run_interpolation_benchmark(source_path=bench_src, config=config)
        return 0

    db = MigrationDB(config.database.path)

    # Handle --reset-file if requested
    if args.reset_file:
        reset_path = Path(args.reset_file).resolve()
        if db.reset_file(reset_path):
            logger.info(f"Successfully reset database record for: {reset_path}")
            MigrationProgressBar.write(f"Reset database record for: {reset_path}")
            return 0
        else:
            logger.warning(f"File not found in database: {reset_path}")
            MigrationProgressBar.write(f"File not found in database: {reset_path}")
            return 1

    engine = MigrationEngine(
        config=config,
        db=db,
        dry_run=args.dry_run,
        limit=args.limit,
        single_file=args.file,
        retry_failed=args.retry_failed,
        no_delete=args.no_delete,
        no_ui=args.no_ui,
    )

    if args.preflight_only:
        ok, lines = engine.run_preflight_checks()
        MigrationProgressBar.write("\n".join(lines))
        return 0 if ok else 1

    try:
        result = engine.execute()
        if result.get("status") == "error":
            return 1
        return 0
    except KeyboardInterrupt:
        MigrationProgressBar.write("\nStopping safely...")
        engine.stop_safely("KeyboardInterrupt")
        return 130
    except Exception as e:
        logger.exception(f"Unhandled error in main migration engine: {e}")
        MigrationProgressBar.write(f"\nFatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
