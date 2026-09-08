"""
Command Line Interface for Galaxy AV1 Migrator.
"""

import argparse
from pathlib import Path
import sys
from rich.console import Console

from av1_migrator.config import load_config
from av1_migrator.db import MigrationDB
from av1_migrator.engine import MigrationEngine
from av1_migrator.logger import setup_logger

console = Console()


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
        choices=["largest_first", "smallest_first"],
        default=None,
        help="Sort queue by file size ('largest_first' or 'smallest_first')",
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

    # Setup logger
    logger = setup_logger(
        log_dir=config.logging.directory,
        filename=config.logging.filename,
        max_bytes=config.logging.max_bytes,
        backup_count=config.logging.backup_count,
        verbose=args.verbose,
        enable_console=args.no_ui or not sys.stdout.isatty(),
    )

    db = MigrationDB(config.database.path)

    # Handle --reset-file if requested
    if args.reset_file:
        reset_path = Path(args.reset_file).resolve()
        if db.reset_file(reset_path):
            logger.info(f"Successfully reset database record for: {reset_path}")
            console.print(f"Reset database record for: {reset_path}")
            return 0
        else:
            logger.warning(f"File not found in database: {reset_path}")
            console.print(f"File not found in database: {reset_path}")
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
        console.print("\n".join(lines))
        return 0 if ok else 1

    try:
        result = engine.execute()
        if result.get("status") == "error":
            return 1
        return 0
    except KeyboardInterrupt:
        console.print("\nStopping safely...")
        engine.stop_safely("KeyboardInterrupt")
        return 130
    except Exception as e:
        logger.exception(f"Unhandled error in main migration engine: {e}")
        console.print(f"\nFatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
