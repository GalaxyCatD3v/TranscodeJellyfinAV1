"""
Standalone CLI helper script to mount or unmount SSHFS for Galaxy AV1 Migrator.
Reads configuration from .env file or command line flags.
"""

import argparse
import sys
from pathlib import Path

from av1_migrator.sshfs import SSHFSConfig, is_drive_accessible, mount_sshfs, unmount_sshfs


def main():
    parser = argparse.ArgumentParser(description="Mount or unmount remote SSHFS storage using .env configuration.")
    parser.add_argument("--env", type=str, default=".env", help="Path to .env configuration file (default: .env)")
    parser.add_argument("--unmount", action="store_true", help="Unmount the SSHFS network drive")
    parser.add_argument("--check", action="store_true", help="Check if the SSHFS network drive is mounted and accessible")
    parser.add_argument("--drive", type=str, default=None, help="Override target mount drive letter (e.g. Y:)")
    parser.add_argument("--host", type=str, default=None, help="Override SSH host / IP")
    parser.add_argument("--user", type=str, default=None, help="Override SSH user")
    parser.add_argument("--port", type=int, default=None, help="Override SSH port")
    parser.add_argument("--password", type=str, default=None, help="Override SSH password")

    args = parser.parse_args()

    cfg = SSHFSConfig.from_env(args.env)
    if args.drive:
        cfg.mount_drive = args.drive if args.drive.endswith(":") else f"{args.drive}:"
    if args.host:
        cfg.host = args.host
    if args.user:
        cfg.user = args.user
    if args.port:
        cfg.port = args.port
    if args.password is not None:
        cfg.password = args.password

    if args.check:
        is_up = is_drive_accessible(cfg.mount_drive)
        status = "ACCESSIBLE" if is_up else "NOT ACCESSIBLE / UNMOUNTED"
        print(f"Drive {cfg.mount_drive} status: {status}")
        sys.exit(0 if is_up else 1)

    if args.unmount:
        ok, msg = unmount_sshfs(cfg.mount_drive)
        print(msg)
        sys.exit(0 if ok else 1)

    # Mount
    print(f"Attempting SSHFS mount: {cfg.user}@{cfg.host}:{cfg.remote_path} -> {cfg.mount_drive} (Port: {cfg.port})")
    ok, msg, _ = mount_sshfs(cfg, env_path=args.env)
    print(msg)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
