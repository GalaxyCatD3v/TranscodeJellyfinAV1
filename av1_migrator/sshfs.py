"""
SSHFS-Win Mounting and Management Module for Galaxy AV1 Migrator.
Handles loading SSH credentials and connection parameters from .env files,
building optimized sshfs arguments, secure password passing via stdin,
mounting/unmounting network drives, and connection health checks.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

from av1_migrator.logger import get_logger
from av1_migrator.utils import find_binary_executable


DEFAULT_SSHFS_EXE = r"C:\Program Files\SSHFS-Win\bin\sshfs.exe"
DEFAULT_SSHFS_X64_EXE = r"C:\Program Files (x86)\SSHFS-Win\bin\sshfs.exe"


def parse_env_file(env_path: Path | str) -> Dict[str, str]:
    """
    Parses a .env file into a dictionary of key-value pairs without requiring external packages.
    Supports comments, single/double quotes, and whitespace trimming.
    """
    env_vars: Dict[str, str] = {}
    p = Path(env_path)
    if not p.is_file():
        return env_vars

    try:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                env_vars[key] = val
    except Exception as e:
        get_logger().warning(f"Error reading .env file ({p}): {e}")

    return env_vars


@dataclass
class SSHFSConfig:
    host: str = "192.168.1.180"
    user: str = "root"
    port: int = 22
    password: str = ""
    remote_path: str = "/"
    mount_drive: str = "Y:"
    volname: str = "jellyfin docker lxc"
    sshfs_exe: str = DEFAULT_SSHFS_EXE
    debug: bool = True
    loglevel: str = "debug1"
    max_readahead: str = "1GB"
    auto_mount: bool = True

    @classmethod
    def from_env(cls, env_path: Optional[Path | str] = None) -> "SSHFSConfig":
        """
        Loads SSHFS configuration by checking the provided or default .env file,
        and falling back to system environment variables and sensible defaults.
        """
        env_dict: Dict[str, str] = {}
        target_env = Path(env_path) if env_path else (Path.cwd() / ".env")
        if not target_env.is_file():
            # Also check project parent directory if called from subdirectories
            parent_env = Path(__file__).resolve().parent.parent / ".env"
            if parent_env.is_file():
                target_env = parent_env

        if target_env.is_file():
            env_dict = parse_env_file(target_env)

        # Helper to get from .env first, then os.environ, then default
        def get_val(keys: List[str], default: str) -> str:
            for k in keys:
                if k in env_dict and env_dict[k]:
                    return env_dict[k]
                if k in os.environ and os.environ[k]:
                    return os.environ[k]
            return default

        host = get_val(["SSH_HOST", "SSHFS_HOST", "HOST"], "192.168.1.180")
        user = get_val(["SSH_USER", "SSHFS_USER", "USER", "USERNAME"], "root")
        port_str = get_val(["SSH_PORT", "SSHFS_PORT", "PORT"], "22")
        try:
            port = int(port_str)
        except ValueError:
            port = 22

        password = get_val(["SSH_PASSWORD", "SSHFS_PASSWORD", "PASSWORD", "SSH_PASS"], "")
        remote_path = get_val(["SSH_REMOTE_PATH", "SSHFS_REMOTE_PATH", "REMOTE_PATH"], "/")
        mount_drive = get_val(["SSHFS_MOUNT_DRIVE", "SSH_MOUNT_DRIVE", "MOUNT_DRIVE"], "Y:")
        if not mount_drive.endswith(":"):
            mount_drive = f"{mount_drive}:"

        volname = get_val(["SSHFS_VOLNAME", "VOLNAME"], "jellyfin docker lxc")
        sshfs_exe = get_val(["SSHFS_EXE", "SSHFS_PATH"], DEFAULT_SSHFS_EXE)
        debug_str = get_val(["SSHFS_DEBUG", "DEBUG"], "true").lower()
        debug = debug_str in ("1", "true", "yes", "on")
        loglevel = get_val(["SSHFS_LOGLEVEL", "LOGLEVEL"], "debug1")
        max_readahead = get_val(["SSHFS_MAX_READAHEAD", "MAX_READAHEAD"], "1GB")
        auto_mount_str = get_val(["SSHFS_AUTO_MOUNT", "AUTO_MOUNT"], "true").lower()
        auto_mount = auto_mount_str in ("1", "true", "yes", "on")

        # Resolve sshfs executable location if default doesn't exist
        if not Path(sshfs_exe).is_file():
            alt_exe = find_binary_executable("sshfs.exe")
            if alt_exe:
                sshfs_exe = alt_exe
            elif Path(DEFAULT_SSHFS_X64_EXE).is_file():
                sshfs_exe = DEFAULT_SSHFS_X64_EXE

        return cls(
            host=host,
            user=user,
            port=port,
            password=password,
            remote_path=remote_path,
            mount_drive=mount_drive,
            volname=volname,
            sshfs_exe=sshfs_exe,
            debug=debug,
            loglevel=loglevel,
            max_readahead=max_readahead,
            auto_mount=auto_mount,
        )


def build_sshfs_args(config: SSHFSConfig) -> List[str]:
    """
    Constructs the list of arguments for sshfs.exe with all optimized parameters:
    - Target: user@host:remote_path
    - Mount drive: Y:
    - Port: -p22
    - Volume name: -ovolname=...
    - Permissions, caching, symlinks, large read, max readahead, stdin password
    """
    target = f"{config.user}@{config.host}:{config.remote_path}"
    args = [
        config.sshfs_exe,
        target,
        config.mount_drive,
        f"-p{config.port}",
        f"-ovolname={config.volname}",
    ]

    if config.debug:
        args.append("-odebug")
        if config.loglevel:
            args.append(f"-ologlevel={config.loglevel}")

    # Standard optimized SSHFS-Win parameters
    args.extend([
        "-oStrictHostKeyChecking=no",
        "-oUserKnownHostsFile=/dev/null",
        "-oidmap=user",
        "-ouid=-1",
        "-ogid=-1",
        "-oumask=000",
        "-ocreate_umask=000",
        f"-omax_readahead={config.max_readahead}",
        "-oallow_other",
        "-olarge_read",
        "-okernel_cache",
        "-ofollow_symlinks",
        "-oPreferredAuthentications=password",
        "-opassword_stdin",
    ])

    return args


def is_drive_accessible(drive_letter: str = "Y:") -> bool:
    """
    Checks whether a drive letter is mounted and readable on Windows.
    """
    clean_drive = drive_letter.strip().rstrip("\\/")
    if not clean_drive.endswith(":"):
        clean_drive = f"{clean_drive}:"
    drive_path = Path(f"{clean_drive}\\")

    if not drive_path.exists():
        return False

    try:
        # Check if we can list the directory without error
        next(drive_path.iterdir(), None)
        return True
    except (StopIteration, PermissionError):
        return True
    except Exception:
        return False


def unmount_sshfs(drive_letter: str = "Y:") -> Tuple[bool, str]:
    """
    Unmounts an existing SSHFS / network drive using standard Windows commands.
    """
    clean_drive = drive_letter.strip().rstrip("\\/")
    if not clean_drive.endswith(":"):
        clean_drive = f"{clean_drive}:"

    logger = get_logger()
    logger.info(f"Unmounting drive {clean_drive}...")

    # 1. Try 'net use Y: /delete /y'
    try:
        proc = subprocess.run(
            ["net", "use", clean_drive, "/delete", "/y"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5.0,
            check=False,
        )
        if proc.returncode == 0:
            logger.info(f"Successfully unmounted {clean_drive} via 'net use'")
            return True, f"Successfully unmounted {clean_drive}"
    except Exception as e:
        logger.debug(f"net use unmount failed: {e}")

    # 2. Try PowerShell Dismount-DiskImage / taskkill if sshfs is running
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-Command", f"(New-Object -ComObject Scripting.FileSystemObject).Drives | Where-Object {{ $_.DriveLetter -eq '{clean_drive[0]}' }}"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except Exception:
        pass

    if not is_drive_accessible(clean_drive):
        return True, f"Drive {clean_drive} is now unmounted."

    return False, f"Could not unmount {clean_drive}. It may still be in use."


def mount_sshfs(
    config: Optional[SSHFSConfig] = None,
    env_path: Optional[Path | str] = None,
    wait_timeout: float = 10.0,
) -> Tuple[bool, str, Optional[subprocess.Popen]]:
    """
    Mounts the SSHFS filesystem using the configured settings.
    Feeds the password securely through stdin.
    Verifies that the drive becomes accessible within wait_timeout seconds.
    """
    if config is None:
        config = SSHFSConfig.from_env(env_path)

    logger = get_logger()

    # Check if already accessible
    if is_drive_accessible(config.mount_drive):
        msg = f"Drive {config.mount_drive} is already mounted and accessible."
        logger.info(msg)
        return True, msg, None

    # Verify sshfs executable exists
    if not Path(config.sshfs_exe).is_file():
        err_msg = (
            f"SSHFS executable not found at '{config.sshfs_exe}'. "
            f"Please install SSHFS-Win or set SSHFS_EXE in your .env file."
        )
        logger.error(err_msg)
        return False, err_msg, None

    cmd = build_sshfs_args(config)
    logger.info(f"Executing SSHFS mount for {config.user}@{config.host}:{config.remote_path} -> {config.mount_drive}")

    try:
        # Create detached background process on Windows
        creationflags = 0
        if os.name == "nt":
            # DETACHED_PROCESS = 0x00000008, CREATE_NEW_PROCESS_GROUP = 0x00000200
            creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
        )

        # Write password to stdin if provided
        if config.password:
            try:
                proc.stdin.write(f"{config.password}\n")
                proc.stdin.flush()
            except Exception as e:
                logger.warning(f"Error sending password to sshfs stdin: {e}")

        # Wait up to wait_timeout seconds for drive to become ready
        start_wait = time.time()
        mounted = False
        while time.time() - start_wait < wait_timeout:
            if is_drive_accessible(config.mount_drive):
                mounted = True
                break
            # Check if process terminated prematurely
            poll_res = proc.poll()
            if poll_res is not None and poll_res != 0:
                stderr_out = ""
                try:
                    stderr_out = proc.stderr.read()
                except Exception:
                    pass
                err_msg = f"sshfs process exited with code {poll_res}: {stderr_out.strip()}"
                logger.error(err_msg)
                return False, err_msg, proc

            time.sleep(0.5)

        if mounted:
            success_msg = f"Drive {config.mount_drive} successfully mounted to {config.user}@{config.host}:{config.remote_path}"
            logger.info(success_msg)
            return True, success_msg, proc
        else:
            timeout_msg = (
                f"Mount command issued, but drive {config.mount_drive} was not accessible within {wait_timeout}s. "
                f"Check SSH host ({config.host}), port ({config.port}), user ({config.user}), and password in .env."
            )
            logger.warning(timeout_msg)
            return False, timeout_msg, proc

    except Exception as e:
        err_msg = f"Failed to start sshfs process: {e}"
        logger.exception(err_msg)
        return False, err_msg, None
