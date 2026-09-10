"""
Unit and integration tests for SSHFS mounting, .env parsing, argument generation,
and drive management in Galaxy AV1 Migrator.
"""

from pathlib import Path
import subprocess
from unittest.mock import MagicMock, patch
import pytest

from av1_migrator.sshfs import (
    DEFAULT_SSHFS_EXE,
    SSHFSConfig,
    build_sshfs_args,
    is_drive_accessible,
    mount_sshfs,
    parse_env_file,
    unmount_sshfs,
)


def test_parse_env_file(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
        # Test SSHFS Configuration
        SSH_HOST=192.168.1.180
        SSH_USER=root
        SSH_PORT=22
        SSH_PASSWORD="supersecretpassword!"
        SSH_REMOTE_PATH=/srv/media
        SSHFS_MOUNT_DRIVE=Y:
        SSHFS_VOLNAME='jellyfin docker lxc'
        export SSHFS_MAX_READAHEAD=1GB
        SSHFS_DEBUG=true
        SSHFS_LOGLEVEL=debug1
        SSHFS_AUTO_MOUNT=true
        """,
        encoding="utf-8",
    )

    data = parse_env_file(env_file)
    assert data["SSH_HOST"] == "192.168.1.180"
    assert data["SSH_USER"] == "root"
    assert data["SSH_PORT"] == "22"
    assert data["SSH_PASSWORD"] == "supersecretpassword!"
    assert data["SSH_REMOTE_PATH"] == "/srv/media"
    assert data["SSHFS_MOUNT_DRIVE"] == "Y:"
    assert data["SSHFS_VOLNAME"] == "jellyfin docker lxc"
    assert data["SSHFS_MAX_READAHEAD"] == "1GB"
    assert data["SSHFS_DEBUG"] == "true"
    assert data["SSHFS_LOGLEVEL"] == "debug1"
    assert data["SSHFS_AUTO_MOUNT"] == "true"


def test_sshfs_config_from_env(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
        SSH_HOST=10.0.0.50
        SSH_USER=admin
        SSH_PORT=2222
        SSH_PASSWORD=mypassword
        SSHFS_MOUNT_DRIVE=Z
        SSHFS_VOLNAME=testvol
        """,
        encoding="utf-8",
    )

    cfg = SSHFSConfig.from_env(env_file)
    assert cfg.host == "10.0.0.50"
    assert cfg.user == "admin"
    assert cfg.port == 2222
    assert cfg.password == "mypassword"
    assert cfg.mount_drive == "Z:"
    assert cfg.volname == "testvol"
    assert cfg.debug is True
    assert cfg.auto_mount is True


def test_build_sshfs_args():
    cfg = SSHFSConfig(
        host="192.168.1.180",
        user="root",
        port=22,
        password="secretpassword",
        remote_path="/",
        mount_drive="Y:",
        volname="jellyfin docker lxc",
        sshfs_exe=r"C:\Program Files\SSHFS-Win\bin\sshfs.exe",
        debug=True,
        loglevel="debug1",
        max_readahead="1GB",
    )

    args = build_sshfs_args(cfg)

    assert args[0] == r"C:\Program Files\SSHFS-Win\bin\sshfs.exe"
    assert args[1] == "root@192.168.1.180:/"
    assert args[2] == "Y:"
    assert "-p22" in args
    assert "-ovolname=jellyfin docker lxc" in args
    assert "-odebug" in args
    assert "-ologlevel=debug1" in args
    assert "-oStrictHostKeyChecking=no" in args
    assert "-oUserKnownHostsFile=/dev/null" in args
    assert "-oidmap=user" in args
    assert "-ouid=-1" in args
    assert "-ogid=-1" in args
    assert "-oumask=000" in args
    assert "-ocreate_umask=000" in args
    assert "-omax_readahead=1GB" in args
    assert "-oallow_other" in args
    assert "-olarge_read" in args
    assert "-okernel_cache" in args
    assert "-ofollow_symlinks" in args
    assert "-oPreferredAuthentications=password" in args
    assert "-opassword_stdin" in args


def test_is_drive_accessible_nonexistent():
    # Random letter unlikely to exist
    assert is_drive_accessible("X:") in (True, False)


@patch("av1_migrator.sshfs.is_drive_accessible")
@patch("pathlib.Path.is_file")
@patch("subprocess.Popen")
def test_mount_sshfs_success(mock_popen, mock_is_file, mock_accessible):
    mock_accessible.side_effect = [False, True]  # Not accessible at first, accessible after mount
    mock_is_file.return_value = True

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.stdin = MagicMock()
    mock_popen.return_value = mock_proc

    cfg = SSHFSConfig(password="testpass")
    ok, msg, proc = mount_sshfs(cfg, wait_timeout=2.0)

    assert ok is True
    assert "successfully mounted" in msg
    mock_proc.stdin.write.assert_called_with("testpass\n")
    mock_proc.stdin.flush.assert_called_once()


@patch("av1_migrator.sshfs.is_drive_accessible")
def test_mount_sshfs_already_mounted(mock_accessible):
    mock_accessible.return_value = True
    cfg = SSHFSConfig()
    ok, msg, proc = mount_sshfs(cfg)
    assert ok is True
    assert "already mounted" in msg
    assert proc is None


@patch("subprocess.run")
@patch("av1_migrator.sshfs.is_drive_accessible")
def test_unmount_sshfs(mock_accessible, mock_run):
    mock_run.return_value = MagicMock(returncode=0)
    mock_accessible.return_value = False

    ok, msg = unmount_sshfs("Y:")
    assert ok is True
    assert "Successfully unmounted" in msg
    mock_run.assert_called_once()
