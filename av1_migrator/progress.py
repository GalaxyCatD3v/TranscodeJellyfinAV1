"""
Progress bar utilities and factories powered by `tqdm`.
Provides responsive, customizable terminal progress bars for media scanning, probing,
individual file encoding, and overall library migration.
"""

import sys
import threading
from typing import Any, Optional
from tqdm import tqdm


class MigrationProgressBar:
    """
    Thread-safe controller wrapper around `tqdm.tqdm`.
    Simplifies lifecycle management (start, absolute updates, kwargs postfix, finish/close).
    """

    def __init__(self, pbar: tqdm, position: int = 0):
        self.pbar = pbar
        self.position = position
        self._lock = threading.Lock()
        self._last_n: float = float(pbar.n) if pbar.n is not None else 0.0
        self._started = False
        self._finished = False

    @property
    def total(self) -> Optional[float]:
        return self.pbar.total

    @total.setter
    def total(self, value: Optional[float]) -> None:
        with self._lock:
            self.pbar.total = value

    def start(self, total: Optional[int | float] = None) -> "MigrationProgressBar":
        with self._lock:
            if not self._started:
                if total is not None:
                    self.pbar.total = total
                self._started = True
                self.pbar.refresh()
        return self

    def update(self, value: int | float = 1, postfix_str: Optional[str] = None, **kwargs: Any) -> None:
        """
        Updates progress bar with an absolute progress position `value`.
        Pass either a clean `postfix_str` or keyword arguments.
        """
        with self._lock:
            if not self._started:
                self._started = True
            if not self._finished:
                try:
                    delta = float(value) - self._last_n
                    if delta > 0:
                        self.pbar.update(delta)
                        self._last_n = float(value)
                    elif delta < 0:
                        self.pbar.n = float(value)
                        self._last_n = float(value)
                    
                    if postfix_str is not None:
                        self.pbar.set_postfix_str(postfix_str, refresh=False)
                    elif kwargs:
                        self.pbar.set_postfix(**kwargs, refresh=False)
                    self.pbar.refresh()
                except Exception:
                    pass

    def set_description(self, desc: str) -> None:
        with self._lock:
            try:
                self.pbar.set_description(desc, refresh=True)
            except Exception:
                pass

    def set_postfix(self, **kwargs: Any) -> None:
        with self._lock:
            try:
                self.pbar.set_postfix(**kwargs, refresh=True)
            except Exception:
                pass

    def set_postfix_str(self, s: str) -> None:
        with self._lock:
            try:
                self.pbar.set_postfix_str(s, refresh=True)
            except Exception:
                pass

    def reset(self, total: Optional[float] = 100.0, desc: Optional[str] = None) -> None:
        """
        Resets the progress bar counter for a new task on the same line/position.
        """
        with self._lock:
            try:
                self.pbar.reset(total=total)
                self._last_n = 0.0
                self._started = True
                self._finished = False
                if desc is not None:
                    self.pbar.set_description(desc, refresh=False)
                self.pbar.set_postfix_str("", refresh=False)
                self.pbar.refresh()
            except Exception:
                pass

    @staticmethod
    def write(msg: str) -> None:
        """Outputs a clean message via tqdm.write without breaking active progress bars."""
        try:
            tqdm.write(msg)
        except Exception:
            print(msg)

    def finish(self) -> None:
        with self._lock:
            if not self._finished:
                try:
                    self.pbar.refresh()
                    self.pbar.close()
                except Exception:
                    pass
                self._finished = True

    def close(self) -> None:
        self.finish()

    def __enter__(self) -> "MigrationProgressBar":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.finish()


def create_scan_progressbar(
    message: str = "Scanning Media Roots",
    fd: Any = sys.stdout,
) -> MigrationProgressBar:
    """
    Creates an indeterminate / dynamic progress bar for recursive directory scanning.
    Displays directories scanned, files found, and active directories/files.
    """
    pbar = tqdm(
        total=None,
        desc=message,
        unit="dirs",
        file=fd,
        dynamic_ncols=True,
        mininterval=0.05,
        bar_format="{desc}: {n_fmt} dirs [{elapsed}, {rate_fmt}] {postfix}",
        leave=True,
    )
    return MigrationProgressBar(pbar)


def create_probe_progressbar(
    max_value: int,
    message: str = "Probing Media Files",
    fd: Any = sys.stdout,
) -> MigrationProgressBar:
    """
    Creates a determinate progress bar for candidate media inspection / FFprobe.
    Displays percentage, visual bar, current / total count, rate, and current file.
    """
    max_val = max(1, max_value)
    pbar = tqdm(
        total=max_val,
        desc=message,
        unit="file",
        file=fd,
        dynamic_ncols=True,
        mininterval=0.05,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        leave=True,
    )
    return MigrationProgressBar(pbar)


def create_encode_progressbar(
    message: str = "Encoding AV1",
    position: int = 0,
    fd: Any = sys.stdout,
) -> MigrationProgressBar:
    """
    Creates a progress bar for single-file FFmpeg encoding (0.0 to 100.0 percent).
    Displays percentage, visual bar, FPS, Speed multiplier, ETA, output size, and GPU load.
    """
    pbar = tqdm(
        total=100.0,
        desc=message,
        unit="%",
        file=fd,
        dynamic_ncols=True,
        mininterval=0.05,
        position=position,
        bar_format="{desc}: {percentage:5.1f}%|{bar}| [{elapsed}<{remaining}] {postfix}",
        leave=True,
    )
    return MigrationProgressBar(pbar, position=position)


def create_worker_progressbar(
    worker_name: str,
    position: int,
    message: Optional[str] = None,
    fd: Any = sys.stdout,
) -> MigrationProgressBar:
    """
    Creates a dedicated worker progress bar (position 1 for CPU, position 2 for GPU).
    Displays percentage, visual bar, elapsed, remaining, throughput, and hardware telemetry.
    """
    desc = message or f"{worker_name} [Idle]"
    pbar = tqdm(
        total=100.0,
        desc=desc,
        unit="%",
        file=fd,
        dynamic_ncols=True,
        mininterval=0.05,
        position=position,
        bar_format="{desc}: {percentage:5.1f}%|{bar}| [{elapsed}<{remaining}] {postfix}",
        leave=True,
    )
    return MigrationProgressBar(pbar, position=position)


def create_overall_progressbar(
    max_value: int,
    message: str = "Overall Migration",
    position: int = 0,
    fd: Any = sys.stdout,
) -> MigrationProgressBar:
    """
    Creates a progress bar for total library migration progress (position 0).
    Displays percentage, visual bar, completed / total count, and storage saved.
    """
    max_val = max(1, max_value)
    pbar = tqdm(
        total=max_val,
        desc=message,
        unit="file",
        file=fd,
        dynamic_ncols=True,
        mininterval=0.05,
        position=position,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        leave=True,
    )
    return MigrationProgressBar(pbar, position=position)
