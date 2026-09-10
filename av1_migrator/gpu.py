"""
GPU monitoring module for Galaxy AV1 Migrator using nvidia-smi.
"""

import shutil
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional
from av1_migrator.logger import get_logger
from av1_migrator.models import GPUStats


def is_nvidia_smi_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def get_available_gpus() -> List[Dict[str, Any]]:
    """
    Returns a list of detected NVIDIA GPUs with index and name.
    e.g. [{'index': 0, 'name': 'NVIDIA GeForce RTX 4070 Ti'}]
    """
    if not is_nvidia_smi_available():
        return []

    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3.0,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return []

        gpus: List[Dict[str, Any]] = []
        for line in proc.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",", 1)]
            if len(parts) >= 2:
                try:
                    idx = int(parts[0])
                    name = parts[1]
                    gpus.append({"index": idx, "name": name})
                except ValueError:
                    continue
            elif len(parts) == 1 and parts[0]:
                gpus.append({"index": len(gpus), "name": parts[0]})
        return gpus
    except Exception as e:
        get_logger().debug(f"Failed to query available GPUs: {e}")
        return []


def fetch_gpu_stats() -> GPUStats:
    """
    Runs nvidia-smi and extracts GPU, NVENC, VRAM, temp, and power metrics.
    """
    if not is_nvidia_smi_available():
        return GPUStats(name="NVIDIA GPU (nvidia-smi not found)", available=False)

    try:
        # Avoid shell=True
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,utilization.encoder,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3.0,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return GPUStats(name="NVIDIA GPU (query failed)", available=False)

        line = proc.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]

        # name, utilization.gpu, utilization.encoder, utilization.memory, memory.used, memory.total, temperature.gpu, power.draw
        name = parts[0] if len(parts) > 0 else "NVIDIA GPU"

        def parse_float(val: str) -> Optional[float]:
            try:
                if val.upper() in ("N/A", "[N/A]", "NOT SUPPORTED", "[NOT SUPPORTED]"):
                    return None
                return float(val)
            except (ValueError, IndexError):
                return None

        gpu_util = parse_float(parts[1]) if len(parts) > 1 else None
        enc_util = parse_float(parts[2]) if len(parts) > 2 else None
        mem_util = parse_float(parts[3]) if len(parts) > 3 else None
        vram_used = parse_float(parts[4]) if len(parts) > 4 else None
        vram_total = parse_float(parts[5]) if len(parts) > 5 else None
        temp = parse_float(parts[6]) if len(parts) > 6 else None
        power = parse_float(parts[7]) if len(parts) > 7 else None

        return GPUStats(
            name=name,
            gpu_util=gpu_util,
            enc_util=enc_util,
            mem_util=mem_util,
            vram_used_mb=vram_used,
            vram_total_mb=vram_total,
            temperature_c=temp,
            power_w=power,
            available=True,
        )
    except Exception as e:
        get_logger().debug(f"Failed to query nvidia-smi: {e}")
        return GPUStats(name="NVIDIA GPU (Error)", available=False)


class GPUMonitorThread:
    """
    Background thread to periodically poll GPU stats.
    """
    def __init__(self, poll_interval: float = 1.0):
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.stats = GPUStats()

    def start(self) -> None:
        self._stop_event.clear()
        self.stats = fetch_gpu_stats()
        self._thread = threading.Thread(target=self._run, name="GPUMonitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.stats = fetch_gpu_stats()
            time.sleep(self.poll_interval)
