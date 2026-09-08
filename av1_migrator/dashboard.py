"""
Rich Terminal Dashboard for Galaxy AV1 Migrator.
Provides a comprehensive live terminal interface with GPU, storage, queue, and current encode metrics.
"""

from datetime import timedelta
from pathlib import Path
import sys
import time
from typing import List, Optional

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text

from av1_migrator.config import format_bytes
from av1_migrator.models import EncodeProgress, GPUStats, MediaFile, StorageStats
from av1_migrator.utils import format_seconds


def make_ascii_bar(percent: float, width: int = 24) -> str:
    """Creates a smooth block progress bar."""
    percent = max(0.0, min(100.0, percent))
    filled_len = int(round(width * percent / 100))
    bar = "█" * filled_len + "░" * (width - filled_len)
    return f"{bar}  {percent:5.1f}%"


class MigrationDashboard:
    """
    Renders the live Rich terminal dashboard.
    """
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.live: Optional[Live] = None
        self.is_active = False

    def start(self) -> None:
        if not self.is_active:
            self.live = Live(
                renderable=self.render_empty(),
                console=self.console,
                screen=False,
                refresh_per_second=4,
                transient=False,
            )
            self.live.start()
            self.is_active = True

    def stop(self) -> None:
        if self.is_active and self.live:
            self.live.stop()
            self.live = None
            self.is_active = False

    def update(
        self,
        gpu_stats: GPUStats,
        storage_stats: StorageStats,
        current_file: Optional[MediaFile],
        encode_progress: Optional[EncodeProgress],
        queue_files: List[MediaFile],
        completed_count: int,
        total_count: int,
        skipped_count: int,
        failed_count: int,
        source_bytes_processed: int,
        output_bytes_created: int,
        remaining_bytes: int,
        avg_processing_bps: float = 0.0,
        status_message: str = "Idle",
        is_paused: bool = False,
        is_emergency: bool = False,
        emergency_msg: str = "",
    ) -> None:
        if not self.is_active or not self.live:
            return

        layout = self.generate_layout(
            gpu_stats=gpu_stats,
            storage_stats=storage_stats,
            current_file=current_file,
            encode_progress=encode_progress,
            queue_files=queue_files,
            completed_count=completed_count,
            total_count=total_count,
            skipped_count=skipped_count,
            failed_count=failed_count,
            source_bytes_processed=source_bytes_processed,
            output_bytes_created=output_bytes_created,
            remaining_bytes=remaining_bytes,
            avg_processing_bps=avg_processing_bps,
            status_message=status_message,
            is_paused=is_paused,
            is_emergency=is_emergency,
            emergency_msg=emergency_msg,
        )
        self.live.update(layout)

    def render_empty(self) -> Panel:
        return Panel(
            Text("Starting Galaxy AV1 Migrator...", style="bold cyan", justify="center"),
            title="GALAXY AV1 MIGRATOR",
            border_style="cyan",
        )

    def generate_layout(
        self,
        gpu_stats: GPUStats,
        storage_stats: StorageStats,
        current_file: Optional[MediaFile],
        encode_progress: Optional[EncodeProgress],
        queue_files: List[MediaFile],
        completed_count: int,
        total_count: int,
        skipped_count: int,
        failed_count: int,
        source_bytes_processed: int,
        output_bytes_created: int,
        remaining_bytes: int,
        avg_processing_bps: float,
        status_message: str,
        is_paused: bool,
        is_emergency: bool,
        emergency_msg: str,
    ) -> Table:
        # Top-level grid table to hold all dashboard panels cleanly
        grid = Table.grid(expand=True, padding=(0, 0))
        grid.add_column()

        # Header Title
        title_text = Text()
        title_text.append("GALAXY AV1 MIGRATOR\n", style="bold cyan")
        title_text.append("Jellyfin Media Optimization  •  NVIDIA AV1 NVENC", style="dim white")
        if is_paused:
            title_text.append("  [PAUSED]", style="bold yellow")
        header_panel = Panel(title_text, border_style="cyan", padding=(0, 1))
        grid.add_row(header_panel)

        # Emergency Stop Banner if triggered
        if is_emergency:
            emerg_text = Text()
            emerg_text.append("!!! STORAGE SAFETY STOP !!!\n", style="bold red blink")
            emerg_text.append(f"{emergency_msg}\n", style="bold red")
            emerg_text.append("FFmpeg terminated. Temporary output removed. Original preserved. Processing stopped.", style="white")
            emerg_panel = Panel(emerg_text, border_style="bold red", padding=(1, 2))
            grid.add_row(emerg_panel)
            return grid

        # Top row: GPU Panel + Storage Panel
        top_row = Table.grid(expand=True, padding=(0, 1))
        top_row.add_column(ratio=1)
        top_row.add_column(ratio=1)

        # 1. GPU Panel
        gpu_table = Table.grid(expand=True, padding=(0, 1))
        gpu_table.add_column(ratio=1)
        gpu_table.add_column(ratio=1)

        gpu_name_str = gpu_stats.name
        temp_str = f"Temp: {int(gpu_stats.temperature_c)}°C" if gpu_stats.temperature_c is not None else "Temp: N/A"
        gpu_table.add_row(
            Text(f"GPU: {gpu_name_str}", style="bold magenta"),
            Text(temp_str, style="yellow", justify="right"),
        )

        gpu_load_bar = make_ascii_bar(gpu_stats.gpu_util or 0.0, 14) if gpu_stats.gpu_util is not None else "N/A"
        gpu_table.add_row(
            Text("GPU Load:", style="dim white"),
            Text(gpu_load_bar, style="magenta", justify="right"),
        )

        nvenc_str = make_ascii_bar(gpu_stats.enc_util, 14) if gpu_stats.enc_util is not None else "NVENC: N/A"
        gpu_table.add_row(
            Text("NVENC:", style="dim white"),
            Text(nvenc_str, style="bold magenta" if gpu_stats.enc_util is not None else "dim white", justify="right"),
        )

        if gpu_stats.vram_used_mb is not None and gpu_stats.vram_total_mb is not None:
            vram_pct = (gpu_stats.vram_used_mb / gpu_stats.vram_total_mb * 100.0) if gpu_stats.vram_total_mb > 0 else 0.0
            vram_bar = make_ascii_bar(vram_pct, 14)
            vram_str = f"{gpu_stats.vram_used_mb / 1024.0:.1f} / {gpu_stats.vram_total_mb / 1024.0:.1f} GB"
        else:
            vram_bar = "N/A"
            vram_str = "N/A"
        gpu_table.add_row(
            Text("VRAM Usage:", style="dim white"),
            Text(vram_bar, style="magenta", justify="right"),
        )
        gpu_table.add_row(
            Text("VRAM:", style="dim white"),
            Text(vram_str, style="white", justify="right"),
        )

        pwr_str = f"{int(gpu_stats.power_w)} W" if gpu_stats.power_w is not None else "N/A"
        gpu_table.add_row(
            Text("Power:", style="dim white"),
            Text(pwr_str, style="yellow", justify="right"),
        )

        gpu_panel = Panel(gpu_table, title="[bold magenta]GPU[/bold magenta]", border_style="magenta")

        # 2. Storage Panel
        storage_table = Table.grid(expand=True, padding=(0, 1))
        storage_table.add_column(ratio=1)
        storage_table.add_column(ratio=1)

        storage_table.add_row(
            Text("Free Space:", style="bold green"),
            Text(format_bytes(storage_stats.free_bytes), style="bold green", justify="right"),
        )
        storage_table.add_row(
            Text("Safety Floor:", style="dim white"),
            Text(format_bytes(storage_stats.minimum_free_bytes), style="dim yellow", justify="right"),
        )
        storage_table.add_row(
            Text("Available Buffer:", style="dim white"),
            Text(format_bytes(storage_stats.buffer_bytes), style="cyan", justify="right"),
        )

        saved_bytes = max(0, source_bytes_processed - output_bytes_created)
        reduc_pct = (saved_bytes / source_bytes_processed * 100.0) if source_bytes_processed > 0 else 0.0

        storage_table.add_row(
            Text("Space Saved:", style="bold blue"),
            Text(format_bytes(saved_bytes), style="bold blue", justify="right"),
        )
        storage_table.add_row(
            Text("Reduction:", style="dim white"),
            Text(f"{reduc_pct:.1f}%", style="bold green", justify="right"),
        )

        storage_panel = Panel(storage_table, title="[bold green]STORAGE[/bold green]", border_style="green")
        top_row.add_row(gpu_panel, storage_panel)
        grid.add_row(top_row)

        # 3. Current File / Encode Panel
        encode_table = Table.grid(expand=True, padding=(0, 0))
        encode_table.add_column()

        if current_file:
            fname = current_file.source.name
            src_codec = (current_file.video_codec or "Video").upper()
            src_w = current_file.width or 0
            src_h = current_file.height or 0
            src_res = f"{src_w}x{src_h}" if src_w else ""
            src_hdr = "HDR10" if current_file.hdr else "SDR"
            tgt_hdr = src_hdr

            # Transformation summary line
            trans_line = Text()
            trans_line.append(f"  {fname}\n\n", style="bold white")
            trans_line.append(f"  SOURCE:  {src_codec} {src_res} {src_hdr}  ({format_bytes(current_file.size)})\n", style="dim white")
            trans_line.append(f"    ↓\n", style="bold yellow")
            trans_line.append(f"  OUTPUT:  AV1 1920x1080 {tgt_hdr} (p010le, CQ28)\n\n", style="bold cyan")
            encode_table.add_row(trans_line)

            prog = encode_progress or EncodeProgress()
            bar_str = make_ascii_bar(prog.percent, 36)
            encode_table.add_row(Text(f"  {bar_str}\n", style="bold cyan"))

            metrics_table = Table.grid(expand=True, padding=(0, 2))
            metrics_table.add_column()
            metrics_table.add_column()
            metrics_table.add_column()
            metrics_table.add_column()

            cur_time_str = format_seconds(prog.out_time_s)
            tot_time_str = format_seconds(current_file.duration)
            time_metric = f"Time: {cur_time_str} / {tot_time_str}"
            speed_metric = f"Speed: {prog.speed:.2f}x"
            fps_metric = f"FPS: {prog.fps:.1f}"
            eta_metric = f"File ETA: {format_seconds(prog.eta_seconds)}"

            metrics_table.add_row(
                Text(time_metric, style="white"),
                Text(speed_metric, style="green"),
                Text(fps_metric, style="magenta"),
                Text(eta_metric, style="yellow"),
            )

            # Output size & Bitrate line
            out_sz = format_bytes(prog.total_size_bytes)
            br_str = f"{prog.bitrate_kbs:.1f} kbps" if prog.bitrate_kbs > 0 else "N/A"
            metrics_table.add_row(
                Text(f"Output: {out_sz}", style="dim white"),
                Text(f"Bitrate: {br_str}", style="dim white"),
                Text("", style="dim white"),
                Text("", style="dim white"),
            )
            encode_table.add_row(metrics_table)
        else:
            encode_table.add_row(Text("  No active encode job.", style="dim italic white"))

        cur_panel = Panel(encode_table, title="[bold cyan]CURRENT ENCODE[/bold cyan]", border_style="cyan")
        grid.add_row(cur_panel)

        # 4. Queue Panel & Library ETA
        queue_grid = Table.grid(expand=True, padding=(0, 0))
        queue_grid.add_column()

        remaining_count = max(0, total_count - completed_count - skipped_count - failed_count)
        overall_pct = (completed_count / total_count * 100.0) if total_count > 0 else 0.0
        overall_bar_str = make_ascii_bar(overall_pct, 36)

        # Overall Progress Bar line
        tot_src_bytes = source_bytes_processed + remaining_bytes
        progress_header = Text()
        progress_header.append(f"  Overall Migration Progress:  ", style="bold white")
        progress_header.append(f"{completed_count} / {total_count} files  •  {format_bytes(source_bytes_processed)} / {format_bytes(tot_src_bytes)}\n", style="dim white")
        progress_header.append(f"  {overall_bar_str}\n\n", style="bold blue")
        queue_grid.add_row(progress_header)

        queue_table = Table.grid(expand=True, padding=(0, 1))
        queue_table.add_column(ratio=1)
        queue_table.add_column(ratio=2)

        # Calculate library ETA
        if avg_processing_bps > 0 and remaining_bytes > 0:
            lib_eta_s = remaining_bytes / avg_processing_bps
            lib_eta_str = f"{format_seconds(lib_eta_s)} ({format_bytes(avg_processing_bps)}/s)"
        else:
            lib_eta_str = "Calculating..."

        q_left = Table.grid(expand=True, padding=(0, 1))
        q_left.add_column()
        q_left.add_column(justify="right")
        q_left.add_row(Text("Completed:", style="bold green"), Text(f"{completed_count} / {total_count}", style="bold green"))
        q_left.add_row(Text("Remaining:", style="white"), Text(f"{remaining_count} ({format_bytes(remaining_bytes)})", style="white"))
        q_left.add_row(Text("Skipped:", style="yellow"), Text(str(skipped_count), style="yellow"))
        q_left.add_row(Text("Failed:", style="red"), Text(str(failed_count), style="red"))
        q_left.add_row(Text("Library ETA:", style="bold yellow"), Text(lib_eta_str, style="bold yellow"))

        q_right = Table.grid(expand=True, padding=(0, 1))
        q_right.add_column()
        q_right.add_row(Text("NEXT UP (Smallest First):", style="bold dim white"))
        
        # Display top 4 next files in queue
        if queue_files:
            for idx, qf in enumerate(queue_files[:4], 1):
                sz_str = format_bytes(qf.size)
                display_name = qf.source.name
                if len(display_name) > 42:
                    display_name = display_name[:39] + "..."
                q_right.add_row(Text(f" {idx}. {sz_str:>9}   {display_name}", style="dim white"))
        else:
            q_right.add_row(Text(" (Queue empty)", style="dim italic"))

        queue_table.add_row(q_left, q_right)
        queue_grid.add_row(queue_table)

        queue_panel = Panel(queue_grid, title="[bold blue]OVERALL PROGRESS & QUEUE[/bold blue]", border_style="blue")
        grid.add_row(queue_panel)

        # 5. Status & Footer Panel
        status_table = Table.grid(expand=True, padding=(0, 1))
        status_table.add_column(ratio=2)
        status_table.add_column(ratio=1, justify="right")

        status_text = Text(f"Status: {status_message}", style="bold yellow" if is_paused else "dim white")
        controls_text = Text("[P] Pause   [R] Resume   [S] Stop   [X] Emergency Stop   [Q] Quit", style="bold cyan")
        status_table.add_row(status_text, controls_text)

        status_panel = Panel(status_table, border_style="dim white", padding=(0, 1))
        grid.add_row(status_panel)

        return grid
