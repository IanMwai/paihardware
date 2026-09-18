from __future__ import annotations

import csv
import datetime as dt
import queue
import threading
from pathlib import Path

import numpy as np

from .processing import ProcessedBlock
from .utils import fmt_hhmmss_ms, rename_with_fallback


def csv_header(labels: list[str]) -> list[str]:
    """Per-GPU V/I/P columns plus the summed power, e.g. gpu1_voltage_v, ...

    total_power_w stays a real column (not derived) so run stats and replay
    read one canonical series regardless of GPU count.
    """
    header = ["time_s"]
    for label in labels:
        gpu = label.lower()
        header.extend([f"{gpu}_voltage_v", f"{gpu}_current_a", f"{gpu}_power_w"])
    header.append("total_power_w")
    return header


class ChunkWriter:
    def __init__(
        self,
        out_dir: Path,
        prefix: str,
        chunk_len_sec: float,
        start_wall_dt: dt.datetime,
        labels: list[str],
        queue_blocks: int = 200,
        file_format: str = "csv",
    ):
        if file_format != "csv":
            raise NotImplementedError("Only CSV writing is implemented")
        self.out_dir = Path(out_dir)
        self.prefix = prefix
        self.labels = list(labels)
        self.header = csv_header(self.labels)
        self.chunk_len = float(chunk_len_sec)
        self.start_wall = start_wall_dt
        self.queue: queue.Queue[ProcessedBlock] = queue.Queue(maxsize=int(queue_blocks))
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.chunk_start_sec = 0.0
        self.chunk_start_wall: dt.datetime | None = None
        self.tmp_path: Path | None = None
        self._file = None
        self.writer = None
        self.closed_paths: list[Path] = []
        self._last_written_t_end = 0.0
        self.thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.thread.start()

    def write_block(self, block: ProcessedBlock) -> None:
        if block.time_s.size == 0:
            return
        try:
            self.queue.put(block, block=False)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except Exception:
                pass
            try:
                self.queue.put(block, block=False)
            except Exception:
                print("[WARNING] Writer queue full; dropping data block")

    def finalize(self, t_end_sec: float) -> list[Path]:
        self.stop_event.set()
        self.queue.join()
        self.thread.join(timeout=10.0)
        with self.lock:
            if self._file is not None:
                self._close_and_rename(float(t_end_sec))
        return list(self.closed_paths)

    def _writer_loop(self) -> None:
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                block = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._write_block_locked(block)
            finally:
                self.queue.task_done()
        with self.lock:
            if self._file is not None:
                self._close_and_rename(self._last_written_t_end)

    def _write_block_locked(self, block: ProcessedBlock) -> None:
        sizes = [block.time_s.size, block.total_power_w.size]
        sizes += [arr.size for arr in block.voltage_v]
        sizes += [arr.size for arr in block.current_a]
        sizes += [arr.size for arr in block.power_w]
        n = min(sizes) if sizes else 0
        if n == 0:
            return
        with self.lock:
            idx = 0
            while idx < n:
                t_now = float(block.time_s[idx])
                self._ensure_chunk_for_time(t_now)
                chunk_end = self.chunk_start_sec + self.chunk_len
                mask_end = idx
                while mask_end < n and float(block.time_s[mask_end]) < chunk_end:
                    mask_end += 1
                if mask_end == idx:
                    self._close_and_rename(chunk_end)
                    continue
                def make_row(k: int) -> list[str]:
                    row = [f"{float(block.time_s[k]):.9f}"]
                    for g in range(len(block.voltage_v)):
                        row.extend(
                            [
                                f"{float(block.voltage_v[g][k]):.9f}",
                                f"{float(block.current_a[g][k]):.9f}",
                                f"{float(block.power_w[g][k]):.9f}",
                            ]
                        )
                    row.append(f"{float(block.total_power_w[k]):.9f}")
                    return row

                rows = (make_row(k) for k in range(idx, mask_end))
                assert self.writer is not None
                self.writer.writerows(rows)
                self._last_written_t_end = max(
                    self._last_written_t_end, float(block.time_s[mask_end - 1])
                )
                idx = mask_end
                if idx < n and float(block.time_s[idx]) >= chunk_end:
                    self._close_and_rename(chunk_end)

    def _ensure_chunk_for_time(self, t_now_sec: float) -> None:
        if self._file is None:
            start = np.floor(float(t_now_sec) / self.chunk_len) * self.chunk_len
            self._open_new(float(start))
        while t_now_sec >= self.chunk_start_sec + self.chunk_len:
            end_sec = self.chunk_start_sec + self.chunk_len
            self._close_and_rename(end_sec)
            self._open_new(end_sec)

    def _open_new(self, chunk_start_sec: float) -> None:
        self.chunk_start_sec = float(chunk_start_sec)
        self.chunk_start_wall = self.start_wall + dt.timedelta(seconds=self.chunk_start_sec)
        date_str = self.chunk_start_wall.strftime("%Y%m%d")
        start_str = fmt_hhmmss_ms(self.chunk_start_wall)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_path = self.out_dir / f"{self.prefix}_{date_str}_{start_str}_to_PENDING.csv"
        self._file = open(self.tmp_path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self._file)
        self.writer.writerow(self.header)

    def _close_and_rename(self, chunk_end_sec: float) -> None:
        if self._file is None:
            return
        self._file.flush()
        self._file.close()
        self._file = None
        self.writer = None
        assert self.chunk_start_wall is not None
        assert self.tmp_path is not None
        chunk_end_wall = self.start_wall + dt.timedelta(seconds=float(chunk_end_sec))
        date_str = self.chunk_start_wall.strftime("%Y%m%d")
        start_str = fmt_hhmmss_ms(self.chunk_start_wall)
        end_str = fmt_hhmmss_ms(chunk_end_wall)
        final_path = self.out_dir / f"{self.prefix}_{date_str}_{start_str}_to_{end_str}.csv"
        rename_with_fallback(self.tmp_path, final_path)
        self.closed_paths.append(final_path)
        self.tmp_path = None
        self.chunk_start_wall = None
