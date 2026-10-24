from __future__ import annotations

import csv
import math
from collections import deque
import datetime as dt
import queue
import threading
import time
from pathlib import Path

import numpy as np

from .processing import ProcessedBlock
from .utils import fmt_hhmmss_ms, rename_with_fallback
from .chunks import begin_publication, publish, digest, validate_pair, logical_name, ChunkIntegrityError


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


class SampleQueue(queue.Queue):
    """Track sample volume as well as block count under Queue's own mutex."""
    def __init__(self, maxsize):
        super().__init__(maxsize=maxsize)
        self.samples = self.max_samples = self.max_depth = 0
        self.arrivals = deque()

    def _put(self, block):
        super()._put(block)
        self.samples += int(block.time_s.size)
        self.arrivals.append(time.monotonic())
        self.max_samples = max(self.max_samples, self.samples)
        self.max_depth = max(self.max_depth, len(self.queue))

    def _get(self):
        block = super()._get()
        self.samples -= int(block.time_s.size)
        self.arrivals.popleft()
        return block

    def snapshot(self):
        with self.mutex:
            return {"queue_depth": len(self.queue), "queued_samples": self.samples,
                    "max_queue_depth": self.max_depth, "max_queued_samples": self.max_samples,
                    "oldest_queued_age_sec": max(0., time.monotonic()-self.arrivals[0]) if self.arrivals else 0.}


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
        compress: bool = False,
        sample_rate_hz: float = 10000.,
        enqueue_timeout_sec: float = 5.,
        compression_queue_chunks: int = 2,
    ):
        if file_format != "csv":
            raise NotImplementedError("Only CSV writing is implemented")
        if queue_blocks < 1 or compression_queue_chunks < 1:
            raise ValueError("Writer and compression queue capacities must be positive")
        if not math.isfinite(enqueue_timeout_sec) or enqueue_timeout_sec < 0:
            raise ValueError("enqueue_timeout_sec must be finite and nonnegative")
        if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be finite and positive")
        self.sample_rate_hz = float(sample_rate_hz)
        self.enqueue_timeout_sec = float(enqueue_timeout_sec)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        begin_publication(self.out_dir)
        self.prefix = prefix
        self.labels = list(labels)
        self.header = csv_header(self.labels)
        self.chunk_len = float(chunk_len_sec)
        self.start_wall = start_wall_dt
        self.queue = SampleQueue(maxsize=int(queue_blocks))
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.chunk_start_sec = 0.0
        self.chunk_start_wall: dt.datetime | None = None
        self.tmp_path: Path | None = None
        self._file = None
        self.writer = None
        self.closed_paths: list[Path] = []
        self.compress = bool(compress)
        self._compression_queue = queue.Queue(maxsize=compression_queue_chunks)
        self._writer_done = threading.Event()
        self._metrics_lock = threading.Lock()
        self._recent_writes = deque(maxlen=256)
        self.samples_enqueued = 0
        self.pending_block_samples = 0
        self.inflight_samples = 0
        self.queue_full_events = 0
        self.enqueue_wait_total_sec = self.enqueue_wait_max_sec = 0.
        self.max_write_sec = self.last_write_sec = 0.
        self.last_write_cpu_sec = self.max_write_cpu_sec = 0.
        self.compression_active = False
        self.compression_started = None
        self.compression_last_sec = self.compression_max_sec = 0.
        self.compression_completed = self.compression_deferred = 0
        self._compressor = None
        self._last_written_t_end = 0.0
        self.error: BaseException | None = None
        self.samples_written = 0
        self.power_sum = self.energy = 0.0
        self.finite_samples = 0
        self.power_peak = None
        self.previous_sample = None
        self.last_progress = time.monotonic()
        if self.compress:
            self._compressor = threading.Thread(target=self._compression_loop, daemon=True)
            self._compressor.start()
        self.thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.thread.start()

    def write_block(self, block: ProcessedBlock, *, on_wait=None, timeout_sec=None) -> None:
        if block.time_s.size == 0:
            return
        self.check_health()
        if self.stop_event.is_set():
            raise RuntimeError("Cannot enqueue after writer finalization")
        timeout = self.enqueue_timeout_sec if timeout_sec is None else min(self.enqueue_timeout_sec, timeout_sec)
        if timeout < 0 or not math.isfinite(timeout):
            raise ValueError("Enqueue timeout must be finite and nonnegative")
        started = time.monotonic()
        deadline = started + timeout
        waited = False
        self.pending_block_samples = int(block.time_s.size)
        try:
            try:
                self.queue.put_nowait(block)
            except queue.Full:
                waited = True
                self.queue_full_events += 1
                while True:
                    self.check_health()
                    if on_wait is not None:
                        on_wait()
                    remaining = deadline-time.monotonic()
                    if remaining <= 0:
                        snapshot = self.diagnostics()
                        raise RuntimeError(
                            f"Writer backlog did not recover within {timeout:g}s; "
                            f"{snapshot['queued_samples']} samples queued "
                            f"({snapshot['queued_seconds']:.3f}s); acquisition stopped to avoid silent data loss"
                        )
                    try:
                        self.queue.put(block, timeout=min(.05, remaining))
                        break
                    except queue.Full:
                        continue
            self.samples_enqueued += int(block.time_s.size)
        finally:
            self.pending_block_samples = 0
            if waited:
                elapsed = time.monotonic()-started
                self.enqueue_wait_total_sec += elapsed
                self.enqueue_wait_max_sec = max(self.enqueue_wait_max_sec, elapsed)

    def diagnostics(self) -> dict:
        snapshot = self.queue.snapshot()
        with self._metrics_lock:
            recent = list(self._recent_writes)
            written = self.samples_written
            snapshot.update(
                sample_rate_hz=self.sample_rate_hz, samples_enqueued=self.samples_enqueued,
                written_samples=written, inflight_samples=self.inflight_samples,
                pending_block_samples=self.pending_block_samples,
                queue_full_events=self.queue_full_events,
                enqueue_wait_total_sec=self.enqueue_wait_total_sec, enqueue_wait_max_sec=self.enqueue_wait_max_sec,
                last_write_sec=self.last_write_sec, max_write_sec=self.max_write_sec,
                last_write_cpu_sec=self.last_write_cpu_sec, max_write_cpu_sec=self.max_write_cpu_sec,
                writer_idle_or_stalled_sec=max(0., time.monotonic()-self.last_progress),
                compression_active=self.compression_active,
                compression_pending_chunks=self._compression_queue.qsize(),
                compression_active_sec=(time.monotonic()-self.compression_started) if self.compression_active else 0.,
                compression_last_sec=self.compression_last_sec, compression_max_sec=self.compression_max_sec,
                compression_completed=self.compression_completed, compression_deferred=self.compression_deferred)
        snapshot['queued_seconds'] = snapshot['queued_samples']/self.sample_rate_hz
        snapshot['unwritten_seconds'] = max(0, self.samples_enqueued-written)/self.sample_rate_hz
        snapshot['writer_service_samples_per_sec'] = (sum(n for n, elapsed in recent)/sum(elapsed for n, elapsed in recent)
                                                     if recent and sum(elapsed for n, elapsed in recent) > 0 else None)
        return snapshot

    def check_health(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"CSV writer failed: {self.error}") from self.error
        if not self.thread.is_alive() and not self.stop_event.is_set():
            raise RuntimeError("CSV writer exited unexpectedly")

    def finalize(self, t_end_sec: float, timeout_sec: float = 30.0) -> list[Path]:
        self.stop_event.set()
        deadline = time.monotonic() + timeout_sec
        # queue.join() can never finish after a writer exception or blocked I/O.
        while self.thread.is_alive() and time.monotonic() < deadline:
            self.thread.join(timeout=min(.1, max(0, deadline-time.monotonic())))
        if self.thread.is_alive():
            raise TimeoutError("CSV writer did not drain before shutdown deadline")
        if self._compressor is not None:
            self._compressor.join(timeout=max(0, deadline-time.monotonic()))
            if self._compressor.is_alive():
                raise TimeoutError("Compression did not finish before shutdown deadline")
        if self.error is not None:
            raise RuntimeError(f"CSV writer failed: {self.error}") from self.error
        return sorted(self.closed_paths, key=logical_name)

    def statistics(self) -> dict:
        return {"sample_count": self.finite_samples,
                "average_power_w": self.power_sum/self.finite_samples if self.finite_samples else None,
                "peak_power_w": self.power_peak,
                "energy_j": self.energy if self.finite_samples else None}

    def _writer_loop(self) -> None:
        try:
            self._drain_writer()
        finally:
            self._writer_done.set()
            if self._file is not None:
                try:
                    self._file.close()
                except OSError as exc:
                    self.error = self.error or exc

    def _drain_writer(self) -> None:
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                block = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            started = time.monotonic()
            cpu_started = time.thread_time()
            self.inflight_samples = int(block.time_s.size)
            try:
                self._write_block_locked(block)
                valid = np.isfinite(block.total_power_w) & np.isfinite(block.time_s)
                values, times = block.total_power_w[valid], block.time_s[valid]
                self.finite_samples += int(times.size)
                if self.previous_sample is not None and times.size:
                    old_t, old_p = self.previous_sample
                    self.energy += (float(times[0])-old_t)*(float(values[0])+old_p)/2
                if times.size > 1:
                    self.energy += float(np.sum(np.diff(times)*(values[1:]+values[:-1])/2))
                if times.size:
                    self.previous_sample = (float(times[-1]), float(values[-1]))
                    peak = float(np.max(values))
                    self.power_peak = peak if self.power_peak is None else max(self.power_peak, peak)
                    self.power_sum += float(np.sum(values))
                finished = time.monotonic()
                with self._metrics_lock:
                    self.samples_written += int(block.time_s.size)
                    self.last_progress = finished
                    self.last_write_sec = finished-started
                    self.last_write_cpu_sec = time.thread_time()-cpu_started
                    self.max_write_cpu_sec = max(self.max_write_cpu_sec, self.last_write_cpu_sec)
                    self.max_write_sec = max(self.max_write_sec, self.last_write_sec)
                    self._recent_writes.append((int(block.time_s.size), self.last_write_sec))
            except BaseException as exc:
                self.error = self.error or exc
                return
            finally:
                self.inflight_samples = 0
                self.queue.task_done()
        try:
            with self.lock:
                if self._file is not None:
                    self._close_and_rename(self._last_written_t_end)
        except BaseException as exc:
            self.error = exc

    def _write_block_locked(self, block: ProcessedBlock) -> None:
        sizes = [block.time_s.size, block.total_power_w.size]
        sizes += [arr.size for arr in block.voltage_v]
        sizes += [arr.size for arr in block.current_a]
        sizes += [arr.size for arr in block.power_w]
        if len(set(sizes)) != 1:
            raise ValueError("Mismatched sample arrays; refusing to truncate data")
        n = min(sizes) if sizes else 0
        if n == 0:
            return
        if np.any(np.diff(block.time_s) < 0):
            raise ValueError("Sample timestamps must be ordered")
        with self.lock:
            idx = 0
            while idx < n:
                t_now = float(block.time_s[idx])
                self._ensure_chunk_for_time(t_now)
                chunk_end = self.chunk_start_sec + self.chunk_len
                mask_end = int(np.searchsorted(block.time_s, chunk_end, side="left"))
                if mask_end <= idx:
                    self._close_and_rename(chunk_end)
                    continue
                columns = [block.time_s[idx:mask_end]]
                for g in range(len(block.voltage_v)):
                    columns.extend((block.voltage_v[g][idx:mask_end], block.current_a[g][idx:mask_end],
                                    block.power_w[g][idx:mask_end]))
                columns.append(block.total_power_w[idx:mask_end])
                # Bulk numeric formatting avoids per-field Python callbacks.
                # Keep the existing nine decimals, column order and CSV newlines.
                np.savetxt(self._file, np.column_stack(columns), fmt="%.9f", delimiter=",", newline="\r\n")
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
        if self.compress:
            try:
                self._compression_queue.put_nowait(final_path)
            except queue.Full:
                # Retaining an intact CSV is preferable to spawning unlimited
                # compression threads or holding the acquisition writer up.
                notice = "compression_deferred; compression queue full, original CSV published"
                publish(final_path, [notice], source_digest=digest(final_path))
                self.closed_paths.append(final_path)
                self.compression_deferred += 1
                print(f"[WARN] {final_path.name}: {notice}", flush=True)
        else:
            publish(final_path, [], source_digest=digest(final_path))
            self.closed_paths.append(final_path)
        self.tmp_path = None
        self.chunk_start_wall = None

    def _compression_loop(self):
        while not self._writer_done.is_set() or not self._compression_queue.empty():
            try:
                path = self._compression_queue.get(timeout=.1)
            except queue.Empty:
                continue
            started = time.monotonic()
            with self._metrics_lock:
                self.compression_active = True
                self.compression_started = started
            try:
                zipped = path.with_name(path.name + ".gz")
                canonical, notices, source_digest = _gzip_chunk(path, zipped)
                publish(canonical, notices, source_digest=source_digest)
                self.closed_paths.append(canonical)
                for notice in notices:
                    print(f"[WARN] {path.name}: {notice}", flush=True)
            except BaseException as exc:
                self.error = self.error or exc
            finally:
                with self._metrics_lock:
                    self.compression_active = False
                    self.compression_last_sec = time.monotonic()-started
                    self.compression_max_sec = max(self.compression_max_sec, self.compression_last_sec)
                    self.compression_completed += 1
                self._compression_queue.task_done()



def _gzip_chunk(src: Path, dst: Path) -> tuple[Path, list[str], str]:
    """Settle representation before publishing; cleanup is never a write failure."""
    import gzip
    import shutil
    notices = []
    source_digest = digest(src)
    tmp = dst.with_name(dst.name + ".tmp")
    try:
        with open(src, "rb") as fin, gzip.open(tmp, "wb", compresslevel=4) as fout:
            shutil.copyfileobj(fin, fout, 1 << 20)
        if not validate_pair(src, tmp):
            raise OSError("Compressed file failed gzip validation")
        rename_with_fallback(tmp, dst)
        # A rename fallback may have copied bytes. Verify the installed file too.
        if not validate_pair(src, dst):
            raise OSError("Installed gzip failed validation")
    except (OSError, EOFError) as exc:
        if digest(src) != source_digest:
            raise ChunkIntegrityError(f"Original CSV changed during compression: {src.name}") from exc
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        notices.append(f"compression_failed; original CSV retained: {exc}")
        return src, notices, source_digest
    for attempt in range(5):
        try:
            src.unlink()
            break
        except FileNotFoundError:
            break
        except OSError as exc:
            if attempt == 4:
                notices.append(f"cleanup_pending; verified gzip published, redundant CSV retained: {exc}")
            else:
                time.sleep(.05)
    return dst, notices, source_digest
