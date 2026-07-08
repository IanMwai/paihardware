"""The acquisition loop behind `pai hardware`.

Reads the DAQ (real or simulated), processes each block, logs it to CSV, and
updates the live snapshot the dashboard reads. The loop is interruptible two
ways: a ``stop_event`` (used when the dashboard runs it in a background thread)
or KeyboardInterrupt (Ctrl+C in a foreground run).
"""

from __future__ import annotations

import datetime as dt
import threading
import time
from pathlib import Path
from typing import Callable

from .config import AcquisitionConfig
from .daq import NIDaqSource, SimulatedDaqSource
from .live_buffer import LiveBuffer
from .logging_writer import ChunkWriter
from .manifest import create_manifest, finalize_manifest, update_manifest
from .processing import PowerProcessor
from .utils import unique_run_dir


def with_measurement_name(config: AcquisitionConfig, name: str | None) -> AcquisitionConfig:
    """Return a copy of config with measurement_name overridden (or config unchanged)."""
    if not name:
        return config
    return config.__class__(**{**config.__dict__, "measurement_name": name})


def acquire(
    config: AcquisitionConfig,
    *,
    simulate: bool = False,
    duration_sec: float | None = None,
    run_dir: Path | None = None,
    stop_event: threading.Event | None = None,
    print_fn: Callable[[str], None] = print,
) -> Path:
    if run_dir is None:
        run_dir = unique_run_dir(config.storage.output_root, config.measurement_name)
    run_dir = Path(run_dir)
    create_manifest(run_dir, config)
    print_fn(f"[INFO] Run directory: {run_dir}")

    source = (
        SimulatedDaqSource(config.sample_rate_hz, config.chunk_size)
        if simulate
        else NIDaqSource(config.channels, config.sample_rate_hz, config.chunk_size)
    )
    processor = PowerProcessor(
        sample_rate_hz=config.sample_rate_hz,
        voltage_scale=config.scaling.voltage_scale,
        current_scale=config.scaling.current_scale,
        voltage_delay_samples=config.processing.voltage_delay_samples,
        current_delay_samples=config.processing.current_delay_samples,
        power_average_samples=config.processing.power_average_samples,
    )
    writer = ChunkWriter(
        out_dir=run_dir,
        prefix=f"{config.logging.file_prefix}_{config.channels.device}_{config.channels.voltage}_{config.channels.current1}_{config.channels.current2}",
        chunk_len_sec=config.logging.chunk_duration_sec,
        start_wall_dt=dt.datetime.now(),
        queue_blocks=config.logging.queue_blocks,
        file_format=config.logging.format,
    )
    live = LiveBuffer(run_dir, config.sample_rate_hz, config.display.window_sec)
    start = time.time()
    failed = False
    source.start()
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            if duration_sec is not None and time.time() - start >= duration_sec:
                break
            available = source.available_samples()
            if available <= 0:
                time.sleep(0.01)
                continue
            n_to_read = min(max(available, 1), config.chunk_size * 10)
            raw_v, raw_i1, raw_i2 = source.read(n_to_read)
            block = processor.process(raw_v, raw_i1, raw_i2)
            writer.write_block(block)
            live.append(block, state="LIVE")
            if simulate:
                time.sleep(len(raw_v) / float(config.sample_rate_hz))
    except KeyboardInterrupt:
        print_fn("[INFO] Acquisition interrupted by user.")
    except Exception as exc:
        failed = True
        update_manifest(run_dir, status="error", archive_status="archive_error", error=str(exc))
        live.mark_state("ERROR")
        raise
    finally:
        source.stop()
        end_time = processor.sample_index / float(config.sample_rate_hz)
        writer.finalize(end_time)
        if not failed:
            # On failure the manifest already says status=error and the live
            # state ERROR; finalizing here would overwrite both with "completed".
            finalize_manifest(run_dir, end_time)
            live.mark_state("COMPLETE")
    print_fn(f"[INFO] Completed run: {run_dir}")
    return run_dir
