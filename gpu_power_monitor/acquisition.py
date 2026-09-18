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
from .remote import NvmlLogger, remote_unready_reason
from .utils import runs_root, unique_run_dir


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
        run_dir = unique_run_dir(
            runs_root(config.storage.output_root, test=config.test_run), config.measurement_name
        )
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    create_manifest(run_dir, config)
    print_fn(f"[INFO] Run directory: {run_dir}")

    gpus = config.channels.gpus
    source = (
        SimulatedDaqSource(
            config.sample_rate_hz,
            config.chunk_size,
            gpus=gpus,
            current_scale=config.scaling.current_scale,
        )
        if simulate
        else NIDaqSource(config.channels, config.sample_rate_hz, config.chunk_size)
    )
    processor = PowerProcessor(
        sample_rate_hz=config.sample_rate_hz,
        gpus=gpus,
        current_scale=config.scaling.current_scale,
        voltage_delay_samples=config.processing.voltage_delay_samples,
        current_delay_samples=config.processing.current_delay_samples,
        power_average_samples=config.processing.power_average_samples,
    )
    # e.g. nidaq_Dev1_ai0_ai7_4GPU (first voltage channel to last current channel)
    prefix = (
        f"{config.logging.file_prefix}_{config.channels.device}"
        f"_{gpus[0].voltage}_{gpus[-1].current}_{len(gpus)}GPU"
    )
    writer = ChunkWriter(
        out_dir=run_dir,
        prefix=prefix,
        chunk_len_sec=config.logging.chunk_duration_sec,
        start_wall_dt=dt.datetime.now(),
        labels=config.channels.labels,
        queue_blocks=config.logging.queue_blocks,
        file_format=config.logging.format,
    )
    live = LiveBuffer(
        run_dir, config.sample_rate_hz, config.display.window_sec, config.channels.labels
    )

    # NVML logging on gamma brackets the power run so both data streams cover
    # the same window. Best-effort throughout: gamma being down never blocks
    # or aborts a power run.
    nvml: NvmlLogger | None = None
    if simulate:
        pass  # simulated data + real GPU telemetry would only mislead
    elif (reason := remote_unready_reason(config.remote)) is not None:
        print_fn(f"[INFO] NVML logging on gamma skipped: {reason}")
    else:
        nvml = NvmlLogger(config.remote, run_dir.name)
        try:
            nvml.start()
            print_fn(
                f"[INFO] NVML logger running on {config.remote.host} "
                f"(~{run_dir.name}/nvml.csv every {config.remote.nvml_interval_ms} ms)"
            )
        except Exception as exc:
            print_fn(f"[WARN] Could not start the NVML logger on gamma: {exc}")
            nvml = None

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
            raw_voltages, raw_currents = source.read(n_to_read)
            if not raw_voltages:
                continue
            block = processor.process(raw_voltages, raw_currents)
            writer.write_block(block)
            live.append(block, state="LIVE")
            if simulate:
                time.sleep(len(raw_voltages[0]) / float(config.sample_rate_hz))
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
        if nvml is not None:
            # Fetch before finalize so the NVML files land in the manifest's
            # checksummed inventory (and therefore in the Globus archive).
            result = nvml.stop_and_fetch(run_dir / "nvml")
            update_manifest(run_dir, nvml=result)
            if result.get("status") == "fetched":
                print_fn(f"[INFO] NVML files fetched from gamma: {', '.join(result['files'])}")
            else:
                print_fn(f"[WARN] NVML fetch failed: {result.get('error')}")
                print_fn(f'[INFO] Retry later with: pai fetch "{run_dir.name}"')
        if not failed:
            # On failure the manifest already says status=error and the live
            # state ERROR; finalizing here would overwrite both with "completed".
            finalize_manifest(run_dir, end_time)
            live.mark_state("COMPLETE")
    print_fn(f"[INFO] Completed run: {run_dir}")
    return run_dir
