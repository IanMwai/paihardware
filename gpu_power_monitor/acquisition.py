"""The acquisition loop behind `pai hardware`.

Reads the DAQ (real or simulated), processes each block, logs it to CSV, and
updates the live snapshot the dashboard reads. The loop is interruptible three
ways: a background ``stop_event``, Ctrl+C in a foreground run, or a STOP file
in the run folder. Each path closes the DAQ and writer, fetches telemetry, and
records the terminal status.
"""

from __future__ import annotations

import datetime as dt
import json
import threading
import time
from pathlib import Path
from typing import Callable

from .config import AcquisitionConfig, select_nvml_gpus
from .daq import NIDaqSource, SimulatedDaqSource
from .live_buffer import LiveBuffer
from .logging_writer import ChunkWriter
from .manifest import create_manifest, finalize_manifest, update_manifest, load_manifest
from .processing import PowerProcessor
from .remote import NvmlLogger, remote_unready_reason
from .utils import runs_root, stop_file, unique_run_dir

# How often the loop looks for a STOP file in the run folder. Cheap on local
# disk; network shares also use this bounded polling interval.
STOP_FILE_POLL_SEC = 0.5


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
    progress_path: Path | None = None,
    run_identity: str | None = None,
) -> Path:
    config = select_nvml_gpus(config, None)
    if not config.channels.gpus:
        raise ValueError("At least one GPU channel pair is required")
    if config.sample_rate_hz <= 0 or config.chunk_size <= 0:
        raise ValueError("Sample rate and chunk size must be positive")
    if run_dir is None:
        run_dir = unique_run_dir(
            runs_root(config.storage.output_root, test=config.test_run), config.measurement_name
        )
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = run_identity or run_dir.name
    create_manifest(run_dir, config)
    update_manifest(run_dir, nvml_selection={"gpu_ids": list(config.remote.gpu_ids), "required": config.remote.required})
    if run_identity:
        update_manifest(run_dir, run_id=identity)
    print_fn(f"[INFO] Run directory: {run_dir}")

    if (not simulate and config.remote.enabled and config.remote.host
            and len(config.channels.gpus) < 4 and not config.remote.gpu_ids):
        message = "Subset measurement needs --nvml-gpus with verified physical IDs, or remote.enabled: false for DAQ-only data"
        update_manifest(run_dir, status="error", archive_status="archive_error", error=message)
        raise ValueError(message)
    if not simulate and config.remote.required:
        reason = remote_unready_reason(config.remote)
        if reason:
            update_manifest(run_dir, status="error", error=reason)
            raise RuntimeError(reason)
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
        compress=config.logging.compress,
        sample_rate_hz=config.sample_rate_hz,
        enqueue_timeout_sec=config.logging.enqueue_timeout_sec,
        compression_queue_chunks=config.logging.compression_queue_chunks,
    )
    live = LiveBuffer(
        run_dir, config.sample_rate_hz, config.display.window_sec, config.channels.labels
    )

    # NVML logging on gamma brackets the power run so both data streams cover
    # the same window. Explicit GPU selection requires successful telemetry;
    # the legacy all-GPU mode remains best effort.
    nvml: NvmlLogger | None = None
    if simulate:
        pass  # simulated data + real GPU telemetry would only mislead
    elif (reason := remote_unready_reason(config.remote)) is not None:
        print_fn(f"[INFO] NVML logging on gamma skipped: {reason}")
    else:
        nvml = NvmlLogger(config.remote, identity)
        try:
            nvml.start()
            print_fn(
                f"[INFO] NVML logger running on {config.remote.host} "
                f"(~{identity}/nvml.csv every {config.remote.nvml_interval_ms} ms)"
            )
        except Exception as exc:
            print_fn(f"[WARN] Could not start the NVML logger on gamma: {exc}")
            nvml = None
            if config.remote.required:
                writer.finalize(0.0)
                update_manifest(run_dir, status="error", error=str(exc))
                raise RuntimeError(f"Required NVML logger failed: {exc}") from exc

    from .utils import write_json, read_json
    sequence = 0
    daq_available = daq_max_available = 0
    block_min = block_max = None
    read_blocks = 0
    driver_capacity = getattr(source, "buffer_capacity_samples", max(config.chunk_size*200, config.sample_rate_hz*60))
    diagnostic_dir = run_dir / "diagnostics"
    diagnostic_dir.mkdir(exist_ok=True)
    diagnostic_path = diagnostic_dir / "writer.jsonl"
    last_diagnostic = float("-inf")
    diagnostic_error = None

    def snapshot(phase):
        return {"phase": phase, "nvml_gpus": list(config.remote.gpu_ids),
                "daq_labels": list(config.channels.labels),
                "power_average_samples": config.processing.power_average_samples,
                "acquired_samples": processor.sample_index, "written_samples": writer.samples_written,
                "queue_depth": writer.queue.qsize(), "writer": writer.diagnostics(),
                "diagnostic_error": diagnostic_error,
                "daq": {"available_samples": daq_available, "max_available_samples": daq_max_available,
                        "backlog_sec": daq_available/config.sample_rate_hz,
                        "buffer_capacity_samples": driver_capacity, "read_blocks": read_blocks,
                        "min_block_samples": block_min, "max_block_samples": block_max}}

    def progress(phase, *, force=False):
        nonlocal sequence, last_diagnostic, diagnostic_error
        state = snapshot(phase)
        now = time.monotonic()
        try:
            if progress_path is not None:
                sequence += 1
                write_json(progress_path, dict(state, sequence=sequence))
            if force or now-last_diagnostic >= 10:
                with diagnostic_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(dict(state, monotonic=now, wall_epoch=time.time())) + "\n")
                last_diagnostic = now
        except OSError as exc:
            if diagnostic_error is None:
                print_fn(f"[WARN] Could not save recording diagnostics: {exc}")
            diagnostic_error = str(exc)
        return state

    def available_samples():
        nonlocal daq_available, daq_max_available
        daq_available = source.available_samples()
        daq_max_available = max(daq_max_available, daq_available)
        if daq_available >= driver_capacity*.9:
            raise RuntimeError(f"DAQ buffer backlog reached {daq_available/config.sample_rate_hz:.3f}s; "
                               "stopping before hardware buffer overflow")
        return daq_available

    def while_waiting():
        nonlocal next_progress
        available_samples()
        if time.monotonic() >= next_progress:
            progress("ready")
            next_progress = time.monotonic() + 1

    def record_samples(count, *, timeout_sec=None):
        nonlocal block_min, block_max, read_blocks
        raw_voltages, raw_currents = source.read(count)
        if (len(raw_voltages) != len(gpus) or len(raw_currents) != len(gpus)
                or any(len(values) != count for values in raw_voltages + raw_currents)):
            raise RuntimeError("DAQ returned an incomplete block; refusing silent sample loss")
        block_min = count if block_min is None else min(block_min, count)
        block_max = count if block_max is None else max(block_max, count)
        read_blocks += 1
        block = processor.process(raw_voltages, raw_currents)
        writer.write_block(block, on_wait=while_waiting, timeout_sec=timeout_sec)
        live.append(block, state="LIVE")

    start = time.monotonic()
    failed = False
    stop_marker = stop_file(run_dir)
    next_stop_check = start
    next_progress = start
    stop_reason = None
    try:
        source.start()
        progress("ready")
        while True:
            writer.check_health()
            if stop_event is not None and stop_event.is_set():
                break
            if duration_sec is not None and time.monotonic() - start >= duration_sec:
                break
            if time.monotonic() >= next_stop_check:
                next_stop_check = time.monotonic() + STOP_FILE_POLL_SEC
                if stop_marker.exists():
                    stop_reason = read_json(stop_marker).get("reason")
                    print_fn("[INFO] Stop requested (STOP file found) - finishing the run.")
                    break
            available = available_samples()
            if available < config.chunk_size:
                # Hardware buffering accumulates a full block without dropping
                # or inventing samples. Keep stop/health checks responsive.
                time.sleep(min(.01, config.chunk_size/config.sample_rate_hz))
                continue
            record_samples(config.chunk_size)
            if time.monotonic() >= next_progress:
                progress("ready")
                next_progress = time.monotonic() + 1
            if simulate:
                time.sleep(config.chunk_size / float(config.sample_rate_hz))
        if not simulate:
            # Preserve all samples already available at the stop boundary,
            # including a final partial block. Do not chase newly arriving data.
            remaining = available_samples()
            deadline = time.monotonic() + 10.
            while remaining:
                timeout = deadline-time.monotonic()
                if timeout <= 0:
                    raise RuntimeError("DAQ stop-boundary samples did not drain within 10s")
                count = min(remaining, config.chunk_size)
                record_samples(count, timeout_sec=timeout)
                remaining -= count
    except KeyboardInterrupt:
        print_fn("[INFO] Acquisition interrupted by user.")
    except Exception as exc:
        failed = True
        update_manifest(run_dir, status="error", archive_status="archive_error", error=str(exc),
                        recording_diagnostics=progress("error", force=True))
        live.mark_state("ERROR")
        raise
    finally:
        progress("finalizing")
        cleanup_error = None
        try:
            source.stop()
        except Exception as exc:
            cleanup_error = exc
        end_time = processor.sample_index / float(config.sample_rate_hz)
        try:
            writer.finalize(end_time)
            progress("finalizing", force=True)
        except Exception as exc:
            cleanup_error = cleanup_error or exc
        if nvml is not None:
            # Fetch before finalize so the NVML files land in the manifest's
            # checksummed inventory (and therefore in the Globus archive).
            result = nvml.stop_and_fetch(run_dir / "nvml")
            if config.remote.required and result.get("status") == "fetched" and end_time > 10:
                if result.get("observed_span_s", 0) < end_time * .9:
                    cleanup_error = cleanup_error or RuntimeError("Required NVML covers less than 90% of the DAQ window")
            update_manifest(run_dir, nvml=result)
            if result.get("status") == "fetched":
                print_fn(f"[INFO] NVML files fetched from gamma: {', '.join(result['files'])}")
            else:
                print_fn(f"[WARN] NVML fetch failed: {result.get('error')}")
                print_fn(f'[INFO] Retry later with: pai fetch "{run_dir.name}"')
                if config.remote.required:
                    cleanup_error = cleanup_error or RuntimeError(f"Required NVML fetch failed: {result.get('error')}")
        # The STOP marker is a request, not data: drop it before the manifest
        # inventories the folder (and so a later run of the same name is not
        # stopped at birth).
        try:
            stop_marker.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            print_fn(f"[WARN] Could not remove {stop_marker.name}: {exc}")
        update_manifest(run_dir, recording_diagnostics=snapshot("error" if failed or cleanup_error else "finalized"))
        if cleanup_error is not None:
            primary_failed = failed
            failed = True
            original_error = load_manifest(run_dir).get("error")
            update_manifest(run_dir, status="error", archive_status="archive_error",
                            error=original_error or str(cleanup_error), cleanup_error=str(cleanup_error))
            live.mark_state("ERROR")
            if not primary_failed:
                raise cleanup_error
        if not failed:
            # On failure the manifest already says status=error and the live
            # state ERROR; finalizing here would overwrite both with "completed".
            if writer.samples_written != processor.sample_index:
                update_manifest(run_dir, status="error", archive_status="archive_error",
                                error="Acquired/written sample count mismatch",
                                recording_diagnostics=snapshot("error"))
                live.mark_state("ERROR")
                raise RuntimeError("Acquired/written sample count mismatch")
            finalize_manifest(run_dir, end_time, stats=writer.statistics())
            if progress_path is not None and stop_reason not in (None, "workload_completed"):
                update_manifest(run_dir, status="interrupted", archive_status="archive_error", error=stop_reason)
                live.mark_state("ERROR")
            else:
                live.mark_state("COMPLETE")
    print_fn(f"[INFO] Completed run: {run_dir}")
    return run_dir
