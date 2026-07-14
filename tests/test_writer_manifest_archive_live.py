import csv
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pytest

from gpu_power_monitor.archive import (
    cleanup_local,
    copy_run,
    delete_test_run,
    mark_archived,
    set_run_kind,
    verify_archive,
)
from gpu_power_monitor.config import AcquisitionConfig
from gpu_power_monitor.live_buffer import LiveBuffer, read_live_status
from gpu_power_monitor.logging_writer import ChunkWriter
from gpu_power_monitor.manifest import create_manifest, finalize_manifest, load_manifest
from gpu_power_monitor.processing import ProcessedBlock


def block(times):
    times = np.asarray(times, dtype=float)
    return ProcessedBlock(
        time_s=times,
        voltage_v=np.ones_like(times) * 12,
        current1_a=np.ones_like(times),
        current2_a=np.ones_like(times) * 2,
        total_current_a=np.ones_like(times) * 3,
        total_power_w=np.ones_like(times) * 36,
    )


def test_csv_chunk_rotation(tmp_path):
    writer = ChunkWriter(tmp_path, "test", 1.0, dt.datetime(2026, 1, 1), queue_blocks=4)
    writer.write_block(block([0.0, 0.5, 1.0, 1.5]))
    paths = writer.finalize(2.0)
    assert len(paths) == 2
    with open(paths[0], newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["time_s", "voltage_v", "current1_a", "current2_a", "total_power_w"]
    assert len(rows) == 3
    assert not list(tmp_path.glob("*PENDING*"))


def test_manifest_checksums_and_stats(tmp_path):
    config = AcquisitionConfig(measurement_name="test")
    create_manifest(tmp_path, config)
    writer = ChunkWriter(tmp_path, "test", 10.0, dt.datetime(2026, 1, 1))
    writer.write_block(block([0.0, 1.0, 2.0]))
    writer.finalize(3.0)
    finalize_manifest(tmp_path, 3.0)
    data = load_manifest(tmp_path)
    assert data["archive_status"] == "ready_to_archive"
    assert data["files"][0]["sha256"]
    assert data["stats"]["sample_count"] == 3
    assert data["stats"]["average_power_w"] == 36


def test_archive_copy_and_verify(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    archive = tmp_path / "archive"
    config = AcquisitionConfig(measurement_name="test")
    create_manifest(run, config)
    (run / "data.csv").write_text("time_s,voltage_v,current1_a,current2_a,total_power_w\n0,1,2,3,5\n", encoding="utf-8")
    finalize_manifest(run, 1.0)
    dest = copy_run(run, archive)
    assert dest.exists()
    data = load_manifest(run)
    assert data["archive_status"] == "archived_verified"
    verify_archive(run, dest)


def test_mark_archived_enables_cleanup(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    config = AcquisitionConfig(measurement_name="test")
    create_manifest(run, config)
    (run / "data.csv").write_text("time_s,voltage_v,current1_a,current2_a,total_power_w\n0,1,2,3,5\n", encoding="utf-8")
    finalize_manifest(run, 1.0)
    dest = "/n/holylabs/lexie_lab/Lab/gpu_power_logs/run"
    # Pass a Path to catch Windows backslash-mangling of the cluster path.
    assert mark_archived(run, Path(dest)) == dest
    data = load_manifest(run)
    assert data["archive_status"] == "archived_verified"
    assert data["archive"]["destination"] == dest
    assert data["archive"]["verified_at"]
    assert cleanup_local(run, retention_days=0, dry_run=True)
    assert run.exists()


def test_archive_copy_refuses_unmounted_root(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    config = AcquisitionConfig(measurement_name="test")
    create_manifest(run, config)
    (run / "data.csv").write_text("time_s,voltage_v,current1_a,current2_a,total_power_w\n0,1,2,3,5\n", encoding="utf-8")
    finalize_manifest(run, 1.0)
    # A cluster path that is not mounted here: neither it nor its parent exists
    # locally (on Windows it would silently resolve to C:\n\holylabs\...).
    with pytest.raises(FileNotFoundError, match="not mounted"):
        copy_run(run, Path("/n/holylabs/lexie_lab/Lab/gpu_power_logs"))
    assert load_manifest(run)["archive_status"] == "ready_to_archive"


def test_archive_verify_detects_mismatch(tmp_path):
    run = tmp_path / "run"
    dest = tmp_path / "dest"
    run.mkdir()
    dest.mkdir()
    config = AcquisitionConfig(measurement_name="test")
    create_manifest(run, config)
    (run / "data.csv").write_text("time_s,voltage_v,current1_a,current2_a,total_power_w\n0,1,2,3,5\n", encoding="utf-8")
    finalize_manifest(run, 1.0)
    (dest / "data.csv").write_text("bad\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        verify_archive(run, dest)
    assert load_manifest(run)["archive_status"] == "archive_error"


def _finalized_run(tmp_path, *, test_run=False):
    run = tmp_path / "run"
    run.mkdir()
    create_manifest(run, AcquisitionConfig(measurement_name="test", test_run=test_run))
    (run / "data.csv").write_text("time_s,voltage_v,current1_a,current2_a,total_power_w\n0,1,2,3,5\n", encoding="utf-8")
    finalize_manifest(run, 1.0)
    return run


def test_test_run_stays_local_and_is_deletable(tmp_path):
    run = _finalized_run(tmp_path, test_run=True)
    data = load_manifest(run)
    assert data["run_kind"] == "test"
    # Completing a test run must not arm archiving.
    assert data["archive_status"] == "local_only"
    with pytest.raises(RuntimeError, match="test run"):
        copy_run(run, tmp_path / "archive")
    delete_test_run(run)
    assert not run.exists()


def test_delete_refuses_archive_bound_run(tmp_path):
    run = _finalized_run(tmp_path)
    with pytest.raises(RuntimeError, match="not a test run"):
        delete_test_run(run)
    assert run.exists()


def test_promote_arms_archiving_and_demote_disarms(tmp_path):
    run = _finalized_run(tmp_path, test_run=True)
    set_run_kind(run, "archive")
    data = load_manifest(run)
    assert data["run_kind"] == "archive"
    assert data["archive_status"] == "ready_to_archive"
    set_run_kind(run, "test")
    data = load_manifest(run)
    assert data["run_kind"] == "test"
    assert data["archive_status"] == "local_only"


def test_demote_refused_once_archived(tmp_path):
    run = _finalized_run(tmp_path)
    mark_archived(run, Path("/n/holylabs/lexie_lab/Lab/gpu_power_logs/run"))
    with pytest.raises(RuntimeError, match="cannot become a test run"):
        set_run_kind(run, "test")
    assert load_manifest(run)["run_kind"] == "archive"


def test_web_latest_payload(tmp_path):
    from gpu_power_monitor.web.server import build_latest_payload

    live = LiveBuffer(tmp_path, sample_rate_hz=10, window_sec=1, max_points=50)
    live.append(block([0.0, 0.1, 0.2]), state="LIVE")
    payload = build_latest_payload(tmp_path, stale_after_sec=10)
    assert payload["state"] == "LIVE"
    assert len(payload["time_s"]) == 3
    assert "current1_a" in payload and "current2_a" in payload
    assert payload["run_peak_power_w"] == 36
    assert payload["run_avg_power_w"] == 36


def test_live_buffer_freshness(tmp_path):
    live = LiveBuffer(tmp_path, sample_rate_hz=10, window_sec=1, max_points=10)
    live.append(block([0.0, 0.1]), state="LIVE")
    assert read_live_status(tmp_path, stale_after_sec=10).state == "LIVE"
    status_path = tmp_path / "latest_status.json"
    data = json.loads(status_path.read_text(encoding="utf-8"))
    data["generated_at_epoch"] = time.time() - 100
    status_path.write_text(json.dumps(data), encoding="utf-8")
    # A LIVE snapshot whose writer went quiet is stale data, not "no data".
    assert read_live_status(tmp_path, stale_after_sec=1).state == "STALE"
