"""Archive-push state machine, with the Globus CLI mocked out.

ready_to_archive -> transfer_pending -> archived_verified | push_failed,
plus the guard rails (unconfigured endpoints, non-completed runs).
"""

from pathlib import Path

import pytest

from gpu_power_monitor import globus_push
from gpu_power_monitor.config import AcquisitionConfig, GlobusConfig, config_from_mapping
from gpu_power_monitor.globus_push import GlobusUnavailable, check_task, push_run
from gpu_power_monitor.manifest import create_manifest, finalize_manifest, load_manifest

ARCHIVE_ROOT = Path("/n/holylabs/lexie_lab/Lab/gpu_power_logs")
GLOBUS_CFG = GlobusConfig(local_endpoint_id="local-uuid", remote_endpoint_id="remote-uuid")


def completed_run(tmp_path) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    create_manifest(run, AcquisitionConfig(measurement_name="test"))
    (run / "data.csv").write_text(
        "time_s,voltage_v,current1_a,current2_a,total_power_w\n0,1,2,3,5\n", encoding="utf-8"
    )
    finalize_manifest(run, 1.0)
    return run


@pytest.fixture
def globus_cli(monkeypatch):
    """Pretend globus-cli is installed and capture/replay its JSON calls."""
    calls = []
    responses = {}

    def fake_run(args):
        calls.append(args)
        return responses[args[0]]

    monkeypatch.setattr(globus_push.shutil, "which", lambda name: "C:/fake/globus.exe")
    monkeypatch.setattr(globus_push, "_run_globus", fake_run)
    return calls, responses


def test_unready_without_endpoint_ids():
    reason = globus_push.globus_unready_reason(GlobusConfig())
    assert "not configured" in reason


def test_push_sets_transfer_pending(tmp_path, globus_cli):
    calls, responses = globus_cli
    responses["transfer"] = {"task_id": "task-123"}
    run = completed_run(tmp_path)
    task_id = push_run(run, GLOBUS_CFG, ARCHIVE_ROOT)
    assert task_id == "task-123"
    data = load_manifest(run)
    assert data["archive_status"] == "transfer_pending"
    assert data["archive"]["task_id"] == "task-123"
    assert data["archive"]["destination"] == "/n/holylabs/lexie_lab/Lab/gpu_power_logs/run"
    transfer_args = calls[0]
    assert "--recursive" in transfer_args and "checksum" in transfer_args


def test_push_refuses_uncompleted_run(tmp_path, globus_cli):
    run = tmp_path / "run"
    run.mkdir()
    create_manifest(run, AcquisitionConfig(measurement_name="test"))  # status: running
    with pytest.raises(RuntimeError, match="not completed"):
        push_run(run, GLOBUS_CFG, ARCHIVE_ROOT)


def test_check_task_success_marks_archived(tmp_path, globus_cli):
    calls, responses = globus_cli
    responses["transfer"] = {"task_id": "task-123"}
    responses["task"] = {"status": "SUCCEEDED"}
    run = completed_run(tmp_path)
    push_run(run, GLOBUS_CFG, ARCHIVE_ROOT)
    assert check_task(run) == "archived_verified"
    data = load_manifest(run)
    assert data["archive_status"] == "archived_verified"
    assert data["archive"]["task_id"] == "task-123"  # kept through mark_archived
    assert data["archive"]["destination"] == "/n/holylabs/lexie_lab/Lab/gpu_power_logs/run"
    assert data["archive"]["verified_at"]


def test_check_task_failure_records_error(tmp_path, globus_cli):
    calls, responses = globus_cli
    responses["transfer"] = {"task_id": "task-123"}
    responses["task"] = {"status": "FAILED", "fatal_error": {"description": "quota exceeded"}}
    run = completed_run(tmp_path)
    push_run(run, GLOBUS_CFG, ARCHIVE_ROOT)
    assert check_task(run) == "push_failed"
    data = load_manifest(run)
    assert data["archive_status"] == "push_failed"
    assert "quota" in data["archive"]["last_error"]


def test_check_task_still_active_stays_pending(tmp_path, globus_cli):
    calls, responses = globus_cli
    responses["transfer"] = {"task_id": "task-123"}
    responses["task"] = {"status": "ACTIVE", "nice_status": "CONNECTION_FAILED"}
    run = completed_run(tmp_path)
    push_run(run, GLOBUS_CFG, ARCHIVE_ROOT)
    # Cluster down: the task keeps retrying on Globus's side; nothing is lost.
    assert check_task(run) == "transfer_pending"
    data = load_manifest(run)
    assert data["archive_status"] == "transfer_pending"
    assert "CONNECTION_FAILED" in data["archive"]["last_status"]
    # The raw code is translated into an actionable hint for the operator.
    assert "Globus Connect Personal" in data["archive"]["last_hint"]


def test_check_task_inactive_hints_reauth(tmp_path, globus_cli):
    calls, responses = globus_cli
    responses["transfer"] = {"task_id": "task-123"}
    responses["task"] = {"status": "INACTIVE", "nice_status": None}
    run = completed_run(tmp_path)
    push_run(run, GLOBUS_CFG, ARCHIVE_ROOT)
    assert check_task(run) == "transfer_pending"
    data = load_manifest(run)
    assert "globus login" in data["archive"]["last_hint"]


def test_config_parses_globus_section():
    config = config_from_mapping(
        {
            "storage": {
                "output_root": "output",
                "globus": {"local_endpoint_id": "a", "remote_endpoint_id": "b", "auto_push": False},
            }
        }
    )
    assert config.storage.globus.local_endpoint_id == "a"
    assert config.storage.globus.auto_push is False
    assert config.storage.globus.deadline_days == 7
