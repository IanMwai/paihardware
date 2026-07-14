"""`pai runs` and the output/ vs output/test/ folder split.

The manifest's run_kind is the source of truth; the folder location is a
derived convenience, so listing/resolution must find runs in either place and
promote/demote must move the folder to match.
"""

from pathlib import Path

import pytest

from gpu_power_monitor import cli
from gpu_power_monitor.config import AcquisitionConfig
from gpu_power_monitor.manifest import create_manifest, finalize_manifest, load_manifest
from gpu_power_monitor.utils import TEST_RUNS_SUBDIR, runs_root


def make_run(parent: Path, name: str, *, test: bool = False) -> Path:
    run = parent / name
    run.mkdir(parents=True)
    create_manifest(run, AcquisitionConfig(measurement_name=name, test_run=test))
    (run / "data.csv").write_text(
        "time_s,voltage_v,current1_a,current2_a,total_power_w\n0,1,2,3,5\n", encoding="utf-8"
    )
    finalize_manifest(run, 1.0)
    return run


@pytest.fixture
def output_root(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_output_root", lambda: tmp_path)
    return tmp_path


def test_runs_root_layout(tmp_path):
    assert runs_root(tmp_path, test=False) == tmp_path
    assert runs_root(tmp_path, test=True) == tmp_path / TEST_RUNS_SUBDIR


def test_list_and_resolve_cover_test_subdir(output_root):
    data_run = make_run(output_root, "GPU Run 0_20260101_000000")
    test_run = make_run(output_root / TEST_RUNS_SUBDIR, "Test Run 0_20260101_000001", test=True)
    names = {r.name for r in cli._list_runs()}
    # The test/ container itself must not be listed as a run.
    assert names == {data_run.name, test_run.name}
    assert cli._resolve_run(data_run.name) == data_run
    assert cli._resolve_run(test_run.name) == test_run


def test_demote_and_promote_move_the_folder(output_root, capsys):
    run = make_run(output_root, "GPU Run 1_20260101_000000")
    assert cli.run_runs("demote", run.name) == 0
    moved = output_root / TEST_RUNS_SUBDIR / run.name
    assert moved.is_dir() and not run.exists()
    assert load_manifest(moved)["run_kind"] == "test"

    assert cli.run_runs("promote", moved.name) == 0
    back = output_root / run.name
    assert back.is_dir() and not moved.exists()
    data = load_manifest(back)
    assert data["run_kind"] == "archive"
    assert data["archive_status"] == "ready_to_archive"


def test_list_flags_folder_mismatch(output_root, capsys):
    # A test run sitting at the output root (e.g. from before the split, or a
    # failed move) is flagged with the command that heals it.
    run = make_run(output_root, "Test Run 9_20260101_000000", test=True)
    assert cli.run_runs("list", None) == 0
    out = capsys.readouterr().out
    assert "folder mismatch" in out
    assert f'pai runs demote "{run.name}"' in out

    # Demote is a no-op on the manifest but still heals the folder.
    assert cli.run_runs("demote", run.name) == 0
    assert (output_root / TEST_RUNS_SUBDIR / run.name).is_dir()
    assert cli.run_runs("list", None) == 0
    assert "folder mismatch" not in capsys.readouterr().out
