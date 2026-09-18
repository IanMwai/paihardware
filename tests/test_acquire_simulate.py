"""End-to-end smoke test: a short simulated 4-GPU run produces the expected
CSV columns, manifest inventory, and live snapshot — no NI hardware needed."""

import csv
import json

import dataclasses

from gpu_power_monitor.acquisition import acquire
from gpu_power_monitor.config import AcquisitionConfig, RemoteConfig


def _fast_config(tmp_path):
    config = AcquisitionConfig(sample_rate_hz=1000, chunk_size=100)
    storage = dataclasses.replace(config.storage, output_root=tmp_path)
    # No gamma in CI: credentials empty disables the NVML integration.
    return dataclasses.replace(config, storage=storage, remote=RemoteConfig(enabled=False))


def test_simulated_run_writes_per_gpu_columns(tmp_path):
    config = _fast_config(tmp_path)
    run_dir = acquire(config, simulate=True, duration_sec=0.5, run_dir=tmp_path / "run")

    csv_files = sorted(run_dir.glob("*.csv"))
    assert csv_files, "expected at least one CSV chunk"
    with open(csv_files[0], newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        first = next(reader)
    labels = [gpu.label.lower() for gpu in config.channels.gpus]
    expected = ["time_s"]
    for gpu in labels:
        expected += [f"{gpu}_voltage_v", f"{gpu}_current_a", f"{gpu}_power_w"]
    expected.append("total_power_w")
    assert header == expected
    assert len(first) == len(expected)

    # Every GPU's scaled values should be plausible: ~12 V and positive current
    # (the sign flip must cancel the simulated negative raw values on GPU2-4).
    row = dict(zip(header, map(float, first)))
    for gpu in labels:
        assert 11.0 < row[f"{gpu}_voltage_v"] < 13.0
        assert row[f"{gpu}_current_a"] > 0

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert len(manifest["channels"]["gpus"]) == 4
    assert manifest["files"], "manifest should inventory the CSV chunks"
    assert (run_dir / "latest_status.json").exists()
