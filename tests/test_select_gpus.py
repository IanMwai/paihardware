import csv
import dataclasses

import pytest

from gpu_power_monitor.acquisition import acquire
from gpu_power_monitor.config import AcquisitionConfig, RemoteConfig, select_gpus


def test_none_blank_and_all_keep_every_gpu():
    config = AcquisitionConfig()
    assert select_gpus(config, None) is config
    assert select_gpus(config, "  ") is config
    assert select_gpus(config, "ALL") is config


def test_select_by_index_label_and_mixed_case():
    config = AcquisitionConfig()
    assert select_gpus(config, "1").channels.labels == ["GPU1"]
    assert select_gpus(config, "gpu3").channels.labels == ["GPU3"]
    assert select_gpus(config, "GPU2, 4").channels.labels == ["GPU2", "GPU4"]


def test_selection_order_does_not_reorder_columns():
    config = AcquisitionConfig()
    assert select_gpus(config, "3,1").channels.labels == ["GPU1", "GPU3"]


def test_unknown_gpu_lists_valid_choices():
    with pytest.raises(ValueError, match="GPU1, GPU2, GPU3, GPU4"):
        select_gpus(AcquisitionConfig(), "GPU9")
    with pytest.raises(ValueError, match="indices 1-4"):
        select_gpus(AcquisitionConfig(), "0")


def test_subset_calibration_survives():
    subset = select_gpus(AcquisitionConfig(), "2")
    gpu = subset.channels.gpus[0]
    assert gpu.label == "GPU2"
    assert gpu.current_sign == -1.0


def test_single_gpu_simulated_run_writes_only_its_columns(tmp_path):
    config = AcquisitionConfig(sample_rate_hz=1000, chunk_size=100)
    config = dataclasses.replace(
        config,
        storage=dataclasses.replace(config.storage, output_root=tmp_path),
        remote=RemoteConfig(enabled=False),
    )
    config = select_gpus(config, "GPU1")
    run_dir = acquire(config, simulate=True, duration_sec=0.3, run_dir=tmp_path / "run")
    csv_files = sorted(run_dir.glob("*.csv"))
    assert csv_files
    with open(csv_files[0], newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header == ["time_s", "gpu1_voltage_v", "gpu1_current_a", "gpu1_power_w", "total_power_w"]
    assert "1GPU" in csv_files[0].name
