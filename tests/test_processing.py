import numpy as np

from gpu_power_monitor.config import GpuChannel
from gpu_power_monitor.processing import PowerProcessor, moving_average_causal


def make_gpus(n=2, voltage_scale=2.0, signs=None):
    signs = signs or [1.0] * n
    return [
        GpuChannel(f"GPU{i + 1}", f"ai{2 * i}", f"ai{2 * i + 1}", voltage_scale, signs[i])
        for i in range(n)
    ]


def test_scaling_and_power_no_delay_no_average():
    processor = PowerProcessor(
        sample_rate_hz=10,
        gpus=make_gpus(2),
        current_scale=10,
        power_average_samples=1,
    )
    block = processor.process([[1, 2], [3, 4]], [[0.1, 0.2], [0.3, 0.4]])
    np.testing.assert_allclose(block.voltage_v[0], [2, 4])
    np.testing.assert_allclose(block.voltage_v[1], [6, 8])
    np.testing.assert_allclose(block.current_a[0], [1, 2])
    np.testing.assert_allclose(block.current_a[1], [3, 4])
    np.testing.assert_allclose(block.power_w[0], [2, 8])
    np.testing.assert_allclose(block.power_w[1], [18, 32])
    np.testing.assert_allclose(block.total_power_w, [20, 40])
    np.testing.assert_allclose(block.total_current_a, [4, 6])


def test_current_sign_flips_inverted_shunts():
    processor = PowerProcessor(
        sample_rate_hz=10,
        gpus=make_gpus(2, voltage_scale=1.0, signs=[1.0, -1.0]),
        current_scale=1,
        power_average_samples=1,
    )
    block = processor.process([[1.0], [1.0]], [[2.0], [-2.0]])
    np.testing.assert_allclose(block.current_a[0], [2.0])
    np.testing.assert_allclose(block.current_a[1], [2.0])
    np.testing.assert_allclose(block.total_power_w, [4.0])


def test_per_gpu_voltage_scales_are_independent():
    gpus = [
        GpuChannel("GPU1", "ai0", "ai1", 2.0),
        GpuChannel("GPU2", "ai2", "ai3", 3.0),
    ]
    processor = PowerProcessor(
        sample_rate_hz=10, gpus=gpus, current_scale=1, power_average_samples=1
    )
    block = processor.process([[1.0], [1.0]], [[1.0], [1.0]])
    np.testing.assert_allclose(block.voltage_v[0], [2.0])
    np.testing.assert_allclose(block.voltage_v[1], [3.0])


def test_moving_average_continues_across_blocks():
    processor = PowerProcessor(
        sample_rate_hz=10,
        gpus=make_gpus(1, voltage_scale=1.0),
        current_scale=1,
        power_average_samples=3,
    )
    first = processor.process([[1, 1]], [[1, 2]])
    second = processor.process([[1, 1]], [[3, 4]])
    np.testing.assert_allclose(first.power_w[0], [1, 1.5])
    np.testing.assert_allclose(second.power_w[0], [2, 3])


def test_positive_voltage_delay_alignment():
    processor = PowerProcessor(
        sample_rate_hz=10,
        gpus=make_gpus(1, voltage_scale=1.0),
        current_scale=1,
        voltage_delay_samples=1,
        current_delay_samples=0,
        power_average_samples=1,
    )
    block = processor.process([[10, 20, 30]], [[1, 1, 1]])
    assert np.isnan(block.voltage_v[0][0])
    np.testing.assert_allclose(block.voltage_v[0][1:], [10, 20])
    np.testing.assert_allclose(block.power_w[0][1:], [10, 20])


def test_negative_current_delay_alignment():
    processor = PowerProcessor(
        sample_rate_hz=10,
        gpus=make_gpus(1, voltage_scale=1.0),
        current_scale=1,
        voltage_delay_samples=0,
        current_delay_samples=1,
        power_average_samples=1,
    )
    block = processor.process([[10, 20, 30]], [[1, 2, 3]])
    assert np.isnan(block.current_a[0][0])
    np.testing.assert_allclose(block.current_a[0][1:], [1, 2])
    np.testing.assert_allclose(block.power_w[0][1:], [20, 60])


def test_causal_moving_average():
    np.testing.assert_allclose(moving_average_causal(np.array([1, 2, 3, 4]), 3), [1, 1.5, 2, 3])
