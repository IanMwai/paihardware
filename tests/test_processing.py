import numpy as np

from gpu_power_monitor.processing import PowerProcessor, moving_average_causal


def test_scaling_and_power_no_delay_no_average():
    processor = PowerProcessor(
        sample_rate_hz=10,
        voltage_scale=2,
        current_scale=10,
        power_average_samples=1,
    )
    block = processor.process([1, 2], [0.1, 0.2], [0.3, 0.4])
    np.testing.assert_allclose(block.voltage_v, [2, 4])
    np.testing.assert_allclose(block.current1_a, [1, 2])
    np.testing.assert_allclose(block.current2_a, [3, 4])
    np.testing.assert_allclose(block.total_power_w, [8, 24])


def test_moving_average_continues_across_blocks():
    processor = PowerProcessor(
        sample_rate_hz=10,
        voltage_scale=1,
        current_scale=1,
        power_average_samples=3,
    )
    first = processor.process([1, 1], [1, 2], [0, 0])
    second = processor.process([1, 1], [3, 4], [0, 0])
    np.testing.assert_allclose(first.total_power_w, [1, 1.5])
    np.testing.assert_allclose(second.total_power_w, [2, 3])


def test_positive_voltage_delay_alignment():
    processor = PowerProcessor(
        sample_rate_hz=10,
        voltage_scale=1,
        current_scale=1,
        voltage_delay_samples=1,
        current_delay_samples=0,
        power_average_samples=1,
    )
    block = processor.process([10, 20, 30], [1, 1, 1], [0, 0, 0])
    assert np.isnan(block.voltage_v[0])
    np.testing.assert_allclose(block.voltage_v[1:], [10, 20])
    np.testing.assert_allclose(block.total_power_w[1:], [10, 20])


def test_negative_current_delay_alignment():
    processor = PowerProcessor(
        sample_rate_hz=10,
        voltage_scale=1,
        current_scale=1,
        voltage_delay_samples=0,
        current_delay_samples=1,
        power_average_samples=1,
    )
    block = processor.process([10, 20, 30], [1, 2, 3], [0, 0, 0])
    assert np.isnan(block.current1_a[0])
    np.testing.assert_allclose(block.current1_a[1:], [1, 2])
    np.testing.assert_allclose(block.total_power_w[1:], [20, 60])


def test_causal_moving_average():
    np.testing.assert_allclose(moving_average_causal(np.array([1, 2, 3, 4]), 3), [1, 1.5, 2, 3])
