from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ProcessedBlock:
    time_s: np.ndarray
    voltage_v: np.ndarray
    current1_a: np.ndarray
    current2_a: np.ndarray
    total_current_a: np.ndarray
    total_power_w: np.ndarray


@dataclass
class PowerProcessor:
    sample_rate_hz: float
    voltage_scale: float
    current_scale: float
    voltage_delay_samples: int = 0
    current_delay_samples: int = 0
    power_average_samples: int = 10
    _sample_index: int = 0
    _v_line: deque[float] | None = field(default=None, init=False)
    _i1_line: deque[float] | None = field(default=None, init=False)
    _i2_line: deque[float] | None = field(default=None, init=False)
    _power_prev: deque[float] = field(default_factory=deque, init=False)

    def __post_init__(self) -> None:
        avg = max(int(self.power_average_samples) - 1, 0)
        self._power_prev = deque(maxlen=avg)
        offset = self.align_offset
        if offset > 0:
            self._v_line = deque(maxlen=offset + 1)
        elif offset < 0:
            need = -offset + 1
            self._i1_line = deque(maxlen=need)
            self._i2_line = deque(maxlen=need)

    @property
    def sample_index(self) -> int:
        return self._sample_index

    @property
    def align_offset(self) -> int:
        return int(self.voltage_delay_samples) - int(self.current_delay_samples)

    def process(self, raw_voltage, raw_current1, raw_current2) -> ProcessedBlock:
        raw_v = np.asarray(raw_voltage, dtype=float)
        raw_i1 = np.asarray(raw_current1, dtype=float)
        raw_i2 = np.asarray(raw_current2, dtype=float)
        n = min(raw_v.size, raw_i1.size, raw_i2.size)
        raw_v = raw_v[:n]
        raw_i1 = raw_i1[:n]
        raw_i2 = raw_i2[:n]

        v_scaled = raw_v * float(self.voltage_scale)
        i1_scaled = raw_i1 * float(self.current_scale)
        i2_scaled = raw_i2 * float(self.current_scale)

        idx = np.arange(self._sample_index, self._sample_index + n, dtype=np.int64)
        t_nom = idx.astype(float) / float(self.sample_rate_hz)
        t_v = t_nom + (float(self.voltage_delay_samples) / float(self.sample_rate_hz))
        t_i = t_nom + (float(self.current_delay_samples) / float(self.sample_rate_hz))

        v_aligned = np.full(n, np.nan, dtype=float)
        i1_aligned = np.full(n, np.nan, dtype=float)
        i2_aligned = np.full(n, np.nan, dtype=float)
        offset = self.align_offset

        if offset == 0:
            v_aligned[:] = v_scaled
            i1_aligned[:] = i1_scaled
            i2_aligned[:] = i2_scaled
            t_common = t_nom
        elif offset > 0:
            t_common = t_i
            assert self._v_line is not None
            for k in range(n):
                self._v_line.append(float(v_scaled[k]))
                if len(self._v_line) > offset:
                    v_aligned[k] = float(self._v_line[0])
                    i1_aligned[k] = float(i1_scaled[k])
                    i2_aligned[k] = float(i2_scaled[k])
        else:
            t_common = t_v
            need = -offset
            assert self._i1_line is not None and self._i2_line is not None
            for k in range(n):
                self._i1_line.append(float(i1_scaled[k]))
                self._i2_line.append(float(i2_scaled[k]))
                if len(self._i1_line) > need:
                    i1_aligned[k] = float(self._i1_line[0])
                    i2_aligned[k] = float(self._i2_line[0])
                    v_aligned[k] = float(v_scaled[k])

        i_total = i1_aligned + i2_aligned
        p_inst = v_aligned * i_total
        power = self._smooth_power(p_inst)
        self._sample_index += n
        return ProcessedBlock(
            time_s=t_common,
            voltage_v=v_aligned,
            current1_a=i1_aligned,
            current2_a=i2_aligned,
            total_current_a=i_total,
            total_power_w=power,
        )

    def _smooth_power(self, power: np.ndarray) -> np.ndarray:
        n = int(self.power_average_samples)
        if n <= 1 or power.size == 0:
            return power.astype(float, copy=True)

        prev = np.fromiter(self._power_prev, dtype=float)
        values = np.concatenate([prev, power]) if prev.size else power
        smoothed = moving_average_causal_ignore_nan(values, n)
        keep = n - 1
        self._power_prev.clear()
        if keep > 0:
            self._power_prev.extend(values[-keep:].tolist())
        return smoothed[-power.size :]


def moving_average_causal(x: np.ndarray, n: int) -> np.ndarray:
    n = int(n)
    x = np.asarray(x, dtype=float)
    if n <= 1 or x.size == 0:
        return x.copy()
    csum = np.cumsum(x)
    out = np.empty_like(x, dtype=float)
    for idx in range(x.size):
        start = max(0, idx - n + 1)
        total = csum[idx] - (csum[start - 1] if start > 0 else 0.0)
        out[idx] = total / float(idx - start + 1)
    return out


def moving_average_causal_ignore_nan(x: np.ndarray, n: int) -> np.ndarray:
    n = int(n)
    x = np.asarray(x, dtype=float)
    if n <= 1 or x.size == 0:
        return x.copy()
    out = np.empty_like(x, dtype=float)
    for idx in range(x.size):
        start = max(0, idx - n + 1)
        window = x[start : idx + 1]
        valid = window[~np.isnan(window)]
        out[idx] = np.mean(valid) if valid.size else np.nan
    return out
