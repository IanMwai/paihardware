from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .config import GpuChannel


@dataclass
class ProcessedBlock:
    """One processed acquisition block: per-GPU arrays plus totals.

    voltage_v / current_a / power_w are indexed like ``labels`` (one array per
    GPU). total_power_w sums the per-GPU power sample-by-sample.
    """

    time_s: np.ndarray
    labels: list[str]
    voltage_v: list[np.ndarray]
    current_a: list[np.ndarray]
    power_w: list[np.ndarray]
    total_power_w: np.ndarray

    @property
    def total_current_a(self) -> np.ndarray:
        if not self.current_a:
            return np.empty(0, dtype=float)
        return np.sum(np.vstack(self.current_a), axis=0)


@dataclass
class PowerProcessor:
    """Scales raw DAQ samples and computes per-GPU power.

    Per GPU: voltage uses its calibrated divider ratio, current uses the shared
    shunt scale times the GPU's wiring sign. The voltage/current delay
    compensation (integer samples, shared across GPUs) and the causal power
    moving average carry over from the single-GPU processor, applied per GPU.
    """

    sample_rate_hz: float
    gpus: Sequence[GpuChannel]
    current_scale: float
    voltage_delay_samples: int = 0
    current_delay_samples: int = 0
    power_average_samples: int = 10
    _sample_index: int = 0
    _v_lines: list[deque[float]] | None = field(default=None, init=False)
    _i_lines: list[deque[float]] | None = field(default=None, init=False)
    _power_prev: list[deque[float]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        avg = max(int(self.power_average_samples) - 1, 0)
        self._power_prev = [deque(maxlen=avg) for _ in self.gpus]
        offset = self.align_offset
        if offset > 0:
            self._v_lines = [deque(maxlen=offset + 1) for _ in self.gpus]
        elif offset < 0:
            self._i_lines = [deque(maxlen=-offset + 1) for _ in self.gpus]

    @property
    def sample_index(self) -> int:
        return self._sample_index

    @property
    def labels(self) -> list[str]:
        return [gpu.label for gpu in self.gpus]

    @property
    def align_offset(self) -> int:
        return int(self.voltage_delay_samples) - int(self.current_delay_samples)

    def process(self, raw_voltages: Sequence, raw_currents: Sequence) -> ProcessedBlock:
        if len(raw_voltages) != len(self.gpus) or len(raw_currents) != len(self.gpus):
            raise ValueError(f"Expected raw data for {len(self.gpus)} GPUs")
        raw_v = [np.asarray(values, dtype=float) for values in raw_voltages]
        raw_i = [np.asarray(values, dtype=float) for values in raw_currents]
        n = min((arr.size for arr in raw_v + raw_i), default=0)
        raw_v = [arr[:n] for arr in raw_v]
        raw_i = [arr[:n] for arr in raw_i]

        idx = np.arange(self._sample_index, self._sample_index + n, dtype=np.int64)
        t_nom = idx.astype(float) / float(self.sample_rate_hz)
        offset = self.align_offset
        if offset > 0:
            t_common = t_nom + (float(self.current_delay_samples) / float(self.sample_rate_hz))
        elif offset < 0:
            t_common = t_nom + (float(self.voltage_delay_samples) / float(self.sample_rate_hz))
        else:
            t_common = t_nom

        voltages: list[np.ndarray] = []
        currents: list[np.ndarray] = []
        powers: list[np.ndarray] = []
        for g, gpu in enumerate(self.gpus):
            v_scaled = raw_v[g] * float(gpu.voltage_scale)
            i_scaled = raw_i[g] * float(self.current_scale) * float(gpu.current_sign)
            v_aligned, i_aligned = self._align(g, v_scaled, i_scaled, n)
            p_inst = v_aligned * i_aligned
            voltages.append(v_aligned)
            currents.append(i_aligned)
            powers.append(self._smooth_power(g, p_inst))

        total = np.sum(np.vstack(powers), axis=0) if powers and n else np.empty(0, dtype=float)
        self._sample_index += n
        return ProcessedBlock(
            time_s=t_common,
            labels=self.labels,
            voltage_v=voltages,
            current_a=currents,
            power_w=powers,
            total_power_w=total,
        )

    def _align(
        self, g: int, v_scaled: np.ndarray, i_scaled: np.ndarray, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        offset = self.align_offset
        if offset == 0:
            return v_scaled.copy(), i_scaled.copy()
        v_aligned = np.full(n, np.nan, dtype=float)
        i_aligned = np.full(n, np.nan, dtype=float)
        if offset > 0:
            assert self._v_lines is not None
            line = self._v_lines[g]
            for k in range(n):
                line.append(float(v_scaled[k]))
                if len(line) > offset:
                    v_aligned[k] = float(line[0])
                    i_aligned[k] = float(i_scaled[k])
        else:
            assert self._i_lines is not None
            line = self._i_lines[g]
            need = -offset
            for k in range(n):
                line.append(float(i_scaled[k]))
                if len(line) > need:
                    i_aligned[k] = float(line[0])
                    v_aligned[k] = float(v_scaled[k])
        return v_aligned, i_aligned

    def _smooth_power(self, g: int, power: np.ndarray) -> np.ndarray:
        n = int(self.power_average_samples)
        if n <= 1 or power.size == 0:
            return power.astype(float, copy=True)

        prev_deque = self._power_prev[g]
        prev = np.fromiter(prev_deque, dtype=float)
        values = np.concatenate([prev, power]) if prev.size else power
        smoothed = moving_average_causal_ignore_nan(values, n)
        keep = n - 1
        prev_deque.clear()
        if keep > 0:
            prev_deque.extend(values[-keep:].tolist())
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
