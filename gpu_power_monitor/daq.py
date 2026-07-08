from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import ChannelConfig


class DaqSource(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def available_samples(self) -> int: ...

    def read(self, n_samples: int) -> tuple[list[float], list[float], list[float]]: ...


def ensure_channel_list(raw):
    if raw is None:
        return []
    if isinstance(raw, (float, int, np.floating, np.integer)):
        return [[float(raw)]]
    if isinstance(raw, list) and (
        len(raw) == 0 or isinstance(raw[0], (float, int, np.floating, np.integer))
    ):
        return [list(raw)]
    return [list(ch) for ch in raw]


class NIDaqSource:
    def __init__(self, channels: ChannelConfig, sample_rate_hz: float, chunk_size: int):
        try:
            import nidaqmx
            from nidaqmx.constants import AcquisitionType, Edge, TerminalConfiguration
        except ImportError as exc:
            raise RuntimeError(
                "nidaqmx is required for hardware acquisition. Use --simulate without NI hardware."
            ) from exc

        self._nidaqmx = nidaqmx
        self._AcquisitionType = AcquisitionType
        self._Edge = Edge
        self._TerminalConfiguration = TerminalConfiguration
        self.channels = channels
        self.sample_rate_hz = float(sample_rate_hz)
        self.chunk_size = int(chunk_size)
        self.task = None

    def start(self) -> None:
        task = self._nidaqmx.Task()
        terminal = self._terminal_config(self.channels.terminal)
        for physical in (
            self.channels.voltage_physical,
            self.channels.current1_physical,
            self.channels.current2_physical,
        ):
            task.ai_channels.add_ai_voltage_chan(
                physical,
                min_val=float(self.channels.min_v),
                max_val=float(self.channels.max_v),
                terminal_config=terminal,
            )
        task.timing.cfg_samp_clk_timing(
            rate=self.sample_rate_hz,
            source="",
            active_edge=self._Edge.RISING,
            sample_mode=self._AcquisitionType.CONTINUOUS,
            samps_per_chan=int(max(self.chunk_size * 200, int(self.sample_rate_hz) * 2)),
        )
        task.start()
        self.task = task

    def stop(self) -> None:
        if self.task is None:
            return
        try:
            self.task.stop()
        finally:
            self.task.close()
            self.task = None

    def available_samples(self) -> int:
        if self.task is None:
            return 0
        return int(self.task.in_stream.avail_samp_per_chan)

    def read(self, n_samples: int) -> tuple[list[float], list[float], list[float]]:
        if self.task is None:
            raise RuntimeError("DAQ task has not been started")
        raw = self.task.read(number_of_samples_per_channel=int(n_samples), timeout=0.0)
        channels = ensure_channel_list(raw)
        if len(channels) < 3:
            raise RuntimeError("Expected 3 channels from task.read()")
        n = min(len(channels[0]), len(channels[1]), len(channels[2]))
        return channels[0][:n], channels[1][:n], channels[2][:n]

    def _terminal_config(self, mode: str):
        mode = mode.strip().upper()
        if mode == "RSE":
            return self._TerminalConfiguration.RSE
        if mode == "DIFF":
            return self._TerminalConfiguration.DIFFERENTIAL
        if mode == "NRSE":
            return self._TerminalConfiguration.NRSE
        raise ValueError("terminal must be RSE, DIFF, or NRSE")


@dataclass
class SimulatedDaqSource:
    sample_rate_hz: float
    chunk_size: int = 1000
    _sample_index: int = 0
    _running: bool = False

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def available_samples(self) -> int:
        return int(self.chunk_size) if self._running else 0

    def read(self, n_samples: int) -> tuple[list[float], list[float], list[float]]:
        if not self._running:
            return [], [], []
        n = int(n_samples)
        idx = np.arange(self._sample_index, self._sample_index + n, dtype=float)
        t = idx / float(self.sample_rate_hz)
        raw_voltage = (12.1 + 0.08 * np.sin(2 * math.pi * 1.5 * t)) / 4.768
        raw_current1 = (4.0 + 1.2 * np.sin(2 * math.pi * 4.0 * t)) / 12.5
        raw_current2 = (3.0 + 0.9 * np.cos(2 * math.pi * 3.0 * t)) / 12.5
        self._sample_index += n
        return raw_voltage.tolist(), raw_current1.tolist(), raw_current2.tolist()
