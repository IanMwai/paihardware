from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from .config import DEFAULT_GPUS, ChannelConfig, GpuChannel


class DaqSource(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def available_samples(self) -> int: ...

    def read(self, n_samples: int) -> tuple[list[list[float]], list[list[float]]]:
        """Per-GPU raw samples: (voltages, currents), each one list per GPU."""
        ...


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
        try:
            terminal = self._terminal_config(self.channels.terminal)
            # Interleaved V/I pairs per GPU: GPU1 V, GPU1 I, GPU2 V, GPU2 I, ...
            # read() unpacks channels back out by this order.
            for gpu in self.channels.gpus:
                for physical in (
                    gpu.voltage_physical(self.channels.device),
                    gpu.current_physical(self.channels.device),
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
                # Host-side buffer. DAQmx raises -200279 ("not able to keep up") the
                # moment the reader falls this far behind. Leave headroom for
                # scheduling or storage stalls.
                samps_per_chan=int(max(self.chunk_size * 200, int(self.sample_rate_hz) * 60)),
            )
            task.start()
        except BaseException:
            task.close()
            raise
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

    @property
    def buffer_capacity_samples(self) -> int:
        return int(max(self.chunk_size * 200, int(self.sample_rate_hz) * 60))

    def read(self, n_samples: int) -> tuple[list[list[float]], list[list[float]]]:
        if self.task is None:
            raise RuntimeError("DAQ task has not been started")
        raw = self.task.read(number_of_samples_per_channel=int(n_samples), timeout=0.0)
        expected = 2 * len(self.channels.gpus)
        if int(n_samples) == 1 and isinstance(raw, list) and len(raw) == expected and all(np.isscalar(v) for v in raw):
            channels = [[v] for v in raw]
        else:
            channels = ensure_channel_list(raw)
        if len(channels) < expected:
            raise RuntimeError(f"Expected {expected} channels from task.read(), got {len(channels)}")
        if len(channels) != expected or any(len(ch) != int(n_samples) for ch in channels):
            raise RuntimeError("DAQ returned mismatched sample arrays; refusing to truncate data")
        n = int(n_samples)
        voltages = [channels[2 * g][:n] for g in range(len(self.channels.gpus))]
        currents = [channels[2 * g + 1][:n] for g in range(len(self.channels.gpus))]
        return voltages, currents


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
    """Synthesizes plausible raw (pre-scaling) samples for each configured GPU.

    Values are divided/signed so that after PowerProcessor's scaling each GPU
    shows ~12 V and a distinct current waveform — including the sign flips on
    GPU2-4, so simulation exercises the same math as hardware.
    """

    sample_rate_hz: float
    chunk_size: int = 1000
    gpus: Sequence[GpuChannel] = DEFAULT_GPUS
    current_scale: float = 12.5
    _sample_index: int = 0
    _running: bool = field(default=False, init=False)

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def available_samples(self) -> int:
        return int(self.chunk_size) if self._running else 0

    def read(self, n_samples: int) -> tuple[list[list[float]], list[list[float]]]:
        if not self._running:
            return [], []
        n = int(n_samples)
        idx = np.arange(self._sample_index, self._sample_index + n, dtype=float)
        t = idx / float(self.sample_rate_hz)
        voltages = []
        currents = []
        for g, gpu in enumerate(self.gpus):
            volts = 12.1 + 0.05 * np.sin(2 * math.pi * (1.0 + 0.5 * g) * t + g)
            amps = (3.0 + 1.5 * g) + 1.2 * np.sin(2 * math.pi * (2.0 + g) * t + 2 * g)
            voltages.append((volts / float(gpu.voltage_scale)).tolist())
            currents.append(
                (amps / (float(self.current_scale) * float(gpu.current_sign))).tolist()
            )
        self._sample_index += n
        return voltages, currents
