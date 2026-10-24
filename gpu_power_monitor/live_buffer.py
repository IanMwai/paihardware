from __future__ import annotations

import io
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .processing import ProcessedBlock
from .utils import atomic_replace_bytes, atomic_replace_text


@dataclass(frozen=True)
class LiveSnapshot:
    state: str
    generated_at_epoch: float
    latest_sample_time_s: float | None
    sample_count: int
    run_avg_power_w: float | None = None
    run_peak_power_w: float | None = None
    energy_j: float | None = None


class LiveBuffer:
    def __init__(
        self,
        run_dir: Path,
        sample_rate_hz: float,
        window_sec: float,
        labels: list[str],
        max_points: int = 2500,
        publish_interval_sec: float = 0.5,
    ):
        self.run_dir = Path(run_dir)
        self.sample_rate_hz = float(sample_rate_hz)
        self.window_sec = float(window_sec)
        self.labels = list(labels)
        self.max_points = int(max_points)
        if self.max_points < 1:
            raise ValueError("max_points must be positive")
        self.publish_interval_sec = max(0.0, float(publish_interval_sec))
        self._next_publish = 0.0
        self._sample_cursor = 0
        self._latest_values = None
        # Decimate only the preview before buffering; full-rate data goes to
        # ChunkWriter independently. Never rebuild millions of Python floats
        # just to publish a 2,500-point dashboard snapshot.
        window_samples = max(1, int(self.sample_rate_hz * self.window_sec))
        self._stride = max(1, int(np.ceil(window_samples / self.max_points)))
        points = max(1, int(np.ceil(window_samples / self._stride)))
        self.time_s = deque(maxlen=points)
        self.voltage_v = [deque(maxlen=points) for _ in self.labels]
        self.current_a = [deque(maxlen=points) for _ in self.labels]
        self.power_w = [deque(maxlen=points) for _ in self.labels]
        self.total_power = deque(maxlen=points)
        self.status_path = self.run_dir / "latest_status.json"
        self.data_path = self.run_dir / "latest.npz"
        # whole-run accumulators (not limited to the on-screen window)
        self._run_sum = 0.0
        self._run_count = 0
        self._run_peak = float("-inf")
        self._energy_j = 0.0

    def append(self, block: ProcessedBlock, state: str = "LIVE") -> None:
        n = block.time_s.size
        if not n:
            return
        offset = (-self._sample_cursor) % self._stride
        selection = slice(offset, None, self._stride)
        self._sample_cursor += n
        self.time_s.extend(block.time_s[selection].tolist())
        latest = {"time_s": float(block.time_s[-1]), "total_power_w": float(block.total_power_w[-1])}
        for g, label in enumerate(self.labels):
            self.voltage_v[g].extend(block.voltage_v[g][selection].tolist())
            self.current_a[g].extend(block.current_a[g][selection].tolist())
            self.power_w[g].extend(block.power_w[g][selection].tolist())
            latest[f"{label.lower()}_voltage_v"] = float(block.voltage_v[g][-1])
            latest[f"{label.lower()}_current_a"] = float(block.current_a[g][-1])
            latest[f"{label.lower()}_power_w"] = float(block.power_w[g][-1])
        self.total_power.extend(block.total_power_w[selection].tolist())
        self._latest_values = latest
        self._accumulate_run_stats(block)
        if time.monotonic() >= self._next_publish:
            self.write_latest(state=state)
            self._next_publish = time.monotonic() + self.publish_interval_sec

    def _accumulate_run_stats(self, block: ProcessedBlock) -> None:
        power = np.asarray(block.total_power_w, dtype=float)
        finite = power[np.isfinite(power)]
        if finite.size:
            self._run_sum += float(finite.sum())
            self._run_count += int(finite.size)
            self._run_peak = max(self._run_peak, float(finite.max()))
            # energy increment ≈ Σ p · Δt over the finite samples in this block
            self._energy_j += float(finite.sum()) / self.sample_rate_hz

    def write_latest(self, state: str) -> None:
        arrays = self._downsample()
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        atomic_replace_bytes(self.data_path, buffer.getvalue())
        latest = float(arrays["time_s"][-1]) if arrays["time_s"].size else None
        snapshot = LiveSnapshot(
            state=state,
            generated_at_epoch=time.time(),
            latest_sample_time_s=latest,
            sample_count=int(arrays["time_s"].size),
            run_avg_power_w=(self._run_sum / self._run_count) if self._run_count else None,
            run_peak_power_w=self._run_peak if self._run_count else None,
            energy_j=self._energy_j if self._run_count else None,
        )
        atomic_replace_text(self.status_path, json.dumps(snapshot.__dict__, indent=2))

    def mark_state(self, state: str) -> None:
        self.write_latest(state)

    def _downsample(self) -> dict[str, np.ndarray]:
        # npz/JSON keys mirror the CSV columns (gpu1_voltage_v, ...); the
        # dashboard discovers the GPUs by scanning for *_voltage_v keys.
        arrays = {"time_s": np.asarray(self.time_s, dtype=float)}
        for g, label in enumerate(self.labels):
            gpu = label.lower()
            arrays[f"{gpu}_voltage_v"] = np.asarray(self.voltage_v[g], dtype=float)
            arrays[f"{gpu}_current_a"] = np.asarray(self.current_a[g], dtype=float)
            arrays[f"{gpu}_power_w"] = np.asarray(self.power_w[g], dtype=float)
        arrays["total_power_w"] = np.asarray(self.total_power, dtype=float)
        if self._latest_values is not None and (
            not arrays["time_s"].size or arrays["time_s"][-1] != self._latest_values["time_s"]
        ):
            arrays = {key: np.append(value, self._latest_values[key]) for key, value in arrays.items()}
        n = arrays["time_s"].size
        if n <= self.max_points:
            return arrays
        idx = np.linspace(0, n - 1, self.max_points).astype(int)
        return {key: value[idx] for key, value in arrays.items()}


def read_live_status(run_dir: Path, stale_after_sec: float) -> LiveSnapshot:
    path = Path(run_dir) / "latest_status.json"
    if not path.exists():
        return LiveSnapshot("IDLE", time.time(), None, 0)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # In-place writes (Defender-blocked rename fallback) are not atomic, so
        # a reader may briefly catch a half-written file. Treat as stale.
        return LiveSnapshot("IDLE", time.time(), None, 0)
    state = str(data.get("state", "IDLE"))
    generated = float(data.get("generated_at_epoch", 0.0))
    if state == "LIVE" and time.time() - generated > float(stale_after_sec):
        # The writer stopped without marking an end state (crash, kill, or a
        # dashboard opened on an old run): the data on disk is real but old.
        state = "STALE"
    return LiveSnapshot(
        state=state,
        generated_at_epoch=generated,
        latest_sample_time_s=data.get("latest_sample_time_s"),
        sample_count=int(data.get("sample_count", 0)),
        run_avg_power_w=data.get("run_avg_power_w"),
        run_peak_power_w=data.get("run_peak_power_w"),
        energy_j=data.get("energy_j"),
    )
