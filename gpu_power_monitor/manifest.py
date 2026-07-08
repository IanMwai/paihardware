from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import AcquisitionConfig
from .utils import atomic_replace_text, utc_now_iso


MANIFEST_NAME = "manifest.json"


@dataclass
class RunManifest:
    measurement_name: str
    run_id: str
    status: str = "running"
    archive_status: str = "local_only"
    created_at: str = field(default_factory=utc_now_iso)
    started_at: str | None = None
    ended_at: str | None = None
    duration_s: float | None = None
    sample_rate_hz: float | None = None
    channels: dict[str, Any] = field(default_factory=dict)
    scaling: dict[str, Any] = field(default_factory=dict)
    processing: dict[str, Any] = field(default_factory=dict)
    files: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    archive: dict[str, Any] = field(default_factory=dict)
    sync: dict[str, Any] = field(default_factory=lambda: {"trigger_time": None, "workload": None})


def create_manifest(run_dir: Path, config: AcquisitionConfig) -> RunManifest:
    manifest = RunManifest(
        measurement_name=config.measurement_name,
        run_id=Path(run_dir).name,
        started_at=utc_now_iso(),
        sample_rate_hz=float(config.sample_rate_hz),
        channels=asdict(config.channels),
        scaling=asdict(config.scaling),
        processing=asdict(config.processing),
    )
    save_manifest(run_dir, manifest)
    return manifest


def save_manifest(run_dir: Path, manifest: RunManifest | dict[str, Any]) -> None:
    data = asdict(manifest) if isinstance(manifest, RunManifest) else manifest
    atomic_replace_text(Path(run_dir) / MANIFEST_NAME, json.dumps(data, indent=2, sort_keys=True))


def load_manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((Path(run_dir) / MANIFEST_NAME).read_text(encoding="utf-8"))


def update_manifest(run_dir: Path, **updates) -> dict[str, Any]:
    data = load_manifest(run_dir)
    data.update(updates)
    save_manifest(run_dir, data)
    return data


def finalize_manifest(run_dir: Path, end_time_s: float) -> dict[str, Any]:
    run_dir = Path(run_dir)
    data = load_manifest(run_dir)
    files = build_file_inventory(run_dir)
    stats = compute_run_stats(run_dir)
    data.update(
        {
            "status": "completed",
            "archive_status": "ready_to_archive",
            "ended_at": utc_now_iso(),
            "duration_s": float(end_time_s),
            "files": files,
            "stats": stats,
        }
    )
    save_manifest(run_dir, data)
    return data


def build_file_inventory(run_dir: Path) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    files = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(run_dir).as_posix()
        if rel in {MANIFEST_NAME, "latest.npz", "latest_status.json"}:
            continue
        files.append(
            {
                "path": rel,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return files


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def compute_run_stats(run_dir: Path) -> dict[str, float | None]:
    powers = []
    times = []
    for csv_path in sorted(Path(run_dir).glob("*.csv")):
        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    p = float(row["total_power_w"])
                    t = float(row["time_s"])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(p):
                    powers.append(p)
                    times.append(t)
    if not powers:
        return {
            "average_power_w": None,
            "peak_power_w": None,
            "energy_j": None,
            "sample_count": 0,
        }
    p_arr = np.asarray(powers, dtype=float)
    t_arr = np.asarray(times, dtype=float)
    # np.trapz was renamed to np.trapezoid in NumPy 2.0 and removed thereafter.
    trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    energy = float(trapezoid(p_arr, t_arr)) if p_arr.size > 1 else 0.0
    return {
        "average_power_w": float(np.mean(p_arr)),
        "peak_power_w": float(np.max(p_arr)),
        "energy_j": energy,
        "sample_count": int(p_arr.size),
    }
