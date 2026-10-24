from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import AcquisitionConfig
from .utils import STOP_FILE_NAME, atomic_replace_text, chunk_paths, open_chunk, utc_now_iso


MANIFEST_NAME = "manifest.json"


@dataclass
class RunManifest:
    measurement_name: str
    run_id: str
    status: str = "running"
    # "archive" runs are destined for the cluster archive; "test" runs are
    # local scratch (never pushed, deletable at will). Manifests from before
    # this field existed count as "archive".
    run_kind: str = "archive"
    archive_status: str = "local_only"
    # Who started the run (the recording account), when
    # known. Informational; never used for permissions.
    operator: str | None = None
    # Dataset keys: the machine the GPUs live in, the GPUs themselves (from
    # the NVML info fetched at the end of the run), and the workload's own
    # description of itself (<run>/workload/meta.json, written by whatever
    # `pai hardware ... -- <command>` ran).
    machine: str | None = None
    gpus: list[dict[str, Any]] = field(default_factory=list)
    workload: dict[str, Any] | None = None
    manifest_version: int = 2
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
        run_kind="test" if config.test_run else "archive",
        operator=getattr(config, "operator", "") or None,
        machine=getattr(config, "machine", "") or None,
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


def finalize_manifest(run_dir: Path, end_time_s: float, *, stats: dict | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    data = load_manifest(run_dir)
    files = build_file_inventory(run_dir)
    stats = compute_run_stats(run_dir) if stats is None else stats
    from .chunks import ready_records
    notices = [dict(chunk=r["logical_name"], message=w) for r in ready_records(run_dir) for w in r.get("warnings", [])]
    data["storage_warnings"] = notices
    workload = load_workload_meta(run_dir)
    if workload is not None:
        data["workload"] = workload
    gpus = parse_gpu_info(run_dir / "nvml" / "gamma_info.txt")
    if gpus:
        data["gpus"] = gpus
    data.setdefault("manifest_version", 2)
    data.update(
        {
            "status": "completed",
            # Test runs stay local: completing one must not arm archiving.
            "archive_status": "local_only" if data.get("run_kind") == "test" else "ready_to_archive",
            "ended_at": utc_now_iso(),
            "duration_s": float(end_time_s),
            "files": files,
            "stats": stats,
        }
    )
    save_manifest(run_dir, data)
    return data


def refresh_file_inventory(run_dir: Path) -> dict[str, Any]:
    """Rebuild the checksummed file list (e.g. after a late NVML fetch added
    files to an already-finalized run)."""
    run_dir = Path(run_dir)
    data = load_manifest(run_dir)
    data["files"] = build_file_inventory(run_dir)
    save_manifest(run_dir, data)
    return data


def build_file_inventory(run_dir: Path, *, deadline: float | None = None) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    from .chunks import inventory_paths, digest, is_chunk
    files = []
    for path in inventory_paths(run_dir):
        rel = path.relative_to(run_dir).as_posix()
        files.append(
            {
                "path": rel,
                "size_bytes": path.stat().st_size,
                "sha256": (digest(path) if deadline is None and path.parent == run_dir and is_chunk(path)
                           else sha256_file(path, deadline=deadline)),
            }
        )
    return files


def sha256_file(path: Path, block_size: int = 1024 * 1024, *, deadline: float | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("File verification deadline exceeded")
            block = handle.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def compute_run_stats(run_dir: Path) -> dict[str, float | None]:
    # Streaming accumulation keeps memory bounded for day-long 10 kHz runs.
    count = 0
    total = energy = 0.0
    peak = None
    previous = None
    for csv_path in chunk_paths(run_dir):
        with open_chunk(csv_path) as handle:
            for row in csv.DictReader(handle):
                try:
                    p, t = float(row["total_power_w"]), float(row["time_s"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not np.isfinite(p) or not np.isfinite(t):
                    continue
                count += 1
                total += p
                peak = p if peak is None else max(peak, p)
                if previous is not None:
                    old_t, old_p = previous
                    energy += (t-old_t) * (p+old_p) / 2
                previous = (t, p)
    return {"average_power_w": total/count if count else None,
            "peak_power_w": peak, "energy_j": energy if count else None,
            "sample_count": count}


WORKLOAD_META_NAME = "meta.json"


def load_workload_meta(run_dir: Path) -> dict[str, Any] | None:
    """The workload's self-description, if it wrote <run>/workload/meta.json."""
    path = Path(run_dir) / "workload" / WORKLOAD_META_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def parse_gpu_info(path: Path) -> list[dict[str, Any]]:
    """GPUs from the `nvidia-smi --query-gpu=index,name,driver_version,
    vbios_version,power.limit --format=csv` block in nvml/gamma_info.txt."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, OSError):
        return []
    gpus: list[dict[str, Any]] = []
    header: list[str] | None = None
    for line in text.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if cells[:2] == ["index", "name"]:
            header = [c.split(" ")[0] for c in cells]
            continue
        if header and len(cells) == len(header) and cells[0].isdigit():
            row = dict(zip(header, cells))
            entry: dict[str, Any] = {"index": int(row["index"]), "name": row["name"]}
            for key in ("driver_version", "vbios_version"):
                if row.get(key):
                    entry[key] = row[key]
            limit = row.get("power.limit", "")
            m = re.search(r"[\d.]+", limit)
            if m:
                entry["power_limit_w"] = float(m.group())
            gpus.append(entry)
        elif header and not cells[0].isdigit():
            header = None
    return gpus
