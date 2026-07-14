from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any

from .manifest import load_manifest, save_manifest, sha256_file
from .utils import utc_now_iso


# archive_status values that mean the run's data is on (or on its way to)
# the cluster, so the local copy is no longer the only one.
_ON_CLUSTER = {"transfer_pending", "archived_verified", "cleanup_eligible"}


def run_kind(data: dict[str, Any]) -> str:
    """A run's kind; manifests from before the field existed are archive runs."""
    return data.get("run_kind", "archive")


def set_run_kind(run_dir: Path, kind: str) -> dict[str, Any]:
    """Reclassify a run as a test run (local scratch) or an archive run.

    Promoting a completed test run re-arms archiving. Demotion is refused once
    the run is on (or on its way to) the cluster: the archive is append-only,
    so the local copy must keep its archived bookkeeping.
    """
    if kind not in ("archive", "test"):
        raise ValueError(f"Unknown run kind: {kind!r} (expected 'archive' or 'test')")
    run_dir = Path(run_dir)
    data = load_manifest(run_dir)
    if run_kind(data) == kind:
        return data
    if kind == "test" and data.get("archive_status") in _ON_CLUSTER:
        raise RuntimeError(
            f"'{run_dir.name}' is already on (or on its way to) the cluster archive - "
            "it cannot become a test run"
        )
    data["run_kind"] = kind
    if kind == "test":
        data["archive_status"] = "local_only"
    elif data.get("status") == "completed":
        data["archive_status"] = "ready_to_archive"
    save_manifest(run_dir, data)
    return data


def delete_test_run(run_dir: Path) -> None:
    """Delete a local test run outright. Refuses anything archive-bound."""
    run_dir = Path(run_dir)
    data = load_manifest(run_dir)
    if run_kind(data) != "test":
        raise RuntimeError(
            f"'{run_dir.name}' is not a test run - archive-bound runs are only deleted by "
            '`pai archive cleanup` after they are archived (or demote it first: pai runs demote "<run>")'
        )
    if data.get("status") == "running":
        raise RuntimeError(f"'{run_dir.name}' is still in progress - stop it before deleting")
    shutil.rmtree(run_dir)


def archive_destination(run_dir: Path, archive_root: Path) -> Path:
    return Path(archive_root) / Path(run_dir).name


def copy_run(run_dir: Path, archive_root: Path) -> Path:
    run_dir = Path(run_dir)
    data = load_manifest(run_dir)
    if run_kind(data) == "test":
        raise RuntimeError(
            f"'{run_dir.name}' is a test run (local only) - promote it first: "
            f'pai runs promote "{run_dir.name}"'
        )
    # On Windows a POSIX cluster path like /n/holylabs/... resolves
    # drive-relative (C:\n\holylabs\...), so an unguarded copytree would
    # silently "archive" to the local disk and mark the run verified. A real
    # mounted archive root exists (or at least its parent does).
    archive_root = Path(archive_root)
    if not archive_root.is_dir() and not archive_root.parent.is_dir():
        raise FileNotFoundError(
            f"Archive root is not mounted on this machine: {archive_root} — "
            "use `pai archive push` (Globus) instead of `copy`"
        )
    dest = archive_destination(run_dir, archive_root)
    if dest.exists():
        raise FileExistsError(f"Archive destination already exists: {dest}")
    shutil.copytree(run_dir, dest, ignore=shutil.ignore_patterns("latest.npz", "latest_status.json"))
    verify_archive(run_dir, dest)
    data["archive_status"] = "archived_verified"
    data["archive"] = {
        "destination": str(dest),
        "verified_at": utc_now_iso(),
    }
    save_manifest(run_dir, data)
    save_manifest(dest, data)
    _write_live_state(run_dir, "ARCHIVED")
    _write_live_state(dest, "ARCHIVED")
    return dest


def mark_archived(run_dir: Path, destination: Path) -> str:
    """Record an externally performed archive (e.g. a Globus transfer).

    Mirrors :func:`copy_run`'s manifest bookkeeping without copying anything,
    so ``cleanup_local`` can later reclaim the local run. The destination is a
    cluster path, so it is stored in POSIX form (Windows ``Path`` would
    otherwise flip ``/n/holylabs/...`` to backslashes).
    """
    run_dir = Path(run_dir)
    dest = Path(destination).as_posix()
    data = load_manifest(run_dir)
    data["archive_status"] = "archived_verified"
    data["archive"] = {
        **data.get("archive", {}),  # keep e.g. the Globus task_id
        "destination": dest,
        "verified_at": utc_now_iso(),
    }
    save_manifest(run_dir, data)
    _write_live_state(run_dir, "ARCHIVED")
    return dest


def verify_archive(run_dir: Path, archived_run_dir: Path | None = None) -> list[str]:
    run_dir = Path(run_dir)
    data = load_manifest(run_dir)
    archived_run_dir = Path(archived_run_dir) if archived_run_dir else Path(data.get("archive", {}).get("destination", ""))
    if not archived_run_dir:
        raise ValueError("No archive destination provided and manifest has no archive.destination")
    failures = []
    for item in data.get("files", []):
        rel = item["path"]
        archived_path = archived_run_dir / rel
        if not archived_path.exists():
            failures.append(f"missing: {rel}")
            continue
        digest = sha256_file(archived_path)
        if digest != item["sha256"]:
            failures.append(f"checksum mismatch: {rel}")
    if failures:
        local = load_manifest(run_dir)
        local["archive_status"] = "archive_error"
        local["archive"] = {
            **local.get("archive", {}),
            "destination": str(archived_run_dir),
            "last_error": failures,
            "checked_at": utc_now_iso(),
        }
        save_manifest(run_dir, local)
        raise RuntimeError("Archive verification failed: " + "; ".join(failures))
    return failures


def archive_status(run_dir: Path) -> dict[str, Any]:
    data = load_manifest(run_dir)
    return {
        "run_id": data.get("run_id"),
        "status": data.get("status"),
        "run_kind": run_kind(data),
        "archive_status": data.get("archive_status", "local_only"),
        "archive": data.get("archive", {}),
        "stats": data.get("stats", {}),
    }


def cleanup_local(run_dir: Path, retention_days: int, *, dry_run: bool = True) -> bool:
    run_dir = Path(run_dir)
    data = load_manifest(run_dir)
    if data.get("archive_status") != "archived_verified":
        return False
    ended_at = data.get("ended_at")
    if not ended_at:
        return False
    ended = dt.datetime.fromisoformat(ended_at)
    if ended.tzinfo is None:
        ended = ended.replace(tzinfo=dt.timezone.utc)
    age = dt.datetime.now(dt.timezone.utc) - ended
    if age.days < int(retention_days):
        return False
    data["archive_status"] = "cleanup_eligible"
    save_manifest(run_dir, data)
    if dry_run:
        return True
    shutil.rmtree(run_dir)
    return True


def _write_live_state(run_dir: Path, state: str) -> None:
    status_path = Path(run_dir) / "latest_status.json"
    if not status_path.exists():
        return
    data = json.loads(status_path.read_text(encoding="utf-8"))
    data["state"] = state
    data["generated_at_epoch"] = dt.datetime.now(dt.timezone.utc).timestamp()
    status_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
