"""Push completed runs to the cluster archive via the Globus CLI.

The archive path is not mounted on the DAQ machine, so runs travel through
Globus (the local endpoint is Globus Connect Personal). Globus queues the
transfer on its own servers: if the cluster is down or this machine goes
offline, the task waits and retries until the configured deadline. A run's
``archive_status`` therefore moves ``ready_to_archive`` -> ``transfer_pending``
-> ``archived_verified`` (Globus checksum-verifies every file), with
``push_failed`` recorded on a fatal task error. Cleanup only ever touches
``archived_verified`` runs, so no failure path here can lose data.

One-time setup: ``pip install globus-cli``, ``globus login``, and both endpoint
UUIDs in ``configs/default.yaml`` (``storage.globus``).
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path

from .archive import mark_archived, run_kind
from .config import GlobusConfig
from .manifest import load_manifest, update_manifest
from .utils import utc_now_iso

# States a Globus task can report; anything not terminal counts as pending.
_TERMINAL_OK = {"SUCCEEDED"}
_TERMINAL_BAD = {"FAILED"}

# Globus ``nice_status`` codes for a transfer that is alive but not moving,
# mapped to what the operator should actually check. Substring-matched against
# the upper-cased code so minor wording variants still hit.
_STUCK_HINTS = {
    "GC_NOT_CONNECTED": (
        "Globus Connect Personal is not running on this machine - "
        "start it (tray icon) and the transfer resumes on its own"
    ),
    "CONNECTION_FAILED": (
        "an endpoint is unreachable - check that Globus Connect Personal is "
        "running here and that the cluster side is up"
    ),
    "CONNECT_FAILED": (
        "an endpoint is unreachable - check that Globus Connect Personal is "
        "running here and that the cluster side is up"
    ),
    "PERMISSION_DENIED": (
        "the archive destination refused the write - check permissions on the archive path"
    ),
    "NO_CREDENTIALS": "an endpoint needs (re)authentication - run: globus login",
    "EXPIRED": "credentials expired - run: globus login",
    "PAUSED": "the transfer was paused by an endpoint admin - it resumes when unpaused",
    "QUOTA": "the destination is out of quota - free up space on the archive side",
}


def stuck_hint(status: str | None, nice_status: str | None) -> str | None:
    """Human explanation of why a pending transfer is not moving, if known."""
    code = str(nice_status or "").upper()
    for key, hint in _STUCK_HINTS.items():
        if key in code:
            return hint
    if str(status or "").upper() == "INACTIVE":
        return "the transfer needs re-authentication - run: globus login"
    return None


class GlobusUnavailable(RuntimeError):
    """Globus pushes cannot run right now; the message says what to fix."""


def globus_unready_reason(cfg: GlobusConfig) -> str | None:
    """Why a push cannot be attempted, or None if it can."""
    if not cfg.local_endpoint_id or not cfg.remote_endpoint_id:
        return (
            "Globus endpoint IDs are not configured - set storage.globus."
            "local_endpoint_id and remote_endpoint_id in configs/default.yaml"
        )
    if shutil.which("globus") is None:
        return "globus-cli is not installed - run: pip install globus-cli"
    return None


def _gcp_path(path: Path) -> str:
    """``path`` in the local endpoint's namespace (GCP exposes ``C:\\`` as ``/C/``)."""
    path = path.resolve()
    if path.drive:
        return "/" + path.drive[0] + path.as_posix()[len(path.drive):]
    return path.as_posix()


def _run_globus(args: list[str]) -> dict:
    """Run a globus CLI command and return its parsed JSON output."""
    try:
        proc = subprocess.run(
            ["globus", *args, "--format", "json"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError as exc:
        raise GlobusUnavailable("globus-cli is not installed - run: pip install globus-cli") from exc
    except subprocess.TimeoutExpired as exc:
        raise GlobusUnavailable("globus CLI timed out - check the network connection") from exc
    if proc.returncode != 0:
        stderr = (proc.stderr or proc.stdout or "").strip()
        low = stderr.lower()
        if any(word in low for word in ("login", "credential", "unauthorized", "expired")):
            raise GlobusUnavailable(
                "Not logged in to Globus (or the session expired) - run: globus login"
            )
        if any(word in low for word in ("could not connect", "connection", "network", "timed out")):
            raise GlobusUnavailable(
                "Cannot reach the Globus service - check this machine's network "
                f"connection. CLI said: {stderr[:200]}"
            )
        if "endpoint" in low and ("not found" in low or "no such" in low):
            raise GlobusUnavailable(
                "Globus rejected an endpoint ID - check storage.globus.local_endpoint_id "
                f"and remote_endpoint_id in configs/default.yaml. CLI said: {stderr[:200]}"
            )
        raise GlobusUnavailable(f"globus CLI failed: {stderr[:500]}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise GlobusUnavailable(f"Unexpected globus CLI output: {proc.stdout[:200]}") from exc


def push_run(run_dir: Path, cfg: GlobusConfig, archive_root: Path) -> str:
    """Submit a transfer of ``run_dir`` to the archive; return the task id.

    Only manifest bookkeeping happens locally — the actual copy runs on
    Globus's side and is checksum-verified there (``--sync-level checksum``).
    """
    run_dir = Path(run_dir).resolve()
    reason = globus_unready_reason(cfg)
    if reason:
        raise GlobusUnavailable(reason)
    data = load_manifest(run_dir)
    if run_kind(data) == "test":
        raise RuntimeError(
            f"'{run_dir.name}' is a test run (local only) - promote it first: "
            f'pai runs promote "{run_dir.name}"'
        )
    if data.get("status") != "completed":
        raise RuntimeError(
            f"Run is not completed (status: {data.get('status')!r}) - only closed runs are archived"
        )
    if data.get("archive_status") == "archived_verified":
        raise RuntimeError("Run is already archived")

    dest = (Path(archive_root) / run_dir.name).as_posix()
    deadline = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=int(cfg.deadline_days))
    result = _run_globus(
        [
            "transfer",
            f"{cfg.local_endpoint_id}:{_gcp_path(run_dir)}",
            f"{cfg.remote_endpoint_id}:{dest}",
            "--recursive",
            "--sync-level", "checksum",
            "--label", f"gpu-power {run_dir.name}"[:128],
            "--exclude", "latest.npz",
            "--exclude", "latest_status.json",
            "--deadline", deadline.strftime("%Y-%m-%d %H:%M:%S"),
            "--notify", "failed,inactive",
        ]
    )
    task_id = str(result.get("task_id", ""))
    if not task_id:
        raise GlobusUnavailable(f"Globus accepted the transfer but returned no task id: {result}")
    update_manifest(
        run_dir,
        archive_status="transfer_pending",
        archive={
            "destination": dest,
            "task_id": task_id,
            "submitted_at": utc_now_iso(),
        },
    )
    return task_id


def check_task(run_dir: Path) -> str:
    """Refresh a pending transfer's status from Globus.

    Returns the run's (possibly updated) ``archive_status``. SUCCEEDED marks
    the run archived (Globus already checksum-verified it); FAILED records
    ``push_failed`` with the error; anything else stays ``transfer_pending``.
    """
    run_dir = Path(run_dir)
    data = load_manifest(run_dir)
    if data.get("archive_status") != "transfer_pending":
        return data.get("archive_status", "local_only")
    task_id = data.get("archive", {}).get("task_id")
    if not task_id:
        update_manifest(run_dir, archive_status="push_failed")
        return "push_failed"

    task = _run_globus(["task", "show", task_id])
    status = str(task.get("status", "")).upper()
    if status in _TERMINAL_OK:
        mark_archived(run_dir, data["archive"]["destination"])
        return "archived_verified"
    if status in _TERMINAL_BAD:
        detail = task.get("fatal_error") or task.get("nice_status_details") or task.get("nice_status")
        update_manifest(
            run_dir,
            archive_status="push_failed",
            archive={**data.get("archive", {}), "last_error": str(detail), "checked_at": utc_now_iso()},
        )
        return "push_failed"
    # ACTIVE (queued/retrying — e.g. the cluster or this endpoint is down) or
    # INACTIVE (needs re-authentication). Either way the transfer is still alive.
    nice = task.get("nice_status")
    hint = stuck_hint(status, nice)
    archive = {**data.get("archive", {}), "last_status": f"{status}: {nice}", "checked_at": utc_now_iso()}
    if hint:
        archive["last_hint"] = hint
    else:
        archive.pop("last_hint", None)
    update_manifest(run_dir, archive=archive)
    return "transfer_pending"
