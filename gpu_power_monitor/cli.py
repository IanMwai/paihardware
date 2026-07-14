"""`pai` command-line entry point.

Two layers over the same underlying workflows:

* Direct subcommands for the **developer** persona (scriptable, no prompts)::

      pai hardware [--simulate] [--test]  # start acquisition (headless by default)
      pai dashboard <run> [--replay]      # open a dashboard on a run
      pai archive push|copy|verify|status|cleanup|mark-archived <run>
      pai runs list|promote|demote|delete [run]  # test runs vs archive-bound runs

* An interactive **menu** for the lab-demo persona, shown when `pai` is run with
  no arguments. It favors fast uptime and quick live/replay viewing.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Auto-generated run names look like "GPU Run 0", "GPU Run 1", ... (test runs
# count their own "Test Run N" sequence). The run folder appends a
# _<timestamp>, e.g. "GPU Run 0_20260624_132551".
_AUTO_NAME_PREFIX = "GPU Run"
_TEST_NAME_PREFIX = "Test Run"


def _auto_name_re(prefix: str) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(prefix)} (\d+)(?:_\d{{8}}_\d{{6}})?(?:_\d+)?$")


def _repo_root() -> Path:
    # cli.py lives in <root>/gpu_power_monitor/, so the repo root is one up.
    return Path(__file__).resolve().parent.parent


def _default_config() -> Path:
    return _repo_root() / "configs" / "default.yaml"


def _output_root() -> Path:
    return _repo_root() / "output"


def _list_runs() -> list[Path]:
    """Run directories under output/ (archive-bound) and output/test/, newest first."""
    from gpu_power_monitor.utils import TEST_RUNS_SUBDIR

    root = _output_root()
    if not root.exists():
        return []
    runs = [p for p in root.iterdir() if p.is_dir() and p.name != TEST_RUNS_SUBDIR]
    test_root = root / TEST_RUNS_SUBDIR
    if test_root.is_dir():
        runs += [p for p in test_root.iterdir() if p.is_dir()]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def _latest_run() -> Path | None:
    runs = _list_runs()
    return runs[0] if runs else None


def _resolve_run(run: str | Path) -> Path:
    """Resolve a run reference to a directory, accepting a bare run name.

    A user-typed name like ``GPU Run 0_20260624_132551`` is resolved against
    ``output/`` (and ``output/test/``) so replay works regardless of the
    current directory.
    """
    from gpu_power_monitor.utils import TEST_RUNS_SUBDIR

    path = Path(run)
    if path.is_dir():
        return path
    for candidate in (_output_root() / path.name, _output_root() / TEST_RUNS_SUBDIR / path.name):
        if candidate.is_dir():
            return candidate
    return path


def auto_run_name(*, test: bool = False) -> str:
    """Next ``GPU Run N`` (or ``Test Run N``) name, counting up from existing runs."""
    prefix = _TEST_NAME_PREFIX if test else _AUTO_NAME_PREFIX
    pattern = _auto_name_re(prefix)
    highest = -1
    for run in _list_runs():
        match = pattern.match(run.name)
        if match:
            highest = max(highest, int(match.group(1)))
    return f"{prefix} {highest + 1}"


# --------------------------------------------------------------------------- #
# Action handlers
# --------------------------------------------------------------------------- #
def run_dashboard(
    run_dir: Path,
    *,
    replay: bool = False,
    fullscreen: bool = False,
    port: int = 8000,
    open_browser: bool = True,
    config_path: Path | None = None,
) -> int:
    from gpu_power_monitor.config import load_config
    from gpu_power_monitor.web import serve

    run_dir = _resolve_run(run_dir)
    config = load_config(config_path or _default_config())
    display = {
        "window_sec": config.display.window_sec,
        "refresh_ms": config.display.refresh_ms,
        "stale_after_sec": config.display.stale_after_sec,
        "y_limits": config.display.y_limits,
        "fullscreen": fullscreen or config.display.fullscreen,
        "replay": replay,
    }
    serve(run_dir, display=display, port=port, open_browser=open_browser)
    return 0


def run_archive(command: str, run_dir: Path, **kwargs) -> int:
    import json

    from gpu_power_monitor.archive import (
        archive_destination,
        archive_status,
        cleanup_local,
        copy_run,
        mark_archived,
        verify_archive,
    )
    from gpu_power_monitor.config import load_config

    run_dir = _resolve_run(run_dir)
    config = load_config(kwargs.get("config_path") or _default_config())
    archive_root = Path(kwargs.get("archive_root") or config.storage.archive_root)
    if command == "copy":
        dest = copy_run(run_dir, archive_root)
        print(f"[INFO] Archived and verified: {dest}")
    elif command == "push":
        from gpu_power_monitor.globus_push import GlobusUnavailable, push_run

        try:
            task_id = push_run(run_dir, config.storage.globus, archive_root)
        except (GlobusUnavailable, RuntimeError) as exc:
            print(f"[ERROR] {exc}")
            return 1
        print(f"[INFO] Transfer submitted (Globus task {task_id}).")
        print("[INFO] Globus queues and retries while the cluster is down; "
              "check with `pai archive status` or menu option 5.")
    elif command == "verify":
        verify_archive(run_dir, kwargs.get("archived_run_dir"))
        print("[INFO] Archive verification passed.")
    elif command == "status":
        data = archive_status(run_dir)
        if data.get("archive_status") == "transfer_pending":
            try:
                from gpu_power_monitor.globus_push import check_task

                check_task(run_dir)
                data = archive_status(run_dir)
            except Exception as exc:  # stale status is still useful
                print(f"[WARN] Could not refresh the Globus task: {exc}")
        print(json.dumps(data, indent=2, sort_keys=True))
        archive = data.get("archive", {})
        if data.get("archive_status") == "transfer_pending" and archive.get("last_hint"):
            print(f"[INFO] Transfer is waiting: {archive['last_hint']}")
        elif data.get("archive_status") == "push_failed":
            error = archive.get("last_error") or "see the manifest for details"
            print(f"[WARN] Push failed: {error}")
            print(f'[INFO] Retry with: pai archive push "{run_dir}"')
    elif command == "mark-archived":
        destination = kwargs.get("destination") or archive_destination(run_dir, archive_root)
        dest = mark_archived(run_dir, destination)
        print(f"[INFO] Marked archived: {dest}")
    elif command == "cleanup":
        delete = kwargs.get("delete", False)
        eligible = cleanup_local(run_dir, config.storage.retention_days, dry_run=not delete)
        if eligible and delete:
            print("[INFO] Local run deleted.")
        elif eligible:
            print("[INFO] Local run is cleanup-eligible. Re-run with --delete to remove it.")
        else:
            print("[INFO] Local run is not cleanup-eligible.")
    return 0


def _relocate_run(run_dir: Path, *, test: bool) -> Path:
    """Move a run folder to match its kind (output/ vs output/test/).

    The manifest's ``run_kind`` is the source of truth; this move is only the
    browsable convenience, so every failure degrades to a warning and the run
    stays where it is (listings find it either way). Returns the current path.
    """
    import shutil

    from gpu_power_monitor.live_buffer import read_live_status
    from gpu_power_monitor.utils import runs_root

    dest_parent = runs_root(_output_root(), test=test)
    if run_dir.parent.resolve() == dest_parent.resolve():
        return run_dir
    if read_live_status(run_dir, stale_after_sec=5.0).state == "LIVE":
        print("[WARN] Run appears to be in use (acquisition/dashboard) - leaving the folder "
              "where it is; it is still listed correctly.")
        return run_dir
    dest = dest_parent / run_dir.name
    if dest.exists():
        print(f"[WARN] Not moving the folder, destination already exists: {dest}")
        return run_dir
    dest_parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(run_dir), str(dest))
    except OSError as exc:
        print(f"[WARN] Could not move the run folder ({exc}) - the run is still marked "
              f"correctly and stays at: {run_dir}")
        return run_dir
    print(f"[INFO] Moved to: {dest}")
    return dest


def run_runs(action: str, run: str | None) -> int:
    """`pai runs` - list local runs, or manage the test-vs-archive split."""
    from gpu_power_monitor.archive import delete_test_run, run_kind, set_run_kind
    from gpu_power_monitor.manifest import load_manifest
    from gpu_power_monitor.utils import runs_root

    if action == "list":
        runs = _list_runs()
        if not runs:
            print("[INFO] No runs found under output/.")
            return 0
        for r in runs:
            note = ""
            try:
                is_test = run_kind(load_manifest(r)) == "test"
            except (FileNotFoundError, ValueError, OSError):
                is_test = None
            if is_test is not None and r.parent.resolve() != runs_root(_output_root(), test=is_test).resolve():
                fix = "demote" if is_test else "promote"
                note = f'   <- folder mismatch, fix with: pai runs {fix} "{r.name}"'
            print(f"  {r.name}   [{_archive_label(r)}]{note}")
        return 0
    if not run:
        print(f"[ERROR] 'pai runs {action}' needs a run name.")
        return 1
    run_dir = _resolve_run(run)
    if not run_dir.is_dir():
        print(f"[ERROR] No such run: {run}")
        return 1
    try:
        if action == "promote":
            set_run_kind(run_dir, "archive")
            run_dir = _relocate_run(run_dir, test=False)
            print(f"[INFO] '{run_dir.name}' is now archive-bound.")
            print(f'[INFO] Push it with: pai archive push "{run_dir.name}"  (or menu option 5)')
        elif action == "demote":
            set_run_kind(run_dir, "test")
            run_dir = _relocate_run(run_dir, test=True)
            print(f"[INFO] '{run_dir.name}' is now a test run - it stays local and is never archived.")
        elif action == "delete":
            delete_test_run(run_dir)
            print(f"[INFO] Deleted test run: {run_dir.name}")
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}")
        return 1
    return 0


def run_devices() -> int:
    """List connected NI-DAQ devices so the user can confirm the device name."""
    try:
        import nidaqmx.system
    except ImportError:
        print(
            "[ERROR] nidaqmx is not installed. Install hardware extras first:\n"
            "        pip install -e .[hardware]"
        )
        return 1

    system = nidaqmx.system.System.local()
    devices = list(system.devices)
    if not devices:
        print("[INFO] No NI-DAQ devices detected. Check connections and NI-DAQmx driver.")
        return 1

    print("Connected NI-DAQ devices:")
    for device in devices:
        try:
            product = device.product_type
        except Exception:
            product = "?"
        print(f"  {device.name}  ({product})")
    print(
        "\nSet channels.device in configs/default.yaml to the name shown above "
        "(currently 'Dev1')."
    )
    return 0


def run_hardware(
    *,
    simulate: bool,
    display: bool,
    name: str | None = None,
    test: bool = False,
    port: int = 8000,
    config_path: Path | None = None,
    duration_sec: float | None = None,
) -> int:
    import dataclasses
    import threading

    from gpu_power_monitor.acquisition import acquire, with_measurement_name
    from gpu_power_monitor.config import load_config
    from gpu_power_monitor.utils import runs_root, unique_run_dir

    config = with_measurement_name(
        load_config(config_path or _default_config()), name or auto_run_name(test=test)
    )
    if test:
        config = dataclasses.replace(config, test_run=True)

    if not display:
        # Headless (developer): acquisition in the foreground, Ctrl+C to stop.
        run_dir = acquire(config, simulate=simulate, duration_sec=duration_sec)
        _auto_push(run_dir, config)
        return 0

    # Demo: acquisition in a background thread, dashboard in the foreground.
    from gpu_power_monitor.web import serve

    run_dir = unique_run_dir(runs_root(config.storage.output_root, test=test), config.measurement_name)
    stop = threading.Event()

    def _background_acquire() -> None:
        try:
            acquire(config, simulate=simulate, run_dir=run_dir, stop_event=stop, duration_sec=duration_sec)
        except Exception as exc:  # surface, don't kill the dashboard
            print(f"[ERROR] Acquisition stopped: {exc}")

    thread = threading.Thread(target=_background_acquire, daemon=True)
    thread.start()

    display_cfg = {
        "window_sec": config.display.window_sec,
        "refresh_ms": config.display.refresh_ms,
        "stale_after_sec": config.display.stale_after_sec,
        "y_limits": config.display.y_limits,
        "fullscreen": config.display.fullscreen,
    }
    try:
        serve(run_dir, display=display_cfg, port=port, open_browser=True)
    finally:
        stop.set()
        thread.join(timeout=15.0)
    _auto_push(run_dir, config)
    return 0


def _auto_push(run_dir: Path, config) -> None:
    """Submit the completed run to Globus if auto-push is on. Never raises —
    every failure leaves the run safely un-archived with a retry hint."""
    from gpu_power_monitor import globus_push
    from gpu_power_monitor.manifest import load_manifest

    gcfg = config.storage.globus
    try:
        data = load_manifest(run_dir)
    except (FileNotFoundError, ValueError):
        return
    if data.get("status") != "completed":
        return  # errored or still-open runs are not archived
    if data.get("run_kind", "archive") == "test":
        print(f'[INFO] Test run - kept local, not archived. Delete anytime with: pai runs delete "{run_dir.name}"')
        return
    if not gcfg.auto_push:
        return
    retry_hint = f'pai archive push "{run_dir}"  (or menu option 5)'
    reason = globus_push.globus_unready_reason(gcfg)
    if reason:
        print(f"[INFO] Skipping archive push: {reason}")
        print(f"[INFO] Archive later with: {retry_hint}")
        return
    try:
        task_id = globus_push.push_run(run_dir, gcfg, config.storage.archive_root)
    except Exception as exc:
        print(f"[WARN] Archive push failed: {exc}")
        print(f"[INFO] Retry with: {retry_hint}")
        return
    print(f"[INFO] Archive transfer submitted (Globus task {task_id}).")
    print("[INFO] Globus queues and retries while the cluster is down; "
          "check with `pai archive status` or menu option 5.")


# --------------------------------------------------------------------------- #
# Interactive menu (lab-demo persona)
# --------------------------------------------------------------------------- #
_ARCHIVE_LABELS = {
    "local_only": "local only",
    "ready_to_archive": "ready to archive",
    "transfer_pending": "transfer pending",
    "archived_verified": "archived",
    "cleanup_eligible": "archived, cleanup eligible",
    "push_failed": "PUSH FAILED",
    "archive_error": "ARCHIVE ERROR",
}


def _archive_label(run: Path) -> str:
    """Short human-readable archive state of a run (for menu listings)."""
    from gpu_power_monitor.manifest import load_manifest

    try:
        data = load_manifest(run)
    except (FileNotFoundError, ValueError, OSError):
        return "no manifest"
    is_test = data.get("run_kind", "archive") == "test"
    if data.get("status") == "running":
        return "test run in progress" if is_test else "run in progress"
    if data.get("status") == "error":
        return "run errored"
    if is_test:
        return "test run (local only)"
    status = data.get("archive_status", "local_only")
    return _ARCHIVE_LABELS.get(status, status)


def _refresh_pending_transfers() -> None:
    """Quick pass over runs with a pending Globus transfer; updates statuses.

    Quiet and best-effort: menu startup must never block on Globus problems.
    """
    from gpu_power_monitor import globus_push
    from gpu_power_monitor.manifest import load_manifest

    for run in _list_runs():
        try:
            data = load_manifest(run)
        except (FileNotFoundError, ValueError, OSError):
            continue
        if data.get("archive_status") != "transfer_pending":
            continue
        try:
            new_status = globus_push.check_task(run)
        except Exception:
            return  # globus unavailable right now - statuses refresh next time
        if new_status == "archived_verified":
            print(f"[INFO] Archive completed: {run.name}")
        elif new_status == "push_failed":
            print(f"[WARN] Archive transfer FAILED: {run.name} - see menu option 5.")
        elif new_status == "transfer_pending":
            try:
                hint = load_manifest(run).get("archive", {}).get("last_hint")
            except (FileNotFoundError, ValueError, OSError):
                continue
            if hint:
                print(f"[INFO] Archive transfer waiting ({run.name}): {hint}")


def _choose_run(prompt: str, annotate=None) -> Path | None:
    """Show a numbered list of past runs (newest first) and return the pick."""
    runs = _list_runs()
    if not runs:
        print("[INFO] No runs found under output/.")
        return None
    print(prompt)
    for idx, run in enumerate(runs, start=1):
        suffix = f"   [{annotate(run)}]" if annotate else ""
        print(f"  {idx}) {run.name}{suffix}")
    raw = input(f"Select [1-{len(runs)}] (blank = cancel): ").strip()
    if not raw:
        return None
    try:
        pick = int(raw)
    except ValueError:
        print("[WARN] Not a number.")
        return None
    if not 1 <= pick <= len(runs):
        print("[WARN] Out of range.")
        return None
    return runs[pick - 1]


def _menu_archive() -> int:
    """Menu option 5: plain-English archive status + push, per run."""
    from gpu_power_monitor import globus_push
    from gpu_power_monitor.config import load_config
    from gpu_power_monitor.manifest import load_manifest

    run = _choose_run("Past runs (newest first):", annotate=_archive_label)
    if run is None:
        return 1
    config = load_config(_default_config())
    data = load_manifest(run)
    archive = data.get("archive", {})
    status = data.get("archive_status", "local_only")

    if data.get("status") == "running":
        print("[INFO] This run is still in progress - it archives after it completes.")
        return 0
    if data.get("run_kind", "archive") == "test":
        print("[INFO] This is a test run - it stays local and is never archived.")
        print("[INFO] To archive it after all, promote it first (menu option 6).")
        return 0
    if status == "transfer_pending":
        print("[INFO] A Globus transfer is pending - checking...")
        try:
            status = globus_push.check_task(run)
        except Exception as exc:
            print(f"[WARN] Could not reach Globus: {exc}")
        data = load_manifest(run)
        archive = data.get("archive", {})
    if status in ("archived_verified", "cleanup_eligible"):
        print(f"[INFO] Archived to: {archive.get('destination')}")
        print(f"[INFO] Verified at: {archive.get('verified_at')}")
        return 0
    if status == "transfer_pending":
        print(f"[INFO] Transfer still pending (Globus task {archive.get('task_id')}).")
        print(f"[INFO] Last status: {archive.get('last_status', 'queued')}")
        print("[INFO] Globus keeps retrying (e.g. while the cluster is down) - nothing to do.")
        return 0
    if status == "push_failed":
        print(f"[WARN] The last transfer failed: {archive.get('last_error', 'unknown error')}")
    if data.get("status") != "completed":
        print(f"[WARN] Run is not completed (status: {data.get('status')!r}) - cannot archive.")
        return 1

    # ready_to_archive / local_only / push_failed retry
    reason = globus_push.globus_unready_reason(config.storage.globus)
    if reason:
        print(f"[INFO] Cannot push from here yet: {reason}")
        print("[INFO] Manual fallback: transfer the folder with Globus Connect Personal, then run:")
        print(f'       pai archive mark-archived "{run}"')
        return 1
    answer = input(f"Push '{run.name}' to the archive now? [Y/n]: ").strip().lower()
    if answer.startswith("n"):
        return 0
    try:
        task_id = globus_push.push_run(run, config.storage.globus, config.storage.archive_root)
    except Exception as exc:
        print(f"[WARN] Push failed: {exc}")
        return 1
    print(f"[INFO] Transfer submitted (Globus task {task_id}).")
    print("[INFO] Globus queues and retries while the cluster is down; this menu refreshes the status.")
    return 0


def _menu_test_runs() -> int:
    """Menu option 6: promote or delete test runs; mark un-archived runs as test."""
    from gpu_power_monitor.archive import delete_test_run, run_kind, set_run_kind
    from gpu_power_monitor.manifest import load_manifest

    run = _choose_run("Runs (newest first):", annotate=_archive_label)
    if run is None:
        return 1
    try:
        data = load_manifest(run)
    except (FileNotFoundError, ValueError, OSError):
        print("[WARN] This run has no readable manifest - manage it by hand.")
        return 1
    if data.get("status") == "running":
        print("[INFO] This run is still in progress - manage it after it completes.")
        return 0
    try:
        if run_kind(data) == "test":
            answer = input(
                "Test run: [p]romote to archive-bound, [d]elete from disk, blank = cancel: "
            ).strip().lower()
            if answer.startswith("p"):
                set_run_kind(run, "archive")
                _relocate_run(run, test=False)
                print(f"[INFO] '{run.name}' is now archive-bound - push it via menu option 5.")
            elif answer.startswith("d"):
                confirm = input(f"Delete '{run.name}' permanently? [y/N]: ").strip().lower()
                if confirm.startswith("y"):
                    delete_test_run(run)
                    print(f"[INFO] Deleted test run: {run.name}")
            return 0
        if data.get("archive_status") in ("transfer_pending", "archived_verified", "cleanup_eligible"):
            print("[INFO] This run is already on (or on its way to) the cluster archive - "
                  "it cannot become a test run.")
            return 0
        answer = input(
            f"Mark '{run.name}' as a test run (stays local, never archived)? [y/N]: "
        ).strip().lower()
        if answer.startswith("y"):
            set_run_kind(run, "test")
            _relocate_run(run, test=True)
            print("[INFO] Marked as a test run - delete it anytime via this menu.")
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"[WARN] {exc}")
        return 1
    return 0


def interactive_menu() -> int:
    print("PAI hardware - GPU power monitor")
    print("=================================")
    _refresh_pending_transfers()
    print("  1) Start measurement (headless)        [developer]")
    print("  2) Start measurement + live dashboard   [demo]")
    print("  3) Open dashboard - latest run (live)    [demo]")
    print("  4) Open dashboard - replay a past run    [demo]")
    print("  5) Archive a run (Globus -> cluster archive)")
    print("  6) Manage a test run (promote / delete)")
    print("  7) Quit")
    choice = input("Select [1-7]: ").strip()
    if choice in {"1", "2"}:
        test = input("Test run? Stays local, never archived [y/N]: ").strip().lower().startswith("y")
        auto = _TEST_NAME_PREFIX if test else _AUTO_NAME_PREFIX
        name = input(f"Measurement name (blank = auto '{auto} N'): ").strip() or None
        simulate = input("Use simulated data (no NI hardware)? [y/N]: ").strip().lower().startswith("y")
        return run_hardware(simulate=simulate, display=(choice == "2"), name=name, test=test)
    if choice == "3":
        latest = _latest_run()
        if latest is None:
            print("[INFO] No runs found under output/.")
            return 1
        print(f"[INFO] Opening latest run: {latest.name}")
        return run_dashboard(latest, fullscreen=True)
    if choice == "4":
        run = _choose_run("Past runs (newest first):")
        if run is None:
            return 1
        return run_dashboard(run, replay=True, fullscreen=True)
    if choice == "5":
        return _menu_archive()
    if choice == "6":
        return _menu_test_runs()
    if choice == "7":
        return 0
    print("[WARN] Unrecognized choice.")
    return 1


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pai", description="PAI GPU power monitor.")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("devices", help="List connected NI-DAQ devices.")

    p_hw = sub.add_parser("hardware", help="Start acquisition.")
    p_hw.add_argument("--simulate", action="store_true", help="Run without NI hardware.")
    p_hw.add_argument("--display", action="store_true", help="Also open the live dashboard.")
    p_hw.add_argument("--name", help="Measurement name for this run.")
    p_hw.add_argument(
        "--test",
        action="store_true",
        help="Test run: stays local, never archived; delete at will with `pai runs delete`.",
    )
    p_hw.add_argument("--port", type=int, default=8000, help="Dashboard port (with --display).")
    p_hw.add_argument("--config", help="Config YAML (default: configs/default.yaml).")
    p_hw.add_argument("--duration-sec", type=float, help="Optional finite duration for smoke tests.")

    p_dash = sub.add_parser("dashboard", help="Open a dashboard on a run.")
    p_dash.add_argument("run", help="Run directory under output/.")
    p_dash.add_argument("--replay", action="store_true", help="Force replay of saved CSVs.")
    p_dash.add_argument("--fullscreen", action="store_true", help="Video-wall fullscreen.")
    p_dash.add_argument("--port", type=int, default=8000, help="Web dashboard port.")
    p_dash.add_argument("--no-browser", action="store_true", help="Do not auto-open a browser.")
    p_dash.add_argument("--config", help="Config YAML (default: configs/default.yaml).")

    p_arch = sub.add_parser("archive", help="Archive or inspect a completed run.")
    p_arch.add_argument("action", choices=["push", "copy", "verify", "status", "cleanup", "mark-archived"])
    p_arch.add_argument("run", help="Run directory under output/.")
    p_arch.add_argument("--archived-run-dir")
    p_arch.add_argument("--delete", action="store_true")
    p_arch.add_argument("--config", help="Config YAML (default: configs/default.yaml).")
    p_arch.add_argument("--archive-root", help="Override storage.archive_root from the config.")
    p_arch.add_argument(
        "--destination",
        help="Archived copy's path for mark-archived (default: <archive_root>/<run_id>).",
    )

    p_runs = sub.add_parser("runs", help="List runs, or manage test runs vs archive-bound runs.")
    p_runs.add_argument("action", choices=["list", "promote", "demote", "delete"])
    p_runs.add_argument("run", nargs="?", help="Run directory or name (not needed for list).")

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return interactive_menu()

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "devices":
        return run_devices()
    if args.command == "hardware":
        return run_hardware(
            simulate=args.simulate,
            display=args.display,
            name=args.name,
            test=args.test,
            port=args.port,
            config_path=Path(args.config) if args.config else None,
            duration_sec=args.duration_sec,
        )
    if args.command == "dashboard":
        return run_dashboard(
            Path(args.run),
            replay=args.replay,
            fullscreen=args.fullscreen,
            port=args.port,
            open_browser=not args.no_browser,
            config_path=Path(args.config) if args.config else None,
        )
    if args.command == "archive":
        return run_archive(
            args.action,
            Path(args.run),
            archived_run_dir=Path(args.archived_run_dir) if args.archived_run_dir else None,
            delete=args.delete,
            config_path=Path(args.config) if args.config else None,
            archive_root=args.archive_root,
            destination=Path(args.destination) if args.destination else None,
        )
    if args.command == "runs":
        return run_runs(args.action, args.run)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
