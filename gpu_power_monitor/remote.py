"""SSH/SFTP link to the GPU workload machine ("gamma").

Runs an NVML logger (nvidia-smi) on gamma for the duration of a power run and
pulls the results into the run folder, so the DAQ power data and the GPU-side
telemetry archive together. Explicit GPU selection requires valid telemetry;
legacy all-GPU recordings can use best-effort telemetry.

Auth: password or key from `.env` (see RemoteConfig). Key auth is preferred
when configured; password auth is also supported. Connections are
opened per operation — runs last hours, holding a session open that long is
less robust than reconnecting twice.
"""

from __future__ import annotations

import csv
import datetime as dt
import posixpath
import shlex
from pathlib import Path
from typing import Any

from .config import RemoteConfig
from .utils import utc_now_iso


class RemoteUnavailable(RuntimeError):
    pass


def remote_unready_reason(config: RemoteConfig) -> str | None:
    """Why gamma cannot be used right now, or None if it is ready."""
    if not config.enabled:
        return "remote NVML integration is disabled in the config"
    if not config.host:
        return "PAI_GAMMA_HOST is not set in .env"
    if not config.user:
        return "PAI_GAMMA_USER is not set in .env"
    if not config.password and not config.key_path:
        return "set PAI_GAMMA_PASSWORD or PAI_GAMMA_KEY_PATH in .env"
    try:
        import paramiko  # noqa: F401
    except ImportError:
        return "paramiko is not installed (pip install -e .[remote])"
    return None


def connect(config: RemoteConfig):
    """Open an SSH connection to gamma. Raises RemoteUnavailable on failure."""
    reason = remote_unready_reason(config)
    if reason:
        raise RemoteUnavailable(reason)
    import paramiko

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kwargs: dict[str, Any] = {
        "username": config.user,
        "timeout": float(config.connect_timeout_sec),
    }
    if config.key_path:
        kwargs["key_filename"] = str(Path(config.key_path).expanduser())
    if config.password:
        kwargs["password"] = config.password
    try:
        client.connect(config.host, **kwargs)
    except Exception as exc:
        raise RemoteUnavailable(f"cannot reach {config.user}@{config.host}: {exc}") from exc
    return client


def run_command(client, command: str, timeout: float = 30.0) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    return exit_code, stdout.read().decode(errors="replace"), stderr.read().decode(errors="replace")


# Reproducibility metadata captured once per run: which GPUs, which driver.
# This is what makes a public dataset row citable/replicable later.
_INFO_COMMANDS = (
    "hostname",
    "date -u +%FT%TZ",
    "nvidia-smi -L",
    "nvidia-smi --query-gpu=index,name,driver_version,vbios_version,power.limit --format=csv",
)

# One NVML sample per line; timestamp comes from gamma's clock (record both
# clocks in the manifest so streams can be aligned in analysis).
_NVML_QUERY = (
    "timestamp,index,uuid,name,power.draw,utilization.gpu,utilization.memory,"
    "temperature.gpu,clocks.sm,memory.used"
)


class NvmlLogger:
    """Start/stop an nvidia-smi CSV logger on gamma for one run.

    Files live under <remote_run_root>/<run_id>/ on gamma (the run folder name
    is the correlation key between the two machines) and are pulled into
    <run_dir>/nvml/ locally.
    """

    def __init__(self, config: RemoteConfig, run_id: str):
        self.config = config
        self.run_id = run_id
        self.remote_dir = posixpath.join(config.remote_run_root, run_id)
        self._quoted_dir = shlex.quote(self.remote_dir)

    def start(self) -> None:
        """Create the remote run dir, record GPU metadata, launch the logger."""
        interval = max(1, int(self.config.nvml_interval_ms))
        info_commands = _INFO_COMMANDS
        if self.config.gpu_ids:
            ids = shlex.quote(",".join(self.config.gpu_ids))
            info_commands = tuple(command + " -i " + ids if command.startswith("nvidia-smi") else command for command in info_commands)
        info_block = " ; ".join(info_commands)
        selection = ""
        if self.config.gpu_ids:
            selection = "-i " + shlex.quote(",".join(self.config.gpu_ids)) + " "
        command = (
            f"mkdir -p {self._quoted_dir} && "
            f"{{ {info_block} ; }} > {self._quoted_dir}/gamma_info.txt 2>&1 ; "
            f"nohup nvidia-smi {selection}--query-gpu={_NVML_QUERY} --format=csv,nounits "
            f"-lms {interval} > {self._quoted_dir}/nvml.csv 2> {self._quoted_dir}/nvml.err "
            f"< /dev/null & echo $! > {self._quoted_dir}/nvml.pid ; cat {self._quoted_dir}/nvml.pid"
        )
        client = connect(self.config)
        try:
            code, out, err = run_command(client, command)
            pid = out.strip()
            if code != 0 or not pid.isdigit():
                raise RemoteUnavailable(
                    f"NVML logger did not start (exit {code}): {err.strip() or out.strip()}"
                )
            if self.config.required:
                alive, detail, _ = run_command(client, f"sleep 0.2; kill -0 {int(pid)} 2>/dev/null || cat {self._quoted_dir}/nvml.err")
                # kill's status must not be masked by a successful cat.
                alive, _, _ = run_command(client, f"kill -0 {int(pid)} 2>/dev/null")
                if alive != 0:
                    raise RemoteUnavailable(f"Required NVML process exited: {detail}")
        finally:
            client.close()

    def stop_and_fetch(self, dest_dir: Path) -> dict[str, Any]:
        """Kill the logger (if running) and pull the remote run dir.

        Never raises: returns a manifest-ready dict with status "fetched" or
        "missing" (plus the error), so callers can record the outcome and move
        on. The remote copy is left in place as a belt-and-braces backup.
        """
        result: dict[str, Any] = {
            "host": self.config.host,
            "remote_dir": self.remote_dir,
            "interval_ms": int(self.config.nvml_interval_ms),
            "gpu_ids": list(self.config.gpu_ids),
            "status": "missing",
        }
        try:
            client = connect(self.config)
        except RemoteUnavailable as exc:
            result["error"] = str(exc)
            return result
        try:
            # Idempotent stop: kill the recorded PID if it is still alive.
            run_command(
                client,
                f"test -f {self._quoted_dir}/nvml.pid && "
                f"kill $(cat {self._quoted_dir}/nvml.pid) 2>/dev/null ; true",
            )
            fetched = self._fetch_dir(client, Path(dest_dir))
            if fetched:
                result["status"] = "fetched"
                result["files"] = fetched
                result["fetched_at"] = utc_now_iso()
                if self.config.required:
                    telemetry = validate_nvml(Path(dest_dir) / "nvml.csv", self.config.gpu_ids)
                    result.update(telemetry)
            else:
                result["error"] = f"no files found at {self.remote_dir} on {self.config.host}"
        except Exception as exc:
            result["status"] = "missing"
            result["error"] = str(exc)
        finally:
            client.close()
        return result

    def _fetch_dir(self, client, dest_dir: Path) -> list[str]:
        sftp = client.open_sftp()
        try:
            try:
                names = sftp.listdir(self.remote_dir)
            except FileNotFoundError:
                return []
            if not names:
                return []
            dest_dir.mkdir(parents=True, exist_ok=True)
            fetched = []
            for name in sorted(names):
                remote_path = posixpath.join(self.remote_dir, name)
                sftp.get(remote_path, str(dest_dir / name))
                fetched.append(name)
            return fetched
        finally:
            sftp.close()


def validate_nvml(path: Path, expected: tuple[str, ...]) -> dict:
    """Require actual samples for every selected GPU, and no unselected samples."""
    counts = {gpu: 0 for gpu in expected}
    first = last = None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        for row in reader:
            row = {k.strip(): str(v).strip() for k, v in row.items() if k}
            matches = [gpu for gpu in expected if gpu == row.get("index") or gpu == row.get("uuid")]
            if len(matches) != 1:
                raise RemoteUnavailable("NVML sample does not match the selected GPU assignment")
            try:
                power = float(row.get("power.draw [W]", row.get("power.draw")))
                import math
                if not math.isfinite(power): raise ValueError("non-finite power")
                stamp = dt.datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f")
            except (KeyError, ValueError) as exc:
                raise RemoteUnavailable(f"Invalid required NVML power sample: {exc}") from exc
            counts[matches[0]] += 1
            first = stamp if first is None else min(first, stamp)
            last = stamp if last is None else max(last, stamp)
    if not counts or any(n == 0 for n in counts.values()):
        raise RemoteUnavailable("Missing required NVML samples")
    return {"sample_counts": counts, "observed_span_s": (last - first).total_seconds(),
            "first_timestamp": first.isoformat(), "last_timestamp": last.isoformat(),
            "clock": "GPU host local clock; align separately with DAQ clock"}
