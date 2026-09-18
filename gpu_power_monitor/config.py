from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Machine/user-specific values come from the environment (or a `.env` file at
# the repo root — see `.env.example`) and override the shared YAML config, so
# one committed config file serves every lab machine. Maps env var -> the
# nested config key it overrides.
_ENV_OVERRIDES = {
    "PAI_GLOBUS_LOCAL_ENDPOINT_ID": ("storage", "globus", "local_endpoint_id"),
    "PAI_GLOBUS_REMOTE_ENDPOINT_ID": ("storage", "globus", "remote_endpoint_id"),
    "PAI_ARCHIVE_ROOT": ("storage", "archive_root"),
    "PAI_OUTPUT_ROOT": ("storage", "output_root"),
    "PAI_GAMMA_HOST": ("remote", "host"),
    "PAI_GAMMA_USER": ("remote", "user"),
    "PAI_GAMMA_PASSWORD": ("remote", "password"),
    "PAI_GAMMA_KEY_PATH": ("remote", "key_path"),
}


@dataclass(frozen=True)
class GpuChannel:
    """One GPU's measurement wiring: a voltage tap and a shunt current tap.

    voltage_scale is the calibrated divider ratio for this GPU's tap;
    current_sign flips shunts wired in the opposite direction so every GPU
    reads positive current under load.
    """

    label: str
    voltage: str
    current: str
    voltage_scale: float
    current_sign: float = 1.0

    def voltage_physical(self, device: str) -> str:
        return f"{device}/{self.voltage}"

    def current_physical(self, device: str) -> str:
        return f"{device}/{self.current}"


# Calibrated 2026-09 against a 12.18 V reference; GPU2-4 shunts are wired
# opposite to GPU1, hence the sign flips.
DEFAULT_GPUS: tuple[GpuChannel, ...] = (
    GpuChannel("GPU1", "ai0", "ai1", 12.18 / 2.569, 1.0),
    GpuChannel("GPU2", "ai2", "ai3", 12.18 / 2.641, -1.0),
    GpuChannel("GPU3", "ai4", "ai5", 12.18 / 2.641, -1.0),
    GpuChannel("GPU4", "ai6", "ai7", 12.18 / 2.644, -1.0),
)


@dataclass(frozen=True)
class ChannelConfig:
    device: str = "Dev1"
    terminal: str = "RSE"
    min_v: float = -0.5
    max_v: float = 3.5
    gpus: tuple[GpuChannel, ...] = DEFAULT_GPUS

    @property
    def labels(self) -> list[str]:
        return [gpu.label for gpu in self.gpus]


@dataclass(frozen=True)
class ScalingConfig:
    # Shared shunt: scaled_current = raw * current_scale (1/0.08 ohm).
    current_scale: float = 12.5


@dataclass(frozen=True)
class ProcessingConfig:
    voltage_delay_samples: int = 0
    current_delay_samples: int = 0
    power_average_samples: int = 10


@dataclass(frozen=True)
class DisplayConfig:
    window_sec: float = 60.0
    refresh_ms: int = 30
    y_limits: dict[str, list[float]] = field(
        default_factory=lambda: {
            "voltage": [11.5, 13.0],
            "current": [-1.0, 15.0],
            "power": [-10.0, 180.0],
        }
    )
    stale_after_sec: float = 2.0
    fullscreen: bool = False


@dataclass(frozen=True)
class GlobusConfig:
    """Globus transfer settings for pushing runs to the archive.

    Both endpoint IDs must be set for pushes to work: the local one comes from
    Globus Connect Personal on this machine, the remote one is the FASRC
    collection (search "FASRC" in the Globus web app). Empty IDs disable the
    feature with a clear message rather than an error.
    """

    local_endpoint_id: str = ""
    remote_endpoint_id: str = ""
    auto_push: bool = True
    deadline_days: int = 7


@dataclass(frozen=True)
class StorageConfig:
    output_root: Path = Path("output")
    archive_root: Path = Path("/n/lab_storage/<pi_lab>/Lab/gpu_power_logs")
    retention_days: int = 14
    globus: GlobusConfig = field(default_factory=GlobusConfig)


@dataclass(frozen=True)
class RemoteConfig:
    """The GPU workload machine ("gamma") reached over SSH.

    Credentials are machine-local: set PAI_GAMMA_HOST / PAI_GAMMA_USER and
    either PAI_GAMMA_PASSWORD or PAI_GAMMA_KEY_PATH in `.env`. With no
    credentials the feature quietly disables itself (a run never depends on
    gamma being reachable).
    """

    enabled: bool = True
    host: str = ""
    user: str = ""
    password: str = ""
    key_path: str = ""
    # Where NVML logs for each run live on gamma: <remote_run_root>/<run_id>/
    remote_run_root: str = "pai_runs"
    # nvidia-smi logging interval; NVML refreshes O(10 Hz), finer adds no info.
    nvml_interval_ms: int = 100
    connect_timeout_sec: float = 10.0


@dataclass(frozen=True)
class LoggingConfig:
    chunk_duration_sec: float = 60.0
    file_prefix: str = "nidaq"
    format: str = "csv"
    queue_blocks: int = 200


@dataclass(frozen=True)
class AcquisitionConfig:
    sample_rate_hz: int = 10000
    chunk_size: int = 1000
    measurement_name: str = "GPU Power Measurement Test"
    # True marks the run as local scratch: never archived to the cluster and
    # deletable at will (`pai runs delete`). Set per run with `pai hardware
    # --test` or the menu prompt, not via the config file.
    test_run: bool = False
    channels: ChannelConfig = field(default_factory=ChannelConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    remote: RemoteConfig = field(default_factory=RemoteConfig)


def load_config(path: str | Path | None = None) -> AcquisitionConfig:
    load_dotenv()
    if path is None:
        data: dict[str, Any] = {}
    else:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        data = _load_mapping(path)
    _apply_env_overrides(data)
    return config_from_mapping(data)


def load_dotenv(path: str | Path | None = None) -> None:
    """Load ``KEY=VALUE`` lines from a ``.env`` file into ``os.environ``.

    Variables already set in the real environment always win. By default looks
    for ``.env`` at the repo root (next to ``configs/``) and in the current
    directory. Minimal on purpose — no quoting rules beyond stripping matched
    single/double quotes, ``#`` starts a comment line.
    """
    if path is not None:
        candidates = [Path(path)]
    else:
        repo_root = Path(__file__).resolve().parent.parent
        candidates = [repo_root / ".env", Path.cwd() / ".env"]
    for env_path in candidates:
        if not env_path.is_file():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            if key:
                os.environ.setdefault(key, value)


def _apply_env_overrides(data: dict[str, Any]) -> None:
    for env_var, key_path in _ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if not value:
            continue
        section = data
        for key in key_path[:-1]:
            nested = section.get(key)
            if not isinstance(nested, dict):
                nested = {}
                section[key] = nested
            section = nested
        section[key_path[-1]] = value


def config_from_mapping(data: dict[str, Any]) -> AcquisitionConfig:
    channels_data = _section(data, "channels")
    if "gpus" in channels_data:
        gpus_data = channels_data.pop("gpus") or []
        if not isinstance(gpus_data, list):
            raise TypeError("Config section 'channels.gpus' must be a list")
        channels_data["gpus"] = tuple(GpuChannel(**gpu) for gpu in gpus_data)
    channels = ChannelConfig(**channels_data)
    scaling = ScalingConfig(**_section(data, "scaling"))
    processing = ProcessingConfig(**_section(data, "processing"))
    display_data = _section(data, "display")
    display = DisplayConfig(**display_data)
    storage_data = _section(data, "storage")
    if "output_root" in storage_data:
        storage_data["output_root"] = Path(storage_data["output_root"])
    if "archive_root" in storage_data:
        storage_data["archive_root"] = Path(storage_data["archive_root"])
    if "globus" in storage_data:
        globus_data = storage_data.pop("globus") or {}
        if not isinstance(globus_data, dict):
            raise TypeError("Config section 'storage.globus' must be a mapping")
        storage_data["globus"] = GlobusConfig(**globus_data)
    storage = StorageConfig(**storage_data)
    logging = LoggingConfig(**_section(data, "logging"))
    remote_data = _section(data, "remote")
    if "enabled" in remote_data:
        remote_data["enabled"] = _as_bool(remote_data["enabled"])
    remote = RemoteConfig(**remote_data)

    top = {
        key: value
        for key, value in data.items()
        if key
        not in {
            "channels",
            "scaling",
            "processing",
            "display",
            "storage",
            "logging",
            "remote",
        }
    }
    return AcquisitionConfig(
        **top,
        channels=channels,
        scaling=scaling,
        processing=processing,
        display=display,
        storage=storage,
        logging=logging,
        remote=remote,
    )


def select_gpus(config: AcquisitionConfig, selection: str | None) -> AcquisitionConfig:
    """Restrict a run to a subset of the configured GPUs.

    ``selection`` is comma-separated labels or 1-based indices ("1,3",
    "GPU2", "gpu1, GPU4"), case-insensitive. None, blank, or "all" keeps
    every configured GPU. Config order is preserved regardless of the
    order given, so CSV columns stay in a stable order.
    """
    if not selection or selection.strip().lower() == "all":
        return config
    gpus = config.channels.gpus
    by_label = {gpu.label.lower(): gpu for gpu in gpus}
    chosen: set[str] = set()
    for token in selection.split(","):
        token = token.strip()
        if not token:
            continue
        gpu = by_label.get(token.lower())
        if gpu is None and token.isdigit() and 1 <= int(token) <= len(gpus):
            gpu = gpus[int(token) - 1]
        if gpu is None:
            valid = ", ".join(g.label for g in gpus)
            raise ValueError(
                f"Unknown GPU {token!r} - valid: {valid}, or indices 1-{len(gpus)}"
            )
        chosen.add(gpu.label)
    subset = tuple(gpu for gpu in gpus if gpu.label in chosen)
    if not subset:
        return config
    channels = dataclasses.replace(config.channels, gpus=subset)
    return dataclasses.replace(config, channels=channels)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Config section {name!r} must be a mapping")
    return dict(value)


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        import json

        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "YAML configs require PyYAML. Install it or use a JSON config."
            ) from exc

        data = yaml.safe_load(text)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError("Config file must contain a top-level mapping")
    return data
