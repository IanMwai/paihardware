"""Env-var / .env overrides for machine-specific config values."""

from pathlib import Path

from gpu_power_monitor.config import load_config, load_dotenv


def test_env_var_overrides_yaml(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "storage:\n"
        "  archive_root: '/n/holylabs/lexie_lab/Lab/gpu_power_logs'\n"
        "  globus:\n"
        "    local_endpoint_id: 'from-yaml'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PAI_GLOBUS_LOCAL_ENDPOINT_ID", "from-env")
    monkeypatch.setenv("PAI_ARCHIVE_ROOT", "/n/holylabs/other_lab/Lab/gpu_power_logs")
    config = load_config(cfg_path)
    assert config.storage.globus.local_endpoint_id == "from-env"
    assert config.storage.archive_root == Path("/n/holylabs/other_lab/Lab/gpu_power_logs")


def test_env_override_without_yaml_section(tmp_path, monkeypatch):
    # Overrides apply even when the YAML has no storage/globus section at all.
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("sample_rate_hz: 1000\n", encoding="utf-8")
    monkeypatch.setenv("PAI_GLOBUS_LOCAL_ENDPOINT_ID", "endpoint-uuid")
    config = load_config(cfg_path)
    assert config.storage.globus.local_endpoint_id == "endpoint-uuid"
    assert config.sample_rate_hz == 1000


def test_dotenv_loads_but_never_overrides_real_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment line\n"
        'PAI_GLOBUS_LOCAL_ENDPOINT_ID="dotenv-uuid"\n'
        "PAI_ARCHIVE_ROOT=/n/from/dotenv\n",
        encoding="utf-8",
    )
    # setenv-then-delenv so monkeypatch restores the pre-test state even after
    # load_dotenv writes the variable.
    monkeypatch.setenv("PAI_GLOBUS_LOCAL_ENDPOINT_ID", "sentinel")
    monkeypatch.delenv("PAI_GLOBUS_LOCAL_ENDPOINT_ID")
    monkeypatch.setenv("PAI_ARCHIVE_ROOT", "/n/from/shell")
    load_dotenv(env_file)
    import os

    assert os.environ["PAI_GLOBUS_LOCAL_ENDPOINT_ID"] == "dotenv-uuid"  # quotes stripped
    assert os.environ["PAI_ARCHIVE_ROOT"] == "/n/from/shell"  # shell wins
