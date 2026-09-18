"""Offline tests for the gamma link: readiness reasons and path handling.
(The live SSH path is exercised manually / by real runs, not in CI.)"""

from gpu_power_monitor.config import RemoteConfig
from gpu_power_monitor.remote import NvmlLogger, remote_unready_reason


def test_disabled_gives_reason():
    reason = remote_unready_reason(RemoteConfig(enabled=False))
    assert "disabled" in reason


def test_missing_credentials_give_reasons():
    assert "PAI_GAMMA_HOST" in remote_unready_reason(RemoteConfig())
    assert "PAI_GAMMA_USER" in remote_unready_reason(RemoteConfig(host="h"))
    assert "PAI_GAMMA_PASSWORD" in remote_unready_reason(RemoteConfig(host="h", user="u"))


def test_password_or_key_satisfies_auth():
    ready_pw = remote_unready_reason(RemoteConfig(host="h", user="u", password="p"))
    ready_key = remote_unready_reason(RemoteConfig(host="h", user="u", key_path="k"))
    # None (ready) unless paramiko is missing in this environment.
    assert ready_pw is None or "paramiko" in ready_pw
    assert ready_key is None or "paramiko" in ready_key


def test_remote_dir_quotes_spaces():
    logger = NvmlLogger(RemoteConfig(host="h", user="u", password="p"), "GPU Run 0_20260918_120000")
    assert logger.remote_dir == "pai_runs/GPU Run 0_20260918_120000"
    # shlex quoting must wrap the space-containing path for the remote shell.
    assert logger._quoted_dir.startswith("'") and logger._quoted_dir.endswith("'")
