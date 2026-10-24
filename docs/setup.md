# Set up the lab PC

Use the Windows PC connected to the NI USB-6341. Install the NI-DAQmx driver and confirm the device appears in NI MAX. Close NI MAX test panels before recording: another task can reserve the device.

Install Python 3.10 or newer. In PowerShell, clone the repository, enter it, and create its environment:

```powershell
git clone https://github.com/IanMwai/paihardware.git
cd paihardware
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e '.[hardware,archive,remote]'
Copy-Item .env.example .env
pai devices
```

If PowerShell blocks activation, run commands through `.\.venv\Scripts\pai.exe` and install through `.\.venv\Scripts\python.exe`. Follow the lab's execution-policy settings rather than changing them globally.

## Configure this machine

Edit `.env` in the repository. It is ignored by Git. Set the output disk, Gamma's SSH host/user/key, and the local Globus collection ID. Existing shell variables take priority over `.env`; `.env` takes priority over `configs/default.yaml`.

Relative output paths are resolved from the repository, even when you run `pai` from another directory. Use an absolute path for a separate data disk. Make sure the Windows account recording the data can write there and has enough free space. Gzip is enabled by default, but 10 kHz recordings still need substantial disk space. Leave room for open CSV chunks while they are being compressed. Measure a short run's size before choosing a long duration.

SSH access belongs to the account running `pai`. Confirm that account can log into Gamma and run `nvidia-smi`. Install the `remote` extra above for SSH support. A subset measurement with paired telemetry needs the physical GPU mapping described in [Collecting data](collecting-data.md).

Globus setup is a separate step: see [Archiving](archiving.md). You can test acquisition before configuring archiving.

## Check the installation

```powershell
pai hardware --simulate --test --name setup_check --duration-sec 10
pai runs list
```

Then make a short real recording with the verified DAQ/NVIDIA selection. Inspect the manifest and diagnostics before increasing the duration.

## Update an existing installation

Finish any recording first. From the repository, activate the existing environment, then:

```powershell
git status --short
git pull --ff-only
python -m pip install -e '.[hardware,archive,remote]'
```

If Git reports local edits, inspect and preserve them before pulling. Keep `.env`, local configurations, `.venv`, and run folders. Recloning source code does not recreate an environment, SSH credentials, or local settings.

[Back to the workflow](../README.md)
