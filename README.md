# Data Collection Workflow for Gamma

This repository runs the power measurement system on the Windows lab PC in room 3.102. The PC reads the shunts connected to Gamma's GPUs through an NI USB-6341. Gamma runs the workloads; the lab PC records the power data. When SSH access is configured, it also collects NVIDIA telemetry from Gamma for the same run.

The [Gamma 2 deployment](https://github.com/IanMwai/paihardware-gamma2) uses a Windows VM instead of a separate lab PC. The two deployments share measurement code, but their wiring and setup differ. Use this repository's calibration for this hardware.

You need access to the lab PC and Gamma to collect measurements. You also need a FASRC account and access to the lab's destination folder to archive runs. Ask Ian or Raja about access and the current hardware connections.

## Start here

Follow [Setup](docs/setup.md) once on the lab PC. Open PowerShell in the repository and activate its environment:

```powershell
.\.venv\Scripts\Activate.ps1
pai devices
pai hardware --simulate --test --name setup_check --duration-sec 10
pai runs list
```

The simulated run checks the software without using the DAQ or producing real measurements. Test runs stay local and are never archived automatically.

Before recording a workload, confirm which DAQ channel pair measures the active GPU. On Gamma, list NVIDIA's identifiers:

```bash
nvidia-smi --query-gpu=index,uuid,name,pci.bus_id --format=csv
```

DAQ labels such as `GPU2` describe the wiring. NVIDIA indices such as `1` describe the server's enumeration. Verify the physical mapping; the numbers alone do not establish it.

For example, **only if DAQ GPU2 is confirmed to measure NVIDIA index 1**, run this on the lab PC:

```powershell
pai hardware --gpus GPU2 --nvml-gpus 1 --test --name gpu2_check --duration-sec 60
```

Start the workload separately on Gamma while recording. This lab-PC command does not launch a workload on the server. Select only the GPUs used by the experiment; collect all four only when all four are part of it. [Collecting data](docs/collecting-data.md) explains selection, timing, and telemetry requirements.

## Commands you will use regularly

| Task | Command |
| --- | --- |
| Record a named run; stop with Ctrl+C | `pai hardware --gpus GPU2 --nvml-gpus 1 --name experiment_01` |
| Record for one hour | `pai hardware --gpus GPU2 --nvml-gpus 1 --name experiment_01 --duration-sec 3600` |
| List saved runs | `pai runs list` |
| Check an archive transfer | `pai archive status latest` |
| Retry an archive transfer | `pai archive push RUN_ID` |
| Retry fetching existing NVIDIA logs | `pai fetch RUN_ID` |
| View a saved recording | `pai dashboard RUN_ID --replay` |
| Delete a disposable test | `pai runs delete RUN_ID` |

The GPU2/index 1 examples assume a verified mapping. Substitute the identifiers for your experiment.

A normal run is saved under `output/`; test runs are under `output/test/`. `PAI_OUTPUT_ROOT` can select a different disk. Recordings are split into one-minute CSV chunks and compressed to `.csv.gz` in the background. Compression is lossless; Python and pandas can read these files directly. `manifest.json` records the configuration, status, statistics, and file checksums. Completion of acquisition and completion of archiving are separate steps.

For long measurements, use the terminal without a live dashboard and leave it open until finalization finishes. The dashboard is optional; recorded files are the data to keep and analyze. On this Windows PC, `--display` opens a local browser. Replay plots are previews, not a replacement for the recorded samples.

## More detail

| Guide | What it covers |
| --- | --- |
| [Setup](docs/setup.md) | Windows environment, NI driver, SSH, and local settings |
| [Collecting data](docs/collecting-data.md) | Selecting GPUs, starting and stopping, and run checks |
| [Data and hardware](docs/data-and-hardware.md) | Wiring, calibration, file contents, and timing limits |
| [Archiving](docs/archiving.md) | Globus, FASRC access, transfer status, and local cleanup |
| [Troubleshooting](docs/troubleshooting.md) | Recording failures, telemetry problems, and diagnostics |
| [Development](docs/development.md) | Tests and changes to the shared measurement code |
