# Data Collection Workflow for Gamma

This workflow records GPU power while workloads run on Gamma. You start the recording from the Windows lab PC in room 3.102. It saves the measurements locally and can archive completed runs to the lab's FASRC storage through Globus.

You need access to the lab PC and Gamma, plus FASRC access for archiving. Reach out to Ian or Raja to get started. The separate [Gamma 2 repository](https://github.com/IanMwai/paihardware-gamma2) covers the VM-based system.

## Collect a recording

Follow the [setup guide](docs/setup.md) the first time. On an already configured lab PC, open PowerShell in this repository and run:

```powershell
.\.venv\Scripts\Activate.ps1
pai devices
pai
```

`pai` opens the menu for recording, viewing, and managing runs. Before recording, confirm which GPU your workload uses and select its matching measurement channels. The [collection guide](docs/collecting-data.md) explains how to do this and gives commands for timed runs.

Start your workload separately on Gamma. When you're done, stop the recording with Ctrl+C and wait for it to finish saving.

## Find your data

```powershell
pai runs list
pai archive status latest
```

Recordings are saved under `output/` by default, with test runs in `output/test/`. Data files are compressed CSVs (`.csv.gz`). A completed recording may still be waiting for its archive transfer.

For long runs, I recommend recording without the live dashboard and reviewing the saved data afterward.

## Guides

- [Setup](docs/setup.md): install the software and connect the machines.
- [Collecting data](docs/collecting-data.md): choose GPUs, record, and check a run.
- [Archiving](docs/archiving.md): transfer, download, and clean up recordings.
- [Troubleshooting](docs/troubleshooting.md): investigate a failed or incomplete run.
- [Hardware and data reference](docs/data-and-hardware.md): wiring, calibration, and file contents.
- [Development](docs/development.md): change the code and run tests.
