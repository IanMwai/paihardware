# Collect data

Choose the workload, GPUs, duration, and run name before starting. Run the workload on Gamma and acquisition on the Windows lab PC. Keep a record of workload settings and their start/end times alongside the measurement. This repository does not orchestrate the server workload.

## Select the active GPUs

`--gpus` selects DAQ pairs by label or one-based position. `--nvml-gpus` selects NVIDIA indices or UUIDs on Gamma, with one identifier per selected pair in configuration order. For example, selecting `GPU3,GPU1` still produces GPU1 followed by GPU3, so supply NVIDIA identifiers in that order.

Check the wiring against the physical cards and use `nvidia-smi --query-gpu=index,uuid,name,pci.bus_id --format=csv` on Gamma to identify them. UUIDs avoid relying on index order across server changes. Do not assume DAQ GPU1 means NVIDIA index 0.

Once **GPU2 → NVIDIA index 1** has been verified:

```powershell
pai hardware --gpus GPU2 --nvml-gpus 1 --test --name gpu2_check --duration-sec 60
```

Explicit NVIDIA selection requires paired telemetry: a missing logger, wrong GPU, invalid power samples, or inadequate coverage makes the run fail rather than silently accepting mismatched data. For recordings longer than ten seconds, telemetry must span at least 90% of the DAQ duration. This checks coverage, not clock synchronization or sensor accuracy.

For a persistent mapping, copy `configs/default.yaml` to `configs/gamma.local.yaml` and add `nvml_id` to each verified channel entry. Keep the existing channel numbers, calibration factors, and current signs. Use `--config configs/gamma.local.yaml` when recording. Local configuration copies are ignored by Git.

DAQ-only recording is possible with `remote.enabled: false` in a local configuration. Label it as DAQ-only in your experiment notes. When remote logging is enabled, a subset run must have an explicit mapping. Leaving out `--gpus` selects all configured DAQ pairs; use that only for an experiment involving all four GPUs. The legacy all-GPU mode allows best-effort telemetry unless `remote.required: true` and all four verified `remote.gpu_ids` are configured.

## Start, stop, and check

Remove `--test` for data you intend to retain and archive. Use `--duration-sec` for a fixed measurement window, or press Ctrl+C once for a manual stop. Allow finalization to finish before closing PowerShell. Finalization drains queued samples, closes chunks, fetches telemetry, and writes the final manifest.

```powershell
pai hardware --gpus GPU2 --nvml-gpus 1 --name steady_1h --duration-sec 3600
pai runs list
pai archive status latest
```

A completed acquisition can still have an archive transfer pending. Confirm both separately. An errored run may contain useful partial data, but it is not a successful recording of the intended duration. Preserve its manifest, diagnostics, and chunks before investigating.

## Optional plots

`--display` opens the live dashboard on the lab PC. Close the dashboard session with Ctrl+C in its terminal and wait for shutdown. For long runs, recording without the dashboard keeps the workflow simpler. Later, use `pai dashboard RUN_ID --replay` to inspect saved data. Replay can take time and memory for large recordings; use the chunks directly for full-resolution analysis.

The live preview is decimated and refreshed at a bounded rate. Recording and summary statistics retain the full processed sample stream.

[Back to the workflow](../README.md)
