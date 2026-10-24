# Data and hardware

The NI USB-6341 measures a voltage tap and shunt signal for each configured GPU. At the default 10 kHz rate, software receives 1,000-sample blocks and writes one-minute chunks. This is the sample rate of the acquisition, not a claim about the bandwidth of every part of the measurement circuit. Confirm which power rails the shunts cover before interpreting the result as whole-board power.

## Lab-PC wiring and calibration

These settings belong to this deployment. GPU2–GPU4 have opposite shunt orientation to GPU1.

| DAQ label | Voltage | Current | Voltage multiplier | Current sign |
| --- | --- | --- | --- | --- |
| GPU1 | ai0 | ai1 | 4.741144 | +1 |
| GPU2 | ai2 | ai3 | 4.611890 | −1 |
| GPU3 | ai4 | ai5 | 4.611890 | −1 |
| GPU4 | ai6 | ai7 | 4.606657 | −1 |

The current multiplier is 12.5 A/V, corresponding to the configured 0.08-ohm shunt. The voltage multipliers use the September 2026 calibration against a 12.18 V reference. Confirm calibration and wiring after hardware changes; do not copy Gamma 2's current signs into this configuration.

The default causal power average uses ten samples, or 1 ms at 10 kHz. Saved power therefore includes this smoothing. Set `processing.power_average_samples: 1` in a local configuration if an experiment requires unsmoothed power, and retain that configuration in the manifest.

## What a run contains

- `manifest.json`: configuration, acquisition status, sample statistics, and checksummed file inventory.
- `nidaq_*.csv.gz` (or `.csv` when compression is disabled or falls back): elapsed `time_s`, per-GPU voltage/current/power, and total power across the selected channels. Units are volts, amperes, and watts.
- `nvml/`: retrieved NVIDIA telemetry and server information when enabled.
- `diagnostics/writer.jsonl`: queue depth, acquired/written sample counts, driver backlog, and writer timings.
- `latest.npz` and `latest_status.json`: dashboard previews; these are not the complete recording.

Summary energy uses joules. The manifest's sample count describes the recorded stream. A successful run requires acquired and written sample counts to agree. A queue or storage failure stops the recording and preserves an error instead of silently dropping blocks.

## CSV and gzip compression

Closed CSV chunks are compressed to `.csv.gz` by default. Compression is lossless and runs in the background. Python and pandas can read gzip files directly. Set `logging.compress: false` in a local configuration to keep plain CSV for debugging. Each logical chunk is published only after its export representation is settled. The hidden `.chunks` directory tracks publication and must stay with the run.

A verified gzip can remain alongside a redundant CSV if Windows temporarily prevents deletion. Readers and inventories count matching representations once; conflicting contents raise an integrity error. If compression fails and the CSV is intact, recording keeps the CSV and records a warning. Do not delete one copy just because both exist without verifying them.

## Compare DAQ and NVIDIA data carefully

NVIDIA logging requests samples every 100 ms by default. The device's power sensor update interval and averaging may differ from that polling interval. It is not a 10 kHz reference.

DAQ time is derived from sample order; NVIDIA timestamps come from Gamma. Synchronize and document the two machines' clocks and workload markers before comparing individual events. Telemetry coverage checks cannot establish timestamp alignment. Keep all source files for later analysis rather than relying on a dashboard screenshot.

[Back to the workflow](../README.md)
