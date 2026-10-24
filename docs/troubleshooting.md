# Troubleshoot a recording

Preserve the run directory before changing anything. Start with `manifest.json`, then `diagnostics/writer.jsonl`, and the terminal's original error. Keep partial recordings labelled as partial; do not turn an error into a successful run by editing status fields.

## The NI device is reserved

NI error −50103 usually means another task owns the resource. Close NI MAX test panels and check for another acquisition process on Windows. Identify the process and run before stopping it. Do not kill every Python process on a shared machine. A stale dashboard status alone does not prove a process is still recording.

## Recording stops under writer pressure

Inspect `acquired_samples`, `written_samples`, queue depth, queued seconds, writer throughput, and DAQ backlog in `diagnostics/writer.jsonl`. A rising queue suggests storage or formatting cannot keep up. Check free space, output location, competing disk activity, and antivirus/file-indexing interference. A driver backlog can also indicate that the acquisition loop is falling behind.

The writer waits briefly for a temporary backlog to clear. A sustained overload becomes an explicit error, preserving the partial data. Increasing the queue only postpones a sustained throughput problem. Validate a short run across several chunk rotations, then an hour, before returning to a longer experiment.

## A CSV cannot be deleted after compression

If the gzip is verified, a locked redundant CSV produces a warning rather than stopping acquisition. The logical chunk is counted once. Keep both files and the `.chunks` records until they can be checked. Compression failure falls back to an intact CSV. Corruption or conflicting CSV/gzip contents is an integrity failure and needs investigation, not forced cleanup.

## NVIDIA data is missing or mismatched

Confirm SSH access from the Windows account running `pai`, install the remote extra, and check the verified GPU mapping. An explicitly selected telemetry run requires valid samples for exactly those GPUs. Unreachable telemetry or insufficient coverage fails that paired recording.

`pai fetch RUN_ID` retries retrieval of logs already created on Gamma. It cannot reconstruct telemetry that was never recorded. All-GPU legacy mode can finish DAQ acquisition without telemetry; inspect the manifest rather than assuming both streams are present.

## A run finished but is not archived

Check `pai archive status RUN_ID`. Completion describes acquisition; transfer pending describes the archive task. Confirm Globus authentication, collection access, destination path, and Connect Personal's allowed folders. Retry with `pai archive push RUN_ID` after resolving access. See [Archiving](archiving.md).

## The plot stops updating

Inspect the recording diagnostics and terminal before treating a stale preview as a DAQ failure. If the dashboard port is occupied, choose another with `--port`. Replay reads saved samples and may be expensive for long recordings. Preserve the full-resolution chunks regardless of what the preview shows.

[Back to the workflow](../README.md)
