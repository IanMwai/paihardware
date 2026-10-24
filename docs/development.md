# Work on the measurement code

Install the development extra in the repository environment and run the tests:

```powershell
python -m pip install -e '.[dev]'
python -m pytest
```

The tests use simulated or mocked sources; they do not establish Windows hardware throughput. The reliability suite exercises temporary and sustained writer pressure, compression locks and failures, chunk integrity, smoothing, GPU selection, and error propagation.

Keep changes shared with Gamma 2 limited to common measurement code. This deployment has a separate Windows lab PC, different current signs, and no VM supervisor. Preserve the hardware configuration when porting fixes.

Before collecting important data, validate real Windows acquisition across several one-minute chunk rotations. Check acquired/written sample counts, diagnostics, telemetry selection, and archive verification. Then validate an hour before extending the duration. Synthetic tests cannot replace those checks.

The README covers the everyday workflow. Keep setup details in `setup.md`, measurement/file semantics in `data-and-hardware.md`, and recovery steps in `troubleshooting.md` so users can find the next action without reading the source.

[Back to the workflow](../README.md)
