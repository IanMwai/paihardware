# Archive and retrieve runs

Acquisition saves locally first. A normal run automatically submits a Globus transfer after clean completion when archiving is configured. Test runs stay local. A submitted transfer is not yet a verified archive.

## Configure Globus once per recording account

Install Globus Connect Personal on the Windows lab PC, sign in, and create its collection. Allow access to the output folder in its settings. Copy that collection's UUID from its collection details into `PAI_GLOBUS_LOCAL_ENDPOINT_ID` in `.env`.

Install the archive extra and sign in to the CLI from the same Windows account used for recording:

```powershell
python -m pip install -e '.[archive]'
globus login
```

The local collection identifies the recording PC. The remote collection identifies the FASRC storage system. They are different IDs. The configured destination path must belong to the remote collection and be writable by your FASRC identity.

The repository retains the lab's existing Holyoke collection and `/n/holylabs/lexie_lab/Lab/gpu_power_logs` default. Confirm the current destination with the lab before using it. FASRC storage migrations may change both the path and collection; a different location needs matching `PAI_ARCHIVE_ROOT` and `PAI_GLOBUS_REMOTE_ENDPOINT_ID` settings. Follow the official [FASRC Globus guide](https://docs.rc.fas.harvard.edu/kb/globus-file-transfer/) and [storage migration information](https://docs.rc.fas.harvard.edu/kb/storage-modernization-initiative-faqs/).

## Check or retry a transfer

```powershell
pai archive status RUN_ID
pai archive push RUN_ID
```

`transfer pending` means Globus has a task to finish. It may be transferring, queued, or waiting for access. Status refreshes the task and can show a waiting reason. `archived_verified` means the archive checks passed. `ready_to_archive` means the data is still waiting for submission. A failed push does not erase the local recording.

If Globus requests consent or authentication, follow its authorization link as the recording account, then retry. FASRC access can require a separate institutional authentication step beyond the initial Globus login. Keep Connect Personal running and the local output path accessible.

For a destination mounted directly on the recording machine, `pai archive copy RUN_ID --archive-root PATH` copies and verifies files. `pai archive verify RUN_ID --archived-run-dir PATH` verifies a locally accessible archive copy. `mark-archived` records a manual assertion; it does not independently verify a transfer, so it should not be used to dismiss an unresolved transfer failure.

## Keep or remove local data

The default retention period is 14 days from the run end, with verified archiving required. Preview eligibility first:

```powershell
pai archive cleanup RUN_ID
```

Only add `--delete` when you intend to remove the eligible local copy. Keep failed or partial measurements until you have inspected and preserved them.

`pai runs promote RUN_ID` makes a test run eligible for the archive workflow; `pai runs demote RUN_ID` makes a run local-only. Promotion is not a transfer. `pai runs delete RUN_ID` deletes a test run and is intended for disposable checks.

Retrieve archived data through Globus using an account with access to the lab folder. Keep the manifest, diagnostics, and telemetry with the chunks. Extract gzip files when needed for analysis; plain CSV needs no extraction.

[Back to the workflow](../README.md)
