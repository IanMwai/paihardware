# PAI Hardware — GPU Power Monitor

High-rate GPU power measurement for the PAI hardware initiative (Le Xie lab and
Minlan Yu lab). The system logs 10 kHz NI-DAQ voltage/current locally on the
DAQ machine, serves a live browser dashboard, and archives completed runs to
shared cluster storage automatically via Globus.

## Deployment context and assumptions

- **Primary deployment:** the left Windows computer in lab **3.102**, which is
  wired to the NI-DAQ box measuring the GPU power rails. The software is not
  tied to that machine and can run on any similar Windows setup with the
  NI-DAQmx driver and NI-DAQ hardware. The only machine-specific state is the
  `.env` file and the DAQ device name (see setup below).
- **Archiving assumes Harvard FASRC:** the default destination is the Le Xie
  lab's holylabs path via the FASRC Globus collection. Another lab (e.g.
  Minlan's) points `PAI_ARCHIVE_ROOT` (and, if needed,
  `PAI_GLOBUS_REMOTE_ENDPOINT_ID`) at its own storage in `.env`. No code or
  shared-config changes.
- **One run at a time per machine:** acquisition owns the DAQ device
  exclusively while running. The dashboard is read-only and can be opened by
  anyone, anytime, without affecting a run.
- **Archived data is shared, not personal:** runs are pushed to a
  group-accessible lab directory, not to any individual's account. See
  [Data access and permissions](#data-access-and-permissions).

## Pipeline at a glance

1. **Acquire** : `pai hardware` reads the NI-DAQ in blocks; full-resolution
   CSVs rotate every 60 s into `output/<run>_<timestamp>/`, and a small rolling
   window (`latest.npz` + `latest_status.json`) is kept for the dashboard.
2. **Display** : the dashboard polls that rolling window; it is a viewer only,
   so closing/pausing it never affects logging.
3. **Close** : on stop (Ctrl+C or `--duration-sec`) the run's `manifest.json`
   is finalized: file inventory with SHA-256 checksums, run stats, status
   `completed` / archive status `ready_to_archive`.
4. **Archive** : a Globus transfer to the lab's archive path is submitted
   automatically after a clean run end (or from menu option 5 / `pai archive
   push`). Globus queues and retries while the cluster is down and
   checksum-verifies every file; success flips the run to `archived_verified`.
5. **Cleanup** : `pai archive cleanup` may delete a local run only once it is
   `archived_verified` **and** older than the retention period (default 14 days).

## Setting up a new machine

Requirements: Windows with the **NI-DAQmx driver** installed (from National
Instruments; reboot after installing) and **Python 3.10+** on PATH.

In PowerShell:

```powershell
git clone https://github.com/IanMwai/paihardware.git
cd paihardware
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[hardware,archive]
```

This installs the acquisition dependencies (`numpy`, `pyyaml`, `nidaqmx`), the
Globus CLI for archiving, and creates the `pai` command. On machines without
the DAQ hardware (e.g. for `--simulate` runs or replaying recorded data),
plain `pip install -e .` is enough. If `pai` is ever not on PATH, every
`pai ...` command below also works as `python -m gpu_power_monitor.cli ...`.

> **Tip:** to avoid activating the venv in every new window, add an
> auto-activation snippet to your PowerShell profile, e.g.
> ```powershell
> # in $PROFILE (adjust the path to where you cloned the project):
> $pai = "$HOME\Desktop\paihardware"
> if ($PWD.Path -like "$pai*" -and (Test-Path "$pai\.venv\Scripts\Activate.ps1")) {
>     & "$pai\.venv\Scripts\Activate.ps1"
> }
> ```

### Machine-specific settings (`.env`)

Shared settings live in `configs/default.yaml` and should not need editing.
Anything specific to a machine or lab goes in a `.env` file at the project
root, which is gitignored:

```powershell
Copy-Item .env.example .env
notepad .env
```

| Variable | What it is | Who needs to set it |
| --- | --- | --- |
| `PAI_GLOBUS_LOCAL_ENDPOINT_ID` | UUID of this machine's Globus Connect Personal endpoint (GCP tray icon → Web: Collection Details) | Every DAQ machine that archives |
| `PAI_GLOBUS_REMOTE_ENDPOINT_ID` | Destination collection UUID; defaults to "Harvard FAS RC Holyoke" (serves `/n/holylabs`) | Only if archiving somewhere else |
| `PAI_ARCHIVE_ROOT` | Archive path on the cluster; defaults to the Le Xie lab path | Other labs (e.g. Minlan's lab sets its own holylabs path) |
| `PAI_OUTPUT_ROOT` | Local run directory; defaults to `output/` | Rarely |

With `PAI_GLOBUS_LOCAL_ENDPOINT_ID` unset, everything still works. Runs just
stay local, and `pai` prints how to archive them once Globus is configured.

### Per-user setup on a shared machine

Each Windows profile keeps its **own clone** (e.g.
`C:\Users\<netid>\Desktop\paihardware`) with its own venv, `.env`, and
`output/` folder. No file in the repo contains an absolute path, so nothing
needs editing when you clone to a different location: the config, output
directory, and dashboard all resolve relative to your clone. Per-user
checklist:

1. Clone + install (top of this section).
2. Copy `.env.example` → `.env`. On an already-configured machine, reuse the
   machine's `PAI_GLOBUS_LOCAL_ENDPOINT_ID` (ask whoever set up Globus Connect
   Personal, or read it from the GCP tray icon → Web: Collection Details).
3. Make sure GCP's accessible paths include *your* clone's folder (GCP tray
   icon → Options → Access), otherwise your pushes fail with a permission
   error while everyone else's work fine.
4. Do the per-profile Globus login (see
   [Archiving — one-time setup](#one-time-setup)).

### Smoke test with simulated data (no hardware needed)

This proves Python, the dashboard, and the browser all work before touching the DAQ:

```powershell
pai hardware --simulate --display --name "sim_test"
```

A browser should open showing live (simulated) voltage, current, and power traces.
Press `Ctrl+C` in PowerShell to stop. If this works, the software is good and any
remaining issue is purely hardware/config.

### Confirm the DAQ device name

The config assumes the device is named `Dev1`. Confirm what your hardware
actually enumerates as:

```powershell
pai devices
```

This lists every connected NI device, e.g. `Dev1  (USB-6212)`. If it shows a
different name (e.g. `Dev2`), set `channels.device` in `configs/default.yaml` to
match. (You can also confirm the name in NI MAX → Devices and Interfaces.)

## Start acquisition

**The easiest way is just:**

```bash
pai
```

That opens an interactive menu: start a run, open a dashboard, replay a past
run, archive. No flags or run names to remember. Most lab members should use
this.

If you prefer the CLI directly, the equivalents are below. The `--name` flag
is **optional** everywhere: without it, runs are auto-named `GPU Run 0`,
`GPU Run 1`, and so on, counting up from your existing runs.

Acquire **and** open the live dashboard:

```bash
pai hardware --display
```

Acquisition only (headless):

```bash
pai hardware --name "training_run_001"
```

Dry-run without NI hardware (add `--simulate` to any of the above):

```bash
pai hardware --simulate --display
```

For a finite smoke test, add `--duration-sec 10`.

Runs are written under `output/<measurement_name>_<timestamp>/`. Press `Ctrl+C`
to stop a run cleanly.

## View dashboard

The dashboard is a local web app (Python stdlib only, no extra dependencies). It
streams from the live snapshot files the acquisition writes and opens in a browser:

```bash
pai dashboard output/<run_id>
```

For a video wall, open the browser fullscreen (or pass `--fullscreen`). Useful
flags: `--port <n>` and `--no-browser`.

Controls (in the browser):

- `p`: pause/resume the display only (the run keeps acquiring and logging).
  During replay this is play/pause.
- `z`: zoom. Pauses the display, then drag a box on any panel to zoom in.
  Resume (`p`) or Reset View (`r`) both exit the zoom and return to live.
- `r`: reset view: default window, autoscale, live scrolling (in replay:
  restart playback from the beginning at 1×).
- `+` / `-`: widen/narrow the live time window.
- `t`: toggle light/dark theme.

Panels draw an oscilloscope-style graticule: minor gridlines land on round
units (…, 10 ms, 100 ms, 1 s, 2 s, 5 s… on time; same 1-2-5 ladder on the
y-axes), with brighter labelled major lines every 5 minors. The steps adapt
to the window width, zoom, and browser size, so each tiny box is always a
known round division; read spans off the grid like on a scope or in LTspice.

Dashboard states:

- **Live** : acquisition is writing samples right now.
- **Stale** : the last snapshot is real data but the writer went quiet
  (crash/kill, or you opened an old run); the badge shows how old the data is.
- **Run Complete** : the run ended cleanly.
- **Replay** : playing back a recorded run (see below).
- **Archived** : the run has been archived to the cluster.
- **Idle** : no data yet.
- **Error** : acquisition died with an error.
- **Disconnected** : this page lost the dashboard server (e.g. the run was
  stopped with Ctrl+C); the last traces stay on screen.

**Replay** loads the recorded run and plays it back like a live run: play/pause,
speed 1× / 10× / 60×, or "Full run" to see the whole trace at once. Opening a
dashboard on a finished/stale run offers a "▶ Replay run" button. Replay data
is served downsampled (up to 50k points); the CSVs keep full resolution.

**Tabs:** the dashboard reuses an already-open tab. If a tab from a previous
run is still open when a new run starts on the same port, that tab reloads
itself into the new run instead of a second tab opening.

## Archive storage

Completed full-resolution runs are archived to persistent lab storage on the
cluster (FASRC **holylabs** for Harvard labs, which has snapshots and disaster
recovery, unlike scratch storage, which auto-purges). Each lab points
`PAI_ARCHIVE_ROOT` at its own path; the default is the Le Xie lab's
`/n/holylabs/lexie_lab/Lab/gpu_power_logs`.

The archive directory is group-writable with **setgid** (new run folders
inherit the lab group) and the **sticky bit** (only a file's owner can delete
or rename it, so no one can remove someone else's runs). To create one with
those modes (on a cluster login node):

```bash
mkdir -p /n/holylabs/<your_lab>/Lab/gpu_power_logs
chmod 3775 /n/holylabs/<your_lab>/Lab/gpu_power_logs
```

Transfers go through **Globus** (the cluster path is not mounted on the DAQ
machine), which checksum-verifies every file.

## Data access and permissions

Where the data lives and who can do what:

- **Runs are archived to the shared lab directory**, not to anyone's home
  directory. Files are *owned by* whichever FASRC identity is logged into
  Globus on the DAQ machine (currently `itoyota` on the 3.102 machine), but
  the directory's setgid bit (mode `3775`) makes every run folder inherit the
  lab group, so ownership does not gate access.
- **Reading archived runs requires your own FASRC account** in the lab's
  group (`lexie_lab` for the Le Xie lab). Then either:
  - `ssh <username>@login.rc.fas.harvard.edu` and read
    `/n/holylabs/lexie_lab/Lab/gpu_power_logs` directly, or
  - use the Globus web app (or CLI) with your own FASRC identity to transfer
    a run to your laptop.

  New members request an FASRC account through the
  [FASRC portal](https://portal.rc.fas.harvard.edu/) with either PI as a sponsor.
- **Globus login on the DAQ machine is only needed to push archives**, and
  tokens are stored per Windows user profile. Each operator logs in once under
  their own Windows account (see
  [Archiving — one-time setup](#one-time-setup)). Nobody needs a Globus login
  just to *read* data on the cluster.
- **Write rights:**
  - *DAQ machine, `output/`* : whoever is logged into Windows and runs `pai`.
  - *Archive directory* : every lab-group member can add new runs (group
    `rwx` + setgid), everyone in the group can read everything, and the
    sticky bit (mode `3775`) means only a file's owner can delete or alter
    their runs. The archive is effectively **append-only**: nothing in this
    tool ever deletes from it, and local cleanup only removes the DAQ
    machine's copy after Globus has checksum-verified the archived one.

## Archiving (automatic, via Globus)

When a run completes cleanly, `pai` automatically submits a Globus transfer of
the run folder to `<archive_root>/<run_id>` (`storage.globus.auto_push: true`).
Menu option 5 does the same thing for any past run, and shows a plain-English
status per run.

### One-time setup

**Per machine** : **Globus Connect Personal** must be installed and running,
with every user's `output/` folder inside its accessible paths (GCP tray icon
→ Options → Access). Put its endpoint UUID in each clone's `.env` as
`PAI_GLOBUS_LOCAL_ENDPOINT_ID` (see above).

**Per Windows profile** : Globus logins do **not** carry across Windows
accounts. Every lab member who operates the DAQ under their own profile must
run both of these once, in their own terminal:

1. Log in to Globus (a browser opens; sign in with your university key):

   ```powershell
   globus login
   ```

2. Authenticate your FASRC identity (FASRC requires this *in addition to*
   `globus login`, and asks again when the session expires; the symptom is a
   "Session reauthentication required" error on any transfer or `globus ls`):

   ```powershell
   globus session update globus.rc.fas.harvard.edu
   ```

If your pushes are skipped with a "Not logged in to Globus" message, you are
in a Windows profile that has not completed these two steps.

### How a run flows to the archive

`ready_to_archive` → `transfer_pending` (Globus task submitted) →
`archived_verified` (task succeeded; Globus checksum-verified every file).
The status refreshes whenever the interactive menu starts, on menu option 5,
and on `pai archive status <run>`. Manual submit/retry:

```bash
pai archive push output/<run_id>
pai archive status output/<run_id>
```

**Duplicates are not possible.** Already-archived runs are refused with "Run
is already archived" (menu option 5 likewise reports rather than resubmits a
pending transfer), every run has one fixed destination
(`<archive_root>/<run_id>`), and transfers use Globus checksum sync. Even a
deliberate re-push only sends files that are missing or differ on the archive
side.

### If the cluster or Globus endpoint is down

Nothing is lost and nothing needs babysitting: the transfer sits queued on
Globus's servers and retries until it succeeds or the deadline passes
(`storage.globus.deadline_days`, default 7). The DAQ machine can even go
offline after submitting. If a transfer is stuck or fails, `pai archive
status` and the interactive menu print what is wrong in plain English and what
to fix (e.g. "Globus Connect Personal is not running on this machine", "run:
globus login"). A run is never cleanup-eligible until it is
`archived_verified`, so a failed or pending push can never cost data.

### Manual fallback

If the Globus CLI is unavailable, transfer the run folder with Globus Connect
Personal by hand, then record it (this is what later unlocks cleanup):

```bash
pai archive mark-archived output/<run_id>
# --destination overrides the default <archive_root>/<run_id>
```

Optionally re-verify the checksums on the cluster (repo checkout + Python there):

```bash
python3 -m gpu_power_monitor.cli archive verify <archived_copy> --archived-run-dir <archived_copy>
```

If the archive target is ever mounted/mapped directly, `copy` transfers,
verifies, and marks in one step (`copy` refuses to run when the archive root
is not actually mounted, so it cannot silently copy to the local disk):

```bash
pai archive copy output/<run_id>
pai archive status output/<run_id>
```

## Local cleanup

Old runs only need deleting when the local disk fills up. Cleanup is
intentionally conservative: a local run is only eligible after it is marked
`archived_verified` and older than the retention period (`storage.retention_days`,
default 14 days).

```bash
pai archive cleanup output/<run_id>
pai archive cleanup output/<run_id> --delete
```

## Hardware settings reference

These match the lab's DAQ wiring (originally from `NI_script_VIP_1GPU_*.py`,
kept in the repo as a legacy reference) and live in `configs/default.yaml`.
No need to change them unless the wiring changes:

- Channels: `Dev1/ai0` (voltage), `Dev1/ai1` + `Dev1/ai2` (current), `RSE`, −0.5…3.5 V
- Voltage scale `4.768`, current scale `12.5`, sample rate `10000` Hz
- Power moving average `10` samples, CSV chunk rotation every `60` s

## Troubleshooting

- **`nidaqmx is not installed`** → run `pip install -e .[hardware]` inside the venv.
- **`pai devices` shows nothing** → check the DAQ USB/cable and that the NI-DAQmx
  driver is installed; confirm the device appears in NI MAX.
- **Dashboard opens but stays IDLE / no data** → the device name in
  `configs/default.yaml` likely does not match `pai devices` output.
- **Port 8000 already in use** → add `--port 8001` to the `pai hardware --display`
  or `pai dashboard` command.
- **Runs stay "local only" / pushes are skipped** → `PAI_GLOBUS_LOCAL_ENDPOINT_ID`
  is not set in `.env`, the Globus CLI is missing (`pip install -e .[archive]`),
  or you are not logged in (`globus login`). The skip message says which.
- **Transfer pending for a long time** → `pai archive status <run>` prints why
  it is waiting (endpoint offline, re-authentication needed, …) and what to do.

## Staying up to date, and contributing

The repo lives at <https://github.com/IanMwai/paihardware>. It is a
**personal repo** (maintained by Ian Toyota): lab members do not have write
access; use it, pull updates, and contribute via fork + pull request. If you
hit an issue, email Ian, ping him on the lab Slack, or open a GitHub issue on
the repo.

**Staying current** : cloning (step 1 of setup) already connects your copy to
GitHub, so there is nothing extra to initialize. Every now and then, from your
clone:

```powershell
git pull
```

Because the package is installed editable (`pip install -e .`), pulled code
changes take effect immediately; you only re-run
`pip install -e .[hardware,archive]` if dependencies changed. If you edited
files locally and `git pull` complains, stash first (`git stash`, pull, then
`git stash pop`). Your `.env`, venv, and `output/` are untouched by pulls
because they are not tracked by git.

**Contributing** : fork + PR.

1. Click **Fork** on <https://github.com/IanMwai/paihardware>.
2. Clone *your fork*, and add this repo as `upstream`:

   ```bash
   git remote add upstream https://github.com/IanMwai/paihardware.git
   ```

3. Branch, make your change, and run the tests (`python -m pytest tests`;
   they pass without DAQ hardware or Globus).
4. Push to your fork and open a pull request against `IanMwai/paihardware`.
   Keep your fork in sync with `git fetch upstream && git merge upstream/main`.

## Ground rules
- **Never commit `.env`** (it is gitignored) or any endpoint UUIDs, usernames,
  or lab-specific paths in code. Those belong in `.env` / `.env.example`.
- Run `python -m pytest tests` before opening a PR (`pip install -e .[dev]`
  for pytest). Tests run without DAQ hardware or Globus.
- Changes to shared defaults in `configs/default.yaml` affect every machine.
  Flag them clearly in the PR description.
