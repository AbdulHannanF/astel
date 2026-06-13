# Task Engine Spike — Temporal on Windows (no admin, no Docker)

Hands-on spike to validate Temporal as Astel's task engine for resumable,
multi-stage GPU pipelines (`l0_seed -> l1_dense -> l2_coarse`, standing in
for L0-L2 of the layer stack).

## Layout

```
bin/temporal.exe      Temporal CLI v1.7.2 (single binary, Windows amd64)
src/shared.py         shared dataclasses + constants
src/activities.py     stub activity: heartbeating "GPU stage" (sleeps in 0.5s ticks)
src/workflows.py      AssetPipelineWorkflow (3 sequential activities + progress query)
src/worker.py         worker process (polls task queue, runs workflow+activity)
src/starter.py        starts a new workflow execution
src/query_progress.py queries progress + status of a running/completed workflow
data/temporal.db      sqlite persistence file for the dev server (created on first run)
logs/                 server/worker stdout captured during the spike
```

## Setup (already done in this checkout)

```powershell
# 1. Temporal CLI binary (no install, just unzip)
#    Downloaded from https://github.com/temporalio/cli/releases (v1.7.2, windows_amd64.zip)
#    -> bin/temporal.exe (579 MB unpacked, single Go binary)

# 2. Python env (system default is 3.14; temporalio needs <=3.13 at time of writing)
uv python install 3.12
uv init --python 3.12 --no-workspace .
uv add temporalio   # resolved temporalio==1.28.x on Python 3.12.13
```

## Running it yourself

Open 3 terminals in `experiments/task-engine-spike/`:

```powershell
# Terminal 1: dev server with persistent sqlite db
.\bin\temporal.exe server start-dev --db-filename data\temporal.db --ip 127.0.0.1

# Terminal 2: worker
uv run src/worker.py

# Terminal 3: start a workflow (asset id, seconds-per-stage)
uv run src/starter.py demo-asset-1 12

# Query progress / status at any time
uv run src/query_progress.py asset-pipeline-demo-asset-1
```

Web UI at http://localhost:8233 shows the workflow execution, history,
and pending activities.

## What was tested

1. **Mid-stage worker crash + resume.** Started `demo-asset-2` with 12s/stage.
   `l0_seed` completed normally. While `l1_dense` was ticking, force-killed
   the worker process (`taskkill /F`). Restarted the worker.
   - `l0_seed` and `l1_dense` were **not** re-executed (workflow history
     already recorded them as complete).
   - `l2_coarse` (which had reached tick 3/24 before the kill) **resumed
     from heartbeat tick 1/24** rather than restarting the whole 24-tick
     stage from zero — confirmed by the worker log line:
     `resuming from heartbeat tick 1/24`.
   - Workflow completed successfully end to end; `query_progress` reported
     `completed_stages=['l0_seed','l1_dense','l2_coarse'], done=True`.

2. **Dev-server restart with `--db-filename` persistence.** Killed
   `temporal.exe` and the worker entirely, restarted the dev server with
   the same `--db-filename data/temporal.db`. `temporal workflow list`
   immediately showed `asset-pipeline-demo-asset-2` as `Completed` —
   full history survived the server restart with zero extra setup.

## Measurements

| Metric | Value |
|---|---|
| `temporal.exe` size on disk | 553 MB (single binary, no install) |
| Download (zip) size | ~109 MB |
| Dev server idle RSS | ~120-130 MB |
| Time to "Server: localhost:7233" ready | ~14-18 s (cold), ~4 s (warm restart with existing db) |
| Time to first workflow start after server ready | a few seconds (worker connect + poll + start_workflow round trip) |
| `data/temporal.db` after 2 short workflows | 676 KB |
| Python venv (`temporalio` + deps) | ~58 MB |
| Python version constraint | temporalio 1.28.x requires Python <3.14; used 3.12.13 via `uv python install 3.12` |

## Cleanup

All `temporal.exe` and `python.exe` processes started by this spike were
force-killed at the end of the session. `data/temporal.db` and `logs/*.log`
are left in place as evidence; safe to delete.
