# 10 — Task Engine Spike: Temporal vs Celery+Redis (hands-on)

*Session: 2026-06-13. Spike code: `experiments/task-engine-spike/`. Resolves the
RA7 open question in [DECISIONS.md](DECISIONS.md) (Task engine row, status 🟡).*

## What was built

A 3-stage toy pipeline mirroring L0→L1→L2 of the layer stack
(`l0_seed -> l1_dense -> l2_coarse`), each stage a heartbeating "GPU activity"
that sleeps in 0.5s ticks (configurable total duration). Workflow exposes a
`progress` query (completed stages, current stage, done flag) — the shape our
SSE/webhook progress layer would read.

Environment: Windows 11, **no admin rights, no Docker**.

- `temporal.exe` v1.7.2 (Server 1.31.1) — downloaded as a zip from
  `temporalio/cli` GitHub releases, unzipped to `experiments/task-engine-spike/bin/`,
  runs with zero install.
- Python env via `uv`: system default is Python 3.14, but `temporalio==1.28.x`
  does not yet support 3.14, so `uv python install 3.12` + pinned venv.
  `uv add temporalio` resolved cleanly on 3.12.13.

## Test 1 — mid-stage worker crash + resume

Started workflow `demo-asset-2` (12s/stage = 24 ticks/stage at 0.5s).
`l0_seed` completed normally (24/24). While `l1_dense` was ticking
(reached tick ~3-4/24), force-killed the worker process (`taskkill /F`).
Restarted the worker against the same dev server.

**Result:**
- `l0_seed` (already-completed activity) was **not** re-executed — workflow
  history already recorded its result. This is the headline property: stages
  are checkpointed for free, no DIY checkpoint table.
- The still-in-flight stage resumed via **activity heartbeat details**: the
  restarted worker logged `resuming from heartbeat tick 1/24` and continued
  ticking from there rather than restarting the 24-tick stage from zero. (The
  exact resume tick depends on the last heartbeat Temporal's server had
  durably recorded before the crash — in our run that was tick 1, i.e. the
  resumed attempt picked up just behind where the killed attempt had reached,
  not from scratch.)
- End to end, the workflow completed successfully; `query_progress` reported
  `completed_stages=['l0_seed','l1_dense','l2_coarse'], done=True`,
  `status: Completed`.

This is exactly the spot-instance-resumability story §1.4/§5 needs: a worker
on a preemptible GPU node can die mid-refine and a fresh worker picks the
activity back up from its last heartbeat, with zero pipeline-specific
checkpoint code beyond calling `activity.heartbeat()` periodically.

## Test 2 — dev-server restart with `--db-filename` persistence

Killed `temporal.exe` and the worker entirely. Restarted the dev server with
`--db-filename data/temporal.db` (same file). `temporal workflow list` and the
web UI immediately showed `asset-pipeline-demo-asset-2` as `Completed` with
full history — durable state survived a full server restart with no extra
config. This is the dev-mode analog of the Postgres-backed production
persistence and is enough for the single-box `astel up` story in "patient
mode."

## Measurements

| Metric | Value | Notes |
|---|---|---|
| `temporal.exe` binary | 553 MB on disk (~109 MB zipped) | single Go binary, no install, no admin |
| Dev server idle RSS | ~120-130 MB | both cold start and warm restart |
| Time to server ready (`localhost:7233` serving) | ~14-18s cold, ~4s warm restart with existing db | "warm" = db file already exists |
| Time to first workflow start | a few seconds after server+worker up | client connect + poll + dispatch |
| `temporal.db` after 2 short workflows | 676 KB | sqlite, grows slowly |
| Python venv footprint (`temporalio`+deps) | ~58 MB | on Python 3.12.13 |
| Python version constraint | temporalio 1.28.x requires <3.14 | must pin 3.12 via `uv python install 3.12`; system 3.14 unaffected for other tools |

None of these numbers are a problem for a single-GPU dev box (64GB RAM target
per CLAUDE.md §6) or for bundling into `astel up`.

## Celery + Redis — analytic comparison (not installed; based on the evidence above)

| Dimension | Temporal (observed) | Celery + Redis (analysis) |
|---|---|---|
| Windows/no-admin friction | Single binary, unzip and run. No native deps. | Redis has no first-party Windows build; options are WSL2, Memurai (commercial), or a Docker container — all unavailable or extra friction on this box. Celery itself is fine on Windows but the `prefork` pool is unsupported (must use `solo`/`threads`/`gevent`), which limits worker concurrency model choices. |
| Resumable multi-stage workflows | Built-in: workflow history is the durable log; completed activities are never re-run; in-flight activities resume from heartbeat details automatically. Demonstrated above with zero pipeline-specific code. | No native concept of a "workflow." Each stage is an independent task; resuming a multi-stage pipeline after a crash requires **DIY checkpoint tables** (Postgres rows per asset/stage/status) plus orchestration logic (a "chain" or custom state machine) to decide what to re-enqueue. Partial-stage progress (mid-refine) is not resumable at all without bespoke checkpointing inside the task itself (e.g., save partial gaussians to disk, have the task check for and load a checkpoint on (re)start) — strictly more code than `activity.heartbeat()`. |
| Progress semantics | First-class: workflow queries (`progress()` above) give live, typed state without extra infrastructure; SSE/webhook layer just polls or subscribes to workflow state. | Celery has task states (`PENDING/STARTED/SUCCESS/...`) and `update_state(meta=...)` for custom progress, backed by the Redis result backend. Workable, but per-stage progress across a multi-stage pipeline again needs the same custom checkpoint table to aggregate "which of N stages done + % through current stage" — Temporal gives this for free via workflow state. |
| Spot-instance resumability | Demonstrated directly (Test 1): kill -9 the worker mid-activity, restart anywhere, pipeline continues from last heartbeat without re-running prior stages. | Possible but DIY: requires (a) task-level checkpointing to durable storage, (b) idempotent task design so re-enqueue doesn't redo completed stages, (c) a supervisor/beat process to detect dead workers and re-enqueue — Temporal's server does (c) automatically via task timeouts/heartbeat timeouts. |
| Self-host footprint for `astel up` | Dev: 1 binary (~550MB unpacked), ~125MB idle RAM, sqlite file. Prod: same binary's server mode or Temporal's Postgres-backed server — and we're running Postgres anyway per §5. | Needs Redis (broker+backend) as an additional service — on this Windows box that's the friction point (no admin, no Docker → WSL2 or a non-trivial manual Redis-for-Windows install). Celery workers themselves are lightweight, but the missing broker is the practical blocker for a clean `astel up` on bare Windows. |
| Operational maturity / UI | Web UI (localhost:8233) ships in the same binary: workflow list, history, pending activities, retries — useful for debugging stuck pipeline runs out of the box. | Flower or similar needed for equivalent visibility; another service to run. |

## FINAL RECOMMENDATION

**Adopt Temporal**, confirming the 🟡 in DECISIONS.md → ✅. The hands-on spike
validated the exact properties the spec requires (resumable multi-stage
workflows, heartbeat-based mid-stage resume, durable progress queries,
spot-tolerance) with **zero custom checkpoint code**, and it runs admin-free
on bare Windows as a single ~550MB binary with ~125MB idle RAM and a sqlite
file for dev persistence. Celery+Redis would require us to hand-roll the
checkpoint/progress/dead-worker-detection machinery Temporal provides natively,
and Redis itself is the weak link on a no-admin/no-Docker Windows box.

**Embedding in `astel up`:**
- **Dev/single-box ("patient mode")**: bundle `temporal.exe` in the repo (or
  fetch on first run) and launch `temporal server start-dev --db-filename
  <data-dir>/temporal.db` as a managed subprocess of `astel up`. ~125MB RAM,
  sqlite persistence — workflows survive `astel` restarts.
- **Prod/cloud**: run `temporalio/server` (or `temporal.exe server start`)
  against the Postgres instance we already provision per §5 (one schema,
  shared with asset/task/credit tables) — no new datastore.
- **Workers**: one Temporal worker process per GPU pool (preview vs refine
  queues, per spec §5/§7), each registering the activities for its stage(s) of
  the layer stack (L0-L2 preview queue, L3+ refine queue, etc.). Activity
  heartbeats double as the wall-time/VRAM telemetry hooks required by
  CLAUDE.md §10.3.
- **Progress/SSE**: gateway (FastAPI) queries workflow state via
  `progress()`-style queries (as built in this spike) and forwards to
  SSE/webhooks — no extra pub/sub needed for the common case.
