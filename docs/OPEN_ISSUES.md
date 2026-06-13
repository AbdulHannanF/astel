# OPEN ISSUES — resolved in session 5 (2026-06-13)

> **RESOLVED.** All three items below (P1, P2, P3) were closed in session 5 —
> see [session-05 retro](retros/session-05.md). Kept as a historical record of
> the end-of-session-4 verification pass; no outstanding action here.

*Written 2026-06-13 (end-of-session-4 verification pass). Every gate is GREEN and
the session-04 code claims are real (SSE CRLF parser, single shared
`useGeneration`, artifact route, honest `origin:"stub"` report). The items below
are the deltas found by actually running the gates and diffing docs against code.*

## Verification baseline (all passed)

`api` ruff·mypy-strict·17+1skip · `web` tsc·15 · `manifest` 10 · `astel_format` 16 ·
`astel_splat_io` 11 · `astel_eval` 36 · `stub` 14.

---

## P1 — `astel_eval` test suite takes 8m30s (510s)

**Real, reproducible.** `libs/astel_eval` is green but pathologically slow; the
other six suites finish in <2s each.

- **Root cause:** [`_fit_strengths`](libs/astel_eval/src/astel_eval/bradley_terry.py:81)
  uses `max_iter=10_000, tol=1e-10` with a pure-Python nested O(n²) loop
  (`for i … for j …`). On **fully-separated** pairwise data — exactly what
  [test_gate.py](libs/astel_eval/tests/test_gate.py) builds (astel wins every one
  of 50 cases) — the Bradley-Terry MLE diverges (loser strength → 0), so it
  **never reaches `tol` and runs all 10 000 iterations on every fit**. The gate
  tests fit per-case across 50 cases × bootstraps × 4 tests → minutes.
- **Impact:** dev-loop pain now; on slower GitHub runners the `libs` CI job risks
  timing out once a remote exists.
- **Fix options (do at least the first two):**
  1. **Vectorize** the inner `for j` loop in `_fit_strengths` with numpy (kills the
     per-iteration Python cost outright).
  2. **Early-stop on relative change** and/or detect separation; cap effective
     iterations. Diverging-MLE-on-separated-data is a known BT pathology.
  3. Add a small smoothing prior (e.g. a fictitious 0.5/0.5 tie per observed pair)
     so estimates stay finite/stable — also improves correctness, not just speed.
  4. Stop-gap: lower `n_bootstrap`/`max_iter` in the gate tests (cheapest, weakest).

---

## P2 — `README.md` is stale and its one runnable instruction is wrong

[README.md](README.md):
- **Quickstart is wrong (lines 43–46):** says `npm install` / `npm run dev` and
  claims `npm run dev` runs "web app + API together." Reality: the repo is **pnpm**
  (`packageManager: pnpm@11.6`); the root `dev` script is `pnpm -C apps/web dev`
  (**web only**). Running both is `pnpm run dev:all` or `pnpm run up`. Fix to
  `pnpm install` + `pnpm run up` (or `dev:all`).
- **Status line (line 14)** still says "Phase R closing → M1 skeleton" — actual
  state is M1 closed + M2 spine landed.
- **Phases list (lines 33–34)** still marks R "closing (session 2)" / M1 "started".

---

## P2 — `ARCHITECTURE.md` is stale (frozen at end of session 2)

[docs/architecture/ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md) predates
sessions 3–4; several statements are now false:
- **Monorepo layout** omits the top-level **`libs/`** (`astel_format`,
  `astel_splat_io`, `astel_eval`) added in session 3.
- **CI line 18** ("web / api / stub / license-gate jobs") omits the **`manifest`**
  and **`libs`** jobs that exist in [ci.yml](.github/workflows/ci.yml).
- **Lines 38–40 & 58:** "renders the checked-in sample regardless of generation
  state" / "Object storage = none … No artifacts produced until M2" — **false now**:
  session 4 added `LocalArtifactStore` + `producer.py`; `l3.ply` +
  `quality-report.json` are produced per task and the viewer loads the per-task
  `l3.ply` on success.
- **Lines 44–48 & 59:** TaskEngine "Next session a TemporalTaskEngine implements…"
  / "Temporal binary integration is next session's work" — **done in session 3**
  (`TemporalTaskEngine` + `temporal/` package exist).
- **Verified-checks counts (lines 63–71)** are stale: vitest 9 → **15**, api 8 →
  **17**; missing the `libs` and `manifest` suites entirely.

---

## P3 — Known gaps (session-5 slice: DONE except where noted)

Resolved in session 5 (CPU-only):
- ✅ Producer now writes the full stub layer stack: `l0.ply` (strided subsample
  of L3), `l3.ply`, `l3.spz`, `l3.sog`, `package.astel` (via
  `astel_format.build_minimal_package`, schema-validated, L0+L3 bound with a
  per-gaussian provenance channel), plus the `astel.quality-report/v0`
  `quality-report.json` the Truth Meter consumes. Content-type map extended for
  `.spz`/`.sog`/`.astel`.
- ✅ `POST /v1/captures` multipart endpoint added (stores raw bytes in the
  artifact store under a `capture-<uuid>` namespace, returns a `CaptureRef`).
  The web Image/Video dock uploads the dropped file there first, then threads
  the returned `capture_id` into `POST /v1/generations` (persisted on the
  `generations` row as a nullable `capture_id` column).

Still open (deferred, not in this slice):
- The producer does **not consume** the capture bytes yet — it still emits the
  procedural splat regardless of the uploaded image/video. Consuming captures =
  the M2 GPU reconstruction path.
- `.sog` is **best-effort**: uniform-quantile codebooks + no spatial sort
  (documented in `astel_splat_io.sog`), so it round-trips but with higher
  quantization error than reference k-means SOGS.
- Artifacts are produced **synchronously at submit** in stub mode (durable async =
  the untouched Temporal path).
- Temporal engine is **untested on this box** — the 1 skipped api test is
  temporal-gated (needs the `temporal` CLI + `ASTEL_TEMPORAL_TESTS=1`).
