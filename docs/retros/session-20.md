# Session 20 retro (2026-06-15)

**M3 CLOSED — preview/refine credit-metering (the billing-semantics deliverable).**
The generative path (sessions 11–16) and the Generation Spec LLM stage
(sessions 15, 17) were already done; this session built the third and final M3
piece from the build plan (§9 M3: "preview/refine billing semantics"), so M3 is
now complete end-to-end.

Mode: Opus, inline. CPU only (no GPU needed). No founder gate touched.

## 1. What shipped

- **`services/api/src/astel_api/billing.py`** — a pure, FastAPI-free credit
  module. `SCHEDULE` prices every layer by tier (preview L0–L2 cheap / refine L3
  the main spend / addon L4–L7 / print), `1 credit == 1¢` (`CREDIT_USD`).
  `price_generation(...)` returns a `CreditLedger` from the delivered artifact
  names ∩ the mode's billable set; `layers_from_artifacts` maps filenames →
  layer codes (l3.ply/spz/sog/package.astel all collapse to L3, billed once);
  `schedule_dict()` feeds the pricing endpoint.
- **Meshy two-stage model.** `POST /v1/generations` gains `mode`
  (`preview`|`refine`, default `refine`) + optional `refine_of`. A keyed refine
  bills only the L3+ increment, never re-charging the preview, and skips the LLM
  spec stage (it belongs to the preview) so it incurs no LLM spend.
- **Ledger + endpoints.** Every generation stores `credit-ledger.json` (schema
  `astel.credit-ledger/v0`) and returns a `billing` summary; `GET /v1/pricing`
  publishes the schedule; `get_generation` returns the persisted billing.
- **LLM fold-in.** The Generation Spec stage's *measured* token cost
  (`astel_llm` ledger) becomes an `LLM_SPEC` credit line, `ceil(usd/0.01)` so a
  real sub-cent call costs ≥1 credit; absent on offline cache-miss / keyed
  refine.
- **Persistence.** `generations` gains `mode`/`refine_of`/`credits`; Alembic
  migration `a1b2c3d4e5f6` (+ `create_all` for dev/test). Verified the migration
  applies on a fresh DB to exactly the expected columns.
- **Docs.** New [`architecture/billing.md`](../architecture/billing.md);
  DECISIONS.md session-20 section; this retro; NEXT_STEPS updated.

## 2. Verified — live HTTP (uvicorn, real requests)

Booted the app and hit it over the wire (not just the ASGI test transport):

- `GET /v1/pricing` → full 9-layer schedule, `credit_usd_rate` 0.01.
- `preview` text gen → billing **1 credit** (only L0 delivered by the stub),
  caveat naming the delivered-but-unbilled L3 (honesty channel intact).
- `refine` keyed on that preview → **20 credits** (L3 only), `refine_of` echoed,
  no L0 re-charge.
- standalone `refine` → **21 credits** (L0 1 + L3 20).
- ∴ preview (1) + keyed-refine (20) = 21 = standalone refine — **no double
  billing**, the core invariant.

Gates green: API **ruff** · **mypy --strict (23 files)** · **51 pytest**
(+16 new: 11 unit in `test_billing.py`, 5 endpoint in `test_api.py`), 1 skipped.

## 3. Honest gaps / carried forward

- **Prices a generation; does not yet debit a balance.** No users/accounts/auth
  in the skeleton, so there is no credit *balance* to deduct from — that lands
  with the accounts milestone (§7 credit-metered). This is the metering half.
- **Stub computes the full stack regardless of tier**, so a preview has an
  unpaid L3 on disk. Surfaced as a ledger caveat, not hidden; the real GPU path
  can gate production by tier later (the billing logic is already correct for
  that split).
- **Add-on (L4–L7) + print billing priced but unexercised** until M4 produces
  those layers. The schedule is published now so it stays stable.
- The credit unit (1¢) and the specific per-layer credit values are a first
  calibration (Meshy-comparable economics); revisit against real GPU cost
  telemetry once measured.

## 4. Next

M4 world-awareness continues: L6 physics-material LLM pass (reuse `astel_llm`),
L4 relighting, metric-scale L5, CoACD + `.3mf` + printability, bind L5 as a
manifest layer. Billing add-on tiers are ready to meter those as they land.
