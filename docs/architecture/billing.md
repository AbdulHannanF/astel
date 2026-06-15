# Billing — preview/refine credit metering

*Status: implemented (M3, session 20, 2026-06-15). Code:
`services/api/src/astel_api/billing.py`; schemas in `schemas.py`; wired in
`main.py`. This is the "preview/refine billing semantics" deliverable of M3
(CLAUDE.md §7, build plan §9 M3).*

## Why

CLAUDE.md §7 and [`docs/meshy-analysis.md`](../meshy-analysis.md) both call for
the same thing: **make exploration nearly free and charge for the keeper.** Our
layer stack (§3) maps onto this perfectly — the cheap preview layers (L0–L2) are
what a user iterates on; the expensive L3 refine is the hero asset they actually
pay for; L4–L7 and the print path are add-ons. Meshy proved the model: a cheap
`preview` task, then a `refine` task keyed on the preview's id. We mirror it.

## The credit unit

A **credit** is a notional internal accounting unit: `1 credit == 1 US cent`
(`billing.CREDIT_USD = 0.01`). This sets no external spend — it is purely how we
meter pipeline work — so it is a free product decision (not a §10.2 cost-flag
item). Every ledger reports the USD-equivalent alongside credits so the number
stays legible and honest.

## The price schedule

`GET /v1/pricing` returns the live schedule (`billing.SCHEDULE`):

| Layer | What | Tier | Credits | ≈ USD |
|-------|------|------|--------:|------:|
| L0 | Seed point cloud | preview | 1 | $0.01 |
| L1 | Dense cloud | preview | 1 | $0.01 |
| L2 | Coarse gaussians | preview | 2 | $0.02 |
| **L3** | **Refined surface gaussians** | **refine** | **20** | **$0.20** |
| L4 | Appearance / lighting | addon | 8 | $0.08 |
| L5 | Collision & solidity | addon | 6 | $0.06 |
| L6 | Physics-material & semantic | addon | 4 | $0.04 |
| L7 | Dynamics | addon | 10 | $0.10 |
| PRINT | Print prep (.3mf/.stl) | print | 12 | $0.12 |

The shape is the whole point: a full preview tier is **4 credits**, while the L3
hero alone is **20** — exploration costs a small fraction of a refine. (L4–L7 /
PRINT are reserved for M4+; they are priced now so the schedule is stable, but
the current pipeline only produces L0/L2/L3.)

## Modes

`POST /v1/generations` takes a `mode` (`"preview"` | `"refine"`, default
`"refine"`) and an optional `refine_of` (a prior preview task id).

- **`preview`** — bills only delivered preview-tier layers (L0–L2). Cheap
  exploration.
- **`refine` standalone** (`refine_of` absent) — bills the full delivered stack
  (preview + refine + add-ons): it had to produce the cheap layers itself, so it
  pays for them.
- **`refine` keyed** (`refine_of` set) — bills only the new refine/add-on work
  (L3+). The preview was already paid on the prior task, so it is **never
  re-charged**. The Generation Spec LLM stage is also skipped (it belongs to the
  preview), so a keyed refine incurs no LLM spend either.

The economics stay coherent: a `preview` (1 credit, given the stub delivers L0)
followed by a keyed `refine` (20 credits) totals **21 credits — identical to a
one-shot standalone refine** (L0 + L3 = 1 + 20). You never pay twice for the same
layer.

## The credit ledger

Every generation stores `credit-ledger.json` (schema `astel.credit-ledger/v0`)
as a downloadable artifact, and returns the same data in the `billing` field of
the `GenerationResource`. It has per-layer line items, an optional measured-LLM
line, totals (credits + USD), and honest caveats. Example (a standalone refine of
a text prompt with a cached spec fixture):

```json
{
  "schema": "astel.credit-ledger/v0",
  "mode": "refine",
  "refine_of": null,
  "items": [
    {"code": "L0", "label": "Seed point cloud", "tier": "preview", "credits": 1.0, "usd": 0.01, "detail": ""},
    {"code": "L3", "label": "Refined surface gaussians", "tier": "refine", "credits": 20.0, "usd": 0.20, "detail": ""},
    {"code": "LLM_SPEC", "label": "Generation Spec (LLM)", "tier": "llm", "credits": 3.0, "usd": 0.023, "detail": "measured token cost; billed at ceil(usd / credit_rate)"}
  ],
  "total_credits": 24.0,
  "total_usd": 0.233,
  "credit_usd_rate": 0.01,
  "caveats": []
}
```

### LLM fold-in

The Generation Spec stage (`generation_spec_stage.py`) already produces a
measured token-cost ledger row (`astel_llm.pricing.ledger_entry`). Billing folds
that **measured** USD cost (not an estimate) into the credit ledger as an
`LLM_SPEC` line, converted to credits with a ceiling (`ceil(usd / 0.01)`) so a
real sub-cent call always costs at least one credit. The line appears only when a
paid/fixture call actually ran this task — offline cache-misses (status
`skipped`) and keyed refines add no LLM line.

## Honesty (CLAUDE.md §10.3, §10.4)

- The current **stub producer computes the full layer stack regardless of
  tier**, so a `preview` task has an L3 file on disk it did not pay for. Rather
  than hide this, the ledger emits a caveat naming the delivered-but-unbilled
  layers. (The real M2+ GPU pipeline can gate production by tier so preview work
  genuinely stops at L2; the billing logic is already correct for that.)
- A `refine` that delivered no L3 is flagged in caveats.
- Credits are always paired with the USD-equivalent; the LLM line carries the
  measured cost, never a guess.

## API surface

- `GET /v1/pricing` → `PricingResource` (the schedule above).
- `POST /v1/generations` accepts `mode` + `refine_of`; returns `mode`,
  `refine_of`, and `billing` (`BillingSummary`) plus a `credit-ledger.json`
  artifact.
- `GET /v1/generations/{id}` returns the persisted `mode`, `refine_of`, and the
  stored billing summary.
- The `generations` table persists `mode`, `refine_of`, and total `credits`
  (Alembic migration `a1b2c3d4e5f6`).

## Not yet (follow-ons)

- No user/account/credit-balance ledger or debiting yet (no auth in the
  skeleton) — this prices a generation; it does not yet *deduct from a balance*.
  That lands with users/accounts (later milestone; cf. §7 credit-metered).
- Tier-gated production (stop a preview at L2 on the GPU path) — billing is
  ready; the producer is not split yet.
- Add-on (L4–L7) and print billing are priced but unexercised until M4 produces
  those layers.
