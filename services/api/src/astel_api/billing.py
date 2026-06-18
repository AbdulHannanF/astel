"""Credit metering for the layered pipeline — preview/refine billing (M3).

This is the "preview/refine billing semantics" half of M3 (CLAUDE.md §7,
``docs/meshy-analysis.md``). The layer stack (§3) maps directly onto credit
psychology: **L0–L2 previews are cheap** (a few cents-equivalent each, so users
can explore freely), **L3 refine is the main spend**, and **L4–L7 / print prep
are add-ons**. We mirror Meshy's two-stage model — a cheap ``preview`` task and a
follow-up ``refine`` task keyed on the preview via ``refine_of`` — so a refine
only pays for the new work, never re-charging the preview layers.

The module is deliberately pure (no FastAPI / no storage coupling): callers pass
the list of *delivered* artifact names plus any measured LLM cost, and get back a
:class:`CreditLedger`. That keeps the credit math unit-testable and lets the same
logic serve the ``/v1/pricing`` schedule endpoint and the per-task ledger.

**Honesty (CLAUDE.md §10.3).** A credit is a notional internal unit
(:data:`CREDIT_USD`); the ledger always reports the USD-equivalent alongside, and
the LLM line carries the *measured* token cost (not an estimate) when the
Generation Spec stage actually ran. Layers that were billed but, in the current
stub, computed identically regardless of tier are flagged in ``caveats``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

#: Notional USD value of one credit. 1 credit == 1 US cent keeps a preview
#: (L0–L2) in the "few cents" band the brief calls for and a full generation in
#: the ~$0.25 band — Meshy-comparable exploration economics. Internal accounting
#: unit only; it sets no external spend (cf. CLAUDE.md §10.2 cost flag).
CREDIT_USD: float = 0.01


@dataclass(frozen=True)
class LayerPrice:
    """The credit cost and tier of a single layer in the price schedule."""

    code: str
    label: str
    tier: str  # "preview" | "refine" | "addon" | "print"
    credits: float


#: The credit schedule, in pipeline order (CLAUDE.md §3 layer stack). Preview
#: layers are cheap by design; L3 (the hero refine) dominates; L4–L7 and the
#: print path are add-ons reserved for M4+. Costs are deliberately round so the
#: psychology reads at a glance: a preview is ~4 credits, a refine adds 20.
SCHEDULE: tuple[LayerPrice, ...] = (
    LayerPrice("L0", "Seed point cloud", "preview", 1.0),
    LayerPrice("L1", "Dense cloud", "preview", 1.0),
    LayerPrice("L2", "Coarse gaussians", "preview", 2.0),
    LayerPrice("L3", "Refined surface gaussians", "refine", 20.0),
    LayerPrice("L4", "Appearance / lighting", "addon", 8.0),
    LayerPrice("L5", "Collision & solidity", "addon", 6.0),
    LayerPrice("L6", "Physics-material & semantic", "addon", 4.0),
    LayerPrice("L7", "Dynamics", "addon", 10.0),
    LayerPrice("PRINT", "Print prep (.3mf/.stl)", "print", 12.0),
)

_PRICE_BY_CODE: dict[str, LayerPrice] = {p.code: p for p in SCHEDULE}

#: Which layer codes each mode is allowed to bill. ``preview`` only ever charges
#: the cheap exploration tier; ``refine`` charges the refine + add-on + print
#: tiers (and, when run standalone, also the preview layers it had to produce —
#: see :func:`_billable_codes`).
PREVIEW_CODES: frozenset[str] = frozenset({"L0", "L1", "L2"})
REFINE_CODES: frozenset[str] = frozenset({"L3", "L4", "L5", "L6", "L7", "PRINT"})

#: Maps a produced artifact filename to the layer code it realises. Multiple
#: artifacts can map to one layer (e.g. ``l3.ply``/``l3.spz``/``l3.sog`` are all
#: deliveries of L3) — billing counts the layer once, not per file.
_ARTIFACT_LAYER: dict[str, str] = {
    "l0.ply": "L0",
    "l1.ply": "L1",
    "l2.ply": "L2",
    "l3.ply": "L3",
    "l3.spz": "L3",
    "l3.sog": "L3",
    "l3.glb": "L3",  # KHR_gaussian_splatting glTF — another L3 delivery format
    "package.astel": "L3",  # the bound asset is anchored on the hero layer
    "l4.ply": "L4",
    "l5.stl": "L5",
    "l5-mass.json": "L5",
    "l6.json": "L6",
    "l7.ply": "L7",
    "print.3mf": "PRINT",
    "print.stl": "PRINT",
}


def layers_from_artifacts(names: list[str]) -> set[str]:
    """Return the set of layer codes realised by the given artifact names.

    Filenames not tied to a billable layer (``quality-report.json``,
    ``generation-spec.json``, ``credit-ledger.json``, ...) are ignored.
    """
    return {_ARTIFACT_LAYER[n] for n in names if n in _ARTIFACT_LAYER}


def _billable_codes(mode: str, refine_of: str | None) -> frozenset[str]:
    """The layer codes ``mode`` may charge for.

    - ``preview`` → only the cheap preview tier.
    - ``refine`` keyed on a prior preview (``refine_of`` set) → only the new
      refine/add-on work; the preview was already paid.
    - ``refine`` standalone → preview + refine (it had to produce the cheap
      layers itself, so it pays for them too).
    """
    if mode == "preview":
        return frozenset(PREVIEW_CODES)
    if refine_of is not None:
        return REFINE_CODES
    return PREVIEW_CODES | REFINE_CODES


def _credits_to_usd(credits: float) -> float:
    return round(credits * CREDIT_USD, 6)


def _usd_to_credits(usd: float) -> float:
    """Convert a measured USD cost to credits, rounded up to the cent/credit.

    Ceiling avoids charging zero credits for a real (tiny) sub-cent LLM call —
    a billed call always costs at least one credit.
    """
    return float(math.ceil(usd / CREDIT_USD)) if usd > 0 else 0.0


@dataclass(frozen=True)
class LineItem:
    """One charge on a generation's credit ledger."""

    code: str
    label: str
    tier: str
    credits: float
    usd: float
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "label": self.label,
            "tier": self.tier,
            "credits": self.credits,
            "usd": self.usd,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CreditLedger:
    """The full credit accounting for one generation task."""

    mode: str
    refine_of: str | None
    items: list[LineItem]
    total_credits: float
    total_usd: float
    credit_usd_rate: float
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "astel.credit-ledger/v0",
            "mode": self.mode,
            "refine_of": self.refine_of,
            "items": [i.to_dict() for i in self.items],
            "total_credits": self.total_credits,
            "total_usd": self.total_usd,
            "credit_usd_rate": self.credit_usd_rate,
            "caveats": self.caveats,
        }


def price_generation(
    *,
    mode: str,
    delivered_artifacts: list[str],
    llm_cost_usd: float | None = None,
    refine_of: str | None = None,
) -> CreditLedger:
    """Build the credit ledger for one generation.

    Bills each delivered layer that ``mode`` is allowed to charge for (see
    :func:`_billable_codes`), folds in the measured LLM cost (the Generation
    Spec stage) as a credit line, and totals both credits and USD-equivalent.

    Args:
        mode: ``"preview"`` or ``"refine"``.
        delivered_artifacts: artifact filenames actually written for the task.
        llm_cost_usd: measured USD cost of the spec-stage LLM call this task, or
            ``None`` when no paid call ran (offline fixture / non-text / refine).
        refine_of: prior preview task id when this is a follow-up refine.
    """
    delivered = layers_from_artifacts(delivered_artifacts)
    billable = _billable_codes(mode, refine_of)
    items: list[LineItem] = []

    for price in SCHEDULE:  # schedule order → deterministic, pipeline-ordered
        if price.code in delivered and price.code in billable:
            items.append(
                LineItem(
                    code=price.code,
                    label=price.label,
                    tier=price.tier,
                    credits=price.credits,
                    usd=_credits_to_usd(price.credits),
                )
            )

    caveats: list[str] = []
    # The LLM line bills the *measured* token cost of the spec stage. It is
    # preview-tier work (it conditions the whole generation), so a refine keyed
    # on a prior preview never re-charges it — callers pass llm_cost_usd=None.
    if llm_cost_usd is not None and llm_cost_usd > 0:
        llm_credits = _usd_to_credits(llm_cost_usd)
        items.append(
            LineItem(
                code="LLM_SPEC",
                label="Generation Spec (LLM)",
                tier="llm",
                credits=llm_credits,
                usd=round(llm_cost_usd, 6),
                detail="measured token cost; billed at ceil(usd / credit_rate)",
            )
        )

    delivered_unbilled = delivered - billable
    if delivered_unbilled:
        # Honesty: the stub producer computes the full stack regardless of tier,
        # so a preview task may have L3 on disk it did not pay for. That is fine
        # (the refine is what's sold) but must not be hidden.
        caveats.append(
            "Some delivered layers were not billed for this tier "
            f"({', '.join(sorted(delivered_unbilled))}); the stub producer "
            "computes all layers regardless of mode."
        )
    missing_billable = (billable & {p.code for p in SCHEDULE}) - delivered
    if mode == "refine" and "L3" in missing_billable:
        caveats.append("Refine requested but no L3 artifact was delivered.")

    total_credits = round(sum(i.credits for i in items), 6)
    total_usd = round(sum(i.usd for i in items), 6)
    return CreditLedger(
        mode=mode,
        refine_of=refine_of,
        items=items,
        total_credits=total_credits,
        total_usd=total_usd,
        credit_usd_rate=CREDIT_USD,
        caveats=caveats,
    )


def schedule_dict() -> dict[str, Any]:
    """The public price schedule for the ``/v1/pricing`` endpoint."""
    return {
        "schema": "astel.pricing/v0",
        "credit_usd_rate": CREDIT_USD,
        "layers": [
            {
                "code": p.code,
                "label": p.label,
                "tier": p.tier,
                "credits": p.credits,
            }
            for p in SCHEDULE
        ],
        "modes": {
            "preview": sorted(PREVIEW_CODES),
            "refine": sorted(REFINE_CODES),
        },
        "notes": [
            "L0–L2 previews are intentionally cheap so exploration is nearly "
            "free; L3 refine is the main spend; L4–L7 and print prep are "
            "add-ons (CLAUDE.md §7).",
            "A refine keyed on a prior preview (refine_of) bills only the new "
            "refine/add-on layers — the preview is never re-charged.",
            "1 credit == 1 US cent (notional internal accounting unit).",
        ],
    }
