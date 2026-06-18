"""Unit tests for the preview/refine credit-metering logic (CLAUDE.md §7).

Pure logic — no FastAPI, no storage. Covers the layer→artifact mapping, the
mode-tier billing rules (preview cheap, refine the main spend, refine_of bills
only the increment), the measured-LLM fold-in, the exploration-is-cheap
invariant, and the public schedule shape.
"""

from __future__ import annotations

from astel_api.billing import (
    _ARTIFACT_LAYER,
    CREDIT_USD,
    PREVIEW_CODES,
    REFINE_CODES,
    SCHEDULE,
    layers_from_artifacts,
    price_generation,
    schedule_dict,
)

# A full stub layer-stack delivery (what produce_artifacts writes).
_FULL_ARTIFACTS = [
    "l0.ply",
    "l3.ply",
    "l3.spz",
    "l3.sog",
    "package.astel",
    "quality-report.json",
    "generation-spec.json",
]


def test_layers_from_artifacts_maps_and_ignores() -> None:
    layers = layers_from_artifacts(_FULL_ARTIFACTS)
    assert layers == {"L0", "L3"}  # spz/sog/package all collapse to L3
    # Non-billable bookkeeping files contribute nothing.
    assert layers_from_artifacts(["quality-report.json", "credit-ledger.json"]) == set()


def test_preview_bills_only_cheap_tier() -> None:
    ledger = price_generation(mode="preview", delivered_artifacts=_FULL_ARTIFACTS)
    codes = {i.code for i in ledger.items}
    assert codes == {"L0"}  # only L0 of the preview tier was delivered by the stub
    assert all(i.tier == "preview" for i in ledger.items)
    # L3 was delivered but not billed in preview mode → flagged honestly.
    assert any("not billed" in c for c in ledger.caveats)


def test_refine_standalone_bills_preview_plus_refine() -> None:
    ledger = price_generation(mode="refine", delivered_artifacts=_FULL_ARTIFACTS)
    codes = {i.code for i in ledger.items}
    assert codes == {"L0", "L3"}
    # No "unbilled delivered layer" caveat: standalone refine pays for everything.
    assert not any("not billed" in c for c in ledger.caveats)


def test_l6_layer_billed_as_addon_when_delivered() -> None:
    # M4: when the physics-material stage delivers l6.json, a standalone refine
    # charges the L6 add-on on top of L0+L3.
    artifacts = [*_FULL_ARTIFACTS, "l6.json"]
    ledger = price_generation(mode="refine", delivered_artifacts=artifacts)
    codes = {i.code for i in ledger.items}
    assert codes == {"L0", "L3", "L6"}
    l6 = next(i for i in ledger.items if i.code == "L6")
    assert l6.tier == "addon"
    assert l6.credits == 4.0
    # A preview never charges an add-on, even if l6.json happens to be present.
    preview = price_generation(mode="preview", delivered_artifacts=artifacts)
    assert "L6" not in {i.code for i in preview.items}


def test_refine_of_bills_only_the_increment() -> None:
    ledger = price_generation(
        mode="refine", delivered_artifacts=_FULL_ARTIFACTS, refine_of="prev-task"
    )
    codes = {i.code for i in ledger.items}
    assert codes == {"L3"}  # the preview (L0) was already paid on the prior task
    assert ledger.refine_of == "prev-task"


def test_preview_is_cheaper_than_refine() -> None:
    """The whole psychology: exploring (preview) costs a fraction of a refine."""
    preview = price_generation(mode="preview", delivered_artifacts=_FULL_ARTIFACTS)
    refine = price_generation(mode="refine", delivered_artifacts=_FULL_ARTIFACTS)
    assert preview.total_credits < refine.total_credits
    # L3 dominates the spend: the hero refine costs more than the entire
    # preview tier put together.
    l3 = next(p for p in SCHEDULE if p.code == "L3")
    preview_tier_total = sum(p.credits for p in SCHEDULE if p.tier == "preview")
    assert l3.credits >= 2 * preview_tier_total


def test_llm_cost_folds_in_as_credits() -> None:
    ledger = price_generation(
        mode="refine",
        delivered_artifacts=_FULL_ARTIFACTS,
        llm_cost_usd=0.023,
    )
    llm_items = [i for i in ledger.items if i.code == "LLM_SPEC"]
    assert len(llm_items) == 1
    # 0.023 USD at $0.01/credit ceils to 3 credits.
    assert llm_items[0].credits == 3.0
    assert llm_items[0].usd == 0.023
    assert llm_items[0].tier == "llm"


def test_zero_or_none_llm_cost_adds_no_line() -> None:
    for cost in (None, 0.0):
        ledger = price_generation(
            mode="refine", delivered_artifacts=_FULL_ARTIFACTS, llm_cost_usd=cost
        )
        assert not any(i.code == "LLM_SPEC" for i in ledger.items)


def test_totals_are_consistent() -> None:
    ledger = price_generation(
        mode="refine", delivered_artifacts=_FULL_ARTIFACTS, llm_cost_usd=0.02
    )
    assert ledger.total_credits == sum(i.credits for i in ledger.items)
    assert ledger.total_usd == round(sum(i.usd for i in ledger.items), 6)
    assert ledger.credit_usd_rate == CREDIT_USD


def test_refine_without_l3_is_flagged() -> None:
    ledger = price_generation(mode="refine", delivered_artifacts=["l0.ply"])
    assert any("no L3" in c for c in ledger.caveats)


def test_ledger_to_dict_round_trips() -> None:
    ledger = price_generation(mode="refine", delivered_artifacts=_FULL_ARTIFACTS)
    d = ledger.to_dict()
    assert d["schema"] == "astel.credit-ledger/v0"
    assert d["mode"] == "refine"
    assert {i["code"] for i in d["items"]} == {"L0", "L3"}


def test_schedule_dict_shape() -> None:
    sched = schedule_dict()
    assert sched["credit_usd_rate"] == CREDIT_USD
    codes = {layer["code"] for layer in sched["layers"]}
    assert {"L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "PRINT"} <= codes
    assert set(sched["modes"]["preview"]) == set(PREVIEW_CODES)
    assert set(sched["modes"]["refine"]) == set(REFINE_CODES)
    assert sched["notes"]


# Artifact names that some real producer path (stub or GPU, services/api or
# pipelines/gpu) currently writes, as of this audit. Cross-checked against:
# - services/api/src/astel_api/producer.py (CPU stub)
# - services/api/src/astel_api/physics_material_stage.py (l6.json)
# - pipelines/gpu/src/astel_gpu/{packaging,generative}.py (GPU path)
_PRODUCED_ARTIFACTS: frozenset[str] = frozenset(
    {
        "l0.ply",
        "l2.ply",
        "l3.ply",
        "l3.spz",
        "l3.sog",
        "l3.glb",
        "package.astel",
        "l5.stl",
        "l5-mass.json",
        "l6.json",
    }
)

# Layer codes mapped in billing._ARTIFACT_LAYER that no producer path emits
# yet (audit recommendation #8). Each entry here is a deliberate "sold but not
# yet shippable" gap tracked in docs/research/15-pipeline-wiring-audit.md §3 —
# removing a key from this set requires a producer to actually start writing
# that artifact.
_NOT_YET_IMPLEMENTED: frozenset[str] = frozenset(
    {"l1.ply", "l4.ply", "l7.ply", "print.3mf", "print.stl"}
)


def test_artifact_layer_entries_are_produced_or_tracked_as_missing() -> None:
    """Every billable artifact is either real or explicitly flagged as a gap.

    Prevents silent drift between the pricing schedule (which markets L1/L4/L7
    and print prep as purchasable add-ons, billing.SCHEDULE) and what any
    producer path can actually deliver (audit §2.3/rec #8). If a key is in
    neither bucket, either a producer now writes it (move it to
    ``_PRODUCED_ARTIFACTS``) or it's a new dead-config entry that must be
    tracked in ``_NOT_YET_IMPLEMENTED``.
    """
    for name in _ARTIFACT_LAYER:
        assert name in _PRODUCED_ARTIFACTS or name in _NOT_YET_IMPLEMENTED, (
            f"{name!r} is mapped in billing._ARTIFACT_LAYER but is neither "
            "produced by a known path nor listed in _NOT_YET_IMPLEMENTED"
        )

    # The two buckets must not overlap, and together must cover every key —
    # otherwise this test could pass vacuously if _ARTIFACT_LAYER shrinks.
    assert frozenset() == _PRODUCED_ARTIFACTS & _NOT_YET_IMPLEMENTED
    assert set(_ARTIFACT_LAYER) == _PRODUCED_ARTIFACTS | _NOT_YET_IMPLEMENTED
