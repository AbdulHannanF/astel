"""CLI entrypoint for the Astel blind-eval harness.

Usage::

    python -m astel_eval list
    python -m astel_eval run --results-dir results/
    python -m astel_eval score --pairwise ratings.csv [--bootstrap 1000]
    python -m astel_eval gate --pairwise ratings.csv [--bootstrap 200]

See ``docs/eval/CORPUS.md`` for the frozen corpus and protocol this implements.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from astel_eval.bradley_terry import fit_per_axis
from astel_eval.corpus import THIN_STRUCTURE_ANCHOR_TRIO, load_corpus
from astel_eval.gate import evaluate_gate
from astel_eval.ratings_io import load_pairwise
from astel_eval.runner import default_plan, run_plan


def _cmd_list(_args: argparse.Namespace) -> int:
    cases = load_corpus()
    print(f"{len(cases)} cases loaded from corpus_v1.json")
    by_modality: dict[str, int] = {}
    for c in cases:
        by_modality[c.modality] = by_modality.get(c.modality, 0) + 1
    for modality, count in sorted(by_modality.items()):
        print(f"  {modality}: {count}")
    print(f"thin-structure anchor trio: {', '.join(THIN_STRUCTURE_ANCHOR_TRIO)}")
    print()
    for c in cases:
        tags = []
        if c.thin_structure:
            tags.append("thin-structure")
        if c.ground_truth_scale is not None:
            tags.append("has-ground-truth-scale")
        tag_str = f"  [{', '.join(tags)}]" if tags else ""
        print(f"{c.id:>4}  ({c.modality:>7}){tag_str}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    plan = default_plan()
    results_dir = Path(args.results_dir)
    print(
        f"Running {plan.job_count} jobs "
        f"({len(plan.cases)} cases x {len(plan.systems)} systems) "
        f"-> {results_dir}"
    )
    results = run_plan(plan, results_dir, overwrite=args.overwrite)
    n_available = sum(1 for r in results if r.available)
    n_errors = sum(1 for r in results if r.status == "error")
    print(f"Done. {len(results)} job results written.")
    print(
        f"  available=True: {n_available} "
        f"(0 expected -- all adapters are STUBS, see CLAUDE.md honesty rules)"
    )
    print(f"  errors: {n_errors}")
    if n_available > 0:
        print(
            "WARNING: some jobs report available=True from STUB adapters. "
            "This should never happen -- investigate before trusting results."
        )
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    pairwise = load_pairwise(Path(args.pairwise))
    if not pairwise:
        print("No pairwise records loaded -- nothing to score.")
        return 1
    per_axis = fit_per_axis(pairwise, n_bootstrap=args.bootstrap)
    for axis, result in per_axis.items():
        print(f"\n=== {axis} ===  ({result.n_comparisons} comparisons)")
        order = sorted(
            range(len(result.systems)),
            key=lambda i: result.strengths[i],
            reverse=True,
        )
        for i in order:
            sysname = result.systems[i]
            strength = result.strengths[i]
            lo, hi = result.ci_low[i], result.ci_high[i]
            print(f"  {sysname:>12}: {strength:6.3f}  (95% CI [{lo:.3f}, {hi:.3f}])")
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    cases = load_corpus()
    pairwise = load_pairwise(Path(args.pairwise))
    result = evaluate_gate(cases, pairwise, n_bootstrap=args.bootstrap)

    print("=== M3 Gate Evaluation (CORPUS.md §4.4) ===\n")
    print(
        f"TRELLIS.2: Astel wins {result.trellis2_wins}/{result.trellis2_total} "
        f"cases (need 50/50) -> {'PASS' if result.trellis2_pass else 'FAIL'}"
    )
    if result.trellis2_losses:
        print(f"  Losing/non-winning cases: {', '.join(result.trellis2_losses)}")

    print(
        f"\nMeshy-free: Astel wins {result.meshy_wins}/{result.meshy_total} "
        f"cases (need >=26/50) -> {'PASS' if result.meshy_pass else 'FAIL'}"
    )
    if result.meshy_losses:
        print(f"  Losing/non-winning cases: {', '.join(result.meshy_losses)}")

    if result.missing_cases:
        print(
            f"\nWARNING: {len(result.missing_cases)} case(s) have no usable "
            f"pairwise data for one or both baselines and were excluded from "
            f"totals: {', '.join(result.missing_cases)}"
        )

    print(f"\nOVERALL GATE: {'PASS' if result.overall_pass else 'FAIL'}")
    return 0 if result.overall_pass else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="astel_eval")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List the frozen corpus cases.")
    p_list.set_defaults(func=_cmd_list)

    p_run = sub.add_parser(
        "run", help="Run the (stubbed) generation plan over the corpus."
    )
    p_run.add_argument(
        "--results-dir",
        default="results",
        help="Directory to write per-job JSON results (default: ./results).",
    )
    p_run.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run jobs even if a result file already exists.",
    )
    p_run.set_defaults(func=_cmd_run)

    p_score = sub.add_parser(
        "score", help="Fit Bradley-Terry strengths per axis from a ratings file."
    )
    p_score.add_argument(
        "--pairwise", required=True, help="Path to a pairwise ratings CSV/JSON file."
    )
    p_score.add_argument(
        "--bootstrap", type=int, default=1000, help="Number of bootstrap resamples."
    )
    p_score.set_defaults(func=_cmd_score)

    p_gate = sub.add_parser(
        "gate", help="Evaluate the M3 gate (CORPUS.md §4.4) from a ratings file."
    )
    p_gate.add_argument(
        "--pairwise", required=True, help="Path to a pairwise ratings CSV/JSON file."
    )
    p_gate.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="Number of bootstrap resamples per case.",
    )
    p_gate.set_defaults(func=_cmd_gate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
