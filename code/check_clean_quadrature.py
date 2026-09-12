#!/usr/bin/env python3
"""Quadrature sensitivity check for the clean paper benchmark.

Reruns the full clean benchmark at 48, 96 and 192 Gauss-Legendre nodes with
every other setting fixed, and tabulates the Type II error of every Bayes rule
-- ``bayes_shared``, ``bayes_tokenwise`` and their Dirichlet and union-tail
counterparts, which are Gumbel-only -- at n = 100, 300, 700 in both
scenarios.  The comparison is paired: the node count feeds only the quadrature
grids, never the simulators, so all three runs see byte-identical pivots.

The point is to check the manuscript's claim that the three node counts give
"identical Type II errors to four decimal places".  The output records the
largest absolute discrepancy so the claim can be accepted or rejected on
evidence rather than assertion.  Plotting is skipped; ``run_benchmark`` is
called directly.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import benchmark_paper_experiment as benchmark


DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "bayesian_paper_benchmark"
)

NODE_COUNTS = (48, 96, 192)
REFERENCE_NODES = 192
HORIZONS = (100, 300, 700)
METHODS = (
    "bayes_shared",
    "bayes_tokenwise",
    "bayes_shared_dirichlet",
    "bayes_tokenwise_dirichlet",
    benchmark.UNIONTAIL_SHARED_METHOD,
    benchmark.UNIONTAIL_TOKENWISE_METHOD,
)
DECISION_RULES = ("fixed_horizon_mc_calibrated", "anytime_bf_ge_1_over_alpha")
FOUR_DECIMAL_TOLERANCE = 5e-5


def collect_type2(
    config: benchmark.BenchmarkConfig,
    horizons: tuple[int, ...] = HORIZONS,
) -> dict[tuple[str, str, str, str, int], float]:
    """Map (scenario, scheme, method, decision_rule, horizon) -> Type II error."""

    rows, _ = benchmark.run_benchmark(config)
    horizons = {h for h in horizons if h <= config.max_horizon}
    return {
        (
            str(row["scenario"]),
            str(row["scheme"]),
            str(row["method"]),
            str(row["decision_rule"]),
            int(row["horizon"]),
        ): float(row["type2_error"])
        for row in rows
        if int(row["horizon"]) in horizons
        and str(row["method"]) in METHODS
        and str(row["decision_rule"]) in DECISION_RULES
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--max-horizon", type=int, default=700)
    parser.add_argument("--n-calibration", type=int, default=10_000)
    parser.add_argument("--n-null", type=int, default=5_000)
    parser.add_argument("--n-alternative", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=24_040_1245)
    return parser.parse_args()


def identical_to_four_decimals(values: list[float]) -> bool:
    """Check the displayed values, not just distance across a rounding boundary."""

    return len({f"{value:.4f}" for value in values}) == 1


def run_check(
    base: benchmark.BenchmarkConfig,
    *,
    node_counts: tuple[int, ...] = NODE_COUNTS,
    reference_nodes: int = REFERENCE_NODES,
    horizons: tuple[int, ...] = HORIZONS,
) -> dict[str, object]:
    """Run paired decision-level checks without writing experiment artifacts."""

    started = time.perf_counter()
    if len(set(node_counts)) < 2 or reference_nodes not in node_counts:
        raise ValueError("node_counts must include the reference and another count")

    per_nodes: dict[int, dict[tuple[str, str, str, str, int], float]] = {}
    for nodes in node_counts:
        per_nodes[nodes] = collect_type2(
            replace(base, bayes_quadrature_nodes=nodes), horizons=horizons
        )

    reference = per_nodes[reference_nodes]
    if not reference:
        raise ValueError("no requested horizons are within max_horizon")
    if any(set(table) != set(reference) for table in per_nodes.values()):
        raise ValueError("quadrature runs did not return the same scored cells")
    cells: list[dict[str, object]] = []
    for key in sorted(reference):
        scenario, scheme, method, decision_rule, horizon = key
        values = {nodes: per_nodes[nodes][key] for nodes in node_counts}
        spread = max(values.values()) - min(values.values())
        cells.append(
            {
                "scenario": scenario,
                "scheme": scheme,
                "method": method,
                "decision_rule": decision_rule,
                "horizon": horizon,
                "type2_error_by_nodes": {
                    str(nodes): values[nodes] for nodes in node_counts
                },
                "abs_difference_from_reference_by_nodes": {
                    str(nodes): abs(values[nodes] - values[reference_nodes])
                    for nodes in node_counts
                },
                **{
                    f"abs_difference_{nodes}_vs_192": abs(values[nodes] - values[192])
                    for nodes in (48, 96)
                    if nodes in values and 192 in values
                },
                "max_abs_difference_across_nodes": spread,
                "within_four_decimal_tolerance": spread < FOUR_DECIMAL_TOLERANCE,
                "identical_to_four_decimals": identical_to_four_decimals(list(values.values())),
            }
        )

    fixed = [
        cell for cell in cells if cell["decision_rule"] == "fixed_horizon_mc_calibrated"
    ]
    worst = max(fixed, key=lambda cell: cell["max_abs_difference_across_nodes"])
    worst_overall = max(cells, key=lambda cell: cell["max_abs_difference_across_nodes"])
    payload: dict[str, object] = {
        "description": (
            "Paired quadrature sensitivity of the clean benchmark: node count "
            "affects only the Bayes quadrature grids, so all runs share pivots."
        ),
        "node_counts": list(node_counts),
        "reference_nodes": reference_nodes,
        "horizons": sorted({int(key[-1]) for key in reference}),
        "methods": list(METHODS),
        "coverage": {
            "quantity": "Type II error estimates after node-specific null calibration",
            "cells_checked": len(cells),
            "gumbel_cells_checked": sum(cell["scheme"] == "gumbel" for cell in cells),
            "actual_method_decision_rule_pairs": sorted({
                (str(cell["scheme"]), str(cell["method"]), str(cell["decision_rule"]))
                for cell in cells
            }),
            "union_tokenwise_state": "document-shared branch state; token-specific parameters within branch",
            "excluded": "method/scenario/decision-rule combinations not evaluated by the clean benchmark",
            "certified_uniform_error_bound": False,
        },
        "config": asdict(base),
        "four_decimal_tolerance": FOUR_DECIMAL_TOLERANCE,
        "cells": cells,
        "fixed_horizon_max_abs_difference": worst["max_abs_difference_across_nodes"],
        "fixed_horizon_worst_cell": {
            key: worst[key]
            for key in ("scenario", "scheme", "method", "horizon")
        },
        "overall_max_abs_difference": worst_overall[
            "max_abs_difference_across_nodes"
        ],
        "overall_worst_cell": {
            key: worst_overall[key]
            for key in ("scenario", "scheme", "method", "decision_rule", "horizon")
        },
        "claim_identical_to_four_decimals_fixed_horizon": all(
            cell["identical_to_four_decimals"] for cell in fixed
        ),
        "claim_identical_to_four_decimals_all_rules": all(
            cell["identical_to_four_decimals"] for cell in cells
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    return payload


def main() -> None:
    args = parse_args()
    base = benchmark.BenchmarkConfig(
        max_horizon=args.max_horizon,
        n_calibration=args.n_calibration,
        n_evaluation_null=args.n_null,
        n_evaluation_alternative=args.n_alternative,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    payload = run_check(base)

    output: Path = args.results_dir / "clean_quadrature_sensitivity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "n_cells": len(payload["cells"]),
                "runtime_seconds": payload["runtime_seconds"],
                "fixed_horizon_max_abs_difference": payload[
                    "fixed_horizon_max_abs_difference"
                ],
                "overall_max_abs_difference": payload["overall_max_abs_difference"],
                "claim_identical_to_four_decimals_fixed_horizon": payload[
                    "claim_identical_to_four_decimals_fixed_horizon"
                ],
                "claim_identical_to_four_decimals_all_rules": payload[
                    "claim_identical_to_four_decimals_all_rules"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
