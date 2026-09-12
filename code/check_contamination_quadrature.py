#!/usr/bin/env python3
"""Paired quadrature sensitivity check for the contamination benchmark."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import benchmark_contamination as contamination


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "bayesian_paper_benchmark"
    / "contamination_quadrature_sensitivity.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON destination (default: %(default)s)",
    )
    return parser.parse_args()


def run_check(
    *,
    n_paths: int = 300,
    node_counts: tuple[int, ...] = (48, 96, 192),
    reference_nodes: int = 192,
    seed: int = 24_040_1247,
    base: contamination.ContaminationConfig | None = None,
) -> dict[str, object]:
    """Compare paired log-BF paths without writing production artifacts.

    The contamination benchmark evaluates shared union rules, not tokenwise
    union rules.  Their shared robust version is included here alongside the
    spike and Dirichlet robust versions; clean tokenwise union decision-level
    sensitivity is covered by check_clean_quadrature.
    """

    started = time.perf_counter()
    if n_paths < 1 or len(set(node_counts)) < 2 or reference_nodes not in node_counts:
        raise ValueError("positive paths and at least two counts including the reference are required")
    if base is None:
        base = contamination.ContaminationConfig(
            n_calibration=1,
            n_evaluation_null=1,
            n_evaluation_alternative=n_paths,
            batch_size=100,
        )
    output: dict[str, object] = {
        "n_paired_paths": n_paths,
        "node_counts": list(node_counts),
        "reference_nodes": reference_nodes,
        "seed": seed,
        "config": {
            "vocabulary_size": base.vocabulary_size,
            "horizons": list(base.horizons),
            "generating_delta_support": [base.delta_low, base.delta_high],
            "prior_support": [base.prior_low, base.prior_high],
            "rho_prior_grid": list(base.rho_prior_grid),
            "rho_prior_weights": list(base.rho_prior_weights),
            "dirichlet_alpha_grid": [contamination.dirichlet.format_alpha(a) for a in base.dirichlet_alpha_grid],
            "dirichlet_c_nodes": base.dirichlet_c_nodes,
            "batch_size": base.batch_size,
        },
        "coverage": {
            "quantity": "log Bayes factors on paired alternative/replacement paths, not Type II decisions",
            "gumbel_methods": [
                "bayes_shared_robust", contamination.DIRICHLET_ROBUST_METHOD,
                contamination.UNIONTAIL_ROBUST_METHOD,
            ],
            "inverse_methods": ["bayes_shared_robust"],
            "generating_replacement_rates": [0.0, 0.4],
            "horizons": list(base.horizons),
            "tokenwise_union": "not evaluated in the contamination benchmark; see the clean checker",
            "certified_uniform_error_bound": False,
        },
        "cells": [],
    }
    scheme_seeds = np.random.SeedSequence(seed).spawn(2)
    for scheme, scheme_seed in zip(("gumbel", "inverse"), scheme_seeds):
        rng_clean, rng_null, rng_mask = [
            np.random.default_rng(seed) for seed in scheme_seed.spawn(3)
        ]
        clean = contamination._simulate(
            scheme, "alternative", rng_clean, n_paths, base
        )
        null = contamination._simulate(scheme, "null", rng_null, n_paths, base)
        masks = rng_mask.uniform(size=clean.shape)
        for rho_true in (0.0, 0.4):
            pivots = contamination.contaminated_pivots(
                clean, null, masks, rho_true
            )
            # Spike robust plus the Gumbel Dirichlet and union-tail robust
            # rules.  The tail layers are not defined for the inverse pivot.
            swept = ("bayes_shared_robust",) + (
                (contamination.DIRICHLET_ROBUST_METHOD, contamination.UNIONTAIL_ROBUST_METHOD)
                if scheme in contamination.DIRICHLET_SCHEMES
                else ()
            )
            for rule in swept:
                paths: dict[int, np.ndarray] = {}
                for nodes in node_counts:
                    config = replace(base, bayes_quadrature_nodes=nodes)
                    if rule == "bayes_shared_robust":
                        paths[nodes] = contamination.shared_bayes_selected_paths(
                            pivots, scheme, config
                        )[rule]
                    elif rule == contamination.DIRICHLET_ROBUST_METHOD:
                        paths[nodes] = (
                            contamination.dirichlet_shared_selected_paths(
                                pivots, scheme, config, config.dirichlet_grid()
                            )[rule]
                        )
                    else:
                        paths[nodes] = contamination.uniontail_shared_selected_paths(
                            pivots, scheme, config, config.uniontail_grid()
                        )[rule]
                reference = paths[reference_nodes]
                for nodes in node_counts:
                    if nodes == reference_nodes:
                        continue
                    absolute = np.abs(paths[nodes] - reference)
                    for horizon_index, horizon in enumerate(base.horizons):
                        values = absolute[:, horizon_index]
                        output["cells"].append(
                            {
                                "scheme": scheme,
                                "rule": rule,
                                "rho_true": rho_true,
                                "horizon": horizon,
                                "nodes": nodes,
                                "max_abs_log_bf_difference": float(values.max()),
                                "mean_abs_log_bf_difference": float(values.mean()),
                                "median_abs_log_bf_difference": float(
                                    np.median(values)
                                ),
                            }
                        )
    cells = output["cells"]
    output["coverage"]["cells_checked"] = len(cells)
    output["max_abs_log_bf_difference_from_reference_by_nodes"] = {
        str(nodes): max(cell["max_abs_log_bf_difference"] for cell in cells if cell["nodes"] == nodes)
        for nodes in node_counts if nodes != reference_nodes
    }
    if 48 in node_counts and reference_nodes == 192:
        output["overall_max_abs_log_bf_difference_48_vs_192"] = output[
            "max_abs_log_bf_difference_from_reference_by_nodes"
        ]["48"]
    output["runtime_seconds"] = time.perf_counter() - started
    return output


def main() -> None:
    args = parse_args()
    output = run_check()
    path = args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(path), "runtime_seconds": output["runtime_seconds"], "overall_max": output[
        "overall_max_abs_log_bf_difference_48_vs_192"
    ]}, indent=2))


if __name__ == "__main__":
    main()
