#!/usr/bin/env python3
"""Does the shared hierarchy actually learn Delta?

Section 3.3 compares the shared mixture against a fixed-Delta_0 score and finds
their power indistinguishable.  That comparison says nothing about whether the
hierarchy recovers Delta, only that recovering it does not buy extra detection
in this configuration.  This script measures the recovery directly: it generates
Gumbel documents at a known deficit and reports the shared component posterior
over the same 96-node grid the benchmark uses.

Writes results/bayesian_paper_benchmark/delta_learning.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bayesian_watermark import SequentialBayesDetector

SEED = 240_401_248
VOCABULARY = 1000
DELTA_LOW, DELTA_HIGH = 0.001, 0.5
NODES = 96
TRUE_DELTAS = (0.005, 0.05, 0.2, 0.4)
HORIZONS = (100, 300, 700)
N_DOCUMENTS = 400
CROSS_CHECK_DOCUMENTS = 3
CREDIBLE_MASS = 0.90
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "bayesian_paper_benchmark"
    / "delta_learning.json"
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


def delta_grid(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """The benchmark's Gauss-Legendre grid on (DELTA_LOW, DELTA_HIGH)."""

    points, weights = np.polynomial.legendre.leggauss(nodes)
    grid = DELTA_LOW + 0.5 * (points + 1.0) * (DELTA_HIGH - DELTA_LOW)
    return grid, weights / weights.sum()


def simulate_gumbel_pivots(
    delta: float, n_documents: int, horizon: int, rng: np.random.Generator
) -> np.ndarray:
    """Exact selected-token pivots for the equal-tail spike NTP at ``delta``."""

    uniforms = rng.random((n_documents, horizon))
    top_selected = rng.random((n_documents, horizon)) < (1.0 - delta)
    exponent = np.where(top_selected, 1.0 - delta, delta / (VOCABULARY - 1))
    return uniforms**exponent


def summarize(grid: np.ndarray, posterior: np.ndarray, truth: float) -> dict:
    mean = float(posterior @ grid)
    sd = float(np.sqrt(posterior @ (grid - mean) ** 2))
    cumulative = np.cumsum(posterior)
    tail = 0.5 * (1.0 - CREDIBLE_MASS)
    low = float(grid[int(np.argmax(cumulative >= tail))])
    high = float(grid[int(np.argmax(cumulative >= 1.0 - tail))])
    return {
        "posterior_mean": mean,
        "posterior_sd": sd,
        "credible_low": low,
        "credible_high": high,
        "covers_truth": bool(low <= truth <= high),
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(SEED)
    grid, weights = delta_grid(NODES)
    prior_mean = float(weights @ grid)
    prior_sd = float(np.sqrt(weights @ (grid - prior_mean) ** 2))

    cells = []
    log_prior = np.log(weights)
    for truth in TRUE_DELTAS:
        pivots = simulate_gumbel_pivots(truth, N_DOCUMENTS, max(HORIZONS), rng)
        log_pivots = np.log(pivots)
        # log spike-family density at every (document, grid node, token)
        tail = np.log(VOCABULARY - 1.0)
        comp = np.logaddexp(
            (grid / (1.0 - grid))[None, :, None] * log_pivots[:, None, :],
            tail + ((VOCABULARY - 1.0) / grid - 1.0)[None, :, None] * log_pivots[:, None, :],
        )
        cumulative = np.cumsum(comp, axis=2)
        for horizon in HORIZONS:
            unnormalized = log_prior[None, :] + cumulative[:, :, horizon - 1]
            unnormalized -= unnormalized.max(axis=1, keepdims=True)
            posterior = np.exp(unnormalized)
            posterior /= posterior.sum(axis=1, keepdims=True)
            rows = [summarize(grid, p, truth) for p in posterior]
            cells.append(
                {
                    "true_delta": truth,
                    "horizon": horizon,
                    "n_documents": N_DOCUMENTS,
                    "mean_posterior_mean": float(np.mean([r["posterior_mean"] for r in rows])),
                    "mean_posterior_sd": float(np.mean([r["posterior_sd"] for r in rows])),
                    "mean_credible_low": float(np.mean([r["credible_low"] for r in rows])),
                    "mean_credible_high": float(np.mean([r["credible_high"] for r in rows])),
                    "credible_coverage": float(np.mean([r["covers_truth"] for r in rows])),
                    "posterior_sd_shrinkage_vs_prior": float(
                        prior_sd / np.mean([r["posterior_sd"] for r in rows])
                    ),
                }
            )
        # Cross-check the vectorised posterior against the reference detector.
        for index in range(CROSS_CHECK_DOCUMENTS):
            detector = SequentialBayesDetector(
                scheme="gumbel", delta_grid=grid, delta_weights=weights,
                structure="shared", gumbel_family="spike", vocabulary_size=VOCABULARY,
            )
            detector.update_many(pivots[index, : HORIZONS[0]])
            reference = detector.component_posterior_weights
            unnormalized = log_prior + cumulative[index, :, HORIZONS[0] - 1]
            unnormalized -= unnormalized.max()
            mine = np.exp(unnormalized); mine /= mine.sum()
            gap = float(np.abs(mine - reference).max())
            if gap > 1e-10:
                raise AssertionError(f"vectorised posterior differs from detector by {gap:g}")

    payload = {
        "question": (
            "Does the shared hierarchy recover Delta, independently of whether "
            "recovering it improves detection power?"
        ),
        "design": {
            "scheme": "gumbel",
            "gumbel_family": "spike",
            "structure": "shared",
            "vocabulary_size": VOCABULARY,
            "delta_grid": f"{NODES} Gauss-Legendre nodes on ({DELTA_LOW}, {DELTA_HIGH})",
            "prior_mean": prior_mean,
            "prior_sd": prior_sd,
            "credible_mass": CREDIBLE_MASS,
            "n_documents_per_cell": N_DOCUMENTS,
            "seed": SEED,
        },
        "cells": cells,
    }

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf8")

    print(f"prior sd over the grid = {prior_sd:.4f}\n")
    header = f"{'true D':>8s} {'n':>5s} {'post mean':>10s} {'post sd':>9s} {'90% interval':>20s} {'cover':>6s}"
    print(header)
    for cell in cells:
        print(
            f"{cell['true_delta']:8.3f} {cell['horizon']:5d} "
            f"{cell['mean_posterior_mean']:10.4f} {cell['mean_posterior_sd']:9.4f} "
            f"  [{cell['mean_credible_low']:.4f},{cell['mean_credible_high']:.4f}] "
            f"{cell['credible_coverage']:6.2f}"
        )
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
