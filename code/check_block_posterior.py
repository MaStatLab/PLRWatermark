"""Posterior mass on the union tail's two blocks as evidence accumulates.

Section~\\ref{sec:tail-width} claims the mixing weight is a starting point
rather than a fixed assumption, because the shared component posterior
reweights the two blocks from the data.  That claim had no stored artifact;
this produces one.

Under an equal-tail generator the Dirichlet (shape) block is the correct one at
full width, so its posterior mass should climb from the prior .5 toward one, and
should climb faster when the per-token signal is stronger.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import dirichlet_detector as dd
import tail_family as tf

RESULTS = Path(__file__).resolve().parents[1] / "results" / "bayesian_paper_benchmark"
VOCAB = 1_000
SEED = 20260902


def block_posterior(grid, pivots: np.ndarray, horizons) -> dict[str, float]:
    weights = np.asarray(grid.log_component_weights, dtype=float)
    per = grid.component_log_ratio(pivots.reshape(-1)).reshape(
        pivots.shape[0], pivots.shape[1], -1
    )
    # The prior weight enters ONCE, not once per token: adding it inside the
    # cumulative sum multiplies it n times and drives the posterior toward
    # whichever block has fewer components.
    cumulative = np.cumsum(per, axis=1) + weights[None, None, :]
    out = {}
    for h in horizons:
        col = cumulative[:, h - 1, :]
        shape = dd._logsumexp_rows(col[:, grid.block_slice(0)])
        width = dd._logsumexp_rows(col[:, grid.block_slice(1)])
        # Posterior mass on the shape block, averaged over documents.
        out[str(h)] = float(np.mean(1.0 / (1.0 + np.exp(width - shape))))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--documents", type=int, default=2_000)
    ap.add_argument("--horizons", type=int, nargs="+", default=[50, 100, 200])
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.005, 0.1])
    ap.add_argument("--output", type=Path, default=RESULTS / "block_posterior.json")
    args = ap.parse_args(argv)

    delta, weights = dd.gauss_legendre_delta_grid(0.001, 0.5, 96)
    grid = dd.UnionTailBayesGrid(
        delta_grid=delta, delta_weights=weights,
        tail_size=VOCAB - 1, dirichlet_weight=0.5,
    )
    horizon = max(args.horizons)
    cells = {}
    for i, value in enumerate(args.deltas):
        rng = np.random.default_rng(SEED + i)
        deltas = np.full((args.documents, horizon), float(value))
        # Equal-tail generator: the shape block at full width is correct.
        pivots = tf.simulate_narrow_width_pivots(rng, deltas, VOCAB - 1)
        cells[f"delta={value:g}"] = block_posterior(grid, pivots, args.horizons)
        print(f"  Delta={value:g}: " + "  ".join(
            f"n={h}:{cells[f'delta={value:g}'][str(h)]:.3f}" for h in args.horizons
        ), flush=True)

    args.output.write_text(json.dumps({
        "description": "Mean posterior mass on the union tail's Dirichlet (shape) block.",
        "generator": "equal tail, so the shape block at full width is correct",
        "prior_weight_on_shape_block": 0.5,
        "documents": args.documents, "vocabulary_size": VOCAB, "seed": SEED,
        "cells": cells,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
