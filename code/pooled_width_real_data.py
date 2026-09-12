"""The pooled tail-width rule on the temperature-matched arms.

The synthetic sweep found nothing: pooling the width differs from the frozen
ladder by at most .0016 anywhere, inside Monte Carlo error.  That is a result
about the generating laws in that sweep, where the width is drawn from the
ladder the detector already averages over.  The released text is the case the
sweep cannot speak to, so the rule is run here too rather than declared
uninteresting by extrapolation.

Only the summary horizon is scored.  The pooled width carries 3168 components
against the deficit rule's 49, so full prefix curves would cost hours to answer
a question the endpoint already answers; the AUC at 100 tokens is what the
matched tables report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import dirichlet_detector as dd
import plot_temperature_prefix_curves as ptc
import width_hierarchy as wh

MATCHED = (Path(__file__).resolve().parents[1] / "results" /
           "bayesian_paper_benchmark" / "real_model" / "temperature_matched")
WINDOW = ("0.2", "0.3", "0.4", "0.5")
REPLICATES = 2_000
SEED = 20260902


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replicates", type=int, default=REPLICATES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--models", nargs="+", default=["1p3B", "2p7B"])
    ap.add_argument("--scratch", type=Path, default=ptc.SCRATCH)
    ap.add_argument("--output", type=Path, default=MATCHED / "pooled_width_real_data.json")
    ap.add_argument("--allow-legacy-archives", action="store_true",
                    help="allow only the eight SHA-allowlisted historical NPZ archives")
    args = ap.parse_args(argv)
    if args.replicates < 2:
        ap.error("replicates must be at least two")
    if len(set(args.models)) != len(args.models) or any(model not in ptc.MODEL_VOCAB for model in args.models):
        ap.error("models must be distinct supported model names")

    captured: list[tuple] = []
    input_provenance = {}
    original = ptc.curve_rows

    def capture(model, temperature, watermarked, null, grids, **kw):
        captured.append((model, str(temperature), watermarked, null))
        return []

    ptc.curve_rows = capture
    try:
        ptc.collect(args.models, args.scratch, scheme="gumbel",
                    allow_legacy=args.allow_legacy_archives, provenance=input_provenance)
    finally:
        ptc.curve_rows = original

    delta, weights = dd.gauss_legendre_delta_grid(ptc.DELTA_LOW, ptc.DELTA_HIGH, 96)
    cells: dict[str, dict] = {}
    for i, (model, temperature, wm, null) in enumerate(captured):
        if temperature not in WINDOW:
            continue
        vocab = ptc.MODEL_VOCAB[model]
        grid = wh.PooledWidthGrid(
            delta_grid=delta, delta_weights=weights, tail_size=vocab - 1,
        )
        h = ptc.SUMMARY_HORIZON
        if wm.shape != null.shape or len(wm) < 2 or wm.shape[1] < h:
            raise ValueError("paired arms with at least two prompts and the requested horizon are required")
        w_paths = grid.shared_paths(np.clip(wm[:, :h], 1e-12, 1 - 1e-12))[:, -1]
        n_paths = grid.shared_paths(np.clip(null[:, :h], 1e-12, 1 - 1e-12))[:, -1]
        auc = ptc.auc_score(w_paths, n_paths)
        rng = np.random.default_rng(args.seed + i)
        n = w_paths.shape[0]
        draws = np.empty(args.replicates)
        for b in range(args.replicates):
            idx = rng.integers(0, n, size=n)
            draws[b] = ptc.auc_score(w_paths[idx], n_paths[idx])
        key = f"{model}_T{temperature}"
        cells[key] = {"auc": float(auc), "cluster_se": float(draws.std(ddof=1)),
                      "n": int(n)}
        print(f"  {key}: auc={auc:.4f} se={draws.std(ddof=1):.4f}", flush=True)

    if not cells:
        raise ValueError("no paired cells in the requested inference window")
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "horizon": ptc.SUMMARY_HORIZON,
        "replicates": args.replicates,
        "seed": args.seed,
        "input_provenance": input_provenance,
        "prior": wh.PooledWidthGrid(
            delta_grid=delta, delta_weights=weights,
            tail_size=ptc.MODEL_VOCAB["1p3B"] - 1,
        ).prior_fingerprint(),
        "cells": cells,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
