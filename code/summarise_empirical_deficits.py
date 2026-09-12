"""Empirical deficit distribution of the matched watermarked arms.

tab:deficit-regime reported these quantiles with no stored artifact behind
them; they were computed once and never written down, so nothing could check
the table and nothing could reproduce it.  This produces the artifact.

The deficit is ``1 - max_w p_{t,w}`` read from the recorded float32 top
probabilities of the WATERMARKED arm, which is the quantity the deficit prior
is a prior for.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matched_data_loader as matched

MATCHED = (Path(__file__).resolve().parents[1] / "results" /
           "bayesian_paper_benchmark" / "real_model" / "temperature_matched")
MODELS = {"1p3B": "OPT-1.3B", "2p7B": "Sheared-LLaMA-2.7B"}
SUPPORT = (0.001, 0.5)


def summarise(deficits: np.ndarray) -> dict[str, float]:
    flat = np.asarray(deficits, dtype=np.float64).reshape(-1)
    if flat.size == 0 or not np.isfinite(flat).all() or np.any((flat < 0) | (flat > 1)):
        raise ValueError("deficits must be nonempty, finite and in [0, 1]")
    low, high = SUPPORT
    return {
        "n": int(flat.size),
        "median": float(np.median(flat)),
        "mean": float(flat.mean()),
        "q90": float(np.quantile(flat, 0.9)),
        "below_support": float(np.mean(flat < low)),
        "in_support": float(np.mean((flat >= low) & (flat <= high))),
        "above_half": float(np.mean(flat > high)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scratch", type=Path, default=MATCHED)
    ap.add_argument("--allow-legacy-archives", action="store_true")
    ap.add_argument("--output", type=Path, help="default: scratch/empirical_deficits.json")
    args = ap.parse_args(argv)
    args.output = args.output or args.scratch / "empirical_deficits.json"

    out: dict[str, dict[str, dict[str, float]]] = {}
    for model in MODELS:
        out[model] = {}
        # Validate complete paired coverage before pooling base and extension.
        for cell in matched.load_cells(
            args.scratch, model, allow_legacy=args.allow_legacy_archives
        ):
            out[model][cell.temperature] = summarise(
                1.0 - np.asarray(cell.watermarked["top_probs"], dtype=np.float64)
            )
        print(f"  {model}: {sorted(out[model], key=float)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "description": "Empirical deficit 1 - max_w p of the matched watermarked arms.",
        "deficit_prior_support": list(SUPPORT),
        "models": MODELS,
        "empirical_deficit_by_temperature": out,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
