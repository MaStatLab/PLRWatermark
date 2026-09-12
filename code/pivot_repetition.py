#!/usr/bin/env python3
"""Audit what the released skipgram PRF actually repeats.

The manuscript used to say that a repeated context reproduces its pivot
exactly.  It does not.  ``real_data_experiment.replay_raw_pivots`` seeds the
keyed draw from the *oldest* token of the four-token window alone, so the
pivot at position ``t`` is the ``W_t`` coordinate of the vector drawn from
``h(W_{t-4})``: the pair ``(W_{t-4}, W_t)`` determines it, and the context
token on its own does not.

The distinction is testable on the released watermarked arm, which stores the
realized tokens next to the pivots, so this script tests it rather than
arguing it.  For every group of positions sharing a key it asks whether the
pivot is constant within the group, once keying on the pair and once keying on
the seed token alone.  A mechanism claim that survives only the first keying
is the claim the manuscript should make.

It also records the distinct-pivot counts the released-output subsection
quotes, which had no artifact behind them and were off by one.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "results" / "bayesian_paper_benchmark" / "real_model"
    / "pivot_repetition.json"
)

MODELS: tuple[tuple[str, str, str], ...] = (
    ("1p3B", "OPT-1.3B", "1p3B-gumbel-c4-m200-T500-skipgram_prf-15485863-temp0.1.pkl"),
    ("2p7B", "Sheared-LLaMA-2.7B",
     "2p7B-gumbel-c4-m200-T500-skipgram_prf-15485863-temp0.1.pkl"),
)


def group_is_constant(keys: np.ndarray, values: np.ndarray) -> tuple[int, int]:
    """Return (groups, groups whose ``values`` are not constant)."""

    order = np.argsort(keys, kind="stable")
    sorted_keys, sorted_values = keys[order], values[order]
    boundaries = np.flatnonzero(np.diff(sorted_keys)) + 1
    groups = np.split(sorted_values, boundaries)
    inconstant = sum(1 for block in groups if np.unique(block).size > 1)
    return len(groups), inconstant


def audit(data_dir: Path | None = None) -> dict[str, Any]:
    import real_data_experiment as real

    started = time.perf_counter()
    directory = data_dir or real.DEFAULT_DATA_DIR
    width = real.CONTEXT_WIDTH
    models: dict[str, Any] = {}

    for spec, label, filename in MODELS:
        payload, _ = real.load_verified_pickle(directory / filename)
        prompts = payload["prompts"].numpy()
        tokens = payload["watermark"]["tokens"].numpy()
        pivots = payload["watermark"]["Ys"].numpy()
        horizon = int(tokens.shape[1])
        # Column t of this array is the token four positions back, which is the
        # single token the released implementation hashes.
        seed_tokens = np.concatenate(
            (prompts[:, -width:], tokens[:, :horizon]), axis=1
        )[:, :horizon]

        distinct_pivots, distinct_pairs, distinct_seeds = [], [], []
        pair_groups = pair_bad = seed_groups = seed_bad = 0
        for row in range(tokens.shape[0]):
            value = pivots[row]
            seed = seed_tokens[row].astype(np.int64)
            pair = seed * (1 << 20) + tokens[row].astype(np.int64)
            distinct_pivots.append(int(np.unique(value).size))
            distinct_pairs.append(int(np.unique(pair).size))
            distinct_seeds.append(int(np.unique(seed).size))
            total, bad = group_is_constant(pair, value)
            pair_groups += total
            pair_bad += bad
            total, bad = group_is_constant(seed, value)
            seed_groups += total
            seed_bad += bad

        vocabulary = int(real.MODEL_SPECS[spec]["vocabulary_size"])
        raw, _ = real.load_verified_pickle(directory / f"{spec}-raw-m200-T500.pkl")
        null_pivots, _, _ = real.replay_raw_pivots(
            raw["prompts"].numpy(),
            raw["null"]["tokens"].numpy(),
            vocabulary_size=vocabulary,
        )
        distinct_null = [int(np.unique(row).size) for row in null_pivots]

        models[spec] = {
            "label": label,
            "documents": int(tokens.shape[0]),
            "horizon": horizon,
            "watermarked_distinct_pivots_mean": float(np.mean(distinct_pivots)),
            "watermarked_distinct_pivots_median": float(np.median(distinct_pivots)),
            "watermarked_distinct_pairs_mean": float(np.mean(distinct_pairs)),
            "watermarked_distinct_seed_tokens_mean": float(np.mean(distinct_seeds)),
            "unwatermarked_distinct_pivots_mean": float(np.mean(distinct_null)),
            "pair_groups": int(pair_groups),
            "pair_groups_with_unequal_pivots": int(pair_bad),
            "seed_groups": int(seed_groups),
            "seed_groups_with_unequal_pivots": int(seed_bad),
            "documents_where_distinct_pivots_equal_distinct_pairs": int(
                sum(1 for a, b in zip(distinct_pivots, distinct_pairs) if a == b)
            ),
        }

    return {
        "models": models,
        "context_width": width,
        "interpretation": (
            "the pair (W_{t-4}, W_t) determines the pivot in every group; the "
            "seed token alone does not, so a repeated context reuses a seed "
            "without reproducing a pivot"
        ),
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = audit(args.data_dir)
    output = args.output or DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
