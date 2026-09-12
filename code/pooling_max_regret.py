"""Maximum regret for the two partial-pooling sweeps.

The regime sweeps of Tables~3--5 report maximum regret; the pooling sweeps did
not, although they are the same kind of object -- Type~II error over a set of
generating regimes -- and the pooled rules are exactly the ones whose case
rests on worst-case behaviour rather than on any single column.

This module adds that summary without re-simulating.  Both pooling sweeps
already ship a per-document boolean miss indicator for every
``regime|rule|horizon`` cell, and the mean of that indicator reproduces the
published Type~II error exactly (asserted below, not assumed), so the stored
indicators are a sufficient statistic for both the point estimate and the
paired document bootstrap.  Regret, the envelope, and the bootstrap standard
error all come from :mod:`regime_sweep`, so the pooling tables and the regime
tables cannot drift apart in convention: a maximum of differences is not a
binomial proportion, and the standard error has to be a paired resample.

The bootstrap is seeded exactly as ``run_regime_sweep`` seeds it -- one stream
per regime from ``[seed, MAX_REGRET_BOOTSTRAP_SEED_SALT, regime_index]`` -- and
resamples documents within a regime, applying one draw to every rule and
horizon at once.  Documents are not shared across regimes, which are separate
simulations, so there is nothing to pair there.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from regime_sweep import (
    DEFAULT_MAX_REGRET_BOOTSTRAP_REPLICATES,
    MAX_REGRET_BOOTSTRAP_SEED_SALT,
    bootstrap_max_regret_mc_se,
    is_benchmark_rule,
    max_regret,
    regret_table,
)

RESULTS = Path(__file__).resolve().parents[1] / "results" / "bayesian_paper_benchmark"

# (payload, indicators, regime labels) for the two sweeps.  The regime order is
# the table's column order, which is also the tie-break order max_regret uses.
SWEEPS = {
    "deficit": (
        "deficit_persistence_sweep.json",
        "deficit_persistence_indicators.npz",
        ("P1", "P2", "P3", "P4", "P5", "P6"),
    ),
    "width": (
        "width_persistence_sweep.json",
        "width_persistence_indicators.npz",
        ("Q1", "Q2", "Q3", "Q4", "Q5"),
    ),
}


def indicator_cube(
    indicators, regime: str, rules: list[str], horizons: list[str]
) -> np.ndarray:
    """``(documents, rules, horizons)`` miss indicators for one regime."""

    return np.stack(
        [
            np.stack(
                [indicators[f"{regime}|{rule}|{horizon}"].astype(float)
                 for horizon in horizons],
                axis=1,
            )
            for rule in rules
        ],
        axis=1,
    )


def summarize(
    payload: dict,
    indicators,
    regimes: tuple[str, ...],
    *,
    replicates: int = DEFAULT_MAX_REGRET_BOOTSTRAP_REPLICATES,
) -> dict:
    """Maximum regret per (horizon, rule), with a paired-bootstrap MC SE.

    Raises if the stored indicators do not reproduce the stored Type~II error,
    since every number here would otherwise describe a different sample than
    the table it is printed beside.
    """

    type2 = payload["type2_error"]
    horizons = list(type2)
    rules = list(type2[horizons[0]])
    seed = int(payload["seed"])

    for horizon in horizons:
        for rule in rules:
            for regime in regimes:
                stored = float(type2[horizon][rule][regime])
                mean = float(indicators[f"{regime}|{rule}|{horizon}"].mean())
                if stored != mean:
                    raise AssertionError(
                        f"{regime}|{rule}|{horizon}: stored Type II {stored} "
                        f"is not the indicator mean {mean}"
                    )

    point = {}
    for horizon in horizons:
        regret, _ = regret_table(
            {rule: {regime: float(type2[horizon][rule][regime])
                    for regime in regimes}
             for rule in rules}
        )
        point[horizon] = max_regret(regret)

    draws = np.empty((replicates, len(regimes), len(rules), len(horizons)))
    for index, regime in enumerate(regimes):
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [seed, MAX_REGRET_BOOTSTRAP_SEED_SALT, index]
            )
        )
        cube = indicator_cube(indicators, regime, rules, horizons)
        documents = cube.shape[0]
        for replicate in range(replicates):
            draws[replicate, index] = cube[
                rng.integers(0, documents, documents)
            ].mean(axis=0)

    mask = np.array([is_benchmark_rule(rule) for rule in rules])
    per_horizon, pooled = bootstrap_max_regret_mc_se(draws, mask)

    return {
        "note": (
            "Largest Type II excess over the best benchmark rule, across the "
            "generating regimes, matching the convention of the regime sweeps. "
            "Standard errors are a paired document bootstrap that recomputes "
            "the envelope on each draw; a maximum of differences is not a "
            "binomial proportion."
        ),
        "bootstrap_replicates": replicates,
        "bootstrap_seed_salt": MAX_REGRET_BOOTSTRAP_SEED_SALT,
        "envelope_rules": [r for r, keep in zip(rules, mask) if keep],
        "by_horizon": {
            horizon: {
                rule: {
                    "max_regret": float(point[horizon][rule]["max_regret"]),
                    "mc_se": float(per_horizon[i, j]),
                    "argmax_regime": point[horizon][rule]["argmax_regime"],
                }
                for i, rule in enumerate(rules)
            }
            for j, horizon in enumerate(horizons)
        },
        "pooled_over_horizons": {
            rule: float(pooled[i]) for i, rule in enumerate(rules)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="update the sweep payloads in place")
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()

    for name, (payload_file, indicator_file, regimes) in SWEEPS.items():
        path = args.results / payload_file
        payload = json.loads(path.read_text(encoding="utf-8"))
        with np.load(args.results / indicator_file) as indicators:
            summary = summarize(payload, indicators, regimes)
        print(f"=== {name} ===")
        for horizon, rows in summary["by_horizon"].items():
            print(f"  n={horizon}")
            for rule, cell in rows.items():
                print(f"    {rule:22s} {cell['max_regret']:.4f} "
                      f"({cell['mc_se']:.4f})  {cell['argmax_regime']}")
        if args.write:
            payload["max_regret"] = summary
            # indent=2 with the original key order, so the diff is additive.
            path.write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
            print(f"  wrote max_regret into {payload_file}")


if __name__ == "__main__":
    main()
