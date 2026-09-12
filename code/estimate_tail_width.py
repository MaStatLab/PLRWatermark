#!/usr/bin/env python3
"""Estimate the effective live-tail width of the released open-model outputs.

The full-width equal-tail Bayesian Gumbel rules assume the residual mass is
spread evenly over all ``K = V - 1`` non-leading coordinates.  That assumption
is testable on the released data, because the recorded top probabilities pin
the deficit exactly.
Conditional on ``Delta_t``, the exact Gumbel pivot density of the NTP vector
``(1-Delta, Delta/J, ..., Delta/J)`` is

    f_{Delta,J}(r) = r**(Delta/(1-Delta)) + J r**(J/Delta - 1),

so ``J`` is the only free parameter and can be profiled out.  The Gumbel null
density is one, so the log-likelihood is also the log Bayes factor.

Three guards make the estimate interpretable.

*Recovery.*  The same estimator is run on simulated pivots with a known ``J`` at
the released sample size, to show the profile is identified rather than flat.

*Dependence.*  The skipgram pseudorandom function reuses a seed whenever the
token four positions back repeats, so positions are not independent; roughly a
quarter of the released values are distinct.  Three reductions are reported.
``all_positions`` uses every informative position and ignores the dependence.
``distinct_values`` keeps one position per rounded pivot value -- which is WRONG
as an estimator, because the likelihood is ``f_{Delta,J}(r)`` and dropping the
other members of a repeated-pivot group discards their deficits: in every one of
the 631 and 776 repeated-pivot groups the deficits differ.  It is retained only
because an earlier version of this study reported it, and because the gap
between it and the others is the point.  ``distinct_addresses`` is the reduction
to use: it groups by the actual PRF address, the token four back, and keeps one
position per address together with that position's own deficit.  A document
cluster bootstrap reports the selection frequency of each profile argmax.

*Goodness of fit.*  A probability-integral transform against
``F(r) = (1-Delta) r**(1/(1-Delta)) + Delta r**(J/Delta)`` with a
Kolmogorov--Smirnov statistic, so a poor fit at every ``J`` cannot masquerade as
an estimate.

``J`` is an effective width.  A real tail is not exactly equal, so a fitted
``J`` of one or two means the residual mass behaves as though carried by one or
two coordinates, not that exactly that many are nonzero.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SEED = 240_401_261
# Only positions whose recorded deficit is resolved well above float32 rounding
# carry information about the tail; below this the pivot is uninformative.
DEFAULT_DELTA_FLOOR = 0.05
DELTA_CEILING = 0.5
DEFAULT_GRID: tuple[int, ...] = (
    1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256, 1024, 4096, 16_384,
)
RECOVERY_TRUTHS: tuple[int, ...] = (1, 2, 4, 16, 256)

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "results" / "bayesian_paper_benchmark" / "tail_width_estimate.json"
)
DEFAULT_QUICK_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "results" / "bayesian_paper_benchmark" / "quick" / "tail_width_estimate"
    / "tail_width_estimate.json"
)


def log_likelihood(pivots: np.ndarray, deltas: np.ndarray, live_tail: int) -> float:
    """``sum_t log f_{Delta_t, J}(r_t)``."""

    log_r = np.log(np.clip(pivots, 1e-300, 1.0))
    head = (deltas / (1.0 - deltas)) * log_r
    tail = math.log(live_tail) + (live_tail / deltas - 1.0) * log_r
    peak = np.maximum(head, tail)
    return float(np.sum(peak + np.log(np.exp(head - peak) + np.exp(tail - peak))))


def profile(
    pivots: np.ndarray, deltas: np.ndarray, grid: Sequence[int]
) -> list[tuple[int, float]]:
    return [(int(j), log_likelihood(pivots, deltas, int(j))) for j in grid]


def fitted_cdf(pivots: np.ndarray, deltas: np.ndarray, live_tail: int) -> np.ndarray:
    """``F(r) = (1-Delta) r**(1/(1-Delta)) + Delta r**(J/Delta)``, which is one at r=1."""

    return (1.0 - deltas) * pivots ** (1.0 / (1.0 - deltas)) + deltas * pivots ** (
        live_tail / deltas
    )


def kolmogorov_smirnov(values: np.ndarray) -> tuple[float, float]:
    """KS statistic against Uniform(0,1) and its asymptotic p-value."""

    values = np.asarray(values)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("KS values must be a nonempty finite vector in [0,1]")
    ordered = np.sort(np.clip(values, 1e-12, 1.0 - 1e-12))
    n = ordered.size
    upper = float(np.max(np.arange(1, n + 1) / n - ordered))
    lower = float(np.max(ordered - np.arange(0, n) / n))
    statistic = max(upper, lower)
    scaled = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * statistic
    # A fixed-length alternating series fails near zero, where its terms have
    # not decayed. SciPy switches to the stable small-argument representation.
    from scipy.special import kolmogorov
    return statistic, float(kolmogorov(scaled))


def simulate_gumbel(deltas: np.ndarray, live_tail: int, rng: np.random.Generator) -> np.ndarray:
    """Exact Gumbel pivots for ``(1-Delta, Delta/J, ..., Delta/J)``."""

    uniform = rng.random(deltas.size)
    leading = rng.random(deltas.size) < (1.0 - deltas)
    selected = np.where(leading, 1.0 - deltas, deltas / live_tail)
    return uniform**selected


def first_per_group(keys: np.ndarray) -> np.ndarray:
    """Index of the FIRST occurrence of each distinct key, in sequence order.

    Not a random occurrence.  A later use of an address sits in a context that
    the earlier use already influenced -- the same pseudorandom vector helped
    choose the intervening tokens -- so its deficit is not independent of the
    earlier draw.  Only the first use of an address is untouched by the reuse,
    which is what the profile likelihood needs.  The selection is made on the
    full sequence, before any deficit filter, or "first" would mean first
    among the informative positions rather than first ever.
    """

    order = np.argsort(keys, kind="stable")      # stable: ties keep sequence order
    sorted_keys = keys[order]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(sorted_keys)) + 1))
    return np.sort(order[starts])


def cluster_bootstrap_estimate(
    pivots: np.ndarray,
    deltas: np.ndarray,
    documents: np.ndarray,
    grid: Sequence[int],
    *,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Resample DOCUMENTS and record where the profile peaks.

    First use of each PRF address removes the dependence between two positions
    that share a pseudorandom vector, but not the dependence between positions
    in the same continuation: they share a prompt, a decoding path, and a
    deficit trajectory.  Resampling positions independently would price only
    the first of those and would therefore understate the spread.  A document
    is the independent unit here -- prompts are distinct C4 records -- so the
    resample draws documents with replacement and takes every retained position
    of each drawn document.

    Documents contribute unequal numbers of positions, so the resampled sample
    size varies between replicates.  That is the honest behaviour of a cluster
    bootstrap and is not corrected for.
    """

    order = np.argsort(documents, kind="stable")
    bounds = np.searchsorted(documents[order], np.unique(documents))
    groups = np.split(order, bounds[1:])
    n_documents = len(groups)
    counts: dict[int, int] = {}
    for _ in range(replicates):
        drawn = rng.integers(0, n_documents, size=n_documents)
        index = np.concatenate([groups[g] for g in drawn])
        curve = profile(pivots[index], deltas[index], grid)
        best = max(curve, key=lambda item: item[1])[0]
        counts[int(best)] = counts.get(int(best), 0) + 1
    total = float(replicates)
    return {
        "replicates": int(replicates),
        "n_addresses": int(pivots.size),
        "n_documents": int(n_documents),
        "resample": (
            "document-cluster bootstrap over the first-use positions: documents "
            "are drawn with replacement and every retained position of a drawn "
            "document is taken, so within-document dependence is priced"
        ),
        "argmax_share": {str(j): counts[j] / total for j in sorted(counts)},
        "argmax_mode": int(max(counts, key=lambda j: counts[j])),
    }


def informative_first_per_group(keys: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Retain informative positions only AFTER selecting each address's first use.

    An uninformative first use still consumes the address.  Keeping a later,
    informative reuse would condition on a context influenced by that draw.
    """

    first_use = first_per_group(keys)
    return first_use[keep[first_use]]


def _summarise(
    pivots: np.ndarray, deltas: np.ndarray, grid: Sequence[int]
) -> dict[str, Any]:
    curve = profile(pivots, deltas, grid)
    ordered = sorted(curve, key=lambda item: item[1], reverse=True)
    best = ordered[0][0]
    statistic, p_value = kolmogorov_smirnov(fitted_cdf(pivots, deltas, best))
    return {
        "n": int(pivots.size),
        "estimate": int(best),
        "log_likelihood_gap_to_runner_up": float(ordered[0][1] - ordered[1][1]),
        "runner_up": int(ordered[1][0]),
        "profile_relative_to_maximum": {
            str(j): float(value - ordered[0][1]) for j, value in curve
        },
        "goodness_of_fit": {
            "fitted": {"live_tail": best, "ks_statistic": statistic, "ks_p_value": p_value},
        },
    }


def run(
    *,
    data_dir: Path | None = None,
    delta_floor: float = DEFAULT_DELTA_FLOOR,
    grid: Sequence[int] = DEFAULT_GRID,
    seed: int = SEED,
    recovery_size: int = 8_000,
    recovery_replicates: int = 200,
    cluster_replicates: int = 2_000,
) -> dict[str, Any]:
    import real_data_experiment as real

    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    directory = data_dir or real.DEFAULT_DATA_DIR

    models: dict[str, Any] = {}
    for name in real.MODEL_SPECS:
        released, _ = real.load_released_model_data(name, directory)
        pivots = np.asarray(released.gumbel_watermarked, dtype=float).ravel()
        deltas = 1.0 - np.asarray(released.top_probabilities_gumbel, dtype=float).ravel()
        vocabulary = int(released.vocabulary_size)
        usable = grid_for_vocabulary = [j for j in grid if j <= vocabulary - 1]

        keep = (
            (deltas >= delta_floor)
            & (deltas <= DELTA_CEILING)
            & (pivots > 0.0)
            & (pivots < 1.0)
        )
        all_addresses = real.prf_addresses(
            released.prompts, released.gumbel_tokens
        ).ravel()
        # FIRST use of each address, chosen on the full sequence and only then
        # filtered for an informative deficit.  Filtering first would pick the
        # first INFORMATIVE use, which may still sit downstream of an earlier
        # use of the same pseudorandom vector.
        first_use = informative_first_per_group(all_addresses, keep)
        addresses = all_addresses[keep]
        r, d = pivots[keep], deltas[keep]
        _, distinct = np.unique(np.round(r, 12), return_index=True)
        # Map the first-use positions into the filtered arrays.
        position_of = np.full(all_addresses.size, -1, dtype=np.int64)
        position_of[np.flatnonzero(keep)] = np.arange(int(keep.sum()))
        by_address = position_of[first_use]
        assert (by_address >= 0).all()

        all_positions = _summarise(r, d, usable)
        deduplicated = _summarise(r[distinct], d[distinct], usable)
        clustered = _summarise(r[by_address], d[by_address], usable)
        subsets = {
            "all_positions": (all_positions, slice(None)),
            "distinct_values": (deduplicated, distinct),
            "distinct_addresses": (clustered, by_address),
        }
        for label, (block, index) in subsets.items():
            wide = kolmogorov_smirnov(
                fitted_cdf(r[index], d[index], vocabulary - 1)
            )
            block["goodness_of_fit"]["assumed_full_width"] = {
                "live_tail": vocabulary - 1,
                "ks_statistic": wide[0],
                "ks_p_value": wide[1],
            }

        # Document of each first-use position.  The released arrays are
        # (documents, horizon) and were flattened, so integer division by the
        # horizon recovers the continuation a position came from.
        horizon = int(np.asarray(released.gumbel_watermarked).shape[1])
        documents_of_first_use = first_use // horizon
        bootstrap = cluster_bootstrap_estimate(
            r[by_address], d[by_address], documents_of_first_use, usable,
            replicates=cluster_replicates, rng=np.random.default_rng(seed + 2),
        )
        # The headline width is the address-clustered one.  The pivot-deduplicated
        # estimate is not usable: see the module docstring.
        estimate = clustered["estimate"]
        median_delta = float(np.median(d))
        models[name] = {
            "vocabulary_size": vocabulary,
            "n_positions_total": int(pivots.size),
            "n_positions_informative": int(keep.sum()),
            "n_distinct_pivot_values": int(distinct.size),
            "fraction_pivot_values_distinct": float(distinct.size / max(keep.sum(), 1)),
            "n_prf_addresses": int(np.unique(addresses).size),
            "all_positions": all_positions,
            "distinct_values": deduplicated,
            "distinct_addresses": clustered,
            "address_cluster_bootstrap": bootstrap,
            "implied_second_largest_ntp_at_median_delta": median_delta / estimate,
            "inverse_limit_condition_log_v_times_p2": (
                math.log(vocabulary) * median_delta / estimate
            ),
            "median_informative_delta": median_delta,
        }

    # -- identifiability check, at the sample sizes actually available -------
    # This used to run at 8,000 simulated positions, which is seven to eight
    # times what either model supplies after first-use selection and the
    # deficit filter.  Recovery at a sample size the data do not have says
    # nothing about whether J is identified HERE, so the check now runs at each
    # model's own informative count, plus the former size for comparison.
    # Repeated, not a single draw.  One simulated dataset per truth can recover
    # all five and still say nothing about reliability; the reported quantity is
    # the share of replicates whose profile peaks at the truth.
    sizes = sorted({int(m["distinct_addresses"]["n"]) for m in models.values()})
    recovery: dict[str, Any] = {}
    for size in sizes + ([recovery_size] if recovery_size not in sizes else []):
        block = []
        for truth in RECOVERY_TRUTHS:
            hits = 0
            gaps = []
            for _ in range(recovery_replicates):
                deltas = rng.uniform(delta_floor, DELTA_CEILING, size=size)
                curve = profile(simulate_gumbel(deltas, truth, rng), deltas, grid)
                ordered = sorted(curve, key=lambda item: item[1], reverse=True)
                hits += int(ordered[0][0] == truth)
                gaps.append(float(ordered[0][1] - ordered[1][1]))
            block.append(
                {
                    "true_live_tail": int(truth),
                    "replicates": int(recovery_replicates),
                    "recovery_rate": hits / float(recovery_replicates),
                    "median_log_likelihood_gap_to_runner_up": float(
                        np.median(gaps)
                    ),
                }
            )
        rates = [b["recovery_rate"] for b in block]
        recovery[str(size)] = {
            "n": int(size),
            "replicates": int(recovery_replicates),
            "min_recovery_rate": float(min(rates)),
            "worst_truth": int(
                block[int(np.argmin(rates))]["true_live_tail"]
            ),
            "recovered_always": all(r == 1.0 for r in rates),
            "by_truth": block,
        }

    return {
        "question": (
            "How many tail coordinates does the released watermarked output "
            "behave as though it has, against the V-1 assumed by the full-width "
            "equal-tail Bayesian Gumbel rules?"
        ),
        "design": {
            "scheme": "gumbel",
            "delta_floor": float(delta_floor),
            "delta_ceiling": float(DELTA_CEILING),
            "live_tail_grid": [int(j) for j in grid],
            "seed": int(seed),
            "note": (
                "J is an effective equal-tail width; a real tail is not exactly "
                "equal, so a fitted J of one or two means the residual mass "
                "behaves as though carried by one or two coordinates."
            ),
        },
        "identifiability_check": {
            "note": (
                "keyed by simulated sample size; the model-sized entries are "
                "the ones that speak to this data, the 8000 entry is the "
                "former default and is kept only for comparison"
            ),
            "by_sample_size": recovery,
        },
        "models": models,
        "runtime_seconds": time.perf_counter() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run(recovery_size=1_000) if args.quick else run()
    output = args.output or (DEFAULT_QUICK_OUTPUT if args.quick else DEFAULT_OUTPUT)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "runtime_seconds": payload["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
