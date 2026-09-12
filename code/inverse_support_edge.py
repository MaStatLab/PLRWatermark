#!/usr/bin/env python3
"""Why tail sparsity strains the inverse-pivot working alternative.

The limiting inverse alternative \\eqref{eq:inverse-alt} is supported on
``(0, 1-Delta)`` while the exact null is supported on ``(0, 1)``.  Its derivation
assumes ``(log M) p_(2) -> 0``.  A tail with ``J`` live coordinates has
``p_(2) = Delta / J``, so that quantity is ``Delta (log M) / J`` and does not
vanish: at ``M = 50272``, ``Delta = .1`` and ``J = 1`` it is ``1.08``.

This script measures the consequence.  It reports three things.

``edge_mass``
    The fraction of watermarked pivots landing at or beyond ``1-Delta``, where
    the working numerator is zero.  Even a small amount of mass outside the
    working support can have a large effect on accumulated log likelihood
    ratios.  This statistic alone does not assess agreement inside the support.

``efficiency``
    Per-token standardised drift of the triangular log likelihood ratio at the
    true deficit, replacing nonfinite values by -40 in both samples.

``rho_rescue``
    The decisive comparison.  Because ``rho + (1-rho) L >= rho``, the
    contamination layer bounds the likelihood ratio away from zero and caps
    exactly this damage.  Generating with a sparse tail and ``rho_true = 0`` --
    no contamination anywhere -- isolates whether the layer's inverse-pivot
    benefit is support-edge robustness in general or protection against
    null-like replacement specifically.

The alternative pivots use a continuous-rank approximation to inverse-transform
decoding, not the limiting triangular pivot law and not the exact finite-M
decoder.  Independent Uniform(0,1) locations replace the distinct permutation
ranks of the live coordinates; sorting preserves their relative order.  This
replacement has no asserted deterministic 1/(M-1) error bound.  Vocabulary size
enters the null simulation, null density, and asymptotic-validity diagnostic,
but not the alternative rank generator.  Historical numerical arrays retain
this approximation; finite-vocabulary support-edge accuracy is not established
by this diagnostic.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import dirichlet_detector as dirichlet
import paired_comparisons as paired
from bayesian_watermark import inverse_alt_logpdf, inverse_null_logpdf

SEED = 240_401_260
DEFAULT_VOCABULARY = 50_272
DEFAULT_HORIZON = 200
DEFAULT_DOCUMENTS = 1_500
DEFAULT_CALIBRATION = 3_000
DEFAULT_NODES = 96
DELTA_LOW, DELTA_HIGH = 0.001, 0.5

# The frozen contamination prior of the robustness study, reused verbatim.
RHO_GRID: tuple[float, ...] = (0.0, 0.1, 0.25, 0.4, 0.6)
RHO_WEIGHTS: tuple[float, ...] = (0.5, 0.125, 0.125, 0.125, 0.125)

EDGE_DELTAS = (0.05, 0.1, 0.2, 0.3, 0.5)
TAIL_WIDTHS = (1, 4, 16, 256)

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "bayesian_paper_benchmark"
    / "inverse_support_edge.json"
)
DEFAULT_QUICK_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "bayesian_paper_benchmark"
    / "quick"
    / "inverse_support_edge"
    / "inverse_support_edge.json"
)


def simulate_inverse(
    n_docs: int,
    n_tokens: int,
    vocabulary_size: int,
    deltas: np.ndarray,
    live_tail: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Continuous-rank approximate pivots for ``live_tail`` live tail atoms.

    ``vocabulary_size`` is retained for API compatibility; this historical
    alternative generator does not use it.  The exact finite-M null is sampled
    separately by :func:`simulate_null`.
    """

    live = int(live_tail) + 1
    out = np.empty((n_docs, n_tokens), dtype=float)
    for row in range(n_docs):
        delta = float(deltas[row])
        eta = np.sort(rng.random((n_tokens, live)), axis=1)
        base = np.concatenate([[1.0 - delta], np.full(live_tail, delta / live_tail)])
        probs = np.tile(base, (n_tokens, 1))
        shuffle = np.argsort(rng.random((n_tokens, live)), axis=1)
        probs = np.take_along_axis(probs, shuffle, axis=1)
        uniform = rng.random(n_tokens)
        selected = (uniform[:, None] < np.cumsum(probs, axis=1)).argmax(axis=1)
        out[row] = np.abs(uniform - eta[np.arange(n_tokens), selected])
    return out


def simulate_null(
    n_docs: int, n_tokens: int, vocabulary_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Exact finite-vocabulary inverse null: ``|U - eta(I)|`` with ``I`` uniform."""

    uniform = rng.random((n_docs, n_tokens))
    ranks = rng.integers(0, vocabulary_size, size=(n_docs, n_tokens))
    return np.abs(uniform - ranks / (vocabulary_size - 1))


def shared_log_bayes_factor(
    pivots: np.ndarray,
    deltas: np.ndarray,
    delta_weights: np.ndarray,
    vocabulary_size: int,
    *,
    rho_grid: Sequence[float],
    rho_weights: Sequence[float],
    batch: int = 100,
) -> np.ndarray:
    """Shared-latent log Bayes factor of the inverse mixture over (Delta, rho)."""

    log_delta = np.log(delta_weights)
    weights = np.asarray(rho_weights, dtype=float)
    log_rho_weight = np.log(weights / weights.sum())
    out = np.empty(pivots.shape[0], dtype=float)
    for start in range(0, pivots.shape[0], batch):
        stop = min(start + batch, pivots.shape[0])
        chunk = pivots[start:stop]
        log_null = inverse_null_logpdf(chunk, vocabulary_size)
        blocks = []
        for rho, log_weight in zip(rho_grid, log_rho_weight):
            totals = np.empty((stop - start, deltas.size), dtype=float)
            for index, delta in enumerate(deltas):
                ratio = inverse_alt_logpdf(chunk, float(delta)) - log_null
                if rho <= 0.0:
                    increment = ratio
                else:
                    increment = np.logaddexp(
                        math.log(rho), math.log1p(-rho) + ratio
                    )
                totals[:, index] = increment.sum(axis=1)
            blocks.append(totals + log_delta[None, :] + log_weight)
        stacked = np.concatenate(blocks, axis=1)
        peak = stacked.max(axis=1)
        # Every component can be annihilated at once: a pivot beyond 1-Delta for
        # the largest node has zero density under all of them.  That is the event
        # this script exists to measure, so it must return -inf rather than the
        # nan that -inf - -inf would produce.
        finite = np.isfinite(peak)
        block_out = np.full(peak.shape, -np.inf, dtype=float)
        if np.any(finite):
            shifted = stacked[finite] - peak[finite][:, None]
            block_out[finite] = peak[finite] + np.log(np.exp(shifted).sum(axis=1))
        out[start:stop] = block_out
    return out


def exact_mcnemar_p_value(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value; 1.0 at perfect concordance.

    Delegates to :mod:`paired_comparisons` so this diagnostic and the paired
    summaries of the clean study cannot drift apart, and so the tail is computed
    in log space: at the document counts used here the discordance can exceed a
    thousand, where ``2.0 ** (b + c)`` overflows a float.
    """

    return paired.exact_mcnemar_p_value(int(b), int(c))


def standardised_drift(alternative: np.ndarray, null: np.ndarray) -> float:
    return float((alternative.mean() - null.mean()) / null.std())


def triangular_log_ratio(
    pivots: np.ndarray, delta: float, vocabulary_size: int, floor: float = -40.0
) -> np.ndarray:
    """Triangular log likelihood ratio, with the zero-density region floored.

    The floor is applied identically to the alternative and to the null so the
    standardised drift compares like with like.
    """

    ratio = inverse_alt_logpdf(pivots, delta) - inverse_null_logpdf(
        pivots, vocabulary_size
    )
    return np.where(np.isfinite(ratio), ratio, floor)


def run(
    *,
    vocabulary_size: int = DEFAULT_VOCABULARY,
    horizon: int = DEFAULT_HORIZON,
    n_documents: int = DEFAULT_DOCUMENTS,
    n_calibration: int = DEFAULT_CALIBRATION,
    nodes: int = DEFAULT_NODES,
    seed: int = SEED,
) -> dict[str, Any]:
    started = time.perf_counter()
    deltas, delta_weights = dirichlet.gauss_legendre_delta_grid(
        DELTA_LOW, DELTA_HIGH, nodes
    )
    rng = np.random.default_rng(seed)
    log_m = math.log(vocabulary_size)

    # -- 1. mass beyond the working support ---------------------------------
    edge_rows: list[dict[str, Any]] = []
    edge_documents = max(200, n_documents // 5)
    for delta in EDGE_DELTAS:
        fixed = np.full(edge_documents, delta)
        for width in TAIL_WIDTHS:
            pivots = simulate_inverse(
                edge_documents, horizon, vocabulary_size, fixed, width, rng
            )
            edge_rows.append(
                {
                    "delta": float(delta),
                    "live_tail": int(width),
                    "second_largest_ntp": float(delta / width),
                    "log_m_times_second_largest": float(log_m * delta / width),
                    "fraction_at_or_beyond_one_minus_delta": float(
                        np.mean(pivots >= 1.0 - delta)
                    ),
                    "mean_pivot": float(pivots.mean()),
                    "triangular_limit_mean_pivot": float((1.0 - delta) / 3.0),
                }
            )

    # -- 2. per-token drift of the triangular score --------------------------
    null_tokens = simulate_null(1, 400_000, vocabulary_size, rng)[0]
    efficiency_rows: list[dict[str, Any]] = []
    for delta in (0.1, 0.3):
        fixed = np.full(edge_documents, delta)
        for width in TAIL_WIDTHS:
            pivots = simulate_inverse(
                edge_documents, horizon, vocabulary_size, fixed, width, rng
            ).reshape(-1)
            efficiency_rows.append(
                {
                    "delta": float(delta),
                    "live_tail": int(width),
                    "standardised_drift": standardised_drift(
                        triangular_log_ratio(pivots, delta, vocabulary_size),
                        triangular_log_ratio(null_tokens, delta, vocabulary_size),
                    ),
                }
            )

    # -- 3. does the rho layer rescue it, with no contamination present? -----
    calibration = simulate_null(n_calibration, horizon, vocabulary_size, rng)
    cutoffs: dict[str, float] = {}
    for label, (grid, weights) in (
        ("clean", ((0.0,), (1.0,))),
        ("robust", (RHO_GRID, RHO_WEIGHTS)),
    ):
        paths = shared_log_bayes_factor(
            calibration,
            deltas,
            delta_weights,
            vocabulary_size,
            rho_grid=grid,
            rho_weights=weights,
        )
        cutoffs[label] = float(np.quantile(paths, 0.95))

    rescue_rows: list[dict[str, Any]] = []
    for width in TAIL_WIDTHS:
        document_deltas = rng.uniform(DELTA_LOW, DELTA_HIGH, size=n_documents)
        pivots = simulate_inverse(
            n_documents, horizon, vocabulary_size, document_deltas, width, rng
        )
        rejected: dict[str, np.ndarray] = {}
        for label, (grid, weights) in (
            ("clean", ((0.0,), (1.0,))),
            ("robust", (RHO_GRID, RHO_WEIGHTS)),
        ):
            paths = shared_log_bayes_factor(
                pivots,
                deltas,
                delta_weights,
                vocabulary_size,
                rho_grid=grid,
                rho_weights=weights,
            )
            rejected[label] = paths > cutoffs[label]
        robust_only = int(np.sum(rejected["robust"] & ~rejected["clean"]))
        clean_only = int(np.sum(rejected["clean"] & ~rejected["robust"]))
        rescue_rows.append(
            {
                "live_tail": int(width),
                "edge_mass": float(
                    np.mean(pivots >= (1.0 - document_deltas)[:, None])
                ),
                "type2_clean_rho_zero": float(1.0 - rejected["clean"].mean()),
                "type2_robust_rho_prior": float(1.0 - rejected["robust"].mean()),
                "robust_minus_clean": float(
                    rejected["clean"].mean() - rejected["robust"].mean()
                ),
                "discordant_robust_only": robust_only,
                "discordant_clean_only": clean_only,
                "exact_mcnemar_p_value": exact_mcnemar_p_value(robust_only, clean_only),
            }
        )

    return {
        "question": (
            "Does tail sparsity push inverse pivots outside the working support, "
            "and is the rho layer's benefit there support-edge robustness rather "
            "than protection against null-like replacement?"
        ),
        "design": {
            "scheme": "inverse",
            "structure": "shared",
            "vocabulary_size": int(vocabulary_size),
            "alternative_rank_model": "independent_continuous_uniform_locations",
            "exact_finite_vocabulary_alternative": False,
            "rank_approximation_error_bound": None,
            "efficiency_nonfinite_log_ratio_floor": -40.0,
            "horizon": int(horizon),
            "n_documents": int(n_documents),
            "n_calibration": int(n_calibration),
            "delta_grid": f"{nodes} Gauss-Legendre nodes on ({DELTA_LOW}, {DELTA_HIGH})",
            "rho_true": 0.0,
            "rho_prior_grid": list(RHO_GRID),
            "rho_prior_weights": list(RHO_WEIGHTS),
            "seed": int(seed),
            "note": (
                "rho_true is zero everywhere: no contamination is simulated, so any "
                "benefit from the rho prior is attributable to the support edge."
            ),
        },
        "edge_mass": edge_rows,
        "efficiency": efficiency_rows,
        "rho_rescue": rescue_rows,
        "cutoffs": cutoffs,
        "runtime_seconds": time.perf_counter() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        payload = run(horizon=50, n_documents=200, n_calibration=300, nodes=16)
        output = args.output or DEFAULT_QUICK_OUTPUT
    else:
        payload = run()
        output = args.output or DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "runtime_seconds": payload["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
