#!/usr/bin/env python3
"""Why does assuming rho=0 cost nothing for Gumbel and everything for inverse?

The contamination table (tab:contamination) shows the contamination-aware
and the clean-rho
shared Bayes rules with the same Type II error at every tested rho_star under
the Gumbel pivot, while the corresponding inverse pair separates by a factor of
four.  Read on its own that looks like the contamination was never applied to
the Gumbel arm.  It was: the Gumbel errors rise monotonically in rho_star and
the rho_star=1 diagnostic returns 1-alpha.  This script measures the two
mechanisms that make the Gumbel rules insensitive to the rho model.

1. Finite expected null cost. The Gumbel component is positive on (0, 1)
   and has integrable log-density endpoints. Its log loss is unbounded as
   the pivot tends to zero, but its expected loss under the null is finite.
   With D1 the component-versus-null Kullback-Leibler divergence and D0 the
   reverse divergence, a fixed-Delta clean rule still drifts upward whenever
   rho_star < D1 / (D1 + D0).  We evaluate that break-even rate by quadrature.

2. No support truncation.  The limiting inverse alternative lives on
   (0, 1 - Delta) while its null lives on (0, 1), so one null-like pivot d
   annihilates every component with Delta > 1 - d.  Contamination is then a
   hard posterior constraint rather than a finite expected null cost. We measure the
   realized ceiling and the surviving prior mass.

The script also locates the Gumbel drift knee directly, replays the stored
rejection indicators to test clean against robust as a paired comparison, and
checks the contaminated errors against the independent clean benchmark at the
surviving horizon (1 - rho_star) n.

Writes results/bayesian_paper_benchmark/contamination_asymmetry.json.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from bayesian_watermark import (
    SequentialBayesDetector,
    gumbel_alt_logpdf_from_probs,
    simulate_inverse_alternative,
    simulate_inverse_null,
    spike_probabilities,
)
from paired_comparisons import exact_mcnemar_p_value

# Distinct from the tail sweep and the other simulation diagnostics.
SEED = 240_401_251
VOCABULARY = 1000
DELTA_LOW, DELTA_HIGH = 0.001, 0.5
NODES = 96
HORIZON = 700
QUADRATURE_NODES = 4000
BREAKEVEN_DELTAS = (0.005, 0.05, 0.1, 0.2, 0.35, 0.5)
TRUNCATION_RHOS = (0.0, 0.1, 0.25, 0.4, 0.6)
DRIFT_RHOS = (0.0, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0)
N_DOCUMENTS = 300
CROSS_CHECK_DOCUMENTS = 3
EVIDENCE_THRESHOLD = 20.0
DOCUMENT_CHUNK = 25

RESULTS = Path(__file__).resolve().parents[1] / "results" / "bayesian_paper_benchmark"


def delta_grid(nodes: int = NODES) -> tuple[np.ndarray, np.ndarray]:
    """The benchmark's Gauss-Legendre grid on (DELTA_LOW, DELTA_HIGH)."""

    points, weights = np.polynomial.legendre.leggauss(nodes)
    grid = DELTA_LOW + 0.5 * (points + 1.0) * (DELTA_HIGH - DELTA_LOW)
    return grid, weights / weights.sum()


def spike_log_density(log_pivots: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """Log of ``f_Delta(r) = r**(Delta/(1-Delta)) + (V-1) r**((V-1)/Delta - 1)``.

    ``log_pivots`` and ``deltas`` broadcast against each other.
    """

    top = deltas / (1.0 - deltas)
    tail = (VOCABULARY - 1.0) / deltas - 1.0
    return np.logaddexp(
        top * log_pivots,
        math.log(VOCABULARY - 1.0) + tail * log_pivots,
    )


def simulate_gumbel_pivots(
    delta: float, n_documents: int, horizon: int, rng: np.random.Generator
) -> np.ndarray:
    """Exact selected-token pivots for the equal-tail spike NTP at ``delta``."""

    uniforms = rng.random((n_documents, horizon))
    top_selected = rng.random((n_documents, horizon)) < (1.0 - delta)
    exponent = np.where(top_selected, 1.0 - delta, delta / (VOCABULARY - 1))
    return uniforms**exponent


# ---------------------------------------------------------------- mechanism 1


def gumbel_breakeven_rates() -> list[dict]:
    """Per-token divergences and the contamination rate that stalls the drift."""

    points, weights = np.polynomial.legendre.leggauss(QUADRATURE_NODES)
    pivots = 0.5 * (points + 1.0)
    quadrature = 0.5 * weights

    rows = []
    for delta in BREAKEVEN_DELTAS:
        log_density = np.asarray(
            gumbel_alt_logpdf_from_probs(
                pivots, spike_probabilities(delta, VOCABULARY)
            )
        )
        density = np.exp(log_density)
        mass = float(quadrature @ density)
        if abs(mass - 1.0) > 1e-6:
            raise AssertionError(f"component at Delta={delta} integrates to {mass!r}")
        forward = float(quadrature @ (density * log_density))  # KL(f || Unif)
        reverse = float(quadrature @ (-log_density))  # KL(Unif || f)
        rows.append(
            {
                "delta": delta,
                "kl_component_vs_null": forward,
                "kl_null_vs_component": reverse,
                "breakeven_rho": forward / (forward + reverse),
            }
        )
    return rows


def gumbel_drift_sweep(rng: np.random.Generator) -> list[dict]:
    """Terminal log Bayes factor of the clean-rho shared mixture versus rho_star."""

    grid, weights = delta_grid()
    log_prior = np.log(weights)
    rows = []
    for rho in DRIFT_RHOS:
        totals = np.empty(N_DOCUMENTS)
        keep = []
        for start in range(0, N_DOCUMENTS, DOCUMENT_CHUNK):
            stop = min(start + DOCUMENT_CHUNK, N_DOCUMENTS)
            size = stop - start
            truth = rng.uniform(DELTA_LOW, DELTA_HIGH, size=size)
            clean = np.stack(
                [simulate_gumbel_pivots(d, 1, HORIZON, rng)[0] for d in truth]
            )
            replaced = rng.random((size, HORIZON))
            mask = rng.random((size, HORIZON)) < rho
            pivots = np.where(mask, replaced, clean)
            log_pivots = np.log(pivots)
            component = spike_log_density(log_pivots[:, None, :], grid[None, :, None])
            unnormalized = log_prior[None, :] + component.sum(axis=2)
            shift = unnormalized.max(axis=1, keepdims=True)
            totals[start:stop] = (
                shift[:, 0] + np.log(np.exp(unnormalized - shift).sum(axis=1))
            )
            if start == 0:
                keep = [pivots[i] for i in range(min(CROSS_CHECK_DOCUMENTS, size))]

        # Cross-check the vectorised evidence against the reference detector.
        for path in keep:
            detector = SequentialBayesDetector(
                scheme="gumbel",
                delta_grid=grid,
                delta_weights=weights,
                structure="shared",
                gumbel_family="spike",
                vocabulary_size=VOCABULARY,
            )
            detector.update_many(path)
            log_pivots = np.log(path)
            component = spike_log_density(log_pivots[None, :], grid[:, None])
            unnormalized = log_prior + component.sum(axis=1)
            shift = unnormalized.max()
            mine = shift + math.log(float(np.exp(unnormalized - shift).sum()))
            gap = abs(mine - detector.log_bayes_factor)
            if gap > 1e-8:
                raise AssertionError(f"vectorised evidence differs by {gap:g}")

        rows.append(
            {
                "rho_star": rho,
                "mean_log_bayes_factor": float(totals.mean()),
                "median_log_bayes_factor": float(np.median(totals)),
                "fraction_above_threshold": float(
                    np.mean(totals > math.log(EVIDENCE_THRESHOLD))
                ),
            }
        )
    return rows


# ---------------------------------------------------------------- mechanism 2


def inverse_support_ceiling(rng: np.random.Generator) -> list[dict]:
    """How far contamination truncates the deficit posterior for inverse pivots."""

    grid, weights = delta_grid()
    rows = []
    for rho in TRUNCATION_RHOS:
        ceilings = np.empty(N_DOCUMENTS)
        for index in range(N_DOCUMENTS):
            truth = rng.uniform(DELTA_LOW, DELTA_HIGH)
            clean = simulate_inverse_alternative(HORIZON, truth, rng)
            replaced = simulate_inverse_null(HORIZON, rng, VOCABULARY)
            mask = rng.random(HORIZON) < rho
            ceilings[index] = 1.0 - np.where(mask, replaced, clean).max()
        surviving = np.array([float(weights[grid < c].sum()) for c in ceilings])
        rows.append(
            {
                "rho_star": rho,
                "median_delta_ceiling": float(np.median(ceilings)),
                "mean_delta_ceiling": float(ceilings.mean()),
                "median_surviving_prior_mass": float(np.median(surviving)),
                "mean_surviving_prior_mass": float(surviving.mean()),
            }
        )
    return rows


# ----------------------------------------------------------------- replay


def paired_clean_vs_robust() -> list[dict]:
    """Exact McNemar on the stored contamination rejection indicators."""

    archive = np.load(RESULTS / "contamination_rejection_indicators.npz")
    rows = []
    for scheme in ("gumbel", "inverse"):
        for rho in ("0", "0.1", "0.25", "0.4", "0.6"):
            clean_key = f"{scheme}|{rho}|{HORIZON}|bayes_shared_clean"
            robust_key = f"{scheme}|{rho}|{HORIZON}|bayes_shared_robust"
            if clean_key not in archive.files or robust_key not in archive.files:
                continue
            clean = archive[clean_key].astype(bool)
            robust = archive[robust_key].astype(bool)
            clean_only = int(np.sum(clean & ~robust))
            robust_only = int(np.sum(robust & ~clean))
            degenerate = clean_only + robust_only == 0
            rows.append(
                {
                    "scheme": scheme,
                    "rho_star": float(rho),
                    "clean_rejects": int(clean.sum()),
                    "robust_rejects": int(robust.sum()),
                    "clean_only": clean_only,
                    "robust_only": robust_only,
                    # With no discordant pair the conditional reference
                    # distribution is the point mass at zero.  The helper
                    # returns 1 so the cell keeps its place in a Holm family;
                    # the flag records that the test itself is uninformative.
                    "exact_mcnemar_p": exact_mcnemar_p_value(clean_only, robust_only),
                    "degenerate_reference_distribution": degenerate,
                }
            )
    return rows


def effective_horizon_match() -> list[dict]:
    """Contaminated error against the clean benchmark at the surviving horizon."""

    with (RESULTS / "benchmark_results.csv").open(newline="", encoding="utf8") as stream:
        clean_rows = [
            row
            for row in csv.DictReader(stream)
            if row["scenario"] == "shared_delta_equal_tail_sensitivity"
            and row["decision_rule"] == "fixed_horizon_mc_calibrated"
            and row["method"] == "bayes_shared"
        ]
    clean = {
        (row["scheme"], int(row["horizon"])): float(row["type2_error"])
        for row in clean_rows
    }

    with (RESULTS / "contamination_results.csv").open(
        newline="", encoding="utf8"
    ) as stream:
        contaminated_rows = [
            row
            for row in csv.DictReader(stream)
            if row["scenario"] == "shared_delta_iid_null_replacement"
            and int(row["horizon"]) == HORIZON
            and row["method"] == "bayes_shared_clean"
        ]

    rows = []
    for row in contaminated_rows:
        rho = float(row["rho_true"])
        if rho >= 1.0:
            continue
        surviving = int(round((1.0 - rho) * HORIZON))
        reference = clean.get((row["scheme"], surviving))
        if reference is None:
            continue
        # The contamination sweep spells the column type_ii_error; the clean
        # benchmark spells it type2_error.
        observed = float(row["type_ii_error"])
        rows.append(
            {
                "scheme": row["scheme"],
                "rho_star": rho,
                "contaminated_type2_error": observed,
                "surviving_horizon": surviving,
                "clean_type2_error_at_surviving_horizon": reference,
                "excess_over_surviving_horizon": observed - reference,
            }
        )
    return rows


def main() -> None:
    rng = np.random.default_rng(SEED)

    breakeven = gumbel_breakeven_rates()
    truncation = inverse_support_ceiling(rng)
    drift = gumbel_drift_sweep(rng)
    paired = paired_clean_vs_robust()
    horizons = effective_horizon_match()

    payload = {
        "question": (
            "Why is the Gumbel shared Bayes rule insensitive to the rho model "
            "at contamination rates that destroy its inverse counterpart?"
        ),
        "design": {
            "vocabulary_size": VOCABULARY,
            "horizon": HORIZON,
            "delta_grid": f"{NODES} Gauss-Legendre nodes on ({DELTA_LOW}, {DELTA_HIGH})",
            "quadrature_nodes": QUADRATURE_NODES,
            "n_documents_per_cell": N_DOCUMENTS,
            "evidence_threshold": EVIDENCE_THRESHOLD,
            "seed": SEED,
        },
        "gumbel_breakeven": breakeven,
        "gumbel_breakeven_note": (
            "A fixed-Delta clean rule accumulates positive drift while "
            "rho_star < D1/(D1+D0). The reverse divergence is finite because "
            "this density has integrable log-density endpoints; positivity "
            "alone would not establish integrability."
        ),
        "gumbel_drift_sweep": drift,
        "inverse_support_ceiling": truncation,
        "inverse_support_note": (
            "The limiting inverse alternative is supported on (0, 1-Delta), so "
            "every component with Delta > 1 - max(d) has zero likelihood.  The "
            "reverse divergence is infinite and no break-even rate exists."
        ),
        "paired_clean_vs_robust": paired,
        "effective_horizon_match": horizons,
        "effective_horizon_note": (
            "Compares the clean-rho rule under contamination with the same rule "
            "on (1-rho_star)n uncontaminated tokens, using the independent clean "
            "benchmark.  A small excess means the replaced tokens cost little "
            "beyond their own information."
        ),
    }

    output = RESULTS / "contamination_asymmetry.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf8")

    print("Gumbel break-even contamination rate (fixed Delta):")
    for row in breakeven:
        print(
            f"  Delta={row['delta']:.3f}  D1={row['kl_component_vs_null']:7.3f}"
            f"  D0={row['kl_null_vs_component']:7.3f}"
            f"  rho_breakeven={row['breakeven_rho']:.3f}"
        )
    print("\nGumbel clean-rho mixture, terminal log Bayes factor at n=700:")
    for row in drift:
        print(
            f"  rho*={row['rho_star']:.2f}  mean={row['mean_log_bayes_factor']:10.1f}"
            f"  median={row['median_log_bayes_factor']:10.1f}"
            f"  reject={row['fraction_above_threshold']:.3f}"
        )
    print("\nInverse deficit-support truncation at n=700:")
    for row in truncation:
        print(
            f"  rho*={row['rho_star']:.2f}  median ceiling={row['median_delta_ceiling']:.4f}"
            f"  median surviving prior mass={row['median_surviving_prior_mass']:.3f}"
        )
    print("\nClean versus robust, paired on the stored indicators:")
    for row in paired:
        shown = (
            "degenerate (no discordant pair)"
            if row["degenerate_reference_distribution"]
            else f"{row['exact_mcnemar_p']:.3g}"
        )
        print(
            f"  {row['scheme']:8s} rho*={row['rho_star']:.2f}"
            f"  clean-only={row['clean_only']:5d} robust-only={row['robust_only']:5d}"
            f"  exact McNemar p={shown}"
        )
    print("\nClean-rho rule versus the same rule on the surviving tokens:")
    for row in horizons:
        print(
            f"  {row['scheme']:8s} rho*={row['rho_star']:.2f}"
            f"  contaminated={row['contaminated_type2_error']:.4f}"
            f"  clean at n={row['surviving_horizon']:3d}: "
            f"{row['clean_type2_error_at_surviving_horizon']:.4f}"
            f"  excess={row['excess_over_surviving_horizon']:+.4f}"
        )
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
