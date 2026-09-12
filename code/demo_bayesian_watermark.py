"""Reproducible examples for the Bayesian watermark detector."""

from __future__ import annotations

import math

import numpy as np

from bayesian_watermark import (
    SequentialBayesDetector,
    beta_grid_prior,
    simulate_gumbel_alternative,
    simulate_inverse_alternative,
    spike_probabilities,
)


def summarize(name: str, detector: SequentialBayesDetector, alpha: float = 0.01) -> None:
    crossing = detector.first_crossing_time(alpha)
    crossing_text = "not crossed" if crossing is None else f"token {crossing}"
    print(name)
    print(f"  observations:          {detector.n_observations}")
    print(f"  log Bayes factor:      {detector.log_bayes_factor:9.3f}")
    print(f"  posterior P(H1|data):  {detector.posterior_probability():9.6f}")
    print(f"  anytime BF >= 1/{alpha:g}: {crossing_text}")
    print(
        "  loss-based decision:   "
        + ("watermarked" if detector.declare_watermark(10.0, 1.0) else "not watermarked")
        + "  (false-positive cost = 10 x false-negative cost)"
    )


def gumbel_example() -> None:
    rng = np.random.default_rng(240401245)
    vocabulary_size = 32
    true_delta = 0.28
    pivots = simulate_gumbel_alternative(
        180,
        spike_probabilities(true_delta, vocabulary_size),
        rng,
    )

    delta_grid, delta_weights = beta_grid_prior(
        2.0,
        3.0,
        low=0.02,
        high=0.65,
        size=81,
    )
    detector = SequentialBayesDetector(
        "gumbel",
        delta_grid,
        delta_weights,
        rho_grid=(0.0, 0.10, 0.25),
        rho_weights=(0.65, 0.25, 0.10),
        structure="shared",
        gumbel_family="spike",
        vocabulary_size=vocabulary_size,
        prior_watermark=0.10,
    )
    detector.update_many(pivots)
    summarize("Gumbel-max pivot (shared Delta and contamination rate)", detector)

    posterior = detector.component_posterior_weights.reshape(
        detector.delta_grid.size, detector.rho_grid.size
    )
    posterior_delta = posterior.sum(axis=1)
    posterior_mean_delta = float(np.sum(detector.delta_grid * posterior_delta))
    print(f"  posterior mean Delta:  {posterior_mean_delta:9.3f} (truth {true_delta:.2f})")


def inverse_example() -> None:
    rng = np.random.default_rng(240401246)
    true_delta = 0.36
    vocabulary_size = 1_000
    pivots = simulate_inverse_alternative(100, true_delta, rng)
    delta_grid, delta_weights = beta_grid_prior(
        2.5,
        3.5,
        low=0.01,
        high=0.70,
        size=101,
    )

    detector = SequentialBayesDetector(
        "inverse",
        delta_grid,
        delta_weights,
        rho_grid=(0.0, 0.15),
        rho_weights=(0.8, 0.2),
        structure="tokenwise",
        inverse_null_vocabulary_size=vocabulary_size,
        prior_watermark=0.10,
    )
    detector.update_many(pivots)
    summarize(
        "Inverse-transform pivot (tokenwise mixture; exact finite-V null)",
        detector,
    )


def main() -> None:
    print("Bayesian pivot-watermark reference implementation")
    print("=" * 50)
    gumbel_example()
    print()
    inverse_example()
    print()
    print(f"For alpha=0.01, the anytime evidence threshold is BF=100 (log BF={math.log(100):.3f}).")


if __name__ == "__main__":
    main()
