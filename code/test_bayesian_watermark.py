"""Tests for the NumPy-only Bayesian watermark reference implementation."""

from __future__ import annotations

import math
import unittest

import numpy as np

from bayesian_watermark import (
    SequentialBayesDetector,
    beta_grid_prior,
    gumbel_alt_pdf_from_probs,
    inverse_alt_pdf,
    inverse_null_pdf,
    least_favorable_probabilities,
    logsumexp,
    simulate_gumbel_alternative,
    simulate_gumbel_null,
    simulate_inverse_alternative,
    simulate_inverse_null,
    spike_probabilities,
)


def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Small compatibility helper for NumPy versions before np.trapezoid."""

    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2.0))


class DensityTests(unittest.TestCase):
    def test_gumbel_density_normalizes(self) -> None:
        grid = np.linspace(0.0, 1.0, 200_001)
        density = gumbel_alt_pdf_from_probs(grid, np.array([0.5, 0.3, 0.2]))
        self.assertAlmostEqual(trapezoid(density, grid), 1.0, places=7)

    def test_inverse_densities_normalize(self) -> None:
        null_grid = np.linspace(0.0, 1.0 - 1e-10, 200_001)
        alt_grid = np.linspace(0.0, 1.0 - 0.37, 200_001, endpoint=False)
        self.assertAlmostEqual(
            trapezoid(inverse_null_pdf(null_grid), null_grid), 1.0, places=7
        )
        self.assertAlmostEqual(
            trapezoid(inverse_alt_pdf(alt_grid, 0.37), alt_grid), 1.0, places=5
        )

        exact_grid = (np.arange(60_000, dtype=float) + 0.5) / 60_000
        self.assertAlmostEqual(
            float(np.mean(inverse_null_pdf(exact_grid, vocabulary_size=7))),
            1.0,
            places=12,
        )

    def test_finite_inverse_null_closed_form_matches_rank_count(self) -> None:
        rng = np.random.default_rng(20260819)
        for vocabulary_size in (2, 3, 7, 31):
            random_grid = rng.uniform(size=500)
            jumps = np.arange(vocabulary_size - 1, dtype=float) / (
                vocabulary_size - 1
            )
            grid = np.concatenate((random_grid, np.nextafter(jumps, 1.0)))
            grid = grid[grid < 1.0]
            ranks = np.arange(vocabulary_size, dtype=float) / (
                vocabulary_size - 1
            )
            expected = (
                np.sum(grid[:, None] < ranks[None, :], axis=1)
                + np.sum(grid[:, None] < (1.0 - ranks)[None, :], axis=1)
            ) / vocabulary_size
            np.testing.assert_allclose(
                inverse_null_pdf(grid, vocabulary_size), expected
            )
            # At a jump, use the manuscript's right-continuous floor version;
            # direct floating-point rank subtraction need not represent the two
            # mathematically symmetric branches identically there.
            expected_at_jumps = 2.0 * (
                vocabulary_size - 1 - np.arange(vocabulary_size - 1)
            ) / vocabulary_size
            np.testing.assert_allclose(
                inverse_null_pdf(jumps, vocabulary_size), expected_at_jumps
            )

    def test_probability_constructions(self) -> None:
        for delta in (0.0, 0.2, 0.5, 0.72):
            probabilities = least_favorable_probabilities(delta, 16)
            self.assertAlmostEqual(float(probabilities.sum()), 1.0)
            self.assertLessEqual(float(probabilities.max()), 1.0 - delta + 1e-12)
        spike = spike_probabilities(0.35, 20)
        self.assertAlmostEqual(float(spike.sum()), 1.0)
        self.assertAlmostEqual(float(spike[0]), 0.65)

    def test_beta_grid_prior(self) -> None:
        grid, weights = beta_grid_prior(2.0, 5.0, low=0.01, high=0.6, size=51)
        self.assertTrue(np.all((grid > 0.01) & (grid < 0.6)))
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertTrue(np.all(weights > 0.0))

        default_grid, default_weights = beta_grid_prior(2.0, 3.0)
        self.assertTrue(np.all((default_grid > 0.0) & (default_grid < 1.0)))
        self.assertAlmostEqual(float(default_weights.sum()), 1.0)

        exact_sample = simulate_inverse_null(
            20, np.random.default_rng(77), vocabulary_size=7
        )
        self.assertTrue(np.all((exact_sample >= 0.0) & (exact_sample < 1.0)))


class SequentialTests(unittest.TestCase):
    def setUp(self) -> None:
        self.deltas = np.array([0.12, 0.28, 0.44])
        self.delta_weights = np.array([0.2, 0.5, 0.3])
        self.rhos = np.array([0.0, 0.2])
        self.rho_weights = np.array([0.75, 0.25])

    def _detector(self, structure: str) -> SequentialBayesDetector:
        return SequentialBayesDetector(
            "gumbel",
            self.deltas,
            self.delta_weights,
            rho_grid=self.rhos,
            rho_weights=self.rho_weights,
            structure=structure,
            gumbel_family="spike",
            vocabulary_size=24,
            prior_watermark=0.2,
        )

    def test_shared_sequential_equals_batch_mixture(self) -> None:
        rng = np.random.default_rng(101)
        observations = simulate_gumbel_alternative(
            45, spike_probabilities(0.28, 24), rng
        )
        detector = self._detector("shared")
        log_ratios = np.vstack(
            [detector.component_log_likelihood_ratios(float(value)) for value in observations]
        )
        expected = float(
            logsumexp(np.log(detector.component_prior_weights) + log_ratios.sum(axis=0))
        )
        actual = detector.update_many(observations)
        self.assertAlmostEqual(actual, expected, places=11)
        self.assertAlmostEqual(float(detector.component_posterior_weights.sum()), 1.0, places=12)

    def test_tokenwise_sequential_equals_product_of_mixtures(self) -> None:
        rng = np.random.default_rng(102)
        observations = simulate_gumbel_alternative(
            45, spike_probabilities(0.35, 24), rng
        )
        detector = self._detector("tokenwise")
        log_prior = np.log(detector.component_prior_weights)
        expected = sum(
            float(logsumexp(log_prior + detector.component_log_likelihood_ratios(float(value))))
            for value in observations
        )
        actual = detector.update_many(observations)
        self.assertAlmostEqual(actual, expected, places=11)

    def test_null_and_alternative_move_posterior_in_opposite_directions(self) -> None:
        null_detector = self._detector("shared")
        alt_detector = self._detector("shared")
        null_detector.update_many(simulate_gumbel_null(600, np.random.default_rng(222)))
        alt_detector.update_many(
            simulate_gumbel_alternative(
                180,
                spike_probabilities(0.35, 24),
                np.random.default_rng(333),
            )
        )
        self.assertLess(null_detector.posterior_probability(), 0.02)
        self.assertGreater(alt_detector.posterior_probability(), 0.98)
        self.assertIsNotNone(alt_detector.first_crossing_time(0.05))

    def test_inverse_alternative_accumulates_evidence(self) -> None:
        grid, weights = beta_grid_prior(2.0, 3.0, low=0.02, high=0.65, size=61)
        detector = SequentialBayesDetector(
            "inverse",
            grid,
            weights,
            rho_grid=(0.0, 0.1),
            rho_weights=(0.8, 0.2),
            structure="tokenwise",
            prior_watermark=0.1,
        )
        detector.update_many(
            simulate_inverse_alternative(160, 0.38, np.random.default_rng(444))
        )
        self.assertGreater(detector.posterior_probability(), 0.99)

    def test_shared_inverse_support_exhaustion_stays_negative_infinity(self) -> None:
        detector = SequentialBayesDetector(
            "inverse",
            delta_grid=(0.2, 0.4),
            delta_weights=(0.5, 0.5),
            structure="shared",
            inverse_null_vocabulary_size=1000,
        )
        # Both components have support d < 1-delta <= 0.8.
        self.assertEqual(detector.update(0.9), -math.inf)
        self.assertEqual(detector.update(0.1), -math.inf)
        self.assertTrue(np.all(np.isneginf(detector.log_bayes_factor_history)))
        self.assertTrue(detector.alternative_killed)
        with self.assertRaisesRegex(RuntimeError, "posterior is undefined"):
            _ = detector.component_posterior_weights
        self.assertEqual(detector.posterior_probability(), 0.0)

    def test_loss_decision_and_anytime_threshold(self) -> None:
        detector = self._detector("shared")
        expected_threshold = (10.0 / 2.0) * (0.8 / 0.2)
        self.assertAlmostEqual(detector.loss_threshold(10.0, 2.0), expected_threshold)
        self.assertFalse(detector.declare_watermark(1.0, 1.0))
        self.assertTrue(detector.declare_watermark(0.1, 1.0))
        self.assertFalse(detector.crosses_anytime_threshold(0.05))


if __name__ == "__main__":
    unittest.main(verbosity=2)
