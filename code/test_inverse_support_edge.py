"""Tests for the inverse support-edge diagnostic (inverse_support_edge.py)."""

from __future__ import annotations

import math
import unittest

import numpy as np

import inverse_support_edge as edge
from bayesian_watermark import inverse_null_logpdf


class SimulatorTests(unittest.TestCase):
    def test_pivots_lie_in_the_unit_interval(self) -> None:
        rng = np.random.default_rng(0)
        for width in (1, 4, 64):
            p = edge.simulate_inverse(20, 50, 5_000, np.full(20, 0.2), width, rng)
            self.assertTrue(np.all((p >= 0.0) & (p < 1.0)))

    def test_zero_deficit_approximates_the_large_vocabulary_null(self) -> None:
        # With Delta = 0 the leading token has all the mass, so the selected
        # continuous rank is independent of U, giving the triangular limiting
        # null.  The finite-M null is close for M=5,000, not identical.
        rng = np.random.default_rng(1)
        p = edge.simulate_inverse(1, 200_000, 5_000, np.zeros(1), 1, rng).ravel()
        q = edge.simulate_null(1, 200_000, 5_000, rng).ravel()
        # Compare means and the mass below 1/2; both are O(1/sqrt(n)) apart.
        self.assertAlmostEqual(p.mean(), q.mean(), places=2)
        self.assertAlmostEqual(np.mean(p < 0.5), np.mean(q < 0.5), places=2)

    def test_wide_tail_matches_the_triangular_limit_mean(self) -> None:
        rng = np.random.default_rng(2)
        delta = 0.2
        p = edge.simulate_inverse(200, 200, 50_272, np.full(200, delta), 4_096, rng)
        self.assertAlmostEqual(float(p.mean()), (1.0 - delta) / 3.0, places=2)

    def test_sparse_tail_puts_mass_beyond_the_working_support(self) -> None:
        # The whole point: at J = 1 the limiting alternative's support is wrong.
        rng = np.random.default_rng(3)
        delta = 0.5
        sparse = edge.simulate_inverse(300, 200, 50_272, np.full(300, delta), 1, rng)
        wide = edge.simulate_inverse(300, 200, 50_272, np.full(300, delta), 4_096, rng)
        self.assertGreater(np.mean(sparse >= 1.0 - delta), 0.02)
        self.assertLess(np.mean(wide >= 1.0 - delta), 0.002)


class NullTests(unittest.TestCase):
    def test_simulated_null_matches_the_exact_null_density(self) -> None:
        rng = np.random.default_rng(4)
        M = 1_000
        draws = edge.simulate_null(1, 400_000, M, rng).ravel()
        edges = np.linspace(0.0, 1.0, 21)
        observed, _ = np.histogram(draws, bins=edges, density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        expected = np.exp(inverse_null_logpdf(centres, M))
        np.testing.assert_allclose(observed, expected, rtol=0.05, atol=0.02)


class BayesFactorTests(unittest.TestCase):
    def test_clean_mixture_is_a_null_martingale(self) -> None:
        rng = np.random.default_rng(5)
        deltas, weights = edge.dirichlet.gauss_legendre_delta_grid(0.001, 0.5, 16)
        pivots = edge.simulate_null(4_000, 3, 1_000, rng)
        paths = edge.shared_log_bayes_factor(
            pivots, deltas, weights, 1_000, rho_grid=(0.0,), rho_weights=(1.0,)
        )
        self.assertAlmostEqual(float(np.mean(np.exp(paths))), 1.0, places=1)

    def test_rho_prior_bounds_the_ratio_below_by_rho(self) -> None:
        # rho + (1-rho) L >= rho is what caps the support-edge damage.
        rng = np.random.default_rng(6)
        deltas, weights = edge.dirichlet.gauss_legendre_delta_grid(0.001, 0.5, 8)
        # A pivot beyond every component's support annihilates the clean rule.
        pivots = np.full((3, 4), 0.999)
        clean = edge.shared_log_bayes_factor(
            pivots, deltas, weights, 1_000, rho_grid=(0.0,), rho_weights=(1.0,)
        )
        robust = edge.shared_log_bayes_factor(
            pivots, deltas, weights, 1_000,
            rho_grid=edge.RHO_GRID, rho_weights=edge.RHO_WEIGHTS,
        )
        self.assertTrue(np.all(np.isneginf(clean)))
        self.assertTrue(np.all(np.isfinite(robust)))


class McNemarTests(unittest.TestCase):
    def test_perfect_concordance_returns_one(self) -> None:
        self.assertEqual(edge.exact_mcnemar_p_value(0, 0), 1.0)

    def test_symmetry_and_known_value(self) -> None:
        self.assertAlmostEqual(
            edge.exact_mcnemar_p_value(7, 1), edge.exact_mcnemar_p_value(1, 7)
        )
        self.assertAlmostEqual(edge.exact_mcnemar_p_value(5, 0), 2.0 / 2**5)


class PayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = edge.run(
            horizon=40, n_documents=150, n_calibration=200, nodes=12
        )

    def test_records_zero_contamination(self) -> None:
        self.assertEqual(self.payload["design"]["rho_true"], 0.0)
        self.assertFalse(self.payload["design"]["exact_finite_vocabulary_alternative"])
        self.assertIsNone(self.payload["design"]["rank_approximation_error_bound"])

    def test_edge_mass_decreases_as_the_tail_widens(self) -> None:
        rows = [r for r in self.payload["edge_mass"] if r["delta"] == 0.5]
        rows.sort(key=lambda r: r["live_tail"])
        masses = [r["fraction_at_or_beyond_one_minus_delta"] for r in rows]
        self.assertGreater(masses[0], masses[-1])

    def test_validity_condition_is_recorded_and_diverges_at_j_one(self) -> None:
        row = next(
            r for r in self.payload["edge_mass"]
            if r["delta"] == 0.1 and r["live_tail"] == 1
        )
        self.assertAlmostEqual(
            row["log_m_times_second_largest"], math.log(50_272) * 0.1, places=6
        )
        self.assertGreater(row["log_m_times_second_largest"], 1.0)

    def test_rescue_rows_are_complete_and_serialisable(self) -> None:
        import json

        widths = {r["live_tail"] for r in self.payload["rho_rescue"]}
        self.assertEqual(widths, set(edge.TAIL_WIDTHS))
        for row in self.payload["rho_rescue"]:
            self.assertGreaterEqual(row["exact_mcnemar_p_value"], 0.0)
            self.assertLessEqual(row["exact_mcnemar_p_value"], 1.0)
        json.dumps(self.payload)


if __name__ == "__main__":
    unittest.main()
