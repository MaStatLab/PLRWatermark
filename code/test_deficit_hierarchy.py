"""Tests for the partial-pooling deficit prior.

The claim that justifies reporting this rule is CONTAINMENT: both existing
hierarchies must be exact members of the family, not limits it approaches.  If
that fails the pooled rule is a third arbitrary detector rather than the one
the other two sit inside, and its regret comparison means nothing.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

import benchmark_paper_experiment as bpe
import deficit_hierarchy as dh
import dirichlet_detector as dd

VOCAB = 1_000
LOW, HIGH = 0.001, 0.5


class BetaQuadratureTests(unittest.TestCase):
    def test_weights_are_normalized_and_reproduce_the_mean(self) -> None:
        for kappa in (0.5, 2.0, 8.0, 32.0, 128.0, 1024.0):
            for mu in (0.02, 0.25, 0.5, 0.8, 0.97):
                x, w = dh.beta_quadrature(kappa * mu, kappa * (1.0 - mu), 64)
                self.assertAlmostEqual(float(w.sum()), 1.0, places=12)
                self.assertAlmostEqual(float((x * w).sum()), mu, places=8)

    def test_reproduces_the_beta_variance(self) -> None:
        """Gauss-Legendre would fail this at large kappa; Gauss-Jacobi does not."""

        for kappa in (2.0, 32.0, 512.0):
            for mu in (0.3, 0.5):
                x, w = dh.beta_quadrature(kappa * mu, kappa * (1.0 - mu), 64)
                mean = float((x * w).sum())
                var = float((w * (x - mean) ** 2).sum())
                self.assertAlmostEqual(var, mu * (1 - mu) / (kappa + 1.0), places=9)

    def test_rejects_nonpositive_parameters(self) -> None:
        for a, b in ((0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)):
            with self.assertRaises(ValueError):
                dh.beta_quadrature(a, b, 16)


class ContainmentTests(unittest.TestCase):
    """The two published hierarchies must be reproduced exactly."""

    def setUp(self) -> None:
        self.pivots = np.random.default_rng(11).uniform(size=(120, 30))

    def test_tokenwise_atom_reproduces_the_tokenwise_rule(self) -> None:
        atom = dh.PooledDeficitGrid(
            vocabulary_size=VOCAB, delta_low=LOW, delta_high=HIGH,
            prior=dh.PooledDeficitPrior(kappas=(dh.TOKENWISE_KAPPA,)),
        )
        self.assertEqual(atom.n_components(), 1)
        reference = bpe.GumbelBayesLookup(bpe.BenchmarkConfig(vocabulary_size=VOCAB))
        pooled = atom.shared_paths(self.pivots)
        expected = np.cumsum(reference(self.pivots), axis=1)
        # The reference is itself an interpolated table, so its own grid error
        # is the floor here.
        self.assertLess(float(np.max(np.abs(pooled - expected))), 5e-6)

    def test_infinite_concentration_reproduces_the_shared_rule(self) -> None:
        layer = dh.PooledDeficitGrid(
            vocabulary_size=VOCAB, delta_low=LOW, delta_high=HIGH,
            prior=dh.PooledDeficitPrior(kappas=(math.inf,), mu_nodes=96),
        )
        deltas, weights = dd.gauss_legendre_delta_grid(LOW, HIGH, 96)
        shared = dd.DirichletBayesGrid(
            delta_grid=deltas, delta_weights=weights,
            alpha_grid=(float("inf"),), tail_size=VOCAB - 1,
        )
        expected, _ = shared.shared_paths(self.pivots)
        self.assertLess(
            float(np.max(np.abs(layer.shared_paths(self.pivots) - expected))), 1e-7
        )

    def test_the_default_grid_contains_both_endpoints(self) -> None:
        grid = dh.PooledDeficitGrid(
            vocabulary_size=VOCAB, delta_low=LOW, delta_high=HIGH
        )
        triples = grid.triples
        self.assertIn(
            (dh.TOKENWISE_MU, dh.TOKENWISE_KAPPA),
            [(mu, k) for mu, k, _ in triples],
        )
        self.assertTrue(any(math.isinf(k) for _, k, _ in triples))
        self.assertAlmostEqual(sum(w for _, _, w in triples), 1.0, places=12)


class PooledGridTests(unittest.TestCase):
    def test_analytic_component_mass_is_one(self) -> None:
        """Each component density integrates to one, by the closed form.

        For the equal-tail spike ``int_0^1 r^(D/(1-D)) dr = 1-D`` and
        ``int_0^1 (V-1) r^((V-1)/D - 1) dr = D``, so every component's mass is
        ``sum_k w_k [(1-D_k) + D_k] = sum_k w_k``.  Numerical integration cannot
        check this directly: the tail term is a spike of height ``V-1`` in a
        neighbourhood of ``r=1`` of width ``O(D/V)``, and adaptive quadrature
        misses it.  Testing the weights instead is exact.
        """

        grid = dh.PooledDeficitGrid(
            vocabulary_size=VOCAB, delta_low=LOW, delta_high=HIGH,
            prior=dh.PooledDeficitPrior(kappas=(0.5, 2.0, 32.0, math.inf)),
        )
        for deltas, weights in zip(grid._deltas, grid._weights):
            mass = float(np.sum(weights * ((1.0 - deltas) + deltas)))
            self.assertAlmostEqual(mass, 1.0, places=12)

    def test_component_density_matches_a_direct_implementation(self) -> None:
        """The tabulated component ratio equals the formula it claims to be."""

        grid = dh.PooledDeficitGrid(
            vocabulary_size=VOCAB, delta_low=LOW, delta_high=HIGH,
            prior=dh.PooledDeficitPrior(kappas=(2.0, 32.0, math.inf), mu_nodes=6),
        )
        r = np.array([1e-6, 0.01, 0.3, 0.7, 0.999, 1.0 - 1e-9])
        got = grid.component_log_ratio(r)
        for j, (mu, kappa, _) in enumerate(grid.triples):
            if math.isinf(kappa):
                x = np.array([mu]); w = np.array([1.0])
            else:
                x, w = dh.beta_quadrature(kappa * mu, kappa * (1.0 - mu), 64)
            d = LOW + (HIGH - LOW) * x
            top = r[:, None] ** (d / (1.0 - d))[None, :]
            tail = (VOCAB - 1.0) * r[:, None] ** ((VOCAB - 1.0) / d - 1.0)[None, :]
            direct = np.log((top + tail) @ w)
            self.assertLess(float(np.max(np.abs(got[:, j] - direct))), 1e-10)

    def test_null_expectation_of_the_bayes_factor_is_one(self) -> None:
        """E_0[BF] = 1, which is what Ville's bound needs.

        Checked at one token: the Bayes factor is heavy tailed under the null,
        so the sample mean converges slowly and a multi-token product is a much
        noisier estimator of the same quantity.  The tolerance below is about
        four standard errors of this estimator at this sample size, not a claim
        about the density, which the two tests above pin exactly.
        """

        grid = dh.PooledDeficitGrid(
            vocabulary_size=VOCAB, delta_low=LOW, delta_high=HIGH,
            prior=dh.PooledDeficitPrior(kappas=(2.0, 32.0, math.inf), mu_nodes=8),
        )
        rng = np.random.default_rng(5)
        pivots = rng.uniform(size=(400_000, 1))
        factors = np.exp(grid.shared_paths(pivots)[:, 0])
        self.assertAlmostEqual(float(np.mean(factors)), 1.0, delta=0.02)

    def test_shared_paths_are_cumulative_and_finite(self) -> None:
        grid = dh.PooledDeficitGrid(
            vocabulary_size=VOCAB, delta_low=LOW, delta_high=HIGH,
            prior=dh.PooledDeficitPrior(kappas=(2.0, math.inf), mu_nodes=6),
        )
        pivots = np.random.default_rng(3).uniform(size=(40, 12))
        paths = grid.shared_paths(pivots)
        self.assertEqual(paths.shape, (40, 12))
        self.assertTrue(np.all(np.isfinite(paths)))
        prefix = grid.shared_paths(pivots[:, :5])
        self.assertLess(float(np.max(np.abs(paths[:, :5] - prefix))), 1e-12)

    def test_rejects_an_invalid_support(self) -> None:
        for low, high in ((0.5, 0.5), (0.6, 0.2), (-0.1, 0.5), (0.1, 1.0)):
            with self.assertRaises(ValueError):
                dh.PooledDeficitGrid(
                    vocabulary_size=VOCAB, delta_low=low, delta_high=high
                )


if __name__ == "__main__":
    unittest.main()
