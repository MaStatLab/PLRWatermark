"""Tests for the deficit persistence sweep.

The sweep exists to give partial pooling something to detect: the published
designs are all-or-nothing on ``Delta``, so a pooled rule can only lose on
them.  Its endpoints must therefore reproduce those two designs exactly, or the
comparison is against something other than the published experiments.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

import deficit_persistence_sweep as sweep


class RegimeTests(unittest.TestCase):
    def test_endpoints_reproduce_the_published_designs(self) -> None:
        labels = [r.label for r in sweep.DEFAULT_REGIMES]
        self.assertEqual(labels, ["P1", "P2", "P3", "P4", "P5", "P6"])
        first, last = sweep.DEFAULT_REGIMES[0], sweep.DEFAULT_REGIMES[-1]
        # Gaussian copula: rho = 1 is the shared design and rho = 0 the
        # tokenwise one.  The ICC is (6/pi) arcsin(rho^2 / 2), NOT rho^2: the
        # latent pair correlates at rho^2 and the probability-integral
        # transform maps that to the uniform marginals' correlation.  At rho=1
        # the value is 1 analytically and 1 - 1.1e-16 in floating point, so the
        # endpoints are asserted to tolerance rather than exactly.
        self.assertEqual(first.rho, 1.0)
        self.assertAlmostEqual(first.icc, 1.0, places=12)
        self.assertEqual(last.rho, 0.0)
        self.assertEqual(last.icc, 0.0)

    def test_unit_rho_holds_the_deficit_constant_within_a_document(self) -> None:
        regime = sweep.DEFAULT_REGIMES[0]
        deltas = regime.deltas(np.random.default_rng(0), 50, 20, 0.001, 0.5)
        self.assertTrue(np.allclose(deltas.std(axis=1), 0.0))
        self.assertGreater(float(deltas.mean(axis=1).std()), 0.1)

    def test_the_tokenwise_endpoint_is_uniform_and_unpooled(self) -> None:
        """P6 must be i.i.d. uniform, matching the tokenwise generator."""

        regime = sweep.DEFAULT_REGIMES[-1]
        deltas = regime.deltas(np.random.default_rng(1), 4_000, 400, 0.001, 0.5)
        # Uniform on the support: mean at the midpoint, sd at range/sqrt(12).
        self.assertAlmostEqual(float(deltas.mean()), 0.2505, delta=0.004)
        self.assertAlmostEqual(float(deltas.std()), 0.499 / math.sqrt(12), delta=0.004)
        # No document-level component beyond sampling noise.
        self.assertLess(float(deltas.mean(axis=1).std()), 0.01)

    def test_persistence_decreases_with_kappa(self) -> None:
        """The realized between-document variance share is the sweep's axis."""

        rng = np.random.default_rng(7)
        seen = []
        for regime in sweep.DEFAULT_REGIMES[:-1]:   # P6 changes the mu law too
            deltas = regime.deltas(rng, 600, 200, 0.001, 0.5)
            within = deltas.std(axis=1).mean()
            between = deltas.mean(axis=1).std()
            seen.append(between**2 / (between**2 + within**2))
        self.assertEqual(seen, sorted(seen, reverse=True))
        self.assertAlmostEqual(seen[0], 1.0, places=6)
        self.assertLess(seen[-1], 0.7)


    def test_every_regime_has_the_same_marginal(self) -> None:
        """The copula exists to move persistence WITHOUT moving the marginal.

        The Beta construction this replaced had
        Var(X_t) = 1/12 + 1/(6(kappa+1)), so lowering the concentration changed
        the marginal and the dependence together and no gain could be
        attributed to either.
        """

        moments = []
        for regime in sweep.DEFAULT_REGIMES:
            deltas = regime.deltas(np.random.default_rng(7), 3_000, 150, 0.001, 0.5)
            moments.append((float(deltas.mean()), float(deltas.std())))
        means = [m for m, _ in moments]
        sds = [s for _, s in moments]
        self.assertLess(max(means) - min(means), 0.006)
        self.assertLess(max(sds) - min(sds), 0.006)

    def test_realized_icc_tracks_rho_squared(self) -> None:
        for regime in sweep.DEFAULT_REGIMES:
            deltas = regime.deltas(np.random.default_rng(11), 3_000, 120, 0.001, 0.5)
            within = deltas.var(axis=1, ddof=1).mean()
            between = max(
                deltas.mean(axis=1).var(ddof=1) - within / deltas.shape[1], 0.0
            )
            icc = between / (between + within) if between + within else 1.0
            self.assertAlmostEqual(icc, regime.icc, delta=0.03)


class SweepTests(unittest.TestCase):
    def test_quick_sweep_runs_and_is_calibrated(self) -> None:
        config = sweep.PersistenceConfig(
            max_horizon=60, horizons=(30, 60), n_calibration=4_000,
            n_evaluation_null=4_000, n_evaluation_alternative=800,
        )
        payload = sweep.run(config)
        self.assertEqual(set(payload["type2_error"]), {"30", "60"})
        for horizon, by_rule in payload["type_i_error"].items():
            self.assertEqual(set(by_rule), set(sweep.RULES))
            for rule, value in by_rule.items():
                # 4,000 null documents at nominal .05: 4 binomial SEs is .0138.
                self.assertAlmostEqual(value, 0.05, delta=0.015, msg=f"{rule} {horizon}")

    def test_type_i_does_not_depend_on_the_alternative(self) -> None:
        """The null arms carry no Delta, so changing the regimes cannot move Type I.

        Under the exact pivot null the Gumbel pivot is Uniform(0,1) whatever the
        next-token distribution is, so the sweep draws its calibration and null
        arms directly and shares them across regimes.  If Type I moved when only
        the alternative changed, that sharing would be broken and every regime
        would be judged at a different realized size.
        """

        base = dict(
            max_horizon=40, horizons=(40,), n_calibration=1_500,
            n_evaluation_null=1_500, n_evaluation_alternative=300,
        )
        full = sweep.run(sweep.PersistenceConfig(**base))
        subset = sweep.run(
            sweep.PersistenceConfig(
                **base, regimes=sweep.DEFAULT_REGIMES[:2]
            )
        )
        self.assertEqual(full["type_i_error"], subset["type_i_error"])
        self.assertEqual(set(subset["regimes"]), {"P1", "P2"})
        # and the shared regimes are scored identically under both
        for rule in sweep.RULES:
            self.assertEqual(
                full["type2_error"]["40"][rule]["P1"],
                subset["type2_error"]["40"][rule]["P1"],
            )


if __name__ == "__main__":
    unittest.main()