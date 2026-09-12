"""Tests for the effective tail-width estimator (estimate_tail_width.py)."""

from __future__ import annotations

import math
import unittest
from unittest.mock import Mock, patch

import numpy as np

import estimate_tail_width as est


class LikelihoodTests(unittest.TestCase):
    def test_density_integrates_to_one_for_every_width(self) -> None:
        # log_likelihood must be built from a probability density, else the
        # profile compares incommensurable things.
        grid = np.linspace(1e-9, 1.0 - 1e-9, 400_001)
        for delta in (0.05, 0.2, 0.5):
            for width in (1, 4, 999):
                d = np.full(grid.size, delta)
                log_r = np.log(grid)
                head = (delta / (1 - delta)) * log_r
                tail = math.log(width) + (width / delta - 1.0) * log_r
                density = np.exp(head) + np.exp(tail)
                self.assertAlmostEqual(float(np.trapezoid(density, grid)), 1.0, places=4)

    def test_cdf_is_the_integral_of_the_density(self) -> None:
        deltas = np.array([0.1, 0.3])
        for width in (1, 5, 100):
            at_one = est.fitted_cdf(np.ones(2), deltas, width)
            np.testing.assert_allclose(at_one, 1.0, atol=1e-12)
            self.assertTrue(np.all(est.fitted_cdf(np.full(2, 0.5), deltas, width) < 1.0))

    def test_cdf_is_monotone(self) -> None:
        r = np.linspace(1e-6, 1 - 1e-6, 500)
        f = est.fitted_cdf(r, np.full(r.size, 0.25), 3)
        self.assertTrue(np.all(np.diff(f) > 0))


class IdentifiabilityTests(unittest.TestCase):
    def test_estimator_recovers_a_known_width(self) -> None:
        rng = np.random.default_rng(0)
        grid = (1, 2, 4, 8, 16, 64, 256)
        for truth in (1, 4, 64):
            deltas = rng.uniform(0.05, 0.5, size=6_000)
            curve = est.profile(est.simulate_gumbel(deltas, truth, rng), deltas, grid)
            best = max(curve, key=lambda item: item[1])[0]
            self.assertEqual(best, truth)

    def test_simulator_matches_the_fitted_cdf(self) -> None:
        # If the generator and the density disagree, the estimate is meaningless.
        rng = np.random.default_rng(1)
        deltas = np.full(200_000, 0.3)
        draws = est.simulate_gumbel(deltas, 4, rng)
        statistic, p_value = est.kolmogorov_smirnov(est.fitted_cdf(draws, deltas, 4))
        self.assertGreater(p_value, 0.01, msg=f"KS D={statistic}")


class KolmogorovSmirnovTests(unittest.TestCase):
    def test_nearly_perfect_uniform_grids_have_survival_probability_one(self) -> None:
        for n in (10_000, 100_000):
            statistic, p_value = est.kolmogorov_smirnov((np.arange(n) + .5) / n)
            self.assertAlmostEqual(statistic, .5 / n, places=14)
            self.assertEqual(p_value, 1.0)

    def test_survival_matches_independent_converged_series_away_from_zero(self) -> None:
        for power in (1.05, 1.2, 2):
            n = 100
            values = ((np.arange(n) + .5) / n) ** power
            statistic, p_value = est.kolmogorov_smirnov(values)
            scaled = (math.sqrt(n) + .12 + .11 / math.sqrt(n)) * statistic
            expected = 2 * math.fsum((-1) ** (k - 1) * math.exp(-2 * k * k * scaled * scaled)
                                     for k in range(1, 10001))
            self.assertAlmostEqual(p_value, expected, places=14)

    def test_invalid_ks_values_are_rejected(self) -> None:
        for values in ([], [[.5]], [np.nan], [-.1], [1.1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                est.kolmogorov_smirnov(np.asarray(values))

    def test_uniform_input_is_not_rejected(self) -> None:
        rng = np.random.default_rng(2)
        _, p_value = est.kolmogorov_smirnov(rng.random(20_000))
        self.assertGreater(p_value, 0.01)

    def test_clearly_non_uniform_input_is_rejected(self) -> None:
        rng = np.random.default_rng(3)
        _, p_value = est.kolmogorov_smirnov(rng.random(20_000) ** 2)
        self.assertLess(p_value, 1e-6)


class DependenceReductionTests(unittest.TestCase):
    def test_first_use_keeps_sequence_order_for_nonadjacent_repeats(self) -> None:
        keys = np.array([30, 10, 30, 20, 10, 40, 20])
        np.testing.assert_array_equal(est.first_per_group(keys), [0, 1, 3, 5])

    def test_uninformative_first_use_excludes_later_informative_reuse(self) -> None:
        # Address 30 first appears below the deficit floor.  Its informative
        # reuse must not enter the likelihood, even though filtering first
        # would keep it.  Address 10 has two informative uses; retain only one.
        keys = np.array([30, 10, 30, 20, 10, 40])
        informative = np.array([False, True, True, True, True, False])
        np.testing.assert_array_equal(
            est.informative_first_per_group(keys, informative), [1, 3]
        )

    def test_bootstrap_preserves_complete_documents_and_multiplicity(self) -> None:
        pivots = np.arange(1, 7, dtype=float) / 10
        deltas = pivots / 2
        documents = np.array([20, 10, 20, 30, 30, 30])
        rng = Mock()
        # Documents sorted by ID are 10, 20, 30.  Draw 30 twice and 10 once,
        # then 20 three times; unequal cluster sizes must not be equalized.
        rng.integers.side_effect = [np.array([2, 2, 0]), np.array([1, 1, 1])]
        with patch.object(est, "profile", side_effect=[[(1, 2.), (2, 1.)],
                                                     [(1, 1.), (2, 2.)]]) as fit:
            result = est.cluster_bootstrap_estimate(
                pivots, deltas, documents, (1, 2), replicates=2, rng=rng
            )
        expected_indices = ([3, 4, 5, 3, 4, 5, 1], [0, 2, 0, 2, 0, 2])
        for call, indices in zip(fit.call_args_list, expected_indices):
            np.testing.assert_array_equal(call.args[0], pivots[list(indices)])
            np.testing.assert_array_equal(call.args[1], deltas[list(indices)])
        self.assertEqual(result["n_documents"], 3)
        self.assertEqual(result["n_addresses"], 6)
        self.assertEqual(result["argmax_share"], {"1": .5, "2": .5})


class PayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Exercise the actual informative sample sizes as well as the larger
        # comparison sample; recovery need not be perfect at the smaller sizes.
        cls.payload = est.run()

    def test_identifiability_is_reported_at_the_observed_sample_sizes(self) -> None:
        # The check used to run only at 8,000 simulated positions, seven to
        # eight times what either model supplies after first-use selection and
        # the deficit filter.  Recovery at a sample size the data do not have
        # says nothing about whether J is identified here.
        blocks = self.payload["identifiability_check"]["by_sample_size"]
        observed = {str(int(m["distinct_addresses"]["n"]))
                    for m in self.payload["models"].values()}
        self.assertTrue(observed <= set(blocks), msg=f"{observed} vs {set(blocks)}")
        for size in observed:
            self.assertGreaterEqual(blocks[size]["replicates"], 100, msg=size)

    def test_the_fitted_widths_are_the_ones_recovery_supports(self) -> None:
        # The reported conclusion is J = 1 or 2.  Those are the widths the
        # check has to support at the observed sizes; the middle rungs need not
        # be, and are not, which is why the wording is qualified.
        blocks = self.payload["identifiability_check"]["by_sample_size"]
        for size in {str(int(m["distinct_addresses"]["n"]))
                     for m in self.payload["models"].values()}:
            rates = {row["true_live_tail"]: row["recovery_rate"]
                     for row in blocks[size]["by_truth"]}
            self.assertEqual(rates[1], 1.0, msg=size)
            self.assertGreaterEqual(rates[2], 0.9, msg=size)

    def test_large_sample_recovery_is_uniform(self) -> None:
        # The contrast the manuscript draws: uniform at 8,000, not at the sizes
        # the released outputs actually supply.
        big = self.payload["identifiability_check"]["by_sample_size"]["8000"]
        self.assertTrue(big["recovered_always"])

    def test_released_tails_are_narrow_and_full_width_fits_worse(self) -> None:
        for name, block in self.payload["models"].items():
            fit = block["distinct_addresses"]["goodness_of_fit"]
            self.assertLessEqual(block["distinct_addresses"]["estimate"], 4, msg=name)
            self.assertLess(
                fit["fitted"]["ks_statistic"],
                fit["assumed_full_width"]["ks_statistic"],
                msg=f"{name}: the fitted width should track the data better",
            )
            # These are nominal KS values; the document bootstrap, not a
            # position-independence test, quantifies width-selection stability.
            self.assertLess(fit["assumed_full_width"]["ks_p_value"], 1e-18, msg=name)

    def test_inverse_validity_condition_is_far_from_zero(self) -> None:
        # The limiting inverse alternative needs (log V) p_(2) -> 0.
        for name, block in self.payload["models"].items():
            self.assertGreater(
                block["inverse_limit_condition_log_v_times_p2"], 1.0, msg=name
            )

    def test_payload_is_json_serialisable(self) -> None:
        import json

        json.dumps(self.payload)
