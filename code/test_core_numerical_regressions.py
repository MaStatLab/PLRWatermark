"""Regression cases found by the September 2026 core numerical audit."""

from decimal import Decimal, localcontext
import math
from pathlib import Path
import unittest

import numpy as np
from scipy.stats import binomtest

import benchmark_paper_experiment as paper
import benchmark_contamination as contamination
import check_clean_quadrature as clean_check
import check_contamination_quadrature as contamination_check
import deficit_persistence_sweep as deficit
import width_persistence_sweep as width
import inverse_support_edge as edge
import regime_sweep


class InverseMarginalStabilityTests(unittest.TestCase):
    @staticmethod
    def reference(d: float, config: paper.BenchmarkConfig) -> float:
        with localcontext() as ctx:
            ctx.prec = 80
            a, b = map(Decimal.from_float, paper.effective_prior_support(config))
            x = Decimal.from_float(d)
            upper = min(b, 1 - x)
            def primitive(delta):
                return 2 * (-(1 - delta).ln() - x / (1 - delta))
            marginal = (primitive(upper) - primitive(a)) / (b - a)
            null = Decimal.from_float(float(paper.inverse_exact_null_density(
                np.array([d]), config.vocabulary_size
            )[0]))
            return float(marginal.ln() - null.ln())

    def test_positive_near_edge_and_narrow_prior_match_high_precision(self):
        for config in (
            paper.BenchmarkConfig(),
            paper.BenchmarkConfig(prior_low=.1, prior_high=.100000001),
        ):
            endpoint = 1 - config.prior_low
            pivots = np.array([0., .2, .6, endpoint - 1e-8,
                               endpoint - 1e-10, np.nextafter(endpoint, 0.)])
            actual = paper.inverse_bayes_log_ratio(pivots, config)
            expected = np.array([self.reference(float(d), config) for d in pivots])
            np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-9)
            self.assertTrue(np.isfinite(actual).all())

    def test_support_zero_and_multitoken_paths(self):
        config = paper.BenchmarkConfig()
        endpoint = 1 - config.prior_low
        near = endpoint - 1e-10
        increments = paper.inverse_bayes_log_ratio(
            np.array([[near, .2, near], [endpoint, .2, .3]]), config
        )
        paths = np.cumsum(increments, axis=1)
        self.assertTrue(np.isfinite(paths[0]).all())
        self.assertTrue(np.isneginf(paths[1]).all())
        for invalid in (-.1, 1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                paper.inverse_bayes_log_ratio(np.array([invalid]), config)

    def test_nan_scores_cannot_be_silently_calibrated_or_counted_as_misses(self):
        with self.assertRaisesRegex(ValueError, "NaN calibration"):
            paper.calibrate_randomized_boundary({"bad": np.array([[np.nan]])}, .05)
        with self.assertRaisesRegex(ValueError, "NaN evaluation"):
            paper.expected_rejection_rate(np.array([[np.nan]]), np.zeros(1), np.zeros(1))
        # A genuine zero likelihood is permitted, unlike NaN.
        cutoffs, gamma = paper.calibrate_randomized_boundary(
            {"zero": np.full((10, 1), -np.inf)}, .05
        )
        self.assertEqual(cutoffs["zero"][0], -np.inf)
        self.assertAlmostEqual(gamma["zero"][0], .05)


class PersistenceSafetyTests(unittest.TestCase):
    def test_smoke_paths_are_isolated_and_explicit_path_wins(self):
        for module in (deficit, width):
            for quick in (False, True):
                args = module.parse_args(["--quick"] if quick else [])
                expected = module.QUICK_RESULTS if quick else module.RESULTS
                self.assertEqual(module.resolve_results_dir(args.results_dir, quick=quick), expected)
            args = module.parse_args(["--quick", "--results-dir", "chosen"])
            self.assertEqual(module.resolve_results_dir(args.results_dir, quick=True), Path("chosen"))
            self.assertNotEqual(module.QUICK_RESULTS, module.RESULTS)

    def test_mcnemar_handles_large_counts_and_degenerate_reference(self):
        for b, c in ((0, 0), (5, 0), (600, 600), (1100, 1000), (1200, 0)):
            expected = binomtest(b, b + c, .5).pvalue if b + c else 1.
            for module in (deficit, width):
                self.assertAlmostEqual(module.mcnemar_exact(b, c), expected, places=11)

    def test_nondefault_support_reaches_the_tokenwise_prior(self):
        config = deficit.PersistenceConfig(delta_low=.1, delta_high=.2)
        self.assertEqual(paper.effective_prior_support(config.benchmark_config()), (.1, .2))
        self.assertEqual((config.benchmark_config().delta_low,
                          config.benchmark_config().delta_high), (.1, .2))

    def test_width_correlation_names_the_continuous_prequantization_quantity(self):
        regime = width.WidthRegime("Q", .8, "test")
        self.assertAlmostEqual(regime.continuous_copula_spearman,
                               6 / math.pi * math.asin(.8 ** 2 / 2))
        self.assertFalse(hasattr(regime, "rank_icc"))


class ContaminationDegenerateConfigurationTests(unittest.TestCase):
    def test_clean_only_prior_and_disabled_point_menu_work_end_to_end(self):
        config = contamination.ContaminationConfig(
            vocabulary_size=40, horizons=(3, 5), rho_true_grid=(0., 1.),
            rho_prior_grid=(0.,), rho_prior_weights=(1.,), rho_point_masses=(),
            n_calibration=20, n_evaluation_null=12, n_evaluation_alternative=12,
            batch_size=12, bayes_quadrature_nodes=8, gumbel_lookup_size=1001,
            dirichlet_c_nodes=2001, dirichlet_alpha_grid=(1., math.inf),
        )
        indicators = {}
        rows, metadata = contamination.run_contamination_benchmark(config, indicator_sink=indicators)
        indexed = {(r["scheme"], r["rho_true"], r["horizon"], r["method"]): r for r in rows}
        for scheme in ("gumbel", "inverse"):
            for rho in config.rho_true_grid:
                for horizon in config.horizons:
                    for prefix in ("bayes_shared", "bayes_tokenwise"):
                        clean = indexed[scheme, rho, horizon, prefix + "_clean"]
                        robust = indexed[scheme, rho, horizon, prefix + "_robust"]
                        self.assertEqual(clean["type_ii_error"], robust["type_ii_error"])
        report = contamination.build_rho_diagnostic(rows, metadata, config, indicators)
        self.assertEqual(report["point_mass_diagnostic_status"], "disabled_empty_point_mass_menu")
        self.assertEqual(report["oracle_reading"]["records"], [])
        self.assertEqual(report["single_fixed_reading"]["records"], [])
        self.assertEqual(report["readings_by_point_mass_family"]["families"], {})
        self.assertEqual(report["mcnemar_robust_vs_point_mass"], [])


class QuadratureCoverageTests(unittest.TestCase):
    def test_four_decimal_equality_checks_rounding_boundaries(self):
        self.assertFalse(clean_check.identical_to_four_decimals([.123449, .123451]))
        self.assertTrue(clean_check.identical_to_four_decimals([.123441, .123449]))

    def test_clean_checker_includes_both_union_hierarchies(self):
        config = paper.BenchmarkConfig(
            vocabulary_size=40, max_horizon=6, n_calibration=20,
            n_evaluation_null=12, n_evaluation_alternative=12, batch_size=12,
            gumbel_lookup_size=1001, dirichlet_c_nodes=2001,
        )
        result = clean_check.run_check(config, node_counts=(8, 12), reference_nodes=12, horizons=(3, 6))
        methods = {cell["method"] for cell in result["cells"]}
        self.assertIn(paper.UNIONTAIL_SHARED_METHOD, methods)
        self.assertIn(paper.UNIONTAIL_TOKENWISE_METHOD, methods)
        self.assertEqual(result["coverage"]["cells_checked"], len(result["cells"]))
        self.assertEqual(result["horizons"], [3, 6])
        self.assertFalse(result["coverage"]["certified_uniform_error_bound"])

    def test_contamination_checker_includes_shared_union_and_records_path_scope(self):
        config = contamination.ContaminationConfig(
            vocabulary_size=40, horizons=(3, 6), dirichlet_c_nodes=2001,
            dirichlet_alpha_grid=(1., math.inf), batch_size=4,
        )
        result = contamination_check.run_check(n_paths=4, node_counts=(8, 12),
                                               reference_nodes=12, base=config)
        methods = {cell["rule"] for cell in result["cells"]}
        self.assertIn(contamination.UNIONTAIL_ROBUST_METHOD, methods)
        self.assertIn("not Type II", result["coverage"]["quantity"])
        self.assertEqual(len(result["cells"]), 16)

    def test_diagnostic_cannot_set_the_regret_minimum(self):
        regret, best = regime_sweep.regret_table({
            "eligible": {"T": .1}, "h_spike_0.01": {"T": .01},
        })
        self.assertEqual(best["T"], .1)
        self.assertAlmostEqual(regret["h_spike_0.01"]["T"], -.09)


class ContinuousRankScopeTests(unittest.TestCase):
    def test_historical_generator_is_explicitly_independent_of_vocabulary(self):
        args = (2, 12)
        a = edge.simulate_inverse(*args, 2, np.zeros(2), 1, np.random.default_rng(5))
        b = edge.simulate_inverse(*args, 50272, np.zeros(2), 1, np.random.default_rng(5))
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
