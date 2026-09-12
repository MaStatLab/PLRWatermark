"""Tests for benchmark_paper_experiment.py."""

from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

import benchmark_paper_experiment as benchmark
import check_clean_quadrature
import paired_comparisons
from bayesian_watermark import SequentialBayesDetector


class BenchmarkExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = benchmark.BenchmarkConfig(
            max_horizon=12,
            n_calibration=40,
            n_evaluation_null=30,
            n_evaluation_alternative=30,
            batch_size=10,
            bayes_quadrature_nodes=32,
            gumbel_lookup_size=8_001,
        )

    def test_inverse_bayes_marginal_normalizes(self) -> None:
        # Integrate f1 = exp(log ratio) f0; avoid the endpoints where a density
        # version can jump without changing the distribution.
        grid = (np.arange(200_000) + 0.5) / 200_000
        f0 = benchmark.inverse_exact_null_density(grid, self.config.vocabulary_size)
        f1 = np.exp(benchmark.inverse_bayes_log_ratio(grid, self.config)) * f0
        self.assertAlmostEqual(float(f1.mean()), 1.0, delta=2e-4)

    def test_gumbel_bayes_marginal_normalizes(self) -> None:
        lookup = benchmark.GumbelBayesLookup(self.config)
        # Midpoint integration is accurate enough to resolve the narrow
        # tail-token peak near one at V=1000.
        grid = (np.arange(200_000) + 0.5) / 200_000
        integral = float(np.exp(lookup(grid)).mean())
        self.assertAlmostEqual(integral, 1.0, delta=8e-3)

    def test_exact_inverse_generators_stay_in_support(self) -> None:
        rng = np.random.default_rng(123)
        for simulator in (
            benchmark.simulate_inverse_null,
            benchmark.simulate_inverse_alternative,
            benchmark.simulate_inverse_alternative_shared_delta,
        ):
            pivots = simulator(rng, 200, 30, self.config)
            self.assertTrue(np.all(pivots >= 0.0))
            self.assertTrue(np.all(pivots < 1.0))

    def test_randomized_boundary_hits_discrete_target(self) -> None:
        values = np.tile(np.arange(5, dtype=float), (20, 1)).T.reshape(20, 5)
        values = np.repeat(values[:, :1], 4, axis=1)
        cutoffs, gammas = benchmark.calibrate_randomized_boundary(
            {"discrete": values}, 0.25
        )
        rate = benchmark.expected_rejection_rate(
            values, cutoffs["discrete"], gammas["discrete"]
        )
        np.testing.assert_allclose(rate, 0.25, atol=1e-12)

    def test_boundary_randomization_is_reproducible_and_uses_the_reported_rule(self) -> None:
        kwargs = {
            "seed": 12345,
            "scenario": "shared_delta_equal_tail_sensitivity",
            "scheme": "gumbel",
            "horizon": 100,
            "n_documents": 20,
        }
        first = benchmark.boundary_randomization_uniforms(**kwargs)
        second = benchmark.boundary_randomization_uniforms(**kwargs)
        np.testing.assert_array_equal(first, second)
        different_cell = benchmark.boundary_randomization_uniforms(
            **{**kwargs, "horizon": 300}
        )
        self.assertFalse(np.array_equal(first, different_cell))

        scores = np.array([0.0, 1.0, 1.0, 2.0])
        uniforms = np.array([0.9, 0.2, 0.8, 0.1])
        rejected = benchmark.randomized_boundary_rejections(
            scores, 1.0, 0.5, uniforms
        )
        np.testing.assert_array_equal(rejected, [False, True, False, True])

    def test_type2_plot_values_mask_zeros_without_moving_positive_rates(self) -> None:
        one_miss = 1.0 / 5_000
        displayed, zero_count = benchmark._type2_plot_values(
            np.array([0.01, one_miss, 0.0, 2.0 * one_miss])
        )
        np.testing.assert_array_equal(zero_count, [False, False, True, False])
        np.testing.assert_array_equal(
            np.ma.getmaskarray(displayed), [False, False, True, False]
        )
        self.assertEqual(float(displayed[1]), one_miss)
        self.assertEqual(float(displayed[3]), 2.0 * one_miss)

        horizons = np.arange(1.0, 22.0)
        zero_run = np.zeros(horizons.size, dtype=bool)
        zero_run[2:20] = True
        selected = benchmark._selected_zero_bound_markers(
            horizons, zero_run, min_spacing=5.0
        )
        # Both endpoints survive, with sparse interior markers; positive-rate
        # horizons outside the run are never selected as upper bounds.
        np.testing.assert_array_equal(
            np.flatnonzero(selected), [2, 7, 12, 17, 19]
        )

    def test_tokenwise_slide_separates_zero_bounds_and_adds_early_zoom(self) -> None:
        config = replace(
            self.config,
            max_horizon=100,
            n_evaluation_alternative=5_000,
        )
        one_miss = 1.0 / config.n_evaluation_alternative
        rows = [
            {
                "scenario": "paper_text_iid_delta_equal_tail",
                "scheme": "gumbel",
                "method": "h_ars",
                "decision_rule": "fixed_horizon_mc_calibrated",
                "horizon": horizon,
                "type2_error": rate,
            }
            for horizon, rate in (
                (1, 0.5),
                (31, 2.0 * one_miss),
                (32, one_miss),
                (34, 0.0),
                (61, 2.0 * one_miss),
                (100, 0.0),
            )
        ]
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
            plt.Figure, "savefig"
        ), mock.patch.object(plt, "close"):
            benchmark.make_slide_plot(
                rows,
                Path(tmpdir) / "tokenwise",
                "paper_text_iid_delta_equal_tail",
                config,
                schemes=("gumbel",),
            )
            figure = plt.gcf()
            self.assertEqual(len(figure.axes), 2)
            full_axis, detail_axis = figure.axes
            self.assertEqual(full_axis.get_xlim(), (1.0, 100.0))
            self.assertEqual(detail_axis.get_xlim(), (1.0, 60.0))
            self.assertLess(full_axis.get_ylim()[0], one_miss)

            estimate_line = next(
                line
                for line in full_axis.lines
                if line.get_label() == r"$h_{\mathrm{ars}}$"
            )
            ydata = estimate_line.get_ydata()
            self.assertTrue(np.ma.isMaskedArray(ydata))
            np.testing.assert_array_equal(
                np.ma.getmaskarray(ydata),
                [False, False, False, True, False, True],
            )
            self.assertEqual(float(ydata[2]), one_miss)

            bound_lines = [
                line
                for line in full_axis.lines
                if line.get_marker() == "v" and line.get_linestyle() == "None"
            ]
            self.assertEqual(len(bound_lines), 1)
            np.testing.assert_allclose(
                bound_lines[0].get_ydata(),
                np.full(2, 3.0 / config.n_evaluation_alternative),
            )
            self.assertFalse(
                any(
                    line.get_linestyle() != "None"
                    and np.size(line.get_ydata()) > 1
                    and np.allclose(
                        np.asarray(line.get_ydata(), dtype=float),
                        3.0 / config.n_evaluation_alternative,
                    )
                    for line in full_axis.lines
                )
            )

        plt.close(figure)

    def test_tiny_benchmark_has_both_scenarios_and_rules(self) -> None:
        rows, metadata = benchmark.run_benchmark(self.config)
        scenarios = {row["scenario"] for row in rows}
        rules = {row["decision_rule"] for row in rows}
        self.assertEqual(
            scenarios,
            {
                "paper_text_iid_delta_equal_tail",
                "shared_delta_equal_tail_sensitivity",
            },
        )
        self.assertEqual(
            rules,
            {"fixed_horizon_mc_calibrated", "anytime_bf_ge_1_over_alpha"},
        )
        self.assertGreater(metadata["runtime_seconds"], 0.0)

    def test_spike_point_mass_matches_direct_ntp_sum(self) -> None:
        # The spike log density must equal log sum_w r**(1/p_w - 1) over the
        # full spike NTP vector, which is the definition it is derived from.
        rng = np.random.default_rng(20240401)
        grid = np.concatenate(
            [
                np.linspace(1e-9, 1.0 - 1e-12, 2_000),
                rng.uniform(size=500),
            ]
        )
        for vocabulary_size, delta in (
            (1000, 0.01),
            (1000, 0.05),
            (1000, 0.499),
            (50, 0.2),
            (3, 0.4),
        ):
            probabilities = np.concatenate(
                (
                    [1.0 - delta],
                    np.full(vocabulary_size - 1, delta / (vocabulary_size - 1)),
                )
            )
            direct = np.log(
                np.sum(
                    grid[:, None] ** (1.0 / probabilities[None, :] - 1.0), axis=1
                )
            )
            np.testing.assert_allclose(
                benchmark.spike_point_mass_score(grid, delta, vocabulary_size),
                direct,
                rtol=0.0,
                atol=1e-13,
            )

    def test_spike_point_mass_density_integrates_to_one(self) -> None:
        # exp(score) is a density on (0,1) because the Gumbel pivot null is
        # Unif(0,1), so f0 = 1 and the score is the log likelihood ratio.
        grid = (np.arange(2_000_000) + 0.5) / 2_000_000
        for delta in (0.01, 0.05):
            integral = float(
                np.exp(
                    benchmark.spike_point_mass_score(
                        grid, delta, self.config.vocabulary_size
                    )
                ).mean()
            )
            self.assertAlmostEqual(integral, 1.0, delta=1e-5)

    def test_spike_methods_are_wired_into_the_gumbel_dispatch(self) -> None:
        lookup = benchmark.GumbelBayesLookup(self.config)
        rng = np.random.default_rng(7)
        pivots = rng.uniform(size=(4, 6))
        for method, delta in (("h_spike_0.01", 0.01), ("h_spike_0.05", 0.05)):
            self.assertIn(method, benchmark.GUMBEL_METHODS)
            self.assertIn(method, benchmark.DIAGNOSTIC_METHODS)
            np.testing.assert_allclose(
                benchmark.gumbel_score(pivots, method, lookup),
                benchmark.spike_point_mass_score(
                    pivots, delta, self.config.vocabulary_size
                ),
                rtol=0.0,
                atol=0.0,
            )

    def test_spike_score_rejects_invalid_parameters(self) -> None:
        grid = np.linspace(0.1, 0.9, 5)
        with self.assertRaises(ValueError):
            benchmark.spike_point_mass_score(grid, 0.0, 1000)
        with self.assertRaises(ValueError):
            benchmark.spike_point_mass_score(grid, 1.0, 1000)
        with self.assertRaises(ValueError):
            benchmark.spike_point_mass_score(grid, 0.1, 1)
        with self.assertRaises(ValueError):
            benchmark.spike_point_mass_score(grid, 0.75, 3)

    def test_config_enforces_equal_tail_spike_delta_constraint(self) -> None:
        benchmark.BenchmarkConfig(
            vocabulary_size=2, delta_low=0.1, delta_high=0.5
        ).validate()
        with self.assertRaisesRegex(ValueError, "equal-tail spike"):
            benchmark.BenchmarkConfig(
                vocabulary_size=2, delta_low=0.1, delta_high=0.5001
            ).validate()
        with self.assertRaisesRegex(ValueError, "Dirichlet tail layer"):
            benchmark.BenchmarkConfig(
                vocabulary_size=1000, delta_low=0.1, delta_high=0.6
            ).validate()

    def test_indicator_sink_is_consistent_and_inert(self) -> None:
        config = benchmark.BenchmarkConfig(
            max_horizon=100,
            n_calibration=200,
            n_evaluation_null=120,
            n_evaluation_alternative=120,
            batch_size=60,
            bayes_quadrature_nodes=32,
            gumbel_lookup_size=8_001,
        )
        rows_without, _ = benchmark.run_benchmark(config)
        sink: dict[str, np.ndarray] = {}
        rows_with, metadata = benchmark.run_benchmark(config, indicator_sink=sink)
        # Supplying the sink must not perturb any reported number.
        self.assertEqual(rows_without, rows_with)

        self.assertTrue(sink)
        indicator_metadata = metadata["per_document_indicators"]
        self.assertEqual(
            indicator_metadata["indicator_rule_version"],
            benchmark.BOUNDARY_RANDOMIZATION_RULE_VERSION,
        )
        self.assertEqual(
            indicator_metadata["boundary_randomization"]["base_seed"], config.seed
        )
        records = indicator_metadata["tie_and_randomized_rejection_counts"]
        self.assertEqual(len(records), len(sink))
        by_key = {
            benchmark.indicator_key(
                str(record["scenario"]),
                str(record["scheme"]),
                int(record["horizon"]),
                str(record["method"]),
            ): record
            for record in records
        }
        self.assertEqual(set(by_key), set(sink))
        for key, indicator in sink.items():
            record = by_key[key]
            self.assertEqual(indicator.dtype, np.bool_)
            self.assertEqual(indicator.size, config.n_evaluation_alternative)
            self.assertEqual(
                int(np.count_nonzero(indicator)), record["n_rejected_randomized"]
            )
            self.assertAlmostEqual(
                1.0 - float(indicator.mean()),
                float(record["type2_error_randomized"]),
                places=12,
            )
            self.assertEqual(
                record["n_rejected_randomized"],
                record["n_rejected_strict"] + record["n_rejected_at_atom"],
            )
        # Indicators are only captured at the selected horizons.
        self.assertEqual(
            {int(key.split("|")[2]) for key in sink},
            {h for h in benchmark.INDICATOR_HORIZONS if h <= config.max_horizon},
        )

    def test_randomized_indicator_matches_reported_rate_without_ties(self) -> None:
        # For continuous scores no evaluation document lands on the atom, so
        # the deterministic score > c rule reproduces the randomized rate.
        config = benchmark.BenchmarkConfig(
            max_horizon=100,
            n_calibration=200,
            n_evaluation_null=120,
            n_evaluation_alternative=120,
            batch_size=60,
            bayes_quadrature_nodes=32,
            gumbel_lookup_size=8_001,
        )
        rows, metadata = benchmark.run_benchmark(config)
        reported = {
            (
                row["scenario"],
                row["scheme"],
                row["method"],
                int(row["horizon"]),
            ): float(row["type2_error"])
            for row in rows
            if row["decision_rule"] == "fixed_horizon_mc_calibrated"
        }
        checked = 0
        for record in metadata["per_document_indicators"][
            "tie_and_randomized_rejection_counts"
        ]:
            if record["n_at_atom"]:
                continue
            key = (
                record["scenario"],
                record["scheme"],
                record["method"],
                int(record["horizon"]),
            )
            self.assertAlmostEqual(
                reported[key], float(record["type2_error_randomized"]), places=12
            )
            checked += 1
        self.assertGreater(checked, 0)

    def test_shared_paths_match_reference_detector(self) -> None:
        nodes, weights = np.polynomial.legendre.leggauss(
            self.config.bayes_quadrature_nodes
        )
        deltas = self.config.delta_low + 0.5 * (nodes + 1.0) * (
            self.config.delta_high - self.config.delta_low
        )
        prior_weights = 0.5 * weights

        for scheme, simulator in (
            ("gumbel", benchmark.simulate_gumbel_null),
            ("inverse", benchmark.simulate_inverse_null),
        ):
            rng_for_path = np.random.default_rng(8128)
            rng_for_pivots = benchmark.clone_rng(rng_for_path)
            path, _ = benchmark.collect_shared_bayes_paths(
                n_rows=1,
                rng=rng_for_path,
                simulator=simulator,
                scheme=scheme,
                config=self.config,
            )
            pivots = simulator(
                rng_for_pivots, 1, self.config.max_horizon, self.config
            )[0]
            detector = SequentialBayesDetector(
                scheme,
                deltas,
                prior_weights,
                structure="shared",
                gumbel_family="spike",
                vocabulary_size=self.config.vocabulary_size,
                inverse_null_vocabulary_size=(
                    self.config.vocabulary_size if scheme == "inverse" else None
                ),
            )
            detector.update_many(pivots)
            np.testing.assert_allclose(
                path[0], detector.log_bayes_factor_history, rtol=1e-11, atol=1e-11
            )


class PairedComparisonTests(unittest.TestCase):
    def test_normal_quantile_inverts_the_normal_cdf(self) -> None:
        self.assertAlmostEqual(paired_comparisons.normal_quantile(0.5), 0.0, places=12)
        self.assertAlmostEqual(
            paired_comparisons.normal_quantile(0.975), 1.959963984540054, places=10
        )
        self.assertAlmostEqual(
            paired_comparisons.normal_quantile(0.995), 2.5758293035489004, places=10
        )
        for probability in (0.001, 0.25, 0.6, 0.999):
            quantile = paired_comparisons.normal_quantile(probability)
            recovered = 0.5 * (1.0 + math.erf(quantile / math.sqrt(2.0)))
            self.assertAlmostEqual(recovered, probability, places=12)

    def test_wilson_interval_matches_closed_form_and_edges(self) -> None:
        # Closed form check against an independently written expression.
        z = 1.959963984540054
        for successes, trials in ((7, 5000), (31, 5000), (225, 5000), (13, 40)):
            phat = successes / trials
            denominator = 1.0 + z * z / trials
            center = (phat + z * z / (2 * trials)) / denominator
            half = (
                z
                * math.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials))
                / denominator
            )
            low, high = paired_comparisons.wilson_interval(successes, trials)
            self.assertAlmostEqual(low, center - half, places=12)
            self.assertAlmostEqual(high, center + half, places=12)
            self.assertLessEqual(low, phat)
            self.assertLessEqual(phat, high)
        # Degenerate counts stay inside [0, 1].
        self.assertEqual(paired_comparisons.wilson_interval(0, 100)[0], 0.0)
        self.assertEqual(paired_comparisons.wilson_interval(100, 100)[1], 1.0)
        with self.assertRaises(ValueError):
            paired_comparisons.wilson_interval(5, 0)
        with self.assertRaises(ValueError):
            paired_comparisons.wilson_interval(11, 10)

    def test_binomial_cdf_half_is_exact(self) -> None:
        for trials in (0, 1, 5, 12, 33):
            total = 0.0
            for quantile in range(trials + 1):
                expected = (
                    sum(math.comb(trials, k) for k in range(quantile + 1)) / 2**trials
                )
                self.assertAlmostEqual(
                    paired_comparisons.binomial_cdf_half(quantile, trials),
                    expected,
                    places=15,
                )
                total = expected
            self.assertAlmostEqual(total, 1.0, places=15)
        self.assertEqual(paired_comparisons.binomial_cdf_half(-1, 10), 0.0)
        self.assertEqual(paired_comparisons.binomial_cdf_half(10, 10), 1.0)

    def test_exact_mcnemar_returns_one_without_discordant_pairs(self) -> None:
        # The conditional reference distribution at b+c=0 is the point mass at
        # zero.  Its exact tail probability is one, while status metadata records
        # the perfect concordance separately.
        self.assertEqual(paired_comparisons.exact_mcnemar_p_value(0, 0), 1.0)
        # b = c > 0 is a genuinely defined test that happens to give one.
        self.assertAlmostEqual(
            paired_comparisons.exact_mcnemar_p_value(5, 5), 1.0, places=15
        )
        self.assertAlmostEqual(
            paired_comparisons.exact_mcnemar_p_value(1, 1), 1.0, places=15
        )

    def test_exact_mcnemar_reproduces_known_values(self) -> None:
        # Hand-computable cases.
        self.assertAlmostEqual(
            paired_comparisons.exact_mcnemar_p_value(0, 1), 1.0, places=15
        )
        self.assertAlmostEqual(
            paired_comparisons.exact_mcnemar_p_value(0, 3), 0.25, places=15
        )
        self.assertAlmostEqual(
            paired_comparisons.exact_mcnemar_p_value(0, 24), 2.0 / 2**24, places=18
        )
        self.assertAlmostEqual(
            paired_comparisons.exact_mcnemar_p_value(5, 5), 1.0, places=15
        )
        # Symmetric in its arguments, never above one.
        for b, c in ((8, 18), (3, 51), (1, 2), (40, 41)):
            forward = paired_comparisons.exact_mcnemar_p_value(b, c)
            self.assertEqual(forward, paired_comparisons.exact_mcnemar_p_value(c, b))
            self.assertLessEqual(forward, 1.0)
            self.assertGreater(forward, 0.0)

    def test_holm_adjustment_is_monotone_and_conservative(self) -> None:
        p_values = [0.01, 0.04, 0.03, 0.5]
        adjusted = paired_comparisons.holm_adjust(p_values)
        self.assertEqual(adjusted, [0.04, 0.09, 0.09, 0.5])
        for raw, cooked in zip(p_values, adjusted):
            self.assertGreaterEqual(cooked, raw)
            self.assertLessEqual(cooked, 1.0)
        ordered = sorted(range(4), key=lambda i: p_values[i])
        previous = 0.0
        for index in ordered:
            self.assertGreaterEqual(adjusted[index] + 1e-15, previous)
            previous = adjusted[index]
        self.assertEqual(paired_comparisons.holm_adjust([]), [])
        self.assertEqual(paired_comparisons.holm_adjust([0.2]), [0.2])

    def test_holm_retains_perfect_concordance_in_prespecified_family(self) -> None:
        adjusted = paired_comparisons.holm_adjust([0.01, 1.0, 0.04, 1.0])
        self.assertEqual(adjusted, [0.04, 1.0, 0.12, 1.0])
        with self.assertRaises(ValueError):
            paired_comparisons.holm_adjust([0.01, math.nan])

    def test_build_comparisons_on_a_hand_built_cell(self) -> None:
        # Reference misses documents {0}; comparator misses {0, 1, 2}.  So the
        # reference-only-miss count b is 0 and the comparator-only count c is 2.
        n = 8
        reference = np.ones(n, dtype=bool)
        reference[0] = False
        comparator = np.ones(n, dtype=bool)
        comparator[[0, 1, 2]] = False
        indicators = {
            benchmark.indicator_key(
                "shared_delta_equal_tail_sensitivity", "gumbel", 700, "bayes_shared"
            ): reference,
            benchmark.indicator_key(
                "shared_delta_equal_tail_sensitivity",
                "gumbel",
                700,
                "h_gum_star_0.005",
            ): comparator,
        }
        payload = paired_comparisons.build_comparisons(indicators)
        self.assertEqual(len(payload["comparisons"]), 1)
        record = payload["comparisons"][0]
        self.assertEqual(record["reference"], "bayes_shared")
        self.assertEqual(record["b_reference_only_miss"], 0)
        self.assertEqual(record["c_comparator_only_miss"], 2)
        self.assertEqual(record["discordant_total"], 2)
        self.assertAlmostEqual(record["p_value"], 0.5, places=15)
        self.assertAlmostEqual(record["reference_type2_error"], 1 / 8)
        self.assertAlmostEqual(record["comparator_type2_error"], 3 / 8)
        self.assertTrue(record["comparator_is_best_paper_score"])
        self.assertEqual(payload["best_comparator_only_holm_family"]["size"], 1)
        self.assertEqual(len(payload["type2_wilson_intervals"]), 2)

    def test_zero_discordance_is_retained_in_holm_family(self) -> None:
        decisions = np.array([True, False, True, True, False], dtype=bool)
        indicators = {
            benchmark.indicator_key(
                "shared_delta_equal_tail_sensitivity",
                "gumbel",
                700,
                method,
            ): decisions.copy()
            for method in ("bayes_shared", "h_gum_star_0.005")
        }
        payload = paired_comparisons.build_comparisons(indicators)
        record = payload["comparisons"][0]
        self.assertEqual(record["discordant_total"], 0)
        self.assertEqual(record["p_value"], 1.0)
        self.assertEqual(
            record["p_value_status"], "no_discordant_pairs_p_equals_one"
        )
        self.assertEqual(record["holm_adjusted_p"], 1.0)
        self.assertEqual(record["holm_family_size"], 1)
        self.assertEqual(record["holm_family_zero_discordance_retained"], 1)

    def test_build_comparisons_picks_the_scenario_specific_reference(self) -> None:
        n = 6
        rng = np.random.default_rng(4)
        indicators = {}
        for scenario in benchmark_scenarios():
            for method in ("bayes_tokenwise", "bayes_shared", "h_gum_star_0.005"):
                if (
                    method == "bayes_shared"
                    and scenario == "paper_text_iid_delta_equal_tail"
                ):
                    continue
                indicators[
                    benchmark.indicator_key(scenario, "gumbel", 300, method)
                ] = rng.uniform(size=n) > 0.3
        payload = paired_comparisons.build_comparisons(indicators)
        references = {
            (record["scenario"], record["reference"])
            for record in payload["comparisons"]
        }
        self.assertEqual(
            references,
            {
                ("paper_text_iid_delta_equal_tail", "bayes_tokenwise"),
                ("shared_delta_equal_tail_sensitivity", "bayes_shared"),
            },
        )
        # A Bayes rule is never a comparator in the reference-score or diagnostic
        # families, so it can never be selected as the best reference score.  The one
        # place a Bayes rule may appear as a comparator is the dirichlet_layer
        # family, which exists precisely to compare a layer rule against the
        # spike-family rule it generalises.
        for record in payload["comparisons"]:
            if str(record["holm_family"]).endswith("|dirichlet_layer"):
                self.assertIn(
                    record["comparator"], paired_comparisons.BAYES_METHODS
                )
                self.assertFalse(record["comparator_is_paper_score"])
                self.assertFalse(record["comparator_is_best_paper_score"])
            else:
                self.assertNotIn(
                    record["comparator"], paired_comparisons.BAYES_METHODS
                )

    def test_layer_family_pairs_each_layer_rule_with_its_spike_rule(self) -> None:
        n = 40
        rng = np.random.default_rng(11)
        indicators = {}
        pairs = paired_comparisons.LAYER_PAIRS
        scenario = "shared_delta_equal_tail_sensitivity"
        for method in (
            "bayes_shared",
            "bayes_tokenwise",
            *pairs,
            "h_gum_star_0.005",
        ):
            indicators[
                benchmark.indicator_key(scenario, "gumbel", 300, method)
            ] = rng.uniform(size=n) > 0.25
        payload = paired_comparisons.build_comparisons(indicators)
        layer_records = [
            record
            for record in payload["comparisons"]
            if str(record["holm_family"]).endswith("|dirichlet_layer")
        ]
        self.assertEqual(
            {(r["reference"], r["comparator"]) for r in layer_records},
            set(pairs.items()),
        )
        for record in layer_records:
            self.assertEqual(record["n_documents"], n)

    def test_layer_family_is_absent_when_no_layer_rule_was_run(self) -> None:
        # The inverse scheme never runs a layer rule, so no layer family may
        # appear for it and no phantom comparison may be emitted.
        n = 20
        rng = np.random.default_rng(12)
        indicators = {}
        for method in ("bayes_shared", "h_dif_star_0.01"):
            indicators[
                benchmark.indicator_key(
                    "shared_delta_equal_tail_sensitivity", "inverse", 300, method
                )
            ] = rng.uniform(size=n) > 0.25
        payload = paired_comparisons.build_comparisons(indicators)
        self.assertFalse(
            any(
                str(record["holm_family"]).endswith("|dirichlet_layer")
                for record in payload["comparisons"]
            )
        )

    def test_diagnostic_scores_are_excluded_from_best_paper_score(self) -> None:
        n = 10
        reference = np.ones(n, dtype=bool)
        reference[[0, 1]] = False
        paper = np.ones(n, dtype=bool)
        paper[[0, 1, 2, 3]] = False
        spike = np.ones(n, dtype=bool)  # strictly better than the reference score
        indicators = {
            benchmark.indicator_key(
                "shared_delta_equal_tail_sensitivity", "gumbel", 700, name
            ): value
            for name, value in (
                ("bayes_shared", reference),
                ("h_gum_star_0.005", paper),
                ("h_spike_0.01", spike),
            )
        }
        payload = paired_comparisons.build_comparisons(indicators)
        best = [
            record
            for record in payload["comparisons"]
            if record["comparator_is_best_paper_score"]
        ]
        self.assertEqual([record["comparator"] for record in best], ["h_gum_star_0.005"])
        spike_record = next(
            record
            for record in payload["comparisons"]
            if record["comparator"] == "h_spike_0.01"
        )
        self.assertFalse(spike_record["comparator_is_paper_score"])
        self.assertIn("diagnostic_scores", spike_record["holm_family"])


class CommandLineSafetyTests(unittest.TestCase):
    def test_default_full_run_uses_paper_results_directory(self) -> None:
        args = benchmark.parse_args([])
        self.assertEqual(
            benchmark.resolve_output_dir(args.output_dir, quick=args.quick),
            benchmark.DEFAULT_RESULTS_DIR,
        )

    def test_default_quick_run_uses_isolated_directory(self) -> None:
        args = benchmark.parse_args(["--quick"])
        self.assertEqual(
            benchmark.resolve_output_dir(args.output_dir, quick=args.quick),
            benchmark.DEFAULT_QUICK_RESULTS_DIR,
        )
        self.assertNotEqual(
            benchmark.DEFAULT_QUICK_RESULTS_DIR, benchmark.DEFAULT_RESULTS_DIR
        )

    def test_explicit_directory_wins_for_quick_run(self) -> None:
        args = benchmark.parse_args(["--quick", "--output-dir", "chosen"])
        self.assertEqual(
            benchmark.resolve_output_dir(args.output_dir, quick=args.quick),
            Path("chosen"),
        )


class CleanQuadratureCheckTests(unittest.TestCase):
    def test_node_count_does_not_perturb_the_simulated_pivots(self) -> None:
        # The paired design depends on quadrature nodes feeding only the Bayes
        # grids, never the random number streams.
        config = benchmark.BenchmarkConfig(
            max_horizon=8, bayes_quadrature_nodes=48, gumbel_lookup_size=2_001
        )
        other = replace(config, bayes_quadrature_nodes=192)
        for simulator in (
            benchmark.simulate_gumbel_alternative_shared_delta,
            benchmark.simulate_inverse_alternative_shared_delta,
        ):
            first = simulator(np.random.default_rng(11), 5, 8, config)
            second = simulator(np.random.default_rng(11), 5, 8, other)
            np.testing.assert_array_equal(first, second)

    def test_collect_type2_returns_only_requested_cells(self) -> None:
        config = benchmark.BenchmarkConfig(
            max_horizon=100,
            n_calibration=120,
            n_evaluation_null=60,
            n_evaluation_alternative=60,
            batch_size=60,
            bayes_quadrature_nodes=16,
            gumbel_lookup_size=2_001,
        )
        table = check_clean_quadrature.collect_type2(config)
        self.assertTrue(table)
        for scenario, scheme, method, rule, horizon in table:
            self.assertIn(method, check_clean_quadrature.METHODS)
            self.assertIn(rule, check_clean_quadrature.DECISION_RULES)
            self.assertEqual(horizon, 100)
            self.assertIn(scheme, ("gumbel", "inverse"))
            self.assertIn(scenario, benchmark_scenarios())
        # bayes_shared exists only in the shared-Delta scenario.
        shared_scenarios = {
            key[0] for key in table if key[2] == "bayes_shared"
        }
        self.assertEqual(shared_scenarios, {"shared_delta_equal_tail_sensitivity"})
        for value in table.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


def benchmark_scenarios() -> tuple[str, ...]:
    return (
        "paper_text_iid_delta_equal_tail",
        "shared_delta_equal_tail_sensitivity",
    )


class DirichletCompetitorTests(unittest.TestCase):
    """The Dirichlet-layer competitor must be paired, contained and Gumbel-only."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = benchmark.BenchmarkConfig(
            vocabulary_size=200,
            max_horizon=30,
            n_calibration=300,
            n_evaluation_null=200,
            n_evaluation_alternative=200,
            batch_size=100,
            bayes_quadrature_nodes=8,
            gumbel_lookup_size=2_001,
            dirichlet_c_nodes=2_001,
        )

    def test_layer_at_alpha_infinity_reproduces_the_spike_rules_exactly(self) -> None:
        # This is the load-bearing invariant.  With the alpha prior collapsed to
        # the equal-tail limit the layer IS the spike family, so every reported
        # number must agree bit for bit.  Agreement proves two things at once:
        # the layer contains the published detector, and the extra collector is
        # driven from a clone that sees identical pivots, so the comparison is
        # paired rather than merely similar.
        rows, _ = benchmark.run_benchmark(
            self.config, dirichlet_alpha_grid=(math.inf,)
        )
        pairs = {
            "bayes_shared": benchmark.DIRICHLET_SHARED_METHOD,
            "bayes_tokenwise": benchmark.DIRICHLET_TOKENWISE_METHOD,
        }
        indexed: dict[tuple[str, str, str, str, int], float] = {}
        for row in rows:
            if row["scheme"] != "gumbel":
                continue
            indexed[
                (
                    str(row["scenario"]),
                    str(row["method"]),
                    str(row["decision_rule"]),
                    "type2",
                    int(row["horizon"]),
                )
            ] = float(row["type2_error"])
        compared = 0
        for scenario in (
            "paper_text_iid_delta_equal_tail",
            "shared_delta_equal_tail_sensitivity",
        ):
            for rule in ("fixed_horizon_mc_calibrated", "anytime_bf_ge_1_over_alpha"):
                for spike, layer in pairs.items():
                    for horizon in range(1, self.config.max_horizon + 1):
                        left = indexed.get((scenario, spike, rule, "type2", horizon))
                        right = indexed.get((scenario, layer, rule, "type2", horizon))
                        if left is None or right is None:
                            continue
                        self.assertEqual(
                            left,
                            right,
                            msg=f"{scenario} {rule} {spike} vs {layer} n={horizon}",
                        )
                        compared += 1
        self.assertGreater(compared, 100)

    def test_layer_rules_are_gumbel_only(self) -> None:
        rows, _ = benchmark.run_benchmark(
            self.config, dirichlet_alpha_grid=(1.0, math.inf)
        )
        inverse_methods = {
            str(row["method"]) for row in rows if row["scheme"] == "inverse"
        }
        for method in (
            benchmark.DIRICHLET_SHARED_METHOD,
            benchmark.DIRICHLET_TOKENWISE_METHOD,
        ):
            self.assertNotIn(method, inverse_methods)
        gumbel_methods = {
            str(row["method"]) for row in rows if row["scheme"] == "gumbel"
        }
        self.assertIn(benchmark.DIRICHLET_SHARED_METHOD, gumbel_methods)
        self.assertIn(benchmark.DIRICHLET_TOKENWISE_METHOD, gumbel_methods)

    def test_shared_layer_appears_only_in_the_shared_scenario(self) -> None:
        # The shared hierarchy is only meaningful where one Delta persists, which
        # is how the existing bayes_shared rule is scoped; the layer must follow it.
        rows, _ = benchmark.run_benchmark(
            self.config, dirichlet_alpha_grid=(1.0, math.inf)
        )
        scenarios = {
            str(row["scenario"])
            for row in rows
            if row["method"] == benchmark.DIRICHLET_SHARED_METHOD
        }
        self.assertEqual(scenarios, {"shared_delta_equal_tail_sensitivity"})

    def test_anytime_maxima_are_tracked_for_every_bayes_method(self) -> None:
        self.assertEqual(
            set(benchmark.ANYTIME_METHODS),
            {"bayes_tokenwise", benchmark.DIRICHLET_TOKENWISE_METHOD},
        )
        rng = np.random.default_rng(4242)
        lookup = benchmark.GumbelBayesLookup(self.config)
        grid = benchmark.build_dirichlet_grid(self.config, (1.0, math.inf))
        dirichlet_lookup = grid.tokenwise_lookup(size=2_001)
        uniontail_lookup = benchmark.build_uniontail_grid(
            self.config
        ).tokenwise_lookup(size=2_001)
        _, maxima = benchmark.collect_paths(
            n_rows=20,
            rng=rng,
            simulator=benchmark.simulate_gumbel_null,
            methods=benchmark.GUMBEL_METHODS,
            scorer=lambda p, m: benchmark.gumbel_score(
                p, m, lookup, dirichlet_lookup=dirichlet_lookup,
                uniontail_lookup=uniontail_lookup,
            ),
            config=self.config,
        )
        self.assertEqual(set(maxima), set(benchmark.ANYTIME_METHODS))
        for values in maxima.values():
            self.assertTrue(np.all(np.diff(values, axis=1) >= -1e-12))

    def test_tokenwise_layer_score_requires_its_lookup(self) -> None:
        lookup = benchmark.GumbelBayesLookup(self.config)
        with self.assertRaises(ValueError):
            benchmark.gumbel_score(
                np.array([0.5]), benchmark.DIRICHLET_TOKENWISE_METHOD, lookup
            )

    def test_config_rejects_a_degenerate_transform_table(self) -> None:
        with self.assertRaises(ValueError):
            benchmark.BenchmarkConfig(dirichlet_c_nodes=4).validate()

    def test_metadata_records_containment_and_normalisation(self) -> None:
        _, metadata = benchmark.run_benchmark(
            self.config, dirichlet_alpha_grid=(1.0, math.inf)
        )
        self.assertEqual(
            metadata["official_code_commit"], benchmark.OFFICIAL_CODE_COMMIT
        )
        self.assertTrue(
            metadata["official_code"].endswith(benchmark.OFFICIAL_CODE_COMMIT)
        )
        self.assertEqual(
            metadata["official_result_arrays"],
            list(benchmark.OFFICIAL_RESULT_ARRAYS),
        )
        self.assertIn("Commit-pinned", metadata["official_code_note"])
        block = metadata["bayes_dirichlet"]
        self.assertEqual(block["scheme"], "gumbel only")
        self.assertIn("inf", block["alpha_prior"]["grid"])
        # The configured table has a small downward numerical mass error, but
        # metadata must not promote this finite diagnostic into a certificate.
        self.assertTrue(block["normalisation"]["diagnostic_only"])
        self.assertFalse(
            block["normalisation"]["certified_one_sided_bound"]
        )
        self.assertLessEqual(
            block["normalisation"]["tokenwise_numerator_mass_minus_one"], 0.0
        )
        self.assertIn("not a certificate", block["normalisation"]["note"])
        outer = block["outer_tokenwise_lookup_validation"]
        self.assertTrue(outer["diagnostic_only"])
        self.assertFalse(outer["certified_one_sided_bound"])
        self.assertEqual(
            outer["midpoint_log_ratio"]["intervals_checked"],
            self.config.gumbel_lookup_size - 1,
        )
        self.assertIn(
            "isolates outer interpolation error",
            outer["midpoint_log_ratio"]["reference"],
        )
        self.assertAlmostEqual(
            outer["interpolated_numerator_mass"]["mass_minus_one"],
            block["normalisation"]["tokenwise_numerator_mass_minus_one"],
        )
        for deviation in block["normalisation"][
            "analytic_component_mass_max_abs_deviation_by_alpha"
        ].values():
            self.assertLess(deviation, 1e-12)
        lookup = metadata["bayes_gumbel_lookup_validation"]
        self.assertTrue(lookup["diagnostic_only"])
        self.assertFalse(lookup["certified_one_sided_bound"])
        self.assertLess(
            lookup["midpoint_log_density"]["max_abs_error"], 1e-3
        )
        self.assertLess(
            abs(lookup["interpolated_density_mass"]["mass_minus_one"]), 2e-5
        )
        # The metadata smoke run deliberately uses only 8,001 lookup nodes.  The
        # manuscript uses the 80,001-node default, whose validation is cheap to
        # check directly and should resolve the reported sub-micro-log-unit error.
        default_lookup = benchmark.GumbelBayesLookup(
            benchmark.BenchmarkConfig()
        ).validation_report()
        self.assertEqual(
            default_lookup["midpoint_log_density"]["intervals_checked"], 80_000
        )
        self.assertLess(
            default_lookup["midpoint_log_density"]["max_abs_error"], 1e-6
        )
        self.assertLess(
            abs(default_lookup["interpolated_density_mass"]["mass_minus_one"]),
            2e-8,
        )



class UnionTailRuleTests(unittest.TestCase):
    """The enlarged shared Gumbel rule: union of the two tail families."""

    def test_ladder_stays_below_the_full_width(self) -> None:
        # The full-width atom lives in the Dirichlet block, so the tail-width
        # ladder must stop short of it or it would be counted twice.
        config = benchmark.BenchmarkConfig(vocabulary_size=1000)
        grid = benchmark.build_uniontail_grid(config)
        self.assertEqual(grid.tail_widths[0], 1)
        self.assertTrue(all(j < config.vocabulary_size - 1 for j in grid.tail_widths))
        self.assertEqual(grid.dirichlet_block.tail_size, config.vocabulary_size - 1)

    def test_widest_atom_reproduces_the_closed_form_spike(self) -> None:
        config = benchmark.BenchmarkConfig(vocabulary_size=1000)
        grid = benchmark.build_uniontail_grid(config)
        rng = np.random.default_rng(3)
        # The equal-tail atom sits in the interpolated Dirichlet block, so
        # agreement is table-limited rather than exact.
        self.assertLess(grid.spike_agreement(rng.uniform(size=4_000)), 1e-6)

    def test_rule_is_gumbel_and_shared_only(self) -> None:
        # It must not appear in the token-sum menus or the anytime tokenwise set.
        self.assertNotIn(benchmark.UNIONTAIL_SHARED_METHOD, benchmark.GUMBEL_METHODS)
        self.assertNotIn(benchmark.UNIONTAIL_SHARED_METHOD, benchmark.INVERSE_METHODS)
        self.assertNotIn(benchmark.UNIONTAIL_SHARED_METHOD, benchmark.ANYTIME_METHODS)

    def test_union_contains_both_parents(self) -> None:
        # "Contains both", operationally: w=1 IS the Dirichlet layer and w=0 IS
        # the tail-width layer.  Agreement is to floating-point rounding rather
        # than bitwise: the collapsed block enters the log-space mixture with a
        # -inf weight, which costs an occasional ulp.
        import dirichlet_detector as dd

        config = benchmark.BenchmarkConfig(
            vocabulary_size=1000, max_horizon=10, bayes_quadrature_nodes=12
        )
        deltas, weights = dd.gauss_legendre_delta_grid(
            config.delta_low, config.delta_high, config.bayes_quadrature_nodes
        )
        shared = dict(delta_grid=deltas, delta_weights=weights, tail_size=999)
        pivots = np.random.default_rng(5).uniform(size=(8, 10))

        left, _ = dd.UnionTailBayesGrid(
            dirichlet_weight=1.0, alpha_grid=benchmark.DIRICHLET_ALPHA_GRID, **shared
        ).shared_paths(pivots, horizons=(10,))
        right, _ = dd.DirichletBayesGrid(
            alpha_grid=benchmark.DIRICHLET_ALPHA_GRID, **shared
        ).shared_paths(pivots, horizons=(10,))
        np.testing.assert_allclose(left, right, rtol=0.0, atol=1e-12)

        widths = tuple(j for j in dd.dyadic_tail_width_grid(999) if j != 999)
        left, _ = dd.UnionTailBayesGrid(
            dirichlet_weight=0.0, **shared
        ).shared_paths(pivots, horizons=(10,))
        right, _ = dd.TailWidthBayesGrid(
            tail_width_grid=widths, **shared
        ).shared_paths(pivots, horizons=(10,))
        np.testing.assert_allclose(left, right, rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
