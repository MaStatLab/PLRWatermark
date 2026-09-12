"""Tests for the Dirichlet tail-layer sweep (tail_regime_sweep.py)."""

from __future__ import annotations

import dataclasses
import inspect
import json
import math
import unittest
from pathlib import Path

import numpy as np

import benchmark_paper_experiment as paper
import regime_sweep as deficit_sweep
import tail_family as tf
import dirichlet_detector as dd
import tail_regime_sweep as sweep


def tiny_config(**overrides: object) -> sweep.TailSweepConfig:
    """A configuration small enough to run a whole sweep inside a test."""

    defaults: dict[str, object] = {
        "vocabulary_size": 40,
        "horizons": (4, 9),
        "alpha_grid": (0.5, 2.0, math.inf),
        "point_alphas": (0.5, 2.0),
        "n_calibration": 60,
        "n_evaluation_null": 30,
        "n_evaluation_alternative": 30,
        "batch_size": 10,
        "bayes_quadrature_nodes": 8,
        "c_nodes": 2_001,
        "seed": 5_151,
        "max_regret_bootstrap_replicates": 50,
        "max_regret_bootstrap_batch_size": 10,
    }
    defaults.update(overrides)
    return sweep.TailSweepConfig(**defaults)  # type: ignore[arg-type]


def logit_quadrature(panels: int = 400, order: int = 32, limit: float = 40.0):
    """Nodes, weights and Jacobian for integrating a density over r in (0,1)."""

    nodes, weights = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(-limit, limit, panels + 1)
    half = 0.5 * (edges[1] - edges[0])
    centres = 0.5 * (edges[:-1] + edges[1:])
    z = (centres[:, None] + half * nodes[None, :]).reshape(-1)
    quadrature = np.tile(weights * half, centres.size)
    r = 1.0 / (1.0 + np.exp(-z))
    return r, quadrature * r * (1.0 - r)


class TailRegimeTests(unittest.TestCase):
    """The regimes must be the tail laws the sweep claims they are."""

    def test_default_regimes_are_the_documented_set(self) -> None:
        labels = [regime.label for regime in sweep.DEFAULT_TAIL_REGIMES]
        self.assertEqual(
            labels, ["T1", "T2", "T3", "T4", "T5", "T6", "W1", "W2", "W3", "W4"]
        )
        kinds = [regime.kind for regime in sweep.DEFAULT_TAIL_REGIMES]
        self.assertEqual(kinds[:5], ["dirichlet"] * 5)
        self.assertEqual(kinds[5], "normalised_uniform")
        self.assertEqual(kinds[6:10], ["width"] * 4)
        # The width arm is APPENDED, never inserted.  Inserting it ahead of T6
        # renumbered T6's per-regime RNG stream and silently moved its published
        # numbers (n=700 Type II .0008 -> .0018) without touching its law.
        self.assertTrue(math.isinf(sweep.DEFAULT_TAIL_REGIMES[0].alpha))
        # The width arm must span narrow to nearly-full so the block it
        # exercises is seen both paying and not paying.
        self.assertEqual(
            [r.tail_width for r in sweep.DEFAULT_TAIL_REGIMES if r.kind == "width"],
            [1, 4, 16, 64],
        )

    def test_exactly_one_default_regime_is_out_of_family(self) -> None:
        """Only T6 is outside every layer of the prior.

        The width regimes are also outside the *Dirichlet* family, but unlike
        T6 they are members of the union tail's width branch, so they are
        misspecified for the shape block alone rather than for the whole prior.
        T6 remains the one law no layer contains.
        """

        outside = [
            regime
            for regime in sweep.DEFAULT_TAIL_REGIMES
            if not regime.in_dirichlet_family and regime.kind != "width"
        ]
        self.assertEqual(len(outside), 1)
        self.assertEqual(outside[0].label, "T6")
        width = [r for r in sweep.DEFAULT_TAIL_REGIMES if r.kind == "width"]
        self.assertTrue(width)
        self.assertTrue(all(not r.in_dirichlet_family for r in width))

    def test_default_alpha_grid_matches_the_detector(self) -> None:
        """The sweep and the detector must not drift apart."""

        self.assertEqual(
            tuple(sweep.DEFAULT_ALPHA_GRID), tuple(dd.DEFAULT_ALPHA_GRID)
        )

    def test_one_in_family_regime_is_off_the_prior_grid(self) -> None:
        # A generating alpha that is an atom of the prior makes the mixture
        # correctly specified for free.  At least one in-family regime must not
        # be an atom, or the sweep cannot distinguish coverage from coincidence.
        grid = sweep.DEFAULT_ALPHA_GRID
        off_grid = [
            regime.label
            for regime in sweep.DEFAULT_TAIL_REGIMES
            if regime.in_dirichlet_family and not regime.on_prior_grid(grid)
        ]
        # Dropping 3 from the support leaves two in-family regimes off the
        # prior grid: T3 (Dirichlet(3)) and T4 (Dirichlet(0.5)).
        self.assertEqual(off_grid, ["T3", "T4"])

    def test_invalid_regimes_are_rejected(self) -> None:
        for bad in (
            sweep.TailRegime("X", "beta", 1.0, "unknown kind"),
            sweep.TailRegime("X", "dirichlet", 0.0, "alpha must be positive"),
            sweep.TailRegime("X", "dirichlet", -1.0, "alpha must be positive"),
            sweep.TailRegime("X", "dirichlet", math.nan, "alpha must not be nan"),
            sweep.TailRegime("X", "normalised_uniform", 3.0, "must carry nan"),
        ):
            with self.assertRaises(ValueError):
                bad.validate()

    def test_equal_tail_regime_reproduces_the_existing_generator(self) -> None:
        # T1 must be the very generator behind shared_delta_equal_tail_sensitivity,
        # so that this sweep's baseline column is the published configuration.
        vocabulary, rows, horizon = 1000, 64, 12
        config = paper.BenchmarkConfig(
            vocabulary_size=vocabulary, max_horizon=horizon
        )
        left = np.random.default_rng(909)
        column = left.uniform(
            config.delta_low, config.delta_high, size=(rows, 1)
        )
        deltas = np.repeat(column, horizon, axis=1)
        got = sweep.DEFAULT_TAIL_REGIMES[0].simulate(left, deltas, vocabulary)

        right = np.random.default_rng(909)
        expected = paper.simulate_gumbel_alternative_shared_delta(
            right, rows, horizon, config
        )
        np.testing.assert_array_equal(got, expected)

    def test_simulated_pivots_stay_in_the_unit_interval(self) -> None:
        for regime in sweep.DEFAULT_TAIL_REGIMES:
            rng = np.random.default_rng(17)
            deltas = np.repeat(rng.uniform(0.01, 0.5, size=(40, 1)), 6, axis=1)
            pivots = regime.simulate(rng, deltas, 200)
            self.assertEqual(pivots.shape, (40, 6))
            self.assertTrue(np.all((pivots >= 0.0) & (pivots <= 1.0)))

    def test_dirichlet_simulator_matches_the_layer_cdf(self) -> None:
        # The size-biased simulator and the closed-form CDF are two independent
        # derivations of the same law; they must agree.
        tail_size, delta = 60, 0.3
        for alpha in (0.4, 2.0, math.inf):
            pivot = tf.dirichlet_tail_pivot(alpha, tail_size, c_nodes=20_001)
            rng = np.random.default_rng(31)
            draws = pivot.simulate(rng, np.full(200_000, delta))
            grid = np.linspace(0.02, 0.98, 25)
            empirical = np.searchsorted(np.sort(draws), grid, side="right") / draws.size
            theoretical = pivot.cdf(grid, np.full(grid.size, delta))
            # 1.63/sqrt(n) is the 99.9% Kolmogorov-Smirnov band at n=200,000.
            band = 1.63 / math.sqrt(draws.size)
            np.testing.assert_allclose(empirical, theoretical, atol=band)

    def test_delta_is_shared_by_every_token_of_a_document(self) -> None:
        config = tiny_config()
        payload_rng = np.random.default_rng(5)
        column = payload_rng.uniform(0.05, 0.4, size=(12, 1))
        deltas = np.repeat(column, 8, axis=1)
        for row in range(deltas.shape[0]):
            self.assertEqual(len(set(deltas[row].tolist())), 1)


class TailDiagnosticTests(unittest.TestCase):
    """The recorded tail moments must support the claims made about them."""

    def test_normalised_uniform_tail_is_bounded_but_dirichlet_is_not(self) -> None:
        # The manuscript's reason the released-code tail is out of family: its
        # scaled marginal is bounded above by 2, the Dirichlet's is not.
        config = tiny_config(vocabulary_size=1000)
        rng = np.random.default_rng(77)
        released = sweep.tail_diagnostics(
            rng, sweep.DEFAULT_TAIL_REGIMES[5], config, 100_000
        )
        dirichlet = sweep.tail_diagnostics(
            rng, sweep.TailRegime("Z", "dirichlet", 3.0, "matched"), config, 100_000
        )
        # Kq -> Uniform(0,2) only in the limit: the normalising sum fluctuates
        # at O(K^-1/2), so a small overshoot past 2 is expected at K=999.  What
        # separates the two laws is the weight of the upper tail, not a hard cap.
        self.assertLess(released["scaled_selected_max"], 2.5)
        self.assertGreater(dirichlet["scaled_selected_max"], 5.0)
        # Matching the variance of the *unselected* marginal q does not match the
        # law the detector actually sees.  Selection size-biases the coordinate,
        # sending Uniform(0,2) to the triangular law on (0,2) with variance 2/9
        # and Gamma(3,3) to Gamma(4,3) with variance 4/9.  The size-biased
        # variances therefore differ by a factor of two even though the
        # unselected ones agree exactly.
        self.assertAlmostEqual(
            released["scaled_selected_mean"], 4.0 / 3.0, places=2
        )
        self.assertAlmostEqual(
            dirichlet["scaled_selected_mean"], 4.0 / 3.0, places=2
        )
        self.assertAlmostEqual(
            released["scaled_selected_variance"], 2.0 / 9.0, places=2
        )
        self.assertAlmostEqual(
            dirichlet["scaled_selected_variance"], 4.0 / 9.0, places=2
        )

    def test_released_tail_is_variance_matched_to_alpha_three(self) -> None:
        # K q -> Uniform(0, 2) has variance 1/3, which is the Dirichlet(alpha)
        # value at alpha = 3.  This is the basis of the "alpha ~ 3" statement.
        config = tiny_config(vocabulary_size=1000)
        rng = np.random.default_rng(78)
        released = sweep.tail_diagnostics(
            rng, sweep.DEFAULT_TAIL_REGIMES[5], config, 200_000
        )
        matched = sweep.tail_diagnostics(
            rng, sweep.TailRegime("Z", "dirichlet", 3.0, "matched"), config, 200_000
        )
        self.assertAlmostEqual(
            released["nominal_scaled_unselected_variance"], 1.0 / 3.0, places=12
        )
        self.assertAlmostEqual(
            matched["nominal_scaled_unselected_variance"], 1.0 / 3.0, places=2
        )

    def test_equal_tail_diagnostics_are_exactly_degenerate(self) -> None:
        config = tiny_config(vocabulary_size=500)
        rng = np.random.default_rng(79)
        report = sweep.tail_diagnostics(
            rng, sweep.DEFAULT_TAIL_REGIMES[0], config, 1_000
        )
        self.assertEqual(report["scaled_selected_variance"], 0.0)
        self.assertAlmostEqual(report["scaled_selected_mean"], 1.0, places=12)


class LayerLikelihoodTests(unittest.TestCase):
    """The layer must be a probability model and must contain the spike family."""

    def test_alpha_infinity_reproduces_the_closed_form_spike_density(self) -> None:
        config = tiny_config(vocabulary_size=1000, c_nodes=200_001)
        layer = sweep.DirichletLayerScores(config)
        rng = np.random.default_rng(101)
        gap = layer.spike_agreement(rng.uniform(size=5_000))
        self.assertLess(gap, 1e-7)

    def test_layer_reproduces_the_existing_shared_bayes_factor(self) -> None:
        # The alpha=inf column of the layer must be the detector already reported
        # in the clean benchmark and the deficit sweep, not merely close to it.
        config = tiny_config(vocabulary_size=1000, c_nodes=200_001, horizons=(3, 11))
        layer = sweep.DirichletLayerScores(config)
        rng = np.random.default_rng(102)
        pivots = rng.uniform(size=(40, config.max_horizon))
        got = sweep.all_rule_paths(pivots, config, layer)[sweep.SPIKE_BAYES_RULE]
        expected = sweep.exact_spike_shared_bayes(pivots, config)
        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_every_discretised_component_integrates_to_one_analytically(self) -> None:
        config = tiny_config(vocabulary_size=1000)
        layer = sweep.DirichletLayerScores(config)
        for alpha, deviation in layer.analytic_normalisation().items():
            self.assertLess(deviation, 1e-12, msg=f"alpha={alpha}")

    def test_interpolated_component_mass_numerical_diagnostic(self) -> None:
        # This regression check records the observed sign and magnitude; finite
        # floating-point quadrature is not a certified one-sided bound.
        config = tiny_config(vocabulary_size=1000, c_nodes=200_001)
        layer = sweep.DirichletLayerScores(config)
        report = layer.normalisation_report(panels=600, gauss_nodes=32)
        self.assertTrue(report["diagnostic_only"])
        self.assertFalse(report["certified_one_sided_bound"])
        self.assertTrue(report["all_masses_at_most_one"])
        self.assertLess(report["max_abs_mass_minus_one"], 1e-6)

    def test_component_densities_are_exact_null_likelihood_ratios(self) -> None:
        # The Gumbel null density is one, so each component density must have
        # unit integral against the null; that is what makes each one-step
        # factor a conditional e-value.
        config = tiny_config(vocabulary_size=1000, c_nodes=200_001)
        layer = sweep.DirichletLayerScores(config)
        r, weight = logit_quadrature(panels=800, order=32)
        density = np.exp(layer.component_log_density(r))
        mass = np.einsum("z,azj->aj", weight, density)
        np.testing.assert_allclose(mass, 1.0, atol=1e-6)

    def test_mixture_is_the_joint_grid_average_of_its_components(self) -> None:
        config = tiny_config(vocabulary_size=200, horizons=(5,))
        layer = sweep.DirichletLayerScores(config)
        rng = np.random.default_rng(103)
        pivots = rng.uniform(size=(9, config.max_horizon))
        paths = sweep.all_rule_paths(pivots, config, layer)

        accumulator = np.zeros(
            (len(layer.alphas), pivots.shape[0], layer.deltas.size)
        )
        for time_index in range(config.max_horizon):
            accumulator += layer.component_log_density(pivots[:, time_index])
        log_weights = (
            np.log(layer.alpha_weights)[:, None] + np.log(layer.delta_weights)[None, :]
        ).reshape(-1)
        flat = accumulator.transpose(1, 0, 2).reshape(pivots.shape[0], -1)
        expected = deficit_sweep._logsumexp_rows(flat + log_weights[None, :])
        np.testing.assert_allclose(paths[sweep.MIXTURE_RULE][:, -1], expected, atol=1e-12)

    def test_shared_mixture_contains_each_weighted_grid_component(
        self,
    ) -> None:
        # Check the nonnegative-sum identity for the joint (Delta, alpha) grid.
        config = tiny_config(vocabulary_size=200, horizons=(7,))
        layer = sweep.DirichletLayerScores(config)
        rng = np.random.default_rng(104)
        pivots = rng.uniform(size=(16, config.max_horizon)) ** 0.4
        paths = sweep.all_rule_paths(pivots, config, layer)
        smallest_weight = float(
            np.min(np.exp(layer._log_joint_weights))
        )
        bound = math.log(1.0 / smallest_weight)
        for name in (
            sweep.SPIKE_BAYES_RULE,
            *(sweep.point_alpha_rule_name(a) for a in config.point_alphas),
        ):
            # Each point-alpha rule is itself a Delta mixture, so it is bounded
            # by the joint mixture up to the alpha weight alone.
            gap = paths[name][:, -1] - paths[sweep.MIXTURE_RULE][:, -1]
            self.assertTrue(np.all(gap <= bound + 1e-9), msg=name)

    def test_pivots_outside_the_unit_interval_are_rejected(self) -> None:
        config = tiny_config()
        layer = sweep.DirichletLayerScores(config)
        for bad in (np.array([-0.1]), np.array([1.5]), np.array([np.nan])):
            with self.assertRaises(ValueError):
                layer.component_log_density(bad)

    def test_exact_zero_pivot_is_handled_without_nan(self) -> None:
        config = tiny_config()
        layer = sweep.DirichletLayerScores(config)
        values = layer.component_log_density(np.array([0.0, 1.0, 0.5]))
        self.assertTrue(np.all(np.isfinite(values)))


class FrozenPriorTests(unittest.TestCase):
    """No rule may be retuned to the regime that generated its data."""

    def test_scoring_entry_point_has_no_regime_parameter(self) -> None:
        parameters = set(inspect.signature(sweep.all_rule_paths).parameters)
        self.assertEqual(parameters, {"pivots", "config", "layer"})

    def test_alpha_prior_ignores_the_regime_list(self) -> None:
        base = tiny_config()
        altered = dataclasses.replace(
            base, regimes=(sweep.DEFAULT_TAIL_REGIMES[5],)
        )
        self.assertEqual(sweep.frozen_alpha_prior(base)[0], sweep.frozen_alpha_prior(altered)[0])
        np.testing.assert_array_equal(
            sweep.frozen_alpha_prior(base)[1], sweep.frozen_alpha_prior(altered)[1]
        )

    def test_alpha_prior_is_uniform_and_contains_the_equal_tail_limit(self) -> None:
        grid, weights = sweep.frozen_alpha_prior(sweep.TailSweepConfig())
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)
        self.assertEqual(len(set(weights.tolist())), 1)
        self.assertTrue(any(math.isinf(a) for a in grid))

    def test_adding_regimes_leaves_the_rules_and_their_results_unchanged(self) -> None:
        base = tiny_config(regimes=sweep.DEFAULT_TAIL_REGIMES[:2])
        extended = dataclasses.replace(
            base, regimes=sweep.DEFAULT_TAIL_REGIMES[:3]
        )
        first = sweep.run_tail_sweep(base)
        second = sweep.run_tail_sweep(extended)
        self.assertEqual(
            [r["name"] for r in first["rules"]], [r["name"] for r in second["rules"]]
        )
        for horizon in base.horizons:
            key = str(horizon)
            for rule in base.rule_names():
                for label in ("T1", "T2"):
                    self.assertEqual(
                        first["type2_error"][key][rule][label],
                        second["type2_error"][key][rule][label],
                        msg=f"{rule} {label} n={horizon}",
                    )

    def test_delta_quadrature_is_the_deficit_sweep_quadrature(self) -> None:
        config = tiny_config()
        layer = sweep.DirichletLayerScores(config)
        deltas, weights = deficit_sweep.frozen_delta_quadrature(
            sweep._deficit_config(config)
        )
        np.testing.assert_array_equal(layer.deltas, deltas)
        np.testing.assert_array_equal(layer.delta_weights, weights)


class ConfigTests(unittest.TestCase):
    def test_seed_is_fresh(self) -> None:
        self.assertNotIn(
            sweep.SWEEP_SEED,
            {24_040_1245, 24_040_1246, 24_040_1247, 24_040_1248, 24_040_1249},
        )

    def test_rule_names_are_the_intended_set(self) -> None:
        names = sweep.TailSweepConfig().rule_names()
        self.assertEqual(
            names,
            (
                "bayes_shared_spike",
                "bayes_shared_alpha_0.1",
                "bayes_shared_alpha_1",
                "bayes_shared_alpha_10",
                "bayes_shared_alpha_100",
                "bayes_shared_alpha_1000",
                "bayes_shared_dirichlet_mixture",
                "bayes_shared_uniontail",
                "h_spike_0.01",
                # Every published tuning and every reference score, as in the
                # deficit sweep: a single tuning cannot show that the constant
                # matters.
                "h_gum_star_0.1",
                "h_gum_star_0.01",
                "h_gum_star_0.005",
                "h_ars",
                "h_log",
                "h_ind_1_over_e",
            ),
        )

    def test_validation_rejects_bad_input(self) -> None:
        for overrides in (
            {"horizons": ()},
            {"horizons": (9, 4)},
            {"horizons": (0, 4)},
            {"alpha_grid": ()},
            {"alpha_grid": (1.0, 1.0)},
            {"alpha_grid": (0.0, 1.0)},
            {"alpha_grid": (1.0, math.nan)},
            {"alpha_grid": (1.0, 2.0), "point_alphas": (3.0,)},
            {"paper_deltas": (0.0,)},
            {"spike_delta": 1.0},
            {"regimes": ()},
            {"c_nodes": 8},
            {"max_regret_bootstrap_replicates": 1},
            {"max_regret_bootstrap_batch_size": 0},
        ):
            with self.subTest(**overrides):
                with self.assertRaises(ValueError):
                    tiny_config(**overrides).validate()

    def test_point_alphas_must_be_atoms_of_the_prior_grid(self) -> None:
        # Every configured point rule must belong to the same comparison menu.
        config = sweep.TailSweepConfig()
        for alpha in config.point_alphas:
            self.assertTrue(
                any(float(alpha) == float(node) for node in config.alpha_grid)
            )


class PayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = tiny_config()
        cls.payload = sweep.run_tail_sweep(cls.config)

    def test_payload_covers_every_rule_regime_horizon_cell(self) -> None:
        expected = {
            (regime, rule, horizon)
            for regime in self.config.regime_labels()
            for rule in self.config.rule_names()
            for horizon in self.config.horizons
        }
        seen = {
            (cell["regime"], cell["rule"], cell["horizon"])
            for cell in self.payload["cells"]
        }
        self.assertEqual(seen, expected)

    def test_payload_regret_is_consistent_with_its_type_two_table(self) -> None:
        for horizon in self.config.horizons:
            key = str(horizon)
            type2 = self.payload["type2_error"][key]
            regret = self.payload["regret"][key]
            for label in self.config.regime_labels():
                eligible = [rule for rule in type2 if deficit_sweep.is_benchmark_rule(rule)]
                best = min(type2[rule][label] for rule in eligible)
                self.assertEqual(
                    self.payload["best_rule_by_regime"][key][label]["rules"],
                    sorted(rule for rule in eligible if type2[rule][label] == best),
                )
                for rule in type2:
                    self.assertAlmostEqual(
                        regret[rule][label], type2[rule][label] - best, places=12
                    )

    def test_payload_max_regret_is_the_row_maximum(self) -> None:
        for horizon in self.config.horizons:
            key = str(horizon)
            for rule, row in self.payload["regret"][key].items():
                self.assertAlmostEqual(
                    self.payload["max_regret"][key][rule]["max_regret"],
                    max(row.values()),
                    places=12,
                )
                self.assertGreaterEqual(
                    self.payload["max_regret"][key][rule]["max_regret_mc_se"],
                    0.0,
                )
                self.assertTrue(
                    math.isfinite(
                        self.payload["max_regret"][key][rule][
                            "max_regret_mc_se"
                        ]
                    )
                )

    def test_pooled_max_regret_dominates_every_horizon(self) -> None:
        pooled = self.payload["pooled_max_regret_across_horizons"]
        for rule, summary in pooled.items():
            for horizon in self.config.horizons:
                self.assertGreaterEqual(
                    summary["max_regret"] + 1e-12,
                    self.payload["max_regret"][str(horizon)][rule]["max_regret"],
                )

    def test_saturated_regimes_are_exactly_the_all_zero_columns(self) -> None:
        for horizon in self.config.horizons:
            key = str(horizon)
            type2 = self.payload["type2_error"][key]
            expected = [
                label
                for label in self.config.regime_labels()
                if all(type2[rule][label] == 0.0 for rule in type2)
            ]
            self.assertEqual(self.payload["saturated_regimes"][key], expected)

    def test_regime_report_flags_family_membership_correctly(self) -> None:
        reports = self.payload["regimes"]
        for regime in self.config.regimes:
            report = reports[regime.label]
            self.assertEqual(
                report["in_dirichlet_family"], regime.in_dirichlet_family
            )
            if not regime.in_dirichlet_family:
                self.assertIsNone(report["alpha"])
                self.assertFalse(report["alpha_is_an_atom_of_the_frozen_prior"])

    def test_payload_is_json_serializable_without_nan(self) -> None:
        text = json.dumps(self.payload, allow_nan=False)
        restored = json.loads(text)
        self.assertEqual(restored["type2_error"], self.payload["type2_error"])
        self.assertEqual(
            restored["config"]["alpha_grid"],
            [sweep._format_alpha(a) for a in self.config.alpha_grid],
        )
        self.assertEqual(
            restored["config"]["point_alphas"],
            [sweep._format_alpha(a) for a in self.config.point_alphas],
        )
        self.assertEqual(restored["config"]["seed"], self.config.seed)

    def test_markdown_tables_have_a_row_for_every_rule(self) -> None:
        for horizon in self.config.horizons:
            tables = self.payload["markdown_tables"][str(horizon)]
            for kind in ("type2", "regret"):
                for rule in self.config.rule_names():
                    self.assertIn(rule, tables[kind])

    def test_calibration_sample_is_shared_and_evaluation_null_is_separate(self) -> None:
        self.assertIn("one exact-null sample", self.payload["calibration"]["sample"])
        self.assertEqual(
            self.payload["calibration"]["n_calibration"], self.config.n_calibration
        )

    def test_validation_block_labels_mass_check_as_uncertified(self) -> None:
        report = self.payload["validation"]["interpolated_component_mass"]
        self.assertTrue(report["diagnostic_only"])
        self.assertFalse(report["certified_one_sided_bound"])
        self.assertTrue(report["all_masses_at_most_one"])

    def test_all_rule_paths_rejects_short_pivot_arrays(self) -> None:
        layer = sweep.DirichletLayerScores(self.config)
        short = np.full((3, self.config.max_horizon - 1), 0.5)
        with self.assertRaises(ValueError):
            sweep.all_rule_paths(short, self.config, layer)


class CommandLineSafetyTests(unittest.TestCase):
    def test_default_full_run_uses_paper_results_directory(self) -> None:
        args = sweep.parse_args([])
        self.assertEqual(
            sweep.resolve_results_dir(args.results_dir, quick=args.quick),
            sweep.DEFAULT_RESULTS_DIR,
        )

    def test_default_quick_run_uses_isolated_directory(self) -> None:
        args = sweep.parse_args(["--quick"])
        self.assertEqual(
            sweep.resolve_results_dir(args.results_dir, quick=args.quick),
            sweep.DEFAULT_QUICK_RESULTS_DIR,
        )
        self.assertNotEqual(sweep.DEFAULT_QUICK_RESULTS_DIR, sweep.DEFAULT_RESULTS_DIR)

    def test_explicit_directory_wins_for_quick_run(self) -> None:
        args = sweep.parse_args(["--quick", "--results-dir", "chosen"])
        self.assertEqual(
            sweep.resolve_results_dir(args.results_dir, quick=args.quick),
            Path("chosen"),
        )



class UnionTailRuleTests(unittest.TestCase):
    def test_rule_is_in_the_menu_once(self) -> None:
        names = sweep.TailSweepConfig().rule_names()
        self.assertEqual(names.count("bayes_shared_uniontail"), 1)

    def test_full_width_atom_lives_in_the_dirichlet_block(self) -> None:
        # The union splits the tail prior: the full-width atom belongs to the
        # Dirichlet block, so the tail-width ladder must stop below it.
        grid = sweep.build_uniontail_grid(sweep.TailSweepConfig())
        self.assertTrue(all(j < grid.tail_size for j in grid.tail_widths))
        self.assertEqual(grid.dirichlet_block.tail_size, grid.tail_size)
        self.assertAlmostEqual(sum(grid.block_weights), 1.0)


class GeneratorIndependenceTests(unittest.TestCase):
    """The generating law must not move when the detector's prior moves.

    The whole point of reporting two priors is that they score the same data.
    A regression here is invisible in every other test -- the sweep still runs,
    the numbers still look plausible -- so it is asserted directly.  This is a
    real defect that once shipped: the alternative sample was drawn from
    ``prior_low``/``prior_high`` instead of the Li et al. bounds, so widening
    the prior silently quadrupled the generated deficit support.
    """

    def _diagnostics(self, payload):
        for value in payload.values():
            if (isinstance(value, dict) and value
                    and isinstance(next(iter(value.values())), dict)
                    and "realized_mean_delta" in next(iter(value.values()))):
                return value
        raise AssertionError("no per-regime delta diagnostics in the payload")

    def test_generated_deltas_do_not_follow_the_prior(self) -> None:
        import dataclasses

        base = tiny_config()
        original = self._diagnostics(sweep.run_tail_sweep(base))
        narrow = self._diagnostics(
            sweep.run_tail_sweep(
                dataclasses.replace(base, prior_low=0.01, prior_high=0.2)
            )
        )
        self.assertEqual(set(original), set(narrow))
        for label in original:
            self.assertAlmostEqual(
                original[label]["realized_mean_delta"],
                narrow[label]["realized_mean_delta"],
                places=12,
                msg=f"regime {label}: the generator moved with the prior",
            )

    def test_generated_deltas_stay_inside_the_li_et_al_bounds(self) -> None:
        low = paper.BenchmarkConfig().delta_low
        high = paper.BenchmarkConfig().delta_high
        diagnostics = self._diagnostics(sweep.run_tail_sweep(tiny_config()))
        for label, entry in diagnostics.items():
            self.assertGreaterEqual(entry["realized_min_delta"], low, msg=label)
            self.assertLessEqual(entry["realized_max_delta"], high, msg=label)



if __name__ == "__main__":
    unittest.main()
