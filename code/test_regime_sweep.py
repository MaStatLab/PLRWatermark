"""Tests for the generating-regime sweep (regime_sweep.py)."""

from __future__ import annotations

import dataclasses
import inspect
import json
import math
import unittest
from pathlib import Path

import numpy as np

import benchmark_paper_experiment as paper
import regime_sweep as sweep
from bayesian_watermark import SequentialBayesDetector


def tiny_config(**overrides: object) -> sweep.RegimeSweepConfig:
    """A configuration small enough to run a whole sweep inside a test."""

    defaults: dict[str, object] = {
        "vocabulary_size": 40,
        "horizons": (4, 9),
        "spike_deltas": (0.005, 0.05, 0.4),
        "n_calibration": 60,
        "n_evaluation_null": 30,
        "n_evaluation_alternative": 30,
        "batch_size": 10,
        "bayes_quadrature_nodes": 8,
        "gumbel_lookup_size": 2_001,
        "seed": 4_242,
        "max_regret_bootstrap_replicates": 50,
        "max_regret_bootstrap_batch_size": 10,
    }
    defaults.update(overrides)
    return sweep.RegimeSweepConfig(**defaults)  # type: ignore[arg-type]


class RegimeSamplerTests(unittest.TestCase):
    """The regime samplers must produce the Delta laws the sweep claims."""

    def test_point_regimes_are_exactly_degenerate(self) -> None:
        for regime in sweep.DEFAULT_REGIMES:
            if not regime.is_point:
                continue
            rng = np.random.default_rng(11)
            deltas = regime.sample(rng, 500)
            self.assertEqual(deltas.shape, (500, 1))
            np.testing.assert_array_equal(
                deltas, np.full((500, 1), regime.low, dtype=float)
            )
            self.assertEqual(regime.mean_delta, regime.low)

    def test_point_regimes_consume_no_randomness(self) -> None:
        # A degenerate branch must not perturb the pivot stream that follows.
        regime = sweep.Regime("Z", "point", 0.3, 0.3, "test")
        rng = np.random.default_rng(12)
        regime.sample(rng, 250)
        after_sampling = rng.uniform(size=5)
        untouched = np.random.default_rng(12).uniform(size=5)
        np.testing.assert_array_equal(after_sampling, untouched)

    def test_uniform_regimes_match_the_intended_uniform_law(self) -> None:
        rows = 200_000
        for regime in sweep.DEFAULT_REGIMES:
            if regime.is_point:
                continue
            with self.subTest(regime=regime.label):
                rng = np.random.default_rng(13)
                deltas = regime.sample(rng, rows)[:, 0]
                width = regime.high - regime.low
                self.assertGreaterEqual(float(deltas.min()), regime.low)
                self.assertLessEqual(float(deltas.max()), regime.high)
                # Mean within five standard errors of (low+high)/2.
                standard_error = width / math.sqrt(12.0 * rows)
                self.assertAlmostEqual(
                    float(deltas.mean()), regime.mean_delta, delta=5.0 * standard_error
                )
                # Empirical CDF matches the uniform CDF at every decile.
                for step in range(1, 10):
                    probability = step / 10.0
                    point = regime.low + probability * width
                    self.assertAlmostEqual(
                        float(np.mean(deltas <= point)), probability, delta=0.01
                    )

    def test_regime_supports_are_the_documented_ones(self) -> None:
        expected = {
            "A": ("uniform", 0.001, 0.5, True),
            "B": ("uniform", 0.001, 0.05, True),
            "C": ("uniform", 0.2, 0.5, True),
            "D": ("point", 0.005, 0.005, True),
            "E": ("point", 0.40, 0.40, True),
            "F": ("point", 0.70, 0.70, False),
            "G": ("point", 0.002, 0.002, True),
            "H": ("point", 0.0075, 0.0075, True),
            "I": ("point", 0.02, 0.02, True),
        }
        self.assertEqual(
            {regime.label for regime in sweep.DEFAULT_REGIMES}, set(expected)
        )
        for regime in sweep.DEFAULT_REGIMES:
            kind, low, high, inside = expected[regime.label]
            self.assertEqual(regime.kind, kind)
            self.assertAlmostEqual(regime.low, low)
            self.assertAlmostEqual(regime.high, high)
            self.assertEqual(regime.within(0.001, 0.5), inside)

    def test_unmatched_point_regimes_avoid_every_tested_deficit(self) -> None:
        """G, H and I exist so that no competitor is matched to them."""

        tested = set(sweep.RegimeSweepConfig().spike_deltas) | {
            sweep.RegimeSweepConfig().paper_deltas[-1]
        }
        for label in ("G", "H", "I"):
            regime = next(r for r in sweep.DEFAULT_REGIMES if r.label == label)
            self.assertEqual(regime.kind, "point")
            self.assertNotIn(regime.low, tested)

    def test_regime_f_really_sits_outside_the_prior_support(self) -> None:
        regime_f = next(r for r in sweep.DEFAULT_REGIMES if r.label == "F")
        config = sweep.RegimeSweepConfig()
        self.assertGreater(regime_f.low, config.prior_high)
        self.assertFalse(regime_f.within(config.prior_low, config.prior_high))

    def test_invalid_regimes_are_rejected(self) -> None:
        for bad in (
            sweep.Regime("X", "beta", 0.1, 0.2, ""),
            sweep.Regime("X", "point", 0.1, 0.2, ""),
            sweep.Regime("X", "uniform", 0.3, 0.3, ""),
            sweep.Regime("X", "uniform", 0.0, 0.2, ""),
            sweep.Regime("X", "point", 1.0, 1.0, ""),
        ):
            with self.assertRaises(ValueError):
                bad.validate()

    def test_delta_is_shared_by_every_token_of_a_document(self) -> None:
        # E[-log R] for one token equals (1-Delta)^2 + Delta^2/(V-1).  If Delta
        # were redrawn per token the per-document means would all collapse to a
        # single value instead of tracking each document's own Delta.
        regime = sweep.Regime("S", "uniform", 0.05, 0.45, "test")
        rng = np.random.default_rng(2024)
        vocabulary_size = 1000
        pivots, deltas = sweep.simulate_regime_pivots(
            rng, 40, 2_000, regime, vocabulary_size
        )
        observed = -np.log(pivots).mean(axis=1)
        expected = (1.0 - deltas) ** 2 + deltas**2 / (vocabulary_size - 1.0)
        np.testing.assert_allclose(observed, expected, atol=0.08)
        # And the documents genuinely differ, so the check has content.
        self.assertGreater(float(expected.max() - expected.min()), 0.3)

    def test_pivots_follow_the_exact_spike_law_of_the_sampled_delta(self) -> None:
        # Under the equal-tail spike with deficit d the pivot is R = U**p with
        # p = 1-d w.p. 1-d and p = d/(V-1) w.p. d, so
        #   P(R <= r) = (1-d) r**(1/(1-d)) + d r**((V-1)/d).
        # Conditioning on the realized per-document deficits makes this an exact
        # check on both the deficit sampler and the pivot construction.
        vocabulary_size = 1000
        for regime in sweep.DEFAULT_REGIMES:
            with self.subTest(regime=regime.label):
                rng = np.random.default_rng(99)
                pivots, deltas = sweep.simulate_regime_pivots(
                    rng, 200, 500, regime, vocabulary_size
                )
                if regime.is_point:
                    np.testing.assert_array_equal(
                        deltas, np.full(200, regime.low)
                    )
                for point in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
                    log_point = math.log(point)
                    expected = float(
                        np.mean(
                            (1.0 - deltas) * np.exp(log_point / (1.0 - deltas))
                            + deltas
                            * np.exp(
                                log_point * (vocabulary_size - 1.0) / deltas
                            )
                        )
                    )
                    self.assertAlmostEqual(
                        float(np.mean(pivots <= point)), expected, delta=0.008
                    )

    def test_regime_a_reproduces_the_existing_shared_delta_generator(self) -> None:
        config = paper.BenchmarkConfig(max_horizon=25, delta_low=0.001, delta_high=0.5)
        regime_a = next(r for r in sweep.DEFAULT_REGIMES if r.label == "A")
        reference = paper.simulate_gumbel_alternative_shared_delta(
            np.random.default_rng(555), 17, 25, config
        )
        produced, _ = sweep.simulate_regime_pivots(
            np.random.default_rng(555), 17, 25, regime_a, config.vocabulary_size
        )
        np.testing.assert_array_equal(produced, reference)

    def test_simulated_pivots_stay_in_the_unit_interval(self) -> None:
        for regime in sweep.DEFAULT_REGIMES:
            rng = np.random.default_rng(7)
            pivots, _ = sweep.simulate_regime_pivots(rng, 20, 50, regime, 1000)
            self.assertTrue(np.all(pivots > 0.0))
            self.assertTrue(np.all(pivots <= 1.0))


class FrozenPriorTests(unittest.TestCase):
    """The prior must be fixed before the sweep and never retuned per regime."""

    def test_quadrature_depends_only_on_the_prior_endpoints(self) -> None:
        config = sweep.RegimeSweepConfig()
        deltas, weights = sweep.frozen_delta_quadrature(config)
        self.assertEqual(deltas.size, config.bayes_quadrature_nodes)
        self.assertGreaterEqual(float(deltas.min()), config.prior_low)
        self.assertLessEqual(float(deltas.max()), config.prior_high)
        # Regime F generates Delta=0.70; the grid must not reach it.
        self.assertLess(float(deltas.max()), 0.7)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)

    def test_quadrature_ignores_the_regime_list(self) -> None:
        full = sweep.RegimeSweepConfig()
        one_regime = dataclasses.replace(
            full, regimes=(next(r for r in sweep.DEFAULT_REGIMES if r.label == "F"),)
        )
        for left, right in zip(
            sweep.frozen_delta_quadrature(full),
            sweep.frozen_delta_quadrature(one_regime),
        ):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(
            sweep.prior_fingerprint(full), sweep.prior_fingerprint(one_regime)
        )

    def test_scoring_entry_point_has_no_regime_parameter(self) -> None:
        # A structural guarantee: no rule can see which regime produced a
        # document, so none of them can be retuned to it.
        for function in (sweep.all_rule_paths, sweep.frozen_delta_quadrature):
            parameters = set(inspect.signature(function).parameters)
            self.assertNotIn("regime", parameters)
            self.assertNotIn("regimes", parameters)
        self.assertEqual(
            list(inspect.signature(sweep.all_rule_paths).parameters),
            ["pivots", "config", "lookup", "dirichlet_grid", "uniontail_grid"],
        )
        # The Dirichlet grid is built from the config alone, so passing it in
        # cannot smuggle regime information into a rule.
        self.assertNotIn(
            "regime", set(inspect.signature(sweep.RegimeSweepConfig.uniontail_grid).parameters)
        )
        self.assertNotIn(
            "regime", set(inspect.signature(sweep.RegimeSweepConfig.dirichlet_grid).parameters)
        )

    def test_shared_bayes_uses_the_frozen_prior_on_out_of_support_data(self) -> None:
        config = tiny_config()
        lookup = paper.GumbelBayesLookup(config.paper_config())
        regime_f = next(r for r in sweep.DEFAULT_REGIMES if r.label == "F")
        pivots, deltas = sweep.simulate_regime_pivots(
            np.random.default_rng(31337), 3, config.max_horizon, regime_f,
            config.vocabulary_size,
        )
        np.testing.assert_array_equal(deltas, np.full(3, 0.7))
        paths = sweep.all_rule_paths(pivots, config, lookup)

        grid, weights = sweep.frozen_delta_quadrature(config)
        selected = [horizon - 1 for horizon in config.horizons]
        for row in range(pivots.shape[0]):
            detector = SequentialBayesDetector(
                "gumbel",
                grid,
                weights,
                structure="shared",
                gumbel_family="spike",
                vocabulary_size=config.vocabulary_size,
            )
            detector.update_many(pivots[row])
            history = detector.log_bayes_factor_history
            np.testing.assert_allclose(
                paths["bayes_shared"][row],
                history[selected],
                rtol=1e-11,
                atol=1e-11,
            )

    def test_tokenwise_bayes_uses_the_same_frozen_prior(self) -> None:
        config = tiny_config()
        lookup = paper.GumbelBayesLookup(config.paper_config())
        pivots = np.random.default_rng(4).uniform(size=(5, config.max_horizon))
        paths = sweep.all_rule_paths(pivots, config, lookup)
        selected = [horizon - 1 for horizon in config.horizons]
        expected = np.cumsum(lookup(pivots), axis=1)[:, selected]
        np.testing.assert_allclose(paths["bayes_tokenwise"], expected, rtol=0, atol=0)
        # The lookup is built from the prior endpoints, not from any regime.
        paper_config = config.paper_config()
        self.assertEqual(paper_config.delta_low, config.prior_low)
        self.assertEqual(paper_config.delta_high, config.prior_high)

    def test_fixed_and_paper_rules_are_wired_to_the_reference_scores(self) -> None:
        config = tiny_config()
        lookup = paper.GumbelBayesLookup(config.paper_config())
        pivots = np.random.default_rng(5).uniform(size=(6, config.max_horizon))
        paths = sweep.all_rule_paths(pivots, config, lookup)
        selected = [horizon - 1 for horizon in config.horizons]
        for delta in config.spike_deltas:
            expected = np.cumsum(
                paper.spike_point_mass_score(
                    pivots, float(delta), config.vocabulary_size
                ),
                axis=1,
            )[:, selected]
            np.testing.assert_allclose(
                paths[sweep.spike_rule_name(delta)], expected, rtol=0, atol=0
            )
        expected_paper = np.cumsum(
            paper.paper_gumbel_optimal_score(
                pivots, float(config.paper_deltas[-1])), axis=1
        )[:, selected]
        np.testing.assert_allclose(
            paths[sweep.paper_rule_name(config.paper_deltas[-1])],
            expected_paper,
            rtol=0,
            atol=0,
        )

    def test_adding_a_rule_leaves_the_other_rules_type_two_unchanged(self) -> None:
        # The regret functionals are min-relative, so adding a rule legitimately
        # moves every regret entry.  Type I, Type II and the cutoffs must NOT
        # move: they are computed per rule from a shared sample, and if they do
        # move it means the new rule perturbed an RNG stream or a calibration.
        base = tiny_config(spike_deltas=(0.005, 0.05))
        extended = dataclasses.replace(base, spike_deltas=(0.005, 0.05, 0.4))
        first = sweep.run_regime_sweep(base)
        second = sweep.run_regime_sweep(extended)
        shared_rules = set(base.rule_names())
        self.assertTrue(shared_rules < set(extended.rule_names()))
        for horizon in base.horizons:
            key = str(horizon)
            for rule in shared_rules:
                for label in base.regime_labels():
                    self.assertEqual(
                        first["type2_error"][key][rule][label],
                        second["type2_error"][key][rule][label],
                        msg=f"type2 moved: {rule} {label} n={horizon}",
                    )
        by_key = {
            (c["rule"], c["regime"], c["horizon"]): c for c in first["cells"]
        }
        for cell in second["cells"]:
            reference = by_key.get((cell["rule"], cell["regime"], cell["horizon"]))
            if reference is None:
                continue
            self.assertEqual(cell["threshold"], reference["threshold"])
            self.assertEqual(cell["type1_error"], reference["type1_error"])
            self.assertEqual(
                cell["boundary_randomization"], reference["boundary_randomization"]
            )

    def test_regret_is_recomputed_against_the_enlarged_rule_set(self) -> None:
        # The complement of the invariance above: the per-regime minimum is taken
        # over all rules, so an added rule that wins anywhere must lower the
        # baseline and change the other rules' regret.  If this ever stops
        # holding, regret is being computed against a stale baseline.
        table = {
            "a": {"R": 0.10, "S": 0.20},
            "b": {"R": 0.30, "S": 0.05},
        }
        regret, best = sweep.regret_table(table)
        self.assertEqual(best, {"R": 0.10, "S": 0.05})
        enlarged = dict(table)
        enlarged["c"] = {"R": 0.01, "S": 0.50}
        regret_after, best_after = sweep.regret_table(enlarged)
        self.assertEqual(best_after, {"R": 0.01, "S": 0.05})
        self.assertGreater(regret_after["a"]["R"], regret["a"]["R"])
        self.assertEqual(regret_after["a"]["S"], regret["a"]["S"])

    def test_dirichlet_rule_is_present_and_carries_its_own_kind(self) -> None:
        payload = sweep.run_regime_sweep(tiny_config())
        kinds = {r["name"]: r["kind"] for r in payload["rules"]}
        families = {r["name"]: r["component_family"] for r in payload["rules"]}
        self.assertEqual(
            kinds[sweep.DIRICHLET_RULE], "bayes_frozen_prior_dirichlet_tail"
        )
        self.assertEqual(kinds["bayes_shared"], "bayes_frozen_prior_spike_tail")
        self.assertEqual(families[sweep.DIRICHLET_RULE], "dirichlet_tail_mixture")
        self.assertEqual(families["bayes_shared"], "equal_tail_spike")
        # The recorded prior must name the alpha factor, or a reader cannot tell
        # which rules the tail prior applies to.
        alpha_block = payload["frozen_prior"]["alpha"]
        self.assertIn("inf", alpha_block["grid"])
        self.assertEqual(alpha_block["used_by"], [sweep.DIRICHLET_RULE])

    def test_config_rejects_a_bad_alpha_grid(self) -> None:
        for grid in ((), (1.0, 1.0), (0.0,), (-1.0,), (math.nan,)):
            with self.subTest(grid=grid):
                with self.assertRaises(ValueError):
                    tiny_config(dirichlet_alpha_grid=grid).validate()

    def test_adding_regimes_leaves_the_rules_and_their_results_unchanged(self) -> None:
        # The decisive non-retuning check: the same rules, calibrated once,
        # produce bit-identical cutoffs and bit-identical Type II errors on the
        # regimes common to both runs, even though the second run adds three
        # further regimes including one outside the prior's support.
        first_three = sweep.DEFAULT_REGIMES[:3]
        small = sweep.run_regime_sweep(tiny_config(regimes=first_three))
        large = sweep.run_regime_sweep(tiny_config(regimes=sweep.DEFAULT_REGIMES))

        self.assertEqual(small["calibration"]["cutoffs"], large["calibration"]["cutoffs"])
        self.assertEqual(
            small["calibration"]["boundary_randomization"],
            large["calibration"]["boundary_randomization"],
        )
        self.assertEqual(small["type1_error"], large["type1_error"])
        self.assertEqual(small["frozen_prior"], large["frozen_prior"])
        for horizon in small["horizons"]:
            key = str(horizon)
            for rule in small["type2_error"][key]:
                for regime in ("A", "B", "C"):
                    self.assertEqual(
                        small["type2_error"][key][rule][regime],
                        large["type2_error"][key][rule][regime],
                        msg=f"{rule} at regime {regime}, n={horizon}",
                    )

    def test_every_regime_records_the_same_frozen_prior(self) -> None:
        payload = sweep.run_regime_sweep(tiny_config())
        fingerprints = [
            entry["frozen_prior_used"]
            for entry in payload["regime_delta_diagnostics"].values()
        ]
        self.assertEqual(len(fingerprints), len(sweep.DEFAULT_REGIMES))
        for fingerprint in fingerprints:
            self.assertEqual(fingerprint, payload["frozen_prior"])
        for entry in payload["rules"]:
            self.assertFalse(entry["retuned_per_regime"])


class RegretTests(unittest.TestCase):
    def test_regret_matches_its_definition_on_a_hand_built_table(self) -> None:
        table = {
            "a": {"X": 0.10, "Y": 0.50},
            "b": {"X": 0.25, "Y": 0.05},
            "c": {"X": 0.10, "Y": 0.30},
        }
        regret, best = sweep.regret_table(table)
        self.assertEqual(best, {"X": 0.10, "Y": 0.05})
        self.assertAlmostEqual(regret["a"]["X"], 0.0)
        self.assertAlmostEqual(regret["a"]["Y"], 0.45)
        self.assertAlmostEqual(regret["b"]["X"], 0.15)
        self.assertAlmostEqual(regret["b"]["Y"], 0.0)
        self.assertAlmostEqual(regret["c"]["X"], 0.0)
        self.assertAlmostEqual(regret["c"]["Y"], 0.25)

    def test_regret_is_nonnegative_and_exactly_zero_for_the_best_rule(self) -> None:
        rng = np.random.default_rng(17)
        table = {
            f"rule_{index}": {
                f"regime_{regime}": float(value)
                for regime, value in enumerate(rng.uniform(size=5))
            }
            for index in range(7)
        }
        regret, best = sweep.regret_table(table)
        for regime in table["rule_0"]:
            zeros = [rule for rule in table if regret[rule][regime] == 0.0]
            self.assertEqual(len(zeros), 1)
            self.assertEqual(table[zeros[0]][regime], best[regime])
            for rule in table:
                self.assertGreaterEqual(regret[rule][regime], 0.0)

    def test_regret_is_zero_everywhere_when_all_rules_tie(self) -> None:
        table = {"a": {"X": 0.2}, "b": {"X": 0.2}}
        regret, best = sweep.regret_table(table)
        self.assertEqual(best, {"X": 0.2})
        self.assertEqual(regret, {"a": {"X": 0.0}, "b": {"X": 0.0}})

        # Paired resampling must also preserve an exact document-level tie.
        base = np.array([[0.0], [1.0], [0.0], [1.0]])
        tied_losses = np.stack((base, base), axis=1)
        means = sweep.paired_bootstrap_means(
            tied_losses,
            np.random.default_rng(19),
            replicates=40,
            batch_size=7,
        )
        per_horizon, pooled = sweep.bootstrap_max_regret_mc_se(
            means[:, None, :, :]
        )
        np.testing.assert_array_equal(per_horizon, np.zeros((2, 1)))
        np.testing.assert_array_equal(pooled, np.zeros(2))

    def test_regret_table_rejects_ragged_or_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            sweep.regret_table({})
        with self.assertRaises(ValueError):
            sweep.regret_table({"a": {"X": 0.1}, "b": {"Y": 0.2}})

    def test_max_regret_reports_the_worst_regime(self) -> None:
        regret = {"a": {"X": 0.0, "Y": 0.4}, "b": {"X": 0.0, "Y": 0.0}}
        summary = sweep.max_regret(regret)
        self.assertAlmostEqual(summary["a"]["max_regret"], 0.4)
        self.assertEqual(summary["a"]["argmax_regime"], "Y")
        self.assertEqual(summary["b"]["max_regret"], 0.0)
        self.assertIsNone(summary["b"]["argmax_regime"])

    def test_pooled_max_regret_takes_the_worst_cell_over_horizons(self) -> None:
        regret = {
            "10": {"a": {"X": 0.1, "Y": 0.0}, "b": {"X": 0.0, "Y": 0.0}},
            "20": {"a": {"X": 0.0, "Y": 0.3}, "b": {"X": 0.0, "Y": 0.0}},
        }
        pooled = sweep.pooled_max_regret(regret)
        self.assertAlmostEqual(pooled["a"]["max_regret"], 0.3)
        self.assertEqual(pooled["a"]["argmax_horizon"], 20)
        self.assertEqual(pooled["a"]["argmax_regime"], "Y")
        self.assertEqual(pooled["b"]["max_regret"], 0.0)
        self.assertIsNone(pooled["b"]["argmax_horizon"])
        self.assertIsNone(pooled["b"]["argmax_regime"])
        with self.assertRaises(ValueError):
            sweep.pooled_max_regret({})

    def test_payload_pooled_max_regret_dominates_every_horizon(self) -> None:
        payload = sweep.run_regime_sweep(tiny_config())
        pooled = payload["pooled_max_regret_across_horizons"]
        for rule, entry in pooled.items():
            per_horizon = [
                payload["max_regret"][str(horizon)][rule]["max_regret"]
                for horizon in payload["horizons"]
            ]
            self.assertAlmostEqual(entry["max_regret"], max(per_horizon), places=15)

    def test_stored_best_rule_lists_exclude_fixed_deficit_diagnostics(self) -> None:
        artifact = (Path(__file__).resolve().parents[1] / "results"
                    / "bayesian_paper_benchmark" / "regime_sweep.json")
        payload = json.loads(artifact.read_text())
        self.assertIn("excluding the h_spike", payload["competitor_scope"])
        for horizon, regimes in payload["best_rule_by_regime"].items():
            errors = payload["type2_error"][horizon]
            benchmark = [rule for rule in errors if sweep.is_benchmark_rule(rule)]
            for regime, best in regimes.items():
                with self.subTest(horizon=horizon, regime=regime):
                    minimum = min(errors[rule][regime] for rule in benchmark)
                    self.assertEqual(best["type2_error"], minimum)
                    self.assertEqual(best["rules"], sorted(
                        rule for rule in benchmark if errors[rule][regime] == minimum
                    ))

    def test_payload_regret_is_consistent_with_its_type_two_table(self) -> None:
        payload = sweep.run_regime_sweep(tiny_config())
        labels = [regime["label"] for regime in payload["regimes"]]
        rules = [rule["name"] for rule in payload["rules"]]
        for horizon in payload["horizons"]:
            key = str(horizon)
            type2 = payload["type2_error"][key]
            regret = payload["regret"][key]
            # The benchmark is the best COMPETITOR: the fixed-Delta_0 equal-tail
            # scores are diagnostics handed the generating family and one true
            # deficit, so letting them set the bar would measure every real rule
            # against an oracle.  A diagnostic may therefore have negative
            # regret, which is the honest way to show that it beat the field.
            benchmark = [r for r in rules if sweep.is_benchmark_rule(r)]
            self.assertTrue(benchmark, "no benchmark rule survived the filter")
            self.assertLess(len(benchmark), len(rules),
                            "this config has no diagnostic, so it cannot test the split")
            for regime in labels:
                # Only benchmark rules may appear as winners. Diagnostics can
                # tie the minimum (zero regret) without joining the benchmark.
                minimum = min(type2[rule][regime] for rule in benchmark)
                winners = [rule for rule in benchmark
                           if type2[rule][regime] == minimum]
                self.assertEqual(
                    payload["best_rule_by_regime"][key][regime]["type2_error"], minimum
                )
                self.assertEqual(
                    payload["best_rule_by_regime"][key][regime]["rules"],
                    sorted(winners),
                )
                for rule in rules:
                    value = regret[rule][regime]
                    if sweep.is_benchmark_rule(rule):
                        self.assertGreaterEqual(value, 0.0)
                    self.assertAlmostEqual(
                        value, type2[rule][regime] - minimum, places=15
                    )
                    self.assertEqual(value == 0.0 and sweep.is_benchmark_rule(rule),
                                     rule in winners)
            for rule in rules:
                self.assertAlmostEqual(
                    payload["max_regret"][key][rule]["max_regret"],
                    max(regret[rule][regime] for regime in labels),
                    places=15,
                )
                self.assertGreaterEqual(
                    payload["max_regret"][key][rule]["max_regret_mc_se"], 0.0
                )
                self.assertTrue(
                    math.isfinite(
                        payload["max_regret"][key][rule]["max_regret_mc_se"]
                    )
                )

    def test_payload_max_regret_ranking_is_sorted_and_complete(self) -> None:
        payload = sweep.run_regime_sweep(tiny_config())
        rules = [rule["name"] for rule in payload["rules"]]
        for horizon in payload["horizons"]:
            key = str(horizon)
            ranking = payload["max_regret_ranking"][key]
            self.assertEqual({entry["rule"] for entry in ranking}, set(rules))
            values = [entry["max_regret"] for entry in ranking]
            self.assertEqual(values, sorted(values))
            for entry in ranking:
                self.assertEqual(
                    entry["max_regret"],
                    payload["max_regret"][key][entry["rule"]]["max_regret"],
                )
                self.assertEqual(
                    entry["max_regret_mc_se"],
                    payload["max_regret"][key][entry["rule"]][
                        "max_regret_mc_se"
                    ],
                )

    def test_lower_envelope_membership_means_zero_regret_everywhere(self) -> None:
        payload = sweep.run_regime_sweep(tiny_config())
        labels = [regime["label"] for regime in payload["regimes"]]
        rules = [rule["name"] for rule in payload["rules"]]
        for horizon in payload["horizons"]:
            key = str(horizon)
            envelope = set(
                payload["rules_attaining_the_lower_envelope"][key][
                    "rules_attaining_every_regime_minimum"
                ]
            )
            for rule in rules:
                everywhere_best = all(
                    payload["regret"][key][rule][regime] == 0.0 for regime in labels
                )
                self.assertEqual(rule in envelope, everywhere_best)
                if everywhere_best:
                    self.assertEqual(
                        payload["max_regret"][key][rule]["max_regret"], 0.0
                    )


class SweepPayloadTests(unittest.TestCase):
    def test_rule_names_are_the_intended_set(self) -> None:
        config = sweep.RegimeSweepConfig()
        self.assertEqual(
            config.rule_names(),
            (
                "h_spike_0.005",
                "h_spike_0.01",
                "h_spike_0.05",
                "h_spike_0.2",
                "h_spike_0.4",
                "bayes_shared",
                "bayes_shared_dirichlet",
                "bayes_shared_uniontail",
                "bayes_tokenwise",
                # Every published tuning of the least-favorable score and every
                # reference score: one tuning could not show that the constant
                # matters, which is what these tables are for.
                "h_gum_star_0.1",
                "h_gum_star_0.01",
                "h_gum_star_0.005",
                "h_ars",
                "h_log",
                "h_ind_1_over_e",
            ),
        )

    def test_seed_is_fresh(self) -> None:
        self.assertNotIn(
            sweep.SWEEP_SEED, {24_040_1245, 24_040_1246, 24_040_1247}
        )
        self.assertEqual(sweep.RegimeSweepConfig().seed, sweep.SWEEP_SEED)

    def test_config_validation_rejects_bad_input(self) -> None:
        base = sweep.RegimeSweepConfig()
        for overrides in (
            {"horizons": ()},
            {"horizons": (300, 100)},
            {"horizons": (0, 100)},
            {"spike_deltas": ()},
            {"spike_deltas": (0.1, 0.1)},
            {"spike_deltas": (1.5,)},
            {
                "vocabulary_size": 3,
                "spike_deltas": (0.75,),
                "regimes": (sweep.Regime("A", "point", 0.5, 0.5, ""),),
            },
            {"paper_deltas": (0.0,)},
            {"max_regret_bootstrap_replicates": 1},
            {"max_regret_bootstrap_batch_size": 0},
            {"regimes": ()},
            {
                "regimes": (
                    sweep.Regime("A", "point", 0.1, 0.1, ""),
                    sweep.Regime("A", "point", 0.2, 0.2, ""),
                )
            },
            {"regimes": (sweep.Regime("A", "uniform", 0.4, 0.2, ""),)},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    dataclasses.replace(base, **overrides).validate()

    def test_payload_covers_every_rule_regime_horizon_cell(self) -> None:
        config = tiny_config()
        payload = sweep.run_regime_sweep(config)
        expected = len(config.rule_names()) * len(config.regimes) * len(
            config.horizons
        )
        self.assertEqual(len(payload["cells"]), expected)
        seen = {
            (cell["rule"], cell["regime"], cell["horizon"]) for cell in payload["cells"]
        }
        self.assertEqual(len(seen), expected)
        for cell in payload["cells"]:
            self.assertGreaterEqual(cell["type2_error"], 0.0)
            self.assertLessEqual(cell["type2_error"], 1.0)
            self.assertAlmostEqual(
                cell["type2_error"] + cell["power"], 1.0, places=12
            )

    def test_calibration_sample_is_shared_and_evaluation_null_is_separate(self) -> None:
        config = tiny_config()
        payload = sweep.run_regime_sweep(config)
        self.assertEqual(
            payload["calibration"]["n_calibration"], config.n_calibration
        )
        # One cutoff per rule per horizon, reused verbatim in every regime cell.
        for cell in payload["cells"]:
            self.assertEqual(
                cell["threshold"],
                payload["calibration"]["cutoffs"][cell["rule"]][str(cell["horizon"])],
            )
        # Realized Type I is measured on a different sample than calibration,
        # so it need not be exactly alpha but must be a probability.
        for rule, by_horizon in payload["type1_error"].items():
            for value in by_horizon.values():
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_delta_diagnostics_report_the_generated_deltas(self) -> None:
        payload = sweep.run_regime_sweep(tiny_config())
        diagnostics = payload["regime_delta_diagnostics"]
        self.assertAlmostEqual(diagnostics["D"]["realized_mean_delta"], 0.005)
        self.assertAlmostEqual(diagnostics["E"]["realized_mean_delta"], 0.40)
        self.assertAlmostEqual(diagnostics["F"]["realized_mean_delta"], 0.70)
        self.assertEqual(diagnostics["F"]["fraction_outside_prior_support"], 1.0)
        self.assertFalse(diagnostics["F"]["support_inside_prior"])
        for label in ("A", "B", "C", "D", "E"):
            self.assertEqual(diagnostics[label]["fraction_outside_prior_support"], 0.0)
            self.assertTrue(diagnostics[label]["support_inside_prior"])
        for label in ("A", "B", "C"):
            entry = diagnostics[label]
            self.assertGreaterEqual(entry["realized_min_delta"], entry["low"])
            self.assertLessEqual(entry["realized_max_delta"], entry["high"])

    def test_saturated_regimes_are_exactly_the_all_zero_columns(self) -> None:
        payload = sweep.run_regime_sweep(tiny_config())
        rules = [rule["name"] for rule in payload["rules"]]
        labels = [regime["label"] for regime in payload["regimes"]]
        for horizon in payload["horizons"]:
            key = str(horizon)
            expected = [
                regime
                for regime in labels
                if all(payload["type2_error"][key][rule][regime] == 0.0 for rule in rules)
            ]
            self.assertEqual(payload["saturated_regimes"][key], expected)

    def test_payload_is_json_serializable_and_round_trips(self) -> None:
        payload = sweep.run_regime_sweep(tiny_config())
        restored = json.loads(json.dumps(payload))
        self.assertEqual(restored["seed"], payload["seed"])
        self.assertEqual(restored["type2_error"], payload["type2_error"])
        self.assertEqual(restored["regret"], payload["regret"])
        self.assertIn("markdown_tables", restored)

    def test_markdown_tables_have_a_row_for_every_rule(self) -> None:
        config = tiny_config()
        payload = sweep.run_regime_sweep(config)
        for horizon in payload["horizons"]:
            for name in ("type2", "regret"):
                text = payload["markdown_tables"][str(horizon)][name]
                for rule in config.rule_names():
                    self.assertIn(f"| {rule} |", text)
                for regime in config.regimes:
                    self.assertIn(regime.label, text.splitlines()[2])
            self.assertIn(
                "MAX REGRET", payload["markdown_tables"][str(horizon)]["regret"]
            )

    def test_probe_is_absent_by_default_and_self_consistent_when_asked(self) -> None:
        config = tiny_config()
        self.assertIsNone(
            sweep.run_regime_sweep(config)["short_horizon_saturation_probe"]
        )
        payload = sweep.run_regime_sweep(config, probe_horizons=(2, 5))
        probe = payload["short_horizon_saturation_probe"]
        self.assertEqual(probe["horizons"], [2, 5])
        # Its own seed, so it cannot silently reuse the main sweep's documents.
        self.assertNotEqual(probe["seed"], payload["seed"])
        self.assertEqual(probe["seed"], config.seed + 1)
        self.assertEqual(set(probe["type2_error"]), {"2", "5"})
        first = probe["shortest_horizon_at_which_a_regime_separates_rules"]
        self.assertEqual(set(first), set(payload["type2_error"]["4"]["bayes_shared"]))
        for regime, horizon in first.items():
            unsaturated = [
                int(key)
                for key in ("2", "5")
                if regime not in probe["saturated_regimes"][key]
            ]
            self.assertEqual(horizon, min(unsaturated) if unsaturated else None)

    def test_probe_leaves_the_reported_design_untouched(self) -> None:
        config = tiny_config()
        plain = sweep.run_regime_sweep(config)
        probed = sweep.run_regime_sweep(config, probe_horizons=(2, 5))
        self.assertEqual(plain["type2_error"], probed["type2_error"])
        self.assertEqual(plain["regret"], probed["regret"])
        self.assertEqual(plain["calibration"]["cutoffs"], probed["calibration"]["cutoffs"])
        self.assertEqual(plain["horizons"], probed["horizons"])

    def test_all_rule_paths_rejects_short_pivot_arrays(self) -> None:
        config = tiny_config()
        lookup = paper.GumbelBayesLookup(config.paper_config())
        with self.assertRaises(ValueError):
            sweep.all_rule_paths(
                np.random.default_rng(1).uniform(size=(3, config.max_horizon - 1)),
                config,
                lookup,
            )


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
        names = sweep.RegimeSweepConfig().rule_names()
        self.assertEqual(names.count(sweep.UNIONTAIL_RULE), 1)

    def test_full_width_atom_lives_in_the_dirichlet_block(self) -> None:
        # The union splits the tail prior: the full-width atom belongs to the
        # Dirichlet block, so the tail-width ladder must stop below it.
        grid = sweep.RegimeSweepConfig().uniontail_grid()
        self.assertTrue(all(j < grid.tail_size for j in grid.tail_widths))
        self.assertEqual(grid.dirichlet_block.tail_size, grid.tail_size)
        self.assertAlmostEqual(sum(grid.block_weights), 1.0)

    def test_grid_does_not_depend_on_the_regime(self) -> None:
        # Same structural guarantee the Dirichlet grid carries: the prior is
        # frozen, so no rule can be retuned to the regime that produced a document.
        self.assertNotIn(
            "regime",
            set(inspect.signature(sweep.RegimeSweepConfig.uniontail_grid).parameters),
        )


if __name__ == "__main__":
    unittest.main()
