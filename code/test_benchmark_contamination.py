"""Tests for the iid null-like contamination robustness benchmark."""

from __future__ import annotations

import dataclasses
import json
import unittest
from pathlib import Path

import numpy as np

import benchmark_contamination as contamination
import benchmark_paper_experiment as paper
import paired_comparisons as paired
from bayesian_watermark import SequentialBayesDetector, logsumexp


class ContaminationBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = contamination.ContaminationConfig(
            vocabulary_size=40,
            horizons=(6, 12),
            rho_true_grid=(0.0, 0.25, 1.0),
            rho_prior_grid=(0.0, 0.25),
            rho_prior_weights=(0.6, 0.4),
            n_calibration=30,
            n_evaluation_null=20,
            n_evaluation_alternative=20,
            batch_size=10,
            bayes_quadrature_nodes=8,
            gumbel_lookup_size=2_001,
        )

    def test_contamination_extremes_and_nested_masks(self) -> None:
        clean = np.arange(12, dtype=float).reshape(3, 4)
        null = clean + 100.0
        masks = np.array(
            [[0.05, 0.20, 0.40, 0.80], [0.15, 0.30, 0.50, 0.90], [0.1, 0.2, 0.3, 0.4]]
        )
        np.testing.assert_array_equal(
            contamination.contaminated_pivots(clean, null, masks, 0.0), clean
        )
        np.testing.assert_array_equal(
            contamination.contaminated_pivots(clean, null, masks, 1.0), null
        )
        low = contamination.contaminated_pivots(clean, null, masks, 0.25) == null
        high = contamination.contaminated_pivots(clean, null, masks, 0.50) == null
        self.assertTrue(np.all(~low | high))

    def test_boundary_uniforms_are_reproducible_and_cell_addressed(self) -> None:
        arguments = {
            "seed": self.config.seed,
            "scheme": "gumbel",
            "rho_true": 0.25,
            "horizon": 12,
            "n_documents": 20,
        }
        first = contamination.boundary_randomization_uniforms(**arguments)
        second = contamination.boundary_randomization_uniforms(**arguments)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all((first >= 0.0) & (first < 1.0)))

        for changed in (
            {"seed": self.config.seed + 1},
            {"scheme": "inverse"},
            {"rho_true": 0.4},
            {"horizon": 6},
        ):
            candidate = contamination.boundary_randomization_uniforms(
                **(arguments | changed)
            )
            self.assertFalse(np.array_equal(first, candidate))

        # The entropy records rho_true exactly rather than via a rounded text
        # label, and does not contain a method component.
        entropy = contamination._boundary_randomization_seed_entropy(
            self.config.seed, "gumbel", 0.25, 12
        )
        self.assertEqual(len(entropy), 6)
        self.assertNotEqual(
            entropy,
            contamination._boundary_randomization_seed_entropy(
                self.config.seed, "gumbel", np.nextafter(0.25, 1.0), 12
            ),
        )

    def test_shared_robust_singleton_zero_equals_clean(self) -> None:
        config = contamination.ContaminationConfig(
            vocabulary_size=40,
            horizons=(5, 10),
            rho_true_grid=(0.0, 1.0),
            rho_prior_grid=(0.0,),
            rho_prior_weights=(1.0,),
            n_calibration=10,
            n_evaluation_null=10,
            n_evaluation_alternative=10,
            batch_size=5,
            bayes_quadrature_nodes=8,
            gumbel_lookup_size=2_001,
        )
        rng = np.random.default_rng(777)
        pivots = paper.simulate_gumbel_null(rng, 9, 10, config.paper_config())
        paths = contamination.shared_bayes_selected_paths(pivots, "gumbel", config)
        np.testing.assert_allclose(
            paths["bayes_shared_clean"],
            paths["bayes_shared_robust"],
            atol=1e-12,
            rtol=1e-12,
        )

    def test_shared_paths_match_reference_detector_for_both_schemes(self) -> None:
        deltas, delta_weights = contamination._delta_quadrature(self.config)
        for scheme, simulator in (
            ("gumbel", paper.simulate_gumbel_null),
            ("inverse", paper.simulate_inverse_null),
        ):
            rng = np.random.default_rng(779)
            pivots = simulator(
                rng, 3, self.config.max_horizon, self.config.paper_config()
            )
            paths = contamination.shared_bayes_selected_paths(
                pivots, scheme, self.config
            )["bayes_shared_robust"]
            for row_index, sequence in enumerate(pivots):
                detector = SequentialBayesDetector(
                    scheme,
                    deltas,
                    delta_weights,
                    rho_grid=self.config.rho_prior_grid,
                    rho_weights=self.config.rho_prior_weights,
                    structure="shared",
                    gumbel_family="spike",
                    vocabulary_size=self.config.vocabulary_size,
                    inverse_null_vocabulary_size=(
                        self.config.vocabulary_size if scheme == "inverse" else None
                    ),
                )
                detector.update_many(sequence)
                expected = detector.log_bayes_factor_history[
                    np.asarray(self.config.horizons) - 1
                ]
                np.testing.assert_allclose(
                    paths[row_index], expected, atol=1e-11, rtol=1e-11
                )

    def test_tokenwise_rho_prior_collapses_to_mean(self) -> None:
        lookup = paper.GumbelBayesLookup(self.config.paper_config())
        rng = np.random.default_rng(778)
        pivots = paper.simulate_gumbel_null(
            rng, 7, self.config.max_horizon, self.config.paper_config()
        )
        paths = contamination.token_and_paper_selected_paths(
            pivots, "gumbel", self.config, lookup
        )
        clean = lookup(pivots)
        mean_rho = 0.1
        expected = np.cumsum(
            np.logaddexp(np.log(mean_rho), np.log1p(-mean_rho) + clean), axis=1
        )[:, np.asarray(self.config.horizons) - 1]
        np.testing.assert_allclose(
            paths["bayes_tokenwise_robust"], expected, atol=1e-12, rtol=1e-12
        )

        same_mean_config = contamination.ContaminationConfig(
            vocabulary_size=40,
            horizons=(6, 12),
            rho_true_grid=(0.0, 1.0),
            rho_prior_grid=(0.05, 0.15),
            rho_prior_weights=(0.5, 0.5),
            n_calibration=30,
            n_evaluation_null=20,
            n_evaluation_alternative=20,
            batch_size=10,
            bayes_quadrature_nodes=8,
            gumbel_lookup_size=2_001,
        )
        same_mean = contamination.token_and_paper_selected_paths(
            pivots, "gumbel", same_mean_config, lookup
        )
        np.testing.assert_allclose(
            paths["bayes_tokenwise_robust"],
            same_mean["bayes_tokenwise_robust"],
            atol=1e-12,
            rtol=1e-12,
        )

    def test_tiny_benchmark_has_diagnostic_and_all_methods(self) -> None:
        rows, metadata = contamination.run_contamination_benchmark(self.config)
        self.assertEqual({row["rho_true"] for row in rows}, {0.0, 0.25, 1.0})
        point_masses = set(self.config.point_mass_methods())
        # The Dirichlet tail layer is Gumbel-only: the inverse limiting
        # alternative depends on Delta alone, so a tail prior is inert for it.
        expected_methods = {
            "gumbel": set(contamination.GUMBEL_PAPER_METHODS)
            | set(contamination.BAYES_METHODS)
            | set(contamination.DIRICHLET_METHODS)
            | set(contamination.UNIONTAIL_METHODS)
            | {contamination.TRGOF_METHOD}
            | point_masses,
            "inverse": set(contamination.INVERSE_PAPER_METHODS)
            | set(contamination.BAYES_METHODS)
            | {contamination.TRGOF_METHOD}
            | point_masses,
        }
        for scheme, methods in expected_methods.items():
            self.assertEqual(
                {row["method"] for row in rows if row["scheme"] == scheme}, methods
            )
        expected_rows = sum(len(methods) for methods in expected_methods.values())
        expected_rows *= len(self.config.rho_true_grid) * len(self.config.horizons)
        self.assertEqual(len(rows), expected_rows)
        for row in rows:
            self.assertAlmostEqual(row["power"] + row["type_ii_error"], 1.0)
        self.assertGreater(metadata["runtime_seconds"], 0.0)
        self.assertLess(
            metadata["rho_one_diagnostic_standardized_gap"], 5.0
        )


class RhoPointMassDiagnosticTests(unittest.TestCase):
    """Tests for the pi_rho = delta_{rho_0} diagnostic detectors."""

    def setUp(self) -> None:
        self.config = contamination.ContaminationConfig(
            vocabulary_size=40,
            horizons=(6, 12),
            rho_true_grid=(0.0, 0.25, 1.0),
            rho_prior_grid=(0.0, 0.25),
            rho_prior_weights=(0.6, 0.4),
            rho_point_masses=(0.1, 0.25),
            n_calibration=30,
            n_evaluation_null=20,
            n_evaluation_alternative=20,
            batch_size=10,
            bayes_quadrature_nodes=8,
            gumbel_lookup_size=2_001,
        )

    def _replace(self, **changes: object) -> contamination.ContaminationConfig:
        return dataclasses.replace(self.config, **changes)

    def test_method_names_and_grouping(self) -> None:
        self.assertEqual(
            contamination.point_mass_method(0.25), "bayes_shared_rho0_0.25"
        )
        self.assertEqual(contamination.point_mass_method(0.4), "bayes_shared_rho0_0.4")
        self.assertEqual(contamination.point_mass_method(0.15), "bayes_shared_rho0_0.15")
        self.assertTrue(contamination.is_point_mass_method("bayes_shared_rho0_0.1"))
        self.assertFalse(contamination.is_point_mass_method("bayes_shared_robust"))
        self.assertFalse(contamination.is_point_mass_method("bayes_shared_clean"))
        self.assertEqual(
            contamination._method_group("bayes_shared_rho0_0.1"),
            "bayes_shared_point_mass_diagnostic",
        )
        # Existing groupings must not move.
        self.assertEqual(
            contamination._method_group("bayes_shared_robust"), "bayes_shared"
        )
        self.assertEqual(
            contamination._method_group("bayes_shared_clean"), "bayes_shared"
        )
        self.assertEqual(
            contamination._method_group("bayes_tokenwise_robust"),
            "bayes_tokenwise_sensitivity",
        )
        self.assertEqual(contamination._method_group("h_dif_star_0.01"), "paper_score")

    def test_config_validation_rejects_bad_point_masses(self) -> None:
        for bad in ((1.0,), (-0.1,), (0.2, 0.2)):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self._replace(rho_point_masses=bad).validate()
        self._replace(rho_point_masses=()).validate()

    def test_point_mass_at_zero_reproduces_the_clean_rule(self) -> None:
        config = self._replace(rho_point_masses=(0.0,))
        for scheme, simulator in (
            ("gumbel", paper.simulate_gumbel_null),
            ("inverse", paper.simulate_inverse_null),
        ):
            with self.subTest(scheme=scheme):
                pivots = simulator(
                    np.random.default_rng(31), 7, config.max_horizon, config.paper_config()
                )
                paths = contamination.shared_bayes_selected_paths(pivots, scheme, config)
                np.testing.assert_array_equal(
                    paths["bayes_shared_rho0_0"], paths["bayes_shared_clean"]
                )

    def test_point_mass_matches_reference_detector(self) -> None:
        deltas, delta_weights = contamination._delta_quadrature(self.config)
        rho_zero = 0.1
        method = contamination.point_mass_method(rho_zero)
        for scheme, simulator in (
            ("gumbel", paper.simulate_gumbel_null),
            ("inverse", paper.simulate_inverse_null),
        ):
            with self.subTest(scheme=scheme):
                pivots = simulator(
                    np.random.default_rng(32),
                    3,
                    self.config.max_horizon,
                    self.config.paper_config(),
                )
                paths = contamination.shared_bayes_selected_paths(
                    pivots, scheme, self.config
                )[method]
                for row_index, sequence in enumerate(pivots):
                    detector = SequentialBayesDetector(
                        scheme,
                        deltas,
                        delta_weights,
                        rho_grid=(rho_zero,),
                        rho_weights=(1.0,),
                        structure="shared",
                        gumbel_family="spike",
                        vocabulary_size=self.config.vocabulary_size,
                        inverse_null_vocabulary_size=(
                            self.config.vocabulary_size if scheme == "inverse" else None
                        ),
                    )
                    detector.update_many(sequence)
                    expected = detector.log_bayes_factor_history[
                        np.asarray(self.config.horizons) - 1
                    ]
                    np.testing.assert_allclose(
                        paths[row_index], expected, atol=1e-11, rtol=1e-11
                    )

    def test_point_mass_is_independent_of_whether_it_sits_on_the_prior_grid(
        self,
    ) -> None:
        """rho_0=0.25 must give the same path whether or not the prior grid has it."""

        on_grid = self._replace(
            rho_prior_grid=(0.0, 0.25), rho_prior_weights=(0.6, 0.4)
        )
        off_grid = self._replace(
            rho_prior_grid=(0.0, 0.3), rho_prior_weights=(0.6, 0.4)
        )
        for scheme, simulator in (
            ("gumbel", paper.simulate_gumbel_null),
            ("inverse", paper.simulate_inverse_null),
        ):
            with self.subTest(scheme=scheme):
                pivots = simulator(
                    np.random.default_rng(33),
                    5,
                    on_grid.max_horizon,
                    on_grid.paper_config(),
                )
                a = contamination.shared_bayes_selected_paths(pivots, scheme, on_grid)
                b = contamination.shared_bayes_selected_paths(pivots, scheme, off_grid)
                np.testing.assert_array_equal(
                    a["bayes_shared_rho0_0.25"], b["bayes_shared_rho0_0.25"]
                )

    def test_adding_point_masses_leaves_existing_paths_bitwise_unchanged(self) -> None:
        without = self._replace(rho_point_masses=())
        with_masses = self._replace(rho_point_masses=(0.1, 0.15, 0.25, 0.4))
        for scheme, simulator in (
            ("gumbel", paper.simulate_gumbel_null),
            ("inverse", paper.simulate_inverse_null),
        ):
            with self.subTest(scheme=scheme):
                pivots = simulator(
                    np.random.default_rng(34),
                    11,
                    without.max_horizon,
                    without.paper_config(),
                )
                base = contamination.shared_bayes_selected_paths(
                    pivots, scheme, without
                )
                extended = contamination.shared_bayes_selected_paths(
                    pivots, scheme, with_masses
                )
                self.assertEqual(
                    set(base), {"bayes_shared_clean", "bayes_shared_robust"}
                )
                for method in base:
                    np.testing.assert_array_equal(base[method], extended[method])

    def test_mixture_prior_is_the_weighted_average_of_its_point_masses(self) -> None:
        """BF for pi_rho = sum_k w_k delta_{rho_k} equals sum_k w_k BF_{rho_k}.

        This is the identity the whole diagnostic rests on: the robust rule is
        exactly a w-weighted average, on the Bayes-factor scale, of the very
        point-mass rules the diagnostic adds, so any advantage it has over all
        of them must come from the averaging itself.
        """

        grid = (0.0, 0.1, 0.25, 0.4)
        weights = (0.4, 0.2, 0.2, 0.2)
        config = self._replace(
            rho_prior_grid=grid, rho_prior_weights=weights, rho_point_masses=grid
        )
        for scheme, simulator in (
            ("gumbel", paper.simulate_gumbel_alternative_shared_delta),
            ("inverse", paper.simulate_inverse_alternative_shared_delta),
        ):
            with self.subTest(scheme=scheme):
                pivots = simulator(
                    np.random.default_rng(35),
                    9,
                    config.max_horizon,
                    config.paper_config(),
                )
                paths = contamination.shared_bayes_selected_paths(
                    pivots, scheme, config
                )
                stacked = np.stack(
                    [paths[contamination.point_mass_method(v)] for v in grid], axis=-1
                )
                expected = logsumexp(stacked + np.log(weights), axis=-1)
                np.testing.assert_allclose(
                    paths["bayes_shared_robust"], expected, atol=1e-10, rtol=1e-10
                )

    def test_point_mass_near_one_sends_the_log_bayes_factor_to_zero(self) -> None:
        """At rho_0 -> 1 the alternative collapses onto the null, so log BF -> 0."""

        config = self._replace(rho_point_masses=(0.1, 0.999999))
        pivots = paper.simulate_gumbel_alternative_shared_delta(
            np.random.default_rng(36), 9, config.max_horizon, config.paper_config()
        )
        paths = contamination.shared_bayes_selected_paths(pivots, "gumbel", config)
        near_one = paths["bayes_shared_rho0_0.999999"]
        self.assertTrue(np.all(np.abs(near_one) < 1e-2))
        self.assertLess(
            float(np.max(np.abs(near_one))),
            float(np.max(np.abs(paths["bayes_shared_rho0_0.1"]))),
        )

    def test_indicator_sink_matches_reported_rates_and_leaves_rows_untouched(
        self,
    ) -> None:
        rows_a, _ = contamination.run_contamination_benchmark(self.config)
        sink: dict[str, np.ndarray] = {}
        rows_b, metadata = contamination.run_contamination_benchmark(
            self.config, indicator_sink=sink
        )
        self.assertEqual(rows_a, rows_b)

        cells = len(self.config.rho_true_grid) * len(self.config.horizons)
        expected_keys = (
            2
            * cells
            * (
                len(contamination.BAYES_METHODS)
                + len(self.config.point_mass_methods())
            )
            + cells
            * (
                len(contamination.GUMBEL_PAPER_METHODS)
                + len(contamination.INVERSE_PAPER_METHODS)
            )
            # Gumbel only, so these are counted once rather than twice.
            + cells * len(contamination.DIRICHLET_METHODS)
            + cells * len(contamination.UNIONTAIL_METHODS)
            # Tr-GoF runs for both schemes, so it is counted twice like the
            # other two-scheme rules.
            + 2 * cells
        )
        self.assertEqual(len(sink), expected_keys)
        for value in sink.values():
            self.assertEqual(value.dtype, np.dtype(bool))
            self.assertEqual(value.size, self.config.n_evaluation_alternative)

        indicator_metadata = metadata["per_document_indicators"]
        self.assertEqual(
            indicator_metadata["indicator_rule_version"],
            contamination.BOUNDARY_RANDOMIZATION_RULE_VERSION,
        )
        self.assertIn(
            "same auxiliary U_i",
            indicator_metadata["boundary_randomization"]["coupling"],
        )
        ties = {
            (
                record["scheme"],
                float(record["rho_true"]),
                int(record["horizon"]),
                record["method"],
            ): record
            for record in indicator_metadata[
                "tie_and_randomized_rejection_counts"
            ]
        }
        self.assertEqual(len(ties), expected_keys)
        seed_entropies: dict[tuple[str, float, int], set[tuple[int, ...]]] = {}
        for row in rows_b:
            key = (
                row["scheme"],
                float(row["rho_true"]),
                int(row["horizon"]),
                row["method"],
            )
            record = ties[key]
            n_documents = int(record["n_documents"])
            expected_power = (
                int(record["n_rejected_strict"])
                + float(record["boundary_randomization_probability"])
                * int(record["n_at_atom"])
            ) / n_documents
            self.assertAlmostEqual(
                expected_power,
                float(row["power"]),
                places=12,
            )
            indicator = contamination.indicator_key(*key)
            self.assertEqual(
                int(np.count_nonzero(sink[indicator])),
                int(record["n_rejected_randomized"]),
            )
            self.assertAlmostEqual(
                float(1.0 - sink[indicator].mean()),
                float(record["type_ii_error_randomized_realization"]),
                places=12,
            )
            if int(record["n_at_atom"]) == 0:
                self.assertAlmostEqual(
                    float(record["type_ii_error_randomized_realization"]),
                    float(row["type_ii_error"]),
                    places=12,
                )
            cell = (str(key[0]), float(key[1]), int(key[2]))
            seed_entropies.setdefault(cell, set()).add(
                tuple(int(value) for value in record["auxiliary_seed_entropy"])
            )
        # All methods in a cell share one auxiliary stream, while rho_true is
        # itself part of that stream's seed address.
        self.assertTrue(all(len(values) == 1 for values in seed_entropies.values()))
        self.assertEqual(
            indicator_metadata["total_documents_at_atom"],
            sum(int(record["n_at_atom"]) for record in ties.values()),
        )
        self.assertEqual(
            indicator_metadata["total_rejections_randomized_at_atom"],
            sum(int(record["n_rejected_at_atom"]) for record in ties.values()),
        )

    def test_indicator_key_is_stable_and_round_trips(self) -> None:
        self.assertEqual(
            contamination.indicator_key("inverse", 0.25, 700, "bayes_shared_robust"),
            "inverse|0.25|700|bayes_shared_robust",
        )
        self.assertEqual(
            contamination.indicator_key("gumbel", 0.0, 100, "bayes_shared_rho0_0.4"),
            "gumbel|0|100|bayes_shared_rho0_0.4",
        )

    def test_headline_rho_one_diagnostic_ignores_point_masses(self) -> None:
        with_masses, metadata_with = contamination.run_contamination_benchmark(
            self.config
        )
        without = self._replace(rho_point_masses=())
        _, metadata_without = contamination.run_contamination_benchmark(without)
        self.assertEqual(
            metadata_with["rho_one_diagnostic_max_abs_power_minus_type_i"],
            metadata_without["rho_one_diagnostic_max_abs_power_minus_type_i"],
        )
        self.assertEqual(
            metadata_with["rho_one_diagnostic_cell"],
            metadata_without["rho_one_diagnostic_cell"],
        )
        self.assertFalse(
            contamination.is_point_mass_method(
                str(metadata_with["rho_one_diagnostic_cell"]["method"])
            )
        )
        self.assertIsNotNone(
            metadata_with["rho_point_mass_diagnostic"][
                "rho_one_max_abs_power_minus_type_i"
            ]
        )
        self.assertIsNone(
            metadata_without["rho_point_mass_diagnostic"][
                "rho_one_max_abs_power_minus_type_i"
            ]
        )

    def test_existing_method_rows_are_unchanged_by_the_new_detectors(self) -> None:
        with_masses, _ = contamination.run_contamination_benchmark(self.config)
        without, _ = contamination.run_contamination_benchmark(
            self._replace(rho_point_masses=())
        )
        kept = [
            row
            for row in with_masses
            if not contamination.is_point_mass_method(str(row["method"]))
        ]
        self.assertEqual(kept, without)
        self.assertGreater(len(with_masses), len(without))


class RhoDiagnosticPayloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = contamination.ContaminationConfig(
            vocabulary_size=40,
            horizons=(6, 12),
            rho_true_grid=(0.0, 0.25, 0.4, 1.0),
            rho_prior_grid=(0.0, 0.25),
            rho_prior_weights=(0.6, 0.4),
            rho_point_masses=(0.1, 0.25),
            n_calibration=40,
            n_evaluation_null=25,
            n_evaluation_alternative=25,
            batch_size=10,
            bayes_quadrature_nodes=8,
            gumbel_lookup_size=2_001,
        )
        self.sink: dict[str, np.ndarray] = {}
        self.rows, self.metadata = contamination.run_contamination_benchmark(
            self.config, indicator_sink=self.sink
        )

    def test_payload_shape_and_consistency(self) -> None:
        payload = contamination.build_rho_diagnostic(
            self.rows, self.metadata, self.config, self.sink, mcnemar_rhos=(0.25, 0.4)
        )
        self.assertEqual(
            payload["point_mass_methods"], list(self.config.point_mass_methods())
        )
        self.assertEqual(
            len(payload["type_ii_tables"]), 2 * len(self.config.horizons)
        )
        for table in payload["type_ii_tables"]:
            paper_method = contamination.PRESPECIFIED_PAPER_SCORE[table["scheme"]]
            methods = [row["method"] for row in table["rows"]]
            self.assertIn("bayes_shared_robust", methods)
            self.assertIn("bayes_shared_clean", methods)
            self.assertIn(paper_method, methods)
            for method in self.config.point_mass_methods():
                self.assertIn(method, methods)
            for row in table["rows"]:
                self.assertEqual(
                    len(row["type_ii_error"]), len(self.config.rho_true_grid)
                )
                for value, rho in zip(row["type_ii_error"], table["rho_true_grid"]):
                    expected = contamination._find(
                        self.rows,
                        table["scheme"],
                        row["method"],
                        table["horizon"],
                        rho,
                    )["type_ii_error"]
                    self.assertAlmostEqual(value, float(expected), places=12)

    def test_oracle_reading_is_a_lower_bound_on_the_single_fixed_reading(self) -> None:
        payload = contamination.build_rho_diagnostic(
            self.rows, self.metadata, self.config, self.sink, mcnemar_rhos=(0.25, 0.4)
        )
        oracle = {
            (r["scheme"], r["horizon"], r["rho_true"]): r
            for r in payload["oracle_reading"]["records"]
        }
        for record in payload["single_fixed_reading"]["records"]:
            for rho_text, gap in record[
                "robust_minus_best_single_fixed_by_rho"
            ].items():
                key = (record["scheme"], record["horizon"], float(rho_text))
                # The hindsight-chosen point mass can only look better, so the
                # robust rule's deficit against it is at least as large.
                self.assertGreaterEqual(
                    oracle[key]["robust_minus_oracle_point_mass"] + 1e-12, gap
                )
            self.assertIn(record["best_by_mean"], self.config.point_mass_methods())
            self.assertIn(record["best_by_max"], self.config.point_mass_methods())

    def test_mcnemar_counts_agree_with_the_indicator_arrays(self) -> None:
        payload = contamination.build_rho_diagnostic(
            self.rows, self.metadata, self.config, self.sink, mcnemar_rhos=(0.25, 0.4)
        )
        records = payload["mcnemar_robust_vs_point_mass"]
        self.assertEqual(
            len(records), 2 * 2 * len(self.config.point_mass_methods())
        )
        horizon = max(self.config.horizons)
        for record in records:
            robust = self.sink[
                contamination.indicator_key(
                    record["scheme"],
                    record["rho_true"],
                    horizon,
                    "bayes_shared_robust",
                )
            ]
            other = self.sink[
                contamination.indicator_key(
                    record["scheme"],
                    record["rho_true"],
                    horizon,
                    record["comparator"],
                )
            ]
            b = int(np.count_nonzero(~robust & other))
            c = int(np.count_nonzero(robust & ~other))
            self.assertEqual(record["b_robust_only_miss"], b)
            self.assertEqual(record["c_point_mass_only_miss"], c)
            self.assertEqual(record["discordant_total"], b + c)
            self.assertAlmostEqual(
                record["robust_type_ii_randomized_realization"],
                float(np.mean(~robust)),
                places=12,
            )
            self.assertAlmostEqual(
                record["comparator_type_ii_randomized_realization"],
                float(np.mean(~other)),
                places=12,
            )
            self.assertAlmostEqual(
                record["robust_type_ii_rao_blackwellized"],
                float(
                    contamination._find(
                        self.rows,
                        record["scheme"],
                        "bayes_shared_robust",
                        horizon,
                        record["rho_true"],
                    )["type_ii_error"]
                ),
                places=12,
            )
            self.assertAlmostEqual(
                record["comparator_type_ii_rao_blackwellized"],
                float(
                    contamination._find(
                        self.rows,
                        record["scheme"],
                        record["comparator"],
                        horizon,
                        record["rho_true"],
                    )["type_ii_error"]
                ),
                places=12,
            )
            expected_p = paired.exact_mcnemar_p_value(b, c)
            if b + c == 0:
                self.assertEqual(expected_p, 1.0)
                self.assertEqual(record["p_value"], 1.0)
                # The directional effect is not estimable, but p=1 is retained
                # as the conservative exact-test bookkeeping value.
                self.assertEqual(
                    record["p_value_status"], "no_discordant_pairs_p_equals_one"
                )
            else:
                self.assertIsNotNone(record["p_value"])
                self.assertEqual(record["p_value_status"], "defined")
                self.assertAlmostEqual(record["p_value"], expected_p, places=12)
                self.assertGreater(record["p_value"], 0.0)
                self.assertLessEqual(record["p_value"], 1.0)
            self.assertEqual(record["reference"], "bayes_shared_robust")
            self.assertTrue(contamination.is_point_mass_method(record["comparator"]))
        self.assertIn("one randomized realization", payload["mcnemar_note"])
        self.assertNotIn("deterministic rule", payload["mcnemar_note"])
        for scheme in ("gumbel", "inverse"):
            for rho in (0.25, 0.4):
                flagged = [
                    r
                    for r in records
                    if r["scheme"] == scheme
                    and r["rho_true"] == rho
                    and r["comparator_is_best_point_mass_at_this_rho"]
                ]
                self.assertEqual(len(flagged), 1)

    def test_family_readings_split_requested_grid_from_the_full_grid(self) -> None:
        """A wider point-mass grid can only strengthen the oracle comparator."""

        config = dataclasses.replace(
            self.config, rho_point_masses=(0.1, 0.15, 0.25, 0.4, 0.6)
        )
        rows, metadata = contamination.run_contamination_benchmark(config)
        payload = contamination.build_rho_diagnostic(rows, metadata, config, None)
        families = payload["readings_by_point_mass_family"]["families"]
        self.assertEqual(set(families), {"all_point_masses", "requested_grid_only"})
        self.assertEqual(
            families["requested_grid_only"]["point_mass_methods"],
            [
                contamination.point_mass_method(v)
                for v in contamination.REQUESTED_RHO_POINT_MASSES
            ],
        )
        self.assertEqual(
            families["all_point_masses"]["point_mass_methods"],
            list(config.point_mass_methods()),
        )
        wide = {
            (r["scheme"], r["horizon"], r["rho_true"]): r
            for r in families["all_point_masses"]["oracle_records"]
        }
        narrow = {
            (r["scheme"], r["horizon"], r["rho_true"]): r
            for r in families["requested_grid_only"]["oracle_records"]
        }
        self.assertEqual(set(wide), set(narrow))
        for key, wide_record in wide.items():
            self.assertLessEqual(
                wide_record["best_point_mass_type_ii"],
                narrow[key]["best_point_mass_type_ii"] + 1e-12,
            )
            self.assertGreaterEqual(
                narrow[key]["robust_minus_oracle_point_mass"] + 1e-12,
                wide_record["robust_minus_oracle_point_mass"],
            )

    def test_no_family_split_when_the_grid_is_exactly_the_requested_one(self) -> None:
        config = dataclasses.replace(
            self.config, rho_point_masses=contamination.REQUESTED_RHO_POINT_MASSES
        )
        rows, metadata = contamination.run_contamination_benchmark(config)
        payload = contamination.build_rho_diagnostic(rows, metadata, config, None)
        self.assertEqual(
            set(payload["readings_by_point_mass_family"]["families"]),
            {"all_point_masses"},
        )

    def test_payload_omits_mcnemar_without_indicators(self) -> None:
        payload = contamination.build_rho_diagnostic(
            self.rows, self.metadata, self.config, None
        )
        self.assertNotIn("mcnemar_robust_vs_point_mass", payload)

    def test_payload_is_json_serialisable(self) -> None:
        payload = contamination.build_rho_diagnostic(
            self.rows, self.metadata, self.config, self.sink, mcnemar_rhos=(0.25,)
        )
        text = json.dumps(payload)
        self.assertEqual(json.loads(text)["point_masses"], [0.1, 0.25])


class CommandLineSafetyTests(unittest.TestCase):
    def test_default_full_run_uses_paper_results_directory(self) -> None:
        args = contamination.parse_args([])
        self.assertEqual(
            contamination.resolve_output_dir(args.output_dir, quick=args.quick),
            contamination.DEFAULT_RESULTS_DIR,
        )

    def test_default_quick_run_uses_isolated_directory(self) -> None:
        args = contamination.parse_args(["--quick"])
        self.assertEqual(
            contamination.resolve_output_dir(args.output_dir, quick=args.quick),
            contamination.DEFAULT_QUICK_RESULTS_DIR,
        )
        self.assertNotEqual(
            contamination.DEFAULT_QUICK_RESULTS_DIR,
            contamination.DEFAULT_RESULTS_DIR,
        )

    def test_explicit_directory_wins_for_quick_run(self) -> None:
        args = contamination.parse_args(
            ["--quick", "--output-dir", "chosen"]
        )
        self.assertEqual(
            contamination.resolve_output_dir(args.output_dir, quick=args.quick),
            Path("chosen"),
        )



class UnionTailRuleTests(unittest.TestCase):
    def test_both_variants_exist_and_are_gumbel_only(self) -> None:
        self.assertEqual(
            set(contamination.UNIONTAIL_METHODS),
            {"bayes_shared_uniontail_clean", "bayes_shared_uniontail_robust"},
        )
        self.assertEqual(contamination.UNIONTAIL_SCHEMES, ("gumbel",))

    def test_robust_variant_reuses_the_frozen_rho_prior(self) -> None:
        # The tail-width robust rule must integrate exactly the same rho prior
        # as the Dirichlet robust rule, so the two differ only in the tail model.
        config = contamination.ContaminationConfig()
        tail = config.uniontail_grid()
        np.testing.assert_array_equal(tail.rhos, np.asarray(config.rho_prior_grid))
        np.testing.assert_allclose(
            tail.rho_weights,
            np.asarray(config.rho_prior_weights) / sum(config.rho_prior_weights),
        )

    def test_full_width_atom_lives_in_the_dirichlet_block(self) -> None:
        # The union splits the tail prior: the full-width atom belongs to the
        # Dirichlet block, so the tail-width ladder must stop below it.
        grid = contamination.ContaminationConfig().uniontail_grid()
        self.assertTrue(all(j < grid.tail_size for j in grid.tail_widths))
        self.assertEqual(grid.dirichlet_block.tail_size, grid.tail_size)
        self.assertAlmostEqual(sum(grid.block_weights), 1.0)
