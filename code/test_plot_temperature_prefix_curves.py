"""Tests for the temperature-matched prefix-curve command.

These cover the pure computation: that the all-prefix score matrices agree
with the single-horizon calls the four-horizon analysis uses, that calibration
never sees the documents it scores, and that the repetition diagnostic counts a
pivot at the prefix where it first appears.  The figures are exercised only far
enough to confirm they render without a display.
"""

import csv
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

import dirichlet_detector as dd
import plot_temperature_prefix_curves as ptc
import trgof


class ConfiguredWeightAndLoaderTests(unittest.TestCase):
    def test_single_weight_computes_only_requested_union_and_preserves_default_menu(self):
        self.assertEqual(ptc.computed_rules(), ptc.COMPUTED_RULES)
        pivots = np.linspace(.1, .9, 24).reshape(12, 2)
        for weight in (.2, .25, .5, 0., 1.):
            rules = ptc.computed_rules(weight)
            unions = [rule for rule in rules if rule.startswith("bayes_uniontail_")]
            self.assertEqual(unions, [ptc.union_rule_name(weight)])
            grids = ptc.build_grids(64, weight)
            with patch.object(ptc, "SPLIT_REPLICATES", 2):
                rows = ptc.curve_rows("m", ".3", pivots, pivots[::-1], grids, rules=rules)
            self.assertEqual({row["rule"] for row in rows}, set(rules))
            order, labels, colors, dashes = ptc._style_for("gumbel", weight)
            self.assertIn(ptc.union_rule_name(weight), order)
            for rule in order:
                self.assertIn(rule, labels)
                self.assertIn(rule, colors)
                self.assertIn(rule, dashes)

    def test_weight_and_scheme_validation_is_early(self):
        for weight in (float("nan"), float("inf"), -.1, 1.1):
            with self.subTest(weight=weight), self.assertRaises(ValueError):
                ptc.computed_rules(weight)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            ptc.main(["--scheme", "inverse", "--union-weight", ".5"])

    def test_collect_uses_paired_loader_and_retains_provenance(self):
        shape = (12, 2)
        pivots = np.full(shape, .5)
        cell = SimpleNamespace(temperature="0.3", watermarked={"Y": pivots},
                               raw={"tokens": np.ones(shape, dtype=int)},
                               prompts=np.ones((12, 50), dtype=int),
                               provenance=[{"mode": "fixture", "path": "fixture.npz"}])
        provenance = {}
        with patch.object(ptc.matched, "load_cells", return_value=[cell]) as load, \
                patch.object(ptc, "build_grids", return_value={}), \
                patch.object(ptc, "curve_rows", return_value=[]) as curves, \
                patch("wrong_key_null.replay_with_key", return_value=(pivots, pivots)), \
                contextlib.redirect_stdout(io.StringIO()):
            ptc.collect(["1p3B"], Path("fixture"), union_weight=.25,
                        allow_legacy=True, provenance=provenance)
        load.assert_called_once_with(Path("fixture"), "1p3B", scheme="gumbel", allow_legacy=True)
        self.assertEqual(curves.call_args.kwargs["rules"], ptc.computed_rules(.25))
        self.assertEqual(provenance["1p3B_T0.3"], cell.provenance)

    def test_collect_rejects_duplicate_or_unknown_model_names(self):
        for models in ([], ["1p3B", "1p3B"], ["other"]):
            with self.subTest(models=models), self.assertRaises(ValueError):
                ptc.collect(models, Path("unused"))

    def test_single_weight_cli_renders_selected_rule_and_reports_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [{"model": "1p3B", "temperature": ".3", "rule": rule,
                     "prefix": prefix, "type2": .5}
                    for rule in ptc.computed_rules(.25) for prefix in (1, 2)]
            csv_path = root / "curves.csv"
            ptc.write_csv(rows, csv_path)
            with contextlib.redirect_stdout(io.StringIO()):
                result = ptc.main(["--union-weight", ".25", "--curves-csv", str(csv_path),
                                   "--output-dir", str(root), "--summary-horizon", "2"])
            self.assertEqual(result, 0)
            summary = json.loads((root / "temperature_prefix_curves_summary_w0p25.json").read_text())
            self.assertIn("bayes_uniontail_w0.25", summary["rules"])
            self.assertNotIn("bayes_uniontail_w0.5", summary["rules"])
            self.assertFalse(summary["input_provenance"]["generation_manifest_validated"])
            self.assertTrue((root / "prefix_curves_matched_1p3B_w0p25.pdf").is_file())

    def test_csv_cannot_silently_supply_wrong_union_weight(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            csv_path = root / "curves.csv"
            ptc.write_csv([{"model": "1p3B", "temperature": ".3", "rule": rule,
                            "prefix": 1, "type2": .5} for rule in ptc.computed_rules(.5)], csv_path)
            with self.assertRaisesRegex(ValueError, "configured plotted rules"):
                ptc.main(["--union-weight", ".25", "--curves-csv", str(csv_path),
                          "--output-dir", str(root)])


class MeanDistinctByPrefixTests(unittest.TestCase):
    def test_counts_first_occurrence_only(self):
        pivots = np.array([[0.1, 0.2, 0.1, 0.3], [0.5, 0.5, 0.5, 0.5]])
        np.testing.assert_allclose(
            ptc.mean_distinct_by_prefix(pivots), [1.0, 1.5, 1.5, 2.0]
        )

    def test_all_distinct_grows_one_per_token(self):
        pivots = np.linspace(0.01, 0.99, 12).reshape(1, 12)
        np.testing.assert_allclose(
            ptc.mean_distinct_by_prefix(pivots), np.arange(1, 13, dtype=float)
        )

    def test_is_nondecreasing_and_bounded_by_prefix(self):
        rng = np.random.default_rng(3)
        pivots = rng.choice([0.2, 0.4, 0.6], size=(20, 30)).astype(float)
        counts = ptc.mean_distinct_by_prefix(pivots)
        self.assertTrue(np.all(np.diff(counts) >= -1e-12))
        self.assertTrue(np.all(counts <= np.arange(1, 31)))
        # only three values exist, so the count can never exceed three
        self.assertLessEqual(counts[-1], 3.0)


class ScorePathTests(unittest.TestCase):
    """Every prefix of the matrix must equal the score computed at that horizon."""

    @classmethod
    def setUpClass(cls):
        cls.vocab = 64
        cls.grids = ptc.build_grids(cls.vocab)
        rng = np.random.default_rng(11)
        cls.pivots = rng.random((16, 9))

    def test_cumulative_rules_match_direct_sums(self):
        paths = ptc.score_paths("h_ars", self.pivots, self.grids)
        for horizon in (1, 4, 9):
            direct = (-np.log1p(-np.clip(self.pivots[:, :horizon], 0, 1 - 1e-12))).sum(1)
            np.testing.assert_allclose(paths[:, horizon - 1], direct)

    def test_trgof_matches_per_horizon_call(self):
        paths = ptc.score_paths("trgof_s2", self.pivots, self.grids)
        for horizon in (2, 5, 9):
            direct = trgof.statistic(trgof.gumbel_p_values(self.pivots[:, :horizon]))
            np.testing.assert_allclose(paths[:, horizon - 1], direct)

    def test_bayes_rules_match_single_horizon_call(self):
        delta, weights = dd.gauss_legendre_delta_grid(ptc.DELTA_LOW, ptc.DELTA_HIGH, 96)
        expected = {
            ptc.union_rule_name(0.5): dd.UnionTailBayesGrid(
                delta_grid=delta, delta_weights=weights, tail_size=self.vocab - 1,
                dirichlet_weight=0.5,
            ),
            "bayes_shared": dd.DirichletBayesGrid(
                delta_grid=delta, delta_weights=weights,
                alpha_grid=(float("inf"),), tail_size=self.vocab - 1,
            ),
        }
        clipped = np.clip(self.pivots, 1e-12, 1 - 1e-12)
        for rule, grid in expected.items():
            paths = ptc.score_paths(rule, self.pivots, self.grids)
            for horizon in (3, 9):
                direct = grid.shared_paths(clipped[:, :horizon], horizons=(horizon,))[0][:, 0]
                np.testing.assert_allclose(paths[:, horizon - 1], direct, atol=1e-12)

    def test_every_rule_returns_one_column_per_prefix(self):
        for rule in ptc.COMPUTED_RULES:
            paths = ptc.score_paths(rule, self.pivots, self.grids)
            self.assertEqual(paths.shape, self.pivots.shape, rule)


class CurveRowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(5)
        cls.grids = ptc.build_grids(64)
        cls.null = rng.random((40, 6))
        # a visibly watermarked arm: pivots pushed toward one
        cls.watermarked = rng.beta(4.0, 1.0, size=(30, 6))
        cls.rows = ptc.curve_rows(
            "1p3B", "0.3", cls.watermarked, cls.null, cls.grids
        )

    def test_one_row_per_rule_and_prefix(self):
        self.assertEqual(len(self.rows), len(ptc.COMPUTED_RULES) * 6)
        self.assertEqual(
            {row["prefix"] for row in self.rows}, set(range(1, 7))
        )

    def test_calibration_uses_the_whole_null_arm(self):
        """Size matching across rules requires one common cutoff basis."""
        for row in self.rows:
            self.assertEqual(row["n_calibration"], 40)
            self.assertEqual(row["n_type1"], 40)
            self.assertEqual(row["n_watermarked"], 30)

    def test_cutoff_never_sees_a_watermarked_document(self):
        """The cutoff is built from the null arm only, so Type II is honest."""
        grids = ptc.build_grids(64)
        rows_a = ptc.curve_rows("m", "0.3", self.watermarked, self.null, grids)
        shuffled = self.watermarked[::-1].copy()
        rows_b = ptc.curve_rows("m", "0.3", shuffled, self.null, grids)
        # permuting the watermarked documents cannot move the cutoff, so the
        # aggregate Type II must be identical
        a = [r["type2"] for r in rows_a]
        b = [r["type2"] for r in rows_b]
        self.assertEqual(a, b)

    def test_split_noise_is_recorded(self):
        """The old design's spread must be visible, not hidden."""
        for row in self.rows:
            self.assertGreaterEqual(row["type2_split_sd"], 0.0)
            self.assertGreaterEqual(row["type2_split_half"], 0.0)
            self.assertLessEqual(row["type2_split_half"], 1.0)

    def test_rates_are_probabilities_and_mcse_consistent(self):
        for row in self.rows:
            for field in ("type1", "type2"):
                self.assertGreaterEqual(row[field], 0.0)
                self.assertLessEqual(row[field], 1.0)
            rate, n = row["type2"], row["n_watermarked"]
            self.assertAlmostEqual(
                row["type2_mcse"], float(np.sqrt(rate * (1 - rate) / n)), places=12
            )

    def test_detects_a_strongly_watermarked_arm(self):
        final = [
            row for row in self.rows
            if row["rule"] == "h_ars" and row["prefix"] == 6
        ]
        self.assertLess(final[0]["type2"], 0.5)

    def test_rows_carry_the_repetition_diagnostic(self):
        for row in self.rows:
            self.assertGreaterEqual(row["distinct_watermarked"], 1.0)
            self.assertLessEqual(row["distinct_watermarked"], row["prefix"])


class UnionWeightTests(unittest.TestCase):
    """The two endpoints of the weight sweep must be the parent priors exactly."""

    @classmethod
    def setUpClass(cls):
        cls.vocab = 64
        cls.grids = ptc.build_grids(cls.vocab)
        cls.pivots = np.random.default_rng(31).random((12, 8))
        cls.horizons = tuple(range(1, 9))
        cls.delta, cls.weights = dd.gauss_legendre_delta_grid(ptc.DELTA_LOW, ptc.DELTA_HIGH, 96)

    def test_every_weight_has_a_grid(self):
        for w in ptc.UNION_WEIGHTS:
            self.assertIn(ptc.union_rule_name(w), self.grids)
            self.assertIn(ptc.union_rule_name(w), ptc.COMPUTED_RULES)

    def test_weight_one_is_the_dirichlet_shape_prior(self):
        import benchmark_paper_experiment as benchmark
        shape = dd.DirichletBayesGrid(
            delta_grid=self.delta, delta_weights=self.weights,
            alpha_grid=benchmark.DIRICHLET_ALPHA_GRID, tail_size=self.vocab - 1,
            allow_large_delta=True,
        )
        got = ptc.score_paths(ptc.union_rule_name(1.0), self.pivots, self.grids)
        want = shape.shared_paths(
            np.clip(self.pivots, 1e-12, 1 - 1e-12), horizons=self.horizons
        )[0]
        np.testing.assert_allclose(got, want, atol=1e-12)

    def test_weight_zero_is_the_width_prior_below_full_width(self):
        """The union's width block excludes J=K, which is the alpha=inf case."""
        ladder = tuple(j for j in dd.dyadic_tail_width_grid(self.vocab - 1)
                       if j < self.vocab - 1)
        width = dd.TailWidthBayesGrid(
            delta_grid=self.delta, delta_weights=self.weights,
            tail_size=self.vocab - 1, tail_width_grid=ladder,
            allow_large_delta=True,
        )
        got = ptc.score_paths(ptc.union_rule_name(0.0), self.pivots, self.grids)
        want = width.shared_paths(
            np.clip(self.pivots, 1e-12, 1 - 1e-12), horizons=self.horizons
        )[0]
        np.testing.assert_allclose(got, want, atol=1e-12)

    def test_interior_weights_lie_between_the_endpoints(self):
        """A mixture cannot exceed both parents at a common observation."""
        lo = ptc.score_paths(ptc.union_rule_name(0.0), self.pivots, self.grids)
        hi = ptc.score_paths(ptc.union_rule_name(1.0), self.pivots, self.grids)
        for w in ptc.UNION_WEIGHTS:
            if w in (0.0, 1.0):
                continue
            mid = ptc.score_paths(ptc.union_rule_name(w), self.pivots, self.grids)
            self.assertTrue(np.all(mid <= np.maximum(lo, hi) + 1e-9))
            self.assertTrue(np.all(mid >= np.minimum(lo, hi) - 1e-9))


class ReferenceMenuTests(unittest.TestCase):
    """Every Gumbel reference score of Li et al. must be computed, not just the
    ones the figure has room for."""

    def test_full_li_et_al_gumbel_menu_is_present(self):
        import benchmark_paper_experiment as benchmark
        expected = {
            m for m in benchmark.GUMBEL_METHODS if m.startswith(("h_", "h*"))
            and not m.startswith("h_spike")
        } | {"h_gum_star_0.1"}
        missing = expected - set(ptc.COMPUTED_RULES)
        self.assertEqual(missing, set(), f"reference scores dropped: {missing}")

    def test_optimal_class_matches_the_benchmark_definition(self):
        import benchmark_paper_experiment as benchmark
        rng = np.random.default_rng(19)
        pivots = rng.random((6, 5))
        grids = ptc.build_grids(64)
        for rule, delta in (("h_gum_star_0.1", 0.1), ("h_gum_star_0.01", 0.01),
                            ("h_gum_star_0.005", 0.005)):
            paths = ptc.score_paths(rule, pivots, grids)
            direct = np.cumsum(
                benchmark.paper_gumbel_optimal_score(pivots, delta), axis=1
            )
            np.testing.assert_allclose(paths, direct)

    def test_plotted_rules_are_a_subset_of_computed(self):
        self.assertTrue(set(ptc.RULE_ORDER) <= set(ptc.COMPUTED_RULES))
        for rule in ptc.COMPUTED_RULES:
            self.assertIn(rule, ptc.RULE_LABEL)


class AucTests(unittest.TestCase):
    """AUC is the calibration-free comparison, so it must be exactly right."""

    def test_perfect_separation(self):
        self.assertEqual(ptc.auc_score(np.array([3.0, 4.0]), np.array([1.0, 2.0])), 1.0)

    def test_reversed_separation(self):
        self.assertEqual(ptc.auc_score(np.array([1.0, 2.0]), np.array([3.0, 4.0])), 0.0)

    def test_all_ties_give_one_half(self):
        self.assertEqual(
            ptc.auc_score(np.array([1.0, 1.0]), np.array([1.0, 1.0])), 0.5
        )

    def test_partial_ties_counted_half(self):
        # one wm above, one tied -> (1 + 0.5) / 2
        self.assertAlmostEqual(
            ptc.auc_score(np.array([2.0, 1.0]), np.array([1.0])), 0.75
        )

    def test_matches_the_mann_whitney_definition(self):
        rng = np.random.default_rng(21)
        wm, null = rng.random(40), rng.random(55)
        brute = np.mean(
            [(1.0 if a > b else 0.5 if a == b else 0.0) for a in wm for b in null]
        )
        self.assertAlmostEqual(ptc.auc_score(wm, null), brute, places=12)

    def test_is_invariant_to_monotone_rescaling(self):
        """A cutoff-free measure must not care about the score's units."""
        rng = np.random.default_rng(22)
        wm, null = rng.random(30), rng.random(30)
        self.assertAlmostEqual(
            ptc.auc_score(wm, null), ptc.auc_score(3.0 * wm + 7.0, 3.0 * null + 7.0),
            places=12,
        )

    def test_rows_carry_auc_in_range(self):
        grids = ptc.build_grids(64)
        rng = np.random.default_rng(23)
        rows = ptc.curve_rows(
            "m", "0.3", rng.beta(4.0, 1.0, size=(25, 5)), rng.random((40, 5)), grids
        )
        for row in rows:
            self.assertGreaterEqual(row["auc"], 0.0)
            self.assertLessEqual(row["auc"], 1.0)


class StyleTests(unittest.TestCase):
    def test_rule_metadata_is_complete_and_unique(self):
        self.assertEqual(len(ptc.RULE_ORDER), len(set(ptc.RULE_ORDER)))
        for rule in ptc.RULE_ORDER:
            self.assertIn(rule, ptc.RULE_LABEL)
            self.assertIn(rule, ptc.RULE_DASH)
            self.assertIn(rule, ptc.RULE_COLOR)

    def test_colour_and_dash_together_identify_every_rule(self):
        """Identity is COMPOSITE: hue names the family, dash the member.

        Nine curves cannot take nine hues and stay separable under colour-vision
        deficiency, so hues repeat by design.  What must stay unique is the
        (colour, dash) PAIR -- the earlier test required distinct dashes
        outright, which a composite scheme cannot satisfy and does not need.
        """

        pairs = [(ptc.RULE_COLOR[r], ptc.RULE_DASH[r]) for r in ptc.RULE_ORDER]
        self.assertEqual(len(pairs), len(set(pairs)), "a (colour, dash) pair repeats")

    def test_rules_sharing_a_hue_have_distinct_dashes(self):
        """Within a family, the dash is the only thing left to tell members apart."""

        by_colour = {}
        for rule in ptc.RULE_ORDER:
            by_colour.setdefault(ptc.RULE_COLOR[rule], []).append(ptc.RULE_DASH[rule])
        for colour, dashes in by_colour.items():
            self.assertEqual(len(dashes), len(set(dashes)),
                             f"hue {colour} repeats a dash pattern")

    def test_identity_never_rests_on_colour_alone(self):
        """No two rules may share a hue AND a dash, in either scheme."""

        for order, colour, dash in (
            (ptc.RULE_ORDER, ptc.RULE_COLOR, ptc.RULE_DASH),
            (ptc.INVERSE_RULE_ORDER, ptc.RULE_COLOR_INV, ptc.RULE_DASH_INV),
        ):
            pairs = [(colour[r], dash[r]) for r in order]
            self.assertEqual(len(pairs), len(set(pairs)))


class OutputTests(unittest.TestCase):
    def _rows(self):
        rows = []
        for model in ("1p3B", "2p7B"):
            for temperature in ("0.1", "0.3"):
                for rule in ptc.RULE_ORDER:
                    for prefix in range(1, 13):
                        rows.append({
                            "model": model, "temperature": temperature,
                            "tokens_generated": 12, "rule": rule, "prefix": prefix,
                            "type1": 0.05, "type1_mcse": 0.01,
                            "type2": 1.0 - prefix / 12.0, "type2_mcse": 0.02,
                            "n_calibration": 20, "n_type1": 20, "n_watermarked": 30,
                            "distinct_watermarked": float(prefix),
                            "distinct_null": float(prefix),
                        })
        return rows

    def test_csv_roundtrip_preserves_every_field(self):
        rows = self._rows()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curves.csv"
            ptc.write_csv(rows, path)
            with path.open(encoding="utf-8") as handle:
                loaded = list(csv.DictReader(handle))
        self.assertEqual(len(loaded), len(rows))
        self.assertEqual(tuple(loaded[0].keys()), ptc.CURVE_FIELDS)

    def test_figures_render(self):
        rows = self._rows()
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / "grid"
            ptc.plot_prefix_grid(rows, "1p3B", stem)
            self.assertTrue(stem.with_suffix(".pdf").exists())
            self.assertTrue(stem.with_suffix(".png").exists())
            summary = Path(directory) / "sweep"
            ptc.plot_temperature_summary(rows, summary, horizon=12)
            self.assertTrue(summary.with_suffix(".pdf").exists())

    def test_prefix_grid_ignores_other_models(self):
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / "absent"
            ptc.plot_prefix_grid(self._rows(), "nonexistent", stem)
            self.assertFalse(stem.with_suffix(".pdf").exists())


if __name__ == "__main__":
    unittest.main()
