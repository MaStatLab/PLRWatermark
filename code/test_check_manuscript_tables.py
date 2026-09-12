"""Regression guards for numerical claims hidden by TeX comments."""

import unittest
from unittest import mock

import check_manuscript_tables as checker


class VisibleTexTests(unittest.TestCase):
    def test_escaped_percent_is_preserved(self):
        self.assertEqual(checker.visible_tex(r"recovery is $100\%$"), r"recovery is $100\%$")

    def test_unescaped_percentage_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "unescaped percentage"):
            checker.visible_tex("recovery is 100% and this would disappear")

    def test_comment_content_is_not_checked_as_visible_prose(self):
        self.assertEqual(checker.visible_tex("% old 100% claim\nvisible"), "\nvisible")

    def test_inline_comment_is_removed(self):
        self.assertEqual(checker.visible_tex("visible % invisible\nnext"), "visible \nnext")

    def test_even_backslashes_do_not_escape_comment(self):
        self.assertEqual(checker.visible_tex(r"row \\% hidden"), "row \\\\")


class TailWidthNarrativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tex = checker.TEX_PATH.read_text(encoding="utf-8")
        cls.payload = checker.read_json("tail_width_estimate.json")

    def test_current_tail_width_narrative_matches_artifact(self):
        self.assertGreater(checker.check_tail_width_fit(self.tex, self.payload), 0)

    def test_reversed_document_counts_are_rejected(self):
        broken = self.tex.replace(
            "resamples 385 OPT-1.3B documents and 368 Sheared-LLaMA-2.7B documents",
            "resamples 368 OPT-1.3B documents and 385 Sheared-LLaMA-2.7B documents",
        )
        self.assertNotEqual(broken, self.tex)
        with self.assertRaisesRegex(AssertionError, "correct models"):
            checker.check_tail_width_fit(broken, self.payload)

    def test_truncated_recovery_percentage_is_rejected(self):
        broken = self.tex.replace(r"$98.5\%$", r"$98\%$")
        self.assertNotEqual(broken, self.tex)
        with self.assertRaisesRegex(AssertionError, "J=2 recovery"):
            checker.check_tail_width_fit(broken, self.payload)

    def test_commented_out_recovery_is_not_evidence(self):
        broken = self.tex.replace(r"$98.5\%$", r"% $98.5\%$")
        self.assertNotEqual(broken, self.tex)
        with self.assertRaisesRegex(AssertionError, "J=2 recovery"):
            checker.check_tail_width_fit(broken, self.payload)


class WidthPoolingNarrativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tex = checker.TEX_PATH.read_text(encoding="utf-8")
        cls.payload = checker.read_json("width_persistence_sweep.json")

    def test_generator_details_in_body_match_artifact(self):
        self.assertGreater(checker.check_width_pooling(self.tex, self.payload), 0)

    def test_changed_persistence_cannot_be_hidden_by_number_elsewhere(self):
        broken = self.tex.replace("$.854$", "$.754$") + "\nUnrelated $.854$.\n"
        self.assertNotEqual(broken, self.tex)
        with self.assertRaisesRegex(AssertionError, "width_persistence .854"):
            checker.check_width_pooling(broken, self.payload)

    def test_changed_latent_correlation_is_rejected(self):
        broken = self.tex.replace("$.9025$", "$.9024$")
        self.assertNotEqual(broken, self.tex)
        with self.assertRaisesRegex(AssertionError, "latent correlation .9025"):
            checker.check_width_pooling(broken, self.payload)


class HoldoutNarrativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tex = checker.visible_tex(checker.TEX_PATH.read_text(encoding="utf-8"))

    def test_current_holdout_narrative_matches_artifacts(self):
        self.assertGreater(checker.check_holdout_claims(self.tex), 0)
        # Reflowing a paragraph must not change validation of its claims.
        self.assertGreater(checker.check_holdout_claims(self.tex.replace(" ", "\n")), 0)
        # The formalized comparison still checks its artifact-derived range.
        broken = self.tex.replace("$.0096$ to $.0568$", "$.0096$ to $.0567$")
        self.assertNotEqual(broken, self.tex)
        with self.assertRaisesRegex(AssertionError, "holdout paragraph"):
            checker.check_holdout_claims(broken)

    def test_incorrect_effect_change_count_is_rejected(self):
        broken = self.tex.replace(
            "twenty point estimates increase and four decrease slightly",
            "twenty point estimates increase and three decrease slightly",
        )
        self.assertNotEqual(broken, self.tex)
        with self.assertRaisesRegex(AssertionError, "effect-change prose"):
            checker.check_holdout_claims(broken)

    def test_interval_width_does_not_establish_nonshrinking_effects(self):
        broken = self.tex.replace(
            "twenty point estimates increase and four decrease slightly",
            "removing the reused prompts does not shrink the effect",
        )
        self.assertNotEqual(broken, self.tex)
        with self.assertRaisesRegex(AssertionError, "effect-change prose"):
            checker.check_holdout_claims(broken)


class MethodLabelTests(unittest.TestCase):
    def test_aliases_preserve_method_identity_and_numeric_validation(self):
        labels = (
            (r"Bayes, shared $\Delta$ $+$ tail shape ($\rho$ mixture)",
             "Bayes, robust shared $+$ tail shape"),
            (r"Bayes, shared $\Delta$ $+$ union tail ($\rho$ mixture)",
             "Bayes, robust shared $+$ union tail"),
            (r"Bayes, tokenwise $\Delta$ $+$ tail shape",
             "Bayes, tokenwise $+$ tail shape"),
            (r"Bayes, hierarchical $\Delta$", r"Bayes, pooled $\Delta$"),
            (r"\quad $+$ hierarchical tail width", r"\quad $+$ tail width, pooled"),
            (r"\quad $+$ shared tail width", r"\quad $+$ tail width, frozen ladder"),
        )
        for current, legacy in labels:
            with self.subTest(label=current):
                suffix = r" & \mcse{.0014}{.0005} \\"
                canonical = checker.canonical_method_labels(current + suffix)
                self.assertEqual(canonical, legacy + suffix)
                self.assertEqual(checker.canonical_method_labels(canonical), canonical)
                checker.assert_row(canonical, legacy, [.0014], [.0005])
                with self.assertRaisesRegex(AssertionError, "drifted"):
                    checker.assert_row(canonical, legacy, [.0024], [.0005])
        source = checker.TEX_PATH.read_text(encoding="utf-8")
        table = checker.table_block(source, "union-weight") + r"\end{table}"
        payload = checker.read_json(
            "real_model/temperature_matched/delong_tests_bootstrap.json"
        )
        self.assertEqual(checker.check_union_weight(table, payload), 12)
        legacy = table.replace("equal-tail baseline", "no tail layer")
        self.assertEqual(checker.check_union_weight(legacy, payload), 12)

        import sync_manuscript_tables as synchronizer
        import update_manuscript_table_mcse as updater

        source = (
            "% leading comment changes offsets when the checker removes it\n"
            "\\begin{table}\n\\label{tab:fixture}\n\\midrule\n"
            r"Bayes, hierarchical $\Delta$ & \mcse{.0030}{.0004} \\"
            " % retain this comment \\mcse{.9999}{.9999}\n\\end{table}\n"
        )
        expected = source.replace(r"\mcse{.0030}{.0004}", r"\mcse{.0030}{.0005}")
        marker = r"Bayes, pooled $\Delta$"
        for body_only in (False, True):
            self.assertEqual(
                updater.update_table(
                    source, "fixture", [(marker, [.0030], [.0005], 0)],
                    body_only=body_only,
                ),
                expected,
            )

            def check_fixture(candidate):
                visible = checker.visible_tex(candidate)
                extract = checker.table_body if body_only else checker.table_block
                checker.assert_row(extract(visible, "fixture"), marker, [.0030], [.0005])

            original_assert = checker.assert_row
            with mock.patch.object(checker, "check_manuscript", side_effect=check_fixture):
                edits = synchronizer.collect_edits(source)
            self.assertIs(checker.assert_row, original_assert)
            self.assertEqual(len(edits), 1)
            changed = source
            for start, stop, replacement in reversed(edits):
                changed = changed[:start] + replacement + changed[stop:]
            self.assertEqual(changed, expected)


class SharedBoundaryMcseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tex = checker.TEX_PATH.read_text(encoding="utf-8")
        cls.rows = checker.read_csv("benchmark_results.csv")
        cls.summary = checker.read_json("benchmark_summary.json")

    def test_shared_and_family_use_conditional_contribution_mcse(self):
        import contextlib
        import io
        import shutil
        import tempfile
        from pathlib import Path

        import update_manuscript_table_mcse as updater

        expected = [.005722490607842028, .004755700156742342, .003928781230507935]
        for horizon, want in zip((100, 300, 700), expected):
            observed = checker.conditional_weight_mc_se(
                self.summary,
                scenario="shared_delta_equal_tail_sensitivity",
                scheme="gumbel",
                method="h_ind_1_over_e",
                horizon=horizon,
            )
            self.assertAlmostEqual(observed, want, places=14)
        regenerated = updater.clean_standard_errors(
            self.rows, self.summary, "h_ind_1_over_e", "gumbel"
        )
        for observed, want in zip(regenerated, expected):
            self.assertAlmostEqual(observed, want, places=14)
        self.assertGreater(
            checker.check_shared_and_family(self.tex, self.rows, self.summary), 0
        )
        # The documented standalone command must also work with the current
        # single-pivot table set, not only the individually tested helpers.
        with tempfile.TemporaryDirectory(prefix="watermarking-mcse-test-") as folder:
            temporary = Path(folder) / checker.TEX_PATH.name
            shutil.copyfile(checker.TEX_PATH, temporary)
            before = temporary.read_bytes()
            with mock.patch.object(checker, "TEX_PATH", temporary):
                with contextlib.redirect_stdout(io.StringIO()):
                    updater.main()
            self.assertEqual(temporary.read_bytes(), before)

    def assert_binomial_substitution_fails(self, label):
        marker = rf"\label{{tab:{label}}}"
        start = self.tex.index(marker)
        end = self.tex.index(r"\end{table}", start)
        block = self.tex[start:end]
        replacement = block.replace(r"\mcse{.2192}{.0057}", r"\mcse{.2192}{.0059}")
        self.assertTrue(block != replacement, "the conditional MCSE must be present")
        broken = self.tex[:start] + replacement + self.tex[end:]
        with self.assertRaisesRegex(AssertionError, "drifted"):
            checker.check_shared_and_family(broken, self.rows, self.summary)

    def test_family_binomial_mcse_is_rejected(self):
        self.assert_binomial_substitution_fails("family")

    def test_shared_binomial_mcse_is_rejected(self):
        self.assert_binomial_substitution_fails("shared")


class PooledAucNarrativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tex = checker.TEX_PATH.read_text(encoding="utf-8")
        cls.cells = checker.read_json(
            "real_model/temperature_matched/delong_tests_bootstrap.json"
        )["cells"]

    def test_current_mean_matches_all_eight_cells_and_ignores_reflow(self):
        self.assertEqual(checker.check_pooled_auc_mean(self.tex, self.cells), 1)
        self.assertEqual(
            checker.check_pooled_auc_mean(self.tex.replace(" ", "\n"), self.cells), 1
        )

    def test_largest_cell_is_not_the_mean(self):
        broken = self.tex.replace(
            "mean AUC improvement of $.0328$", "mean AUC improvement of $.0555$"
        )
        self.assertTrue(broken != self.tex, "the artifact-derived mean must be present")
        with self.assertRaisesRegex(AssertionError, "mean AUC improvement"):
            checker.check_pooled_auc_mean(broken, self.cells)

    def test_correct_number_in_another_section_is_not_evidence(self):
        broken = self.tex.replace(
            "mean AUC improvement of $.0328$", "mean AUC improvement of $.0555$"
        )
        broken += "\n\\section{Unrelated}\nmean AUC improvement of $.0328$\n"
        with self.assertRaisesRegex(AssertionError, "hierarchical-deficit section"):
            checker.check_pooled_auc_mean(broken, self.cells)


class PooledAucBoldfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = checker.TEX_PATH.read_text(encoding="utf-8")
        cls.table = checker.table_block(source, "pooled-real") + r"\end{table}"

    def test_current_within_block_maxima_are_marked(self):
        import sync_manuscript_tables as synchronizer

        self.assertEqual(checker.check_boldface(self.table), 16)
        unchanged, count = synchronizer.renormalise_bold(self.table)
        self.assertEqual(unchanged, self.table)
        self.assertEqual(count, 0)
        broken = self.table.replace(r"\bestmcse{.7837}", r"\mcse{.7837}")
        repaired, count = synchronizer.renormalise_bold(broken)
        self.assertEqual(count, 1)
        self.assertEqual(repaired, self.table)

    def test_removing_all_maximum_markers_is_rejected(self):
        broken = self.table.replace(r"\bestmcse", r"\mcse")
        self.assertTrue(broken != self.table, "the table must have maximum markers")
        with self.assertRaisesRegex(AssertionError, "tab:pooled-real.*boldface"):
            checker.check_boldface(broken)

    def test_marking_the_smaller_auc_is_rejected(self):
        broken = self.table.replace(r"\mcse{.7828}", r"\bestmcse{.7828}")
        broken = broken.replace(r"\bestmcse{.7837}", r"\mcse{.7837}")
        self.assertTrue(broken != self.table, "the AUC comparison must be present")
        with self.assertRaisesRegex(AssertionError, "tab:pooled-real.*boldface"):
            checker.check_boldface(broken)

    def test_difference_row_is_not_an_absolute_auc_maximum(self):
        broken = self.table.replace(
            r"Difference & $+$\mcse", r"Difference & $+$\bestmcse", 1
        )
        self.assertTrue(broken != self.table, "the descriptive Difference row must be present")
        with self.assertRaisesRegex(AssertionError, "Difference row is descriptive"):
            checker.check_boldface(broken)

    def test_legacy_method_label_does_not_bypass_latest_bolding_check(self):
        # table_block already canonicalizes hierarchical to the legacy pooled
        # label; this must not change the current document's display contract.
        self.assertIn(r"Bayes, pooled $\Delta$", self.table)
        self.assertEqual(checker.check_boldface(self.table), 16)

    def test_unbold_archived_table_remains_compatible(self):
        archived = self.table.replace(r"\bestmcse", r"\mcse")
        with mock.patch.object(checker, "GUMBEL_ONLY", False):
            self.assertEqual(checker.check_boldface(archived), 0)


if __name__ == "__main__":
    unittest.main()
