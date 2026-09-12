"""Negative controls for reporting eligibility, table edits, and audit gates."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

import check_figure_geometry as geometry
import check_manuscript_tables as checker
import diff_artifacts as diff
import paired_comparisons as paired
import sync_manuscript_tables as sync


def table(name, rows):
    return (f"\\begin{{table}}\n\\label{{tab:{name}}}\n\\midrule\n"
            + rows + "\n\\end{table}\n")


POOLED = table("pooled-real", "\n".join((
    r"Bayes, shared $\Delta$ & \mcse{.7000}{.0100} & \mcse{.9000}{.0100} \\",
    r"Bayes, hierarchical $\Delta$ & \mcse{.8000}{.0100} & \mcse{.9000}{.0100} \\",
    r"Difference & $+$\mcse{.1000}{.0100} & $+$\mcse{.0000}{.0100} \\",
    r"\multicolumn{3}{l}{\textit{Reference scores}}\\",
    r"ars & \mcse{.8200}{.0100} & \mcse{.9100}{.0100} \\",
    r"log & \mcse{.7100}{.0100} & \mcse{.9000}{.0100} \\",
)))


class ReferenceEligibilityTests(unittest.TestCase):
    def test_bayes_union_and_trgof_cannot_be_best_reference(self):
        prefix = "paper_text_iid_delta_equal_tail|gumbel|100|"
        ind = {prefix + name: np.ones(10, dtype=bool) for name in paired.BAYES_METHODS}
        ind[prefix + "trgof_s2"] = np.ones(10, dtype=bool)
        ind[prefix + "h_ars"] = np.array([False, True] * 5)
        ind[prefix + "h_spike_0.01"] = np.ones(10, dtype=bool)
        rows = paired.build_comparisons(ind)["comparisons"]
        selected = [row for row in rows if row["comparator_is_best_paper_score"]]
        self.assertEqual([row["comparator"] for row in selected], ["h_ars"])
        self.assertTrue(selected[0]["comparator_is_paper_score"])
        self.assertFalse(any(row["comparator"] == "bayes_tokenwise_uniontail" for row in rows))

    def test_each_scheme_uses_its_explicit_reference_family(self):
        ind = {}
        for scheme, scores in paired.PAPER_REFERENCE_BY_SCHEME.items():
            for horizon in (100, 300, 700):
                prefix = f"shared_delta_equal_tail_sensitivity|{scheme}|{horizon}|"
                ind[prefix + "bayes_shared"] = np.ones(10, dtype=bool)
                for score in scores | {"trgof_s2", "h_spike_0.01"}:
                    ind[prefix + score] = np.ones(10, dtype=bool)
        rows = paired.build_comparisons(ind)["comparisons"]
        for scheme, expected_size in (("gumbel", 18), ("inverse", 12)):
            family = [row for row in rows if row["scheme"] == scheme
                      and row["comparator_is_paper_score"]]
            self.assertEqual(len(family), expected_size)
            self.assertTrue(all(row["holm_family_size"] == expected_size for row in family))
            self.assertTrue(all(row["holm_adjusted_p"] == 1 for row in family))
            self.assertTrue(all(row["holm_family"].endswith("|paper_scores") for row in family))
        self.assertTrue(all(row["comparator_is_paper_score"] for row in rows
                            if row["comparator_is_best_paper_score"]))

    def test_best_reference_tie_break_is_deterministic(self):
        prefix = "shared_delta_equal_tail_sensitivity|inverse|100|"
        names = ["h_neg", "h_dif_star_0.1", "bayes_shared", "trgof_s2"]
        ind = {prefix + name: np.ones(5, dtype=bool) for name in names}
        rows = paired.build_comparisons(ind)["comparisons"]
        self.assertEqual([x["comparator"] for x in rows if x["comparator_is_best_paper_score"]],
                         ["h_dif_star_0.1"])

    def test_unknown_scheme_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown pivot scheme"):
            paired.build_comparisons({
                "shared_delta_equal_tail_sensitivity|unknown|100|bayes_shared":
                    np.ones(5, dtype=bool),
            })


class SignedRowTests(unittest.TestCase):
    def test_probability_one_is_a_recognized_numeric_cell(self):
        checker.assert_row(r"row & \mcse{1.0000}{.0000} \\", "row", [1.0], [0.0])

    def test_signed_row_checks_positive_negative_and_zero(self):
        row = (r"Difference & $+$\mcse{.0010}{.0001} & $-$\mcse{.0020}{.0002}"
               r" & $+$\mcse{.0000}{.0003} \\")
        checker.assert_row(row, "Difference &", [.001, -.002, 0], [.0001, .0002, .0003],
                           signed=True)
        with self.assertRaisesRegex(AssertionError, "signs drifted"):
            checker.assert_row(row.replace("$-$", "$+$"), "Difference &",
                               [.001, -.002, 0], [.0001, .0002, .0003], signed=True)

    def test_missing_sign_does_not_pass_as_positive(self):
        with self.assertRaisesRegex(AssertionError, "signs drifted"):
            checker.assert_row(r"Difference & \mcse{.0010}{.0001} \\",
                               "Difference &", [.001], [.0001], signed=True)

    def test_unequal_estimate_and_se_counts_fail(self):
        with self.assertRaisesRegex(AssertionError, "counts differ"):
            checker.assert_row(r"row & \mcse{.0010}{.0001} \\", "row", [.001, .002], [.0001])

    def test_synchronizer_repairs_sign_and_preserves_comment(self):
        source = table("fixture", r"Difference & $-$\mcse{.0010}{.0001} \\ % retain this")
        def audit(candidate):
            checker.assert_row(checker.table_body(candidate, "fixture"), "Difference &",
                               [.001], [.0001], signed=True)
        with mock.patch.object(checker, "check_manuscript", side_effect=audit):
            edits = sync.collect_edits(source)
        self.assertEqual(len(edits), 1)
        start, stop, replacement = edits[0]
        repaired = source[:start] + replacement + source[stop:]
        self.assertEqual(repaired, source.replace("$-$", "$+$"))
        audit(repaired)


class TableEmphasisTests(unittest.TestCase):
    def test_all_missing_markers_are_restored_with_displayed_ties(self):
        with self.assertRaisesRegex(AssertionError, "boldface"):
            checker.check_boldface(POOLED)
        repaired, count = sync.renormalise_bold(POOLED)
        self.assertEqual(count, 5)
        self.assertEqual(checker.check_boldface(repaired), 4)
        self.assertEqual(sync.renormalise_bold(repaired), (repaired, 0))

    def test_difference_row_is_always_unemphasised(self):
        broken = POOLED.replace(r"Difference & $+$\mcse", r"Difference & $+$\bestmcse")
        repaired, _ = sync.renormalise_bold(broken)
        self.assertNotIn(r"Difference & $+$\bestmcse", repaired)
        checker.check_boldface(repaired)

    def test_deficit_support_bolds_only_reference_scores(self):
        source = table("deficit-support", "\n".join((
            r"prior & narrow & \bestmcse{.9900}{.0010} \\",
            r" & wide & \mcse{.9900}{.0010} \\",
            r"\multicolumn{3}{l}{\textit{Reference scores}}\\",
            r"ars & --- & \mcse{.8000}{.0010} \\",
            r"log & --- & \mcse{.7000}{.0010} \\",
        )))
        repaired, count = sync.renormalise_bold(source)
        self.assertEqual(count, 2)
        self.assertIn(r"ars & --- & \bestmcse", repaired)
        self.assertNotIn(r"\bestmcse{.9900}", repaired)
        self.assertEqual(checker.check_boldface(repaired), 1)

    def test_type_i_columns_are_unemphasised_even_if_erroneously_bold(self):
        source = table("released-output", "\n".join((
            r"a & \bestmcse{.0400}{.0010} & \mcse{.1000}{.0010} \\",
            r"b & \mcse{.0500}{.0010} & \mcse{.2000}{.0010} \\",
        )))
        repaired, count = sync.renormalise_bold(source)
        self.assertEqual(count, 2)
        self.assertNotIn(r"\bestmcse{.0400}", repaired)
        self.assertIn(r"\bestmcse{.1000}", repaired)
        checker.check_boldface(repaired)

    def test_missing_reference_block_cannot_hide_deficit_support_policy(self):
        source = table("deficit-support", r"prior & \mcse{.9000}{.0010} \\")
        with self.assertRaisesRegex(AssertionError, "missing its Reference scores"):
            sync.renormalise_bold(source)

    def test_maximum_of_one_is_bolded(self):
        source = table("pooled-real", "\n".join((
            r"a & \mcse{.9000}{.0010} \\",
            r"b & \mcse{1.0000}{.0000} \\",
        )))
        repaired, count = sync.renormalise_bold(source)
        self.assertEqual(count, 1)
        self.assertIn(r"\bestmcse{1.0000}", repaired)
        checker.check_boldface(repaired)

    def test_inline_comment_macros_are_not_rewritten(self):
        source = POOLED.replace("\n\\end{table}", "\n% \\bestmcse{.9999}{.0001}\n\\end{table}")
        repaired, count = sync.renormalise_bold(source)
        self.assertEqual(count, 5)
        self.assertIn(r"% \bestmcse{.9999}{.0001}", repaired)


class SynchronizerSafetyTests(unittest.TestCase):
    def test_dry_run_reports_emphasis_and_does_not_write(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "manuscript.tex"
            target.write_text(POOLED, encoding="utf-8")
            out = io.StringIO()
            with mock.patch.object(checker, "TEX_PATH", target), \
                    mock.patch.object(sync, "collect_edits", return_value=[]), \
                    mock.patch.object(checker, "check_manuscript", side_effect=checker.check_boldface), \
                    contextlib.redirect_stdout(out):
                self.assertEqual(sync.main(["--dry-run"]), 0)
            self.assertEqual(target.read_text(), POOLED)
            self.assertIn("updated 5", out.getvalue())
            self.assertIn("dry run: no files written", out.getvalue())

    def test_validation_failure_leaves_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "manuscript.tex"
            target.write_text(POOLED, encoding="utf-8")
            with mock.patch.object(checker, "TEX_PATH", target), \
                    mock.patch.object(sync, "collect_edits", return_value=[]), \
                    mock.patch.object(checker, "check_manuscript", side_effect=AssertionError("prose")), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(AssertionError, "prose"):
                    sync.main([])
            self.assertEqual(target.read_text(), POOLED)
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_numeric_collection_failure_is_not_retried_or_swallowed(self):
        source = table("fixture", r"row & \mcse{.1000}{.0100} \\")
        original = checker.assert_row
        def bad_check(candidate):
            checker.assert_row(checker.table_body(candidate, "fixture"), "row", [.2], [.01])
            raise AssertionError("unrelated prose failure")
        with mock.patch.object(checker, "check_manuscript", side_effect=bad_check) as audit:
            with self.assertRaisesRegex(AssertionError, "unrelated prose"):
                sync.collect_edits(source)
        self.assertEqual(audit.call_count, 1)
        self.assertIs(checker.assert_row, original)

    def test_atomic_write_preserves_mode_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "manuscript.tex"
            target.write_text("old")
            target.chmod(0o640)
            sync.atomic_write(target, "new", "old")
            self.assertEqual(target.read_text(), "new")
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o640)
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_atomic_write_preserves_concurrent_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "manuscript.tex"
            target.write_text("user edit")
            with self.assertRaisesRegex(RuntimeError, "changed during validation"):
                sync.atomic_write(target, "candidate", "old")
            self.assertEqual(target.read_text(), "user edit")
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_replace_failure_does_not_damage_original(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "manuscript.tex"
            target.write_text("old")
            with mock.patch.object(sync.os, "replace", side_effect=OSError("disk")):
                with self.assertRaisesRegex(OSError, "disk"):
                    sync.atomic_write(target, "new", "old")
            self.assertEqual(target.read_text(), "old")
            self.assertEqual(list(Path(directory).iterdir()), [target])


class GeometryGuardsTests(unittest.TestCase):
    def boxes(self):
        result = {}
        for label, (plate, inside, outside) in geometry.EXPECTED.items():
            result[label] = {plate: (0., 0., 10., 10.)}
            result[label].update({name: (1., 1., 2., 2.) for name in inside})
            result[label].update({name: (11., 11., 12., 12.) for name in outside})
        return result

    def test_exact_expected_geometry_passes(self):
        self.assertEqual(geometry.validate_geometry(self.boxes()), [])

    def test_missing_expected_node_fails(self):
        boxes = self.boxes()
        del boxes["A"]["delta"]
        self.assertIn("panel A: missing nodes ['delta']", geometry.validate_geometry(boxes))

    def test_extra_node_or_panel_fails(self):
        boxes = self.boxes()
        boxes["A"]["unrelated"] = (1., 1., 2., 2.)
        boxes["Z"] = {}
        failures = geometry.validate_geometry(boxes)
        self.assertIn("panel A: unexpected nodes ['unrelated']", failures)
        self.assertIn("unexpected panel Z", failures)

    def test_plate_membership_still_checked(self):
        boxes = self.boxes()
        boxes["A"]["delta"] = (1., 1., 2., 2.)
        self.assertIn("panel A: delta must sit OUTSIDE plateU", geometry.validate_geometry(boxes))

    def test_extra_node_style_options_do_not_hide_probe(self):
        body = r"\node[latentnode, xshift=2pt] (delta) {};\end{tikzpicture}"
        self.assertIn("GEO A delta", geometry.instrument(body, "A", "plateU"))

    def test_duplicate_probes_fail(self):
        line = "GEO A q SW 1pt 1pt NE 2pt 2pt\n"
        with self.assertRaisesRegex(ValueError, "duplicate geometry"):
            geometry.parse_geometry(line + line)

    def test_latex_failure_is_not_a_partial_geometry_success(self):
        result = subprocess.CompletedProcess([], 1, stdout="broken TeX", stderr="")
        with mock.patch.object(geometry.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "pdflatex failed"):
                geometry.compile_geometry("bad document")


class ArtifactDiffTests(unittest.TestCase):
    def run_case(self, old, new):
        with tempfile.TemporaryDirectory() as directory:
            before, after = Path(directory) / "old", Path(directory) / "new"
            before.mkdir(); after.mkdir()
            for root, files in ((before, old), (after, new)):
                for name, value in files.items():
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(value, encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as out:
                status = diff.main([str(before), str(after)])
            return status, out.getvalue()

    def test_identical_recursive_inventory_passes(self):
        files = {"nested/data.csv": "a\n1\n", "result.json": '{"x": 1}'}
        self.assertEqual(self.run_case(files, files)[0], 0)

    def test_extra_or_missing_csv_columns_fail(self):
        for old, new in (("a\n1\n", "a,b\n1,2\n"), ("a,b\n1,2\n", "a\n1\n")):
            status, output = self.run_case({"a.csv": old}, {"a.csv": new})
            self.assertEqual(status, 1)
            self.assertIn("<columns>", output)

    def test_missing_artifact_fails(self):
        status, output = self.run_case({"a.json": "{}"}, {})
        self.assertEqual(status, 1)
        self.assertIn("MISSING", output)

    def test_new_only_artifact_fails(self):
        self.assertEqual(self.run_case({"a.json": "{}"}, {"a.json": "{}", "b.json": "{}"})[0], 1)

    def test_empty_inventory_is_not_success(self):
        self.assertEqual(self.run_case({}, {})[0], 1)

    def test_malformed_json_fails(self):
        self.assertEqual(self.run_case({"a.json": "{}"}, {"a.json": "{"})[0], 1)

    def test_documented_volatile_exclusions_are_retained(self):
        old = {"a.json": '{"value": 1, "timestamp": 1}', "provenance.json": "{}"}
        new = {"a.json": '{"value": 1, "timestamp": 2}'}
        self.assertEqual(self.run_case(old, new)[0], 0)

    def test_missing_key_messages_name_the_correct_side(self):
        self.assertEqual(diff.walk({}, {"x": 1}), [(".x", "<missing in old>", 1)])
        self.assertEqual(diff.walk({"x": 1}, {}), [(".x", 1, "<missing in new>")])


class DriftReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "contexts" / "drift_report.py"
        spec = importlib.util.spec_from_file_location("audit_drift_report", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_current_wrapper_passes_and_restores_checker(self):
        original = checker.assert_row
        with mock.patch.object(checker, "main", return_value=None), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.module.main(), 0)
        self.assertIs(checker.assert_row, original)

    def test_signed_failure_is_collected_without_false_success_banner(self):
        def run():
            checker.assert_row(r"Difference & $-$\mcse{.0010}{.0001} \\",
                               "Difference &", [.001], [.0001], signed=True)
            print("manuscript table check passed")
        with mock.patch.object(checker, "main", side_effect=run), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(self.module.main(), 1)
        self.assertIn("1 DRIFTED ROWS", out.getvalue())
        self.assertNotIn("table check passed", out.getvalue())

    def test_system_exit_is_incomplete_not_clean(self):
        with mock.patch.object(checker, "main", side_effect=SystemExit(2)), \
                contextlib.redirect_stdout(io.StringIO()) as out, \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.module.main(), 1)
        self.assertIn("INCOMPLETE", out.getvalue())
        self.assertNotIn("no drift", out.getvalue())


if __name__ == "__main__":
    unittest.main()
