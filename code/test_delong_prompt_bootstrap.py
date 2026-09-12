"""Tests for the prompt-cluster bootstrap on the temperature-matched AUCs."""

from __future__ import annotations

import json
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import delong_prompt_bootstrap as dpb

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = (REPO / "results/bayesian_paper_benchmark/real_model"
             / "temperature_matched")


def paired_scores(n: int, seed: int, ties: bool = False) -> dict:
    """Scores for every rule the contrast lists name, sharing one document set."""
    rng = np.random.default_rng(seed)
    rules = {r for pair in dpb.CONTRASTS + dpb.DESCRIPTIVE_CONTRASTS for r in pair}
    base_w, base_z = rng.normal(1.0, 1.0, n), rng.normal(0.0, 1.0, n)
    out = {}
    for i, rule in enumerate(sorted(rules)):
        jitter = 0.0 if ties else 0.01 * i * rng.normal(0.0, 1.0, n)
        out[rule] = (base_w + jitter, base_z + jitter)
    return out


class PercentileTest(unittest.TestCase):
    def test_invalid_bootstrap_configuration_and_pairing_are_rejected(self) -> None:
        scores = paired_scores(10, seed=4)
        for replicates in (0, 1, -1):
            with self.subTest(replicates=replicates), self.assertRaises(ValueError):
                dpb.bootstrap_cell(scores, replicates, seed=1)
        for replacement in ((np.zeros(9), np.zeros(10)),
                            (np.zeros(10), np.full(10, np.nan)),
                            (np.zeros((10, 1)), np.zeros(10))):
            invalid = dict(scores)
            invalid["bayes_shared"] = replacement
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                dpb.bootstrap_cell(invalid, 2, seed=1)

    def test_missing_contrast_rules_are_rejected(self) -> None:
        for scores in ({}, {"bayes_shared": (np.zeros(10), np.zeros(10))}):
            with self.assertRaisesRegex(ValueError, "contrast rules"):
                dpb.bootstrap_cell(scores, 2, seed=1)

    def test_invalid_cli_counts_models_and_holdout_are_rejected_early(self) -> None:
        for options in (["--replicates", "1"], ["--prompt-start", "-1"],
                        ["--models", "1p3B", "1p3B"], ["--models", "bogus"]):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                dpb.main(options)

    def test_main_passes_legacy_opt_in_and_retains_source_provenance(self) -> None:
        captured = {}
        def collect(models, scratch, scheme, **kwargs):
            captured.update(kwargs)
            kwargs["provenance"]["1p3B_T0.3"] = [{"mode": "fixture", "path": "fixture.npz"}]
            dpb.ptc.curve_rows("1p3B", "0.3", np.zeros((4, 100)), np.zeros((4, 100)), {})
            return []
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "bootstrap.json"
            with patch.object(dpb.ptc, "collect", side_effect=collect), \
                    patch.object(dpb, "cell_scores", return_value=paired_scores(4, seed=3)), \
                    contextlib.redirect_stdout(io.StringIO()):
                dpb.main(["--models", "1p3B", "--replicates", "2",
                          "--allow-legacy-archives", "--output", str(output)])
            payload = json.loads(output.read_text())
        self.assertTrue(captured["allow_legacy"])
        self.assertEqual(payload["input_provenance"]["1p3B_T0.3"][0]["mode"], "fixture")

    def test_missing_inference_window_does_not_write_empty_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "empty.json"
            with patch.object(dpb.ptc, "collect", return_value=[]), \
                    self.assertRaisesRegex(ValueError, "no paired cells"):
                dpb.main(["--replicates", "2", "--output", str(output)])
            self.assertFalse(output.exists())

    def test_identically_zero_contrast_gives_probability_one(self) -> None:
        # Both bootstrap tails are then the whole sample, so twice the smaller
        # is 2.  The record used to store that, which is not a probability.
        scores = paired_scores(60, seed=11, ties=True)
        cell = dpb.bootstrap_cell(scores, replicates=50, seed=3)
        every = {**cell["tests"], **cell["descriptive_differences"]}
        self.assertTrue(every)
        for name, body in every.items():
            with self.subTest(contrast=name):
                self.assertEqual(body["dauc"], 0.0)
                self.assertEqual(body["p_percentile_two_sided"], 1.0)

    def test_every_percentile_is_a_probability(self) -> None:
        scores = paired_scores(80, seed=5)
        cell = dpb.bootstrap_cell(scores, replicates=64, seed=7)
        every = {**cell["tests"], **cell["descriptive_differences"]}
        for name, body in every.items():
            with self.subTest(contrast=name):
                self.assertGreaterEqual(body["p_percentile_two_sided"], 0.0)
                self.assertLessEqual(body["p_percentile_two_sided"], 1.0)

    def test_percentile_is_floored_at_the_bootstrap_resolution(self) -> None:
        # A contrast no replicate crosses cannot be resolved past 1/replicates,
        # and reporting zero would claim more than the resample supports.
        scores = paired_scores(120, seed=9)
        for rule, (w, z) in scores.items():
            if rule == "bayes_shared":
                scores[rule] = (w - 5.0, z)
        cell = dpb.bootstrap_cell(scores, replicates=40, seed=2)
        floors = [b["p_percentile_two_sided"]
                  for b in cell["tests"].values() if b["dauc"] > 0]
        self.assertTrue(floors)
        self.assertGreaterEqual(min(floors), 1.0 / 40)


class ArtifactTest(unittest.TestCase):
    def test_legacy_inference_is_explicitly_marked_as_historical(self) -> None:
        archive = (REPO / "results/bayesian_paper_benchmark_delta_half"
                   / "real_model/temperature_matched")
        names = ("delong_tests.json", "delong_tests_weights.json",
                 "delong_tests_wideprior.json", "delong_tests_inverse.json")
        for path in [directory / name for directory in (ARTIFACTS, archive)
                     for name in names]:
            with self.subTest(artifact=str(path.relative_to(REPO))):
                payload = json.loads(path.read_text())
                status = payload["_inference_status"]
                self.assertEqual(status["status"], "historical_non_inferential")
                self.assertFalse(status["valid_for_current_inference"])
                self.assertIn("sign_test_p", status["historical_fields"])
                if "inverse" in path.name:
                    self.assertIsNone(status["current_artifact"])
                    self.assertIn("No current inverse", status["current_section"])
                else:
                    current = (path.parent / status["current_artifact"]).resolve()
                    self.assertEqual(current, ARTIFACTS / "delong_tests_bootstrap.json")
                    self.assertTrue(current.exists())
                if path.parent == archive:
                    self.assertIn("Archived", status["scope"])

    def test_current_artifacts_do_not_report_sign_test_summaries(self) -> None:
        for name in ("delong_tests_bootstrap.json", "delong_tests_holdout.json"):
            payload = json.loads((ARTIFACTS / name).read_text())
            with self.subTest(artifact=name):
                self.assertIn("Joint prompt-cluster bootstrap", payload["method"])
                self.assertNotIn("sign_test_p", json.dumps(payload))
                self.assertNotIn("summary", payload)

    def test_stored_percentiles_are_probabilities(self) -> None:
        for name in ("delong_tests_bootstrap.json", "delong_tests_holdout.json"):
            path = ARTIFACTS / name
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            with self.subTest(artifact=name):
                for cell, body in payload["cells"].items():
                    for section in ("tests", "descriptive_differences"):
                        for contrast, test in body.get(section, {}).items():
                            p = test["p_percentile_two_sided"]
                            self.assertLessEqual(
                                p, 1.0, f"{name} {cell} {contrast} stores p={p}")

    def test_holdout_excludes_the_released_prompts(self) -> None:
        path = ARTIFACTS / "delong_tests_holdout.json"
        if not path.exists():
            self.skipTest("holdout artifact not present")
        payload = json.loads(path.read_text())
        self.assertEqual(payload["prompt_start"], 500)
        # The informative window carries 2500 documents; dropping the 500
        # released prompts must leave exactly 2000.
        sizes = {body["n"] for body in payload["cells"].values()}
        self.assertEqual(sizes, {2000})


if __name__ == "__main__":
    unittest.main()
