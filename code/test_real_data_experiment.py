"""Focused tests for the released-output real-data reanalysis."""

from __future__ import annotations

import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import real_data_experiment as realdata
from bayesian_watermark import SequentialBayesDetector


class RealDataExperimentTests(unittest.TestCase):
    def test_allowlist_has_exactly_the_six_commit_pinned_pickles(self) -> None:
        self.assertEqual(len(realdata.ASSET_SHA256), 6)
        for model in realdata.MODEL_SPECS:
            self.assertEqual(
                set(realdata.asset_names(model).values())
                & set(realdata.ASSET_SHA256),
                set(realdata.asset_names(model).values()),
            )

    def test_hash_mismatch_stops_before_torch_or_pickle_load(self) -> None:
        filename = next(iter(realdata.ASSET_SHA256))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / filename
            path.write_bytes(b"not the allowlisted pickle")
            with mock.patch.object(
                realdata, "_torch_module", side_effect=AssertionError("torch imported")
            ):
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    realdata.load_verified_pickle(path)

    def test_non_allowlisted_pickle_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unknown.pkl"
            path.write_bytes(b"data")
            with self.assertRaisesRegex(ValueError, "not one of the six"):
                realdata.verify_asset(path)

    def test_inverse_stored_eta_is_corrected_by_one_grid_step(self) -> None:
        model = "1p3B"
        vocabulary_size = realdata.MODEL_SPECS[model]["vocabulary_size"]
        rank = 25_000
        args = argparse.Namespace(
            model=realdata.MODEL_SPECS[model]["model_id"], m=200, T=500
        )
        prompts = np.zeros((500, 50), dtype=np.int64)
        raw_tokens = np.zeros((500, 220), dtype=np.int64)
        gumbel_pivots = np.full((500, 200), 0.4)
        u = np.full((500, 200), 0.25)
        eta_shifted = np.full(
            (500, 200), (rank - 1) / (vocabulary_size - 1.0), dtype=np.float32
        )
        top = np.full((500, 200), 0.9)
        raw = {
            "args": args,
            "prompts": prompts,
            "null": {"tokens": raw_tokens},
        }
        gumbel = {
            "args": args,
            "prompts": prompts.copy(),
            "watermark": {
                "Ys": gumbel_pivots,
                "top_probs": top,
                "tokens": np.zeros((500, 200), dtype=np.int64),
            },
        }
        inverse = {
            "args": args,
            "prompts": prompts.copy(),
            "watermark": {"Us": u, "etas": eta_shifted, "top_probs": top},
        }
        released = realdata.extract_released_model_data(
            model, raw, gumbel, inverse
        )
        np.testing.assert_allclose(
            released.inverse_watermarked_formal,
            abs(0.25 - rank / (vocabulary_size - 1.0)),
            rtol=0.0,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            released.inverse_watermarked_shifted,
            np.float32(abs(np.float32(0.25) - eta_shifted[0, 0])),
            rtol=0.0,
            atol=1e-14,
        )

    def test_prompt_mismatch_is_not_silently_ignored(self) -> None:
        model = "1p3B"
        args = argparse.Namespace(
            model=realdata.MODEL_SPECS[model]["model_id"], m=200, T=500
        )
        prompts = np.zeros((500, 50), dtype=np.int64)
        top = np.full((500, 200), 0.9)
        raw = {
            "args": args,
            "prompts": prompts,
            "null": {"tokens": np.zeros((500, 220), dtype=np.int64)},
        }
        gumbel = {
            "args": args,
            "prompts": prompts.copy(),
            "watermark": {
                "Ys": np.full((500, 200), 0.5),
                "top_probs": top,
                "tokens": np.zeros((500, 200), dtype=np.int64),
            },
        }
        inverse_prompts = prompts.copy()
        inverse_prompts[0, 0] = 1
        inverse = {
            "args": args,
            "prompts": inverse_prompts,
            "watermark": {
                "Us": np.full((500, 200), 0.25),
                "etas": np.full((500, 200), 0.5),
                "top_probs": top,
            },
        }
        with self.assertRaisesRegex(ValueError, "prompt tensors differ"):
            realdata.extract_released_model_data(model, raw, gumbel, inverse)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
    def test_cpu_prf_replay_matches_official_reference_values(self) -> None:
        # Values were evaluated directly from alternative_prf_schemes.py and
        # sampling.py at the pinned commit.  skipgram_prf hashes the oldest of
        # the four context tokens, giving seed 406910 for context token 11.
        prompts = np.array([[11, 22, 33, 44]], dtype=np.int64)
        raw_tokens = np.array([[7]], dtype=np.int64)
        gumbel, formal, shifted = realdata.replay_raw_pivots(
            prompts, raw_tokens, vocabulary_size=17, horizon=1
        )
        self.assertAlmostEqual(gumbel[0, 0], 0.728571891784668, places=15)
        self.assertAlmostEqual(formal[0, 0], 0.1372307538986206, places=15)
        self.assertAlmostEqual(shifted[0, 0], 0.1997307538986206, places=15)

    def test_shared_spike_paths_match_sequential_detector(self) -> None:
        rng = np.random.default_rng(20240827)
        deltas, weights = realdata.dirichlet.gauss_legendre_delta_grid(
            0.001, 0.5, 16
        )
        horizons = (2, 5, 8)
        for scheme, pivots in (
            ("gumbel", rng.uniform(0.05, 0.95, size=(3, 8))),
            ("inverse", rng.uniform(0.05, 0.8, size=(3, 8))),
        ):
            observed = realdata.shared_spike_paths(
                pivots,
                scheme=scheme,
                vocabulary_size=31,
                deltas=deltas,
                delta_weights=weights,
                horizons=horizons,
            )
            expected = np.empty_like(observed)
            for row in range(pivots.shape[0]):
                detector = SequentialBayesDetector(
                    scheme,
                    deltas,
                    weights,
                    structure="shared",
                    gumbel_family="spike",
                    vocabulary_size=31,
                    inverse_null_vocabulary_size=31,
                )
                detector.update_many(pivots[row])
                expected[row] = detector.log_bayes_factor_history[
                    np.asarray(horizons) - 1
                ]
            np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)

    def test_uniform_grid_has_the_requested_measure(self) -> None:
        for low, high in ((0.001, 0.5), (0.0, 0.1)):
            with self.subTest(low=low, high=high):
                deltas, weights = realdata.dirichlet.gauss_legendre_delta_grid(
                    low, high, 96
                )
                self.assertTrue(np.all(deltas > low))
                self.assertTrue(np.all(deltas < high))
                self.assertAlmostEqual(float(weights.sum()), 1.0, places=15)
                self.assertAlmostEqual(
                    float(np.dot(weights, deltas)),
                    0.5 * (low + high),
                    places=13,
                )
                self.assertAlmostEqual(
                    float(np.dot(weights, deltas**2)),
                    (low * low + low * high + high * high) / 3.0,
                    places=13,
                )

    def test_uniform_inverse_integral_matches_direct_quadrature(self) -> None:
        pivots = np.array([0.0, 0.2, 0.49, 0.500001, 0.9, 0.999, 0.99998])
        raw_nodes, raw_weights = np.polynomial.legendre.leggauss(256)
        for bounds in ((0.001, 0.5), (0.0, 0.1)):
            with self.subTest(bounds=bounds):
                config = realdata._benchmark_config(
                    realdata.RealDataConfig(
                        delta_low=bounds[0], delta_high=bounds[1]
                    ),
                    1000,
                )
                observed = np.exp(
                    realdata.real_data_inverse_score(
                        pivots, "bayes_tokenwise", config
                    )
                ) * realdata.benchmark.inverse_exact_null_density(pivots, 1000)
                # The detector integrates over its PRIOR support, which the
                # config now keeps separate from any generating law.
                low, high = config.prior_low, config.prior_high
                expected = np.zeros_like(pivots)
                for index, pivot in enumerate(pivots):
                    upper = min(high, 1.0 - pivot)
                    if upper <= low:
                        continue
                    delta = low + 0.5 * (raw_nodes + 1.0) * (upper - low)
                    density = 2.0 / (1.0 - delta) * (
                        1.0 - pivot / (1.0 - delta)
                    )
                    expected[index] = (
                        0.5 * (upper - low) / (high - low)
                        * np.dot(raw_weights, density)
                    )
                np.testing.assert_allclose(
                    observed, expected, rtol=2e-11, atol=2e-13
                )

    def test_all_real_bayes_grids_use_identical_uniform_nodes(self) -> None:
        config = realdata.RealDataConfig(
            bayes_quadrature_nodes=8,
            gumbel_lookup_size=1_001,
            dirichlet_c_nodes=1_001,
        )
        scorers = realdata.ModelScorers(config, 31)
        expected_delta, expected_weight = (
            realdata.dirichlet.gauss_legendre_delta_grid(
                *realdata.benchmark.effective_prior_support(
                    realdata._benchmark_config(config, VOCABULARY)
                    if "VOCABULARY" in dir() else
                    realdata._benchmark_config(config, 1000)
                ),
                config.bayes_quadrature_nodes,
            )
        )
        np.testing.assert_array_equal(scorers.deltas, expected_delta)
        np.testing.assert_array_equal(scorers.delta_weights, expected_weight)
        np.testing.assert_array_equal(scorers.dirichlet_grid.deltas, expected_delta)
        np.testing.assert_allclose(
            scorers.dirichlet_grid.delta_weights,
            expected_weight,
            rtol=0.0,
            atol=2e-16,
        )
        np.testing.assert_array_equal(
            scorers.gumbel_lookup._top_exponents,
            expected_delta / (1.0 - expected_delta),
        )
        probe_logits = np.array([-4.0, 0.3, 6.0])
        log_r = -np.logaddexp(0.0, -probe_logits)
        manual_density = (
            np.exp(
                log_r[:, None]
                * (expected_delta / (1.0 - expected_delta))[None, :]
            )
            + (31 - 1)
            * np.exp(
                log_r[:, None]
                * ((31 - 1) / expected_delta - 1.0)[None, :]
            )
        ) @ expected_weight
        np.testing.assert_allclose(
            scorers.gumbel_lookup._direct_log_density_at_logits(probe_logits),
            np.log(manual_density),
            rtol=0.0,
            atol=2e-15,
        )

    def test_real_config_restores_uniform_prior(self) -> None:
        config = realdata.RealDataConfig()
        self.assertEqual(config.delta_prior, "uniform")
        self.assertEqual((config.delta_low, config.delta_high), (0.001, 0.5))
        config.validate()
        with self.assertRaisesRegex(ValueError, "requires uniform Delta"):
            realdata.RealDataConfig(delta_prior="other").validate()
        args = realdata.parse_args(
            ["--delta-low", "0", "--delta-high", "0.1", "--no-download"]
        )
        self.assertEqual((args.delta_low, args.delta_high), (0.0, 0.1))
        config = realdata.RealDataConfig(
            delta_low=args.delta_low,
            delta_high=args.delta_high,
        )
        config.validate()

    def test_real_data_inverse_reference_uses_normalized_one_e_minus_six_floor(self) -> None:
        config = realdata._benchmark_config(realdata.RealDataConfig(), 1000)
        pivots = np.array([[0.2, 0.999]])
        delta = 0.01
        expected = np.log(
            np.maximum(1.0 - pivots / (1.0 - delta), 1e-6)
            / (1.0 - pivots)
            / (1.0 - delta)
        )
        observed = realdata.real_data_inverse_score(
            pivots, "h_dif_star_0.01", config
        )
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)

    def test_method_menus_include_full_frozen_sets_and_real_gumbel_point(self) -> None:
        self.assertTrue(set(realdata.benchmark.GUMBEL_METHODS) < set(realdata.GUMBEL_METHODS))
        self.assertTrue(set(realdata.benchmark.INVERSE_METHODS) < set(realdata.INVERSE_METHODS))
        self.assertIn("h_gum_star_0.1", realdata.GUMBEL_METHODS)
        self.assertIn(realdata.benchmark.DIRICHLET_SHARED_METHOD, realdata.GUMBEL_METHODS)
        self.assertIn("bayes_shared", realdata.INVERSE_METHODS)

    def test_model_stream_codes_do_not_depend_on_requested_order(self) -> None:
        self.assertEqual(realdata.MODEL_CODES["1p3B"], 0)
        self.assertEqual(realdata.MODEL_CODES["2p7B"], 1)

    def test_replay_hash_validation_fails_closed(self) -> None:
        zeros = np.zeros((2, 3), dtype=float)
        with self.assertRaisesRegex(RuntimeError, "fail-closed PRF replay mismatch"):
            realdata.validate_replay_hashes("1p3B", zeros, zeros, zeros)

    def test_rao_blackwellized_mcse_integrates_boundary_randomization(self) -> None:
        scores = np.array([0.0, 1.0, 1.0, 2.0])
        rate, mcse, contributions = realdata._rao_blackwellized_summary(
            scores, 1.0, 0.25
        )
        np.testing.assert_array_equal(contributions, [0.0, 0.25, 0.25, 1.0])
        self.assertEqual(rate, 0.375)
        self.assertAlmostEqual(
            mcse, float(np.std(contributions, ddof=1) / np.sqrt(4)), places=15
        )

    def test_seed_context_summary_matches_skipgram_addressing(self) -> None:
        prompts = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        tokens = np.array([[9, 10, 11], [9, 12, 13]])
        summary = realdata._seed_context_summary(prompts, tokens, horizon=3)
        # The addressed contexts are [1,2,3] and [5,6,7].
        self.assertEqual(summary["n_positions"], 6)
        self.assertEqual(summary["n_distinct_context_tokens"], 6)
        self.assertEqual(summary["fraction_positions_beyond_first_occurrence"], 0.0)

    def test_calibration_randomized_boundary_targets_alpha(self) -> None:
        values = np.repeat(np.arange(10, dtype=float), 100).reshape(100, 10)
        cutoffs, gammas = realdata._calibrate({"score": values}, 0.05)
        rate = realdata.benchmark.expected_rejection_rate(
            values, cutoffs["score"], gammas["score"]
        )
        np.testing.assert_allclose(rate, 0.05, rtol=0.0, atol=1e-15)



class UnionTailRuleTests(unittest.TestCase):
    def test_ladder_uses_the_real_vocabulary_not_the_simulation_one(self) -> None:
        # The whole point of the layer is that the tail width must scale with
        # the deployed vocabulary; hardcoding the simulation's 999 would undo it.
        import dirichlet_detector as dd

        for model, spec in realdata.MODEL_SPECS.items():
            tail_size = int(spec["vocabulary_size"]) - 1
            ladder = dd.dyadic_tail_width_grid(tail_size)
            self.assertEqual(ladder[-1], tail_size)
            self.assertNotEqual(ladder[-1], 999)
            self.assertGreater(len(ladder), 6)

    def test_rule_is_registered_for_gumbel_only(self) -> None:
        import benchmark_paper_experiment as benchmark

        self.assertIn(benchmark.UNIONTAIL_SHARED_METHOD, realdata.GUMBEL_METHODS)
        self.assertNotIn(benchmark.UNIONTAIL_SHARED_METHOD, realdata.INVERSE_METHODS)
        self.assertEqual(
            realdata.METHOD_ORIGINS[benchmark.UNIONTAIL_SHARED_METHOD],
            "bayesian_union_tail",
        )


if __name__ == "__main__":
    unittest.main()
