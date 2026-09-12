"""Resume and provenance regressions without downloads or an upstream clone."""

import contextlib
import copy
import hashlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import generate_temperature_matched as gen
import build_prompts as builder


class GenerationResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / "run.npz"
        self.prompts = torch.zeros((2, 50), dtype=torch.long)
        self.prompt_record = {
            "prompt_source": "released tensor",
            "prompt_table_sha256": gen.prompt_sha256(self.prompts.numpy()),
            "selected_prompt_sha256": gen.prompt_sha256(self.prompts.numpy()),
        }
        self.argv = ["--model", "1p3B", "--temps", "0.1", "--methods", "raw",
                     "--documents", "2", "--tokens", "3", "--batch", "1",
                     "--device", "cpu", "--out", str(self.out)]
        self.upstream = patch.object(gen, "upstream_commit", return_value=gen.EXPECTED_UPSTREAM_COMMIT).start()
        self.addCleanup(patch.stopall)
        self.model = patch.object(gen, "load_model", return_value=object()).start()
        self.watermarks = patch.object(gen, "load_upstream", return_value=(object(), object())).start()
        self.prompt_loader = patch.object(gen, "load_prompts", return_value=(self.prompts, self.prompt_record)).start()
        self.generate = patch.object(gen, "generate", side_effect=self.fake_generate).start()

    @staticmethod
    def fake_generate(model, prompts, *, m, method, **kwargs):
        shape = (len(prompts), m)
        tokens = np.ones(shape, dtype=np.int64)
        top = np.full(shape, 0.5)
        y = np.full(shape, -0.2 if method == "transform" else 0.2) if method != "raw" else None
        u = np.full(shape, 0.3) if method == "transform" else None
        eta = np.full(shape, 0.5) if method == "transform" else None
        return tokens, y, top, u, eta

    def run_main(self, extra=()):
        with contextlib.redirect_stdout(io.StringIO()):
            gen.main(self.argv + list(extra))

    def read(self):
        with np.load(self.out, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    def test_completed_resume_is_byte_preserving_and_does_not_load_model(self):
        self.run_main()
        before = self.out.read_bytes()
        self.model.reset_mock()
        self.generate.reset_mock()
        self.run_main()
        self.assertEqual(before, self.out.read_bytes())
        self.model.assert_not_called()
        self.generate.assert_not_called()

    def test_extending_methods_and_temperatures_preserves_old_arms(self):
        self.run_main()
        first = self.read()["t0.1_raw__tokens"].copy()
        self.run_main(["--temps", "0.1,0.5", "--methods", "raw,gumbel,transform"])
        payload = self.read()
        record = json.loads(payload[gen.METADATA_KEY].item())
        self.assertEqual(len(record["arms"]), 6)
        np.testing.assert_array_equal(first, payload["t0.1_raw__tokens"])
        self.assertEqual(record["batch"], 1)
        self.assertEqual(record["selected_prompt_sha256"], self.prompt_record["selected_prompt_sha256"])
        self.assertIn("t0.5_transform__eta", payload)

    def test_incompatible_resume_fails_before_model_load_without_mutation(self):
        self.run_main()
        before = self.out.read_bytes()
        changes = [("--model", "2p7B"), ("--documents", "3"), ("--tokens", "4"),
                   ("--batch", "2"), ("--seed", "3"), ("--prompt-offset", "1")]
        for change in changes:
            with self.subTest(change=change):
                self.model.reset_mock()
                with self.assertRaisesRegex(ValueError, "Use a new --out"):
                    self.run_main(change)
                self.model.assert_not_called()
                self.assertEqual(before, self.out.read_bytes())

    def test_changed_actual_prompt_bytes_are_rejected(self):
        self.run_main()
        self.prompt_record["selected_prompt_sha256"] = "changed"
        self.model.reset_mock()
        with self.assertRaisesRegex(ValueError, "selected_prompt_sha256"):
            self.run_main()
        self.model.assert_not_called()

    def test_legacy_npz_is_not_relabelled(self):
        np.savez_compressed(self.out, t0_1_raw__tokens=np.ones((1, 1)))
        before = self.out.read_bytes()
        with self.assertRaisesRegex(ValueError, "no trustworthy run metadata"):
            self.run_main()
        self.model.assert_not_called()
        self.assertEqual(before, self.out.read_bytes())

    def test_existing_companion_and_shape_failures_prevent_skip(self):
        self.run_main()
        original = self.read()
        for defect in ("missing", "shape", "stray"):
            with self.subTest(defect=defect):
                payload = dict(original)
                if defect == "missing":
                    del payload["t0.1_raw__top_probs"]
                elif defect == "shape":
                    payload["t0.1_raw__top_probs"] = np.zeros((1, 1))
                else:
                    payload["t0.5_raw__tokens"] = np.ones((2, 3), dtype=np.int64)
                np.savez_compressed(self.out, **payload)
                before = self.out.read_bytes()
                self.model.reset_mock()
                with self.assertRaisesRegex(ValueError, "Cannot resume"):
                    self.run_main()
                self.model.assert_not_called()
                self.assertEqual(before, self.out.read_bytes())

    def test_changed_environment_and_model_revision_are_rejected(self):
        self.run_main()
        payload = self.read()
        saved = json.loads(payload[gen.METADATA_KEY].item())
        for field, value in (("device", "mps"), ("model_revision", "other"),
                             ("upstream_commit", "other"), ("versions", {})):
            record = copy.deepcopy(saved)
            record[field] = value
            payload[gen.METADATA_KEY] = np.array(json.dumps(record))
            np.savez_compressed(self.out, **payload)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                self.run_main()

    def test_failed_atomic_write_preserves_checkpoint_and_cleans_temporary(self):
        self.run_main()
        before = self.out.read_bytes()
        with patch.object(gen.np, "savez_compressed", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.run_main(["--temps", "0.1,0.5"])
        self.assertEqual(before, self.out.read_bytes())
        self.assertEqual(list(self.out.parent.iterdir()), [self.out])

    def test_new_bad_arm_does_not_replace_previous_checkpoint(self):
        self.run_main()
        before = self.out.read_bytes()
        self.generate.side_effect = lambda *a, **kw: (np.ones((1, 1), dtype=int), None, np.zeros((1, 1)), None, None)
        with self.assertRaisesRegex(ValueError, "shape"):
            self.run_main(["--temps", "0.1,0.5"])
        self.assertEqual(before, self.out.read_bytes())

    def test_v2_manifest_cannot_resume_as_per_prompt_rng(self):
        self.run_main()
        payload = self.read()
        meta = json.loads(payload[gen.METADATA_KEY].item())
        meta["schema_version"] = 2
        payload[gen.METADATA_KEY] = np.array(json.dumps(meta))
        np.savez_compressed(self.out, **payload)
        before = self.out.read_bytes()
        self.model.reset_mock()
        with self.assertRaisesRegex(ValueError, "unsupported or legacy"):
            self.run_main()
        self.model.assert_not_called()
        self.assertEqual(before, self.out.read_bytes())

    def test_in_range_array_edit_is_detected_by_manifest_digest(self):
        self.run_main()
        payload = self.read()
        payload["t0.1_raw__tokens"][0, 0] = 2
        np.savez_compressed(self.out, **payload)
        before = self.out.read_bytes()
        with self.assertRaisesRegex(ValueError, "array digests"):
            self.run_main()
        self.assertEqual(before, self.out.read_bytes())

    def test_command_assigns_distinct_global_prompt_streams(self):
        self.run_main()
        streams = [call.kwargs["raw_generators"][0]
                   for call in self.generate.call_args_list]
        self.assertEqual(len(streams), 2)
        self.assertNotEqual(streams[0].initial_seed(), streams[1].initial_seed())
        meta = json.loads(self.read()[gen.METADATA_KEY].item())
        self.assertEqual(meta["schema_version"], 3)
        self.assertEqual(meta["raw_rng_scheme"], gen.RAW_RNG_SCHEME)


class PromptValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "prompts.npy"
        self.reference = np.arange(100, dtype=np.int64).reshape(2, 50)
        self.table = np.vstack((self.reference, np.full((1, 50), 2)))
        self.args = SimpleNamespace(model="1p3B", prompts=self.path, prompt_offset=0, documents=2)
        self.patcher = patch("real_data_experiment.load_released_model_data",
                             return_value=(SimpleNamespace(prompts=self.reference), []))
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.write_table()

    def write_table(self):
        np.save(self.path, self.table)
        self.meta = {
            "model": "1p3B", "tokenizer": gen.SPECS["1p3B"][0],
            "tokenizer_revision": gen.SPECS["1p3B"][2],
            "c4_repo": builder.C4_REPO, "c4_revision": builder.C4_REVISION,
            "c4_file": builder.C4_FILE, "c4_sha256": builder.C4_SHA256,
            "prompt_tokens": 50, "continuation_tokens": 200,
            "buffer_tokens": 20, "truncate_at": 2028,
            "prompts": len(self.table), "released_rows_verified": len(self.reference),
            "output_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
        }
        self.path.with_suffix(".json").write_text(json.dumps(self.meta))

    def test_custom_and_released_selected_prompt_digest_agree(self):
        custom, cmeta = gen.load_prompts(self.args)
        self.args.prompts = None
        released, rmeta = gen.load_prompts(self.args)
        torch.testing.assert_close(custom, released)
        self.assertEqual(cmeta["selected_prompt_sha256"], rmeta["selected_prompt_sha256"])
        self.assertNotEqual(cmeta["prompt_table_sha256"], rmeta["prompt_table_sha256"])

    def test_released_prompt_offset_is_applied_and_bounds_checked(self):
        self.args.prompts = None
        self.args.prompt_offset, self.args.documents = 1, 1
        selected, record = gen.load_prompts(self.args)
        np.testing.assert_array_equal(selected, self.reference[1:])
        self.assertEqual(record["selected_prompt_sha256"], gen.prompt_sha256(self.reference[1:]))
        for offset, documents in ((1, 2), (-1, 1), (0, 3), (0, 0)):
            self.args.prompt_offset, self.args.documents = offset, documents
            with self.subTest(offset=offset, documents=documents), self.assertRaisesRegex(ValueError, "requested"):
                gen.load_prompts(self.args)

    def test_missing_sidecar_and_missing_or_wrong_fields_fail(self):
        self.path.with_suffix(".json").unlink()
        with self.assertRaisesRegex(ValueError, "sidecar is missing"):
            gen.load_prompts(self.args)
        for key in ("output_sha256", "model", "tokenizer_revision", "prompts", "c4_sha256"):
            self.write_table()
            del self.meta[key]
            self.path.with_suffix(".json").write_text(json.dumps(self.meta))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                gen.load_prompts(self.args)

    def test_digest_mismatch_is_rejected(self):
        self.table[2, 0] = 3
        np.save(self.path, self.table)
        with self.assertRaisesRegex(ValueError, "output_sha256"):
            gen.load_prompts(self.args)

    def test_first_released_rows_must_match_even_with_matching_file_digest(self):
        self.table[0, 0] = 5
        self.write_table()
        with self.assertRaisesRegex(ValueError, "reproduce all released"):
            gen.load_prompts(self.args)

    def test_prompt_shape_dtype_and_token_range_are_checked(self):
        for table in (np.zeros((3, 49), dtype=int), self.table.astype(float),
                      np.full((3, 50), -1), np.full((3, 50), gen.SPECS["1p3B"][1])):
            self.table = table
            self.write_table()
            with self.subTest(shape=table.shape, dtype=table.dtype), self.assertRaisesRegex(ValueError, "integer"):
                gen.load_prompts(self.args)


class PayloadValidationTests(unittest.TestCase):
    @staticmethod
    def transform_payload(vocab_size, uniform):
        from inverse_batched_sampling import transform_Y_batched

        tag = "t0.1_transform"
        args = SimpleNamespace(model="1p3B", seed=20260902)
        y, u, eta = transform_Y_batched(
            torch.tensor([[0]]),
            torch.arange(vocab_size).reshape(1, vocab_size),
            torch.tensor([[uniform]], dtype=torch.float32),
        )
        metadata = {
            "documents": 1,
            "tokens": 1,
            "vocab_size": vocab_size,
            "model_key": args.model,
            "seed": args.seed,
            "arms": {tag: gen.arm_spec(args, 0.1, "transform")},
        }
        payload = {
            gen.METADATA_KEY: np.array(json.dumps(metadata)),
            f"{tag}__tokens": np.array([[0]], dtype=np.int64),
            f"{tag}__top_probs": np.array([[0.5]], dtype=np.float32),
            f"{tag}__Y": y.numpy(),
            f"{tag}__U": u.numpy(),
            f"{tag}__eta": eta.numpy(),
        }
        return payload, metadata, tag

    def test_rank_zero_eta_is_valid_at_actual_opt_vocabulary(self):
        vocab_size = gen.SPECS["1p3B"][1]
        payload, metadata, tag = self.transform_payload(vocab_size, 0.4)
        eta = payload[f"{tag}__eta"][0, 0]
        self.assertLess(eta, 0)
        self.assertAlmostEqual(float(eta), -1 / (vocab_size - 1), places=11)
        gen.validate_payload(payload, metadata)

    def test_small_vocabulary_accepts_shifted_eta_and_y_below_minus_one(self):
        payload, metadata, tag = self.transform_payload(7, 0.99)
        self.assertAlmostEqual(float(payload[f"{tag}__eta"][0, 0]), -1 / 6, places=6)
        self.assertLess(payload[f"{tag}__Y"][0, 0], -1)
        gen.validate_payload(payload, metadata)

    def test_shifted_bounds_allow_roundoff_but_reject_out_of_support_values(self):
        payload, metadata, tag = self.transform_payload(7, 1.0)
        eta_key, y_key = f"{tag}__eta", f"{tag}__Y"
        payload[eta_key][0, 0] = np.nextafter(payload[eta_key][0, 0], -np.inf)
        payload[y_key][0, 0] = np.nextafter(payload[y_key][0, 0], -np.inf)
        gen.validate_payload(payload, metadata)

        for key, value in ((eta_key, -1 / 6 - 0.01),
                           (eta_key, 5 / 6 + 0.01),
                           (y_key, -7 / 6 - 0.01),
                           (y_key, 0.01)):
            candidate = {name: array.copy() for name, array in payload.items()}
            candidate[key][0, 0] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(
                    ValueError, f"invalid values in {key}"):
                gen.validate_payload(candidate, metadata)


class ConfigurationTests(unittest.TestCase):
    @staticmethod
    def flat_model():
        class Model:
            def __call__(self, inputs, **kwargs):
                return SimpleNamespace(logits=torch.zeros((len(inputs), inputs.shape[1], 7)),
                                       past_key_values=None)
        return Model()

    def draw_prompts(self, prompts, offset=0, tokens=12):
        return gen.generate(self.flat_model(), prompts, vocab_size=7, m=tokens,
                            temperature=.3, method="raw", device="cpu",
                            raw_generators=gen.prompt_generators(4223981722, prompts, offset))[0]

    def test_per_prompt_rng_is_chunk_and_batch_invariant(self):
        prompts = torch.arange(300).reshape(6, 50)
        whole = self.draw_prompts(prompts)
        for batch in (1, 2, 4):
            chunks = [self.draw_prompts(prompts[start:start + batch], start)
                      for start in range(0, len(prompts), batch)]
            np.testing.assert_array_equal(whole, np.concatenate(chunks))
        np.testing.assert_array_equal(whole[3:], self.draw_prompts(prompts[3:], 3))

    def test_extension_does_not_restart_base_even_for_identical_prompt_text(self):
        prompts = torch.ones((2, 50), dtype=torch.long)
        base = gen.prompt_generators(4223981722, prompts, 0)
        extension = gen.prompt_generators(4223981722, prompts, 500)
        self.assertFalse({g.initial_seed() for g in base} & {g.initial_seed() for g in extension})
        self.assertFalse(np.array_equal(self.draw_prompts(prompts), self.draw_prompts(prompts, 500)))

    def test_prompt_content_and_arm_seed_are_part_of_stream_identity(self):
        prompt = torch.ones((1, 50), dtype=torch.long)
        seed = gen.prompt_generators(1, prompt, 0)[0].initial_seed()
        self.assertNotEqual(seed, gen.prompt_generators(2, prompt, 0)[0].initial_seed())
        self.assertNotEqual(seed, gen.prompt_generators(1, prompt + 1, 0)[0].initial_seed())

    def test_raw_stream_configuration_fails_closed(self):
        prompts = torch.zeros((2, 50), dtype=torch.long)
        common = dict(vocab_size=7, m=1, temperature=.3, method="raw", device="cpu")
        stream = torch.Generator()
        for kwargs in ({}, {"raw_generators": [stream]},
                       {"raw_generators": [stream, stream]},
                       {"raw_generator": stream, "raw_generators": [stream, torch.Generator()]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                gen.generate(self.flat_model(), prompts, **common, **kwargs)

    def test_raw_generation_uses_arm_rng_without_importing_upstream(self):
        class Model:
            def __call__(self, inputs, **kwargs):
                logits = torch.zeros((len(inputs), inputs.shape[1], 7))
                return SimpleNamespace(logits=logits, past_key_values=None)

        prompts = torch.zeros((3, 50), dtype=torch.long)
        with patch.object(gen, "load_upstream", side_effect=AssertionError("upstream imported")):
            first = gen.generate(Model(), prompts, vocab_size=7, m=8, temperature=0.5,
                                 method="raw", device="cpu",
                                 raw_generator=torch.Generator().manual_seed(42))
            # Unrelated global RNG draws must not perturb the arm's RNG stream.
            torch.rand(100)
            repeated = gen.generate(Model(), prompts, vocab_size=7, m=8, temperature=0.5,
                                    method="raw", device="cpu",
                                    raw_generator=torch.Generator().manual_seed(42))
        np.testing.assert_array_equal(first[0], repeated[0])
        self.assertEqual(first[0].shape, (3, 8))
        self.assertIsNone(first[1])

    def test_auto_device_order_and_explicit_unavailability(self):
        for cuda, mps, expected in ((True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")):
            with patch.object(torch.cuda, "is_available", return_value=cuda), \
                 patch.object(torch.backends.mps, "is_available", return_value=mps):
                self.assertEqual(gen.select_device("auto"), expected)
                self.assertEqual(gen.select_device("cpu"), "cpu")
                for device, available in (("cuda", cuda), ("mps", mps)):
                    if available:
                        self.assertEqual(gen.select_device(device), device)
                    else:
                        with self.assertRaisesRegex(ValueError, "unavailable"):
                            gen.select_device(device)

    def test_upstream_requires_expected_clean_commit(self):
        complete = lambda code, out: subprocess.CompletedProcess([], code, stdout=out, stderr="")
        for sha, dirty in (("wrong", ""), (gen.EXPECTED_UPSTREAM_COMMIT, " M sampling.py")):
            with patch("subprocess.run", side_effect=[complete(0, sha), complete(0, dirty)]):
                with self.assertRaises(ValueError):
                    gen.upstream_commit()
        with patch("subprocess.run", side_effect=[complete(0, gen.EXPECTED_UPSTREAM_COMMIT), complete(0, "")]):
            self.assertEqual(gen.upstream_commit(), gen.EXPECTED_UPSTREAM_COMMIT)

    def test_help_needs_no_upstream_or_model_import(self):
        with patch.object(gen, "load_upstream", side_effect=AssertionError("upstream imported")), \
             patch.object(gen, "load_model", side_effect=AssertionError("model loaded")), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                gen.main(["--help"])
        self.assertEqual(raised.exception.code, 0)

    def test_invalid_batch_counts_temperatures_and_methods_fail_early(self):
        base = ["--model", "1p3B", "--out", "unused.npz"]
        for option in (("--batch", "0"), ("--documents", "-1"), ("--tokens", "0"),
                       ("--prompt-offset", "-1"), ("--temps", "nan"),
                       ("--temps", "0.1,0.10000001"), ("--methods", "raw,bogus")):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    gen.parse_args(base + list(option))


if __name__ == "__main__":
    unittest.main()
