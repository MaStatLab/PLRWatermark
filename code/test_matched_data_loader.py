"""Manifest, legacy allowlist, and paired global-prompt loading regressions."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import warnings
import zipfile
from unittest.mock import patch

import numpy as np

import generate_temperature_matched as gen
import matched_data_loader as loader


class MatchedLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prompts = np.arange(300, dtype=np.int64).reshape(6, 50)

    def archive(self, name, offset=0, documents=2, tokens=3, methods=("raw", "gumbel"),
                temperature=.3, **changes):
        args = SimpleNamespace(model="1p3B", prompt_offset=offset, documents=documents,
                               tokens=tokens, batch=2, seed=7)
        prompt_record = {"prompt_source": "fixture.npy",
                         "prompt_table_sha256": gen.prompt_sha256(self.prompts),
                         "selected_prompt_sha256": gen.prompt_sha256(self.prompts[offset:offset + documents])}
        with patch.object(gen, "upstream_commit", return_value=gen.EXPECTED_UPSTREAM_COMMIT):
            metadata = gen.run_metadata(args, gen.SPECS[args.model][0],
                                        gen.SPECS[args.model][2], "cpu", prompt_record)
        arrays = {}
        for method in methods:
            tag = f"t{temperature:g}_{method}"
            metadata["arms"][tag] = gen.arm_spec(args, temperature, method)
            arrays[tag + "__tokens"] = np.ones((documents, tokens), dtype=np.int64)
            arrays[tag + "__top_probs"] = np.full((documents, tokens), .5, dtype=np.float32)
            if method != "raw":
                arrays[tag + "__Y"] = np.full((documents, tokens), .2, dtype=np.float32)
        metadata["array_sha256"] = {key: gen.array_sha256(value) for key, value in arrays.items()}
        metadata.update(changes)
        arrays[gen.METADATA_KEY] = np.array(json.dumps(metadata))
        path = self.root / name
        np.savez_compressed(path, **arrays)
        return path

    def cells(self, paths, **kwargs):
        return loader.paired_cells(paths, self.prompts, "1p3B", **kwargs)

    def test_disjoint_blocks_join_in_global_order_and_preserve_metadata(self):
        base = self.archive("base.npz")
        extension = self.archive("ext.npz", offset=2, documents=4)
        cell, = self.cells([extension, base])
        np.testing.assert_array_equal(cell.prompts, self.prompts)
        np.testing.assert_array_equal(cell.prompt_indices, np.arange(6))
        self.assertEqual(cell.raw["tokens"].shape, (6, 3))
        self.assertEqual(len(cell.provenance), 2)
        self.assertEqual(cell.provenance[0]["generation_manifest"]["schema_version"], 3)
        self.assertEqual(cell.provenance[0]["mode"], "verified_manifest_v3")

    def test_separate_paired_arms_with_same_prompt_identity_are_valid(self):
        raw = self.archive("raw.npz", methods=("raw",))
        wm = self.archive("wm.npz", methods=("gumbel",))
        cell, = self.cells([raw, wm])
        self.assertEqual(len(cell.prompts), 2)
        self.assertEqual(len(cell.provenance), 2)

    def test_unmanifested_unlisted_archive_is_always_rejected(self):
        path = self.root / "unknown.npz"
        np.savez_compressed(path, t0_3_raw__tokens=np.ones((2, 3)))
        for allowed in (False, True):
            with self.subTest(allowed=allowed), self.assertRaises(ValueError):
                loader.read_archive(path, allow_legacy=allowed)

    def test_known_filename_with_wrong_bytes_is_not_allowlisted(self):
        path = self.root / "matched_1p3B_lo.npz"
        np.savez_compressed(path, t0_3_raw__tokens=np.ones((2, 3)))
        with self.assertRaisesRegex(ValueError, "not SHA-allowlisted"):
            loader.read_archive(path, allow_legacy=True)

    def test_v2_manifest_is_not_implicitly_upgraded(self):
        path = self.archive("old.npz", schema_version=2)
        with self.assertRaisesRegex(ValueError, "schema v3"):
            loader.read_archive(path, allow_legacy=True)

    def test_missing_rng_scheme_and_wrong_model_revision_are_rejected(self):
        for change in ({"raw_rng_scheme": None}, {"model_revision": "other"},
                       {"upstream_commit": "other"}, {"seed": "seven"}):
            path = self.archive("bad.npz", **change)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "manifest"):
                loader.read_archive(path)

    def test_unknown_payload_array_is_rejected(self):
        path = self.archive("bad.npz")
        with np.load(path, allow_pickle=False) as archive:
            arrays = dict(archive)
        arrays["undocumented"] = np.ones(1)
        np.savez_compressed(path, **arrays)
        with self.assertRaisesRegex(ValueError, "completed-arm manifest"):
            loader.read_archive(path)

    def test_in_range_array_tampering_is_rejected(self):
        path = self.archive("bad.npz")
        with np.load(path, allow_pickle=False) as archive:
            arrays = dict(archive)
        arrays["t0.3_raw__tokens"][0, 0] = 7
        np.savez_compressed(path, **arrays)
        with self.assertRaisesRegex(ValueError, "array digests"):
            loader.read_archive(path)

    def test_duplicate_npz_members_are_rejected_before_dictionary_conversion(self):
        path = self.archive("duplicate.npz")
        with zipfile.ZipFile(path, "a") as archive:
            member = archive.namelist()[0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr(member, archive.read(member))
        with self.assertRaisesRegex(ValueError, "duplicate NPZ"):
            loader.read_archive(path)

    def test_global_offset_must_match_actual_selected_prompt_digest(self):
        path = self.archive("bad.npz", prompt_offset=1)
        with self.assertRaisesRegex(ValueError, "digest/offset"):
            self.cells([path])

    def test_reordered_actual_prompt_rows_are_rejected(self):
        path = self.archive("base.npz")
        self.prompts[[0, 1]] = self.prompts[[1, 0]]
        with self.assertRaisesRegex(ValueError, "selected prompt"):
            self.cells([path])

    def test_duplicate_and_overlapping_blocks_are_rejected(self):
        base = self.archive("base.npz")
        for offset in (0, 1):
            other = self.archive("overlap.npz", offset=offset)
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, "overlapping"):
                self.cells([base, other])
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.cells([base, base])

    def test_gap_or_missing_leading_prompts_is_rejected(self):
        base = self.archive("base.npz")
        later = self.archive("gap.npz", offset=3)
        for paths in ([base, later], [later]):
            with self.subTest(paths=paths), self.assertRaisesRegex(ValueError, "gap"):
                self.cells(paths)

    def test_extension_cannot_change_token_budget(self):
        base = self.archive("base.npz")
        extension = self.archive("ext.npz", offset=2, tokens=4)
        with self.assertRaisesRegex(ValueError, "token budgets"):
            self.cells([base, extension])

    def test_paired_arms_cannot_silently_truncate_documents_or_tokens(self):
        raw = self.archive("raw.npz", methods=("raw",))
        for options in ({"documents": 3}, {"tokens": 4}):
            wm = self.archive("wm.npz", methods=("gumbel",), **options)
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "pairing or token budgets"):
                self.cells([raw, wm])

    def test_incompatible_generation_identity_is_rejected(self):
        base = self.archive("base.npz")
        other = self.archive("ext.npz", offset=2, device="mps")
        with self.assertRaisesRegex(ValueError, "generation identities"):
            self.cells([base, other])

    def test_rounded_tags_cannot_hide_different_arm_temperatures(self):
        raw = self.archive("raw.npz", methods=("raw",), temperature=.1)
        wm = self.archive("wm.npz", methods=("gumbel",), temperature=.10000001)
        with self.assertRaisesRegex(ValueError, "sampling temperatures differ"):
            self.cells([raw, wm])

    def test_rounded_tags_cannot_hide_different_extension_temperatures(self):
        base = self.archive("base.npz", temperature=.1)
        extension = self.archive("ext.npz", offset=2, temperature=.10000001)
        with self.assertRaisesRegex(ValueError, "unequal temperatures"):
            self.cells([base, extension])

    def test_unpaired_temperature_cannot_be_silently_skipped(self):
        path = self.archive("raw.npz", methods=("raw",))
        with self.assertRaisesRegex(ValueError, "incomplete or unpaired"):
            self.cells([path])

    def test_different_temperatures_may_share_prompt_rows(self):
        paths = [self.archive("low.npz", temperature=.2),
                 self.archive("high.npz", temperature=.3)]
        self.assertEqual([cell.temperature for cell in self.cells(paths)], ["0.2", "0.3"])

    def test_ambiguous_extension_locations_are_rejected(self):
        (self.root / "extension").mkdir()
        for folder in (self.root, self.root / "extension"):
            (folder / "ext_1p3B_lo.npz").touch()
        with self.assertRaisesRegex(ValueError, "duplicate extension"):
            loader.discover_archives(self.root, "1p3B")


class LegacyArchiveTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1] / "results/bayesian_paper_benchmark/real_model/temperature_matched"

    def test_all_eight_actual_archives_require_opt_in_and_retain_unknown_history(self):
        for name, (digest, model, offset, documents, tokens) in loader.LEGACY_ARCHIVES.items():
            path = self.ROOT / ("extension" if offset else "") / name
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "allow-legacy-archives"):
                    loader.read_archive(path)
                archive = loader.read_archive(path, allow_legacy=True)
                self.assertEqual((archive.model, archive.offset, archive.documents, archive.tokens),
                                 (model, offset, documents, tokens))
                self.assertIsNone(archive.metadata)
                self.assertIsNone(archive.provenance["generation_manifest"])
                self.assertEqual(archive.provenance["historical_rng_and_model_revision"], "unknown")
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_all_current_paired_cells_keep_original_sizes_and_global_order(self):
        for model in gen.SPECS:
            prompts = loader.load_prompt_table(self.ROOT, model)
            for offset, documents in ((0, 500), (500, 2000)):
                normalized = prompts[offset:offset + documents].astype(">i4")
                self.assertEqual(gen.prompt_sha256(normalized),
                                 loader.LEGACY_PROMPT_BLOCK_SHA256[(model, offset, documents)])
            cells = loader.load_cells(self.ROOT, model, allow_legacy=True)
            self.assertEqual(len(cells), 8)
            for cell in cells:
                n = 2500 if float(cell.temperature) in (.2, .3, .4, .5) else 500
                self.assertEqual(cell.raw["tokens"].shape, cell.watermarked["tokens"].shape)
                self.assertEqual(len(cell.prompts), n)
                np.testing.assert_array_equal(cell.prompt_indices, np.arange(n))
                band = "lo" if float(cell.temperature) <= .3 else "hi"
                paths = [self.ROOT / f"matched_{model}_{band}.npz"]
                if n == 2500:
                    paths.append(self.ROOT / "extension" / f"ext_{model}_{band}.npz")
                for method, actual in (("raw", cell.raw), ("gumbel", cell.watermarked)):
                    for field, values in actual.items():
                        pieces = []
                        for path in paths:
                            with np.load(path, allow_pickle=False) as stored:
                                pieces.append(stored[f"t{cell.temperature}_{method}__{field}"])
                        expected = np.concatenate(pieces)
                        self.assertEqual(values.dtype, expected.dtype)
                        np.testing.assert_array_equal(values, expected)

    def test_changed_legacy_extension_prompt_is_rejected_with_unchanged_archives(self):
        for model in gen.SPECS:
            prompts = loader.load_prompt_table(self.ROOT, model).copy()
            prompts[500, 0] = (prompts[500, 0] + 1) % gen.SPECS[model][1]
            paths = loader.discover_archives(self.ROOT, model)
            with self.subTest(model=model), self.assertRaisesRegex(ValueError, "historical prompt block"):
                loader.paired_cells(paths, prompts, model, allow_legacy=True)

    def test_changed_legacy_base_prompt_is_rejected_with_unchanged_archives(self):
        for model in gen.SPECS:
            prompts = loader.load_prompt_table(self.ROOT, model).copy()
            prompts[0, 0] = (prompts[0, 0] + 1) % gen.SPECS[model][1]
            paths = loader.discover_archives(self.ROOT, model)
            with self.subTest(model=model), self.assertRaisesRegex(ValueError, "historical prompt block"):
                loader.paired_cells(paths, prompts, model, allow_legacy=True)

    def test_historical_and_new_generation_cannot_be_mixed(self):
        fixture = MatchedLoaderTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        path = fixture.archive("new.npz")
        legacy = self.ROOT / "matched_1p3B_lo.npz"
        with self.assertRaisesRegex(ValueError, "cannot mix historical"):
            fixture.cells([legacy, path], allow_legacy=True)


if __name__ == "__main__":
    unittest.main()
