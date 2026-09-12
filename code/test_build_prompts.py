"""Tests for the C4 prompt-table builder.

The builder's correctness claim is that it reproduces upstream's filter, and
the evidence for that claim is the checked-in tables: their first 500 rows are
the released prompt tensor.  These tests guard the two constants that claim
turns on and the digest checks that keep the inputs and outputs pinned.
"""

from __future__ import annotations

import gzip
import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import build_prompts as bp

REPO = Path(__file__).resolve().parents[1]
EXTENSION = (REPO / "results/bayesian_paper_benchmark/real_model"
             / "temperature_matched/extension")


class FakeTokens:
    """The slice-then-``.numpy()`` surface ``build`` uses off a torch tensor."""

    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def __len__(self) -> int:
        return int(self.values.size)

    def __getitem__(self, item):
        return FakeTokens(self.values[item])

    def numpy(self) -> np.ndarray:
        return self.values


class FakeTokenizer:
    """One token per whitespace-separated word, so lengths are readable."""

    def encode(self, text, return_tensors=None, truncation=False, max_length=None):
        values = np.array([abs(hash(w)) % 50000 for w in text.split()], dtype=np.int64)
        if truncation and max_length is not None:
            values = values[:max_length]
        return [FakeTokens(values)]


class ConstantsTest(unittest.TestCase):
    def test_truncation_is_context_window_less_buffer(self) -> None:
        # 2028, not 2047.  The 20-token difference is invisible on documents
        # short enough to escape truncation and shifts the window by exactly
        # 20 positions on every document long enough to hit it; the released
        # prompts show the shifted window, so this constant is load-bearing.
        self.assertEqual(bp.BUFFER_TOKENS, 20)
        self.assertEqual(bp.TRUNCATE_AT, 2028)

    def test_window_sits_immediately_before_the_scored_continuation(self) -> None:
        self.assertEqual(bp.PROMPT_TOKENS, 50)
        self.assertEqual(bp.CONTINUATION_TOKENS, 200)


class BuildTest(unittest.TestCase):
    def _shard(self, lengths) -> Path:
        path = Path(self.tmp) / "shard.json.gz"
        with gzip.open(path, "wt") as fh:
            for i, n in enumerate(lengths):
                fh.write(json.dumps({"text": " ".join(f"w{i}x{j}" for j in range(n))}) + "\n")
        return path

    def setUp(self) -> None:
        self._tmpdir = __import__("tempfile").TemporaryDirectory()
        self.tmp = self._tmpdir.name
        self.addCleanup(self._tmpdir.cleanup)
        patcher = mock.patch.dict(
            "sys.modules",
            {"transformers": mock.Mock(AutoTokenizer=mock.Mock(
                from_pretrained=lambda name, revision=None: FakeTokenizer()))},
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_short_documents_are_skipped_not_padded(self) -> None:
        # 249 tokens is one short of the 250 a prompt plus its continuation
        # needs.  Accepting it would put continuation tokens in the prompt.
        shard = self._shard([249, 300, 249, 400])
        table, scanned = bp.build("1p3B", 2, shard)
        self.assertEqual(table.shape, (2, 50))
        self.assertEqual(scanned, 4)

    def test_prompt_is_the_window_before_the_continuation(self) -> None:
        shard = self._shard([300])
        table, _ = bp.build("1p3B", 1, shard)
        tok = FakeTokenizer()
        with gzip.open(shard, "rt") as fh:
            full = tok.encode(json.loads(fh.readline())["text"])[0].numpy()
        np.testing.assert_array_equal(table[0], full[-250:-200])

    def test_truncation_moves_the_window_to_a_fixed_offset(self) -> None:
        # Two documents of different lengths, both past the truncation point,
        # must yield windows at the same absolute offset -- that invariance is
        # what identified the constant in the first place.
        shard = self._shard([bp.TRUNCATE_AT + 100, bp.TRUNCATE_AT + 900])
        table, _ = bp.build("1p3B", 2, shard)
        tok = FakeTokenizer()
        with gzip.open(shard, "rt") as fh:
            docs = [tok.encode(json.loads(l)["text"])[0].numpy() for l in fh]
        offset = bp.TRUNCATE_AT - 250
        for row, full in zip(table, docs):
            np.testing.assert_array_equal(row, full[offset:offset + 50])

    def test_tokenizer_is_loaded_at_a_pinned_revision(self) -> None:
        # The tokenizer ships in the model repository, so an unpinned load
        # follows the branch head and a tokenizer change moves every prompt.
        seen = {}

        def capture(name, revision=None):
            seen["name"], seen["revision"] = name, revision
            return FakeTokenizer()

        with mock.patch.dict("sys.modules", {"transformers": mock.Mock(
                AutoTokenizer=mock.Mock(from_pretrained=capture))}):
            bp.build("1p3B", 1, self._shard([300]))
        self.assertEqual(seen["name"], bp.TOKENIZERS["1p3B"][0])
        self.assertEqual(seen["revision"], bp.TOKENIZERS["1p3B"][1])
        self.assertEqual(len(seen["revision"]), 40)

    def test_exhausted_shard_fails_rather_than_returning_short(self) -> None:
        shard = self._shard([300, 300])
        with self.assertRaises(SystemExit):
            bp.build("1p3B", 5, shard)


class DigestTest(unittest.TestCase):
    def test_corrupted_cache_is_rejected(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            (cache / Path(bp.C4_FILE).name).write_bytes(b"not the shard")
            with self.assertRaises(SystemExit) as ctx:
                bp.fetch_shard(cache)
            self.assertIn("digest mismatch", str(ctx.exception))

    def test_pinned_c4_revision_is_a_commit_not_a_branch(self) -> None:
        self.assertEqual(len(bp.C4_REVISION), 40)
        self.assertTrue(all(c in "0123456789abcdef" for c in bp.C4_REVISION))


class CheckedInTablesTest(unittest.TestCase):
    """The tables in the repository must still match the record beside them."""

    def test_tables_match_their_recorded_digest_and_shape(self) -> None:
        for model in ("1p3B", "2p7B"):
            npy = EXTENSION / f"prompts_{model}.npy"
            meta_path = EXTENSION / f"prompts_{model}.json"
            if not npy.exists() or not meta_path.exists():
                self.skipTest(f"{model} prompt table not present")
            meta = json.loads(meta_path.read_text())
            with self.subTest(model=model):
                self.assertEqual(bp.sha256(npy), meta["output_sha256"])
                self.assertEqual(np.load(npy).shape, (meta["prompts"], 50))
                self.assertEqual(meta["truncate_at"], bp.TRUNCATE_AT)
                self.assertEqual(meta["c4_sha256"], bp.C4_SHA256)
                self.assertEqual(meta["released_rows_verified"], 500)

    def test_first_rows_are_the_released_prompt_tensor(self) -> None:
        data_dir = (REPO / "results/bayesian_paper_benchmark/real_model"
                    / "upstream_assets")
        if not data_dir.exists():
            self.skipTest("upstream assets not present")
        import real_data_experiment as real
        for model in ("1p3B", "2p7B"):
            npy = EXTENSION / f"prompts_{model}.npy"
            if not npy.exists():
                self.skipTest(f"{model} prompt table not present")
            released, _ = real.load_released_model_data(model, data_dir)
            reference = np.asarray(released.prompts)
            with self.subTest(model=model):
                np.testing.assert_array_equal(
                    np.load(npy)[: reference.shape[0]], reference)


if __name__ == "__main__":
    unittest.main()
