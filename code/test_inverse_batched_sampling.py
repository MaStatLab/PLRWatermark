"""The batched inverse path must equal the released one, row for row.

The released ``transform_key_func`` asserts a batch of one, so the only way to
show the batched rewrite is faithful is to run the released functions one row
at a time and require element-wise equality with the batched call on the stacked
rows -- including the order in which draws are taken from the generator, which
determines the keys.
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

UPSTREAM = Path(
    os.environ.get(
        "WATERMARK_FRAMEWORK_DIR",
        Path(__file__).resolve().parents[1] / "third_party" / "WatermarkFramework",
    )
) / "real data"

if UPSTREAM.is_dir() and str(UPSTREAM) not in sys.path:
    sys.path.insert(0, str(UPSTREAM))

_previous_dont_write_bytecode = sys.dont_write_bytecode
try:
    # Provenance checks require a clean upstream tree.  Test collection must
    # not create an untracked __pycache__ in that checkout.
    sys.dont_write_bytecode = True
    try:
        import sampling as upstream
    except Exception:  # pragma: no cover - exercised only without the clone
        upstream = None
finally:
    sys.dont_write_bytecode = _previous_dont_write_bytecode

import inverse_batched_sampling as batched


@unittest.skipIf(upstream is None, f"upstream clone not present at {UPSTREAM}")
class EquivalenceTests(unittest.TestCase):
    """Every batched function reproduces the released one at batch size one."""

    VOCAB = 97
    KEY = 15_485_863
    C = 5
    SCHEME = "minhash_prf"

    def _prompts(self, rows, length=11, seed=0):
        rng = np.random.default_rng(seed)
        return torch.from_numpy(
            rng.integers(0, self.VOCAB, size=(rows, length)).astype(np.int64)
        )

    def test_key_func_matches_released_row_by_row(self):
        prompts = self._prompts(6, seed=1)
        got_xi, got_pi = batched.transform_key_func_batched(
            torch.Generator(), prompts, self.VOCAB, self.KEY, self.C, self.SCHEME
        )
        for row in range(prompts.shape[0]):
            want_xi, want_pi = upstream.transform_key_func(
                torch.Generator(), prompts[row].unsqueeze(0),
                self.VOCAB, self.KEY, self.C, self.SCHEME,
            )
            torch.testing.assert_close(got_xi[row].unsqueeze(0), want_xi, rtol=0, atol=0)
            torch.testing.assert_close(got_pi[row].unsqueeze(0), want_pi, rtol=0, atol=0)

    def test_key_depends_on_the_row_not_the_batch(self):
        """The released bug: seeding on the whole batch makes rows identical."""
        prompts = self._prompts(4, seed=2)
        _, pi = batched.transform_key_func_batched(
            torch.Generator(), prompts, self.VOCAB, self.KEY, self.C, self.SCHEME
        )
        self.assertFalse(
            bool((pi[0] == pi[1]).all()), "distinct contexts must give distinct keys"
        )

    def test_repeated_context_reproduces_the_key(self):
        """The seed is a function of the last c tokens, so a repeat must repeat."""
        prompts = self._prompts(2, seed=3)
        prompts[1] = prompts[0]
        xi, pi = batched.transform_key_func_batched(
            torch.Generator(), prompts, self.VOCAB, self.KEY, self.C, self.SCHEME
        )
        torch.testing.assert_close(xi[0], xi[1], rtol=0, atol=0)
        torch.testing.assert_close(pi[0], pi[1], rtol=0, atol=0)

    def test_inverse_permutation_matches_released(self):
        generator = torch.Generator().manual_seed(7)
        pi = torch.vstack([
            torch.randperm(self.VOCAB, generator=generator).unsqueeze(0)
            for _ in range(5)
        ])
        got = batched.inverse_permutation_batched(pi)
        for row in range(pi.shape[0]):
            torch.testing.assert_close(
                got[row], upstream.inverse_permutation(pi[row]), rtol=0, atol=0
            )

    def test_inverse_permutation_is_a_true_inverse(self):
        generator = torch.Generator().manual_seed(8)
        pi = torch.vstack([
            torch.randperm(self.VOCAB, generator=generator).unsqueeze(0)
            for _ in range(4)
        ])
        inv = batched.inverse_permutation_batched(pi)
        expected = torch.arange(self.VOCAB).expand_as(pi)
        torch.testing.assert_close(torch.gather(inv, 1, pi), expected, rtol=0, atol=0)

    def _probs(self, rows, seed):
        generator = torch.Generator().manual_seed(seed)
        raw = torch.rand((rows, self.VOCAB), generator=generator)
        return raw / raw.sum(dim=1, keepdim=True)

    def test_sampling_matches_released_row_by_row(self):
        rows = 5
        probs = self._probs(rows, 11)
        generator = torch.Generator().manual_seed(12)
        pi = torch.vstack([
            torch.randperm(self.VOCAB, generator=generator).unsqueeze(0)
            for _ in range(rows)
        ])
        xi = torch.rand((rows, 1), generator=generator)
        got = batched.transform_sampling_batched(probs, pi, xi)
        for row in range(rows):
            want = upstream.transform_sampling(
                probs[row].unsqueeze(0), pi[row].unsqueeze(0), xi[row].unsqueeze(0)
            )
            torch.testing.assert_close(got[row].unsqueeze(0), want, rtol=0, atol=0)

    def test_pivot_matches_released_row_by_row(self):
        rows = 5
        generator = torch.Generator().manual_seed(13)
        pi = torch.vstack([
            torch.randperm(self.VOCAB, generator=generator).unsqueeze(0)
            for _ in range(rows)
        ])
        xi = torch.rand((rows, 1), generator=generator)
        s = torch.randint(0, self.VOCAB, (rows, 1), generator=generator)
        got_y, got_u, got_eta = batched.transform_Y_batched(s, pi, xi)
        for row in range(rows):
            want_y, want_u, want_eta = upstream.transform_Y(
                s[row].unsqueeze(0), pi[row].unsqueeze(0), xi[row].unsqueeze(0)
            )
            torch.testing.assert_close(
                got_y[row].reshape(-1), want_y.reshape(-1), rtol=0, atol=0
            )
            torch.testing.assert_close(
                got_eta[row].reshape(-1), want_eta.reshape(-1), rtol=0, atol=0
            )
            torch.testing.assert_close(
                got_u[row].reshape(-1), want_u.reshape(-1), rtol=0, atol=0
            )

    def test_pivot_avoids_the_broadcast_trap(self):
        """A (batch,1) minus a (batch,) would silently become (batch, batch)."""
        rows = 6
        generator = torch.Generator().manual_seed(14)
        pi = torch.vstack([
            torch.randperm(self.VOCAB, generator=generator).unsqueeze(0)
            for _ in range(rows)
        ])
        xi = torch.rand((rows, 1), generator=generator)
        s = torch.randint(0, self.VOCAB, (rows, 1), generator=generator)
        y, _, _ = batched.transform_Y_batched(s, pi, xi)
        self.assertEqual(tuple(y.shape), (rows, 1))

    def test_pivot_is_in_range(self):
        rows = 8
        generator = torch.Generator().manual_seed(15)
        pi = torch.vstack([
            torch.randperm(self.VOCAB, generator=generator).unsqueeze(0)
            for _ in range(rows)
        ])
        xi = torch.rand((rows, 1), generator=generator)
        s = torch.randint(0, self.VOCAB, (rows, 1), generator=generator)
        y, u, eta = batched.transform_Y_batched(s, pi, xi)
        self.assertTrue(bool(((y <= 0) & (y >= -1.001)).all()))
        self.assertTrue(bool(((eta >= -0.001) & (eta <= 1.001)).all()))
        self.assertTrue(bool(((u >= 0) & (u <= 1)).all()))


@unittest.skipIf(upstream is None, f"upstream clone not present at {UPSTREAM}")
class EndToEndGenerationTests(unittest.TestCase):
    """Generating a batch must equal generating each row on its own.

    This is the claim the whole inverse arm rests on.  A stub model supplies
    row-wise deterministic logits, so any difference between the batched run and
    the row-by-row run can only come from the key, sampler or pivot code.
    """

    VOCAB = 61

    class _Output:
        def __init__(self, logits):
            self.logits = logits
            self.past_key_values = None

    class _RowwiseStub:
        """Logits depend only on the row's own tokens, never on its neighbours."""

        device = "cpu"

        def __init__(self, vocab):
            self.vocab = vocab

        def __call__(self, ids, past_key_values=None, attention_mask=None):
            rows = []
            for row in range(ids.shape[0]):
                generator = torch.Generator().manual_seed(
                    int(ids[row].sum().item()) % 100_000
                )
                rows.append(
                    torch.rand((1, ids.shape[1], self.vocab), generator=generator)
                )
            return EndToEndGenerationTests._Output(torch.cat(rows, dim=0))

    def test_batched_generation_equals_row_by_row(self):
        import generate_temperature_matched as gtm

        torch.manual_seed(0)
        prompts = torch.randint(0, self.VOCAB, (5, 9))
        model = self._RowwiseStub(self.VOCAB)

        batched_out = gtm.generate(
            model, prompts, vocab_size=self.VOCAB, m=6,
            temperature=0.3, method="transform", device="cpu",
        )
        for row in range(prompts.shape[0]):
            single = gtm.generate(
                model, prompts[row : row + 1], vocab_size=self.VOCAB, m=6,
                temperature=0.3, method="transform", device="cpu",
            )
            for index, name in enumerate(("tokens", "Y", "top_probs", "U", "eta")):
                np.testing.assert_allclose(
                    batched_out[index][row], single[index][0],
                    rtol=0, atol=0,
                    err_msg=f"row {row} differs in {name} when batched",
                )

    def test_pivot_identity_holds(self):
        """Y must be exactly -|U - eta| for every generated token."""
        import generate_temperature_matched as gtm

        torch.manual_seed(1)
        prompts = torch.randint(0, self.VOCAB, (4, 9))
        _, y, _, u, eta = gtm.generate(
            self._RowwiseStub(self.VOCAB), prompts, vocab_size=self.VOCAB, m=5,
            temperature=0.5, method="transform", device="cpu",
        )
        np.testing.assert_allclose(-y, np.abs(u - eta), rtol=0, atol=1e-12)
        self.assertTrue(bool(((-y >= 0) & (-y <= 1)).all()))


class ImportTests(unittest.TestCase):
    def test_module_imports_without_the_upstream_clone(self):
        """Only the key function touches upstream, and only when called."""
        self.assertTrue(hasattr(batched, "generate_inv_batched"))
        self.assertTrue(hasattr(batched, "inverse_permutation_batched"))


if __name__ == "__main__":
    unittest.main()
