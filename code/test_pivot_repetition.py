"""Tests for the released-PRF repetition audit."""

from __future__ import annotations

import unittest

import numpy as np

from pivot_repetition import group_is_constant


class GroupIsConstantTest(unittest.TestCase):
    def test_counts_groups_and_inconstant_groups(self) -> None:
        keys = np.array([2, 1, 2, 1, 3])
        values = np.array([0.5, 0.1, 0.5, 0.2, 0.7])
        groups, inconstant = group_is_constant(keys, values)
        self.assertEqual(groups, 3)
        # key 1 carries .1 and .2; key 2 carries .5 twice; key 3 is a singleton.
        self.assertEqual(inconstant, 1)

    def test_all_constant(self) -> None:
        keys = np.array([7, 7, 7])
        values = np.array([0.25, 0.25, 0.25])
        self.assertEqual(group_is_constant(keys, values), (1, 0))

    def test_singletons_are_never_inconstant(self) -> None:
        keys = np.arange(5)
        values = np.linspace(0.1, 0.9, 5)
        self.assertEqual(group_is_constant(keys, values), (5, 0))

    def test_unsorted_input_groups_correctly(self) -> None:
        # A sort-order bug would split a key into two adjacent runs and read
        # each as constant, so the audit would report zero violations wrongly.
        keys = np.array([1, 2, 1, 2, 1])
        values = np.array([0.1, 0.3, 0.1, 0.4, 0.9])
        groups, inconstant = group_is_constant(keys, values)
        self.assertEqual(groups, 2)
        self.assertEqual(inconstant, 2)


if __name__ == "__main__":
    unittest.main()


class PrfAddressTest(unittest.TestCase):
    """The address is the token four back, which is what the seed hashes."""

    def test_address_is_the_token_four_positions_back(self) -> None:
        import real_data_experiment as real

        prompts = np.arange(100, 110).reshape(1, 10)
        tokens = np.arange(200, 208).reshape(1, 8)
        addresses = real.prf_addresses(prompts, tokens, horizon=8)
        # positions 0..3 are seeded by the last four prompt tokens, and from
        # position 4 onward by the continuation token four back.
        expected = np.array([[106, 107, 108, 109, 200, 201, 202, 203]])
        self.assertTrue(np.array_equal(addresses, expected))

    def test_short_prompt_is_rejected(self) -> None:
        import real_data_experiment as real

        with self.assertRaises(ValueError):
            real.prf_addresses(np.zeros((1, 2), dtype=int),
                               np.zeros((1, 8), dtype=int), horizon=8)
