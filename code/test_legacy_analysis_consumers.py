"""Legacy diagnostics must not bypass archive and pairing validation."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import analyse_temperature_matched as legacy
import summarise_empirical_deficits as deficits


class LegacyConsumersTests(unittest.TestCase):
    def test_base_only_requires_both_bands(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "both base"):
                legacy.load_base_cells(directory, "1p3B")

    def test_base_loader_uses_validated_pairing_and_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            for band in ("lo", "hi"):
                (Path(directory) / f"matched_1p3B_{band}.npz").touch()
            with patch.object(legacy.matched, "load_prompt_table", return_value="prompts"), \
                 patch.object(legacy.matched, "paired_cells", return_value=[]) as paired:
                legacy.load_base_cells(directory, "1p3B", allow_legacy=True)
                self.assertTrue(paired.call_args.kwargs["allow_legacy"])
                self.assertEqual(len(paired.call_args.args[0]), 2)

    def test_invalid_deficits_fail(self):
        for values in ([], [np.nan], [-.1], [1.1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                deficits.summarise(values)

    def test_valid_summary_unchanged(self):
        record = deficits.summarise(np.array([0., .1, .5, .6]))
        self.assertEqual(record["n"], 4)
        self.assertEqual(record["in_support"], .5)


if __name__ == "__main__":
    unittest.main()
