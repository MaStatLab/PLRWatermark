"""Safe-destination and validated-loader integration for pooled real data."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import pooled_width_real_data as pooled


class PooledRealDriverTests(unittest.TestCase):
    def test_invalid_counts_and_models_fail_before_loading(self):
        for options in (["--replicates", "1"], ["--models", "bad"],
                        ["--models", "1p3B", "1p3B"]):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                pooled.main(options)

    def test_no_window_cells_cannot_write_empty_output(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "empty.json"
            with patch.object(pooled.ptc, "collect", return_value=[]), \
                    self.assertRaisesRegex(ValueError, "no paired cells"):
                pooled.main(["--replicates", "2", "--output", str(output)])
            self.assertFalse(output.exists())

    def test_validated_source_opt_in_and_provenance_reach_explicit_output(self):
        seen = {}
        def collect(models, scratch, scheme, **kwargs):
            seen.update(kwargs)
            kwargs["provenance"]["1p3B_T0.3"] = [{"mode": "fixture"}]
            pooled.ptc.curve_rows("1p3B", "0.3", np.full((4, 100), .6),
                                  np.full((4, 100), .5), {})
            return []
        class Grid:
            def shared_paths(self, values):
                return np.cumsum(values, axis=1)
            def prior_fingerprint(self):
                return {"fixture": True}
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "nested" / "pooled.json"
            with patch.object(pooled.ptc, "collect", side_effect=collect), \
                    patch.object(pooled.wh, "PooledWidthGrid", return_value=Grid()), \
                    contextlib.redirect_stdout(io.StringIO()):
                pooled.main(["--models", "1p3B", "--replicates", "2",
                             "--allow-legacy-archives", "--output", str(output)])
            result = json.loads(output.read_text())
        self.assertTrue(seen["allow_legacy"])
        self.assertEqual(result["input_provenance"]["1p3B_T0.3"], [{"mode": "fixture"}])
        self.assertEqual(result["cells"]["1p3B_T0.3"]["n"], 4)


if __name__ == "__main__":
    unittest.main()
