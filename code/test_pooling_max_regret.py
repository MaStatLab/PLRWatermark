"""Tests for the pooling sweeps' maximum-regret summary."""
import unittest

import numpy as np

import pooling_max_regret as pmr


def _indicators(table, horizons=("10",)):
    """``{regime: {rule: [miss, ...]}}`` to the npz-style flat mapping."""

    out = {}
    for regime, rules in table.items():
        for rule, misses in rules.items():
            for horizon in horizons:
                out[f"{regime}|{rule}|{horizon}"] = np.asarray(misses, dtype=bool)
    return out


def _payload(indicators, regimes, rules, horizons=("10",), seed=7):
    return {
        "seed": seed,
        "type2_error": {
            h: {r: {g: float(indicators[f"{g}|{r}|{h}"].mean()) for g in regimes}
                for r in rules}
            for h in horizons
        },
    }


class SummarizeTests(unittest.TestCase):
    def setUp(self):
        # Four documents per regime.  "good" misses one document in A and none
        # in B; "bad" misses three in A and two in B.  The envelope is "good",
        # so its regret is zero everywhere and "bad" has regret .5 at each
        # regime, hence max regret .5.
        self.regimes = ("A", "B")
        self.rules = ("good", "bad")
        self.ind = _indicators({
            "A": {"good": [1, 0, 0, 0], "bad": [1, 1, 1, 0]},
            "B": {"good": [0, 0, 0, 0], "bad": [1, 1, 0, 0]},
        })
        self.payload = _payload(self.ind, self.regimes, self.rules)

    def test_max_regret_is_the_worst_excess_over_the_envelope(self):
        out = pmr.summarize(self.payload, self.ind, self.regimes, replicates=8)
        rows = out["by_horizon"]["10"]
        self.assertAlmostEqual(rows["bad"]["max_regret"], 0.5)
        self.assertAlmostEqual(rows["good"]["max_regret"], 0.0)

    def test_the_envelope_rule_reports_no_argmax_regime(self):
        out = pmr.summarize(self.payload, self.ind, self.regimes, replicates=8)
        rows = out["by_horizon"]["10"]
        self.assertIsNone(rows["good"]["argmax_regime"])
        # Ties break to the first regime in the table's column order.
        self.assertEqual(rows["bad"]["argmax_regime"], "A")

    def test_stored_type2_must_equal_the_indicator_mean(self):
        # The whole point of reading the indicators is that they are the same
        # sample the table reports; a mismatch has to be loud.
        broken = _payload(self.ind, self.regimes, self.rules)
        broken["type2_error"]["10"]["bad"]["A"] = 0.9
        with self.assertRaises(AssertionError):
            pmr.summarize(broken, self.ind, self.regimes, replicates=8)

    def test_bootstrap_se_is_positive_and_reproducible(self):
        first = pmr.summarize(self.payload, self.ind, self.regimes, replicates=64)
        again = pmr.summarize(self.payload, self.ind, self.regimes, replicates=64)
        a = first["by_horizon"]["10"]["bad"]
        b = again["by_horizon"]["10"]["bad"]
        self.assertGreater(a["mc_se"], 0.0)
        self.assertEqual(a["mc_se"], b["mc_se"])

    def test_indicator_cube_is_documents_by_rules_by_horizons(self):
        cube = pmr.indicator_cube(self.ind, "A", list(self.rules), ["10"])
        self.assertEqual(cube.shape, (4, 2, 1))
        np.testing.assert_allclose(cube[:, 0, 0], [1, 0, 0, 0])
        np.testing.assert_allclose(cube[:, 1, 0], [1, 1, 1, 0])


class ShippedArtifactTests(unittest.TestCase):
    """The committed payloads carry the block the tables are printed from."""

    def test_both_sweeps_record_max_regret_for_every_cell(self):
        import json

        for _, (payload_file, _, regimes) in pmr.SWEEPS.items():
            payload = json.loads(
                (pmr.RESULTS / payload_file).read_text(encoding="utf-8")
            )
            block = payload["max_regret"]
            self.assertEqual(block["bootstrap_replicates"], 2000)
            for horizon, rows in block["by_horizon"].items():
                self.assertEqual(
                    set(rows), set(payload["type2_error"][horizon]),
                    f"{payload_file} n={horizon} rule set drifted",
                )
                for rule, cell in rows.items():
                    self.assertGreaterEqual(cell["max_regret"], 0.0)
                    self.assertGreater(cell["mc_se"], 0.0)
                    if cell["max_regret"] == 0.0:
                        self.assertIsNone(cell["argmax_regime"])
                    else:
                        self.assertIn(cell["argmax_regime"], regimes)

    def test_stored_max_regret_agrees_with_the_stored_type2_table(self):
        # The recorded maxima have to be the ones the Type II columns imply.
        # Note this is a maximum over regimes of a per-regime excess, so it is
        # NOT generally zero for any rule: a rule is on the envelope at every
        # regime only if it wins all of them, which none of these does.
        import json

        from regime_sweep import max_regret, regret_table

        for _, (payload_file, _, regimes) in pmr.SWEEPS.items():
            payload = json.loads(
                (pmr.RESULTS / payload_file).read_text(encoding="utf-8")
            )
            type2 = payload["type2_error"]
            for horizon, rows in payload["max_regret"]["by_horizon"].items():
                regret, best = regret_table(
                    {rule: {g: float(type2[horizon][rule][g]) for g in regimes}
                     for rule in type2[horizon]}
                )
                recomputed = max_regret(regret)
                for rule, cell in rows.items():
                    self.assertAlmostEqual(
                        cell["max_regret"], recomputed[rule]["max_regret"],
                        places=12, msg=f"{payload_file} n={horizon} {rule}",
                    )
                    self.assertEqual(
                        cell["argmax_regime"], recomputed[rule]["argmax_regime"]
                    )
                # Every regime's envelope is attained by some scored rule.
                for g in regimes:
                    self.assertIn(
                        best[g],
                        [float(type2[horizon][r][g]) for r in type2[horizon]],
                    )


if __name__ == "__main__":
    unittest.main()
