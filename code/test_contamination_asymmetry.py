"""Tests for the contamination-asymmetry diagnostic (contamination_asymmetry.py)."""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np

import contamination_asymmetry as asym
from bayesian_watermark import (
    gumbel_alt_logpdf_from_probs,
    inverse_alt_logpdf,
    spike_probabilities,
)

RESULTS = Path(__file__).resolve().parents[1] / "results" / "bayesian_paper_benchmark"
PAYLOAD_PATH = RESULTS / "contamination_asymmetry.json"


def payload() -> dict:
    with PAYLOAD_PATH.open(encoding="utf8") as stream:
        return json.load(stream)


class SpikeDensityTest(unittest.TestCase):
    """The vectorised component must agree with the library density."""

    def test_matches_the_library_density(self) -> None:
        pivots = np.linspace(1e-6, 1.0 - 1e-9, 257)
        for delta in (0.005, 0.1, 0.5):
            mine = asym.spike_log_density(np.log(pivots), np.array(delta))
            reference = np.asarray(
                gumbel_alt_logpdf_from_probs(
                    pivots, spike_probabilities(delta, asym.VOCABULARY)
                )
            )
            self.assertLess(float(np.abs(mine - reference).max()), 1e-9)

    def test_component_integrates_to_one(self) -> None:
        points, weights = np.polynomial.legendre.leggauss(4000)
        pivots = 0.5 * (points + 1.0)
        quadrature = 0.5 * weights
        for delta in (0.005, 0.2, 0.5):
            density = np.exp(asym.spike_log_density(np.log(pivots), np.array(delta)))
            self.assertAlmostEqual(float(quadrature @ density), 1.0, places=6)


class SupportTest(unittest.TestCase):
    """The two mechanisms rest on opposite support properties."""

    def test_gumbel_component_is_positive_on_the_unit_interval(self) -> None:
        pivots = np.linspace(1e-9, 1.0 - 1e-9, 1001)
        for delta in (0.001, 0.25, 0.5):
            values = asym.spike_log_density(np.log(pivots), np.array(delta))
            self.assertTrue(np.all(np.isfinite(values)))

    def test_inverse_alternative_vanishes_above_one_minus_delta(self) -> None:
        delta = 0.4
        self.assertEqual(inverse_alt_logpdf(1.0 - delta + 1e-6, delta), -math.inf)
        self.assertTrue(math.isfinite(inverse_alt_logpdf(1.0 - delta - 1e-6, delta)))


class BreakevenTest(unittest.TestCase):
    """A bounded reverse divergence gives the Gumbel rule a usable margin."""

    def test_breakeven_exceeds_primary_rates_through_point_six(self) -> None:
        rows = asym.gumbel_breakeven_rates()
        self.assertEqual(len(rows), len(asym.BREAKEVEN_DELTAS))
        for row in rows:
            self.assertGreater(row["breakeven_rho"], 0.6)
            self.assertLess(row["breakeven_rho"], 1.0)
            self.assertGreater(row["kl_null_vs_component"], 0.0)

    def test_breakeven_decreases_in_the_deficit(self) -> None:
        rates = [row["breakeven_rho"] for row in asym.gumbel_breakeven_rates()]
        self.assertEqual(rates, sorted(rates, reverse=True))


class PayloadTest(unittest.TestCase):
    """Guard the figures the manuscript quotes from the stored artifact."""

    def setUp(self) -> None:
        if not PAYLOAD_PATH.exists():
            self.skipTest("run contamination_asymmetry.py first")
        self.payload = payload()

    def test_design_matches_the_module_constants(self) -> None:
        design = self.payload["design"]
        self.assertEqual(design["vocabulary_size"], asym.VOCABULARY)
        self.assertEqual(design["horizon"], asym.HORIZON)
        self.assertEqual(design["seed"], asym.SEED)

    def test_gumbel_drift_decreases_and_turns_negative_only_at_full_replacement(
        self,
    ) -> None:
        rows = self.payload["gumbel_drift_sweep"]
        means = [row["mean_log_bayes_factor"] for row in rows]
        self.assertEqual(means, sorted(means, reverse=True))
        for row in rows:
            if row["rho_star"] < 1.0:
                self.assertGreater(row["mean_log_bayes_factor"], 0.0)
            else:
                self.assertLess(row["mean_log_bayes_factor"], 0.0)

    def test_inverse_ceiling_tightens_with_contamination(self) -> None:
        rows = self.payload["inverse_support_ceiling"]
        ceilings = [row["median_delta_ceiling"] for row in rows]
        self.assertEqual(ceilings, sorted(ceilings, reverse=True))
        heaviest = rows[-1]
        self.assertEqual(heaviest["rho_star"], 0.6)
        self.assertLess(heaviest["median_surviving_prior_mass"], 0.15)

    def test_gumbel_pair_is_unresolved_and_inverse_pair_is_not(self) -> None:
        rows = self.payload["paired_clean_vs_robust"]
        gumbel = [row for row in rows if row["scheme"] == "gumbel"]
        inverse = [row for row in rows if row["scheme"] == "inverse"]
        self.assertTrue(
            all(
                row["degenerate_reference_distribution"]
                for row in gumbel
                if row["rho_star"] <= 0.4
            )
        )
        heaviest = [row for row in inverse if row["rho_star"] == 0.6][0]
        self.assertLess(heaviest["exact_mcnemar_p"], 1e-100)
        self.assertGreater(heaviest["robust_only"], 10 * heaviest["clean_only"])

    def test_gumbel_loses_only_the_replaced_tokens(self) -> None:
        rows = self.payload["effective_horizon_match"]
        gumbel = [row for row in rows if row["scheme"] == "gumbel"]
        inverse = [row for row in rows if row["scheme"] == "inverse"]
        self.assertTrue(
            all(abs(row["excess_over_surviving_horizon"]) < 0.002 for row in gumbel)
        )
        heaviest = [row for row in inverse if row["rho_star"] == 0.6][0]
        self.assertGreater(heaviest["excess_over_surviving_horizon"], 0.2)


if __name__ == "__main__":
    unittest.main()
