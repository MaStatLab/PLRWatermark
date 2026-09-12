"""Regression tests for analytical-versus-implemented model descriptions."""

import unittest
from unittest.mock import Mock

import numpy as np
import benchmark_contamination as contamination

import dirichlet_detector as dd
from experiment_metadata import union_tail_metadata


class UnionMetadataTests(unittest.TestCase):
    def test_complete_union_is_not_certified_closed_form(self):
        grid = dd.UnionTailBayesGrid(
            delta_grid=(.1, .2), delta_weights=(.5, .5), tail_size=15,
            alpha_grid=(1., float("inf")), dirichlet_weight=.7, c_nodes=1001,
        )
        record = union_tail_metadata(grid, methods=["shared", "tokenwise"], tokenwise=True)
        normalisation = record["normalisation"]
        self.assertFalse(normalisation["closed_form"])
        self.assertEqual(normalisation["closed_form_scope"], "width branch only")
        self.assertTrue(normalisation["interpolated"])
        self.assertFalse(normalisation["certified_one_sided_bound"])
        self.assertIn("outer lookups", normalisation["interpolation_scope"])
        self.assertEqual(record["branch_prior"]["shape_weight"], .7)
        self.assertEqual(record["branch_prior"]["component_counts"],
                         [block.n_components for block in grid.blocks])
        self.assertNotIn(15, record["tail_width_prior"]["grid"])
        self.assertIn("shape branch", record["tail_width_prior"]["note"])
        self.assertNotIn("outer_tokenwise_lookup_validation", record)

    def test_aggregate_does_not_invent_model_specific_diagnostics(self):
        record = union_tail_metadata(methods=["shared"])
        self.assertNotIn("prior_fingerprint", record)
        self.assertNotIn("analytic_component_mass_max_abs_deviation", record["normalisation"])
        self.assertIn("by_model", record["branch_prior"])


class ContaminationDisplayTests(unittest.TestCase):
    def test_positive_errors_below_zero_bound_are_not_clipped(self):
        axis = Mock()
        contamination.plot_error_estimates(
            axis, [0, .4, .6], [0, .0002, .001], .0006, color="black"
        )
        line, marker = axis.plot.call_args_list
        np.testing.assert_allclose(line.args[1], [np.nan, .0002, .001])
        np.testing.assert_array_equal(marker.args[0], [0])
        self.assertEqual(marker.kwargs["linestyle"], "None")
        self.assertEqual(marker.kwargs["marker"], "v")

    def test_axis_includes_all_positive_errors(self):
        self.assertLess(contamination.error_axis_floor(
            [{"type2_error": 0}, {"type2_error": .0002}], .0006), .0002)


if __name__ == "__main__":
    unittest.main()
