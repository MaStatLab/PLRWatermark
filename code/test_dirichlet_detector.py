"""Tests for the shared Dirichlet-layer detector (dirichlet_detector.py)."""

from __future__ import annotations

import math
import unittest

import numpy as np

import benchmark_paper_experiment as paper
import dirichlet_detector as dd
import width_hierarchy as wh
import regime_sweep as deficit_sweep
import tail_family as tf


VOCABULARY = 1000
TAIL_SIZE = VOCABULARY - 1


def small_grid(**overrides: object) -> dd.DirichletBayesGrid:
    """A grid cheap enough to build inside a test."""

    deltas, weights = dd.gauss_legendre_delta_grid(0.001, 0.5, 8)
    defaults: dict[str, object] = {
        "delta_grid": deltas,
        "delta_weights": weights,
        "alpha_grid": (0.5, 2.0, math.inf),
        "tail_size": 39,
        "c_nodes": 2_001,
    }
    defaults.update(overrides)
    return dd.DirichletBayesGrid(**defaults)  # type: ignore[arg-type]


def full_grid(**overrides: object) -> dd.DirichletBayesGrid:
    """The production grid: 96 deficit nodes, the frozen alpha prior, M=1000.

    The deficit bounds are read from the sweep config rather than restated, so
    these agreement tests compare two implementations of the same prior instead
    of pinning one particular prior.
    """
    import regime_sweep as _sweep

    _cfg = _sweep.RegimeSweepConfig(bayes_quadrature_nodes=96)
    deltas, weights = dd.gauss_legendre_delta_grid(
        _cfg.prior_low, _cfg.prior_high, 96
    )
    defaults: dict[str, object] = {
        "delta_grid": deltas,
        "delta_weights": weights,
        "tail_size": TAIL_SIZE,
        "allow_large_delta": True,
    }
    defaults.update(overrides)
    return dd.DirichletBayesGrid(**defaults)  # type: ignore[arg-type]


def logit_quadrature(panels: int = 800, order: int = 32, limit: float = 40.0):
    """Nodes and dr-weights for integrating a density over r in (0,1)."""

    nodes, weights = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(-limit, limit, panels + 1)
    half = 0.5 * (edges[1] - edges[0])
    centres = 0.5 * (edges[:-1] + edges[1:])
    z = (centres[:, None] + half * nodes[None, :]).reshape(-1)
    quadrature = np.tile(weights * half, centres.size)
    r = 1.0 / (1.0 + np.exp(-z))
    return r, quadrature * r * (1.0 - r)


class GridConstructionTests(unittest.TestCase):
    def test_component_ordering_is_alpha_delta_rho(self) -> None:
        grid = small_grid(rho_grid=(0.0, 0.5))
        self.assertEqual(
            grid.n_components,
            len(grid.alphas) * grid.deltas.size * grid.rhos.size,
        )
        # alpha slowest, rho fastest
        block = grid.deltas.size * grid.rhos.size
        for index, _ in enumerate(grid.alphas):
            self.assertTrue(
                np.all(grid.component_alpha[index * block:(index + 1) * block] == index)
            )
        np.testing.assert_array_equal(
            grid.component_rho[: grid.rhos.size], grid.rhos
        )

    def test_component_weights_sum_to_one(self) -> None:
        grid = small_grid(rho_grid=(0.0, 0.25, 0.5), rho_weights=(0.5, 0.3, 0.2))
        total = float(np.exp(grid.log_component_weights).sum())
        self.assertAlmostEqual(total, 1.0, places=12)

    def test_alpha_slice_selects_that_alphas_components(self) -> None:
        grid = small_grid(rho_grid=(0.0, 0.4))
        for alpha in grid.alphas:
            chunk = grid.component_slice_for_alpha(alpha)
            index = grid.alpha_index[dd.format_alpha(alpha)]
            self.assertTrue(np.all(grid.component_alpha[chunk] == index))
            self.assertEqual(
                grid.component_alpha[chunk].size, grid.deltas.size * grid.rhos.size
            )

    def test_invalid_input_is_rejected(self) -> None:
        deltas, weights = dd.gauss_legendre_delta_grid(0.001, 0.5, 8)
        bad = [
            {"delta_grid": np.array([0.0, 0.2])},
            {"delta_grid": np.array([0.2, 1.0])},
            {"delta_grid": np.array([0.2, 0.5001])},
            {"delta_grid": np.array([])},
            {"alpha_grid": ()},
            {"alpha_grid": (1.0, 1.0)},
            {"alpha_grid": (0.0,)},
            {"alpha_grid": (math.nan,)},
            {"rho_grid": ()},
            {"rho_grid": (-0.1,)},
            {"rho_grid": (1.5,)},
            {"tail_size": 0},
            {"delta_weights": np.zeros(8)},
        ]
        for overrides in bad:
            with self.subTest(**{k: str(v) for k, v in overrides.items()}):
                kwargs: dict[str, object] = {
                    "delta_grid": deltas,
                    "delta_weights": weights,
                    "alpha_grid": (1.0,),
                    "tail_size": 39,
                    "c_nodes": 2_001,
                }
                kwargs.update(overrides)
                if "delta_grid" in overrides and "delta_weights" not in overrides:
                    kwargs["delta_weights"] = np.ones(
                        max(np.asarray(overrides["delta_grid"]).size, 1)
                    )
                with self.assertRaises(ValueError):
                    dd.DirichletBayesGrid(**kwargs)  # type: ignore[arg-type]

    def test_gauss_legendre_helper_matches_the_existing_grid(self) -> None:
        # The Dirichlet rule must integrate the same deficit prior, the same way,
        # as every spike-family rule already reported.
        # Derive the bounds from the config rather than restating them, so the
        # invariant is "the same prior, integrated the same way" and not "this
        # particular prior".
        config = deficit_sweep.RegimeSweepConfig(bayes_quadrature_nodes=96)
        deltas, weights = dd.gauss_legendre_delta_grid(
            config.prior_low, config.prior_high, 96
        )
        reference_deltas, reference_weights = deficit_sweep.frozen_delta_quadrature(
            config
        )
        np.testing.assert_array_equal(deltas, reference_deltas)
        np.testing.assert_array_equal(weights, reference_weights)

    def test_gauss_legendre_helper_rejects_bad_ranges(self) -> None:
        for args in ((0.5, 0.5, 8), (0.5, 0.2, 8), (-0.1, 0.5, 8), (0.1, 1.0, 8)):
            with self.assertRaises(ValueError):
                dd.gauss_legendre_delta_grid(*args)
        with self.assertRaises(ValueError):
            dd.gauss_legendre_delta_grid(0.1, 0.5, 1)


class ContainmentTests(unittest.TestCase):
    """The layer must contain the detector already reported, exactly."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.grid = full_grid()
        cls.spike_only = full_grid(alpha_grid=(math.inf,))

    def test_alpha_infinity_reproduces_the_closed_form_spike_density(self) -> None:
        rng = np.random.default_rng(3)
        gap = self.grid.spike_agreement(rng.uniform(size=20_000))
        self.assertLess(gap, 1e-7)

    def test_alpha_infinity_reproduces_the_published_shared_path(self) -> None:
        # regime_sweep._shared_bayes_batch is the routine behind every shared-Delta
        # Gumbel number in the manuscript.  The layer at alpha=inf must match it.
        rng = np.random.default_rng(4)
        pivots = rng.uniform(size=(60, 40))
        config = deficit_sweep.RegimeSweepConfig(
            vocabulary_size=VOCABULARY,
            horizons=tuple(range(1, 41)),
            bayes_quadrature_nodes=96,
        )
        deltas, weights = deficit_sweep.frozen_delta_quadrature(config)
        expected = deficit_sweep._shared_bayes_batch(
            pivots, deltas, np.log(weights), config
        )
        got, _ = self.spike_only.shared_paths(pivots)
        np.testing.assert_allclose(got, expected, atol=1e-6)

    def test_alpha_infinity_tokenwise_matches_the_existing_lookup(self) -> None:
        # The spike-family tokenwise lookup and the layer's tokenwise lookup at
        # alpha=inf are two tabulations of the same integral.
        config = paper.BenchmarkConfig(vocabulary_size=VOCABULARY)
        reference = paper.GumbelBayesLookup(config)
        lookup = self.spike_only.tokenwise_lookup()
        rng = np.random.default_rng(5)
        r = rng.uniform(size=20_000) ** rng.uniform(0.05, 1.0, size=20_000)
        np.testing.assert_allclose(lookup(r), reference(r), atol=2e-5)


class NormalisationTests(unittest.TestCase):
    """Every component must be a probability density, or e-validity fails."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.grid = full_grid()

    def test_analytic_component_mass_is_exactly_one(self) -> None:
        for alpha, deviation in self.grid.analytic_normalisation().items():
            self.assertLess(deviation, 1e-12, msg=f"alpha={alpha}")

    def test_interpolated_component_mass_numerical_diagnostic(self) -> None:
        report = self.grid.normalisation_report(panels=600, gauss_nodes=32)
        self.assertTrue(report["diagnostic_only"])
        self.assertFalse(report["certified_one_sided_bound"])
        self.assertTrue(report["all_masses_at_most_one"])
        self.assertLess(report["max_abs_mass_minus_one"], 1e-7)

    def test_component_ratios_numerically_integrate_to_one(self) -> None:
        # The analytic components are null e-values.  This checks that the fast
        # interpolated implementation remains numerically close to unit mass.
        r, weight = logit_quadrature()
        ratios = np.exp(self.grid.component_log_ratio(r))
        mass = weight @ ratios
        np.testing.assert_allclose(mass, 1.0, atol=1e-6)

    def test_diluted_component_ratios_are_also_null_e_values(self) -> None:
        grid = full_grid(rho_grid=(0.0, 0.25, 0.6), rho_weights=(0.5, 0.25, 0.25))
        r, weight = logit_quadrature()
        mass = weight @ np.exp(grid.component_log_ratio(r))
        np.testing.assert_allclose(mass, 1.0, atol=1e-6)

    def test_tokenwise_numerator_numerical_mass_is_near_one(self) -> None:
        lookup = self.grid.tokenwise_lookup()
        mass = lookup.normalisation()
        # Observed for this frozen grid; not a certified integral inequality.
        self.assertLessEqual(mass, 1.0)
        self.assertGreater(mass, 1.0 - 1e-6)

    def test_dilution_with_rho_one_is_exactly_the_null(self) -> None:
        grid = small_grid(rho_grid=(1.0,))
        rng = np.random.default_rng(6)
        ratios = grid.component_log_ratio(rng.uniform(size=200))
        np.testing.assert_allclose(ratios, 0.0, atol=0.0)


class SequentialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.grid = full_grid()

    def test_sparse_horizons_match_the_dense_path_exactly(self) -> None:
        rng = np.random.default_rng(7)
        pivots = rng.uniform(size=(24, 60))
        dense, _ = self.grid.shared_paths(pivots)
        sparse, _ = self.grid.shared_paths(pivots, horizons=(10, 30, 60))
        np.testing.assert_array_equal(sparse, dense[:, [9, 29, 59]])

    def test_running_maximum_is_the_true_supremum(self) -> None:
        rng = np.random.default_rng(8)
        pivots = rng.uniform(size=(24, 60)) ** 0.4
        dense, maxima = self.grid.shared_paths(pivots, running_max=True)
        expected = np.maximum.accumulate(dense, axis=1)
        np.testing.assert_allclose(maxima, expected, atol=0.0)

    def test_running_maximum_at_sparse_horizons_covers_skipped_tokens(self) -> None:
        # The supremum must be over every token, not only the reported ones.
        rng = np.random.default_rng(9)
        pivots = rng.uniform(size=(16, 50)) ** 0.3
        dense, _ = self.grid.shared_paths(pivots)
        _, sparse_max = self.grid.shared_paths(
            pivots, horizons=(10, 50), running_max=True
        )
        np.testing.assert_allclose(
            sparse_max[:, 0], np.max(dense[:, :10], axis=1), atol=0.0
        )
        np.testing.assert_allclose(
            sparse_max[:, 1], np.max(dense[:, :50], axis=1), atol=0.0
        )

    def test_shared_path_is_the_log_mixture_of_component_products(self) -> None:
        rng = np.random.default_rng(10)
        pivots = rng.uniform(size=(11, 7))
        got, _ = self.grid.shared_paths(pivots, horizons=(7,))
        accumulator = np.zeros((11, self.grid.n_components))
        for time_index in range(7):
            accumulator += self.grid.component_log_ratio(pivots[:, time_index])
        expected = dd._logsumexp_rows(
            accumulator + self.grid.log_component_weights[None, :]
        )
        np.testing.assert_allclose(got[:, 0], expected, atol=1e-12)

    def test_tokenwise_lookup_matches_the_direct_mixture(self) -> None:
        lookup = self.grid.tokenwise_lookup()
        rng = np.random.default_rng(11)
        r = rng.uniform(size=5_000)
        direct = dd._logsumexp_rows(
            self.grid.component_log_ratio(r) + self.grid.log_component_weights[None, :]
        )
        np.testing.assert_allclose(lookup(r), direct, atol=1e-6)

    def test_shared_mixture_contains_each_weighted_grid_component(
        self,
    ) -> None:
        rng = np.random.default_rng(12)
        pivots = rng.uniform(size=(20, 40)) ** 0.35
        mixture, _ = self.grid.shared_paths(pivots, horizons=(40,))
        accumulator = np.zeros((20, self.grid.n_components))
        for time_index in range(40):
            accumulator += self.grid.component_log_ratio(pivots[:, time_index])
        bound = -float(np.min(self.grid.log_component_weights))
        best = np.max(accumulator, axis=1)
        self.assertTrue(np.all(mixture[:, 0] >= best - bound - 1e-9))
        self.assertTrue(np.all(mixture[:, 0] <= best + 1e-9 + math.log(
            self.grid.n_components
        )))

    def test_shared_paths_rejects_bad_horizons(self) -> None:
        pivots = np.full((4, 10), 0.5)
        for horizons in ((0, 5), (5, 5), (5, 3), (5, 11), ()):
            with self.subTest(horizons=horizons):
                with self.assertRaises(ValueError):
                    self.grid.shared_paths(pivots, horizons=horizons)
        with self.assertRaises(ValueError):
            self.grid.shared_paths(np.full(10, 0.5))

    def test_exact_zero_and_one_pivots_are_finite(self) -> None:
        values = self.grid.component_log_ratio(np.array([0.0, 1.0, 0.5]))
        self.assertTrue(np.all(np.isfinite(values)))

    def test_pivots_outside_the_unit_interval_are_rejected(self) -> None:
        for bad in (np.array([-0.1]), np.array([1.5]), np.array([np.nan])):
            with self.assertRaises(ValueError):
                self.grid.component_log_density(bad)


class SimulationConsistencyTests(unittest.TestCase):
    """The detector and the simulators must describe the same law."""

    def test_component_density_matches_the_size_biased_simulator(self) -> None:
        delta, tail_size = 0.25, 200
        deltas = np.array([delta])
        grid = dd.DirichletBayesGrid(
            delta_grid=deltas,
            delta_weights=np.array([1.0]),
            alpha_grid=(0.7,),
            tail_size=tail_size,
            c_nodes=20_001,
        )
        pivot = tf.dirichlet_tail_pivot(0.7, tail_size, c_nodes=20_001)
        rng = np.random.default_rng(13)
        draws = pivot.simulate(rng, np.full(300_000, delta))
        # Compare the empirical mean of a bounded test function against the
        # density-weighted integral; this exercises density and simulator jointly.
        r, weight = logit_quadrature()
        density = np.exp(grid.component_log_density(r)).reshape(-1)
        for power in (1.0, 2.0, 4.0):
            predicted = float(np.dot(weight, density * r**power))
            observed = float(np.mean(draws**power))
            self.assertAlmostEqual(predicted, observed, delta=4e-3, msg=f"r^{power}")

    def test_prior_fingerprint_records_the_frozen_prior(self) -> None:
        grid = full_grid(rho_grid=(0.0, 0.4), rho_weights=(0.7, 0.3))
        report = grid.prior_fingerprint()
        self.assertEqual(report["delta_nodes"], 96)
        self.assertEqual(
            report["alpha_grid"], ["0.1", "1", "10", "100", "1000", "inf"]
        )
        self.assertAlmostEqual(report["rho_mean"], 0.12, places=12)
        self.assertEqual(report["n_components"], 6 * 96 * 2)

    def test_uniform_alpha_prior_helper(self) -> None:
        grid, weights = dd.uniform_alpha_prior()
        self.assertEqual(len(grid), len(dd.DEFAULT_ALPHA_GRID))
        self.assertEqual(tuple(grid), dd.DEFAULT_ALPHA_GRID)
        self.assertTrue(any(math.isinf(a) for a in grid))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)
        for bad in ((), (1.0, 1.0), (0.0,), (math.nan,)):
            with self.assertRaises(ValueError):
                dd.uniform_alpha_prior(bad)

    def test_format_alpha_is_stable(self) -> None:
        self.assertEqual(dd.format_alpha(math.inf), "inf")
        self.assertEqual(dd.format_alpha(1.0), "1")
        self.assertEqual(dd.format_alpha(0.1), "0.1")


class ImplementationAgreementTests(unittest.TestCase):
    """The two independent layer implementations must not drift apart.

    ``tail_regime_sweep.DirichletLayerScores`` predates this module and flattens
    its component axes in its own order.  Nothing in either implementation would
    raise if the two orderings diverged -- the posterior would simply be
    transposed -- so the agreement is asserted rather than assumed.
    """

    def test_component_log_densities_agree_exactly(self) -> None:
        import tail_regime_sweep as trs

        config = trs.TailSweepConfig(
            vocabulary_size=VOCABULARY,
            bayes_quadrature_nodes=96,
            alpha_grid=dd.DEFAULT_ALPHA_GRID,
            point_alphas=trs.DEFAULT_POINT_ALPHAS,
            horizons=(5, 20),
        )
        layer = trs.DirichletLayerScores(config)
        grid = full_grid()
        rng = np.random.default_rng(99)
        pivots = rng.uniform(size=400)
        left = layer.component_log_density(pivots)
        right = grid.component_log_density(pivots).reshape(
            pivots.size, len(grid.alphas), grid.deltas.size
        ).transpose(1, 0, 2)
        np.testing.assert_array_equal(left, right)

    def test_shared_mixture_paths_agree_exactly(self) -> None:
        import tail_regime_sweep as trs

        config = trs.TailSweepConfig(
            vocabulary_size=VOCABULARY,
            bayes_quadrature_nodes=96,
            alpha_grid=dd.DEFAULT_ALPHA_GRID,
            point_alphas=trs.DEFAULT_POINT_ALPHAS,
            horizons=(5, 20),
        )
        layer = trs.DirichletLayerScores(config)
        grid = full_grid()
        rng = np.random.default_rng(100)
        pivots = rng.uniform(size=(25, 20))
        left = trs.all_rule_paths(pivots, config, layer)[trs.MIXTURE_RULE][:, -1]
        right, _ = grid.shared_paths(pivots, horizons=(20,))
        np.testing.assert_array_equal(left, right[:, 0])

    def test_tail_family_exposes_no_conflicting_flattening_helper(self) -> None:
        # A previous helper flattened (Delta, alpha) with Delta slowest, the
        # opposite of this module's order.  It was unused and has been removed;
        # this guards against it coming back and being mixed with grid order.
        self.assertNotIn("joint_delta_alpha_grid", tf.__all__)
        self.assertFalse(hasattr(tf, "joint_delta_alpha_grid"))



def small_tailwidth_grid(**overrides: object) -> dd.TailWidthBayesGrid:
    """A tail-width grid cheap enough to build inside a test."""

    deltas, weights = dd.gauss_legendre_delta_grid(0.001, 0.5, 8)
    defaults: dict[str, object] = {
        "delta_grid": deltas,
        "delta_weights": weights,
        "tail_size": 39,
    }
    defaults.update(overrides)
    return dd.TailWidthBayesGrid(**defaults)  # type: ignore[arg-type]


class TailWidthGridTests(unittest.TestCase):
    """The enlarged Gumbel layer with a latent number of live tail coordinates."""

    def test_dyadic_ladder_starts_at_one_and_ends_at_the_full_tail(self) -> None:
        for tail_size in (39, 999, 31_999, 50_271):
            grid = dd.dyadic_tail_width_grid(tail_size)
            self.assertEqual(grid[0], 1)
            self.assertEqual(grid[-1], tail_size)
            self.assertEqual(len(set(grid)), len(grid))
            self.assertEqual(list(grid), sorted(grid))
            for value in grid[:-1]:
                self.assertLess(value, tail_size)

    def test_dyadic_ladder_is_powers_of_four_below_the_full_tail(self) -> None:
        self.assertEqual(dd.dyadic_tail_width_grid(999), (1, 4, 16, 64, 256, 999))
        self.assertEqual(
            dd.dyadic_tail_width_grid(50_271),
            (1, 4, 16, 64, 256, 1024, 4096, 16_384, 50_271),
        )

    def test_ladder_rejects_bad_arguments(self) -> None:
        with self.assertRaises(ValueError):
            dd.dyadic_tail_width_grid(0)
        with self.assertRaises(ValueError):
            dd.dyadic_tail_width_grid(10, base=1)

    def test_uniform_prior_rejects_duplicate_or_nonpositive_widths(self) -> None:
        with self.assertRaises(ValueError):
            dd.uniform_tail_width_prior(())
        with self.assertRaises(ValueError):
            dd.uniform_tail_width_prior((1, 1, 4))
        with self.assertRaises(ValueError):
            dd.uniform_tail_width_prior((0, 4))

    def test_grid_rejects_widths_wider_than_the_vocabulary_tail(self) -> None:
        deltas, weights = dd.gauss_legendre_delta_grid(0.001, 0.5, 4)
        with self.assertRaises(ValueError):
            dd.TailWidthBayesGrid(
                delta_grid=deltas,
                delta_weights=weights,
                tail_size=39,
                tail_width_grid=(1, 40),
            )

    def test_grid_rejects_deficits_above_one_half(self) -> None:
        with self.assertRaises(ValueError):
            dd.TailWidthBayesGrid(
                delta_grid=np.array([0.2, 0.7]),
                delta_weights=np.array([0.5, 0.5]),
                tail_size=39,
            )

    def test_component_count_and_ordering(self) -> None:
        grid = small_tailwidth_grid(rho_grid=(0.0, 0.25))
        self.assertEqual(
            grid.n_components,
            len(grid.tail_widths) * grid.deltas.size * grid.rhos.size,
        )
        # (J, Delta, rho) with J slowest and rho fastest.
        self.assertEqual(grid.component_rho[0], 0.0)
        self.assertEqual(grid.component_rho[1], 0.25)
        self.assertEqual(grid.component_tail_width[0], grid.tail_widths[0])
        self.assertEqual(grid.component_tail_width[-1], grid.tail_widths[-1])
        block = grid.deltas.size * grid.rhos.size
        for index, width in enumerate(grid.tail_widths):
            chunk = grid.component_tail_width[index * block : (index + 1) * block]
            self.assertTrue(np.all(chunk == width))
            self.assertEqual(grid.component_slice_for_tail_width(width),
                             slice(index * block, (index + 1) * block))

    def test_widest_atom_is_exactly_the_closed_form_spike(self) -> None:
        # This is the containment property the enlarged family rests on: the
        # last atom must BE the equal-tail component every other rule evaluates.
        grid = small_tailwidth_grid()
        rng = np.random.default_rng(4)
        self.assertEqual(grid.spike_agreement(rng.uniform(size=5_000)), 0.0)

    def test_atoms_match_an_independent_exact_spike_implementation(self) -> None:
        import bayesian_watermark as bw

        grid = small_tailwidth_grid(tail_width_grid=(1, 4, 39))
        r = np.array([1e-9, 0.25, 0.5, 0.9, 0.999999, 1.0])
        density = grid.component_log_density(r).reshape(
            r.size, len(grid.tail_widths), grid.deltas.size
        )
        for index, width in enumerate(grid.tail_widths):
            reference = np.stack(
                [
                    bw.gumbel_alt_logpdf_from_probs(
                        r, bw.spike_probabilities(float(delta), int(width) + 1)
                    )
                    for delta in grid.deltas
                ],
                axis=1,
            )
            np.testing.assert_allclose(
                density[:, index, :], reference, rtol=0.0, atol=1e-12
            )

    def test_components_are_analytically_normalised(self) -> None:
        grid = small_tailwidth_grid()
        for width, deviation in grid.analytic_normalisation().items():
            self.assertLess(abs(deviation), 1e-15, msg=f"tail width {width}")

    def test_component_density_integrates_to_one_by_quadrature(self) -> None:
        grid = small_tailwidth_grid(tail_width_grid=(1, 4, 39))
        nodes, weights = np.polynomial.legendre.leggauss(200)
        edges = np.linspace(-40.0, 40.0, 4_001)
        half = 0.5 * (edges[1] - edges[0])
        centres = 0.5 * (edges[:-1] + edges[1:])
        z = (centres[:, None] + half * nodes[None, :]).reshape(-1)
        quadrature = np.tile(weights * half, centres.size)
        sigmoid = 1.0 / (1.0 + np.exp(-z))
        jacobian = sigmoid * (1.0 - sigmoid)
        density = np.exp(
            grid.component_log_density(sigmoid).reshape(
                z.size, len(grid.tail_widths), grid.deltas.size
            )
        )
        mass = np.einsum("z,zjd->jd", quadrature * jacobian, density)
        np.testing.assert_allclose(mass, 1.0, rtol=0.0, atol=5e-7)

    def test_dilution_ratio_is_rho_plus_one_minus_rho_times_the_clean_ratio(self) -> None:
        rho = 0.3
        clean = small_tailwidth_grid()
        diluted = small_tailwidth_grid(rho_grid=(rho,))
        r = np.array([0.05, 0.4, 0.95])
        expected = np.logaddexp(
            math.log(rho), math.log1p(-rho) + clean.component_log_ratio(r)
        )
        np.testing.assert_allclose(diluted.component_log_ratio(r), expected, atol=1e-15)

    def test_one_step_ratio_has_null_expectation_one(self) -> None:
        # The martingale increment property.  Checked by deterministic
        # quadrature rather than Monte Carlo: at J = 1 and small Delta the
        # component is a sharp spike near r = 1, so the likelihood ratio is
        # heavy tailed and its sample mean is dominated by a single draw.
        grid = small_tailwidth_grid()
        nodes, weights = np.polynomial.legendre.leggauss(400)
        edges = np.linspace(-45.0, 45.0, 6_001)
        half = 0.5 * (edges[1] - edges[0])
        centres = 0.5 * (edges[:-1] + edges[1:])
        z = (centres[:, None] + half * nodes[None, :]).reshape(-1)
        quadrature = np.tile(weights * half, centres.size)
        sigmoid = 1.0 / (1.0 + np.exp(-z))
        jacobian = sigmoid * (1.0 - sigmoid)
        mixture = np.exp(
            dd._logsumexp_rows(
                grid.component_log_ratio(sigmoid) + grid.log_component_weights
            )
        )
        self.assertAlmostEqual(float(np.dot(quadrature * jacobian, mixture)), 1.0, places=9)

    def test_dilution_preserves_the_null_expectation(self) -> None:
        # rho + (1-rho) L also has null expectation one, so the diluted rule is
        # still a test martingale.
        grid = small_tailwidth_grid(rho_grid=(0.0, 0.4), rho_weights=(0.5, 0.5))
        nodes, weights = np.polynomial.legendre.leggauss(400)
        edges = np.linspace(-45.0, 45.0, 6_001)
        half = 0.5 * (edges[1] - edges[0])
        centres = 0.5 * (edges[:-1] + edges[1:])
        z = (centres[:, None] + half * nodes[None, :]).reshape(-1)
        quadrature = np.tile(weights * half, centres.size)
        sigmoid = 1.0 / (1.0 + np.exp(-z))
        jacobian = sigmoid * (1.0 - sigmoid)
        mixture = np.exp(
            dd._logsumexp_rows(
                grid.component_log_ratio(sigmoid) + grid.log_component_weights
            )
        )
        self.assertAlmostEqual(float(np.dot(quadrature * jacobian, mixture)), 1.0, places=9)

    def test_shared_paths_match_a_direct_reference_computation(self) -> None:
        grid = small_tailwidth_grid()
        rng = np.random.default_rng(12)
        pivots = rng.uniform(size=(7, 6))
        values, maxima = grid.shared_paths(pivots, running_max=True)
        accumulator = np.zeros((7, grid.n_components))
        for time_index in range(6):
            accumulator += grid.component_log_ratio(pivots[:, time_index])
            expected = dd._logsumexp_rows(accumulator + grid.log_component_weights)
            np.testing.assert_allclose(values[:, time_index], expected, atol=1e-12)
        np.testing.assert_allclose(maxima, np.maximum.accumulate(values, axis=1))

    def test_collapsing_to_the_widest_atom_reproduces_the_shared_spike_rule(self) -> None:
        # End-to-end containment: with the prior collapsed to J = K the enlarged
        # rule must reproduce the existing shared spike detector bit for bit.
        config = paper.BenchmarkConfig(
            vocabulary_size=VOCABULARY,
            max_horizon=12,
            bayes_quadrature_nodes=16,
        )
        deltas_c, weights_c = dd.gauss_legendre_delta_grid(
            config.delta_low, config.delta_high, config.bayes_quadrature_nodes
        )
        grid = dd.TailWidthBayesGrid(
            delta_grid=deltas_c,
            delta_weights=weights_c,
            tail_size=VOCABULARY - 1,
            tail_width_grid=(VOCABULARY - 1,),
        )
        rng = np.random.default_rng(13)
        pivots = rng.uniform(size=(9, 12))
        mine, _ = grid.shared_paths(pivots, horizons=(12,))
        deltas, weights = dd.gauss_legendre_delta_grid(
            config.delta_low, config.delta_high, config.bayes_quadrature_nodes
        )
        log_r = np.log(pivots)
        top = (deltas / (1.0 - deltas))[None, None, :] * log_r[:, :, None]
        tail = math.log(VOCABULARY - 1) + (
            (VOCABULARY - 1) / deltas - 1.0
        )[None, None, :] * log_r[:, :, None]
        per_token = np.logaddexp(top, tail).sum(axis=1)
        reference = dd._logsumexp_rows(per_token + np.log(weights))
        np.testing.assert_allclose(mine[:, 0], reference, rtol=0.0, atol=1e-9)

    def test_prior_fingerprint_records_the_frozen_ladder(self) -> None:
        grid = small_tailwidth_grid()
        fingerprint = grid.prior_fingerprint()
        self.assertEqual(fingerprint["tail_width_grid"], list(grid.tail_widths))
        self.assertEqual(fingerprint["tail_size"], 39)
        self.assertTrue(fingerprint["closed_form"])
        self.assertAlmostEqual(sum(fingerprint["tail_width_weights"]), 1.0)
        self.assertEqual(fingerprint["n_components"], grid.n_components)

    def test_layer_is_shared_only(self) -> None:
        # The tokenwise hierarchy cannot learn a document-level tail width, so
        # the class deliberately exposes no tokenwise lookup.
        self.assertFalse(hasattr(dd.TailWidthBayesGrid, "tokenwise_lookup"))



class UnionTailTokenwiseStateTests(unittest.TestCase):
    """The tokenwise union rule must keep the tail state at DOCUMENT level.

    Section 3 and Figure 1 both place B outside the token plate in every
    hierarchy.  An earlier implementation reused DirichletTokenwiseLookup,
    which collapses every component at each token and therefore redraws B per
    position.  The two models differ by tens of nats on 200-token paths, so
    this is a modelling error rather than a numerical one.
    """

    def setUp(self) -> None:
        deltas, weights = dd.gauss_legendre_delta_grid(0.001, 0.5, 96)
        self.grid = dd.UnionTailBayesGrid(
            delta_grid=deltas, delta_weights=weights,
            tail_size=999, dirichlet_weight=0.5,
        )
        self.lookup = self.grid.tokenwise_lookup(size=80_001)

    def test_mixes_branch_products_not_per_token_marginals(self) -> None:
        pivots = np.random.default_rng(0).uniform(size=(200, 60)) ** 0.3
        got = self.lookup.paths(pivots)

        weights = np.asarray(self.grid.log_component_weights, dtype=float)
        per = self.grid.component_log_ratio(pivots.reshape(-1)).reshape(
            pivots.shape[0], pivots.shape[1], -1
        )
        terms = []
        for index in range(2):
            block = self.grid.block_slice(index)
            total = float(np.log(np.exp(weights[block]).sum()))
            ratio = dd._logsumexp_rows(per[:, :, block] + weights[None, None, block])
            terms.append(total + np.cumsum(ratio - total, axis=1))
        expected = np.logaddexp(terms[0], terms[1])
        # The gap is the branch tables' own interpolation error accumulated
        # over 60 tokens, not a modelling difference; the wrong model is off by
        # tens of nats, which the next test pins.
        self.assertLess(float(np.max(np.abs(got - expected))), 1e-4)

    def test_differs_from_collapsing_the_union_at_every_token(self) -> None:
        """The wrong model must be measurably different, or the test is vacuous."""

        pivots = np.random.default_rng(1).uniform(size=(200, 200)) ** 0.3
        correct = self.lookup.paths(pivots)
        wrong = np.cumsum(
            dd.DirichletTokenwiseLookup(self.grid, size=20_001)(pivots), axis=1
        )
        self.assertGreater(float(np.max(np.abs(correct - wrong))), 1.0)

    def test_degenerate_weights_collapse_to_a_single_branch(self) -> None:
        """At w=0 and w=1 the union is one branch, so B carries no information."""

        deltas, weights = dd.gauss_legendre_delta_grid(0.001, 0.5, 96)
        pivots = np.random.default_rng(2).uniform(size=(80, 40)) ** 0.3
        for weight, block in ((0.0, 1), (1.0, 0)):
            grid = dd.UnionTailBayesGrid(
                delta_grid=deltas, delta_weights=weights,
                tail_size=999, dirichlet_weight=weight,
            )
            lookup = grid.tokenwise_lookup(size=80_001)
            component = np.asarray(grid.log_component_weights, dtype=float)
            live = grid.block_slice(block)
            total = float(np.log(np.exp(component[live]).sum()))
            per = grid.component_log_ratio(pivots.reshape(-1)).reshape(
                pivots.shape[0], pivots.shape[1], -1
            )
            ratio = dd._logsumexp_rows(per[:, :, live] + component[None, None, live])
            expected = np.cumsum(ratio - total, axis=1)
            self.assertLess(
                float(np.max(np.abs(lookup.paths(pivots) - expected))), 1e-4
            )


class PooledWidthEndpointTests(unittest.TestCase):
    """The lambda=inf endpoint must BE the frozen ladder, not merely resemble it.

    The width-pooling experiment silently compared two supports: the pooled
    grid drops the full-width rung, the default TailWidthBayesGrid keeps it,
    and the two scores differed by about .15 log-BF units.  Any gain measured
    that way is partly a support difference, so the endpoint identity is the
    property that makes the experiment about pooling at all.
    """

    def setUp(self) -> None:
        self.k = 999
        self.deltas, self.weights = dd.gauss_legendre_delta_grid(0.001, 0.5, 96)
        self.ladder = tuple(
            j for j in dd.dyadic_tail_width_grid(self.k) if j != self.k
        )

    def test_pooled_grid_uses_the_union_width_support(self) -> None:
        pooled = wh.PooledWidthGrid(
            delta_grid=self.deltas, delta_weights=self.weights, tail_size=self.k
        )
        self.assertEqual(tuple(pooled.tail_widths), self.ladder)
        self.assertNotIn(self.k, pooled.tail_widths)

    def test_infinite_concentration_reproduces_the_frozen_ladder(self) -> None:
        pooled = wh.PooledWidthGrid(
            delta_grid=self.deltas, delta_weights=self.weights, tail_size=self.k,
            prior=wh.PooledWidthPrior(lambdas=(math.inf,)),
        )
        frozen = dd.TailWidthBayesGrid(
            delta_grid=self.deltas, delta_weights=self.weights,
            tail_size=self.k, tail_width_grid=self.ladder,
        )
        pivots = np.random.default_rng(0).uniform(size=(120, 50)) ** 0.3
        got = pooled.shared_paths(pivots)
        want = frozen.shared_paths(pivots)
        want = want[0] if isinstance(want, tuple) else want
        self.assertLess(float(np.max(np.abs(got - want))), 1e-10)

    def test_zero_concentration_is_not_the_frozen_ladder(self) -> None:
        """lambda=0 redraws per token; the module docstring once said otherwise.

        Both ends of the dial share the frozen prior's marginal, so an
        endpoint claim that names the wrong one is not caught by any identity
        the other tests check.  This pins the distinction directly.
        """

        flat = wh.PooledWidthGrid(
            delta_grid=self.deltas, delta_weights=self.weights, tail_size=self.k,
            prior=wh.PooledWidthPrior(lambdas=(0.0,)),
        )
        frozen = dd.TailWidthBayesGrid(
            delta_grid=self.deltas, delta_weights=self.weights,
            tail_size=self.k, tail_width_grid=self.ladder,
        )
        pivots = np.random.default_rng(2).uniform(size=(120, 50)) ** 0.3
        want = frozen.shared_paths(pivots)
        want = want[0] if isinstance(want, tuple) else want
        self.assertGreater(
            float(np.max(np.abs(flat.shared_paths(pivots) - want))), 1e-3
        )

    def test_a_mismatched_support_is_detectably_different(self) -> None:
        """Guard against the test above passing vacuously."""

        pooled = wh.PooledWidthGrid(
            delta_grid=self.deltas, delta_weights=self.weights, tail_size=self.k,
            prior=wh.PooledWidthPrior(lambdas=(math.inf,)),
        )
        default = dd.TailWidthBayesGrid(
            delta_grid=self.deltas, delta_weights=self.weights, tail_size=self.k
        )
        self.assertIn(self.k, default.tail_widths)
        pivots = np.random.default_rng(1).uniform(size=(120, 50)) ** 0.3
        other = default.shared_paths(pivots)
        other = other[0] if isinstance(other, tuple) else other
        self.assertGreater(
            float(np.max(np.abs(pooled.shared_paths(pivots) - other))), 1e-3
        )


if __name__ == "__main__":
    unittest.main()
