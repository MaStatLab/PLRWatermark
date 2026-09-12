"""Tests for the Tr-GoF statistic (trgof.py).

The load-bearing test in this module is :class:`ReferenceAgreementTests`, which
inlines the authors' ``compute_score`` from

    github.com/lx10077/TrGoF, ``simulation_codes/histogram_TrGoF.py``

verbatim and asserts *exact* floating-point equality with
:func:`trgof.statistic` across the Cressie--Read family.  Everything else in
this module is a property test around that anchor: the Higher-Criticism
identity at ``s = 2``, the truncation and one-sidedness, the two p-value maps,
power, and the calibration behaviour that makes the randomised boundary
necessary.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

import trgof
from bayesian_watermark import simulate_inverse_null

try:  # The randomised-boundary calibration path every rule is routed through.
    from benchmark_paper_experiment import (
        calibrate_randomized_boundary,
        expected_rejection_rate,
    )
except Exception:  # pragma: no cover - exercised only if the harness moves
    calibrate_randomized_boundary = None
    expected_rejection_rate = None


# --------------------------------------------------------------------------
# Reference implementation, reproduced verbatim.
# --------------------------------------------------------------------------
# Copied from github.com/lx10077/TrGoF, simulation_codes/histogram_TrGoF.py.
# ``acc_eps`` and ``mask`` are module-level globals there; their released
# values are 1e-10 and True.  Nothing below is reformatted: the point of this
# block is that it can be diffed character by character against the release.

acc_eps = 1e-10
mask = True


def compute_score(Ys, alpha=1., s=2, eps=acc_eps):
    ps = 1 - Ys
    ps = np.sort(ps)
    n = len(ps)
    first = int(len(ps) * alpha)
    ps = ps[:first]
    rk = np.arange(1, 1+first)/n

    if mask:
        ind = (ps >= 1/n) * (rk >= ps)
        ps = ps[ind]
        rk = rk[ind]

    if s == 1:
        final = rk * np.log(rk+eps) - rk*np.log(ps+eps) + (1-rk+eps) * np.log(1-rk+eps) - (1-rk) * np.log(1-ps+eps)
    elif s == 0:
        final = ps * np.log(ps+eps) - ps*np.log(rk+eps) + (1-ps+eps) * np.log(1-ps+eps) - (1-ps) * np.log(1-rk+eps)
    elif s == 2:
        final = (rk - ps)**2/(ps*(1-ps)+eps)/2
    elif s == 1/2:
        final = 2*(np.sqrt(rk)-np.sqrt(ps))**2 + 2*(np.sqrt(1-rk)-np.sqrt(1-ps))**2
    elif s >= 0:
        final = (1-(rk**s)*(ps+eps)**(1-s)-((1-rk)**s)*((1-ps+eps)**(1-s)))/(s*(1-s))
    elif s == -1:
        final = (rk - ps)**2/(rk*(1-rk)+eps)/2
    else:
        final = (1-ps**(1-s)/(rk+eps)**(-s)-(1-ps)**(1-s)/(1-rk+eps)**(-s))/(s*(1-s))

    return n*np.max(final)


# The full Cressie--Read sweep the tests run over: the five indices the
# authors' simulation code uses, plus two negative indices that exercise the
# two remaining branches (``s == -1`` and the generic negative one).
S_GRID: tuple[float, ...] = (2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0)


def _reference_scores(gumbel_pivots: np.ndarray, s: float) -> np.ndarray:
    """Row-wise reference score, with the empty-mask case defined.

    ``compute_score`` calls ``np.max`` on the masked array, which raises when
    the truncated index set is empty.  The vectorised implementation defines
    that degenerate maximum to be zero (the atom at zero); the only edit made
    here is to supply that value instead of letting NumPy raise.
    """

    rows = np.atleast_2d(np.asarray(gumbel_pivots, dtype=float))
    out = np.empty(rows.shape[0])
    for i, row in enumerate(rows):
        ps = np.sort(1 - row)
        n = ps.size
        rk = np.arange(1, n + 1) / n
        if mask and not np.any((ps >= 1 / n) * (rk >= ps)):
            out[i] = 0.0
        else:
            out[i] = compute_score(row, s=s)
    return out


def _ks_statistic(sample: np.ndarray) -> float:
    """One-sample Kolmogorov--Smirnov distance from Uniform(0, 1)."""

    u = np.sort(np.asarray(sample, dtype=float))
    n = u.size
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - u), np.max(u - (i - 1) / n)))


# Asymptotic one-sample KS critical value at the 0.1% level.  Every caller
# uses a fixed seed, so these checks are deterministic; the level is set this
# far out only so the tests stay comfortable under a change of seed (over
# twenty spare seeds the largest observed distance was 0.88 of this value).
def _ks_critical(n: int) -> float:
    return 1.95 / math.sqrt(n)


def _watermarked_gumbel(rows: int, n: int, delta: float, rng) -> np.ndarray:
    """Gumbel-max pivots under a single-token alternative, ``Y = U**P``."""

    return rng.uniform(size=(rows, n)) ** delta


class ReferenceAgreementTests(unittest.TestCase):
    """Exact agreement with the authors' released ``compute_score``."""

    def test_matches_reference_on_null_pivots(self) -> None:
        rng = np.random.default_rng(0)
        for n in (5, 37, 200, 1_000):
            pivots = rng.uniform(size=(24, n))
            p_values = trgof.gumbel_p_values(pivots)
            for s in S_GRID:
                expected = _reference_scores(pivots, s)
                observed = trgof.statistic(p_values, s=s)
                # Exact equality, not allclose: the reimplementation must
                # perform the same arithmetic, branch for branch.
                np.testing.assert_array_equal(
                    observed, expected, err_msg=f"n={n}, s={s}"
                )

    def test_matches_reference_on_watermarked_pivots(self) -> None:
        rng = np.random.default_rng(1)
        for delta in (0.2, 0.5, 0.8):
            for n in (37, 200):
                pivots = _watermarked_gumbel(16, n, delta, rng)
                p_values = trgof.gumbel_p_values(pivots)
                for s in S_GRID:
                    np.testing.assert_array_equal(
                        trgof.statistic(p_values, s=s),
                        _reference_scores(pivots, s),
                        err_msg=f"delta={delta}, n={n}, s={s}",
                    )

    def test_matches_reference_on_mixed_and_extreme_rows(self) -> None:
        # Rows deliberately chosen to land on the branch boundaries: pivots at
        # the ends of the unit interval, a row with a heavy excess of small
        # p-values, and a row with none at all.
        rows = np.array(
            [
                [0.0, 0.25, 0.5, 0.75, 1.0, 0.999999, 1e-9, 0.5, 0.5, 0.5],
                [0.99] * 10,
                [0.01] * 10,
                [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45],
            ]
        )
        p_values = trgof.gumbel_p_values(rows)
        with np.errstate(divide="ignore", invalid="ignore"):
            for s in S_GRID:
                np.testing.assert_array_equal(
                    trgof.statistic(p_values, s=s),
                    _reference_scores(rows, s),
                    err_msg=f"s={s}",
                )

    def test_matches_reference_with_the_mask_switched_off(self) -> None:
        # The release exposes the truncation as a module-level ``mask`` flag;
        # trgof exposes it as ``truncate``.  Both settings must agree.
        global mask
        rng = np.random.default_rng(2)
        pivots = rng.uniform(size=(12, 150))
        p_values = trgof.gumbel_p_values(pivots)
        mask = False
        try:
            for s in S_GRID:
                np.testing.assert_array_equal(
                    trgof.statistic(p_values, s=s, truncate=False),
                    _reference_scores(pivots, s),
                    err_msg=f"s={s}",
                )
        finally:
            mask = True

    def test_module_constants_match_the_release(self) -> None:
        # These constants are what the run metadata records as the sensitivity
        # sweep, so they are part of the contract.
        self.assertEqual(trgof.ACCURACY_EPS, acc_eps)
        self.assertEqual(trgof.DEFAULT_S, 2.0)
        self.assertEqual(trgof.DEFAULT_S_VALUES, (2.0, 1.5, 1.0, 0.5, 0.0))
        self.assertIn(trgof.DEFAULT_S, trgof.DEFAULT_S_VALUES)


class HigherCriticismTests(unittest.TestCase):
    """At s = 2 the statistic is one half the squared one-sided HC."""

    @staticmethod
    def _hc_plus(p_values: np.ndarray, truncate: bool = True) -> np.ndarray:
        values = np.atleast_2d(np.asarray(p_values, dtype=float))
        n = values.shape[1]
        ordered = np.sort(values, axis=1)
        rank = (np.arange(1, n + 1) / n)[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.sqrt(n) * (rank - ordered) / np.sqrt(ordered * (1.0 - ordered))
        keep = rank >= ordered
        if truncate:
            keep = keep & (ordered >= 1.0 / n)
        best = np.where(keep, z, -np.inf).max(axis=1)
        return np.where(np.isfinite(best), best, 0.0)

    def test_s2_is_half_the_squared_one_sided_hc(self) -> None:
        rng = np.random.default_rng(3)
        for pivots in (
            rng.uniform(size=(64, 300)),
            _watermarked_gumbel(64, 300, 0.4, rng),
        ):
            p_values = trgof.gumbel_p_values(pivots)
            hc = self._hc_plus(p_values)
            # eps=0 removes the release's numerical guard, which is the only
            # thing standing between the two expressions.
            observed = trgof.statistic(p_values, s=trgof.DEFAULT_S, eps=0.0)
            np.testing.assert_allclose(observed, 0.5 * hc**2, rtol=1e-12, atol=0.0)

    def test_the_released_eps_perturbs_the_identity_only_negligibly(self) -> None:
        rng = np.random.default_rng(4)
        p_values = trgof.gumbel_p_values(rng.uniform(size=(64, 300)))
        guarded = trgof.statistic(p_values, s=trgof.DEFAULT_S)
        exact = trgof.statistic(p_values, s=trgof.DEFAULT_S, eps=0.0)
        np.testing.assert_allclose(guarded, exact, rtol=1e-6, atol=1e-9)

    def test_hc_over_the_untruncated_set_is_at_least_as_large(self) -> None:
        # Dropping the truncation can only widen the maximisation set, so the
        # truncated statistic never exceeds the untruncated one.  This is what
        # "Tr" buys: it discards the unstable p < 1/n region.
        rng = np.random.default_rng(5)
        p_values = trgof.gumbel_p_values(rng.uniform(size=(200, 100)))
        truncated = trgof.statistic(p_values, s=trgof.DEFAULT_S)
        full = trgof.statistic(p_values, s=trgof.DEFAULT_S, truncate=False)
        self.assertTrue(np.all(full >= truncated - 1e-12))
        self.assertTrue(np.any(full > truncated + 1e-9))


class TruncationTests(unittest.TestCase):
    """The truncation and the one-sidedness both bite."""

    def test_no_excess_of_small_p_values_scores_zero(self) -> None:
        # Every order statistic sits strictly above its rank, p_(t) > t/n, so
        # nothing but the degenerate t = n term survives the one-sided mask.
        n = 10
        p_values = np.array([[t / n + 0.5 / n for t in range(1, n)] + [1.0]])
        self.assertEqual(trgof.statistic(p_values, s=trgof.DEFAULT_S)[0], 0.0)
        for s in S_GRID:
            # Other members of the family pick up an O(eps log eps) artefact
            # from the release's numerical guard; none of them see evidence.
            self.assertLessEqual(float(trgof.statistic(p_values, s=s)[0]), 1e-6)

    def test_all_p_values_below_one_over_n_score_exactly_zero(self) -> None:
        # The truncation empties the index set, and the maximum degenerates.
        n = 10
        p_values = np.array([[0.5 * t / n**2 for t in range(1, n + 1)]])
        for s in S_GRID:
            self.assertEqual(float(trgof.statistic(p_values, s=s)[0]), 0.0)
        # Without the truncation the same row is overwhelming evidence, which
        # is exactly the instability the truncation is there to remove.
        self.assertGreater(
            float(trgof.statistic(p_values, s=trgof.DEFAULT_S, truncate=False)[0]),
            50.0,
        )

    def test_an_excess_of_small_p_values_scores_positive(self) -> None:
        n = 100
        p_values = np.concatenate(
            [np.full(10, 0.02), np.linspace(0.2, 0.99, n - 10)]
        )[None, :]
        for s in S_GRID:
            self.assertGreater(float(trgof.statistic(p_values, s=s)[0]), 1.0)

    def test_statistic_depends_only_on_the_order_statistics(self) -> None:
        rng = np.random.default_rng(6)
        p_values = rng.uniform(size=(8, 120))
        shuffled = np.array([rng.permutation(row) for row in p_values])
        for s in S_GRID:
            np.testing.assert_array_equal(
                trgof.statistic(shuffled, s=s), trgof.statistic(p_values, s=s)
            )

    def test_statistic_is_not_a_running_sum_over_tokens(self) -> None:
        # The reason Tr-GoF has to be recomputed at every horizon rather than
        # accumulated: the per-horizon path is neither additive nor monotone.
        rng = np.random.default_rng(7)
        row = trgof.gumbel_p_values(rng.uniform(size=400))
        path = np.array(
            [float(trgof.statistic(row[None, :k], s=trgof.DEFAULT_S)[0])
             for k in range(1, 401)]
        )
        self.assertTrue(np.any(np.diff(path) < 0.0))
        halves = sum(
            float(trgof.statistic(part[None, :], s=trgof.DEFAULT_S)[0])
            for part in (row[:200], row[200:])
        )
        self.assertNotAlmostEqual(path[-1], halves, places=6)

    def test_one_dimensional_input_is_treated_as_a_single_row(self) -> None:
        rng = np.random.default_rng(8)
        row = trgof.gumbel_p_values(rng.uniform(size=50))
        flat = trgof.statistic(row, s=trgof.DEFAULT_S)
        self.assertEqual(flat.shape, (1,))
        np.testing.assert_array_equal(flat, trgof.statistic(row[None, :]))


class GumbelPValueTests(unittest.TestCase):
    def test_transform_is_the_released_one(self) -> None:
        rng = np.random.default_rng(9)
        pivots = rng.uniform(size=1_000)
        np.testing.assert_array_equal(trgof.gumbel_p_values(pivots), 1 - pivots)

    def test_p_values_are_uniform_under_the_gumbel_null(self) -> None:
        rng = np.random.default_rng(10)
        p = trgof.gumbel_p_values(rng.uniform(size=400_000))
        self.assertTrue(np.all((p >= 0.0) & (p <= 1.0)))
        self.assertLess(_ks_statistic(p), _ks_critical(p.size))

    def test_p_values_are_stochastically_small_under_the_alternative(self) -> None:
        rng = np.random.default_rng(11)
        p = trgof.gumbel_p_values(_watermarked_gumbel(1, 200_000, 0.5, rng).ravel())
        self.assertGreater(_ks_statistic(p), 10.0 * _ks_critical(p.size))
        self.assertLess(float(p.mean()), 0.45)


class InversePValueTests(unittest.TestCase):
    """Our extension: p = F_0(d) at the exact finite-vocabulary null.

    The released implementation applies Tr-GoF to Gumbel-max pivots only.  The
    inverse-transform p-value below is not the authors'; it is validated here
    against a brute-force sum over ranks and against the simulated exact null.
    """

    @staticmethod
    def _brute_force_cdf(d: np.ndarray, m: int) -> np.ndarray:
        """F_0(d) = (1/M) sum_k P(|U - k/(M-1)| <= d), summed over ranks."""

        eta = np.arange(m) / (m - 1.0)
        d = np.atleast_1d(np.asarray(d, dtype=float))
        upper = np.minimum(1.0, eta[None, :] + d[:, None])
        lower = np.maximum(0.0, eta[None, :] - d[:, None])
        return (upper - lower).mean(axis=1)

    def test_matches_a_brute_force_sum_over_ranks_at_small_vocabulary(self) -> None:
        grid = np.linspace(0.0, 1.0, 97)
        for m in (2, 3, 5, 8, 13, 64, 257):
            np.testing.assert_allclose(
                trgof.inverse_p_values(grid, m),
                self._brute_force_cdf(grid, m),
                rtol=0.0,
                atol=1e-12,
                err_msg=f"M={m}",
            )

    def test_endpoints_and_monotonicity(self) -> None:
        for m in (2, 50, 1_000, 50_272):
            p = trgof.inverse_p_values(np.linspace(0.0, 1.0, 5_001), m)
            self.assertEqual(float(p[0]), 0.0)
            self.assertAlmostEqual(float(p[-1]), 1.0, places=12)
            self.assertTrue(np.all(np.diff(p) >= -1e-15))
            self.assertTrue(np.all((p >= 0.0) & (p <= 1.0)))

    def test_p_values_are_uniform_under_the_exact_finite_v_null(self) -> None:
        rng = np.random.default_rng(12)
        for m in (1_000, 50_272):
            d = simulate_inverse_null(400_000, rng, vocabulary_size=m)
            p = trgof.inverse_p_values(d, m)
            self.assertLess(
                _ks_statistic(p), _ks_critical(p.size), msg=f"M={m}"
            )
            self.assertAlmostEqual(float(p.mean()), 0.5, places=2)

    def test_small_d_is_the_evidence_direction(self) -> None:
        # The one-sided choice p = F_0(d), not 1 - F_0(d): the watermark
        # concentrates the inverse pivot near zero.
        rng = np.random.default_rng(13)
        m = 50_272
        alt = 0.6 * simulate_inverse_null(200_000, rng, vocabulary_size=m)
        p = trgof.inverse_p_values(alt, m)
        self.assertLess(float(p.mean()), 0.45)
        self.assertGreater(float(np.mean(p < 0.05)), 0.05)


class PowerTests(unittest.TestCase):
    def test_gumbel_statistic_is_larger_under_the_watermark(self) -> None:
        rng = np.random.default_rng(14)
        n = 200
        null = trgof.statistic(
            trgof.gumbel_p_values(rng.uniform(size=(2_000, n))), s=trgof.DEFAULT_S
        )
        for delta in (0.8, 0.5, 0.2):
            alt = trgof.statistic(
                trgof.gumbel_p_values(_watermarked_gumbel(2_000, n, delta, rng)),
                s=trgof.DEFAULT_S,
            )
            self.assertGreater(float(alt.mean()), float(null.mean()))
            self.assertGreater(float(np.median(alt)), float(np.median(null)))

    def test_inverse_statistic_is_larger_under_the_watermark(self) -> None:
        rng = np.random.default_rng(15)
        n, m, rows = 200, 50_272, 2_000
        null_d = simulate_inverse_null(rows * n, rng, vocabulary_size=m)
        alt_d = 0.5 * simulate_inverse_null(rows * n, rng, vocabulary_size=m)
        null = trgof.statistic(
            trgof.inverse_p_values(null_d.reshape(rows, n), m), s=trgof.DEFAULT_S
        )
        alt = trgof.statistic(
            trgof.inverse_p_values(alt_d.reshape(rows, n), m), s=trgof.DEFAULT_S
        )
        self.assertGreater(float(alt.mean()), 3.0 * float(null.mean()))

    def test_power_is_monotone_in_the_signal(self) -> None:
        rng = np.random.default_rng(16)
        n = 200
        means = [
            float(
                trgof.statistic(
                    trgof.gumbel_p_values(_watermarked_gumbel(3_000, n, delta, rng)),
                    s=trgof.DEFAULT_S,
                ).mean()
            )
            for delta in (1.0, 0.8, 0.6, 0.4)
        ]
        self.assertTrue(all(a < b for a, b in zip(means, means[1:])), means)


class CalibrationTests(unittest.TestCase):
    """Calibration, and why the randomised boundary matters here."""

    def test_null_95th_percentile_gives_about_a_five_percent_rejection_rate(
        self,
    ) -> None:
        rng = np.random.default_rng(17)
        n, rows = 100, 20_000
        calibration = trgof.statistic(
            trgof.gumbel_p_values(rng.uniform(size=(rows, n))), s=trgof.DEFAULT_S
        )
        threshold = float(np.quantile(calibration, 0.95))
        evaluation = trgof.statistic(
            trgof.gumbel_p_values(rng.uniform(size=(rows, n))), s=trgof.DEFAULT_S
        )
        rate = float(np.mean(evaluation > threshold))
        # Monte Carlo error is about 0.0022 once the noise in the threshold is
        # counted, so 0.008 is roughly a 3.5-sigma band.
        self.assertAlmostEqual(rate, 0.05, delta=0.008)

    def test_the_atom_at_zero_is_present_at_short_horizons(self) -> None:
        # The truncated index set is empty exactly when every p-value falls
        # below 1/n, which has null probability n**(-n).  That is negligible at
        # long horizons but dominant at the short prefixes the curves start
        # from, so the null law of the statistic has a genuine atom at zero.
        rng = np.random.default_rng(18)
        rows = 200_000
        for n in (1, 2, 3, 5):
            scores = trgof.statistic(
                trgof.gumbel_p_values(rng.uniform(size=(rows, n))),
                s=trgof.DEFAULT_S,
            )
            observed = float(np.mean(scores == 0.0))
            self.assertAlmostEqual(observed, n ** (-n), delta=0.004, msg=f"n={n}")
            self.assertGreater(observed, 0.0, msg=f"n={n}")
        # At n = 1 the null statistic is degenerate at zero, so no fixed
        # threshold can have size 0.05: strict comparison rejects never and
        # non-strict comparison rejects always.  Only the randomised boundary
        # attains the nominal level.
        degenerate = trgof.statistic(
            trgof.gumbel_p_values(rng.uniform(size=(10_000, 1))), s=trgof.DEFAULT_S
        )
        self.assertTrue(np.all(degenerate == 0.0))
        self.assertEqual(float(np.mean(degenerate > 0.0)), 0.0)
        self.assertEqual(float(np.mean(degenerate >= 0.0)), 1.0)

    def test_the_atom_is_gone_at_the_horizons_the_tables_report(self) -> None:
        rng = np.random.default_rng(19)
        for n in (50, 200):
            scores = trgof.statistic(
                trgof.gumbel_p_values(rng.uniform(size=(20_000, n))),
                s=trgof.DEFAULT_S,
            )
            self.assertEqual(float(np.mean(scores == 0.0)), 0.0, msg=f"n={n}")

    @unittest.skipIf(
        calibrate_randomized_boundary is None,
        "benchmark_paper_experiment calibration helpers unavailable",
    )
    def test_the_shared_randomized_boundary_path_hits_the_nominal_level(self) -> None:
        # Tr-GoF must go through the same calibration path as every other rule.
        # With the atom at zero present, the cutoff-plus-gamma rule still lands
        # on 5% exactly at every horizon, including the degenerate n = 1.
        rng = np.random.default_rng(20)
        alpha = 0.05
        rows, horizons = 20_000, (1, 2, 3, 10, 100)
        pivots = rng.uniform(size=(rows, max(horizons)))
        p_values = trgof.gumbel_p_values(pivots)
        paths = np.column_stack(
            [
                trgof.statistic(p_values[:, :k], s=trgof.DEFAULT_S)
                for k in horizons
            ]
        )
        cutoffs, gammas = calibrate_randomized_boundary({"trgof_s2": paths}, alpha)
        rates = expected_rejection_rate(
            paths, cutoffs["trgof_s2"], gammas["trgof_s2"]
        )
        np.testing.assert_allclose(rates, alpha, atol=1e-12)
        # The n = 1 column is pure boundary mass: the cutoff is the atom
        # itself and the whole level comes from the randomisation.
        self.assertEqual(float(cutoffs["trgof_s2"][0]), 0.0)
        self.assertAlmostEqual(float(gammas["trgof_s2"][0]), alpha, places=12)

    def test_calibration_is_deterministic_given_the_null_sample(self) -> None:
        # No hidden randomness inside the statistic: adding this rule cannot
        # perturb any other rule's draws.
        rng = np.random.default_rng(21)
        p_values = trgof.gumbel_p_values(rng.uniform(size=(500, 120)))
        first = trgof.statistic(p_values, s=trgof.DEFAULT_S)
        second = trgof.statistic(p_values.copy(), s=trgof.DEFAULT_S)
        np.testing.assert_array_equal(first, second)


class SensitivitySweepTests(unittest.TestCase):
    """The other Cressie--Read indices are recorded, not promoted to rules."""

    def test_every_swept_index_is_computable_and_agrees_with_the_reference(
        self,
    ) -> None:
        rng = np.random.default_rng(22)
        pivots = _watermarked_gumbel(32, 200, 0.5, rng)
        p_values = trgof.gumbel_p_values(pivots)
        for s in trgof.DEFAULT_S_VALUES:
            observed = trgof.statistic(p_values, s=s)
            np.testing.assert_array_equal(observed, _reference_scores(pivots, s))
            self.assertTrue(np.all(np.isfinite(observed)))

    def test_every_swept_index_separates_null_from_alternative(self) -> None:
        rng = np.random.default_rng(23)
        n = 200
        null_p = trgof.gumbel_p_values(rng.uniform(size=(2_000, n)))
        alt_p = trgof.gumbel_p_values(_watermarked_gumbel(2_000, n, 0.5, rng))
        for s in trgof.DEFAULT_S_VALUES:
            null = trgof.statistic(null_p, s=s)
            alt = trgof.statistic(alt_p, s=s)
            threshold = float(np.quantile(null, 0.95))
            self.assertGreater(
                float(np.mean(alt > threshold)), 0.5, msg=f"s={s}"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
