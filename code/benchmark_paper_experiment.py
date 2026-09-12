#!/usr/bin/env python3
"""Benchmark Bayesian detectors in clean designs adapted from Li et al. (2025).

The tokenwise-deficit design follows their written synthetic specification:
the vocabulary size is 1,000, every Delta_t is drawn iid from
Uniform(0.001, 0.5), and the NTP is the spike vector

    (1 - Delta_t, Delta_t/(V-1), ..., Delta_t/(V-1)).

All fixed-horizon tests are calibrated on the same independent null sample.
At atoms (the indicator score), a randomized boundary test is used so that
the calibration target is exactly alpha up to Monte Carlo resolution.  The
Bayes-factor anytime rule is reported separately and is never calibrated on
the fixed horizon: it rejects when max_{s <= t} BF_s >= 1/alpha.

The truncated goodness-of-fit statistic of Li et al. (2024) is carried as a
competitor for both pivot schemes.  It is not a token sum, so it is recomputed
at every reported prefix, and it is calibrated exactly like every other rule.

The implementation is vectorized and depends only on NumPy and Matplotlib.
It writes a long-form CSV, a metadata/summary JSON, and PNG/PDF figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from experiment_metadata import union_tail_metadata

import dirichlet_detector as dirichlet
import trgof


PAPER_URL = "https://arxiv.org/abs/2404.01245"
OFFICIAL_CODE_COMMIT = "05b7ffda9279fc9e645f38807e4a0e2dbcff4330"
OFFICIAL_CODE_URL = (
    "https://github.com/lx10077/WatermarkFramework/tree/"
    f"{OFFICIAL_CODE_COMMIT}"
)
OFFICIAL_RESULT_ARRAYS = (
    "simulation/results_data/"
    "K1000N5000c5key23333T700Delta0.5-alpha0.05-max-result.json",
    "simulation/results_data/"
    "K1000N5000c5key23333T700Delta0.5-alpha0.05-inv-result.json",
)
DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "bayesian_paper_benchmark"
)
DEFAULT_QUICK_RESULTS_DIR = DEFAULT_RESULTS_DIR / "quick" / "clean_benchmark"

GUMBEL_METHODS = (
    "h_ars",
    "h_gum_star_0.1",
    "h_log",
    "h_ind_1_over_e",
    "h_gum_star_0.01",
    "h_gum_star_0.005",
    "h_spike_0.01",
    "h_spike_0.05",
    "bayes_tokenwise",
    "bayes_tokenwise_dirichlet",
    "bayes_tokenwise_uniontail",
)

# Bayes-factor paths whose running maximum is tracked for the anytime comparison.
# Exact e-process guarantees apply to analytic or exactly normalized versions;
# lookup-table interpolation is the numerical caveat recorded in the metadata
# and manuscript.  Frequentist fixed-horizon scores are not tracked here.
ANYTIME_METHODS = ("bayes_tokenwise", "bayes_tokenwise_dirichlet")

# The Dirichlet tail layer applies to the Gumbel pivot only.  For the inverse
# pivot the limiting alternative depends on Delta alone, so a prior on the tail
# shape is inert by construction and there is nothing to add.
DIRICHLET_SHARED_METHOD = "bayes_shared_dirichlet"
DIRICHLET_TOKENWISE_METHOD = "bayes_tokenwise_dirichlet"
DIRICHLET_ALPHA_GRID: tuple[float, ...] = dirichlet.DEFAULT_ALPHA_GRID

# The enlarged Gumbel layer places a prior on the number of live tail
# coordinates J instead of fixing the tail to be equal over all V-1 of them.
# J = V-1 is exactly the equal-tail spike that every other Gumbel rule assumes,
# so this family contains them and the finite-grid evidence bound applies.  The
# component is closed form, so unlike the Dirichlet tail-shape layer it needs no
# transform table and no interpolation.  Shared hierarchy only: the tokenwise
# rule averages over J before multiplying and so cannot learn a document-level
# tail width.  It is reported anyway, as the tokenwise arm, because that
# prediction is worth showing rather than asserting: the two hierarchies then
# differ only in where the width is integrated, and the shared arm's advantage
# over it is the value of learning a document-level width.
UNIONTAIL_SHARED_METHOD = "bayes_shared_uniontail"
UNIONTAIL_TOKENWISE_METHOD = "bayes_tokenwise_uniontail"
INVERSE_METHODS = (
    "h_neg",
    "h_dif_star_0.1",
    "h_dif_star_0.01",
    "h_dif_star_0.001",
    "bayes_tokenwise",
)

# The truncated goodness-of-fit competitor of Li et al. (2024), evaluated for
# both pivot schemes.  It is deliberately absent from GUMBEL_METHODS and
# INVERSE_METHODS: those tuples drive the token-sum path in ``collect_paths``,
# and Tr-GoF is not a token sum.  It sorts the p-values of the entire prefix, so
# it has to be recomputed at every horizon, exactly like the shared-latent Bayes
# rules, and it is collected by :func:`collect_trgof_paths` instead.
TRGOF_METHOD = "trgof_s2"
TRGOF_S = trgof.DEFAULT_S
# The remaining Cressie--Read indices the authors sweep.  Recorded as a
# sensitivity in the metadata only; s = 2 (Higher Criticism) is the single
# evaluated rule, because one competitor row per scheme is what the tables can
# carry.
TRGOF_S_SENSITIVITY: tuple[float, ...] = tuple(
    s for s in trgof.DEFAULT_S_VALUES if s != TRGOF_S
)

# Diagnostic scores defined here rather than evaluated by Li et al. (2025).
# They use the data-generating spike likelihood family at one fixed Delta with
# no Bayesian averaging, isolating component-family choice from averaging over
# Delta.
DIAGNOSTIC_METHODS = ("h_spike_0.01", "h_spike_0.05")

# The reference scores of Li et al. (2025) that the shared-deficit table
# displays, and the family the manuscript Holm-corrects across.  Defining this
# explicitly matters: paired_comparisons.py used to treat every non-diagnostic
# comparator as a reference score, so our OWN Bayes rules and the untabulated
# h*_gum,.1 were corrected across as though they were competitors from the
# cited paper, inflating the family from 15 to 21.
PAPER_REFERENCE_SCORES: tuple[str, ...] = (
    "h_ars",
    "h_log",
    "h_ind_1_over_e",
    # All three published tunings.  ".1" was omitted while it went untabulated;
    # the tables now display it, so leaving it out would correct over a family
    # smaller than the one a reader scans.
    "h_gum_star_0.1",
    "h_gum_star_0.01",
    "h_gum_star_0.005",
)

# Horizons at which per-document rejection indicators are persisted.
INDICATOR_HORIZONS = (100, 300, 700)

# The paired indicators realise the calibrated randomisation at score atoms.
# A dedicated, addressable seed keeps those auxiliary uniforms independent of
# every pivot stream and invariant to method ordering or future added methods.
BOUNDARY_RANDOMIZATION_RULE_VERSION = "randomized_boundary_v1"
BOUNDARY_RANDOMIZATION_SEED_SALT = 0x42574D31
_BOUNDARY_SCENARIO_CODES = {
    "paper_text_iid_delta_equal_tail": 0,
    "shared_delta_equal_tail_sensitivity": 1,
}
_BOUNDARY_SCHEME_CODES = {"gumbel": 0, "inverse": 1}


@dataclass(frozen=True)
class BenchmarkConfig:
    vocabulary_size: int = 1000
    max_horizon: int = 700
    alpha: float = 0.05
    # The generating law of Li et al. (2025): Delta ~ Uniform(delta_low,
    # delta_high) per document or per token.  These define the experiment and
    # are not changed.
    delta_low: float = 0.001
    delta_high: float = 0.5
    # The detector's prior support, kept separate from the generator so the
    # detector is not told the generating range.  The default is the whole unit
    # interval, represented by an interior pair because the component density
    # has removable singularities at both endpoints: Delta/(1-Delta) diverges at
    # one and J/Delta - 1 at zero.
    prior_low: float = 0.001
    prior_high: float = 0.5
    # Above one half the designated leading coordinate is no longer the largest
    # at every tail width, so Delta stops naming a top-probability deficit.  The
    # whole-interval prior accepts that deliberately, by opting in.
    allow_large_delta: bool = False
    n_calibration: int = 10_000
    n_evaluation_null: int = 5_000
    n_evaluation_alternative: int = 5_000
    batch_size: int = 1_000
    seed: int = 24_040_1245
    bayes_quadrature_nodes: int = 96
    gumbel_lookup_size: int = 80_001
    gumbel_lookup_logit_limit: float = 30.0
    # Resolution of the tabulated Dirichlet tail transform.  On the separate
    # deterministic grid documented in the manuscript, the 20,001-node
    # tail_family default has a 9.5e-6 largest component log-density discrepancy
    # from direct quadrature, versus 9.5e-8 at 200,001 nodes.  These are empirical
    # grid diagnostics, not uniform bounds; this setting is JSON-safe as a plain int.
    dirichlet_c_nodes: int = dirichlet.DEFAULT_C_NODES

    def validate(self) -> None:
        if self.vocabulary_size < 2:
            raise ValueError("vocabulary_size must be at least 2")
        if self.max_horizon < 1:
            raise ValueError("max_horizon must be positive")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if not 0.0 <= self.delta_low < self.delta_high < 1.0:
            raise ValueError("require 0 <= delta_low < delta_high < 1")
        if not 0.0 <= self.prior_low < self.prior_high < 1.0:
            raise ValueError("require 0 <= prior_low < prior_high < 1")
        max_delta = 1.0 - 1.0 / self.vocabulary_size
        if self.delta_high > max_delta + 1e-12:
            raise ValueError(
                "the equal-tail spike requires "
                f"delta_high <= 1 - 1/vocabulary_size = {max_delta:g}"
            )
        largest_bound = max(self.delta_high, self.prior_high)
        if largest_bound > 0.5 + 1e-12 and not self.allow_large_delta:
            raise ValueError(
                "the Dirichlet tail layer requires delta_high <= 0.5 so that "
                "the designated leading coordinate remains largest"
            )
        for name in ("n_calibration", "n_evaluation_null", "n_evaluation_alternative"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.bayes_quadrature_nodes < 8:
            raise ValueError("bayes_quadrature_nodes must be at least 8")
        if self.gumbel_lookup_size < 1001:
            raise ValueError("gumbel_lookup_size must be at least 1001")
        if not (
            math.isfinite(self.gumbel_lookup_logit_limit)
            and self.gumbel_lookup_logit_limit > 0.0
        ):
            raise ValueError("gumbel_lookup_logit_limit must be positive and finite")
        if self.dirichlet_c_nodes < 16:
            raise ValueError("dirichlet_c_nodes must be at least 16")
        if not isinstance(self.seed, (int, np.integer)) or int(self.seed) < 0:
            raise ValueError("seed must be a non-negative integer")


def _safe_logit(probability: np.ndarray) -> np.ndarray:
    tiny = np.finfo(float).tiny
    epsilon = np.finfo(float).eps
    clipped = np.clip(probability, tiny, 1.0 - epsilon)
    return np.log(clipped) - np.log1p(-clipped)


class GumbelBayesLookup:
    """Fast interpolation of the exact spike-prior marginal density.

    Under a spike NTP with Delta=delta,

      f(r | delta) = r**(delta/(1-delta))
                     + (V-1) r**((V-1)/delta - 1).

    We average this exact density over the Uniform(delta_low, delta_high)
    prior by Gauss-Legendre quadrature.  A uniform logit(r) grid resolves both
    endpoints, including the narrow tail-token contribution near r=1.
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        nodes, weights = np.polynomial.legendre.leggauss(config.bayes_quadrature_nodes)
        low, high = effective_prior_support(config)
        deltas = low + 0.5 * (nodes + 1.0) * (
            high - low
        )
        normalized_weights = 0.5 * weights
        logits = np.linspace(
            -config.gumbel_lookup_logit_limit,
            config.gumbel_lookup_logit_limit,
            config.gumbel_lookup_size,
        )
        log_density = np.empty_like(logits)
        top_exponents = deltas / (1.0 - deltas)
        tail_exponents = (config.vocabulary_size - 1.0) / deltas - 1.0
        log_tail_count = math.log(config.vocabulary_size - 1.0)

        chunk = 4_000
        for start in range(0, logits.size, chunk):
            stop = min(start + chunk, logits.size)
            z = logits[start:stop]
            # log(sigmoid(z)), stable at both endpoints.
            log_r = -np.logaddexp(0.0, -z)
            top = np.exp(log_r[:, None] * top_exponents[None, :])
            tail = np.exp(
                log_tail_count + log_r[:, None] * tail_exponents[None, :]
            )
            density = (top + tail) @ normalized_weights
            log_density[start:stop] = np.log(density)

        self.logits = logits
        self.log_density = log_density
        self.vocabulary_size = config.vocabulary_size
        self._top_exponents = top_exponents
        self._tail_exponents = tail_exponents
        self._log_tail_count = log_tail_count
        self._quadrature_weights = normalized_weights

    def __call__(self, pivots: np.ndarray) -> np.ndarray:
        return np.interp(
            _safe_logit(pivots),
            self.logits,
            self.log_density,
            left=self.log_density[0],
            right=self.log_density[-1],
        )

    def _direct_log_density_at_logits(self, logits: np.ndarray) -> np.ndarray:
        """Evaluate the configured quadrature without lookup interpolation."""

        values = np.asarray(logits, dtype=float)
        output = np.empty_like(values)
        chunk = 4_000
        for start in range(0, values.size, chunk):
            stop = min(start + chunk, values.size)
            log_r = -np.logaddexp(0.0, -values[start:stop])
            top = np.exp(log_r[:, None] * self._top_exponents[None, :])
            tail = np.exp(
                self._log_tail_count
                + log_r[:, None] * self._tail_exponents[None, :]
            )
            output[start:stop] = np.log(
                (top + tail) @ self._quadrature_weights
            )
        return output

    def validation_report(
        self, *, panels: int = 2_000, gauss_nodes: int = 40, limit: float = 40.0
    ) -> dict[str, object]:
        """Record reproducible accuracy diagnostics for the interpolated lookup.

        Midpoint error is measured against direct evaluation of the same frozen
        deficit quadrature in every table interval.  Density mass is integrated
        in logit coordinates.  Both are floating-point diagnostics, not
        certified one-sided error bounds.
        """

        midpoints = 0.5 * (self.logits[:-1] + self.logits[1:])
        interpolated = np.interp(midpoints, self.logits, self.log_density)
        direct = self._direct_log_density_at_logits(midpoints)
        error = np.abs(interpolated - direct)
        worst = int(np.argmax(error))

        nodes, weights = np.polynomial.legendre.leggauss(int(gauss_nodes))
        edges = np.linspace(-float(limit), float(limit), int(panels) + 1)
        half = 0.5 * (edges[1] - edges[0])
        centres = 0.5 * (edges[:-1] + edges[1:])
        mass = 0.0
        for start in range(0, centres.size, 128):
            stop = min(start + 128, centres.size)
            z = (centres[start:stop, None] + half * nodes[None, :]).reshape(-1)
            quadrature = np.tile(weights * half, stop - start)
            sigmoid = 1.0 / (1.0 + np.exp(-z))
            jacobian = sigmoid * (1.0 - sigmoid)
            log_density = np.interp(
                z,
                self.logits,
                self.log_density,
                left=self.log_density[0],
                right=self.log_density[-1],
            )
            mass += float(np.sum(quadrature * jacobian * np.exp(log_density)))

        return {
            "diagnostic_only": True,
            "certified_one_sided_bound": False,
            "midpoint_log_density": {
                "intervals_checked": int(midpoints.size),
                "max_abs_error": float(error[worst]),
                "argmax_logit": float(midpoints[worst]),
                "reference": (
                    "direct evaluation of the same frozen Gauss-Legendre "
                    "deficit quadrature"
                ),
            },
            "interpolated_density_mass": {
                "mass": mass,
                "mass_minus_one": mass - 1.0,
                "logit_quadrature": {
                    "panels": int(panels),
                    "gauss_nodes": int(gauss_nodes),
                    "limit": float(limit),
                },
            },
        }


def effective_prior_support(config: "BenchmarkConfig") -> tuple[float, float]:
    """The prior support actually used, clamped to what the vocabulary allows.

    The whole-interval default is stated vocabulary-free, but no NTP on $V$
    tokens has a deficit above ``1 - 1/V``: the leading coordinate cannot fall
    below a uniform share.  Quadrature nodes beyond that describe nothing, so
    the upper limit is clamped rather than rejected, which lets one prior
    specification serve every vocabulary in the study.
    """
    return config.prior_low, min(config.prior_high, 1.0 - 1.0 / config.vocabulary_size)


def build_dirichlet_grid(
    config: BenchmarkConfig,
    alpha_grid: tuple[float, ...] | None = None,
) -> dirichlet.DirichletBayesGrid:
    """The frozen joint prior on (Delta, alpha) for the Gumbel pivot.

    The deficit factor is the same Gauss-Legendre grid every spike-family rule
    uses, so the Dirichlet rule differs from the existing shared rule only in
    carrying a second latent component.  The alpha factor is uniform on
    ``DIRICHLET_ALPHA_GRID`` and includes the equal-tail limit, which means the
    existing shared rule is one component of this mixture.
    """

    deltas, weights = dirichlet.gauss_legendre_delta_grid(
        *effective_prior_support(config), config.bayes_quadrature_nodes
    )
    return dirichlet.DirichletBayesGrid(
        delta_grid=deltas,
        delta_weights=weights,
        alpha_grid=DIRICHLET_ALPHA_GRID if alpha_grid is None else alpha_grid,
        tail_size=config.vocabulary_size - 1,
        c_nodes=config.dirichlet_c_nodes,
        allow_large_delta=config.allow_large_delta,
    )


def build_uniontail_grid(
    config: BenchmarkConfig,
    tail_width_grid: tuple[int, ...] | None = None,
) -> dirichlet.UnionTailBayesGrid:
    """The frozen union prior on (Delta, alpha, J) for the Gumbel pivot.

    The deficit factor is the same Gauss-Legendre grid every spike-family rule
    uses.  The tail factor is a union, not a product: weight
    ``DEFAULT_UNION_DIRICHLET_WEIGHT`` on the Dirichlet tail-shape family at the
    full width, and the remainder on the tail-width ladder ``1, 4, ..., V/4`` at
    an equal tail.  Both existing Gumbel layers, and the equal-tail spike, are
    components of this mixture.
    """

    deltas, weights = dirichlet.gauss_legendre_delta_grid(
        *effective_prior_support(config), config.bayes_quadrature_nodes
    )
    return dirichlet.UnionTailBayesGrid(
        delta_grid=deltas,
        delta_weights=weights,
        tail_size=config.vocabulary_size - 1,
        alpha_grid=DIRICHLET_ALPHA_GRID,
        tail_width_grid=tail_width_grid,
        c_nodes=config.dirichlet_c_nodes,
        allow_large_delta=config.allow_large_delta,
    )


def inverse_exact_null_density(d: np.ndarray, vocabulary_size: int) -> np.ndarray:
    """Exact density of |U-J/(V-1)| for uniform discrete J."""

    counts_one_side = vocabulary_size - 1 - np.floor(
        d * (vocabulary_size - 1)
    )
    return 2.0 * counts_one_side / vocabulary_size


def inverse_bayes_log_ratio(d: np.ndarray, config: BenchmarkConfig) -> np.ndarray:
    """Tokenwise Bayes log ratio using the triangular H1 limit of Li et al. (2025).

    The integral over a Uniform(a,b) prior is analytic.  With
    c=min(b,1-d), it is

      2/(b-a) [ -log(1-delta) - d/(1-delta) ]_a^c.

    The denominator is the exact finite-V pivot null density, retaining an
    exact null Bayes factor even though the alternative is asymptotic.
    """

    d = np.asarray(d, dtype=float)
    if np.any(~np.isfinite(d)) or np.any((d < 0.0) | (d >= 1.0)):
        raise ValueError("inverse pivots must be finite and lie in [0, 1)")
    a, b = effective_prior_support(config)
    upper = np.minimum(b, 1.0 - d)
    valid = (upper > a) & (d < 1.0 - a)

    marginal = np.zeros_like(d)
    if np.any(valid):
        # With s=1-a, u=max(1-b,d), and t=(upper-a)/s, the integral is
        # 2/(b-a) * [-log(1-t)-t + t*(u-d)/u].  Both terms are nonnegative.
        # Subtracting antiderivatives loses the entire positive answer near
        # d=1-a.  Evaluate -log(1-t)-t by its positive series when t is small.
        width = upper[valid] - a
        u = np.maximum(1.0 - b, d[valid])
        t = width / (1.0 - a)
        remainder = np.empty_like(t)
        small = t < 0.01
        series = np.full_like(t[small], 1.0 / 12.0)
        for order in range(11, 1, -1):
            series = 1.0 / order + t[small] * series
        remainder[small] = t[small] ** 2 * series
        remainder[~small] = np.log1p(width[~small] / u[~small]) - t[~small]
        # Compute u-d as (1-d)-upper to avoid another cancellation after
        # rounding 1-b when the prior interval is very narrow.
        residual = (1.0 - d[valid]) - upper[valid]
        marginal[valid] = 2.0 / (b - a) * (
            remainder + t * (residual / u)
        )
    f0 = inverse_exact_null_density(d, config.vocabulary_size)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(marginal) - np.log(f0)


def paper_gumbel_optimal_score(r: np.ndarray, delta: float) -> np.ndarray:
    """Least-favorable Gumbel log likelihood ratio evaluated by Li et al. (2025)."""

    cap = 1.0 - delta
    count = int(math.floor(1.0 / cap + 1e-12))
    residual = 1.0 - count * cap
    log_r = np.log(r)
    first = math.log(count) + (delta / cap) * log_r
    if residual <= 1e-14:
        return first
    second = (1.0 / residual - 1.0) * log_r
    return np.logaddexp(first, second)


def spike_point_mass_score(
    r: np.ndarray, delta: float, vocabulary_size: int
) -> np.ndarray:
    """Log likelihood ratio of the data-generating spike family at one Delta.

    This is a diagnostic competitor introduced by this manuscript; it is NOT
    one of the scores in Li et al. (2025).  The Bayes Gumbel detector averages
    the exact data-generating spike density

        f^sp_Delta(r) = r**(Delta/(1-Delta)) + (M-1) r**((M-1)/Delta - 1)

    over the Uniform(delta_low, delta_high) prior, whereas h*_gum,Delta as
    evaluated by Li et al. (2025) uses the least-favorable family.  This
    score keeps the data-generating family but freezes a single Delta and
    performs no Bayesian averaging, isolating the component-family choice.

    Because the Gumbel pivot is Unif(0,1) under the null, f0(r)=1 and the log
    likelihood ratio equals the log spike density.  Evaluated as a logaddexp
    of the two exponent terms so that both the r->0 and r->1 ends are stable.
    """

    if not isinstance(vocabulary_size, (int, np.integer)) or vocabulary_size < 2:
        raise ValueError("vocabulary_size must be an integer of at least 2")
    max_delta = 1.0 - 1.0 / int(vocabulary_size)
    if not (
        math.isfinite(delta) and 0.0 < delta <= max_delta + 1e-12
    ):
        raise ValueError(
            f"delta must lie in (0, {max_delta:g}] for this vocabulary"
        )
    log_r = np.log(r)
    top = (delta / (1.0 - delta)) * log_r
    tail = math.log(vocabulary_size - 1.0) + (
        (vocabulary_size - 1.0) / delta - 1.0
    ) * log_r
    return np.logaddexp(top, tail)


def paper_inverse_optimal_score(d: np.ndarray, delta: float) -> np.ndarray:
    """Numerical h*_dif,delta in the pinned upstream simulation snapshot.

    The snapshot at ``OFFICIAL_CODE_COMMIT`` evaluates
    ``log(max(1 - d/(1-delta), 1e-4)/(1-d))``.  Relative to the normalized
    limiting likelihood ratio, it omits ``1/(1-delta)`` and clips the
    triangular factor at ``1e-4``.  Reproducing that convention avoids
    negative infinity and matches the upstream fixed-horizon comparison.
    """

    numerator = np.maximum(1.0 - d / (1.0 - delta), 1e-4)
    denominator = np.maximum(1.0 - d, np.finfo(float).tiny)
    return np.log(numerator / denominator)


def gumbel_score(
    pivots: np.ndarray,
    method: str,
    bayes_lookup: GumbelBayesLookup,
    vocabulary_size: int | None = None,
    dirichlet_lookup: dirichlet.DirichletTokenwiseLookup | None = None,
    uniontail_lookup: dirichlet.DirichletTokenwiseLookup | None = None,
) -> np.ndarray:
    if method == "h_ars":
        return -np.log1p(-pivots)
    if method == "h_log":
        return np.log(pivots)
    if method == "h_ind_1_over_e":
        return (pivots >= math.exp(-1.0)).astype(float)
    if method == "h_gum_star_0.1":
        return paper_gumbel_optimal_score(pivots, 0.1)
    if method == "h_gum_star_0.01":
        return paper_gumbel_optimal_score(pivots, 0.01)
    if method == "h_gum_star_0.005":
        return paper_gumbel_optimal_score(pivots, 0.005)
    if method == "h_spike_0.01":
        return spike_point_mass_score(
            pivots, 0.01, vocabulary_size or bayes_lookup.vocabulary_size
        )
    if method == "h_spike_0.05":
        return spike_point_mass_score(
            pivots, 0.05, vocabulary_size or bayes_lookup.vocabulary_size
        )
    if method == "bayes_tokenwise":
        return bayes_lookup(pivots)  # f0(r)=1
    if method == DIRICHLET_TOKENWISE_METHOD:
        if dirichlet_lookup is None:
            raise ValueError(
                f"{DIRICHLET_TOKENWISE_METHOD} requires a dirichlet_lookup"
            )
        return dirichlet_lookup(pivots)  # f0(r)=1
    if method == UNIONTAIL_TOKENWISE_METHOD:
        if uniontail_lookup is None:
            raise ValueError(
                f"{UNIONTAIL_TOKENWISE_METHOD} requires a uniontail_lookup"
            )
        # The tail state is document level, so this path is a mixture of two
        # branch PRODUCTS and is not the running sum of any per-token quantity.
        # collect_paths cumsums whatever it is handed, so hand it the first
        # differences: cumsum then reconstructs the path exactly, and the
        # anytime running maximum is taken on the reconstructed path.
        paths = uniontail_lookup.paths(pivots)
        return np.diff(paths, axis=1, prepend=0.0)
    raise KeyError(method)


def inverse_score(pivots: np.ndarray, method: str, config: BenchmarkConfig) -> np.ndarray:
    if method == "h_neg":
        return -pivots
    if method == "h_dif_star_0.1":
        return paper_inverse_optimal_score(pivots, 0.1)
    if method == "h_dif_star_0.01":
        return paper_inverse_optimal_score(pivots, 0.01)
    if method == "h_dif_star_0.001":
        return paper_inverse_optimal_score(pivots, 0.001)
    if method == "bayes_tokenwise":
        return inverse_bayes_log_ratio(pivots, config)
    raise KeyError(method)


def simulate_gumbel_null(
    rng: np.random.Generator, rows: int, horizon: int, config: BenchmarkConfig
) -> np.ndarray:
    del config
    return rng.uniform(size=(rows, horizon))


def simulate_gumbel_alternative(
    rng: np.random.Generator, rows: int, horizon: int, config: BenchmarkConfig
) -> np.ndarray:
    deltas = rng.uniform(
        config.delta_low, config.delta_high, size=(rows, horizon)
    )
    selected_top = rng.uniform(size=(rows, horizon)) >= deltas
    selected_probability = np.where(
        selected_top,
        1.0 - deltas,
        deltas / (config.vocabulary_size - 1.0),
    )
    # Given selected-token probability p, R ~ Beta(1/p,1), i.e. U**p.
    return rng.uniform(size=(rows, horizon)) ** selected_probability


def simulate_gumbel_alternative_shared_delta(
    rng: np.random.Generator, rows: int, horizon: int, config: BenchmarkConfig
) -> np.ndarray:
    """Equal-tail sensitivity run with one Delta shared by each document."""

    deltas = np.repeat(
        rng.uniform(config.delta_low, config.delta_high, size=(rows, 1)),
        horizon,
        axis=1,
    )
    selected_top = rng.uniform(size=(rows, horizon)) >= deltas
    selected_probability = np.where(
        selected_top,
        1.0 - deltas,
        deltas / (config.vocabulary_size - 1.0),
    )
    return rng.uniform(size=(rows, horizon)) ** selected_probability


def simulate_inverse_null(
    rng: np.random.Generator, rows: int, horizon: int, config: BenchmarkConfig
) -> np.ndarray:
    uniforms = rng.uniform(size=(rows, horizon))
    ranks = rng.integers(0, config.vocabulary_size, size=(rows, horizon))
    return np.abs(uniforms - ranks / (config.vocabulary_size - 1.0))


def simulate_inverse_alternative(
    rng: np.random.Generator, rows: int, horizon: int, config: BenchmarkConfig
) -> np.ndarray:
    """Exact inverse-transform pivots for an equal-tail spike NTP."""

    v = config.vocabulary_size
    deltas = rng.uniform(config.delta_low, config.delta_high, size=(rows, horizon))
    top_probability = 1.0 - deltas
    tail_probability = deltas / (v - 1.0)
    top_rank = rng.integers(0, v, size=(rows, horizon))
    uniforms = rng.uniform(size=(rows, horizon))
    mass_before_top = tail_probability * top_rank

    before = uniforms < mass_before_top
    after = uniforms > mass_before_top + top_probability
    selected_rank = top_rank.copy()
    selected_rank[before] = np.floor(
        uniforms[before] / tail_probability[before]
    ).astype(int)
    selected_rank[after] = top_rank[after] + 1 + np.floor(
        (uniforms[after] - mass_before_top[after] - top_probability[after])
        / tail_probability[after]
    ).astype(int)
    np.clip(selected_rank, 0, v - 1, out=selected_rank)
    return np.abs(uniforms - selected_rank / (v - 1.0))


def simulate_inverse_alternative_shared_delta(
    rng: np.random.Generator, rows: int, horizon: int, config: BenchmarkConfig
) -> np.ndarray:
    """Exact equal-tail inverse watermark with one Delta per document."""

    v = config.vocabulary_size
    deltas = np.repeat(
        rng.uniform(config.delta_low, config.delta_high, size=(rows, 1)),
        horizon,
        axis=1,
    )
    top_probability = 1.0 - deltas
    tail_probability = deltas / (v - 1.0)
    top_rank = rng.integers(0, v, size=(rows, horizon))
    uniforms = rng.uniform(size=(rows, horizon))
    mass_before_top = tail_probability * top_rank
    before = uniforms < mass_before_top
    after = uniforms > mass_before_top + top_probability
    selected_rank = top_rank.copy()
    selected_rank[before] = np.floor(
        uniforms[before] / tail_probability[before]
    ).astype(int)
    selected_rank[after] = top_rank[after] + 1 + np.floor(
        (uniforms[after] - mass_before_top[after] - top_probability[after])
        / tail_probability[after]
    ).astype(int)
    np.clip(selected_rank, 0, v - 1, out=selected_rank)
    return np.abs(uniforms - selected_rank / (v - 1.0))


def collect_paths(
    *,
    n_rows: int,
    rng: np.random.Generator,
    simulator: Callable[[np.random.Generator, int, int, BenchmarkConfig], np.ndarray],
    methods: tuple[str, ...],
    scorer: Callable[[np.ndarray, str], np.ndarray],
    config: BenchmarkConfig,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return all-prefix score sums and, per anytime method, running maxima.

    The second return value is keyed by method name and contains an entry for
    every element of :data:`ANYTIME_METHODS` present in ``methods``.  Only
    Bayes-factor paths are tracked.  Exact anytime validity additionally
    requires analytic or exactly normalized predictive densities; interpolated
    paths are numerical implementations of that rule, as documented separately.
    """

    paths = {
        method: np.empty((n_rows, config.max_horizon), dtype=float)
        for method in methods
    }
    running_maxima = {
        method: np.empty((n_rows, config.max_horizon), dtype=float)
        for method in ANYTIME_METHODS
        if method in methods
    }
    for start in range(0, n_rows, config.batch_size):
        stop = min(start + config.batch_size, n_rows)
        pivots = simulator(rng, stop - start, config.max_horizon, config)
        for method in methods:
            values = scorer(pivots, method)
            np.cumsum(values, axis=1, out=values)
            paths[method][start:stop] = values
            if method in running_maxima:
                np.maximum.accumulate(values, axis=1, out=values)
                running_maxima[method][start:stop] = values
    return paths, running_maxima


def clone_rng(rng: np.random.Generator) -> np.random.Generator:
    """Clone a Generator before sampling so two methods see identical pivots."""

    clone = np.random.default_rng()
    clone.bit_generator.state = rng.bit_generator.state
    return clone


def _row_logsumexp(values: np.ndarray) -> np.ndarray:
    maximum = np.max(values, axis=1)
    output = np.full(values.shape[0], -np.inf)
    finite = np.isfinite(maximum)
    if np.any(finite):
        output[finite] = maximum[finite] + np.log(
            np.exp(values[finite] - maximum[finite, None]).sum(axis=1)
        )
    return output


def collect_shared_bayes_paths(
    *,
    n_rows: int,
    rng: np.random.Generator,
    simulator: Callable[[np.random.Generator, int, int, BenchmarkConfig], np.ndarray],
    scheme: str,
    config: BenchmarkConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Bayes factor for one latent Delta shared across the whole document.

    This is log integral prod_t f1(y_t | Delta) pi(Delta) dDelta, evaluated
    by Gauss-Legendre quadrature.  It is distinct from summing tokenwise
    marginal log likelihood ratios.
    """

    nodes, weights = np.polynomial.legendre.leggauss(config.bayes_quadrature_nodes)
    low, high = effective_prior_support(config)
    deltas = low + 0.5 * (nodes + 1.0) * (
        high - low
    )
    log_prior_weights = np.log(0.5 * weights)
    output = np.empty((n_rows, config.max_horizon), dtype=float)
    running_max = np.empty_like(output)

    if scheme == "gumbel":
        top_exponents = deltas / (1.0 - deltas)
        tail_exponents = (config.vocabulary_size - 1.0) / deltas - 1.0
        log_tail_count = math.log(config.vocabulary_size - 1.0)
    elif scheme == "inverse":
        inverse_scales = 1.0 - deltas
        inverse_constants = np.log(2.0 / inverse_scales)
    else:
        raise ValueError("scheme must be gumbel or inverse")

    for start in range(0, n_rows, config.batch_size):
        stop = min(start + config.batch_size, n_rows)
        pivots = simulator(rng, stop - start, config.max_horizon, config)
        component_log_likelihood_ratio = np.zeros(
            (stop - start, deltas.size), dtype=float
        )
        batch_path = np.empty((stop - start, config.max_horizon), dtype=float)
        for time_index in range(config.max_horizon):
            observation = pivots[:, time_index, None]
            if scheme == "gumbel":
                log_r = np.log(observation)
                component_log_ratio = np.logaddexp(
                    log_r * top_exponents[None, :],
                    log_tail_count + log_r * tail_exponents[None, :],
                )
            else:
                inside = 1.0 - observation / inverse_scales[None, :]
                with np.errstate(divide="ignore", invalid="ignore"):
                    component_log_density = np.where(
                        inside > 0.0,
                        inverse_constants[None, :] + np.log(inside),
                        -np.inf,
                    )
                log_f0 = np.log(
                    inverse_exact_null_density(
                        observation[:, 0], config.vocabulary_size
                    )
                )
                component_log_ratio = component_log_density - log_f0[:, None]
            component_log_likelihood_ratio += component_log_ratio
            batch_path[:, time_index] = _row_logsumexp(
                component_log_likelihood_ratio + log_prior_weights[None, :]
            )
        output[start:stop] = batch_path
        np.maximum.accumulate(batch_path, axis=1, out=batch_path)
        running_max[start:stop] = batch_path
    return output, running_max


def collect_dirichlet_shared_paths(
    *,
    n_rows: int,
    rng: np.random.Generator,
    simulator: Callable[[np.random.Generator, int, int, BenchmarkConfig], np.ndarray],
    grid: dirichlet.DirichletBayesGrid | dirichlet.UnionTailBayesGrid,
    config: BenchmarkConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared-latent log Bayes factor for a joint (Delta, tail) prior.

    Drives either the Dirichlet tail-shape grid or the tail-width grid; both
    expose the same shared-latent component protocol.

    The signature mirrors :func:`collect_shared_bayes_paths` so that the caller
    can drive both from clones of one generator and be sure the two detectors saw
    identical pivots.  Gumbel only: the layer is inert for the inverse pivot.
    """

    output = np.empty((n_rows, config.max_horizon), dtype=float)
    running_max = np.empty_like(output)
    for start in range(0, n_rows, config.batch_size):
        stop = min(start + config.batch_size, n_rows)
        pivots = simulator(rng, stop - start, config.max_horizon, config)
        values, maxima = grid.shared_paths(pivots, running_max=True)
        output[start:stop] = values
        running_max[start:stop] = maxima
    return output, running_max


def collect_trgof_paths(
    *,
    n_rows: int,
    rng: np.random.Generator,
    simulator: Callable[[np.random.Generator, int, int, BenchmarkConfig], np.ndarray],
    scheme: str,
    config: BenchmarkConfig,
    s: float = TRGOF_S,
) -> np.ndarray:
    """Tr-GoF of :mod:`trgof` evaluated at every prefix of every document.

    The statistic sorts the p-values of the whole prefix, so unlike every score
    in :func:`collect_paths` it cannot be accumulated with ``cumsum`` and is
    recomputed from scratch at each horizon.  The signature mirrors
    :func:`collect_shared_bayes_paths` so the caller can drive it from a clone of
    the scheme generator and be sure it saw the same pivots as every other rule.

    Only one array is returned: Tr-GoF is a fixed-horizon test statistic, not a
    Bayes factor, so there is no running maximum to report against ``1/alpha``.

    The Gumbel p-value ``1 - Y`` is the authors' own.  The inverse-transform
    p-value is this study's extension; see :func:`trgof.inverse_p_values`.
    """

    if scheme == "gumbel":
        to_p_values = trgof.gumbel_p_values
    elif scheme == "inverse":
        to_p_values = lambda pivots: trgof.inverse_p_values(
            pivots, config.vocabulary_size
        )
    else:
        raise ValueError("scheme must be gumbel or inverse")

    output = np.empty((n_rows, config.max_horizon), dtype=float)
    for start in range(0, n_rows, config.batch_size):
        stop = min(start + config.batch_size, n_rows)
        p_values = to_p_values(simulator(rng, stop - start, config.max_horizon, config))
        for time_index in range(config.max_horizon):
            output[start:stop, time_index] = trgof.statistic(
                p_values[:, : time_index + 1], s=s
            )
    return output


def calibrate_randomized_boundary(
    calibration_paths: dict[str, np.ndarray], alpha: float
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Calibrate score > c plus gamma * 1{score=c} at each horizon."""

    cutoffs: dict[str, np.ndarray] = {}
    boundary_probabilities: dict[str, np.ndarray] = {}
    for method, values in calibration_paths.items():
        if np.any(np.isnan(values)):
            raise ValueError(f"NaN calibration scores for {method}")
        n_rows, n_horizons = values.shape
        target = alpha * n_rows
        cutoff = np.empty(n_horizons)
        gamma = np.empty(n_horizons)
        order_index = int(math.floor((1.0 - alpha) * n_rows))
        order_index = min(max(order_index, 0), n_rows - 1)
        for column in range(n_horizons):
            vector = values[:, column]
            c = np.partition(vector, order_index)[order_index]
            n_greater = int(np.count_nonzero(vector > c))
            n_equal = int(np.count_nonzero(vector == c))
            cutoff[column] = c
            gamma[column] = np.clip((target - n_greater) / n_equal, 0.0, 1.0)
        cutoffs[method] = cutoff
        boundary_probabilities[method] = gamma
    return cutoffs, boundary_probabilities


def expected_rejection_rate(
    values: np.ndarray, cutoff: np.ndarray, boundary_probability: np.ndarray
) -> np.ndarray:
    if np.any(np.isnan(values)) or np.any(np.isnan(cutoff)):
        raise ValueError("NaN evaluation scores or cutoffs")
    greater = np.mean(values > cutoff[None, :], axis=0)
    equal = np.mean(values == cutoff[None, :], axis=0)
    return greater + boundary_probability * equal


def _boundary_randomization_seed_entropy(
    seed: int, scenario: str, scheme: str, horizon: int
) -> tuple[int, ...]:
    """Stable SeedSequence entropy for one paired evaluation cell."""

    if scenario not in _BOUNDARY_SCENARIO_CODES:
        raise ValueError(f"unknown benchmark scenario: {scenario}")
    if scheme not in _BOUNDARY_SCHEME_CODES:
        raise ValueError(f"unknown pivot scheme: {scheme}")
    if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(horizon, (int, np.integer)) or int(horizon) < 1:
        raise ValueError("horizon must be a positive integer")
    return (
        int(seed),
        BOUNDARY_RANDOMIZATION_SEED_SALT,
        _BOUNDARY_SCENARIO_CODES[scenario],
        _BOUNDARY_SCHEME_CODES[scheme],
        int(horizon),
    )


def boundary_randomization_uniforms(
    *, seed: int, scenario: str, scheme: str, horizon: int, n_documents: int
) -> np.ndarray:
    """Reproducible auxiliary uniforms for a randomized-boundary cell.

    The seed address deliberately omits the method.  Consequently every method
    evaluated on the same documents at one scenario/scheme/horizon reuses the
    same auxiliary uniform for document ``i``.  This common-random-number
    coupling makes McNemar discordances reflect score differences instead of
    avoidable Monte Carlo noise from independently randomizing two boundaries.
    """

    if not isinstance(n_documents, (int, np.integer)) or int(n_documents) < 0:
        raise ValueError("n_documents must be a non-negative integer")
    entropy = _boundary_randomization_seed_entropy(seed, scenario, scheme, horizon)
    return np.random.default_rng(np.random.SeedSequence(entropy)).random(
        int(n_documents)
    )


def randomized_boundary_rejections(
    values: np.ndarray,
    cutoff: float,
    boundary_probability: float,
    uniforms: np.ndarray,
) -> np.ndarray:
    """Realise ``score>c`` or ``score==c and U<gamma`` document by document."""

    scores = np.asarray(values, dtype=float)
    auxiliary = np.asarray(uniforms, dtype=float)
    if scores.ndim != 1 or auxiliary.shape != scores.shape:
        raise ValueError("values and uniforms must be one-dimensional with equal shape")
    gamma = float(boundary_probability)
    if not (math.isfinite(gamma) and 0.0 <= gamma <= 1.0):
        raise ValueError("boundary_probability must lie in [0, 1]")
    if np.any(~np.isfinite(auxiliary)) or np.any(
        (auxiliary < 0.0) | (auxiliary >= 1.0)
    ):
        raise ValueError("uniforms must contain finite values in [0, 1)")
    strict = scores > float(cutoff)
    at_boundary = scores == float(cutoff)
    return strict | (at_boundary & (auxiliary < gamma))


def _seeded_rngs(seed: int, count: int) -> list[np.random.Generator]:
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(count)]


def indicator_key(scenario: str, scheme: str, horizon: int, method: str) -> str:
    """Stable flat key for the persisted per-document rejection indicators."""

    return f"{scenario}|{scheme}|{horizon}|{method}"


def _record_alternative_indicators(
    *,
    scenario: str,
    scheme: str,
    method: str,
    values: np.ndarray,
    cutoff: np.ndarray,
    boundary_probability: np.ndarray,
    boundary_uniforms: dict[int, np.ndarray],
    randomization_seed: int,
    horizons: tuple[int, ...],
    sink: dict[str, np.ndarray] | None,
    tie_records: list[dict[str, object]],
) -> None:
    """Store a reproducible realization of the calibrated randomized test."""

    for horizon in horizons:
        index = horizon - 1
        column = values[:, index]
        threshold = float(cutoff[index])
        gamma = float(boundary_probability[index])
        uniforms = boundary_uniforms[horizon]
        strict = column > threshold
        at_boundary = column == threshold
        rejected = randomized_boundary_rejections(
            column, threshold, gamma, uniforms
        )
        boundary_rejected = at_boundary & rejected
        ties = int(np.count_nonzero(at_boundary))
        if sink is not None:
            sink[indicator_key(scenario, scheme, horizon, method)] = rejected
        tie_records.append(
            {
                "scenario": scenario,
                "scheme": scheme,
                "horizon": horizon,
                "method": method,
                "threshold": threshold,
                "n_documents": int(column.size),
                "n_at_atom": ties,
                "boundary_randomization_probability": gamma,
                "n_rejected_strict": int(np.count_nonzero(strict)),
                "n_rejected_at_atom": int(np.count_nonzero(boundary_rejected)),
                "n_rejected_randomized": int(np.count_nonzero(rejected)),
                "type2_error_randomized": float(1.0 - rejected.mean()),
                "auxiliary_seed_entropy": list(
                    _boundary_randomization_seed_entropy(
                        randomization_seed, scenario, scheme, horizon
                    )
                ),
            }
        )


def run_benchmark(
    config: BenchmarkConfig,
    indicator_sink: dict[str, np.ndarray] | None = None,
    dirichlet_alpha_grid: tuple[float, ...] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Run the clean benchmark.

    If ``indicator_sink`` is given it is filled in place with per-document
    boolean rejection indicators on the alternative evaluation sample, keyed by
    :func:`indicator_key`, at the horizons in ``INDICATOR_HORIZONS``.  The
    numerical results are unaffected by whether the sink is supplied.
    """

    config.validate()
    started = time.perf_counter()
    bayes_lookup = GumbelBayesLookup(config)
    # Gumbel only: the Dirichlet tail layer is inert for the inverse pivot,
    # whose limiting alternative depends on Delta alone.
    gumbel_dirichlet_grid = build_dirichlet_grid(config, dirichlet_alpha_grid)
    gumbel_uniontail_grid = build_uniontail_grid(config)
    dirichlet_lookup = gumbel_dirichlet_grid.tokenwise_lookup(
        logit_limit=config.gumbel_lookup_logit_limit, size=config.gumbel_lookup_size
    )
    uniontail_lookup = gumbel_uniontail_grid.tokenwise_lookup(
        logit_limit=config.gumbel_lookup_logit_limit, size=config.gumbel_lookup_size
    )
    rngs = _seeded_rngs(config.seed, 8)
    indicator_horizons = tuple(
        h for h in INDICATOR_HORIZONS if h <= config.max_horizon
    )
    tie_records: list[dict[str, object]] = []

    gumbel_scorer = lambda p, m: gumbel_score(
        p, m, bayes_lookup, dirichlet_lookup=dirichlet_lookup,
        uniontail_lookup=uniontail_lookup,
    )
    inverse_scorer = lambda p, m: inverse_score(p, m, config)

    all_rows: list[dict[str, object]] = []
    scheme_specs = (
        (
            "gumbel",
            GUMBEL_METHODS,
            gumbel_scorer,
            simulate_gumbel_null,
            (
                ("paper_text_iid_delta_equal_tail", simulate_gumbel_alternative),
                ("shared_delta_equal_tail_sensitivity", simulate_gumbel_alternative_shared_delta),
            ),
            rngs[:4],
            gumbel_dirichlet_grid,
            gumbel_uniontail_grid,
        ),
        (
            "inverse",
            INVERSE_METHODS,
            inverse_scorer,
            simulate_inverse_null,
            (
                ("paper_text_iid_delta_equal_tail", simulate_inverse_alternative),
                ("shared_delta_equal_tail_sensitivity", simulate_inverse_alternative_shared_delta),
            ),
            rngs[4:],
            None,
            None,
        ),
    )

    for (
        scheme,
        methods,
        scorer,
        null_sim,
        alternative_specs,
        scheme_rngs,
        dirichlet_grid,
        uniontail_grid,
    ) in scheme_specs:
        # Shared-latent rules are not token sums, so they live outside `methods`
        # and are collected separately.  Every collector is driven from a *clone*
        # of the scheme generator taken before that generator is advanced, so all
        # detectors in this scheme see bit-for-bit identical pivots and every
        # comparison below is paired.  Adding a method to `methods` consumes no
        # extra randomness, so the published spike-family numbers are unchanged.
        shared_specs: list[tuple[str, Callable[..., tuple[np.ndarray, np.ndarray]]]] = [
            (
                "bayes_shared",
                lambda *, n_rows, rng, simulator: collect_shared_bayes_paths(
                    n_rows=n_rows,
                    rng=rng,
                    simulator=simulator,
                    scheme=scheme,
                    config=config,
                ),
            )
        ]
        if dirichlet_grid is not None:
            shared_specs.append(
                (
                    DIRICHLET_SHARED_METHOD,
                    lambda *, n_rows, rng, simulator: collect_dirichlet_shared_paths(
                        n_rows=n_rows,
                        rng=rng,
                        simulator=simulator,
                        grid=dirichlet_grid,
                        config=config,
                    ),
                )
            )
        if uniontail_grid is not None:
            shared_specs.append(
                (
                    UNIONTAIL_SHARED_METHOD,
                    lambda *, n_rows, rng, simulator: collect_dirichlet_shared_paths(
                        n_rows=n_rows,
                        rng=rng,
                        simulator=simulator,
                        grid=uniontail_grid,
                        config=config,
                    ),
                )
            )

        # Whole-prefix competitors that are neither token sums nor Bayes factors.
        # Tr-GoF is reported in *both* alternative scenarios, unlike the
        # shared-latent Bayes rules, whose contrast is specific to the
        # shared-deficit design, so it is collected through its own list.
        prefix_specs: list[tuple[str, Callable[..., np.ndarray]]] = [
            (
                TRGOF_METHOD,
                lambda *, n_rows, rng, simulator: collect_trgof_paths(
                    n_rows=n_rows,
                    rng=rng,
                    simulator=simulator,
                    scheme=scheme,
                    config=config,
                ),
            )
        ]

        def collect_shared(
            source: np.random.Generator, n_rows: int, simulator
        ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
            """Run every shared-latent collector on the same pivot stream."""

            clones = [clone_rng(source) for _ in shared_specs]
            return {
                name: collector(n_rows=n_rows, rng=clone, simulator=simulator)
                for (name, collector), clone in zip(shared_specs, clones)
            }

        def collect_prefix(
            source: np.random.Generator, n_rows: int, simulator
        ) -> dict[str, np.ndarray]:
            """Run every whole-prefix collector on the same pivot stream.

            Each collector gets its own clone of ``source``, taken before that
            generator is advanced, so these rules see bit-for-bit the same
            pivots as the token-sum and shared-latent rules and draw no new
            randomness from any stream an existing rule uses.
            """

            clones = [clone_rng(source) for _ in prefix_specs]
            return {
                name: collector(n_rows=n_rows, rng=clone, simulator=simulator)
                for (name, collector), clone in zip(prefix_specs, clones)
            }

        shared_calibration = collect_shared(
            scheme_rngs[0], config.n_calibration, null_sim
        )
        prefix_calibration = collect_prefix(
            scheme_rngs[0], config.n_calibration, null_sim
        )
        calibration, _ = collect_paths(
            n_rows=config.n_calibration,
            rng=scheme_rngs[0],
            simulator=null_sim,
            methods=methods,
            scorer=scorer,
            config=config,
        )
        cutoffs, gammas = calibrate_randomized_boundary(calibration, config.alpha)
        shared_cutoffs, shared_gammas = calibrate_randomized_boundary(
            {name: paths for name, (paths, _) in shared_calibration.items()},
            config.alpha,
        )
        # Same common exact-null sample, same nominal level, same randomised
        # boundary.  Tr-GoF has an atom at zero -- when the truncated index set
        # is empty nothing survives and the score is zero -- so it is routed
        # through the shared calibration path rather than special-cased.
        prefix_cutoffs, prefix_gammas = calibrate_randomized_boundary(
            prefix_calibration, config.alpha
        )
        del calibration
        del shared_calibration
        del prefix_calibration

        shared_null = collect_shared(
            scheme_rngs[1], config.n_evaluation_null, null_sim
        )
        prefix_null = collect_prefix(
            scheme_rngs[1], config.n_evaluation_null, null_sim
        )
        null_paths, null_running_max = collect_paths(
            n_rows=config.n_evaluation_null,
            rng=scheme_rngs[1],
            simulator=null_sim,
            methods=methods,
            scorer=scorer,
            config=config,
        )
        for alt_index, (scenario, alt_sim) in enumerate(alternative_specs):
            shared_alternative: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            if scenario == "shared_delta_equal_tail_sensitivity":
                shared_alternative = collect_shared(
                    scheme_rngs[2 + alt_index],
                    config.n_evaluation_alternative,
                    alt_sim,
                )
            prefix_alternative = collect_prefix(
                scheme_rngs[2 + alt_index],
                config.n_evaluation_alternative,
                alt_sim,
            )
            alt_paths, alt_running_max = collect_paths(
                n_rows=config.n_evaluation_alternative,
                rng=scheme_rngs[2 + alt_index],
                simulator=alt_sim,
                methods=methods,
                scorer=scorer,
                config=config,
            )
            boundary_uniforms = {
                horizon: boundary_randomization_uniforms(
                    seed=config.seed,
                    scenario=scenario,
                    scheme=scheme,
                    horizon=horizon,
                    n_documents=config.n_evaluation_alternative,
                )
                for horizon in indicator_horizons
            }

            def record_fixed_horizon(
                method: str,
                null_values: np.ndarray,
                alt_values: np.ndarray,
                cutoff: np.ndarray,
                gamma: np.ndarray,
            ) -> None:
                """One fixed-horizon row per prefix, plus the paired indicators."""

                type1 = expected_rejection_rate(null_values, cutoff, gamma)
                power = expected_rejection_rate(alt_values, cutoff, gamma)
                _record_alternative_indicators(
                    scenario=scenario,
                    scheme=scheme,
                    method=method,
                    values=alt_values,
                    cutoff=cutoff,
                    boundary_probability=gamma,
                    boundary_uniforms=boundary_uniforms,
                    randomization_seed=config.seed,
                    horizons=indicator_horizons,
                    sink=indicator_sink,
                    tie_records=tie_records,
                )
                for index, horizon in enumerate(range(1, config.max_horizon + 1)):
                    type2 = 1.0 - power[index]
                    all_rows.append(
                        {
                            "scenario": scenario,
                            "scheme": scheme,
                            "method": method,
                            "decision_rule": "fixed_horizon_mc_calibrated",
                            "horizon": horizon,
                            "alpha": config.alpha,
                            "threshold": float(cutoff[index]),
                            "boundary_randomization": float(gamma[index]),
                            "type1_error": float(type1[index]),
                            "power": float(power[index]),
                            "type2_error": float(type2),
                            "type1_mc_se": float(
                                math.sqrt(
                                    type1[index]
                                    * (1.0 - type1[index])
                                    / config.n_evaluation_null
                                )
                            ),
                            "type2_mc_se": float(
                                math.sqrt(
                                    type2 * (1.0 - type2) / config.n_evaluation_alternative
                                )
                            ),
                        }
                    )

            def record_anytime(
                method: str, null_maxima: np.ndarray, alt_maxima: np.ndarray
            ) -> None:
                """One anytime row per prefix, at the theoretical BF threshold.

                The anytime rule uses log(1/alpha), not a Monte Carlo cutoff; the
                same Bayes likelihood drives it and the fixed-horizon rule.
                """

                threshold = -math.log(config.alpha)
                type1 = np.mean(null_maxima >= threshold, axis=0)
                power = np.mean(alt_maxima >= threshold, axis=0)
                for index, horizon in enumerate(range(1, config.max_horizon + 1)):
                    type2 = 1.0 - power[index]
                    all_rows.append(
                        {
                            "scenario": scenario,
                            "scheme": scheme,
                            "method": method,
                            "decision_rule": "anytime_bf_ge_1_over_alpha",
                            "horizon": horizon,
                            "alpha": config.alpha,
                            "threshold": threshold,
                            "boundary_randomization": 0.0,
                            "type1_error": float(type1[index]),
                            "power": float(power[index]),
                            "type2_error": float(type2),
                            "type1_mc_se": float(
                                math.sqrt(
                                    type1[index]
                                    * (1.0 - type1[index])
                                    / config.n_evaluation_null
                                )
                            ),
                            "type2_mc_se": float(
                                math.sqrt(
                                    type2 * (1.0 - type2) / config.n_evaluation_alternative
                                )
                            ),
                        }
                    )

            for method in methods:
                record_fixed_horizon(
                    method,
                    null_paths[method],
                    alt_paths[method],
                    cutoffs[method],
                    gammas[method],
                )
            for method, (alt_values, alt_maxima) in shared_alternative.items():
                record_fixed_horizon(
                    method,
                    shared_null[method][0],
                    alt_values,
                    shared_cutoffs[method],
                    shared_gammas[method],
                )
            for method, alt_values in prefix_alternative.items():
                record_fixed_horizon(
                    method,
                    prefix_null[method],
                    alt_values,
                    prefix_cutoffs[method],
                    prefix_gammas[method],
                )
            for method, maxima in alt_running_max.items():
                record_anytime(method, null_running_max[method], maxima)
            for method, (_, alt_maxima) in shared_alternative.items():
                record_anytime(method, shared_null[method][1], alt_maxima)
            # No anytime row for the whole-prefix competitors: Tr-GoF is a
            # fixed-horizon statistic, not an e-process, so log(1/alpha) is not
            # a valid threshold for it.

            del alt_paths, alt_running_max, shared_alternative, prefix_alternative
    dirichlet_outer_validation = dirichlet_lookup.validation_report()
    # The union's tokenwise path has its own two-branch table and was not
    # being recorded, so the supplementary checks covered every interpolated
    # rule except this one.
    uniontail_outer_validation = uniontail_lookup.validation_report()
    elapsed = time.perf_counter() - started
    metadata: dict[str, object] = {
        "paper": PAPER_URL,
        "official_code": OFFICIAL_CODE_URL,
        "official_code_commit": OFFICIAL_CODE_COMMIT,
        "official_result_arrays": list(OFFICIAL_RESULT_ARRAYS),
        "official_code_note": (
            "Commit-pinned upstream snapshot used only to document the simulator "
            "and reference-score conventions attributed to Li et al. (2025). "
            "This repository's provenance.json separately identifies the code "
            "that produced the Bayesian benchmark artifacts."
        ),
        "config": asdict(config),
        "runtime_seconds": elapsed,
        "primary_scenario": {
            "name": "paper_text_iid_delta_equal_tail",
            "delta": "Delta_t iid Uniform(0.001, 0.5) at every token",
            "ntp": "(1-Delta_t, Delta_t/(V-1), ..., Delta_t/(V-1))",
            "gumbel_H1": "exact selected-token Beta(1/P_w,1) pivot law",
            "inverse_H1": "exact finite-V inverse-CDF watermark for equal-tail spike",
        },
        "shared_delta_sensitivity": {
            "name": "shared_delta_equal_tail_sensitivity",
            "delta": "one Delta ~ Uniform(0.001,0.5) shared by the document",
            "ntp": "equal-tail spike at every token",
            "purpose": (
                "Isolates the commit-pinned simulation snapshot's document-shared "
                "deficit while "
                "retaining the equal-tail spike stated by Li et al. (2025)."
            ),
        },
        "calibration": (
            "Independent exact-null Monte Carlo. Fixed-horizon rule is score>c plus "
            "a randomized score=c boundary chosen to target alpha."
        ),
        "anytime_rule": "max_{s<=t} log BF_s >= log(1/alpha); no horizon calibration",
        "bayes_gumbel": (
            "Exact spike-family f1(r|Delta), integrated over the data-generating "
            "Uniform(0.001,0.5) prior; tokenwise independent Delta."
        ),
        "bayes_gumbel_lookup_validation": bayes_lookup.validation_report(),
        "bayes_inverse": (
            "Large-V triangular f1(d|Delta) of Li et al. (2025), analytically integrated over "
            "Uniform(0.001,0.5), divided by the exact finite-V null density."
        ),
        "bayes_shared": (
            "For the shared-Delta sensitivity scenario, integrates the joint "
            "document likelihood integral prod_t f1(y_t|Delta) pi(Delta)dDelta."
        ),
        "bayes_dirichlet": {
            "methods": [DIRICHLET_SHARED_METHOD, DIRICHLET_TOKENWISE_METHOD],
            "scheme": "gumbel only",
            "why_gumbel_only": (
                "The inverse limiting alternative depends on Delta alone, so a "
                "prior on the tail shape is inert for it by construction."
            ),
            "component": (
                "f_{Delta,alpha}(r) = r**(Delta/(1-Delta)) + K E_q[r**(1/(Delta q)-1)] "
                "with q ~ Beta(alpha, (K-1)alpha) the tail marginal of a "
                "Dirichlet(alpha,...,alpha) tail on K = V-1 coordinates."
            ),
            "alpha_prior": {
                "family": "uniform on a fixed grid",
                "grid": [dirichlet.format_alpha(a) for a in gumbel_dirichlet_grid.alphas],
                "note": (
                    "inf is the equal-tail spike limit, so the existing "
                    "bayes_shared rule is one component of this mixture."
                ),
            },
            "prior_fingerprint": gumbel_dirichlet_grid.prior_fingerprint(),
            "outer_tokenwise_lookup_validation": dirichlet_outer_validation,
            "containment_check": {
                "alpha_inf_vs_closed_form_spike_max_abs_log_density_difference": (
                    gumbel_dirichlet_grid.spike_agreement(
                        np.random.default_rng(config.seed).uniform(size=20_000)
                    )
                ),
            },
            "normalisation": {
                "diagnostic_only": True,
                "certified_one_sided_bound": False,
                "analytic_component_mass_max_abs_deviation_by_alpha": (
                    gumbel_dirichlet_grid.analytic_normalisation()
                ),
                "tokenwise_numerator_mass_minus_one": (
                    dirichlet_outer_validation["interpolated_numerator_mass"][
                        "mass_minus_one"
                    ]
                ),
                "note": (
                    "Components are exactly normalised analytically. The reported "
                    "tokenwise mass includes both the inner component transform and "
                    "the outer prior-mixture lookup. It is a numerical quadrature "
                    "diagnostic for the configured tables, not a certificate that "
                    "interpolation error is one-sided for every pivot or configuration."
                ),
            },
        },
        "bayes_uniontail": {
            **union_tail_metadata(gumbel_uniontail_grid, methods=[UNIONTAIL_SHARED_METHOD, UNIONTAIL_TOKENWISE_METHOD], tokenwise=True),
            'outer_tokenwise_lookup_validation': uniontail_outer_validation,
            'containment_check': {
                "shape_equal_tail_atom_vs_closed_form_spike_max_abs_log_density_difference": (
                    gumbel_uniontail_grid.spike_agreement(
                        np.random.default_rng(config.seed).uniform(size=20_000)
                    )
                ),
            },
        },
        "trgof": {
            "methods": [TRGOF_METHOD],
            "scheme": "gumbel and inverse",
            "citation": "Li et al. (2026; preprint 2024), li2024robust",
            "citation_title": (
                "Robust Detection of Watermarks for Large Language Models "
                "Under Human Edits"
            ),
            "reference_implementation": "github.com/lx10077/TrGoF, compute_score",
            "statistic": (
                "S_n(s) = n * max_t K_s(t/n, p_(t)) over the sorted prefix "
                "p-values p_(1) <= ... <= p_(n), with K_s the Cressie-Read "
                "phi-divergence between the empirical and null occupancy of "
                "each order statistic."
            ),
            "s": float(TRGOF_S),
            "s_note": (
                "s = 2 is Higher Criticism, the index the authors' figures "
                "lead with."
            ),
            "s_sensitivity_grid": [float(s) for s in trgof.DEFAULT_S_VALUES],
            "s_sensitivity_not_evaluated": [
                float(s) for s in TRGOF_S_SENSITIVITY
            ],
            "s_sensitivity_note": (
                "The authors' simulation code sweeps "
                f"{[float(s) for s in trgof.DEFAULT_S_VALUES]}. Only s = "
                f"{float(TRGOF_S)} is evaluated as a rule here, to keep one "
                "Tr-GoF row per scheme in tables that are already large; the "
                "remaining indices are recorded as this sensitivity grid and "
                "were not run."
            ),
            "truncation": (
                "The maximum is taken over the truncated, one-sided index set "
                "p_(t) >= 1/n and t/n >= p_(t): the first condition is the "
                "truncation the method is named for, the second is the "
                "one-sided restriction K_s^+, so only an excess of small "
                "p-values counts as evidence. Rows whose truncated index set "
                "is empty score zero, which is the value the reference "
                "implementation's maximum degenerates to."
            ),
            "atom_at_zero": (
                "The empty truncated index set gives the statistic an atom at "
                "zero, most of it at short prefixes. It is calibrated on the "
                "same exact-null sample, at the same nominal alpha, with the "
                "same randomized score == c boundary as every other rule; no "
                "special case."
            ),
            "not_a_token_sum": (
                "The statistic sorts the p-values of the whole prefix, so it "
                "is not additive over tokens and cannot be accumulated with a "
                "cumulative sum. It is recomputed from scratch at every "
                "reported prefix, like the shared-latent Bayes rules, and is "
                "therefore not a member of GUMBEL_METHODS or INVERSE_METHODS."
            ),
            "no_anytime_row": (
                "Tr-GoF is a fixed-horizon statistic, not an e-process, so it "
                "is reported only under the fixed-horizon calibrated rule."
            ),
            "gumbel_p_value": {
                "definition": "p = 1 - Y, exactly uniform under the null",
                "attribution": (
                    "The authors'. Their released implementation applies "
                    "Tr-GoF to the Gumbel-max pivot only."
                ),
            },
            "inverse_p_value": {
                "definition": (
                    "p = F_0(d), the exact finite-vocabulary null CDF of the "
                    "inverse-transform pivot |U - eta(I)|; small d is the "
                    "evidence direction, so the one-sided p-value is the CDF "
                    "rather than its complement."
                ),
                "attribution": (
                    "OURS, not the authors'. Li et al. (2024) define no "
                    "p-value for the inverse-transform pivot and their "
                    "released code does not apply Tr-GoF to it. This "
                    "extension is labelled as ours wherever it is reported."
                ),
                "exact_null_vocabulary_size": int(config.vocabulary_size),
                "exact_null_note": (
                    "The null is the exact finite-M law at M = "
                    f"{int(config.vocabulary_size)}, whose density is "
                    "piecewise constant, so the p-values are uniform at finite "
                    "vocabulary rather than only asymptotically."
                ),
            },
            "accuracy_eps": float(trgof.ACCURACY_EPS),
            "implementation": (
                "code/trgof.py, a term-by-term reimplementation of the "
                "authors' released compute_score, kept as the single owner of "
                "the statistic so every caller in this study shares it."
            ),
        },
        "paper_score_parameters": {
            "gumbel_optimal_delta": [0.1, 0.01, 0.005],
            "inverse_optimal_delta": [0.1, 0.01, 0.001],
            "indicator_threshold": "1/e",
            "inverse_numerical_clip": 1e-4,
        },
        "diagnostic_scores_not_from_the_paper": {
            "methods": list(DIAGNOSTIC_METHODS),
            "definition": (
                "log f^sp_Delta(r) with f^sp_Delta(r) = r**(Delta/(1-Delta)) + "
                "(V-1) r**((V-1)/Delta - 1), the exact data-generating spike "
                "density at a single fixed Delta; f0(r)=1 so this is the log "
                "likelihood ratio."
            ),
            "delta": [0.01, 0.05],
            "purpose": (
                "This diagnostic is not among the scores evaluated by Li et al. (2025). "
                "It uses the data-generating spike family with one fixed Delta and no "
                "Bayesian averaging, isolating the contribution of component-family "
                "choice relative to averaging over Delta. The "
                "h*_gum,Delta as evaluated by Li et al. (2025) instead uses the "
                "least-favorable family."
            ),
        },
        "per_document_indicators": {
            "indicator_rule_version": BOUNDARY_RANDOMIZATION_RULE_VERSION,
            "horizons": list(indicator_horizons),
            "sample": "alternative evaluation sample",
            "rule": (
                "score > c or (score == c and U_i < gamma) at each method's own "
                "calibrated fixed-horizon cutoff c and boundary probability gamma"
            ),
            "boundary_randomization": {
                "base_seed": int(config.seed),
                "seed_salt": BOUNDARY_RANDOMIZATION_SEED_SALT,
                "seed_entropy_order": [
                    "base_seed",
                    "seed_salt",
                    "scenario_code",
                    "scheme_code",
                    "horizon",
                ],
                "scenario_codes": dict(_BOUNDARY_SCENARIO_CODES),
                "scheme_codes": dict(_BOUNDARY_SCHEME_CODES),
                "coupling": (
                    "The same auxiliary U_i is reused for document i by every "
                    "method in a scenario/scheme/horizon cell; cells have separate "
                    "addressable SeedSequence streams independent of pivot RNGs."
                ),
                "aggregate_rate_note": (
                    "CSV power is the conditional expectation greater + gamma*equal; "
                    "the persisted indicators are one reproducible realization of "
                    "that reported randomized-boundary procedure."
                ),
            },
            "tie_and_randomized_rejection_counts": tie_records,
            "total_documents_at_atom": int(
                sum(int(record["n_at_atom"]) for record in tie_records)
            ),
            "total_rejections_randomized_at_atom": int(
                sum(int(record["n_rejected_at_atom"]) for record in tie_records)
            ),
        },
        "important_reproduction_note": (
            "The written specification of Li et al. (2025) makes Delta_t iid across "
            "tokens with an equal tail. The commit-pinned accompanying simulation "
            "snapshot instead samples one Delta per document and redraws "
            "random normalized-uniform tail probabilities at each token. This benchmark "
            "reports the written specification and a shared-Delta, equal-tail sensitivity "
            "that isolates the consequential dependence change."
        ),
    }
    return all_rows, metadata


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _selected_summary(rows: list[dict[str, object]], horizons: tuple[int, ...]) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if int(row["horizon"]) in horizons
        and (
            row["decision_rule"] == "anytime_bf_ge_1_over_alpha"
            or row["method"]
            in {
                "h_ars",
                "h_gum_star_0.005",
                "h_spike_0.01",
                "h_spike_0.05",
                "h_neg",
                "h_dif_star_0.001",
                "bayes_tokenwise",
                "bayes_shared",
                DIRICHLET_TOKENWISE_METHOD,
                DIRICHLET_SHARED_METHOD,
                UNIONTAIL_SHARED_METHOD,
                TRGOF_METHOD,
            }
        )
    ]


def write_json(rows: list[dict[str, object]], metadata: dict[str, object], path: Path) -> None:
    horizons = tuple(h for h in (50, 100, 200, 300, 500, 700) if h <= metadata["config"]["max_horizon"])
    payload = dict(metadata)
    payload["selected_results"] = _selected_summary(rows, horizons)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def write_selected_csv(
    rows: list[dict[str, object]], max_horizon: int, path: Path
) -> None:
    horizons = tuple(h for h in (50, 100, 200, 300, 500, 700) if h <= max_horizon)
    write_csv(_selected_summary(rows, horizons), path)


def make_plot(rows: list[dict[str, object]], output_stem: Path, scenario: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display = {
        "h_ars": r"$h_{ars}$",
        "h_log": r"$h_{log}$",
        "h_ind_1_over_e": r"$h_{ind,1/e}$",
        "h_gum_star_0.1": r"$h^*_{gum,.1}$",
        "h_gum_star_0.01": r"$h^*_{gum,.01}$",
        "h_gum_star_0.005": r"$h^*_{gum,.005}$",
        # Diagnostics introduced here, not scores from Li et al. (2025).
        "h_spike_0.01": r"$h^{spike}_{.01}$ (diagnostic; fixed $\Delta$)",
        "h_spike_0.05": r"$h^{spike}_{.05}$ (diagnostic; fixed $\Delta$)",
        "h_neg": r"$h_{neg}$",
        "h_dif_star_0.1": r"$h^*_{dif,.1}$",
        "h_dif_star_0.01": r"$h^*_{dif,.01}$",
        "h_dif_star_0.001": r"$h^*_{dif,.001}$",
        "bayes_tokenwise": "Bayes mixture",
        "bayes_shared": "Bayes shared-Delta",
        "bayes_tokenwise_dirichlet": "Bayes mixture + Dirichlet tail",
        "bayes_tokenwise_uniontail": "Bayes mixture + union tail",
        "bayes_shared_dirichlet": "Bayes shared-Delta + Dirichlet tail",
        "bayes_shared_uniontail": "Bayes shared-Delta + union tail",
        # Inverse-scheme p-value is ours; the Gumbel one is the authors'.
        TRGOF_METHOD: r"Tr-GoF ($s=2$)",
    }
    colors = {
        "h_ars": "#1f77b4",
        "h_log": "#ff7f0e",
        "h_ind_1_over_e": "#7f7f7f",
        "h_gum_star_0.1": "#8c6d1f",
        "h_gum_star_0.01": "#111111",
        "h_gum_star_0.005": "#d62728",
        "h_spike_0.01": "#8c564b",
        "h_spike_0.05": "#2ca02c",
        "h_neg": "#1f77b4",
        "h_dif_star_0.1": "#ff7f0e",
        "h_dif_star_0.01": "#111111",
        "h_dif_star_0.001": "#d62728",
        "bayes_tokenwise": "#6f2dbd",
        "bayes_shared": "#00876c",
        "bayes_tokenwise_dirichlet": "#c2185b",
        "bayes_tokenwise_uniontail": "#7b5d10",
        "bayes_shared_dirichlet": "#0b6fa4",
        "bayes_shared_uniontail": "#8c6d1f",
        TRGOF_METHOD: "#6b4c9a",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.5), sharex=True)
    for row_index, scheme in enumerate(("gumbel", "inverse")):
        scheme_rows = [
            row
            for row in rows
            if row["scheme"] == scheme
            and row["scenario"] == scenario
            and row["decision_rule"] == "fixed_horizon_mc_calibrated"
        ]
        methods = list(GUMBEL_METHODS if scheme == "gumbel" else INVERSE_METHODS)
        # Not a token sum, so it is not in those tuples, but it is reported at
        # every prefix in both scenarios and belongs on both panels.
        methods.append(TRGOF_METHOD)
        if scenario == "shared_delta_equal_tail_sensitivity":
            methods.append("bayes_shared")
            if scheme == "gumbel":
                methods.append(DIRICHLET_SHARED_METHOD)
                methods.append(UNIONTAIL_SHARED_METHOD)
        for method in methods:
            subset = [row for row in scheme_rows if row["method"] == method]
            if not subset:
                continue
            horizon = np.asarray([row["horizon"] for row in subset])
            type1 = np.asarray([row["type1_error"] for row in subset])
            type2 = np.asarray([row["type2_error"] for row in subset])
            width = 2.6 if method.startswith("bayes_") else 1.35
            # The layer curves sit almost on top of their spike counterparts
            # under an equal-tail generator, so they must be dashed or the
            # earlier curve is silently hidden underneath.
            style = "--" if method.endswith("_dirichlet") else "-"
            axes[row_index, 0].plot(
                horizon,
                type1,
                label=display[method],
                color=colors[method],
                lw=width,
                ls=style,
            )
            axes[row_index, 1].plot(
                horizon,
                np.maximum(type2, 1e-4),
                label=display[method],
                color=colors[method],
                lw=width,
                ls=style,
            )

        # Every anytime rule that produced rows, so no computed rule is silently
        # absent from the figure.  Layer rules are dotted to separate them from
        # their spike counterparts, which they otherwise sit on top of.
        anytime_methods = ["bayes_tokenwise", DIRICHLET_TOKENWISE_METHOD]
        if scenario == "shared_delta_equal_tail_sensitivity":
            anytime_methods += ["bayes_shared", DIRICHLET_SHARED_METHOD]
        anytime_labels = {
            "bayes_tokenwise": "Bayes tokenwise anytime BF>=20",
            DIRICHLET_TOKENWISE_METHOD: "Bayes tokenwise + Dirichlet anytime",
            "bayes_shared": "Bayes shared anytime",
            DIRICHLET_SHARED_METHOD: "Bayes shared + Dirichlet anytime",
        }
        anytime_colors = {
            "bayes_tokenwise": "#c51b8a",
            DIRICHLET_TOKENWISE_METHOD: "#c2185b",
            "bayes_shared": "#00876c",
            DIRICHLET_SHARED_METHOD: "#0b6fa4",
        }
        for method in anytime_methods:
            anytime = [
                row
                for row in rows
                if row["scheme"] == scheme
                and row["scenario"] == scenario
                and row["decision_rule"] == "anytime_bf_ge_1_over_alpha"
                and row["method"] == method
            ]
            if not anytime:
                continue
            horizon = np.asarray([row["horizon"] for row in anytime])
            style = ":" if method.endswith("_dirichlet") else "--"
            axes[row_index, 0].plot(
                horizon,
                [row["type1_error"] for row in anytime],
                color=anytime_colors[method],
                lw=1.8,
                ls=style,
                label=anytime_labels[method],
            )
            axes[row_index, 1].plot(
                horizon,
                np.maximum([row["type2_error"] for row in anytime], 1e-4),
                color=anytime_colors[method],
                lw=1.8,
                ls=style,
                label=anytime_labels[method],
            )
        axes[row_index, 0].axhline(0.05, color="black", ls=":", lw=1.0)
        axes[row_index, 0].set_ylim(0.0, 0.09)
        axes[row_index, 1].set_yscale("log")
        axes[row_index, 1].set_ylim(5e-4, 1.05)
        axes[row_index, 0].set_ylabel(f"{scheme.title()} Type I error")
        axes[row_index, 1].set_ylabel(f"{scheme.title()} Type II error")
        axes[row_index, 1].legend(fontsize=8.5, ncol=2, loc="upper right")

    axes[1, 0].set_xlabel("Text length")
    axes[1, 1].set_xlabel("Text length")
    axes[0, 0].set_title("Independent fixed-horizon 5% calibration")
    alternative_title = (
        r"$H_1$: $\Delta_t \sim U(.001,.5)$ independently"
        if scenario == "paper_text_iid_delta_equal_tail"
        else r"$H_1$: one $\Delta \sim U(.001,.5)$ per document"
    )
    axes[0, 1].set_title(alternative_title)
    for axis in axes.flat:
        axis.grid(alpha=0.18)
        axis.set_xlim(1, max(int(row["horizon"]) for row in rows))
    scenario_title = (
        "tokenwise-deficit equal-tail design"
        if scenario == "paper_text_iid_delta_equal_tail"
        else "shared-deficit equal-tail design"
    )
    fig.suptitle(
        f"Bayesian detectors and reference scores evaluated by Li et al. (2025): {scenario_title}",
        fontsize=15,
    )
    fig.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _bayes_linestyle(method: str, scenario: str):
    """Family/member style for the compact comparison figures.

    Reference-score hue identifies a family and dash identifies the member.
    For Bayes curves, hierarchy sets solid versus dashed in the shared-deficit
    scenario; the Dirichlet layer remains dotted or long-dashed so it cannot
    silently hide a nearly coincident equal-tail curve.
    """

    reference_styles = {
        "h_ars": "-",
        "h_log": (0, (2.5, 1.5)),
        "h_ind_1_over_e": (0, (1, 1.4)),
        "h_gum_star_0.1": "-",
        "h_gum_star_0.01": (0, (3.5, 1.2)),
        "h_gum_star_0.005": (0, (1.2, 1.2)),
        "h_spike_0.01": (0, (5, 1.5)),
        "h_spike_0.05": (0, (1, 1.4)),
        "h_neg": (0, (2.5, 1.5)),
        "h_dif_star_0.1": "-",
        "h_dif_star_0.01": (0, (3.5, 1.2)),
        "h_dif_star_0.001": (0, (1.2, 1.2)),
        TRGOF_METHOD: (0, (6, 1.5, 1, 1.5)),
    }
    if method in reference_styles:
        return reference_styles[method]

    tokenwise = method in {
        "bayes_tokenwise", DIRICHLET_TOKENWISE_METHOD, UNIONTAIL_TOKENWISE_METHOD
    }
    layer = method.endswith("_dirichlet")
    shared_scenario = scenario == "shared_delta_equal_tail_sensitivity"
    if layer:
        return ":" if (tokenwise and shared_scenario) else (0, (4, 1.6))
    if tokenwise and shared_scenario:
        return "--"
    return "-"


def _type2_plot_values(type2: np.ndarray) -> tuple[np.ma.MaskedArray, np.ndarray]:
    """Keep observed rates on curves and separate zero-count upper bounds.

    Replacing a zero observation by ``3/N`` inside a line can manufacture an
    upturn when adjacent positive observations are below that bound.  Masking
    zeros breaks the point-estimate line; callers draw their upper-bound
    markers separately.
    """

    values = np.asarray(type2, dtype=float)
    if np.any(values < 0.0):
        raise ValueError("Type II error rates must be nonnegative")
    zero_count = values == 0.0
    return np.ma.masked_where(zero_count, values), zero_count


def _selected_zero_bound_markers(
    horizon: np.ndarray,
    zero_count: np.ndarray,
    *,
    min_spacing: float,
) -> np.ndarray:
    """Thin bound markers while retaining each zero-count run's endpoints."""

    x = np.asarray(horizon, dtype=float)
    zeros = np.asarray(zero_count, dtype=bool)
    if x.shape != zeros.shape:
        raise ValueError("horizon and zero-count masks must have the same shape")
    if min_spacing <= 0.0:
        raise ValueError("minimum marker spacing must be positive")

    selected = np.zeros_like(zeros)
    zero_indices = np.flatnonzero(zeros)
    if zero_indices.size == 0:
        return selected

    breaks = np.flatnonzero(
        (np.diff(zero_indices) > 1)
        | (np.diff(x[zero_indices]) > 1.0 + 1e-12)
    )
    starts = np.r_[0, breaks + 1]
    stops = np.r_[breaks + 1, zero_indices.size]
    for start, stop in zip(starts, stops):
        run = zero_indices[start:stop]
        selected[run[0]] = True
        last_selected = run[0]
        for index in run[1:-1]:
            if x[index] - x[last_selected] >= min_spacing:
                selected[index] = True
                last_selected = index
        selected[run[-1]] = True
    return selected


def make_slide_plot(
    rows: list[dict[str, object]],
    output_stem: Path,
    scenario: str,
    config: BenchmarkConfig,
    schemes: tuple[str, ...] = ("gumbel", "inverse"),
    exclude_methods: tuple[str, ...] = (),
) -> None:
    """Write a compact fixed-horizon Type II comparison for the lecture deck.

    ``schemes`` selects which pivot panels to draw and ``exclude_methods`` drops
    named rules.  Both default to the full figure; they exist so a
    Gumbel-only variant of the manuscript can reuse this code unchanged rather
    than fork it.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "h_ars": r"$h_{\mathrm{ars}}$",
        "h_log": r"$h_{\mathrm{log}}$",
        "h_ind_1_over_e": r"$h_{\mathrm{ind},1/e}$",
        "h_gum_star_0.1": r"$h^\star_{\mathrm{gum},.1}$",
        "h_gum_star_0.01": r"$h^\star_{\mathrm{gum},.01}$",
        "h_gum_star_0.005": r"$h^\star_{\mathrm{gum},.005}$",
        "h_spike_0.01": r"$h^{\mathrm{sp}}_{.01}$ (equal tail)",
        "h_spike_0.05": r"$h^{\mathrm{sp}}_{.05}$ (equal tail)",
        "h_neg": r"$h_{\mathrm{neg}}$",
        "h_dif_star_0.1": r"$h^*_{\mathrm{dif},.1}$",
        "h_dif_star_0.01": r"$h^*_{\mathrm{dif},.01}$",
        "h_dif_star_0.001": r"$h^*_{\mathrm{dif},.001}$",
        "bayes_tokenwise": r"Bayes, tokenwise $\Delta$",
        "bayes_shared": r"Bayes, shared $\Delta$",
        "bayes_tokenwise_dirichlet": r"Bayes, tokenwise $\Delta$ + tail shape",
        "bayes_tokenwise_uniontail": r"Bayes, tokenwise $\Delta$ + union tail",
        "bayes_shared_dirichlet": r"Bayes, shared $\Delta$ + tail shape",
        "bayes_shared_uniontail": r"Bayes, shared $\Delta$ + union tail",
        TRGOF_METHOD: r"Tr-GoF ($s=2$)",
    }
    colors = {
        # Match the family encoding used by the temperature-matched figures:
        # hue identifies the family and dash identifies the member/hierarchy.
        "h_ars": "#D55E00",
        "h_log": "#D55E00",
        "h_ind_1_over_e": "#D55E00",
        "h_gum_star_0.1": "#009E73",
        "h_gum_star_0.01": "#009E73",
        "h_gum_star_0.005": "#009E73",
        "h_spike_0.01": "#7F7F7F",
        "h_spike_0.05": "#7F7F7F",
        "h_neg": "#D55E00",
        "h_dif_star_0.1": "#009E73",
        "h_dif_star_0.01": "#009E73",
        "h_dif_star_0.001": "#009E73",
        "bayes_tokenwise": "#542788",
        "bayes_shared": "#542788",
        "bayes_tokenwise_dirichlet": "#9B2226",
        "bayes_tokenwise_uniontail": "#0B6FA4",
        "bayes_shared_dirichlet": "#9B2226",
        "bayes_shared_uniontail": "#0B6FA4",
        TRGOF_METHOD: "#6B4C9A",
    }
    methods_by_scheme = {
        "gumbel": [
            "h_ars",
            "h_log",
            "h_ind_1_over_e",
            "h_gum_star_0.1",
            "h_gum_star_0.01",
            "h_gum_star_0.005",
            TRGOF_METHOD,
            "bayes_tokenwise",
            DIRICHLET_TOKENWISE_METHOD,
            UNIONTAIL_TOKENWISE_METHOD,
        ],
        "inverse": [
            "h_neg",
            "h_dif_star_0.1",
            "h_dif_star_0.01",
            "h_dif_star_0.001",
            TRGOF_METHOD,
            "bayes_tokenwise",
        ],
    }
    if scenario == "shared_delta_equal_tail_sensitivity":
        # The Dirichlet layer is Gumbel-only, so only that panel gains a curve.
        methods_by_scheme = {
            name: (
                methods
                + ["bayes_shared"]
                + (
                    [DIRICHLET_SHARED_METHOD, UNIONTAIL_SHARED_METHOD]
                    if name == "gumbel"
                    else []
                )
            )
            for name, methods in methods_by_scheme.items()
        }

    if exclude_methods:
        methods_by_scheme = {
            name: [m for m in methods if m not in exclude_methods]
            for name, methods in methods_by_scheme.items()
        }

    zero_miss_bound = 3.0 / config.n_evaluation_alternative
    max_horizon = (
        min(config.max_horizon, 300)
        if scenario == "paper_text_iid_delta_equal_tail"
        else config.max_horizon
    )
    titles = {"gumbel": "Gumbel-max watermark",
              "inverse": "Inverse-transform watermark"}
    tokenwise_detail = (
        scenario == "paper_text_iid_delta_equal_tail" and len(schemes) == 1
    )
    width = 9.2 if tokenwise_detail else (11.8 if len(schemes) > 1 else 7.2)
    height = 5.25 if len(schemes) == 1 else 5.1
    n_panels = 2 if tokenwise_detail else len(schemes)
    gridspec_kw = {"width_ratios": (1.35, 1.0)} if tokenwise_detail else None
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(width, height),
        sharey=True,
        gridspec_kw=gridspec_kw,
    )
    axes = np.atleast_1d(axes)
    if tokenwise_detail:
        scheme = schemes[0]
        panel_specs = (
            (axes[0], scheme, rf"Prefixes through $n={max_horizon}$", max_horizon),
            (
                axes[1],
                scheme,
                rf"Early-token detail ($n\leq {min(60, max_horizon)}$)",
                min(60, max_horizon),
            ),
        )
    else:
        panel_specs = tuple(
            (axis, scheme, titles[scheme], max_horizon)
            for axis, scheme in zip(axes, schemes)
        )

    plotted_rates = [
        float(row["type2_error"])
        for row in rows
        if row["scheme"] in schemes
        and row["scenario"] == scenario
        and row["decision_rule"] == "fixed_horizon_mc_calibrated"
        and row["method"] in methods_by_scheme[str(row["scheme"])]
        and int(row["horizon"]) <= max_horizon
        and float(row["type2_error"]) > 0.0
    ]
    one_miss_rate = 1.0 / config.n_evaluation_alternative
    smallest_positive = min(plotted_rates, default=one_miss_rate)
    y_floor = 0.75 * min(one_miss_rate, smallest_positive)

    for axis, scheme, title, panel_horizon in panel_specs:
        for method in methods_by_scheme[scheme]:
            subset = [
                row
                for row in rows
                if row["scheme"] == scheme
                and row["scenario"] == scenario
                and row["decision_rule"] == "fixed_horizon_mc_calibrated"
                and row["method"] == method
                and int(row["horizon"]) <= panel_horizon
            ]
            if not subset:
                continue
            subset.sort(key=lambda row: int(row["horizon"]))
            horizon = np.asarray([row["horizon"] for row in subset], dtype=float)
            type2 = np.asarray([row["type2_error"] for row in subset], dtype=float)
            displayed_type2, zero_count = _type2_plot_values(type2)
            is_bayes = method.startswith("bayes_")
            axis.plot(
                horizon,
                displayed_type2,
                color=colors[method],
                label=labels[method],
                lw=2.5 if is_bayes else 1.5,
                # Four Bayes curves share this panel and the two layer rules sit
                # almost exactly on their spike counterparts under an equal-tail
                # generator.  Hierarchy sets dash vs solid, the tail prior sets
                # dotted, so no curve is hidden beneath another.
                ls=_bayes_linestyle(method, scenario),
            )
            if np.any(zero_count):
                # Keep roughly 25 bound symbols across a panel at most, while
                # retaining the first and last horizon of every zero-count run.
                # Point-estimate curves above are untouched and remain masked
                # at every zero, so no bound is joined to an estimate.
                marker_spacing = max(1, int(math.ceil(panel_horizon / 25.0)))
                selected_bounds = _selected_zero_bound_markers(
                    horizon, zero_count, min_spacing=marker_spacing
                )
                axis.plot(
                    horizon[selected_bounds],
                    np.full(np.count_nonzero(selected_bounds), zero_miss_bound),
                    color=colors[method],
                    ls="None",
                    marker="v",
                    ms=4.8,
                    markerfacecolor="none",
                    markeredgewidth=1.0,
                    label="_nolegend_",
                    zorder=4,
                )
        axis.set_title(title, fontsize=12)
        axis.set_xlabel(r"Scored prefix length $n$")
        axis.set_xlim(1, panel_horizon)
        axis.set_yscale("log")
        # Type II error cannot exceed one.  The legend lives in its own band
        # below the panels, so the y range need not be expanded to make room.
        axis.set_ylim(y_floor, 1.05)
        axis.grid(alpha=0.18)

    axes[0].set_ylabel("Type II error")
    description = (
        "Tokenwise-deficit equal-tail design"
        if scenario == "paper_text_iid_delta_equal_tail"
        else "Shared-deficit equal-tail design"
    )
    if tokenwise_detail:
        description += f": {titles[schemes[0]]}"
    fig.suptitle(description, fontsize=13.5, y=0.995)
    # Combine panel legends in a reserved band, retaining first appearance so
    # family order remains the same as the plotted method order.
    legend_items = {}
    for axis in axes:
        handles, legend_labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, legend_labels):
            legend_items.setdefault(label, handle)
    legend_columns = (3 if tokenwise_detail else 2) if len(schemes) == 1 else 4
    legend_rows = int(np.ceil(len(legend_items) / legend_columns))
    legend_fontsize = 10.0 if tokenwise_detail else 8.0
    fig.legend(
        legend_items.values(), legend_items.keys(),
        loc="lower center", bbox_to_anchor=(0.5, 0.052),
        ncol=legend_columns, frameon=False, fontsize=legend_fontsize,
        handlelength=2.8, columnspacing=1.35, labelspacing=0.45,
    )
    fig.text(
        0.99,
        0.012,
        "Open triangles mark selected zero-miss horizons at the one-sided "
        f"bound 3/N = {zero_miss_bound:.4f}; bounds, not point estimates.",
        ha="right",
        va="bottom",
        fontsize=10.0 if tokenwise_detail else 8.5,
        color="#555555",
    )
    legend_band = 0.18 + 0.02 * legend_rows
    fig.tight_layout(rect=(0.0, legend_band, 1.0, 0.95))
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def resolve_output_dir(requested: Path | None, *, quick: bool) -> Path:
    """Keep smoke-run artifacts separate unless the caller chooses a path."""

    if requested is not None:
        return requested
    return DEFAULT_QUICK_RESULTS_DIR if quick else DEFAULT_RESULTS_DIR


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "artifact directory (default: the benchmark results directory, or its "
            "isolated quick/clean_benchmark subdirectory with --quick)"
        ),
    )
    parser.add_argument("--n-calibration", type=int, default=10_000)
    parser.add_argument("--n-null", type=int, default=5_000)
    parser.add_argument("--n-alternative", type=int, default=5_000)
    parser.add_argument("--max-horizon", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=24_040_1245)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small smoke run (500 calibration, 250+250 evaluation, T=80).",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write CSV/JSON results without importing Matplotlib.",
    )
    parser.add_argument(
        "--replot",
        action="store_true",
        help=(
            "Redraw the figures from the existing benchmark_results.csv without "
            "rerunning the simulation.  A figure fix should not cost a full run."
        ),
    )
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, object]]:
    """Load a written benchmark_results.csv back into plottable rows.

    Only the columns the figures read are converted; everything else stays a
    string, which is harmless because the plotting code selects by name.
    """

    numeric = {
        "horizon": int,
        "alpha": float,
        "threshold": float,
        "boundary_randomization": float,
        "type1_error": float,
        "power": float,
        "type2_error": float,
        "type1_mc_se": float,
        "type2_mc_se": float,
    }
    with path.open(newline="", encoding="utf-8") as handle:
        rows: list[dict[str, object]] = []
        for raw in csv.DictReader(handle):
            row: dict[str, object] = dict(raw)
            for column, cast in numeric.items():
                if column in row:
                    row[column] = cast(str(row[column]))
            rows.append(row)
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.quick:
        args.n_calibration = 500
        args.n_null = 250
        args.n_alternative = 250
        args.max_horizon = min(args.max_horizon, 80)
        args.batch_size = min(args.batch_size, 250)
    config = BenchmarkConfig(
        max_horizon=args.max_horizon,
        n_calibration=args.n_calibration,
        n_evaluation_null=args.n_null,
        n_evaluation_alternative=args.n_alternative,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    output_dir = resolve_output_dir(args.output_dir, quick=args.quick)
    if args.replot:
        if args.skip_plots:
            raise SystemExit("--replot and --skip-plots are mutually exclusive")
        rows = read_rows(output_dir / "benchmark_results.csv")
        for scenario, stem in (
            ("paper_text_iid_delta_equal_tail", "bayesian_vs_paper_tokenwise_delta"),
            ("shared_delta_equal_tail_sensitivity", "bayesian_vs_paper_shared_delta"),
        ):
            make_plot(rows, output_dir / stem, scenario)
        make_slide_plot(
            rows,
            output_dir / "slide_bayes_tokenwise_delta",
            "paper_text_iid_delta_equal_tail",
            config,
        )
        make_slide_plot(
            rows,
            output_dir / "slide_bayes_shared_delta",
            "shared_delta_equal_tail_sensitivity",
            config,
        )
        print(
            json.dumps(
                {"replotted_from": str(output_dir / "benchmark_results.csv")},
                indent=2,
            )
        )
        return
    indicators: dict[str, np.ndarray] = {}
    rows, metadata = run_benchmark(config, indicator_sink=indicators)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "rejection_indicators.npz",
        **{key: value for key, value in indicators.items()},
    )
    write_csv(rows, output_dir / "benchmark_results.csv")
    write_selected_csv(
        rows, config.max_horizon, output_dir / "benchmark_selected_horizons.csv"
    )
    write_json(rows, metadata, output_dir / "benchmark_summary.json")
    if not args.skip_plots:
        make_plot(
            rows,
            output_dir / "bayesian_vs_paper_tokenwise_delta",
            "paper_text_iid_delta_equal_tail",
        )
        make_plot(
            rows,
            output_dir / "bayesian_vs_paper_shared_delta",
            "shared_delta_equal_tail_sensitivity",
        )
        make_slide_plot(
            rows,
            output_dir / "slide_bayes_tokenwise_delta",
            "paper_text_iid_delta_equal_tail",
            config,
        )
        make_slide_plot(
            rows,
            output_dir / "slide_bayes_shared_delta",
            "shared_delta_equal_tail_sensitivity",
            config,
        )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "rows": len(rows),
                "runtime_seconds": metadata["runtime_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
