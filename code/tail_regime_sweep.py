#!/usr/bin/env python3
"""Tail-shape regime sweep for the Dirichlet-layer Gumbel detector.

Every other Gumbel experiment in this study uses an equal residual tail,
``Delta/(M-1)`` on each of the ``K = M-1`` non-leading coordinates.  This is
the equal-tail specification stated by Li et al. (2025) and the
``alpha -> infinity`` limit of the Dirichlet tail layer.  Because ``Delta``
constrains only the largest coordinate, the verifier cannot infer the residual
tail shape from the deficit.

This experiment evaluates tail-shape misspecification while holding the
deficit prior fixed.  Every rule is calibrated once at nominal 5% on the exact
null and then applied unchanged to six
generating tail laws, five inside the Dirichlet family and one outside it.  The
out-of-family regime is the normalized-uniform tail used by the released simulator,
which normalises i.i.d. uniforms instead of drawing a Dirichlet; the scaled tail
marginal ``K q`` then tends to ``Uniform(0, 2)`` rather than to
``Gamma(alpha, alpha)``, so it is not a member of the layer at any ``alpha``.

The frozen prior is the product of the ``Uniform(0.001, 0.5)`` deficit prior used
throughout and a uniform prior on ``alpha`` over a fixed log-spaced grid that
includes the equal-tail limit.  Neither factor is ever a function of the regime.

The headline is worst-case Type II regret across the six tail laws, matching the
deficit sweep: regret is a rule's excess Type II error over the best rule at
that regime, which is what an analyst pays for committing to a tail shape before
seeing the document.

A second enlargement of the same spike family runs alongside the Dirichlet
layer as an extra competitor: ``bayes_shared_uniontail`` puts a uniform prior on
the *number* of live tail coordinates ``J`` over the dyadic ladder
``1, 4, 16, ..., K`` and keeps the live ones equal, instead of keeping all ``K``
live and putting a prior on their dispersion.  Its component is closed form and
its widest atom ``J = K`` is exactly the equal-tail spike, so it too contains the
manuscript's existing Gumbel detector.  It is added without touching any
pre-existing rule; because maximum regret is measured against the best rule on
the displayed menu, the regret column can move even where Type II errors do not.

Scope
-----
Gumbel scheme only, shared-``Delta`` hierarchy, fixed-horizon rules only.  The
layer is inert for the inverse pivot because its limiting alternative depends on
``Delta`` alone, so there is nothing to sweep there.  "Best rule" and regret
refer only to the finite, prespecified competitor menu below; this sweep does
not optimize over all point values of ``alpha`` or all priors.  Depends only on
NumPy.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator

import numpy as np

from experiment_metadata import union_tail_metadata

import benchmark_paper_experiment as paper
import dirichlet_detector as dirichlet
import regime_sweep as deficit_sweep
import tail_family as tf


DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "bayesian_paper_benchmark"
)
DEFAULT_QUICK_RESULTS_DIR = DEFAULT_RESULTS_DIR / "quick" / "tail_regime_sweep"

# Distinct from 240401245 (clean benchmark), 240401246 (contamination),
# 240401247 (contamination quadrature check), 240401248 (deficit sweep and the
# Delta-learning check), 240401249 (short-horizon probe), and 240401251
# (contamination-asymmetry diagnostic).
SWEEP_SEED = 24_040_1250

# Separate bootstrap address from both simulation streams and the deficit
# sweep's maximum-regret bootstrap.
MAX_REGRET_BOOTSTRAP_SEED_SALT = 91_730_042

# The frozen prior on the tail-concentration parameter: uniform on this grid.
# math.inf is the equal-tail (spike) limit stated by Li et al. (2025).
DEFAULT_ALPHA_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0, math.inf)

# Point-alpha competitors: the layer with a single tail shape assumed, the
# tail-side analogue of the fixed-Delta_0 diagnostic in the deficit sweep.
DEFAULT_POINT_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)

# psi/chi table resolution.  On the manuscript's separate deterministic
# validation grid (20,000 logits equally spaced on [-30, 30], paired cyclically
# with the 96 deficit nodes, for every configured alpha), direct non-tabulated
# quadrature gives worst component log-density discrepancies of 9.5e-6 with
# 20,001 nodes and 9.5e-8 with 200,001.  These are empirical grid diagnostics,
# not certified uniform bounds.  The coarser error can accumulate over 700
# tokens, so the experiment uses the finer table.
DEFAULT_C_NODES = 200_001

# Comparators carried over from the clean benchmark and its family diagnostic.
DEFAULT_PAPER_DELTAS: tuple[float, ...] = deficit_sweep.DEFAULT_PAPER_DELTAS
DEFAULT_SPIKE_DELTA = 0.01


def _format_alpha(alpha: float) -> str:
    return "inf" if math.isinf(float(alpha)) else f"{float(alpha):g}"


def point_alpha_rule_name(alpha: float) -> str:
    """Name of the shared-Delta layer rule that fixes one tail shape."""

    return f"bayes_shared_alpha_{_format_alpha(alpha)}"


SPIKE_BAYES_RULE = "bayes_shared_spike"
MIXTURE_RULE = "bayes_shared_dirichlet_mixture"

# A second enlargement of the same spike family, added alongside the Dirichlet
# tail-shape layer rather than in place of it.  Instead of keeping the tail
# equal over all K = V-1 coordinates and putting a prior on how dispersed it is,
# this rule puts a prior on the number of live tail coordinates J and keeps the
# live ones equal: conditional on (Delta, J) the working NTP vector is
# (1-Delta, Delta/J, ..., Delta/J).  J = K is exactly the equal-tail spike every
# other Gumbel rule in this study assumes, so the family CONTAINS them and the
# finite-grid evidence bound applies.  The component is closed form,
#     f_{Delta,J}(r) = r**(Delta/(1-Delta)) + J * r**(J/Delta - 1),
# so unlike the Dirichlet layer it needs no transform table and no
# interpolation.  Shared hierarchy only: a tokenwise version would average over
# J before multiplying and so could not learn a document-level tail width.
# Named to match ``benchmark_paper_experiment.UNIONTAIL_SHARED_METHOD``.
UNIONTAIL_RULE = "bayes_shared_uniontail"


@dataclass(frozen=True)
class TailRegime:
    """One data-generating law for the tail of the NTP vector.

    ``kind='dirichlet'`` draws the tail as ``Delta * q`` with
    ``q ~ Dirichlet(alpha, ..., alpha)`` on ``K = M-1`` coordinates;
    ``alpha=math.inf`` is the equal tail, which is exactly the generator behind
    ``shared_delta_equal_tail_sensitivity`` in ``benchmark_paper_experiment``.

    ``kind='width'`` puts the deficit on ``tail_width`` coordinates and zero on
    the rest: the width branch of the union tail prior, and the shape the
    released outputs actually take, where the fitted width is one rather than the
    full vocabulary.  ``tail_width = M-1`` is again the equal tail, so the width
    regimes contain T1 as an endpoint and isolate the one factor the shape
    regimes hold fixed.  Without these the tail-width block is never exercised
    against data it is designed for, and every synthetic experiment in the study
    is equal tailed.

    ``kind='normalised_uniform'`` builds the tail by normalising i.i.d. uniforms,
    which is what the released simulator accompanying Li et al. (2025) does.  That law is outside the
    Dirichlet family at every ``alpha``, so it is the honest misspecification
    case rather than a member of the prior's support.

    In both cases one ``Delta`` is drawn per document and shared by all of its
    tokens; only the tail law changes across regimes.
    """

    label: str
    kind: str
    alpha: float
    description: str
    tail_width: int = 0

    def validate(self) -> None:
        if self.kind not in ("dirichlet", "normalised_uniform", "width"):
            raise ValueError(
                "regime kind must be 'dirichlet', 'width' or 'normalised_uniform'"
            )
        if self.kind == "dirichlet":
            value = float(self.alpha)
            if math.isnan(value) or value <= 0.0:
                raise ValueError("a dirichlet regime needs alpha > 0 (inf allowed)")
        elif self.kind == "width":
            if not math.isnan(float(self.alpha)):
                raise ValueError("a width regime must carry alpha=nan")
            if int(self.tail_width) < 1:
                raise ValueError("a width regime needs tail_width >= 1")
        elif not math.isnan(float(self.alpha)):
            raise ValueError("a normalised_uniform regime must carry alpha=nan")
        if self.kind != "width" and int(self.tail_width) != 0:
            raise ValueError("tail_width is only meaningful for a width regime")

    @property
    def in_dirichlet_family(self) -> bool:
        return self.kind == "dirichlet"

    def on_prior_grid(self, alpha_grid: tuple[float, ...]) -> bool:
        """Whether this regime's tail law is an atom of the frozen alpha prior."""

        if not self.in_dirichlet_family:
            return False
        return any(float(self.alpha) == float(node) for node in alpha_grid)

    def simulate(
        self,
        rng: np.random.Generator,
        deltas: np.ndarray,
        vocabulary_size: int,
    ) -> np.ndarray:
        """Exact Gumbel pivots for this tail law at the given per-token deficits."""

        self.validate()
        if self.in_dirichlet_family:
            pivot = tf.dirichlet_tail_pivot(
                float(self.alpha), vocabulary_size - 1, c_nodes=DEFAULT_C_NODES
            )
            return pivot.simulate(rng, deltas)
        if self.kind == "width":
            return tf.simulate_narrow_width_pivots(
                rng, deltas, int(self.tail_width)
            )
        return tf.simulate_normalised_uniform_pivots(rng, deltas, vocabulary_size)


# ORDER MATTERS.  Each regime draws from its own spawned stream, indexed by
# position, so inserting a regime renumbers every stream after it and silently
# changes the published numbers of the regimes that follow.  New regimes are
# APPENDED; the shape laws T1..T6 keep the positions they were published with.
DEFAULT_TAIL_REGIMES = (
    TailRegime(
        "T1",
        "dirichlet",
        math.inf,
        "equal tail (alpha -> inf); the specification stated by Li et al. (2025) and the "
        "generator behind every other Gumbel result in this study",
    ),
    TailRegime(
        "T2",
        "dirichlet",
        10.0,
        "Dirichlet(10) tail; mildly dispersed, close to the equal-tail limit",
    ),
    TailRegime(
        "T3",
        "dirichlet",
        3.0,
        "Dirichlet(3) tail; variance-matched to the released simulator's normalized-uniform tail, "
        "and not an atom of the frozen alpha prior",
    ),
    TailRegime(
        "T4",
        "dirichlet",
        0.5,
        "Dirichlet(0.5) tail; sparse, and deliberately NOT an atom of the "
        "frozen alpha prior, and further from every atom than T3",
    ),
    TailRegime(
        "T5",
        "dirichlet",
        0.1,
        "Dirichlet(0.1) tail; strongly sparse, most of the deficit on a few "
        "coordinates",
    ),
    TailRegime(
        "T6",
        "normalised_uniform",
        math.nan,
        "tail built by normalising i.i.d. uniforms, as in the released simulator of Li et al. (2025); "
        "OUTSIDE the Dirichlet family at every alpha",
    ),
    TailRegime(
        "W1",
        "width",
        math.nan,
        "deficit on a single tail coordinate; the width fitted on the released outputs",
        tail_width=1,
    ),
    TailRegime(
        "W2",
        "width",
        math.nan,
        "deficit spread equally over 4 tail coordinates",
        tail_width=4,
    ),
    TailRegime(
        "W3",
        "width",
        math.nan,
        "deficit spread equally over 16 tail coordinates",
        tail_width=16,
    ),
    TailRegime(
        "W4",
        "width",
        math.nan,
        "deficit spread equally over 64 tail coordinates; approaching the full width",
        tail_width=64,
    ),
)


@dataclass(frozen=True)
class TailSweepConfig:
    vocabulary_size: int = 1000
    horizons: tuple[int, ...] = (100, 300, 700)
    alpha_level: float = 0.05
    # The frozen prior.  Never a function of the regime.
    # Detector prior only.  The generating regimes below are unchanged; the
    # detector is deliberately not told their range.
    prior_low: float = 0.001
    prior_high: float = 0.5
    allow_large_delta: bool = False
    alpha_grid: tuple[float, ...] = DEFAULT_ALPHA_GRID
    point_alphas: tuple[float, ...] = DEFAULT_POINT_ALPHAS
    paper_deltas: tuple[float, ...] = DEFAULT_PAPER_DELTAS
    spike_delta: float = DEFAULT_SPIKE_DELTA
    regimes: tuple[TailRegime, ...] = DEFAULT_TAIL_REGIMES
    n_calibration: int = 10_000
    n_evaluation_null: int = 5_000
    n_evaluation_alternative: int = 5_000
    batch_size: int = 500
    seed: int = SWEEP_SEED
    bayes_quadrature_nodes: int = 96
    c_nodes: int = DEFAULT_C_NODES
    max_regret_bootstrap_replicates: int = (
        deficit_sweep.DEFAULT_MAX_REGRET_BOOTSTRAP_REPLICATES
    )
    max_regret_bootstrap_batch_size: int = (
        deficit_sweep.DEFAULT_MAX_REGRET_BOOTSTRAP_BATCH_SIZE
    )

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)

    @property
    def tail_size(self) -> int:
        return self.vocabulary_size - 1

    def paper_config(self) -> paper.BenchmarkConfig:
        """The clean-benchmark config carrying the frozen deficit prior."""

        return paper.BenchmarkConfig(
            vocabulary_size=self.vocabulary_size,
            max_horizon=self.max_horizon,
            alpha=self.alpha_level,
            prior_low=self.prior_low,
            prior_high=self.prior_high,
            n_calibration=self.n_calibration,
            n_evaluation_null=self.n_evaluation_null,
            n_evaluation_alternative=self.n_evaluation_alternative,
            batch_size=self.batch_size,
            seed=self.seed,
            bayes_quadrature_nodes=self.bayes_quadrature_nodes,
        )

    def rule_names(self) -> tuple[str, ...]:
        return (
            (SPIKE_BAYES_RULE,)
            + tuple(point_alpha_rule_name(a) for a in self.point_alphas)
            + (MIXTURE_RULE, UNIONTAIL_RULE)
            + (
                deficit_sweep.spike_rule_name(self.spike_delta),
                *(deficit_sweep.paper_rule_name(d) for d in self.paper_deltas),
                *deficit_sweep.REFERENCE_SCORES,
            )
        )

    def regime_labels(self) -> tuple[str, ...]:
        return tuple(regime.label for regime in self.regimes)

    def validate(self) -> None:
        self.paper_config().validate()
        max_delta = 1.0 - 1.0 / self.vocabulary_size
        if not self.horizons:
            raise ValueError("at least one horizon is required")
        if tuple(sorted(set(self.horizons))) != tuple(self.horizons):
            raise ValueError("horizons must be strictly increasing and distinct")
        if self.horizons[0] < 1:
            raise ValueError("horizons must be positive")
        if not self.alpha_grid:
            raise ValueError("the alpha prior needs at least one node")
        if len({_format_alpha(a) for a in self.alpha_grid}) != len(self.alpha_grid):
            raise ValueError("alpha_grid values must be distinct")
        for value in self.alpha_grid:
            if math.isnan(float(value)) or float(value) <= 0.0:
                raise ValueError("alpha_grid values must be positive (inf allowed)")
        for value in self.point_alphas:
            if not any(float(value) == float(node) for node in self.alpha_grid):
                raise ValueError("every point alpha must be an atom of alpha_grid")
        if not self.paper_deltas:
            raise ValueError("paper_deltas must not be empty")
        for delta in self.paper_deltas:
            if not 0.0 < float(delta) <= max_delta + 1e-12:
                raise ValueError(
                    "paper_deltas must lie in (0, 1 - 1/vocabulary_size]"
                )
        if not 0.0 < float(self.spike_delta) <= max_delta + 1e-12:
            raise ValueError(
                "spike_delta must lie in (0, 1 - 1/vocabulary_size]"
            )
        if not self.regimes:
            raise ValueError("at least one regime is required")
        if len(set(self.regime_labels())) != len(self.regimes):
            raise ValueError("regime labels must be distinct")
        for regime in self.regimes:
            regime.validate()
        if len(set(self.rule_names())) != len(self.rule_names()):
            raise ValueError("rule names must be distinct")
        if self.c_nodes < 16:
            raise ValueError("c_nodes must be at least 16")
        if self.max_regret_bootstrap_replicates < 2:
            raise ValueError("max_regret_bootstrap_replicates must be at least 2")
        if self.max_regret_bootstrap_batch_size < 1:
            raise ValueError("max_regret_bootstrap_batch_size must be positive")


def frozen_alpha_prior(config: TailSweepConfig) -> tuple[tuple[float, ...], np.ndarray]:
    """The frozen tail-concentration prior: uniform on ``config.alpha_grid``."""

    grid = tuple(float(a) for a in config.alpha_grid)
    return grid, np.full(len(grid), 1.0 / len(grid), dtype=float)


class DirichletLayerScores:
    """Shared-Delta log Bayes factors for the Dirichlet tail layer.

    One :class:`tail_family.DirichletTailPivot` per ``alpha`` supplies the
    tabulated transform ``psi_alpha``.  All of them share the same log-spaced
    ``c`` grid, so the tables can be stacked and evaluated with one arithmetic
    index instead of one binary search per ``alpha``: the grid is uniform in
    ``log c``, which turns interpolation into a gather and is what makes a
    700-token, 45,000-document sweep tractable.

    The component densities are exactly normalised by construction *before
    interpolation* (the tail quadrature nodes are rescaled so that ``K E[q] =
    1``), so the corresponding non-interpolated shared Bayes factor is a null
    martingale.  Linear interpolation perturbs the density slightly.
    :meth:`normalisation_report` records a high-accuracy numerical mass check,
    but it is not a certified one-sided bound and therefore does not by itself
    prove that the fast realised process is a supermartingale.
    """

    def __init__(self, config: TailSweepConfig) -> None:
        config.validate()
        self.config = config
        self.alphas, self.alpha_weights = frozen_alpha_prior(config)
        self.pivots = tuple(
            tf.dirichlet_tail_pivot(a, config.tail_size, c_nodes=config.c_nodes)
            for a in self.alphas
        )
        grid = self.pivots[0].log_c_grid
        for pivot in self.pivots[1:]:
            if not np.array_equal(pivot.log_c_grid, grid):
                raise ValueError("all alpha tables must share one log-c grid")
        self._log_c_grid = grid
        self._log_c_min = float(grid[0])
        self._inv_step = 1.0 / float(grid[1] - grid[0])
        self._n_grid = int(grid.size)
        # log psi_alpha(c) + c, stacked over alpha.  Shifting by +c makes the
        # table nearly linear in log c, so linear interpolation is accurate.
        self._table = np.stack([pivot._psi_shifted for pivot in self.pivots])

        self.deltas, delta_weights = deficit_sweep.frozen_delta_quadrature(
            _deficit_config(config)
        )
        self.delta_weights = delta_weights
        self._log_delta_weights = np.log(delta_weights)
        self._log_joint_weights = (
            np.log(self.alpha_weights)[:, None] + self._log_delta_weights[None, :]
        ).reshape(-1)
        self._top_exponent = self.deltas / (1.0 - self.deltas)
        self._inverse_delta = 1.0 / self.deltas
        self._log_tail_size = math.log(config.tail_size)
        self._alpha_index = {_format_alpha(a): i for i, a in enumerate(self.alphas)}

    # -- component log densities ---------------------------------------------

    def component_log_density(self, pivots: np.ndarray) -> np.ndarray:
        """``log f_{Delta,alpha}(r)`` with shape ``(alpha, rows, Delta)``.

        ``pivots`` is a one-dimensional batch of Gumbel pivots.  Exact zeros are
        clamped to the smallest positive double: they occur with probability
        ``2**-53`` per draw, and clamping keeps ``c = -log(r)/Delta`` finite and
        inside the tabulated range instead of producing ``inf - inf``.
        """

        r = np.asarray(pivots, dtype=float).reshape(-1, 1)
        if np.any(~np.isfinite(r)) or np.any((r < 0.0) | (r > 1.0)):
            raise ValueError("Gumbel pivots must lie in [0, 1]")
        r = np.maximum(r, np.finfo(float).tiny)
        log_r = np.log(r)
        deficit = -log_r
        c = deficit * self._inverse_delta[None, :]
        with np.errstate(divide="ignore"):
            position = (np.log(c) - self._log_c_min) * self._inv_step
        np.clip(position, 0.0, self._n_grid - 1.000001, out=position)
        lower_index = position.astype(np.intp)
        fraction = position - lower_index
        lower = self._table[:, lower_index]
        upper = self._table[:, lower_index + 1]
        shifted = lower + (upper - lower) * fraction[None, :, :]
        log_tail = self._log_tail_size + deficit[None, :, :] + (shifted - c[None, :, :])
        log_top = self._top_exponent[None, None, :] * log_r[None, :, :]
        return np.logaddexp(log_top, log_tail)

    # -- diagnostics ----------------------------------------------------------

    def spike_agreement(self, pivots: np.ndarray) -> float:
        """Worst ``|log f_{Delta,inf} - log f^{sp}_Delta|`` over the Delta nodes.

        The ``alpha -> inf`` member of the layer must reproduce the closed-form
        equal-tail spike density, which is the density the manuscript's existing
        Gumbel detector evaluates.  Both sides are read at the same Delta
        quadrature nodes, so nothing but the layer's own tabulated transform is
        being tested.
        """

        if "inf" not in self._alpha_index:
            raise ValueError("the alpha grid does not contain the equal-tail limit")
        r = np.asarray(pivots, dtype=float).reshape(-1)
        layer = self.component_log_density(r)[self._alpha_index["inf"]]
        log_r = np.log(np.maximum(r, np.finfo(float).tiny))[:, None]
        closed_form = np.logaddexp(
            self._top_exponent[None, :] * log_r,
            self._log_tail_size
            + ((self.config.tail_size * self._inverse_delta) - 1.0)[None, :] * log_r,
        )
        return float(np.max(np.abs(layer - closed_form)))

    def analytic_normalisation(self) -> dict[str, float]:
        """Analytic component mass ``(1-Delta) + Delta K E[q]`` per alpha.

        Exactly one for every alpha once the tail nodes are rescaled, which is
        what keeps the discretised layer a probability model and the shared Bayes
        factor a null martingale.
        """

        report: dict[str, float] = {}
        for alpha, pivot in zip(self.alphas, self.pivots):
            mass = pivot.normalisation(self.deltas)
            report[_format_alpha(alpha)] = float(np.max(np.abs(mass - 1.0)))
        return report

    def normalisation_report(
        self, panels: int = 2_000, gauss_nodes: int = 40, limit: float = 40.0
    ) -> dict[str, object]:
        """Numerically estimated mass of the interpolated density per alpha.

        Integrated in ``z = logit(r)`` so that both the ``r -> 0`` end and the
        very sharp ``r -> 1`` tail spike are resolved, and evaluated at exactly
        the Delta quadrature nodes the detector uses.  Because the integral is
        truncated and evaluated in floating point, its sign is diagnostic rather
        than a certificate that the true interpolant is a sub-probability density.
        """

        nodes, weights = np.polynomial.legendre.leggauss(int(gauss_nodes))
        edges = np.linspace(-float(limit), float(limit), int(panels) + 1)
        half = 0.5 * (edges[1] - edges[0])
        centres = 0.5 * (edges[:-1] + edges[1:])
        mass = np.zeros((len(self.alphas), self.deltas.size), dtype=float)
        block = 64
        for start in range(0, centres.size, block):
            stop = min(start + block, centres.size)
            z = (centres[start:stop, None] + half * nodes[None, :]).reshape(-1)
            quadrature = np.tile(weights * half, stop - start)
            sigmoid = 1.0 / (1.0 + np.exp(-z))
            jacobian = sigmoid * (1.0 - sigmoid)
            density = np.exp(self.component_log_density(sigmoid))
            mass += np.einsum("z,azj->aj", quadrature * jacobian, density)
        deviation = mass - 1.0
        per_alpha = {
            _format_alpha(alpha): float(deviation[index][
                np.argmax(np.abs(deviation[index]))
            ])
            for index, alpha in enumerate(self.alphas)
        }
        return {
            "diagnostic_only": True,
            "certified_one_sided_bound": False,
            "delta_nodes_checked": int(self.deltas.size),
            "logit_quadrature": {
                "panels": int(panels),
                "gauss_nodes": int(gauss_nodes),
                "limit": float(limit),
            },
            "max_abs_mass_minus_one": float(np.max(np.abs(deviation))),
            "signed_worst_by_alpha": per_alpha,
            "all_masses_at_most_one": bool(np.all(mass <= 1.0)),
        }


def _deficit_config(config: TailSweepConfig) -> deficit_sweep.RegimeSweepConfig:
    """A deficit-sweep config carrying only the frozen Delta prior.

    Reused so that the Delta quadrature in this sweep is literally the same
    function of the same prior endpoints as in the deficit sweep.
    """

    return deficit_sweep.RegimeSweepConfig(
        vocabulary_size=config.vocabulary_size,
        horizons=config.horizons,
        alpha=config.alpha_level,
        prior_low=config.prior_low,
        prior_high=config.prior_high,
        n_calibration=config.n_calibration,
        n_evaluation_null=config.n_evaluation_null,
        n_evaluation_alternative=config.n_evaluation_alternative,
        batch_size=config.batch_size,
        seed=config.seed,
        bayes_quadrature_nodes=config.bayes_quadrature_nodes,
    )


def build_uniontail_grid(config: TailSweepConfig) -> dirichlet.UnionTailBayesGrid:
    """The frozen joint prior on ``(Delta, J)`` for the Gumbel pivot.

    The deficit factor is :func:`regime_sweep.frozen_delta_quadrature` on the
    frozen prior endpoints, which is node-for-node the same Gauss-Legendre rule
    as ``dirichlet_detector.gauss_legendre_delta_grid(low, high, nodes)`` and is
    the single Delta grid every Bayes score in this sweep is built from.  So
    this rule differs from ``bayes_shared_spike`` only in carrying a latent tail
    width.  The width factor is uniform on the frozen dyadic ladder
    ``1, 4, 16, ..., K``, whose last atom is the equal-tail spike, so
    ``bayes_shared_spike`` is one component of this mixture.

    Like :func:`frozen_alpha_prior` this is a pure function of ``config`` and
    therefore cannot see a regime.  It is also cheap and deterministic, which is
    why :func:`all_rule_paths` can rebuild it rather than take it as an
    argument: that keeps the scoring entry point's signature free of anything a
    regime could be smuggled in through.
    """

    deltas, weights = deficit_sweep.frozen_delta_quadrature(_deficit_config(config))
    return dirichlet.UnionTailBayesGrid(
        delta_grid=deltas,
        delta_weights=weights,
        tail_size=config.tail_size,
        alpha_grid=config.alpha_grid,
        # None -> the frozen dyadic ladder below the full width; the full-width
        # atom already lives in the Dirichlet block of the union.
        tail_width_grid=None,
        c_nodes=config.c_nodes,
        allow_large_delta=config.allow_large_delta,
    )


def _additive_increments(
    batch: np.ndarray, config: TailSweepConfig
) -> Iterator[tuple[str, np.ndarray]]:
    """Per-token increments for the two rules that are plain token sums."""

    yield (
        deficit_sweep.spike_rule_name(config.spike_delta),
        paper.spike_point_mass_score(
            batch, float(config.spike_delta), config.vocabulary_size
        ),
    )
    for delta in config.paper_deltas:
        yield (
            deficit_sweep.paper_rule_name(delta),
            paper.paper_gumbel_optimal_score(batch, float(delta)),
        )
    for name in deficit_sweep.REFERENCE_SCORES:
        yield name, paper.gumbel_score(batch, name, None)


def all_rule_paths(
    pivots: np.ndarray, config: TailSweepConfig, layer: DirichletLayerScores
) -> dict[str, np.ndarray]:
    """Every rule's statistic at the selected horizons.

    There is deliberately no ``regime`` argument: the frozen priors enter only
    through ``config`` and ``layer``, so no rule can be retuned to the tail law
    that produced ``pivots``.  The tail-width grid is rebuilt here from
    ``config`` alone for the same reason.

    Adding the tail-width rule consumes no randomness: every rule is a pure
    function of the ``pivots`` already drawn by the caller, so the pivot streams
    seen by the pre-existing rules are bit-for-bit unchanged.
    """

    if pivots.ndim != 2 or pivots.shape[1] < config.max_horizon:
        raise ValueError("pivots must be (rows, >= max_horizon)")
    rows = pivots.shape[0]
    selected = np.asarray(config.horizons, dtype=int) - 1
    rules = config.rule_names()
    output = {
        name: np.empty((rows, len(config.horizons)), dtype=float) for name in rules
    }
    record = {horizon - 1: index for index, horizon in enumerate(config.horizons)}

    n_alpha = len(layer.alphas)
    n_delta = layer.deltas.size
    spike_index = layer._alpha_index["inf"]
    point_indices = {
        point_alpha_rule_name(a): layer._alpha_index[_format_alpha(a)]
        for a in config.point_alphas
    }
    log_delta_weights = layer._log_delta_weights
    log_joint_weights = layer._log_joint_weights
    tailwidth = build_uniontail_grid(config)

    for start in range(0, rows, config.batch_size):
        stop = min(start + config.batch_size, rows)
        batch = pivots[start:stop, : config.max_horizon]
        for name, increments in _additive_increments(batch, config):
            output[name][start:stop] = np.cumsum(increments, axis=1)[:, selected]

        # The tail-width layer carries its own (Delta, J) component recursion.
        # It is closed form, so there is no table to share with the Dirichlet
        # layer and nothing to interpolate.
        output[UNIONTAIL_RULE][start:stop] = tailwidth.shared_paths(
            batch, horizons=config.horizons
        )[0]

        accumulator = np.zeros((n_alpha, stop - start, n_delta), dtype=float)
        for time_index in range(config.max_horizon):
            accumulator += layer.component_log_density(batch[:, time_index])
            column = record.get(time_index)
            if column is None:
                continue
            # The Gumbel null density is one, so the accumulated component log
            # density is already the accumulated component log likelihood ratio.
            output[SPIKE_BAYES_RULE][start:stop, column] = deficit_sweep._logsumexp_rows(
                accumulator[spike_index] + log_delta_weights[None, :]
            )
            for name, index in point_indices.items():
                output[name][start:stop, column] = deficit_sweep._logsumexp_rows(
                    accumulator[index] + log_delta_weights[None, :]
                )
            output[MIXTURE_RULE][start:stop, column] = deficit_sweep._logsumexp_rows(
                accumulator.transpose(1, 0, 2).reshape(stop - start, -1)
                + log_joint_weights[None, :]
            )
    return output


def exact_spike_shared_bayes(
    pivots: np.ndarray, config: TailSweepConfig
) -> np.ndarray:
    """Closed-form shared-Delta spike Bayes factor, for cross-checking.

    This is the routine the clean benchmark and the deficit sweep use.  Running
    it beside the ``alpha=inf`` member of the layer is how we verify that the
    layer contains the manuscript's existing Gumbel detector as a special case.
    """

    deficit_config = _deficit_config(config)
    deltas, weights = deficit_sweep.frozen_delta_quadrature(deficit_config)
    return deficit_sweep._shared_bayes_batch(
        pivots[:, : config.max_horizon], deltas, np.log(weights), deficit_config
    )


def _binomial_se(rate: float, trials: int) -> float:
    return math.sqrt(max(rate * (1.0 - rate), 0.0) / trials)


def tail_diagnostics(
    rng: np.random.Generator, regime: TailRegime, config: TailSweepConfig, draws: int
) -> dict[str, object]:
    """Moments of the *selected* tail coordinate under a regime's tail law.

    Selection size-biases the tail, so this reports the law of ``K B`` with
    ``B`` the chosen coordinate.  For ``Dirichlet(alpha)`` the unselected scaled
    marginal ``K q`` has variance ``(K-1)/(K alpha + 1)``; for the released
    normalised-uniform tail it tends to ``Uniform(0, 2)``, variance ``1/3``; for
    a width regime it is degenerate at ``K/J`` with variance ``K/J - 1``.
    """

    regime.validate()
    tail_size = config.tail_size
    if regime.in_dirichlet_family:
        alpha = float(regime.alpha)
        if math.isinf(alpha):
            selected = np.full(draws, 1.0 / tail_size)
        else:
            selected = rng.beta(alpha + 1.0, (tail_size - 1.0) * alpha, size=draws)
        nominal_unselected_variance = float(
            (tail_size - 1.0) / (tail_size * alpha + 1.0)
        ) if math.isfinite(alpha) else 0.0
    elif regime.kind == "width":
        # A width regime's live coordinates all carry 1/J, so the size-biased
        # selected coordinate is deterministic at 1/J and the scaled marginal
        # K q takes K/J on J coordinates and zero on the rest: variance K/J - 1.
        # Falling through to the normalised-uniform branch here reported the
        # wrong law for every width regime, and did so identically for all of
        # them, which is how it was caught.
        width = float(regime.tail_width)
        selected = np.full(draws, 1.0 / width)
        nominal_unselected_variance = float(tail_size / width - 1.0)
    else:
        selected = tf.normalised_uniform_tail(rng, draws, tail_size)
        nominal_unselected_variance = 1.0 / 3.0
    scaled = selected * tail_size
    return {
        "law_of_the_selected_tail_coordinate": "K * B, B size-biased",
        "n_draws": int(draws),
        "scaled_selected_mean": float(scaled.mean()),
        "scaled_selected_variance": float(scaled.var()),
        "scaled_selected_max": float(scaled.max()),
        "nominal_scaled_unselected_variance": nominal_unselected_variance,
    }


def run_tail_sweep(config: TailSweepConfig) -> dict[str, object]:
    """Run the sweep and return the complete JSON-serialisable payload."""

    config.validate()
    started = time.perf_counter()
    layer = DirichletLayerScores(config)
    # Same pure function of `config` that `all_rule_paths` calls, so the grid
    # reported here is the grid that scored the documents.
    tailwidth = build_uniontail_grid(config)
    rules = list(config.rule_names())
    horizons = list(config.horizons)

    seeds = np.random.SeedSequence(config.seed).spawn(4 + len(config.regimes))
    rng_calibration = np.random.default_rng(seeds[0])
    rng_evaluation_null = np.random.default_rng(seeds[1])
    rng_diagnostics = np.random.default_rng(seeds[2])
    rng_validation = np.random.default_rng(seeds[3])
    regime_rngs = [np.random.default_rng(seed) for seed in seeds[4:]]

    paper_config = config.paper_config()

    # One exact-null calibration sample serves every regime: the Gumbel pivot is
    # Uniform(0,1) under the null whatever the alternative's tail looks like.
    calibration_pivots = paper.simulate_gumbel_null(
        rng_calibration, config.n_calibration, config.max_horizon, paper_config
    )
    calibration_paths = all_rule_paths(calibration_pivots, config, layer)
    # Cross-check the layer against the closed form on real null pivots.
    closed_form = exact_spike_shared_bayes(calibration_pivots[:1000], config)
    spike_path_gap = float(
        np.max(np.abs(closed_form - calibration_paths[SPIKE_BAYES_RULE][:1000]))
    )
    cutoffs, gammas = paper.calibrate_randomized_boundary(
        calibration_paths, config.alpha_level
    )
    del calibration_pivots, calibration_paths

    evaluation_null_pivots = paper.simulate_gumbel_null(
        rng_evaluation_null, config.n_evaluation_null, config.max_horizon, paper_config
    )
    null_paths = all_rule_paths(evaluation_null_pivots, config, layer)
    type1 = {
        rule: paper.expected_rejection_rate(
            null_paths[rule], cutoffs[rule], gammas[rule]
        )
        for rule in rules
    }
    del evaluation_null_pivots, null_paths

    cells: list[dict[str, object]] = []
    regime_reports: dict[str, dict[str, object]] = {}
    type2: dict[str, dict[str, dict[str, float]]] = {
        str(horizon): {rule: {} for rule in rules} for horizon in horizons
    }
    bootstrap_type2 = np.empty(
        (
            config.max_regret_bootstrap_replicates,
            len(config.regimes),
            len(rules),
            len(horizons),
        ),
        dtype=float,
    )

    for regime_index, (regime, rng) in enumerate(zip(config.regimes, regime_rngs)):
        # The generating law of Li et al. (2025), never the detector prior.
        # The whole point of the sweep is that the prior may widen while the
        # data-generating law stays fixed; reading prior_low/prior_high here
        # would make the two prior arms score different data.
        column = rng.uniform(
            paper_config.delta_low, paper_config.delta_high,
            size=(config.n_evaluation_alternative, 1),
        )
        deltas = np.repeat(column, config.max_horizon, axis=1)
        pivots = regime.simulate(rng, deltas, config.vocabulary_size)
        regime_reports[regime.label] = {
            "kind": regime.kind,
            "alpha": None if not regime.in_dirichlet_family else (
                "inf" if math.isinf(float(regime.alpha)) else float(regime.alpha)
            ),
            "description": regime.description,
            "in_dirichlet_family": bool(regime.in_dirichlet_family),
            "alpha_is_an_atom_of_the_frozen_prior": bool(
                regime.on_prior_grid(tuple(layer.alphas))
            ),
            "realized_mean_delta": float(column.mean()),
            "realized_min_delta": float(column.min()),
            "realized_max_delta": float(column.max()),
            "tail_diagnostics": tail_diagnostics(
                rng_diagnostics, regime, config, 200_000
            ),
        }
        alternative_paths = all_rule_paths(pivots, config, layer)
        del pivots, deltas
        loss_cube = np.stack(
            [
                deficit_sweep.conditional_miss_contributions(
                    alternative_paths[rule], cutoffs[rule], gammas[rule]
                )
                for rule in rules
            ],
            axis=1,
        )
        bootstrap_rng = np.random.default_rng(
            np.random.SeedSequence(
                [config.seed, MAX_REGRET_BOOTSTRAP_SEED_SALT, regime_index]
            )
        )
        bootstrap_type2[:, regime_index] = deficit_sweep.paired_bootstrap_means(
            loss_cube,
            bootstrap_rng,
            replicates=config.max_regret_bootstrap_replicates,
            batch_size=config.max_regret_bootstrap_batch_size,
        )
        del loss_cube
        for rule in rules:
            rejection = paper.expected_rejection_rate(
                alternative_paths[rule], cutoffs[rule], gammas[rule]
            )
            for index, horizon in enumerate(horizons):
                error = float(1.0 - rejection[index])
                type2[str(horizon)][rule][regime.label] = error
                cells.append(
                    {
                        "regime": regime.label,
                        "rule": rule,
                        "horizon": int(horizon),
                        "alpha_level": float(config.alpha_level),
                        "threshold": float(cutoffs[rule][index]),
                        "boundary_randomization": float(gammas[rule][index]),
                        "type1_error": float(type1[rule][index]),
                        "power": float(rejection[index]),
                        "type2_error": error,
                        "type2_mc_se": _binomial_se(
                            error, config.n_evaluation_alternative
                        ),
                    }
                )
        del alternative_paths

    # Same benchmark as the point estimate: the equal-tail diagnostics are
    # excluded from the envelope, so the SE describes the reported quantity.
    benchmark_mask = np.array(
        [deficit_sweep.is_benchmark_rule(rule) for rule in rules], dtype=bool
    )
    max_regret_mc_se, pooled_max_regret_mc_se = (
        deficit_sweep.bootstrap_max_regret_mc_se(bootstrap_type2, benchmark_mask)
    )
    # Arm-restricted standard errors, matching the arm-restricted regret above.
    # The regret column of each table is a maximum over that table's regimes, so
    # its Monte Carlo error must be bootstrapped over the same subset; the
    # pooled SE belongs to a maximum neither table displays.
    arm_indices = {
        arm: [
            index
            for index, regime in enumerate(config.regimes)
            if regime.label.startswith(prefix)
        ]
        for arm, prefix in (("shape", "T"), ("width", "W"))
    }
    arm_mc_se: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for arm, indices in arm_indices.items():
        if not indices:
            continue
        arm_mc_se[arm] = deficit_sweep.bootstrap_max_regret_mc_se(
            bootstrap_type2[:, indices, :, :], benchmark_mask
        )
    del bootstrap_type2

    regret: dict[str, dict[str, dict[str, float]]] = {}
    best_by_regime: dict[str, dict[str, object]] = {}
    max_regret_by_horizon: dict[str, dict[str, dict[str, object]]] = {}
    ranking: dict[str, list[dict[str, object]]] = {}
    saturated: dict[str, list[str]] = {}
    labels = list(config.regime_labels())

    for horizon_index, horizon in enumerate(horizons):
        key = str(horizon)
        horizon_regret, best = deficit_sweep.regret_table(type2[key])
        regret[key] = horizon_regret
        best_by_regime[key] = {
            label: {
                "type2_error": best[label],
                "rules": sorted(
                    rule for rule in rules
                    if deficit_sweep.is_benchmark_rule(rule)
                    and type2[key][rule][label] == best[label]
                ),
            }
            for label in labels
        }
        max_regret_by_horizon[key] = deficit_sweep.max_regret(horizon_regret)
        # Regret restricted to each arm.  The pooled figure above maximizes over
        # every regime in the sweep, so adding the width arm would silently
        # change the shape arm's published regret column; each table reports
        # regret over the laws it displays.
        for arm, prefix in (("shape", "T"), ("width", "W")):
            subset = {
                rule: {
                    label: value
                    for label, value in by_label.items()
                    if label.startswith(prefix)
                }
                for rule, by_label in horizon_regret.items()
            }
            if not next(iter(subset.values()), {}):
                continue
            arm_regret = deficit_sweep.max_regret(subset)
            for rule in rules:
                max_regret_by_horizon[key][rule][f"max_regret_{arm}"] = (
                    arm_regret[rule]["max_regret"]
                )
                max_regret_by_horizon[key][rule][f"argmax_regime_{arm}"] = (
                    arm_regret[rule]["argmax_regime"]
                )
        for rule_index, rule in enumerate(rules):
            max_regret_by_horizon[key][rule]["max_regret_mc_se"] = float(
                max_regret_mc_se[rule_index, horizon_index]
            )
            for arm, (per_horizon_se, _) in arm_mc_se.items():
                max_regret_by_horizon[key][rule][f"max_regret_{arm}_mc_se"] = float(
                    per_horizon_se[rule_index, horizon_index]
                )
        ranking[key] = [
            {
                "rule": rule,
                "max_regret": max_regret_by_horizon[key][rule]["max_regret"],
                "max_regret_mc_se": max_regret_by_horizon[key][rule][
                    "max_regret_mc_se"
                ],
                "argmax_regime": max_regret_by_horizon[key][rule]["argmax_regime"],
            }
            for rule in sorted(
                rules,
                key=lambda name: (
                    max_regret_by_horizon[key][name]["max_regret"],
                    name,
                ),
            )
        ]
        saturated[key] = [
            label
            for label in labels
            if all(type2[key][rule][label] == 0.0 for rule in rules)
        ]

    pooled = deficit_sweep.pooled_max_regret(regret)
    for rule_index, rule in enumerate(rules):
        pooled[rule]["max_regret_mc_se"] = float(
            pooled_max_regret_mc_se[rule_index]
        )
        for arm, (_, pooled_arm_se) in arm_mc_se.items():
            pooled[rule][f"max_regret_{arm}_mc_se"] = float(
                pooled_arm_se[rule_index]
            )

    tables: dict[str, dict[str, str]] = {}
    for horizon in horizons:
        key = str(horizon)
        tables[key] = {
            "type2": deficit_sweep.format_table(
                f"Type II error at n={horizon} "
                f"(nominal 5%, {config.n_evaluation_alternative:,} documents per cell)",
                rules,
                labels,
                type2[key],
            ),
            "regret": deficit_sweep.format_table(
                f"Regret at n={horizon} (Type II minus the best rule at that regime)",
                rules,
                labels,
                regret[key],
                extra_columns=[
                    (
                        "MAX REGRET",
                        {
                            rule: float(
                                max_regret_by_horizon[key][rule]["max_regret"]
                            )
                            for rule in rules
                        },
                    )
                ],
            ),
        }

    check_pivots = rng_validation.uniform(size=20_000)
    validation = {
        "alpha_inf_layer_vs_closed_form_spike_max_abs_log_density_difference": (
            layer.spike_agreement(check_pivots)
        ),
        "alpha_inf_layer_vs_closed_form_spike_max_abs_log_bayes_factor_difference": (
            spike_path_gap
        ),
        "analytic_component_mass_max_abs_deviation_by_alpha": (
            layer.analytic_normalisation()
        ),
        "interpolated_component_mass": layer.normalisation_report(),
        "realized_type1_error_range": [
            float(min(float(v) for rule in rules for v in type1[rule])),
            float(max(float(v) for rule in rules for v in type1[rule])),
        ],
        "type1_error_by_rule": {
            rule: [float(v) for v in type1[rule]] for rule in rules
        },
    }

    return {
        "description": (
            "Tail-shape and tail-width regime sweep on the Gumbel "
            "pivot.  Every rule is calibrated once on the exact null and applied "
            "unchanged to six shape laws, five inside the Dirichlet "
            "family and one (the released simulator's normalised-uniform tail) "
            "outside it, and four restricted-width laws."
        ),
        "scheme": "gumbel",
        "hierarchy": "shared_delta",
        "seed": int(config.seed),
        "seed_note": (
            "Fresh seed, distinct from 240401245 (clean benchmark), 240401246 "
            "(contamination), 240401247 (contamination quadrature check), "
            "240401248 (deficit sweep) and 240401249 (short-horizon probe)."
        ),
        "config": {
            "vocabulary_size": config.vocabulary_size,
            "horizons": list(config.horizons),
            "alpha_level": config.alpha_level,
            "prior_low": config.prior_low,
            "prior_high": config.prior_high,
            "alpha_grid": [_format_alpha(a) for a in config.alpha_grid],
            "point_alphas": [_format_alpha(a) for a in config.point_alphas],
            "paper_deltas": list(config.paper_deltas),
            "spike_delta": config.spike_delta,
            "n_calibration": config.n_calibration,
            "n_evaluation_null": config.n_evaluation_null,
            "n_evaluation_alternative": config.n_evaluation_alternative,
            "batch_size": config.batch_size,
            "seed": config.seed,
            "bayes_quadrature_nodes": config.bayes_quadrature_nodes,
            "c_nodes": config.c_nodes,
            "max_regret_bootstrap_replicates": (
                config.max_regret_bootstrap_replicates
            ),
            "max_regret_bootstrap_batch_size": (
                config.max_regret_bootstrap_batch_size
            ),
        },
        "frozen_prior": {
            "delta": {
                "family": "uniform",
                "low": config.prior_low,
                "high": config.prior_high,
                "quadrature": "gauss_legendre",
                "nodes": config.bayes_quadrature_nodes,
            },
            "alpha": {
                "family": "uniform on a fixed grid",
                "grid": [_format_alpha(a) for a in layer.alphas],
                "weights": [float(w) for w in layer.alpha_weights],
                "note": (
                    "inf is the equal-tail (spike) limit stated by Li et al. (2025). "
                    "The grid is frozen before simulation and is "
                    "never a function of the regime."
                ),
            },
        },
        "rules": [
            {
                "name": SPIKE_BAYES_RULE,
                "kind": "layer_at_alpha_inf",
                "note": (
                    "the manuscript's existing shared-Delta Gumbel Bayes rule; "
                    "the equal-tail spike component"
                ),
            },
        ]
        + [
            {
                "name": point_alpha_rule_name(a),
                "kind": "layer_at_point_alpha",
                "alpha": _format_alpha(a),
            }
            for a in config.point_alphas
        ]
        + [
            {
                "name": MIXTURE_RULE,
                "kind": "layer_with_frozen_alpha_prior",
                "note": "the method added by this section",
            },
            {
                "name": UNIONTAIL_RULE,
                "kind": "union_tail_shape_and_width_with_frozen_prior",
                "note": (
                    "A document-level union of the full-width Dirichlet shape "
                    "branch and the closed-form width branch at J < K. The "
                    "equal-tail spike is the alpha=inf atom in the shape branch."
                ),
            },
            {
                "name": deficit_sweep.spike_rule_name(config.spike_delta),
                "kind": "point_delta_spike_diagnostic",
            },
            {
                "name": [deficit_sweep.paper_rule_name(d)
                         for d in config.paper_deltas],
                "kind": "paper_least_favorable",
            },
        ],
        "competitor_scope": (
            "Best-rule and regret summaries are relative only to the configured "
            "finite rule menu. Point-alpha competitors are exactly "
            "config.point_alphas plus the alpha=inf spike; alpha-grid atoms used "
            "inside the mixture are not automatically point-rule competitors. "
            "This is not an optimization over all positive alpha or all priors."
        ),
        "tail_width_layer": {
            **union_tail_metadata(tailwidth, methods=[UNIONTAIL_RULE], tokenwise=False),
            "containment_check": {
                "shape_equal_tail_atom_vs_closed_form_spike_max_abs_log_density_difference": (
                    tailwidth.spike_agreement(check_pivots)
                ),
                "note": (
                    "The equal-tail atom lies in the interpolated shape branch; "
                    "this is a numerical containment check, not an exact "
                    "arithmetic identity."
                ),
            },
        },
        "regimes": regime_reports,
        "calibration": {
            "sample": "one exact-null sample shared by every regime",
            "n_calibration": config.n_calibration,
            "note": (
                "The Gumbel pivot null is Uniform(0,1) and does not depend on the "
                "alternative's tail, so one calibration sample legitimately serves "
                "all configured regimes; the evaluation null is a fresh, separate sample."
            ),
        },
        "max_regret_mc_se_method": {
            "estimator": "paired nonparametric document bootstrap",
            "replicates": int(config.max_regret_bootstrap_replicates),
            "batch_size": int(config.max_regret_bootstrap_batch_size),
            "seed_address": [
                int(config.seed),
                int(MAX_REGRET_BOOTSTRAP_SEED_SALT),
                "zero-based regime index",
            ],
            "resampling": (
                "Documents are resampled independently within each generating "
                "tail law and jointly across all rules and horizons.  Each draw "
                "recomputes the best rule within every tail law and then the "
                "maximum regret across tail laws."
            ),
            "conditional_on_realized_calibration": True,
            "excluded_uncertainty": (
                "Calibration-sample, quadrature, finite-menu selection and "
                "model uncertainty are not included."
            ),
        },
        "validation": validation,
        "type2_error": type2,
        "regret": regret,
        "best_rule_by_regime": best_by_regime,
        "max_regret": max_regret_by_horizon,
        "max_regret_ranking": ranking,
        "pooled_max_regret_across_horizons": pooled,
        "saturated_regimes": saturated,
        "cells": cells,
        "markdown_tables": tables,
        "runtime_seconds": float(time.perf_counter() - started),
    }


def resolve_results_dir(requested: Path | None, *, quick: bool) -> Path:
    """Keep smoke-run artifacts separate unless the caller chooses a path."""

    if requested is not None:
        return requested
    return DEFAULT_QUICK_RESULTS_DIR if quick else DEFAULT_RESULTS_DIR


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "artifact directory (default: the benchmark results directory, or its "
            "isolated quick/tail_regime_sweep subdirectory with --quick)"
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="small smoke run: fewer documents, shorter horizons, coarser tables",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = TailSweepConfig()
    if args.quick:
        config = replace(
            config,
            horizons=(25, 50),
            n_calibration=400,
            n_evaluation_null=200,
            n_evaluation_alternative=200,
            batch_size=100,
            c_nodes=20_001,
            max_regret_bootstrap_replicates=200,
        )
    payload = run_tail_sweep(config)
    results_dir = resolve_results_dir(args.results_dir, quick=args.quick)
    results_dir.mkdir(parents=True, exist_ok=True)
    destination = results_dir / "tail_regime_sweep.json"
    destination.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    print(f"wrote {destination}")
    for horizon in config.horizons:
        print()
        print(payload["markdown_tables"][str(horizon)]["type2"])
        print()
        print(payload["markdown_tables"][str(horizon)]["regret"])
    print()
    print("validation:", json.dumps(payload["validation"], indent=1))
    print(f"runtime: {payload['runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()
