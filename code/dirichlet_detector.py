#!/usr/bin/env python3
"""Gumbel Bayes factors under a joint prior on the deficit, tail shape and dilution.

This module is the single owner of the Dirichlet-layer detector used by every
experiment in this study.  It exists because four scripts need the same object at
three different levels of detail:

* ``benchmark_paper_experiment`` needs the shared-``Delta`` log Bayes factor at
  **all** 700 prefixes, plus a running maximum for the anytime rule;
* ``regime_sweep`` and ``tail_regime_sweep`` need it at three horizons only;
* ``benchmark_contamination`` needs it with a third latent axis, the dilution
  rate ``rho``, and its component grid is therefore ``Delta x alpha x rho``.

Model
-----
The leading next-token probability is ``1 - Delta``; the remaining
``K = M - 1`` coordinates carry ``Delta * q`` with
``q ~ Dirichlet(alpha, ..., alpha)``.  Exchangeability leaves a single univariate
transform per ``alpha`` (see :mod:`tail_family`), so the component density is

    f_{Delta,alpha}(r) = r**(Delta/(1-Delta)) + (K/r) psi_alpha(-log r / Delta),

and ``alpha = math.inf`` recovers the equal-tail spike family exactly.  The
Gumbel pivot null is ``Uniform(0,1)``, so ``log f`` *is* the component log
likelihood ratio.  With a dilution rate ``rho`` the component ratio becomes
``rho + (1-rho) f_{Delta,alpha}(r)``, which still has unit null expectation.

Why the gather
--------------
:class:`tail_family.DirichletTailPivot` tabulates ``log psi_alpha(c) + c`` on a
grid that is **uniform in log c**.  A uniform grid means the interpolation index
is arithmetic rather than a binary search, which is what makes evaluating
hundreds of components at every one of 700 tokens for tens of thousands of
documents tractable.  Stacking the per-``alpha`` tables lets one gather serve
every ``alpha`` at once.

Normalisation
-------------
Each discretised component integrates to exactly one *before interpolation*
because the tail quadrature nodes are rescaled so that ``K E[q] = 1``.  The
fast detector linearly interpolates a tabulated transform.  At the default
resolution, high-order numerical quadrature estimates the resulting mass as
about ``1 - 7e-9``.  That is an accuracy diagnostic, not a certified one-sided
integration bound: exact martingale/Ville claims apply to the normalized
non-interpolated model, while the fast interpolated implementation is a
numerically checked approximation.  See
:meth:`DirichletBayesGrid.normalisation_report`.

Depends only on NumPy.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

import tail_family as tf


# On the manuscript's separate deterministic validation grid (20,000 logits
# equally spaced on [-30, 30], paired cyclically with the 96 deficit nodes, for
# every configured alpha), direct non-tabulated quadrature gives worst component
# log-density discrepancies of 9.5e-6 with 20,001 nodes and 9.5e-8 with 200,001.
# These are empirical grid diagnostics, not certified uniform bounds.  Over 700
# tokens the coarser error can accumulate, so the detector uses the finer grid;
# the one-off build cost is well under a minute and is cached.
DEFAULT_C_NODES = 200_001

# Frozen prior on tail concentration.  math.inf is the equal-tail (spike) limit,
# i.e. the specification every non-Dirichlet Gumbel rule in this study assumes.
DEFAULT_ALPHA_GRID: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0, math.inf)


def format_alpha(alpha: float) -> str:
    """Stable string key for an ``alpha`` value, with ``inf`` spelled out."""

    return "inf" if math.isinf(float(alpha)) else f"{float(alpha):g}"


def uniform_alpha_prior(
    alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
) -> tuple[tuple[float, ...], np.ndarray]:
    """The frozen tail-concentration prior: uniform on ``alpha_grid``."""

    grid = tuple(float(a) for a in alpha_grid)
    if not grid:
        raise ValueError("alpha_grid must be nonempty")
    for value in grid:
        if math.isnan(value) or value <= 0.0:
            raise ValueError("alpha_grid values must be positive (math.inf allowed)")
    if len({format_alpha(a) for a in grid}) != len(grid):
        raise ValueError("alpha_grid values must be distinct")
    return grid, np.full(len(grid), 1.0 / len(grid), dtype=float)


def _normalised(weights: Sequence[float] | np.ndarray, size: int, name: str) -> np.ndarray:
    array = np.asarray(weights, dtype=float)
    if array.ndim != 1 or array.size != size:
        raise ValueError(f"{name} must be one-dimensional of length {size}")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    total = float(array.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive total mass")
    return array / total


def _logsumexp_rows(values: np.ndarray) -> np.ndarray:
    """``log sum exp`` over the last axis, safe against all-``-inf`` rows."""

    maximum = np.max(values, axis=-1)
    output = np.full(maximum.shape, -np.inf, dtype=float)
    finite = np.isfinite(maximum)
    if np.any(finite):
        output[finite] = maximum[finite] + np.log(
            np.exp(values[finite] - maximum[finite][..., None]).sum(axis=-1)
        )
    return output


def _shared_latent_paths(
    grid: "DirichletBayesGrid | TailWidthBayesGrid",
    pivots: np.ndarray,
    *,
    horizons: Sequence[int] | None = None,
    running_max: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Shared-latent recursion for any object exposing the component protocol.

    The object must provide ``n_components``, ``log_component_weights`` and
    ``component_log_ratio``.  Factored out so the Dirichlet tail-shape grid and
    the tail-width grid share one implementation.
    """

    values = np.asarray(pivots, dtype=float)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("pivots must be a two-dimensional (rows, tokens) array")
    rows, tokens = values.shape
    if horizons is None:
        record = {t: t for t in range(tokens)}
        n_columns = tokens
    else:
        wanted = [int(h) for h in horizons]
        if not wanted or sorted(set(wanted)) != wanted:
            raise ValueError("horizons must be strictly increasing and distinct")
        if wanted[0] < 1 or wanted[-1] > tokens:
            raise ValueError("horizons must lie in 1..pivots.shape[1]")
        record = {h - 1: i for i, h in enumerate(wanted)}
        n_columns = len(wanted)

    output = np.empty((rows, n_columns), dtype=float)
    maxima = np.empty((rows, n_columns), dtype=float) if running_max else None
    accumulator = np.zeros((rows, grid.n_components), dtype=float)
    weights = grid.log_component_weights[None, :]
    best = np.full(rows, -np.inf) if running_max else None
    for time_index in range(tokens):
        accumulator += grid.component_log_ratio(values[:, time_index])
        if running_max or time_index in record:
            current = _logsumexp_rows(accumulator + weights)
            if running_max:
                np.maximum(best, current, out=best)
            column = record.get(time_index)
            if column is not None:
                output[:, column] = current
                if maxima is not None:
                    maxima[:, column] = best
        # When no running maximum is needed and this token is not reported,
        # the logsumexp is skipped entirely; that is the whole saving.
    return output, maxima


class DirichletBayesGrid:
    """Joint ``(Delta, alpha, rho)`` component grid for the Gumbel pivot.

    Components are flattened in ``(alpha, Delta, rho)`` order with ``alpha``
    slowest and ``rho`` fastest, matching ``np.meshgrid(..., indexing='ij')``.

    Parameters
    ----------
    delta_grid, delta_weights:
        Quadrature approximation to the deficit prior.  Weights are renormalised.
    alpha_grid, alpha_weights:
        Tail-concentration prior.  ``math.inf`` is the equal-tail limit.
    tail_size:
        ``K = M - 1``, the number of non-leading vocabulary coordinates.
    rho_grid, rho_weights:
        Optional dilution prior.  The default ``(0.0,)`` is the clean model.
    """

    def __init__(
        self,
        *,
        delta_grid: Sequence[float] | np.ndarray,
        delta_weights: Sequence[float] | np.ndarray,
        alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
        alpha_weights: Sequence[float] | np.ndarray | None = None,
        tail_size: int,
        allow_large_delta: bool = False,
        rho_grid: Sequence[float] = (0.0,),
        rho_weights: Sequence[float] | np.ndarray | None = None,
        c_nodes: int = DEFAULT_C_NODES,
    ) -> None:
        deltas = np.asarray(delta_grid, dtype=float)
        if deltas.ndim != 1 or deltas.size == 0:
            raise ValueError("delta_grid must be a nonempty one-dimensional array")
        if np.any(~np.isfinite(deltas)) or np.any((deltas <= 0.0) | (deltas >= 1.0)):
            raise ValueError("delta_grid values must lie strictly in (0, 1)")
        if np.any(deltas > 0.5 + 1e-12) and not allow_large_delta:
            raise ValueError(
                "delta_grid values above 0.5 require allow_large_delta=True: "
                "the designated leading coordinate is then no longer the "
                "largest under every tail draw, so Delta stops being a "
                "top-probability deficit"
            )
        if not isinstance(tail_size, (int, np.integer)) or int(tail_size) < 1:
            raise ValueError("tail_size must be a positive integer")

        alphas = tuple(float(a) for a in alpha_grid)
        if not alphas:
            raise ValueError("alpha_grid must be nonempty")
        for value in alphas:
            if math.isnan(value) or value <= 0.0:
                raise ValueError("alpha_grid values must be positive (math.inf allowed)")
        if len({format_alpha(a) for a in alphas}) != len(alphas):
            raise ValueError("alpha_grid values must be distinct")

        rhos = np.asarray(rho_grid, dtype=float)
        if rhos.ndim != 1 or rhos.size == 0:
            raise ValueError("rho_grid must be a nonempty one-dimensional array")
        if np.any(~np.isfinite(rhos)) or np.any((rhos < 0.0) | (rhos > 1.0)):
            raise ValueError("rho_grid values must lie in [0, 1]")

        self.deltas = deltas.copy()
        self.delta_weights = _normalised(delta_weights, deltas.size, "delta_weights")
        self.alphas = alphas
        if alpha_weights is None:
            self.alpha_weights = np.full(len(alphas), 1.0 / len(alphas), dtype=float)
        else:
            self.alpha_weights = _normalised(alpha_weights, len(alphas), "alpha_weights")
        self.rhos = rhos.copy()
        if rho_weights is None:
            self.rho_weights = np.full(rhos.size, 1.0 / rhos.size, dtype=float)
        else:
            self.rho_weights = _normalised(rho_weights, rhos.size, "rho_weights")
        self.tail_size = int(tail_size)
        self.c_nodes = int(c_nodes)

        self.pivots = tuple(
            tf.dirichlet_tail_pivot(a, self.tail_size, c_nodes=self.c_nodes)
            for a in self.alphas
        )
        grid = self.pivots[0].log_c_grid
        for pivot in self.pivots[1:]:
            if not np.array_equal(pivot.log_c_grid, grid):
                raise ValueError("all alpha tables must share one log-c grid")
        self._log_c_min = float(grid[0])
        self._inv_step = 1.0 / float(grid[1] - grid[0])
        self._n_grid = int(grid.size)
        self._table = np.stack([pivot._psi_shifted for pivot in self.pivots])

        self._top_exponent = self.deltas / (1.0 - self.deltas)
        self._inverse_delta = 1.0 / self.deltas
        self._log_tail_size = math.log(self.tail_size)
        self.alpha_index = {format_alpha(a): i for i, a in enumerate(self.alphas)}

        with np.errstate(divide="ignore"):
            self._log_rho = np.log(self.rhos)
            self._log_one_minus_rho = np.log1p(-self.rhos)
        self._clean = bool(self.rhos.size == 1 and self.rhos[0] == 0.0)

        # log prior weight per flattened component, (alpha, Delta, rho).
        with np.errstate(divide="ignore"):
            log_joint = (
                np.log(self.alpha_weights)[:, None, None]
                + np.log(self.delta_weights)[None, :, None]
                + np.log(self.rho_weights)[None, None, :]
            )
        self.log_component_weights = log_joint.reshape(-1)
        self.component_alpha = np.repeat(
            np.arange(len(self.alphas)), self.deltas.size * self.rhos.size
        )
        self.component_delta = np.tile(
            np.repeat(self.deltas, self.rhos.size), len(self.alphas)
        )
        self.component_rho = np.tile(
            self.rhos, len(self.alphas) * self.deltas.size
        )

    # -- shape bookkeeping ----------------------------------------------------

    @property
    def n_components(self) -> int:
        return len(self.alphas) * self.deltas.size * self.rhos.size

    def component_slice_for_alpha(self, alpha: float) -> slice:
        """Flat slice of the components belonging to one ``alpha``."""

        index = self.alpha_index[format_alpha(alpha)]
        block = self.deltas.size * self.rhos.size
        return slice(index * block, (index + 1) * block)

    # -- component likelihoods ------------------------------------------------

    def component_log_density(self, pivots: np.ndarray) -> np.ndarray:
        """``log f_{Delta,alpha}(r)`` with shape ``(rows, alpha * Delta)``.

        Exact zeros are clamped to the smallest positive double: they occur with
        probability ``2**-53`` per draw, and clamping keeps ``c`` finite and
        inside the tabulated range rather than producing ``inf - inf``.
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
        # (alpha, rows, Delta) -> (rows, alpha * Delta)
        stacked = np.logaddexp(log_top, log_tail)
        return np.ascontiguousarray(stacked.transpose(1, 0, 2)).reshape(r.shape[0], -1)

    def component_log_ratio(self, pivots: np.ndarray) -> np.ndarray:
        """Component log likelihood ratios, shape ``(rows, n_components)``.

        The Gumbel null density is one, so with no dilution this is exactly
        :meth:`component_log_density`.  With dilution the component ratio is
        ``rho + (1-rho) f``, whose null expectation is still one.
        """

        log_density = self.component_log_density(pivots)
        if self._clean:
            return log_density
        rows = log_density.shape[0]
        diluted = np.logaddexp(
            self._log_rho[None, None, :],
            self._log_one_minus_rho[None, None, :] + log_density[:, :, None],
        )
        return diluted.reshape(rows, -1)

    # -- sequential Bayes factors --------------------------------------------

    def shared_paths(
        self,
        pivots: np.ndarray,
        *,
        horizons: Sequence[int] | None = None,
        running_max: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Shared-latent log Bayes factor: average the product over the prior.

        Returns ``(values, maxima)``.  If ``horizons`` is ``None`` the returned
        array has one column per prefix ``1..pivots.shape[1]``; otherwise one
        column per requested horizon.  ``maxima`` is the running maximum of the
        log Bayes factor up to each reported column, used by the anytime rule,
        and is ``None`` unless ``running_max`` is set.

        The running maximum is accumulated over **every** token, not only the
        reported ones, so it is the true supremum of the process up to that
        horizon even when ``horizons`` is sparse.
        """

        return _shared_latent_paths(
            self, pivots, horizons=horizons, running_max=running_max
        )

    def tokenwise_lookup(
        self,
        *,
        logit_limit: float = 30.0,
        size: int = 80_001,
    ) -> "DirichletTokenwiseLookup":
        """Tabulate the tokenwise prior-predictive log likelihood ratio.

        Under the tokenwise hierarchy a fresh ``(Delta, alpha, rho)`` is drawn at
        every position, so the per-token increment is a fixed scalar function of
        the pivot and can be tabulated once.
        """

        return DirichletTokenwiseLookup(self, logit_limit=logit_limit, size=size)

    # -- diagnostics ----------------------------------------------------------

    def spike_agreement(self, pivots: np.ndarray) -> float:
        """Worst ``|log f_{Delta,inf} - log f^{sp}_Delta|`` over the Delta nodes.

        The ``alpha = inf`` member must reproduce the closed-form equal-tail
        spike density, which is the density every other Gumbel rule in this study
        evaluates.  Both sides are read at the same Delta nodes.
        """

        key = "inf"
        if key not in self.alpha_index:
            raise ValueError("the alpha grid does not contain the equal-tail limit")
        r = np.asarray(pivots, dtype=float).reshape(-1)
        density = self.component_log_density(r).reshape(
            r.size, len(self.alphas), self.deltas.size
        )
        layer = density[:, self.alpha_index[key], :]
        log_r = np.log(np.maximum(r, np.finfo(float).tiny))[:, None]
        closed_form = np.logaddexp(
            self._top_exponent[None, :] * log_r,
            self._log_tail_size
            + ((self.tail_size * self._inverse_delta) - 1.0)[None, :] * log_r,
        )
        return float(np.max(np.abs(layer - closed_form)))

    def analytic_normalisation(self) -> dict[str, float]:
        """Analytic component mass ``(1-Delta) + Delta K E[q]``, per alpha."""

        report: dict[str, float] = {}
        for alpha, pivot in zip(self.alphas, self.pivots):
            mass = pivot.normalisation(self.deltas)
            report[format_alpha(alpha)] = float(np.max(np.abs(mass - 1.0)))
        return report

    def normalisation_report(
        self, *, panels: int = 2_000, gauss_nodes: int = 40, limit: float = 40.0
    ) -> dict[str, object]:
        """Numerical quadrature mass of the interpolated component density.

        Integrated in ``z = logit(r)`` so both the ``r -> 0`` end and the very
        sharp ``r -> 1`` tail spike are resolved, at exactly the Delta nodes the
        detector uses.  This finite, truncated floating-point calculation is a
        diagnostic only.  In particular, an estimated mass at most one is not a
        proof that the true interpolant is a sub-probability density.
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
            density = np.exp(
                self.component_log_density(sigmoid).reshape(
                    z.size, len(self.alphas), self.deltas.size
                )
            )
            mass += np.einsum("z,zaj->aj", quadrature * jacobian, density)
        deviation = mass - 1.0
        per_alpha = {
            format_alpha(alpha): float(
                deviation[index][np.argmax(np.abs(deviation[index]))]
            )
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

    def prior_fingerprint(self) -> dict[str, object]:
        """A small numeric signature of the frozen prior, safe to record."""

        return {
            "delta_nodes": int(self.deltas.size),
            "delta_node_sum": float(self.deltas.sum()),
            "delta_weight_sum": float(self.delta_weights.sum()),
            "delta_node_min": float(self.deltas.min()),
            "delta_node_max": float(self.deltas.max()),
            "alpha_grid": [format_alpha(a) for a in self.alphas],
            "alpha_weights": [float(w) for w in self.alpha_weights],
            "rho_grid": [float(r) for r in self.rhos],
            "rho_weights": [float(w) for w in self.rho_weights],
            "rho_mean": float(np.dot(self.rho_weights, self.rhos)),
            "n_components": int(self.n_components),
            "c_nodes": int(self.c_nodes),
        }


class DirichletTokenwiseLookup:
    """Interpolated tokenwise prior-predictive log likelihood ratio.

    ``log int lambda_theta(r) Pi(d theta)`` as a function of the pivot alone,
    tabulated on a uniform ``logit(r)`` grid.  This mirrors the existing
    spike-family tokenwise lookup, so the two hierarchies are computed the same
    way and differ only in the component family.

    Any grid exposing ``log_component_weights`` and ``component_log_ratio``
    works here, so the union tail uses it unchanged.  Note what that rule then
    is: the tokenwise hierarchy integrates each token separately, so it averages
    over the tail width at every position and cannot accumulate evidence about a
    document-level width.  Its width block is inert by construction, which is
    the point the tokenwise arm is reported to demonstrate.
    """

    def __init__(
        self,
        grid: "DirichletBayesGrid | UnionTailBayesGrid",
        *,
        logit_limit: float = 30.0,
        size: int = 80_001,
    ) -> None:
        if not (math.isfinite(logit_limit) and logit_limit > 0.0):
            raise ValueError("logit_limit must be positive and finite")
        if not isinstance(size, (int, np.integer)) or int(size) < 1_001:
            raise ValueError("size must be an integer of at least 1001")
        self.grid = grid
        self.logits = np.linspace(-float(logit_limit), float(logit_limit), int(size))
        values = np.empty(self.logits.size, dtype=float)
        weights = grid.log_component_weights[None, :]
        chunk = 4_000
        for start in range(0, self.logits.size, chunk):
            stop = min(start + chunk, self.logits.size)
            z = self.logits[start:stop]
            r = 1.0 / (1.0 + np.exp(-z))
            values[start:stop] = _logsumexp_rows(
                grid.component_log_ratio(r) + weights
            )
        self.log_ratio = values

    def __call__(self, pivots: np.ndarray) -> np.ndarray:
        array = np.asarray(pivots, dtype=float)
        tiny = np.finfo(float).tiny
        epsilon = np.finfo(float).eps
        clipped = np.clip(array, tiny, 1.0 - epsilon)
        z = np.log(clipped) - np.log1p(-clipped)
        return np.interp(
            z,
            self.logits,
            self.log_ratio,
            left=self.log_ratio[0],
            right=self.log_ratio[-1],
        )

    def _direct_log_ratio_at_logits(self, logits: np.ndarray) -> np.ndarray:
        """Evaluate the frozen component mixture without outer interpolation."""

        values = np.asarray(logits, dtype=float)
        output = np.empty_like(values)
        weights = self.grid.log_component_weights[None, :]
        chunk = 4_000
        for start in range(0, values.size, chunk):
            stop = min(start + chunk, values.size)
            z = values[start:stop]
            pivots = 1.0 / (1.0 + np.exp(-z))
            output[start:stop] = _logsumexp_rows(
                self.grid.component_log_ratio(pivots) + weights
            )
        return output

    def validation_report(
        self, *, panels: int = 2_000, gauss_nodes: int = 40, limit: float = 40.0
    ) -> dict[str, object]:
        """Reproducible diagnostics for the outer tokenwise interpolation.

        Midpoint error is measured against the same frozen component mixture
        evaluated without the outer lookup.  The inner component transform may
        itself be interpolated, so this report isolates only the additional
        tokenwise interpolation layer.  Neither diagnostic is a certified bound.
        """

        midpoints = 0.5 * (self.logits[:-1] + self.logits[1:])
        interpolated = np.interp(midpoints, self.logits, self.log_ratio)
        direct = self._direct_log_ratio_at_logits(midpoints)
        error = np.abs(interpolated - direct)
        worst = int(np.argmax(error))
        mass = self.normalisation(
            panels=panels, gauss_nodes=gauss_nodes, limit=limit
        )
        return {
            "diagnostic_only": True,
            "certified_one_sided_bound": False,
            "midpoint_log_ratio": {
                "intervals_checked": int(midpoints.size),
                "max_abs_error": float(error[worst]),
                "argmax_logit": float(midpoints[worst]),
                "reference": (
                    "direct evaluation of the same frozen component mixture; "
                    "this isolates outer interpolation error"
                ),
            },
            "interpolated_numerator_mass": {
                "mass": mass,
                "mass_minus_one": mass - 1.0,
                "logit_quadrature": {
                    "panels": int(panels),
                    "gauss_nodes": int(gauss_nodes),
                    "limit": float(limit),
                },
            },
        }

    def normalisation(
        self, *, panels: int = 2_000, gauss_nodes: int = 40, limit: float = 40.0
    ) -> float:
        """Numerically estimate the mass of the interpolated tokenwise numerator.

        The return value is an accuracy diagnostic, not a certified upper or
        lower bound on the integral.
        """

        nodes, weights = np.polynomial.legendre.leggauss(int(gauss_nodes))
        edges = np.linspace(-float(limit), float(limit), int(panels) + 1)
        half = 0.5 * (edges[1] - edges[0])
        centres = 0.5 * (edges[:-1] + edges[1:])
        total = 0.0
        block = 256
        for start in range(0, centres.size, block):
            stop = min(start + block, centres.size)
            z = (centres[start:stop, None] + half * nodes[None, :]).reshape(-1)
            quadrature = np.tile(weights * half, stop - start)
            sigmoid = 1.0 / (1.0 + np.exp(-z))
            jacobian = sigmoid * (1.0 - sigmoid)
            total += float(np.dot(quadrature * jacobian, np.exp(self(sigmoid))))
        return total


def gauss_legendre_delta_grid(
    low: float, high: float, nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss--Legendre nodes and weights for a uniform prior on ``(low, high)``.

    Identical to the grid the spike-family detectors use, so the Dirichlet rule
    and the existing rules integrate the same deficit prior the same way.
    """

    if not (0.0 <= low < high < 1.0):
        raise ValueError("require 0 <= low < high < 1")
    if not isinstance(nodes, (int, np.integer)) or int(nodes) < 2:
        raise ValueError("nodes must be an integer of at least 2")
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(int(nodes))
    deltas = low + 0.5 * (raw_nodes + 1.0) * (high - low)
    return deltas, 0.5 * raw_weights


__all__ = [
    "DEFAULT_ALPHA_GRID",
    "DEFAULT_C_NODES",
    "DirichletBayesGrid",
    "DirichletTokenwiseLookup",
    "format_alpha",
    "gauss_legendre_delta_grid",
    "uniform_alpha_prior",
]


# ---------------------------------------------------------------------------
# Latent tail width
# ---------------------------------------------------------------------------

# Base of the frozen tail-width ladder.  The grid is 1, 4, 16, ..., K, so it is
# scale free: it spans the sparsest possible tail up to the equal tail over the
# whole vocabulary whatever K is, and it always contains K itself.
DEFAULT_TAIL_WIDTH_BASE = 4


def dyadic_tail_width_grid(
    tail_size: int, base: int = DEFAULT_TAIL_WIDTH_BASE
) -> tuple[int, ...]:
    """``(1, base, base**2, ..., tail_size)`` -- the frozen tail-width ladder.

    The final atom is always ``tail_size = K = M - 1``, the equal-tail spike
    over the full vocabulary, so the enlarged family contains the tail
    specification every other Gumbel rule in this study assumes.
    """

    if not isinstance(tail_size, (int, np.integer)) or int(tail_size) < 1:
        raise ValueError("tail_size must be a positive integer")
    if not isinstance(base, (int, np.integer)) or int(base) < 2:
        raise ValueError("base must be an integer of at least 2")
    tail_size = int(tail_size)
    widths: list[int] = []
    value = 1
    while value < tail_size:
        widths.append(value)
        value *= int(base)
    widths.append(tail_size)
    return tuple(widths)


def uniform_tail_width_prior(
    tail_width_grid: Sequence[int],
) -> tuple[tuple[int, ...], np.ndarray]:
    """The frozen tail-width prior: uniform on ``tail_width_grid``."""

    grid = tuple(int(j) for j in tail_width_grid)
    if not grid:
        raise ValueError("tail_width_grid must be nonempty")
    if any(j < 1 for j in grid):
        raise ValueError("tail_width_grid values must be positive integers")
    if len(set(grid)) != len(grid):
        raise ValueError("tail_width_grid values must be distinct")
    return grid, np.full(len(grid), 1.0 / len(grid), dtype=float)


class TailWidthBayesGrid:
    """Joint ``(J, Delta, rho)`` component grid for the Gumbel pivot.

    ``J`` is the number of live tail coordinates.  Conditional on ``(Delta, J)``
    the working NTP vector is ``(1-Delta, Delta/J, ..., Delta/J)`` with ``J``
    tail entries, whose exact Gumbel pivot density is

        f_{Delta,J}(r) = r**(Delta/(1-Delta)) + J * r**(J/Delta - 1).

    Setting ``J = tail_size`` recovers the equal-tail spike over the whole
    vocabulary, so this family *contains* the component every other Gumbel rule
    in this study uses; ``J = 1`` is the sparsest tail at that deficit.

    The density is closed form.  Unlike the Dirichlet tail-shape layer there is
    no transform table and no interpolation, so each component integrates to one
    analytically and the test-martingale statement applies to the evaluated
    density rather than to a numerical approximation of it.

    Components are flattened in ``(J, Delta, rho)`` order with ``J`` slowest and
    ``rho`` fastest, matching the ``(alpha, Delta, rho)`` convention of
    :class:`DirichletBayesGrid`.
    """

    def __init__(
        self,
        *,
        delta_grid: Sequence[float] | np.ndarray,
        delta_weights: Sequence[float] | np.ndarray,
        tail_size: int,
        allow_large_delta: bool = False,
        tail_width_grid: Sequence[int] | None = None,
        tail_width_weights: Sequence[float] | np.ndarray | None = None,
        rho_grid: Sequence[float] = (0.0,),
        rho_weights: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        deltas = np.asarray(delta_grid, dtype=float)
        if deltas.ndim != 1 or deltas.size == 0:
            raise ValueError("delta_grid must be a nonempty one-dimensional array")
        if np.any(~np.isfinite(deltas)) or np.any((deltas <= 0.0) | (deltas >= 1.0)):
            raise ValueError("delta_grid values must lie strictly in (0, 1)")
        if np.any(deltas > 0.5 + 1e-12) and not allow_large_delta:
            raise ValueError(
                "delta_grid values above 0.5 require allow_large_delta=True: "
                "at tail width J the residual coordinates carry Delta/J each, "
                "which exceeds the leading 1-Delta once Delta > J/(1+J)"
            )
        if not isinstance(tail_size, (int, np.integer)) or int(tail_size) < 1:
            raise ValueError("tail_size must be a positive integer")
        self.tail_size = int(tail_size)

        if tail_width_grid is None:
            tail_width_grid = dyadic_tail_width_grid(self.tail_size)
        widths, uniform = uniform_tail_width_prior(tail_width_grid)
        if any(j > self.tail_size for j in widths):
            raise ValueError("tail_width_grid values must not exceed tail_size")
        self.tail_widths = widths

        rhos = np.asarray(rho_grid, dtype=float)
        if rhos.ndim != 1 or rhos.size == 0:
            raise ValueError("rho_grid must be a nonempty one-dimensional array")
        if np.any(~np.isfinite(rhos)) or np.any((rhos < 0.0) | (rhos > 1.0)):
            raise ValueError("rho_grid values must lie in [0, 1]")

        self.deltas = deltas.copy()
        self.delta_weights = _normalised(delta_weights, deltas.size, "delta_weights")
        if tail_width_weights is None:
            self.tail_width_weights = uniform
        else:
            self.tail_width_weights = _normalised(
                tail_width_weights, len(widths), "tail_width_weights"
            )
        self.rhos = rhos.copy()
        if rho_weights is None:
            self.rho_weights = np.full(rhos.size, 1.0 / rhos.size, dtype=float)
        else:
            self.rho_weights = _normalised(rho_weights, rhos.size, "rho_weights")

        self._top_exponent = self.deltas / (1.0 - self.deltas)
        self._inverse_delta = 1.0 / self.deltas
        # (J, Delta): log J and the tail exponent J/Delta - 1.
        widths_array = np.asarray(self.tail_widths, dtype=float)
        self._log_width = np.log(widths_array)[:, None]
        self._tail_exponent = widths_array[:, None] * self._inverse_delta[None, :] - 1.0
        self.tail_width_index = {int(j): i for i, j in enumerate(self.tail_widths)}

        with np.errstate(divide="ignore"):
            self._log_rho = np.log(self.rhos)
            self._log_one_minus_rho = np.log1p(-self.rhos)
        self._clean = bool(self.rhos.size == 1 and self.rhos[0] == 0.0)

        with np.errstate(divide="ignore"):
            log_joint = (
                np.log(self.tail_width_weights)[:, None, None]
                + np.log(self.delta_weights)[None, :, None]
                + np.log(self.rho_weights)[None, None, :]
            )
        self.log_component_weights = log_joint.reshape(-1)
        self.component_tail_width = np.repeat(
            np.asarray(self.tail_widths), self.deltas.size * self.rhos.size
        )
        self.component_delta = np.tile(
            np.repeat(self.deltas, self.rhos.size), len(self.tail_widths)
        )
        self.component_rho = np.tile(
            self.rhos, len(self.tail_widths) * self.deltas.size
        )

    # -- shape bookkeeping ----------------------------------------------------

    @property
    def n_components(self) -> int:
        return len(self.tail_widths) * self.deltas.size * self.rhos.size

    def component_slice_for_tail_width(self, tail_width: int) -> slice:
        """Flat slice of the components belonging to one tail width."""

        index = self.tail_width_index[int(tail_width)]
        block = self.deltas.size * self.rhos.size
        return slice(index * block, (index + 1) * block)

    # -- component likelihoods ------------------------------------------------

    def component_log_density(self, pivots: np.ndarray) -> np.ndarray:
        """``log f_{Delta,J}(r)`` with shape ``(rows, J * Delta)``.

        Exact zeros are clamped to the smallest positive double, matching
        :meth:`DirichletBayesGrid.component_log_density`.
        """

        r = np.asarray(pivots, dtype=float).reshape(-1, 1)
        if np.any(~np.isfinite(r)) or np.any((r < 0.0) | (r > 1.0)):
            raise ValueError("Gumbel pivots must lie in [0, 1]")
        r = np.maximum(r, np.finfo(float).tiny)
        log_r = np.log(r)
        log_top = self._top_exponent[None, None, :] * log_r[None, :, :]
        log_tail = (
            self._log_width[:, None, :]
            + self._tail_exponent[:, None, :] * log_r[None, :, :]
        )
        # (J, rows, Delta) -> (rows, J * Delta)
        stacked = np.logaddexp(log_top, log_tail)
        return np.ascontiguousarray(stacked.transpose(1, 0, 2)).reshape(r.shape[0], -1)

    def component_log_ratio(self, pivots: np.ndarray) -> np.ndarray:
        """Component log likelihood ratios, shape ``(rows, n_components)``."""

        log_density = self.component_log_density(pivots)
        if self._clean:
            return log_density
        rows = log_density.shape[0]
        diluted = np.logaddexp(
            self._log_rho[None, None, :],
            self._log_one_minus_rho[None, None, :] + log_density[:, :, None],
        )
        return diluted.reshape(rows, -1)

    # -- sequential Bayes factors --------------------------------------------

    def shared_paths(
        self,
        pivots: np.ndarray,
        *,
        horizons: Sequence[int] | None = None,
        running_max: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Shared-latent log Bayes factor; see :func:`_shared_latent_paths`."""

        return _shared_latent_paths(
            self, pivots, horizons=horizons, running_max=running_max
        )

    # -- diagnostics ----------------------------------------------------------

    def spike_agreement(self, pivots: np.ndarray) -> float:
        """Worst ``|log f_{Delta,K} - log f^{sp}_Delta|`` over the Delta nodes.

        The widest atom must reproduce the closed-form equal-tail spike density
        that every other Gumbel rule evaluates.  Both sides are computed here in
        closed form, so this is an arithmetic identity check rather than an
        interpolation-accuracy check, and it should return zero.
        """

        if self.tail_size not in self.tail_width_index:
            raise ValueError("the tail-width grid does not contain the full tail")
        r = np.asarray(pivots, dtype=float).reshape(-1)
        density = self.component_log_density(r).reshape(
            r.size, len(self.tail_widths), self.deltas.size
        )
        layer = density[:, self.tail_width_index[self.tail_size], :]
        log_r = np.log(np.maximum(r, np.finfo(float).tiny))[:, None]
        closed_form = np.logaddexp(
            self._top_exponent[None, :] * log_r,
            math.log(self.tail_size)
            + ((self.tail_size * self._inverse_delta) - 1.0)[None, :] * log_r,
        )
        return float(np.max(np.abs(layer - closed_form)))

    def analytic_normalisation(self) -> dict[str, float]:
        """Analytic component mass ``(1-Delta) + J * (Delta/J) = 1``, per width.

        Each component is a finite sum of exact Beta densities, so the mass is
        one identically; the value returned is the floating-point deviation.
        """

        report: dict[str, float] = {}
        for width in self.tail_widths:
            mass = (1.0 - self.deltas) + width * (self.deltas / width)
            report[str(int(width))] = float(np.max(np.abs(mass - 1.0)))
        return report

    def prior_fingerprint(self) -> dict[str, object]:
        """A small numeric signature of the frozen prior, safe to record."""

        return {
            "delta_nodes": int(self.deltas.size),
            "delta_node_sum": float(self.deltas.sum()),
            "delta_weight_sum": float(self.delta_weights.sum()),
            "delta_node_min": float(self.deltas.min()),
            "delta_node_max": float(self.deltas.max()),
            "tail_width_grid": [int(j) for j in self.tail_widths],
            "tail_width_weights": [float(w) for w in self.tail_width_weights],
            "tail_size": int(self.tail_size),
            "rho_grid": [float(r) for r in self.rhos],
            "rho_weights": [float(w) for w in self.rho_weights],
            "rho_mean": float(np.dot(self.rho_weights, self.rhos)),
            "n_components": int(self.n_components),
            "closed_form": True,
        }


# ---------------------------------------------------------------------------
# Union of the two tail enlargements
# ---------------------------------------------------------------------------

# Prior mass on the Dirichlet tail-shape block.  The remainder goes to the
# tail-width block.  Half and half, matching the convention the contamination
# prior already uses in placing .5 on the incumbent clean model.
#
# The equal-tail synthetic designs and exploratory narrow-width estimates on
# released outputs motivate retaining both components.  The default is a fixed
# prior choice, not an optimized weight.  The current synthetic regime sweep
# evaluates w = .5; temperature-matched AUC sensitivity is reported separately.
DEFAULT_UNION_DIRICHLET_WEIGHT = 0.5


class UnionTailTokenwiseLookup:
    """Tokenwise union tail with a DOCUMENT-LEVEL tail state.

    Section 3 keeps the tail state ``B`` at the document level in every
    hierarchy, and Figure 1 draws it outside the token plate even in the
    tokenwise panel.  That model is

        w * prod_t m_shape(Y_t)  +  (1 - w) * prod_t m_width(Y_t),

    which mixes the two BRANCH PRODUCTS.  Collapsing the whole union at every
    token instead gives

        prod_t { w m_shape(Y_t) + (1 - w) m_width(Y_t) },

    which is a different model -- it redraws ``B`` at each position.  The two
    disagree by up to 26 nats on 200-token paths, so this is not a numerical
    detail: the second cannot accumulate evidence about which branch the
    document is in, which is the whole content of a document-level state.

    Within each branch the deficit and shape or width are still integrated per
    token, so this is the tokenwise hierarchy in every coordinate except ``B``.
    """

    def __init__(
        self,
        grid: "UnionTailBayesGrid",
        *,
        logit_limit: float = 30.0,
        size: int = 80_001,
    ) -> None:
        self.grid = grid
        weights = np.asarray(grid.log_component_weights, dtype=float)
        self.block_log_weights: list[float] = []
        self.block_tables: list[np.ndarray] = []
        self.logits = np.linspace(-float(logit_limit), float(logit_limit), int(size))
        for index in range(2):
            block = grid.block_slice(index)
            total = float(_logsumexp_rows(weights[block][None, :])[0])
            self.block_log_weights.append(total)
            table = np.zeros(self.logits.size, dtype=float)
            if not math.isfinite(total):
                # w = 0 or w = 1 kills a branch outright.  Renormalizing inside
                # a branch of zero mass is 0/0, so leave its table at zero and
                # let the -inf document weight remove it from the mixture; the
                # rule is then exactly the surviving branch.
                self.block_tables.append(table)
                continue
            chunk = 4_000
            for start in range(0, self.logits.size, chunk):
                stop = min(start + chunk, self.logits.size)
                r = 1.0 / (1.0 + np.exp(-self.logits[start:stop]))
                # renormalize inside the branch: the branch weight is carried
                # once, at the document level, not once per token.
                table[start:stop] = _logsumexp_rows(
                    grid.component_log_ratio(r)[:, block]
                    + weights[block][None, :]
                ) - total
            self.block_tables.append(table)

    def _interpolate(self, table: np.ndarray, pivots: np.ndarray) -> np.ndarray:
        array = np.asarray(pivots, dtype=float)
        tiny = np.finfo(float).tiny
        epsilon = np.finfo(float).eps
        clipped = np.clip(array, tiny, 1.0 - epsilon)
        z = np.log(clipped) - np.log1p(-clipped)
        return np.interp(z, self.logits, table)

    def paths(self, pivots: np.ndarray) -> np.ndarray:
        """Document log Bayes factor at every prefix, shape (rows, horizon)."""

        values = np.asarray(pivots, dtype=float)
        terms = []
        for log_weight, table in zip(self.block_log_weights, self.block_tables):
            terms.append(
                log_weight + np.cumsum(self._interpolate(table, values), axis=1)
            )
        return np.logaddexp(terms[0], terms[1])

    def validation_report(self) -> dict[str, float]:
        """Largest interpolation error of each branch table, on a check grid."""

        probe = 1.0 / (1.0 + np.exp(-np.linspace(-25.0, 25.0, 4_001)))
        weights = np.asarray(self.grid.log_component_weights, dtype=float)
        report: dict[str, float] = {}
        for index, (total, table) in enumerate(
            zip(self.block_log_weights, self.block_tables)
        ):
            if not math.isfinite(total):
                report[f"block_{index}_max_abs_interpolation_error"] = 0.0
                continue
            block = self.grid.block_slice(index)
            direct = _logsumexp_rows(
                self.grid.component_log_ratio(probe)[:, block]
                + weights[block][None, :]
            ) - total
            report[f"block_{index}_max_abs_interpolation_error"] = float(
                np.max(np.abs(self._interpolate(table, probe) - direct))
            )
        return report


class UnionTailBayesGrid:
    """Union of the Dirichlet tail-shape and tail-width families.

    Two blocks share one deficit prior and one dilution prior:

    * weight ``w`` on :class:`DirichletBayesGrid` at the full tail width
      ``J = K``, which spreads the residual mass over every non-leading
      coordinate as ``Dirichlet(alpha)``;
    * weight ``1 - w`` on :class:`TailWidthBayesGrid` restricted to ``J < K``,
      which keeps the tail equal but lets the number of live coordinates vary.

    Both parents are exact sub-models, and so is the equal-tail spike, which is
    the ``(alpha = inf, J = K)`` atom inside the first block.  The full-width
    atom is deliberately excluded from the second block so it is not counted
    twice.

    Union rather than product.  ``alpha`` and ``J`` both describe tail
    concentration, so a joint ``(Delta, alpha, J)`` grid mostly manufactures
    duplicate atoms: measured at production resolution it is worse than either
    parent, because it pays ``log 6`` extra nats of dilution for shapes the
    union already reaches.  The union pays only ``log 2``.
    """

    def __init__(
        self,
        *,
        delta_grid: Sequence[float] | np.ndarray,
        delta_weights: Sequence[float] | np.ndarray,
        tail_size: int,
        allow_large_delta: bool = False,
        dirichlet_weight: float = DEFAULT_UNION_DIRICHLET_WEIGHT,
        alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
        tail_width_grid: Sequence[int] | None = None,
        rho_grid: Sequence[float] = (0.0,),
        rho_weights: Sequence[float] | np.ndarray | None = None,
        c_nodes: int = DEFAULT_C_NODES,
    ) -> None:
        weight = float(dirichlet_weight)
        if not (math.isfinite(weight) and 0.0 <= weight <= 1.0):
            raise ValueError("dirichlet_weight must be finite and lie in [0, 1]")
        if not isinstance(tail_size, (int, np.integer)) or int(tail_size) < 1:
            raise ValueError("tail_size must be a positive integer")
        self.tail_size = int(tail_size)
        self.dirichlet_weight = weight

        if tail_width_grid is None:
            widths = tuple(
                j for j in dyadic_tail_width_grid(self.tail_size) if j != self.tail_size
            )
        else:
            widths, _ = uniform_tail_width_prior(tail_width_grid)
            if any(j >= self.tail_size for j in widths):
                raise ValueError(
                    "tail_width_grid must hold widths strictly below tail_size; "
                    "the full-width atom already lies in the Dirichlet block"
                )
        if not widths:
            raise ValueError("the tail-width block needs at least one width below tail_size")
        self.tail_widths = widths

        shared = dict(
            delta_grid=delta_grid,
            delta_weights=delta_weights,
            rho_grid=rho_grid,
            rho_weights=rho_weights,
        )
        self.dirichlet_block = DirichletBayesGrid(
            allow_large_delta=allow_large_delta,
            alpha_grid=alpha_grid, tail_size=self.tail_size, c_nodes=c_nodes, **shared
        )
        # Closed form, so the tail-width block needs no transform table.
        self.tail_width_block = TailWidthBayesGrid(
            allow_large_delta=allow_large_delta,
            tail_size=self.tail_size, tail_width_grid=widths, **shared
        )
        self.blocks = (self.dirichlet_block, self.tail_width_block)
        self.block_weights = (weight, 1.0 - weight)

        with np.errstate(divide="ignore"):
            self.log_component_weights = np.concatenate(
                [
                    block.log_component_weights + math.log(bw)
                    if bw > 0.0
                    else np.full(block.n_components, -np.inf)
                    for block, bw in zip(self.blocks, self.block_weights)
                ]
            )
        total = float(np.exp(_logsumexp_rows(self.log_component_weights[None, :]))[0])
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"union component weights sum to {total!r}, not one")

        self.deltas = self.dirichlet_block.deltas
        self.delta_weights = self.dirichlet_block.delta_weights
        self.alphas = self.dirichlet_block.alphas
        self.rhos = self.dirichlet_block.rhos
        self.rho_weights = self.dirichlet_block.rho_weights
        self.tail_width_weights = self.tail_width_block.tail_width_weights
        self.c_nodes = int(c_nodes)

        # Per-component coordinates, concatenated in block order so they line up
        # with component_log_ratio and log_component_weights.  Callers that
        # select components by dilution rate -- the clean-rho reading of the
        # contamination study, for instance -- index through these.
        self.component_delta = np.concatenate(
            [block.component_delta for block in self.blocks]
        )
        self.component_rho = np.concatenate(
            [block.component_rho for block in self.blocks]
        )

    # -- shape bookkeeping ----------------------------------------------------

    @property
    def n_components(self) -> int:
        return int(sum(block.n_components for block in self.blocks))

    def block_slice(self, index: int) -> slice:
        start = int(sum(b.n_components for b in self.blocks[:index]))
        return slice(start, start + self.blocks[index].n_components)

    # -- component likelihoods ------------------------------------------------

    def component_log_density(self, pivots: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [block.component_log_density(pivots) for block in self.blocks], axis=1
        )

    def component_log_ratio(self, pivots: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [block.component_log_ratio(pivots) for block in self.blocks], axis=1
        )

    # -- sequential Bayes factors --------------------------------------------

    def shared_paths(
        self,
        pivots: np.ndarray,
        *,
        horizons: Sequence[int] | None = None,
        running_max: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Shared-latent log Bayes factor; see :func:`_shared_latent_paths`."""

        return _shared_latent_paths(
            self, pivots, horizons=horizons, running_max=running_max
        )

    # -- diagnostics ----------------------------------------------------------

    def spike_agreement(self, pivots: np.ndarray) -> float:
        """Interpolation accuracy of the equal-tail atom, which sits in the
        Dirichlet block; the tail-width block is closed form and exact."""

        return self.dirichlet_block.spike_agreement(pivots)

    def analytic_normalisation(self) -> dict[str, float]:
        report = {
            f"alpha={format_alpha(a)},J={self.tail_size}": value
            for a, value in self.dirichlet_block.analytic_normalisation().items()
            for _ in (0,)
        }
        for key, value in self.tail_width_block.analytic_normalisation().items():
            report[f"alpha=inf,J={key}"] = value
        return report

    def tokenwise_lookup(
        self,
        *,
        logit_limit: float = 30.0,
        size: int = 80_001,
    ) -> "UnionTailTokenwiseLookup":
        """The tokenwise union rule, with the tail state left at document level.

        Deliberately NOT a DirichletTokenwiseLookup: that collapses every
        component at each token, which redraws the tail state per position and
        contradicts Section 3 and Figure 1.  See UnionTailTokenwiseLookup.
        """

        return UnionTailTokenwiseLookup(self, logit_limit=logit_limit, size=size)

    def prior_fingerprint(self) -> dict[str, object]:
        return {
            "family": "union of the Dirichlet tail-shape and tail-width families",
            "dirichlet_weight": self.dirichlet_weight,
            "tail_width_weight": 1.0 - self.dirichlet_weight,
            "alpha_grid": [format_alpha(a) for a in self.alphas],
            "tail_width_grid": [int(j) for j in self.tail_widths],
            "tail_size": int(self.tail_size),
            "block_weights": [float(w) for w in self.block_weights],
            "block_components": [int(b.n_components) for b in self.blocks],
            "delta_nodes": int(self.deltas.size),
            "delta_node_sum": float(self.deltas.sum()),
            "rho_grid": [float(r) for r in self.rhos],
            "rho_weights": [float(w) for w in self.rho_weights],
            "rho_mean": float(np.dot(self.rho_weights, self.rhos)),
            "n_components": int(self.n_components),
            "contains_dirichlet_tail_shape_family": True,
            "contains_tail_width_family": True,
            "contains_equal_tail_spike": True,
        }
