#!/usr/bin/env python3
"""Dirichlet tail families for the Gumbel-max pivot.

Motivation
----------
The Gumbel-max pivot density for a *fixed* next-token-probability vector ``P``
is

    f_P(r) = sum_w r**(1/P_w - 1),      0 <= r <= 1.

It is a **sum over coordinates**, so a prior on ``P`` passes inside by
linearity.  Condition on the deficit, ``P_(1) = 1 - Delta``, and let the tail
be ``Delta * q`` with ``q ~ Dirichlet(alpha, ..., alpha)`` on ``K = M - 1``
components.  Exchangeability then leaves only the *marginal* of one tail
coordinate, ``q ~ Beta(alpha, (K-1) alpha)``, and

    f_{Delta,alpha}(r) = r**(Delta/(1-Delta)) + K * E_q[ r**(1/(Delta q) - 1) ].

Efficient form
--------------
With ``L = -log r`` and ``c = L / Delta``,

    E_q[r**(1/(Delta q))] = E_q[exp(-c/q)] =: psi_alpha(c),

a single univariate function per ``alpha``.  Tabulating ``psi_alpha`` once on a
log-spaced ``c`` grid turns every ``(Delta, r)`` evaluation into interpolation:

    f_{Delta,alpha}(r) = r**(Delta/(1-Delta)) + (K/r) * psi_alpha(L/Delta).

Everything is carried in log space.  The companion transform

    chi_alpha(c) = E_q[q exp(-c/q)]

gives the CDF in closed form,

    F_{Delta,alpha}(r) = (1-Delta) r**(1/(1-Delta)) + K Delta chi_alpha(L/Delta),

which is what the validation tests compare against a literal argmax race.

Analytic and pre-interpolation normalisation
--------------------------------------------
``int_0^1 f dr = (1-Delta) + Delta K E[q] = 1`` because ``E[q] = 1/K``.  The
analytic component is therefore automatically normalised and the martingale /
e-process argument needs no change when it is evaluated directly.  When the
tail prior is discretised, the nodes are **rescaled** so that ``K * (weighted
mean of q) = 1`` holds to floating-point exactness; the resulting quadrature
component is normalised before interpolation.  Linear interpolation of the
tabulated transform is a further numerical approximation: its measured mass
error is small, but exact e-process validity requires conservative
renormalisation or a certified mass bound.

Limits
------
``alpha -> infinity`` recovers the equal-tail spike family exactly (``q`` is
degenerate at ``1/K``); it is supported directly as ``alpha = math.inf``.
Small ``alpha`` gives sparse tails.

Out of family
-------------
The released simulator accompanying Li et al. (2025) builds its tail by normalising i.i.d. *uniforms*.
That is **not** a Dirichlet law: normalising i.i.d. uniforms sends the scaled
marginal ``K q`` to ``Uniform(0, 2)`` (variance ``1/3``), whereas
``Dirichlet(alpha)`` sends it to ``Gamma(alpha, alpha)`` (variance ``1/alpha``,
unbounded).  The variance-matched Dirichlet member is ``alpha ~ 3``, and even
that differs in shape: the released law has essentially no upper tail past the
limiting endpoint 2, while ``Gamma(3, 3)`` does.  The endpoint is a limit, not a
hard bound -- the normalising sum fluctuates at ``O(K^{-1/2})``, so at ``K=999``
about ``0.7%`` of coordinates exceed 2 and the observed maximum is near 2.1 --
but the contrast in tail weight is what makes this an out-of-family generator.  The normalised-uniform law is
provided here as :func:`simulate_normalised_uniform_pivots` precisely because
it is outside the Dirichlet family, which makes it the honest out-of-family
robustness case.

Depends only on NumPy.
"""

from __future__ import annotations

import math
from typing import Sequence, Union

import numpy as np


ArrayLike = Union[float, Sequence[float], np.ndarray]

# Quadrature defaults for the tail marginal q ~ Beta(alpha, (K-1) alpha).
# The grid is composite Gauss-Legendre in z = logit(q); the integrand
# q**alpha (1-q)**((K-1)alpha) / B is analytic there and decays exponentially
# at both ends, so short panels with a high-order rule are very accurate.
DEFAULT_PANEL_WIDTH = 0.5
DEFAULT_GAUSS_NODES = 16
DEFAULT_TRUNCATION_MASS = 1e-14

# Panel refinement for concentrated tail marginals.  The tolerance is loose
# enough that every alpha which already converged keeps its original panel
# count, and tight enough to catch the resolution loss that appears once the
# Laplace scale 1/sqrt(alpha) falls well below the panel width.
# Accepting the legacy panel count whenever it already meets the historical
# guard keeps every previously tabulated alpha bit-identical.
LEGACY_MASS_TOLERANCE = 1e-10
# Once refinement starts, converge on the *change* between successive panel
# doublings.  The captured mass tends to a value offset from one by the
# analytic truncation floor (order 1e-9 for concentrated marginals), which no
# amount of refinement removes and which the subsequent renormalisation and
# mean-rescaling absorb exactly.
PANEL_CONVERGENCE_TOLERANCE = 1e-12
PANEL_MASS_SANITY = 1e-6
MAX_PANEL_REFINEMENTS = 12

# Log-spaced grid for the psi/chi tables.  ``c = -log(r)/Delta`` is zero at
# r = 1 and grows without bound as r -> 0; the tail term of the density is
# already negligible long before the upper end of this grid.
DEFAULT_C_MIN = 1e-30
DEFAULT_C_MAX = 1e6
DEFAULT_C_NODES = 20_001

_CHUNK = 4_096


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _logsumexp_last(values: np.ndarray) -> np.ndarray:
    """log(sum(exp(values))) over the last axis, without SciPy."""

    maximum = np.max(values, axis=-1)
    output = np.full(maximum.shape, -np.inf, dtype=float)
    finite = np.isfinite(maximum)
    if np.any(finite):
        shifted = values[finite] - maximum[finite][..., None]
        output[finite] = maximum[finite] + np.log(np.exp(shifted).sum(axis=-1))
    return output


def _validated_alpha(alpha: float) -> float:
    value = float(alpha)
    if math.isnan(value) or value <= 0.0:
        raise ValueError("alpha must be positive (math.inf is the spike limit)")
    return value


def _validated_tail_size(tail_size: int) -> int:
    if not isinstance(tail_size, (int, np.integer)) or int(tail_size) < 1:
        raise ValueError("tail_size must be an integer of at least 1")
    return int(tail_size)


def tail_marginal_quadrature(
    alpha: float,
    tail_size: int,
    *,
    panel_width: float = DEFAULT_PANEL_WIDTH,
    gauss_nodes: int = DEFAULT_GAUSS_NODES,
    truncation_mass: float = DEFAULT_TRUNCATION_MASS,
) -> tuple[np.ndarray, np.ndarray]:
    """Discretise the tail marginal ``q ~ Beta(alpha, (K-1) alpha)``.

    Returns ``(nodes, weights)`` with ``weights`` summing to one and the nodes
    **rescaled** so that ``tail_size * sum(weights * nodes) == 1`` to
    floating-point exactness.  That rescaling is what keeps
    ``int_0^1 f_{Delta,alpha} dr = 1`` exact, hence the Bayes factor a
    martingale under the null.

    ``alpha = math.inf`` returns the degenerate spike node ``1/K``.
    """

    alpha = _validated_alpha(alpha)
    tail_size = _validated_tail_size(tail_size)

    if tail_size == 1:
        # A single tail coordinate carries the whole deficit: q == 1.
        return np.ones(1, dtype=float), np.ones(1, dtype=float)
    if math.isinf(alpha):
        # Degenerate Dirichlet limit: every tail coordinate equals 1/K.
        return np.full(1, 1.0 / tail_size, dtype=float), np.ones(1, dtype=float)

    if not (math.isfinite(panel_width) and panel_width > 0.0):
        raise ValueError("panel_width must be positive and finite")
    if not isinstance(gauss_nodes, (int, np.integer)) or int(gauss_nodes) < 2:
        raise ValueError("gauss_nodes must be an integer of at least 2")
    if not (0.0 < truncation_mass < 1e-3):
        raise ValueError("truncation_mass must lie in (0, 1e-3)")

    a = alpha
    b = (tail_size - 1.0) * alpha
    log_b_constant = _log_beta(a, b)

    # In z = logit(q) the density is exp(a*log q + b*log(1-q) - log B), which
    # behaves like exp(a z)/B as z -> -inf and exp(-b z)/B as z -> +inf.  Solve
    # for the truncation points analytically and add a safety margin.
    log_mass = math.log(truncation_mass)
    z_low = (log_mass + math.log(a) + log_b_constant) / a - 2.0
    z_high = -(log_mass + math.log(b) + log_b_constant) / b + 2.0
    if not z_low < z_high:  # pragma: no cover - defensive
        raise ValueError("degenerate quadrature range for the tail marginal")

    # The log-density in z peaks at the mode q = 1/K with Laplace scale
    # sd_z = 1/sqrt(alpha (1 - 1/K)), so the peak narrows as alpha grows and a
    # fixed panel width eventually fails to resolve it.  Start from the legacy
    # panel count and refine only if the captured mass says we must; for every
    # alpha that already converged this loop exits on the first pass and
    # reproduces the previous nodes and weights bit for bit.
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(int(gauss_nodes))
    n_panels = max(8, int(math.ceil((z_high - z_low) / panel_width)))

    def _panels(count: int) -> tuple[np.ndarray, np.ndarray, float]:
        edges = np.linspace(z_low, z_high, count + 1)
        half_width = 0.5 * (edges[1] - edges[0])
        centres = 0.5 * (edges[:-1] + edges[1:])
        z = (centres[:, None] + half_width * gl_nodes[None, :]).reshape(-1)
        quadrature_weights = np.tile(gl_weights * half_width, count)
        node_log_q = -np.logaddexp(0.0, -z)
        log_one_minus_q = -np.logaddexp(0.0, z)
        log_density = a * node_log_q + b * log_one_minus_q - log_b_constant
        node_weights = quadrature_weights * np.exp(log_density)
        return node_log_q, node_weights, float(node_weights.sum())

    log_q, weights, captured = _panels(n_panels)
    if abs(captured - 1.0) > LEGACY_MASS_TOLERANCE:
        previous = captured
        for _ in range(MAX_PANEL_REFINEMENTS):
            n_panels *= 2
            log_q, weights, captured = _panels(n_panels)
            if abs(captured - previous) <= PANEL_CONVERGENCE_TOLERANCE:
                break
            previous = captured
        else:  # pragma: no cover - defensive
            raise ValueError(
                "tail-marginal quadrature failed to converge for "
                f"alpha={alpha!r}, tail_size={tail_size!r} after "
                f"{MAX_PANEL_REFINEMENTS} refinements"
            )
        if abs(captured - 1.0) > PANEL_MASS_SANITY:  # pragma: no cover
            raise ValueError(
                f"tail-marginal quadrature captured {captured!r} of the unit "
                f"mass for alpha={alpha!r}, tail_size={tail_size!r}"
            )

    weights = weights / captured

    nodes = np.exp(log_q)
    mean = float(np.dot(weights, nodes))
    # Force K * E[q] = 1 so that the component density integrates to exactly
    # one and the Bayes factor stays a martingale under the null.
    nodes = nodes * (1.0 / (tail_size * mean))
    return nodes, weights


class DirichletTailPivot:
    """Gumbel-pivot law for a ``Delta``-deficit NTP with Dirichlet tails.

    The leading coordinate is ``1 - Delta`` and the remaining ``K = M - 1``
    coordinates are ``Delta * q`` with ``q ~ Dirichlet(alpha, ..., alpha)``.
    Only the marginal of one tail coordinate survives, so a single tabulated
    transform per ``alpha`` serves every ``Delta``.

    ``alpha = math.inf`` is the equal-tail spike family.
    """

    def __init__(
        self,
        alpha: float,
        tail_size: int,
        *,
        c_min: float = DEFAULT_C_MIN,
        c_max: float = DEFAULT_C_MAX,
        c_nodes: int = DEFAULT_C_NODES,
        panel_width: float = DEFAULT_PANEL_WIDTH,
        gauss_nodes: int = DEFAULT_GAUSS_NODES,
        truncation_mass: float = DEFAULT_TRUNCATION_MASS,
    ) -> None:
        self.alpha = _validated_alpha(alpha)
        self.tail_size = _validated_tail_size(tail_size)
        if not 0.0 < c_min < c_max:
            raise ValueError("require 0 < c_min < c_max")
        if not isinstance(c_nodes, (int, np.integer)) or int(c_nodes) < 16:
            raise ValueError("c_nodes must be an integer of at least 16")

        self.nodes, self.weights = tail_marginal_quadrature(
            self.alpha,
            self.tail_size,
            panel_width=panel_width,
            gauss_nodes=gauss_nodes,
            truncation_mass=truncation_mass,
        )
        # Far-tail quadrature weights can underflow to exactly zero; log(0) is
        # the correct -inf here and logsumexp handles it, so silence the warning
        # rather than perturbing the weights.
        with np.errstate(divide="ignore"):
            self._log_weights = np.log(self.weights)
            self._log_weighted_nodes = self._log_weights + np.log(self.nodes)
        self._log_tail_size = math.log(self.tail_size)

        self.log_c_grid = np.linspace(math.log(c_min), math.log(c_max), int(c_nodes))
        c_grid = np.exp(self.log_c_grid)
        # Store log psi + c and log chi + c.  For large c both transforms decay
        # like exp(-c), so the shifted tables are nearly linear in log c and
        # linear interpolation on them is well conditioned.
        self._psi_shifted = self._log_psi_direct(c_grid) + c_grid
        self._chi_shifted = self._log_chi_direct(c_grid) + c_grid

    # -- exact (unbtabulated) transforms -------------------------------------

    def _log_transform_direct(self, c: np.ndarray, log_base: np.ndarray) -> np.ndarray:
        flat = np.asarray(c, dtype=float).reshape(-1)
        output = np.empty(flat.size, dtype=float)
        inverse_nodes = 1.0 / self.nodes
        for start in range(0, flat.size, _CHUNK):
            stop = min(start + _CHUNK, flat.size)
            block = flat[start:stop, None] * inverse_nodes[None, :]
            output[start:stop] = _logsumexp_last(log_base[None, :] - block)
        return output.reshape(np.shape(c))

    def _log_psi_direct(self, c: ArrayLike) -> np.ndarray:
        """``log E_q[exp(-c/q)]`` summed directly over the quadrature nodes."""

        return self._log_transform_direct(np.asarray(c, dtype=float), self._log_weights)

    def _log_chi_direct(self, c: ArrayLike) -> np.ndarray:
        """``log E_q[q exp(-c/q)]`` summed directly over the quadrature nodes."""

        return self._log_transform_direct(
            np.asarray(c, dtype=float), self._log_weighted_nodes
        )

    # -- tabulated transforms -------------------------------------------------

    def _interpolate(self, c: np.ndarray, table: np.ndarray, at_zero: float) -> np.ndarray:
        c = np.asarray(c, dtype=float)
        if np.any(c < 0.0):
            raise ValueError("the transform argument c must be nonnegative")
        safe = np.maximum(c, math.exp(self.log_c_grid[0]))
        with np.errstate(divide="ignore"):
            log_c = np.log(safe)
        shifted = np.interp(
            log_c,
            self.log_c_grid,
            table,
            left=table[0],
            right=table[-1],
        )
        with np.errstate(invalid="ignore"):
            value = shifted - c
        return np.where(c == 0.0, at_zero, value)

    def log_psi(self, c: ArrayLike, *, exact: bool = False) -> np.ndarray:
        """``log E_q[exp(-c/q)]``.  ``psi(0) = 1`` exactly."""

        array = np.asarray(c, dtype=float)
        if exact:
            direct = self._log_psi_direct(array)
            return np.where(array == 0.0, 0.0, direct)
        return self._interpolate(array, self._psi_shifted, 0.0)

    def log_chi(self, c: ArrayLike, *, exact: bool = False) -> np.ndarray:
        """``log E_q[q exp(-c/q)]``.  ``chi(0) = E[q] = 1/K`` exactly."""

        array = np.asarray(c, dtype=float)
        at_zero = -self._log_tail_size
        if exact:
            direct = self._log_chi_direct(array)
            return np.where(array == 0.0, at_zero, direct)
        return self._interpolate(array, self._chi_shifted, at_zero)

    # -- pivot law ------------------------------------------------------------

    @staticmethod
    def _validated_delta(delta: ArrayLike) -> np.ndarray:
        array = np.asarray(delta, dtype=float)
        if np.any(~np.isfinite(array)) or np.any((array <= 0.0) | (array >= 1.0)):
            raise ValueError("delta must lie strictly in (0, 1)")
        return array

    def log_pdf(self, r: ArrayLike, delta: ArrayLike, *, exact: bool = False) -> np.ndarray:
        """Log density of the pivot at deficit ``delta``.

        ``f(r) = r**(Delta/(1-Delta)) + (K/r) psi(L/Delta)`` with ``L = -log r``.
        """

        pivots = np.asarray(r, dtype=float)
        if np.any(~np.isfinite(pivots)) or np.any((pivots < 0.0) | (pivots > 1.0)):
            raise ValueError("Gumbel pivots must lie in [0, 1]")
        deltas = self._validated_delta(delta)
        with np.errstate(divide="ignore"):
            log_r = np.log(pivots)
        deficit = -log_r  # L >= 0, +inf at r = 0
        log_top = (deltas / (1.0 - deltas)) * log_r
        log_tail = (
            self._log_tail_size
            + deficit
            + self.log_psi(deficit / deltas, exact=exact)
        )
        # At r = 0 both branches are -inf; logaddexp propagates that correctly.
        log_top = np.where(np.isnan(log_top), -np.inf, log_top)
        log_tail = np.where(np.isnan(log_tail), -np.inf, log_tail)
        return np.logaddexp(log_top, log_tail)

    def pdf(self, r: ArrayLike, delta: ArrayLike, *, exact: bool = False) -> np.ndarray:
        return np.exp(self.log_pdf(r, delta, exact=exact))

    def cdf(self, r: ArrayLike, delta: ArrayLike, *, exact: bool = False) -> np.ndarray:
        """``F(r) = (1-Delta) r**(1/(1-Delta)) + K Delta chi(L/Delta)``."""

        pivots = np.asarray(r, dtype=float)
        if np.any(~np.isfinite(pivots)) or np.any((pivots < 0.0) | (pivots > 1.0)):
            raise ValueError("Gumbel pivots must lie in [0, 1]")
        deltas = self._validated_delta(delta)
        with np.errstate(divide="ignore"):
            log_r = np.log(pivots)
        deficit = -log_r
        top = (1.0 - deltas) * np.exp(log_r / (1.0 - deltas))
        tail = (
            self.tail_size
            * deltas
            * np.exp(self.log_chi(deficit / deltas, exact=exact))
        )
        return np.where(pivots == 0.0, 0.0, top + tail)

    def normalisation(self, delta: ArrayLike) -> np.ndarray:
        """Analytic ``int_0^1 f dr`` for the discretised tail prior.

        Equals ``(1-Delta) + Delta * K * sum(w q)``, which the node rescaling
        pins to one.
        """

        deltas = self._validated_delta(delta)
        mass = self.tail_size * float(np.dot(self.weights, self.nodes))
        return (1.0 - deltas) + deltas * mass

    # -- simulation -----------------------------------------------------------

    def _size_biased_tail(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Draw the *selected* tail coordinate ``B ~ Beta(alpha+1, (K-1)alpha)``.

        Selecting a token size-biases the tail coordinate, so the chosen tail
        probability is ``Delta * B``.
        """

        if self.tail_size == 1:
            return np.ones(size, dtype=float)
        if math.isinf(self.alpha):
            return np.full(size, 1.0 / self.tail_size, dtype=float)
        return rng.beta(self.alpha + 1.0, (self.tail_size - 1.0) * self.alpha, size=size)

    def simulate(self, rng: np.random.Generator, deltas: ArrayLike) -> np.ndarray:
        """Exact, fully vectorised pivot simulator.

            R = U**(1-Delta)   with probability 1-Delta
            R = U**(Delta B)   with probability Delta,  B ~ Beta(alpha+1, (K-1)alpha)

        ``deltas`` may have any shape; the output matches it.
        """

        array = self._validated_delta(deltas)
        chose_tail = rng.uniform(size=array.shape) < array
        probability = np.where(chose_tail, 0.0, 1.0 - array)
        count = int(np.count_nonzero(chose_tail))
        if count:
            probability[chose_tail] = (
                array[chose_tail] * self._size_biased_tail(rng, count)
            )
        return rng.uniform(size=array.shape) ** probability


_TABLE_CACHE: dict[tuple[float, int, float, float, int], DirichletTailPivot] = {}


def dirichlet_tail_pivot(
    alpha: float,
    tail_size: int,
    *,
    c_min: float = DEFAULT_C_MIN,
    c_max: float = DEFAULT_C_MAX,
    c_nodes: int = DEFAULT_C_NODES,
) -> DirichletTailPivot:
    """Cached :class:`DirichletTailPivot`; the tables are expensive to build."""

    key = (float(alpha), int(tail_size), float(c_min), float(c_max), int(c_nodes))
    pivot = _TABLE_CACHE.get(key)
    if pivot is None:
        pivot = DirichletTailPivot(
            alpha, tail_size, c_min=c_min, c_max=c_max, c_nodes=c_nodes
        )
        _TABLE_CACHE[key] = pivot
    return pivot


# -- generators used for validation and for the out-of-family case ------------


def dirichlet_argmax_race(
    rng: np.random.Generator,
    deltas: ArrayLike,
    vocabulary_size: int,
    alpha: float,
) -> np.ndarray:
    """Literal ``argmax_w log(U_w)/P_w`` race with explicit Dirichlet tails.

    This is the definitional generator, used only to validate the closed-form
    density: it draws the whole NTP vector and the whole key, so its cost is
    ``O(M)`` per token.  Keep ``vocabulary_size`` small.
    """

    array = DirichletTailPivot._validated_delta(deltas)
    if not isinstance(vocabulary_size, (int, np.integer)) or int(vocabulary_size) < 2:
        raise ValueError("vocabulary_size must be an integer of at least 2")
    vocabulary_size = int(vocabulary_size)
    tail_size = vocabulary_size - 1
    alpha = _validated_alpha(alpha)
    flat = array.reshape(-1)
    rows = flat.size

    if math.isinf(alpha):
        tails = np.full((rows, tail_size), 1.0 / tail_size, dtype=float)
    elif tail_size == 1:
        tails = np.ones((rows, 1), dtype=float)
    else:
        tails = rng.dirichlet(np.full(tail_size, alpha), size=rows)
    probabilities = np.empty((rows, vocabulary_size), dtype=float)
    probabilities[:, 0] = 1.0 - flat
    probabilities[:, 1:] = flat[:, None] * tails

    keys = rng.uniform(size=(rows, vocabulary_size))
    with np.errstate(divide="ignore"):
        scores = np.log(keys) / probabilities
    winners = np.argmax(scores, axis=1)
    return keys[np.arange(rows), winners].reshape(array.shape)


def normalised_uniform_tail(
    rng: np.random.Generator, size: int, tail_size: int
) -> np.ndarray:
    """Size-biased coordinate of a tail built by normalising i.i.d. uniforms.

    This is the construction in the released simulator, ``generated/generated.sum()``.
    It is **not** a Dirichlet law: the scaled marginal ``K q`` tends to
    ``Uniform(0, 2)`` rather than ``Gamma(alpha, alpha)``.  Returns the value of
    the coordinate picked with probability proportional to itself, which is
    what token selection does.

    Exact, but ``O(K)`` work per draw, so it is chunked.
    """

    tail_size = _validated_tail_size(tail_size)
    if size < 0:
        raise ValueError("size must be nonnegative")
    if tail_size == 1:
        return np.ones(size, dtype=float)
    output = np.empty(size, dtype=float)
    block = max(1, min(size, max(1, 8_000_000 // tail_size)))
    for start in range(0, size, block):
        stop = min(start + block, size)
        draws = rng.uniform(size=(stop - start, tail_size))
        total = draws.sum(axis=1)
        cumulative = np.cumsum(draws, axis=1)
        target = rng.uniform(size=stop - start) * total
        chosen = np.argmax(cumulative >= target[:, None], axis=1)
        output[start:stop] = (
            draws[np.arange(stop - start), chosen] / total
        )
    return output


def simulate_narrow_width_pivots(
    rng: np.random.Generator,
    deltas: np.ndarray,
    tail_width: int,
) -> np.ndarray:
    """Exact Gumbel pivots when the deficit sits on ``tail_width`` coordinates.

    The next-token vector is ``(1 - Delta, Delta / J, ..., Delta / J, 0, ..., 0)``
    with ``J = tail_width`` nonzero tail coordinates: the width branch of the
    union tail prior, and the law the released outputs actually look like, where
    the fitted width is one rather than the full vocabulary.

    This is the equal-tail generator with ``V - 1`` replaced by ``J``, so
    ``tail_width = V - 1`` reproduces it exactly.  That containment is what makes
    a width sweep a clean generalization of the existing experiments rather than
    a different setup: only the number of coordinates carrying the deficit moves.

    Given the selected token's probability ``p``, the Gumbel pivot is
    ``R = U ** p`` with ``U`` uniform, i.e. ``Beta(1/p, 1)``.
    """

    width = int(tail_width)
    if width < 1:
        raise ValueError("tail_width must be at least 1")
    deltas = np.asarray(deltas, dtype=float)
    if np.any(deltas < 0.0) or np.any(deltas > 1.0):
        raise ValueError("deltas must lie in [0, 1]")
    on_top = rng.uniform(size=deltas.shape) >= deltas
    selected = np.where(on_top, 1.0 - deltas, deltas / width)
    return rng.uniform(size=deltas.shape) ** selected


def simulate_normalised_uniform_pivots(
    rng: np.random.Generator, deltas: ArrayLike, vocabulary_size: int
) -> np.ndarray:
    """Exact pivots for the released code's normalised-uniform tail.

    Out of family for the Dirichlet layer by construction; see the module
    docstring.
    """

    array = DirichletTailPivot._validated_delta(deltas)
    if not isinstance(vocabulary_size, (int, np.integer)) or int(vocabulary_size) < 2:
        raise ValueError("vocabulary_size must be an integer of at least 2")
    tail_size = int(vocabulary_size) - 1
    chose_tail = rng.uniform(size=array.shape) < array
    probability = np.where(chose_tail, 0.0, 1.0 - array)
    count = int(np.count_nonzero(chose_tail))
    if count:
        probability[chose_tail] = array[chose_tail] * normalised_uniform_tail(
            rng, count, tail_size
        )
    return rng.uniform(size=array.shape) ** probability


def normalised_uniform_argmax_race(
    rng: np.random.Generator, deltas: ArrayLike, vocabulary_size: int
) -> np.ndarray:
    """Literal argmax race for the released code's normalised-uniform tail."""

    array = DirichletTailPivot._validated_delta(deltas)
    if not isinstance(vocabulary_size, (int, np.integer)) or int(vocabulary_size) < 2:
        raise ValueError("vocabulary_size must be an integer of at least 2")
    vocabulary_size = int(vocabulary_size)
    tail_size = vocabulary_size - 1
    flat = array.reshape(-1)
    rows = flat.size
    draws = rng.uniform(size=(rows, tail_size))
    tails = draws / draws.sum(axis=1, keepdims=True)
    probabilities = np.empty((rows, vocabulary_size), dtype=float)
    probabilities[:, 0] = 1.0 - flat
    probabilities[:, 1:] = flat[:, None] * tails
    keys = rng.uniform(size=(rows, vocabulary_size))
    with np.errstate(divide="ignore"):
        scores = np.log(keys) / probabilities
    winners = np.argmax(scores, axis=1)
    return keys[np.arange(rows), winners].reshape(array.shape)


__all__ = [
    "DEFAULT_C_MAX",
    "DEFAULT_C_MIN",
    "DEFAULT_C_NODES",
    "DirichletTailPivot",
    "dirichlet_argmax_race",
    "dirichlet_tail_pivot",
    "normalised_uniform_argmax_race",
    "normalised_uniform_tail",
    "simulate_normalised_uniform_pivots",
    "tail_marginal_quadrature",
]
