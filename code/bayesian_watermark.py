"""Bayesian detectors for pivot-based language-model watermarks.

The module implements the likelihoods developed in Li et al. (2025) and a
Bayesian alternative to fixing the regularity parameter Delta.  It deliberately
depends only on NumPy so that the examples accompanying the manuscript
are easy to reproduce.

Two model structures are available:

``shared``
    One latent (Delta, rho) pair is shared by the whole document.  Sequential
    updating learns that pair as evidence accumulates.

``tokenwise``
    A fresh (Delta, rho) pair is drawn at every token.  This is a simple model
    for strongly heterogeneous next-token distributions.

Here rho is an optional contamination probability.  Conditional on rho, the
alternative density is (1-rho) f_signal + rho f_0.
"""

from __future__ import annotations

import math
from typing import Iterable, Literal, Sequence, Union

import numpy as np


ArrayLike = Union[float, Sequence[float], np.ndarray]
Scheme = Literal["gumbel", "inverse"]
Structure = Literal["shared", "tokenwise"]
GumbelFamily = Literal["least_favorable", "spike"]


def _return_scalar_if_scalar(original: ArrayLike, value: np.ndarray) -> float | np.ndarray:
    """Return a Python float when the input was scalar, otherwise an array."""

    if np.asarray(original).ndim == 0:
        return float(np.asarray(value))
    return value


def _normalized_weights(weights: ArrayLike, expected_size: int, name: str) -> np.ndarray:
    array = np.asarray(weights, dtype=float)
    if array.ndim != 1 or array.size != expected_size:
        raise ValueError(f"{name} must be a one-dimensional array of length {expected_size}")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain finite, nonnegative values")
    total = float(array.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive total mass")
    return array / total


def logsumexp(values: ArrayLike, axis: int | tuple[int, ...] | None = None) -> float | np.ndarray:
    """Compute log(sum(exp(values))) without SciPy and without overflow."""

    array = np.asarray(values, dtype=float)
    maximum = np.max(array, axis=axis, keepdims=True)
    finite_maximum = np.isfinite(maximum)
    shifted = np.full_like(array, -np.inf, dtype=float)
    np.subtract(array, maximum, out=shifted, where=np.broadcast_to(finite_maximum, array.shape))
    with np.errstate(under="ignore", divide="ignore", invalid="ignore"):
        result = np.where(
            finite_maximum,
            maximum + np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True)),
            maximum,
        )
    if axis is not None:
        result = np.squeeze(result, axis=axis)
    elif result.ndim == 0:
        return float(result)
    else:
        result = np.squeeze(result)
    if np.asarray(result).ndim == 0:
        return float(result)
    return result


def beta_grid_prior(
    alpha: float,
    beta: float,
    *,
    low: float = 0.0,
    high: float = 1.0,
    size: int = 101,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate a scaled Beta prior by midpoint quadrature.

    The returned grid lies strictly inside ``(low, high)``.  Equal-width
    quadrature factors cancel during normalization, so the weights are simply
    the normalized Beta density evaluated at the midpoints.
    """

    if not (math.isfinite(alpha) and alpha > 0.0):
        raise ValueError("alpha must be positive and finite")
    if not (math.isfinite(beta) and beta > 0.0):
        raise ValueError("beta must be positive and finite")
    if not (math.isfinite(low) and math.isfinite(high) and 0.0 <= low < high <= 1.0):
        raise ValueError("require 0 <= low < high <= 1")
    if not isinstance(size, int) or size < 2:
        raise ValueError("size must be an integer of at least 2")

    unit_grid = (np.arange(size, dtype=float) + 0.5) / size
    grid = low + (high - low) * unit_grid
    log_beta_constant = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    log_density = (
        (alpha - 1.0) * np.log(unit_grid)
        + (beta - 1.0) * np.log1p(-unit_grid)
        - log_beta_constant
    )
    log_density -= float(logsumexp(log_density))
    return grid, np.exp(log_density)


def least_favorable_probabilities(
    delta: float, vocabulary_size: int | None = None
) -> np.ndarray:
    """Construct the least-favorable NTP vector for the Delta-regular class.

    Its positive entries are ``m`` copies of ``1-delta`` and, when nonzero,
    the residual ``1-m(1-delta)``, where ``m=floor(1/(1-delta))``.
    """

    if not (math.isfinite(delta) and 0.0 <= delta < 1.0):
        raise ValueError("delta must lie in [0, 1)")
    if vocabulary_size is not None and (
        not isinstance(vocabulary_size, int) or vocabulary_size < 1
    ):
        raise ValueError("vocabulary_size must be a positive integer")

    cap = 1.0 - delta
    # The tolerance makes exact boundaries such as delta=1/2 stable in binary
    # floating point, while not changing values away from a boundary.
    count = int(math.floor(1.0 / cap + 1e-12))
    residual = 1.0 - count * cap
    if abs(residual) < 1e-12:
        residual = 0.0
    positive_size = count + int(residual > 0.0)

    if vocabulary_size is not None and positive_size > vocabulary_size:
        max_delta = 1.0 - 1.0 / vocabulary_size
        raise ValueError(
            f"delta={delta:g} is incompatible with vocabulary_size={vocabulary_size}; "
            f"require delta <= {max_delta:g}"
        )

    output_size = vocabulary_size if vocabulary_size is not None else positive_size
    probabilities = np.zeros(output_size, dtype=float)
    probabilities[:count] = cap
    if residual > 0.0:
        probabilities[count] = residual
    probabilities /= probabilities.sum()
    return probabilities


def spike_probabilities(delta: float, vocabulary_size: int) -> np.ndarray:
    """Construct ``(1-delta, delta/(V-1), ..., delta/(V-1))``.

    The validation enforces membership in the Delta-regular class, including
    the requirement that no tail probability exceeds the leading probability.
    """

    if not isinstance(vocabulary_size, int) or vocabulary_size < 2:
        raise ValueError("vocabulary_size must be an integer of at least 2")
    max_delta = 1.0 - 1.0 / vocabulary_size
    if not (math.isfinite(delta) and 0.0 <= delta <= max_delta + 1e-12):
        raise ValueError(f"delta must lie in [0, {max_delta:g}] for this vocabulary")
    probabilities = np.full(vocabulary_size, delta / (vocabulary_size - 1), dtype=float)
    probabilities[0] = 1.0 - delta
    return probabilities


def _validated_probabilities(probabilities: ArrayLike) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError("probabilities must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("probabilities must be finite and nonnegative")
    total = float(probs.sum())
    if not np.isclose(total, 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("probabilities must sum to one")
    return probs[probs > 0.0] / total


def gumbel_null_logpdf(r: ArrayLike) -> float | np.ndarray:
    """Log density of the Gumbel-max pivot under H0: Uniform(0, 1)."""

    array = np.asarray(r, dtype=float)
    if np.any(~np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("Gumbel pivots must lie in [0, 1]")
    output = np.zeros_like(array, dtype=float)
    return _return_scalar_if_scalar(r, output)


def gumbel_alt_logpdf_from_probs(
    r: ArrayLike, probabilities: ArrayLike
) -> float | np.ndarray:
    """Exact Gumbel-pivot log density under H1.

    For positive token probabilities P_w,

        f_1(r | P) = sum_w r**(1/P_w - 1),  0 <= r <= 1.
    """

    array = np.asarray(r, dtype=float)
    if np.any(~np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("Gumbel pivots must lie in [0, 1]")
    probs = _validated_probabilities(probabilities)
    exponents = 1.0 / probs - 1.0

    flat = array.reshape(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_r = np.log(flat)
        terms = log_r[:, None] * exponents[None, :]
    # 0**0=1 for a singular P_w=1 component; 0**a=0 for a>0.
    at_zero = flat == 0.0
    if np.any(at_zero):
        terms[at_zero, :] = np.where(np.isclose(exponents, 0.0, atol=1e-14), 0.0, -np.inf)
    output = np.asarray(logsumexp(terms, axis=1)).reshape(array.shape)
    return _return_scalar_if_scalar(r, output)


def gumbel_alt_pdf_from_probs(
    r: ArrayLike, probabilities: ArrayLike
) -> float | np.ndarray:
    """Exact Gumbel-pivot density under H1."""

    log_density = np.asarray(gumbel_alt_logpdf_from_probs(r, probabilities))
    density = np.exp(log_density)
    return _return_scalar_if_scalar(r, density)


def inverse_null_logpdf(
    d: ArrayLike, vocabulary_size: int | None = None
) -> float | np.ndarray:
    """Inverse-pivot log density under H0.

    If ``vocabulary_size`` is omitted, this returns the large-vocabulary
    triangular limit ``f0(d)=2(1-d)``.  For finite ``vocabulary_size=M``, it
    returns the exact density of ``|U-eta(I)|``, where ``U`` is uniform,
    ``I`` is uniform on ``{1,...,M}``, and ``eta(i)=(i-1)/(M-1)``.  Using the
    exact version preserves the Bayes-factor anytime guarantee at finite M.
    """

    array = np.asarray(d, dtype=float)
    if np.any(~np.isfinite(array)) or np.any((array < 0.0) | (array >= 1.0)):
        raise ValueError("inverse-transform pivots must lie in [0, 1)")
    if vocabulary_size is None:
        output = math.log(2.0) + np.log1p(-array)
    else:
        if not isinstance(vocabulary_size, int) or vocabulary_size < 2:
            raise ValueError("vocabulary_size must be an integer of at least 2")
        # Conditional on rank eta=j/(M-1), |U-eta| has one unit-density
        # branch in each feasible direction.  On either side, exactly
        # M-1-floor((M-1)d) ranks contribute, hence
        #
        #   f_{0,M}(d) = 2/M * (M-1-floor((M-1)d)).
        #
        # This is the O(1)-per-pivot form in the manuscript.  It is equivalent
        # to counting all M ranks, but avoids materialising an input-by-M
        # broadcast.  The floor selects the same strict-inequality density
        # version at the measure-zero jump points.
        count_one_side = vocabulary_size - 1 - np.floor(
            (vocabulary_size - 1) * array
        )
        density = 2.0 * count_one_side / vocabulary_size
        output = np.log(density)
    return _return_scalar_if_scalar(d, output)


def inverse_null_pdf(
    d: ArrayLike, vocabulary_size: int | None = None
) -> float | np.ndarray:
    """Finite-vocabulary exact or large-vocabulary limiting null density."""

    density = np.exp(np.asarray(inverse_null_logpdf(d, vocabulary_size)))
    return _return_scalar_if_scalar(d, density)


def inverse_alt_logpdf(d: ArrayLike, delta: float) -> float | np.ndarray:
    """Large-vocabulary inverse-transform log density under H1."""

    if not (math.isfinite(delta) and 0.0 <= delta < 1.0):
        raise ValueError("delta must lie in [0, 1)")
    array = np.asarray(d, dtype=float)
    if np.any(~np.isfinite(array)) or np.any((array < 0.0) | (array >= 1.0)):
        raise ValueError("inverse-transform pivots must lie in [0, 1)")
    scale = 1.0 - delta
    inside = 1.0 - array / scale
    with np.errstate(divide="ignore", invalid="ignore"):
        output = np.where(inside > 0.0, math.log(2.0 / scale) + np.log(inside), -np.inf)
    return _return_scalar_if_scalar(d, output)


def inverse_alt_pdf(d: ArrayLike, delta: float) -> float | np.ndarray:
    """Large-vocabulary inverse-transform density under H1."""

    density = np.exp(np.asarray(inverse_alt_logpdf(d, delta)))
    return _return_scalar_if_scalar(d, density)


class SequentialBayesDetector:
    """Sequential Bayes-factor detector for Gumbel or inverse pivots.

    Parameters
    ----------
    scheme:
        ``"gumbel"`` or ``"inverse"``.
    delta_grid, delta_weights:
        Discrete quadrature approximation to the prior on Delta.
    rho_grid, rho_weights:
        Optional contamination prior.  ``rho=0`` means no contamination.
    structure:
        ``"shared"`` learns one latent component for the whole document;
        ``"tokenwise"`` integrates a fresh component at each observation.
    inverse_null_vocabulary_size:
        For the inverse scheme, use the exact finite-vocabulary pivot null for
        this vocabulary size.  If omitted, use its triangular limiting law.
    prior_watermark:
        Prior probability assigned to H1, used only to convert the Bayes factor
        into a posterior probability or a loss-based decision.
    """

    def __init__(
        self,
        scheme: Scheme,
        delta_grid: ArrayLike,
        delta_weights: ArrayLike | None = None,
        *,
        rho_grid: ArrayLike = (0.0,),
        rho_weights: ArrayLike | None = None,
        structure: Structure = "shared",
        gumbel_family: GumbelFamily = "least_favorable",
        vocabulary_size: int = 50,
        inverse_null_vocabulary_size: int | None = None,
        prior_watermark: float = 0.5,
    ) -> None:
        if scheme not in ("gumbel", "inverse"):
            raise ValueError("scheme must be 'gumbel' or 'inverse'")
        if structure not in ("shared", "tokenwise"):
            raise ValueError("structure must be 'shared' or 'tokenwise'")
        if gumbel_family not in ("least_favorable", "spike"):
            raise ValueError("gumbel_family must be 'least_favorable' or 'spike'")
        if not isinstance(vocabulary_size, int) or vocabulary_size < 2:
            raise ValueError("vocabulary_size must be an integer of at least 2")
        if inverse_null_vocabulary_size is not None and (
            not isinstance(inverse_null_vocabulary_size, int)
            or inverse_null_vocabulary_size < 2
        ):
            raise ValueError("inverse_null_vocabulary_size must be an integer of at least 2")
        if not (math.isfinite(prior_watermark) and 0.0 < prior_watermark < 1.0):
            raise ValueError("prior_watermark must lie strictly between zero and one")

        deltas = np.asarray(delta_grid, dtype=float)
        rhos = np.asarray(rho_grid, dtype=float)
        if deltas.ndim != 1 or deltas.size == 0:
            raise ValueError("delta_grid must be a nonempty one-dimensional array")
        if np.any(~np.isfinite(deltas)) or np.any((deltas < 0.0) | (deltas >= 1.0)):
            raise ValueError("delta_grid values must lie in [0, 1)")
        if rhos.ndim != 1 or rhos.size == 0:
            raise ValueError("rho_grid must be a nonempty one-dimensional array")
        if np.any(~np.isfinite(rhos)) or np.any((rhos < 0.0) | (rhos > 1.0)):
            raise ValueError("rho_grid values must lie in [0, 1]")

        if delta_weights is None:
            normalized_delta_weights = np.full(deltas.size, 1.0 / deltas.size)
        else:
            normalized_delta_weights = _normalized_weights(
                delta_weights, deltas.size, "delta_weights"
            )
        if rho_weights is None:
            normalized_rho_weights = np.full(rhos.size, 1.0 / rhos.size)
        else:
            normalized_rho_weights = _normalized_weights(rho_weights, rhos.size, "rho_weights")

        component_delta, component_rho = np.meshgrid(deltas, rhos, indexing="ij")
        component_weights = np.outer(normalized_delta_weights, normalized_rho_weights).reshape(-1)

        self.scheme = scheme
        self.structure = structure
        self.gumbel_family = gumbel_family
        self.vocabulary_size = vocabulary_size
        self.inverse_null_vocabulary_size = inverse_null_vocabulary_size
        self.prior_watermark = float(prior_watermark)
        self.delta_grid = deltas.copy()
        self.delta_weights = normalized_delta_weights.copy()
        self.rho_grid = rhos.copy()
        self.rho_weights = normalized_rho_weights.copy()
        self.component_delta = component_delta.reshape(-1)
        self.component_rho = component_rho.reshape(-1)
        self.component_prior_weights = component_weights
        self._component_log_prior = np.log(component_weights)

        if self.scheme == "gumbel":
            for delta in self.delta_grid:
                self._gumbel_probabilities(float(delta))
        self.reset()

    def _gumbel_probabilities(self, delta: float) -> np.ndarray:
        if self.gumbel_family == "least_favorable":
            return least_favorable_probabilities(delta, self.vocabulary_size)
        return spike_probabilities(delta, self.vocabulary_size)

    @property
    def log_bayes_factor(self) -> float:
        """Current log(m1/m0)."""

        return self._log_bayes_factor

    @property
    def bayes_factor(self) -> float:
        """Current Bayes factor; may overflow to infinity for decisive evidence."""

        if self._log_bayes_factor > math.log(np.finfo(float).max):
            return math.inf
        return math.exp(self._log_bayes_factor)

    @property
    def n_observations(self) -> int:
        return self._n_observations

    @property
    def component_posterior_weights(self) -> np.ndarray:
        """Posterior component weights (shared model) or prior weights (tokenwise)."""

        if self._alternative_killed:
            raise RuntimeError(
                "component posterior is undefined after every alternative "
                "component assigns zero predictive density"
            )
        return np.exp(self._component_log_weights)

    @property
    def alternative_killed(self) -> bool:
        """Whether the alternative has assigned zero density to an observation."""

        return self._alternative_killed

    @property
    def log_bayes_factor_history(self) -> np.ndarray:
        return np.asarray(self._log_bayes_factor_history, dtype=float)

    def reset(self) -> None:
        """Reset all sequential state while retaining the model specification."""

        self._component_log_weights = self._component_log_prior.copy()
        self._log_bayes_factor = 0.0
        self._log_bayes_factor_history: list[float] = []
        self._n_observations = 0
        self._alternative_killed = False

    def _null_logpdf(self, observation: float) -> float:
        if self.scheme == "gumbel":
            return float(gumbel_null_logpdf(observation))
        return float(
            inverse_null_logpdf(observation, self.inverse_null_vocabulary_size)
        )

    def _signal_logpdf(self, observation: float) -> np.ndarray:
        if self.scheme == "inverse":
            return np.asarray(
                [inverse_alt_logpdf(observation, float(delta)) for delta in self.component_delta],
                dtype=float,
            )
        cache: dict[float, float] = {}
        for delta in np.unique(self.component_delta):
            probabilities = self._gumbel_probabilities(float(delta))
            cache[float(delta)] = float(
                gumbel_alt_logpdf_from_probs(observation, probabilities)
            )
        return np.asarray([cache[float(delta)] for delta in self.component_delta], dtype=float)

    def component_log_likelihood_ratios(self, observation: float) -> np.ndarray:
        """Return log likelihood ratios for every (Delta, rho) component."""

        observation = float(observation)
        log_f0 = self._null_logpdf(observation)
        signal_log_ratio = self._signal_logpdf(observation) - log_f0
        rho = self.component_rho
        with np.errstate(divide="ignore"):
            log_clean = np.log1p(-rho) + signal_log_ratio
            log_contaminated = np.log(rho)
        return np.logaddexp(log_clean, log_contaminated)

    def update(self, observation: float) -> float:
        """Consume one pivot and return the updated log Bayes factor."""

        component_log_ratios = self.component_log_likelihood_ratios(float(observation))
        if self._alternative_killed:
            predictive_log_ratio = -math.inf
        elif self.structure == "shared":
            predictive_log_ratio = float(
                logsumexp(self._component_log_weights + component_log_ratios)
            )
            if math.isfinite(predictive_log_ratio):
                self._component_log_weights += component_log_ratios - predictive_log_ratio
            else:
                self._alternative_killed = True
        else:
            predictive_log_ratio = float(
                logsumexp(self._component_log_prior + component_log_ratios)
            )

        self._log_bayes_factor += predictive_log_ratio
        self._n_observations += 1
        self._log_bayes_factor_history.append(self._log_bayes_factor)
        return self._log_bayes_factor

    def update_many(self, observations: Iterable[float]) -> float:
        """Consume an iterable of pivots and return the final log Bayes factor."""

        for observation in observations:
            self.update(float(observation))
        return self._log_bayes_factor

    def posterior_probability(self, prior_watermark: float | None = None) -> float:
        """Return P(H1 | data) for the supplied or configured prior probability."""

        prior = self.prior_watermark if prior_watermark is None else float(prior_watermark)
        if not (math.isfinite(prior) and 0.0 < prior < 1.0):
            raise ValueError("prior_watermark must lie strictly between zero and one")
        log_odds = math.log(prior) - math.log1p(-prior) + self._log_bayes_factor
        if log_odds >= 0.0:
            return 1.0 / (1.0 + math.exp(-log_odds))
        exponential = math.exp(log_odds)
        return exponential / (1.0 + exponential)

    def loss_threshold(
        self,
        false_positive_cost: float = 1.0,
        false_negative_cost: float = 1.0,
        *,
        prior_watermark: float | None = None,
    ) -> float:
        """Return the Bayes-factor threshold for declaring a watermark."""

        c_fp = float(false_positive_cost)
        c_fn = float(false_negative_cost)
        if not (math.isfinite(c_fp) and c_fp > 0.0):
            raise ValueError("false_positive_cost must be positive and finite")
        if not (math.isfinite(c_fn) and c_fn > 0.0):
            raise ValueError("false_negative_cost must be positive and finite")
        prior = self.prior_watermark if prior_watermark is None else float(prior_watermark)
        if not (math.isfinite(prior) and 0.0 < prior < 1.0):
            raise ValueError("prior_watermark must lie strictly between zero and one")
        return (c_fp / c_fn) * ((1.0 - prior) / prior)

    def declare_watermark(
        self,
        false_positive_cost: float = 1.0,
        false_negative_cost: float = 1.0,
        *,
        prior_watermark: float | None = None,
    ) -> bool:
        """Apply the posterior expected-loss decision rule."""

        threshold = self.loss_threshold(
            false_positive_cost,
            false_negative_cost,
            prior_watermark=prior_watermark,
        )
        return self._log_bayes_factor > math.log(threshold)

    def crosses_anytime_threshold(self, alpha: float = 0.05) -> bool:
        """Return whether BF >= 1/alpha at the current time.

        Exact finite-vocabulary inverse-pivot validity requires constructing
        the detector with ``inverse_null_vocabulary_size``.  Omitting it uses
        the triangular limiting null and gives only asymptotic calibration.
        """

        if not (math.isfinite(alpha) and 0.0 < alpha < 1.0):
            raise ValueError("alpha must lie strictly between zero and one")
        return self._log_bayes_factor >= -math.log(alpha)

    def first_crossing_time(self, alpha: float = 0.05) -> int | None:
        """Return the first 1-based time BF >= 1/alpha, or None."""

        if not (math.isfinite(alpha) and 0.0 < alpha < 1.0):
            raise ValueError("alpha must lie strictly between zero and one")
        indices = np.flatnonzero(self.log_bayes_factor_history >= -math.log(alpha))
        return None if indices.size == 0 else int(indices[0] + 1)


def simulate_gumbel_null(
    n: int, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Simulate n Gumbel pivots under H0."""

    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    generator = np.random.default_rng() if rng is None else rng
    return generator.uniform(size=n)


def simulate_gumbel_alternative(
    n: int,
    probabilities: ArrayLike,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate exact Gumbel pivots under H1 for a fixed NTP vector."""

    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    probs = _validated_probabilities(probabilities)
    generator = np.random.default_rng() if rng is None else rng
    selected = generator.choice(probs.size, size=n, p=probs)
    uniforms = generator.uniform(size=n)
    # Conditional on W=w, the pivot is Beta(1/P_w, 1), sampled as U**P_w.
    return uniforms ** probs[selected]


def simulate_inverse_null(
    n: int,
    rng: np.random.Generator | None = None,
    vocabulary_size: int | None = None,
) -> np.ndarray:
    """Simulate inverse pivots from the exact or limiting null.

    Omitting ``vocabulary_size`` simulates the continuous large-vocabulary
    limit ``|U-V|``.  Supplying ``M`` uses an independent uniform rank on the
    exact grid ``{0, 1/(M-1), ..., 1}``.
    """

    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    generator = np.random.default_rng() if rng is None else rng
    uniforms = generator.uniform(size=n)
    if vocabulary_size is None:
        ranks = generator.uniform(size=n)
    else:
        if not isinstance(vocabulary_size, int) or vocabulary_size < 2:
            raise ValueError("vocabulary_size must be an integer of at least 2")
        ranks = generator.integers(0, vocabulary_size, size=n) / (vocabulary_size - 1)
    return np.abs(uniforms - ranks)


def simulate_inverse_alternative(
    n: int,
    delta: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate from the large-vocabulary inverse-transform alternative."""

    if not (math.isfinite(delta) and 0.0 <= delta < 1.0):
        raise ValueError("delta must lie in [0, 1)")
    return (1.0 - delta) * simulate_inverse_null(n, rng)


__all__ = [
    "SequentialBayesDetector",
    "beta_grid_prior",
    "gumbel_alt_logpdf_from_probs",
    "gumbel_alt_pdf_from_probs",
    "gumbel_null_logpdf",
    "inverse_alt_logpdf",
    "inverse_alt_pdf",
    "inverse_null_logpdf",
    "inverse_null_pdf",
    "least_favorable_probabilities",
    "logsumexp",
    "simulate_gumbel_alternative",
    "simulate_gumbel_null",
    "simulate_inverse_alternative",
    "simulate_inverse_null",
    "spike_probabilities",
]
