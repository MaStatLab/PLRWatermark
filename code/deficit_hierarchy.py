"""Partial pooling of the token-level deficit.

The study currently offers two hierarchies and nothing between them.  The
shared hierarchy draws one ``Delta`` for a whole document, so every token is
informative about the same scalar.  The tokenwise hierarchy draws ``Delta_t``
i.i.d. from a *fixed* prior, so tokens are informative about nothing in common
and the document likelihood factorizes.  Real next-token distributions sit
between: the deficit drifts with the context but is not constant.

This module adds the missing middle by giving the token-level prior free
parameters and integrating them at the document level:

    psi = (mu, kappa),      X_t | psi ~ Beta(kappa*mu, kappa*(1-mu))  i.i.d.,
    Delta_t = low + (high - low) * X_t,      psi ~ pi_psi.

Given ``psi`` the tokens are independent, so the per-token increment is again a
scalar function of the pivot; the coupling enters only when ``psi`` is
integrated after the token likelihoods are multiplied.  That is exactly the
shared hierarchy's integration order applied to a hyperparameter rather than to
``Delta`` itself, so the existing shared-path machinery carries over unchanged.

BOTH EXISTING HIERARCHIES ARE MEMBERS, not approximations:

  * ``kappa = 2, mu = 1/2`` is ``Beta(1,1)``, i.e. ``Delta_t`` i.i.d. uniform on
    the working support.  That single atom reproduces the tokenwise rule.
  * ``kappa = inf`` degenerates ``X_t`` to ``mu`` for every ``t``, so the whole
    document shares one deficit.  Mixing that layer over ``mu`` with the
    quadrature weights reproduces the shared rule.

so the family contains both and the finite-grid evidence bound of the study
applies to it.  Anything the pooled rule gains over both endpoints is the value
of partial pooling, and anything it loses is the prior dilution that buys the
extra flexibility.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import betaln, roots_jacobi

# kappa = 2 with mu = 1/2 is Beta(1,1); the tokenwise rule is that atom alone.
TOKENWISE_KAPPA = 2.0
TOKENWISE_MU = 0.5
DEFAULT_KAPPAS: tuple[float, ...] = (2.0, 8.0, 32.0, 128.0, math.inf)
DEFAULT_MU_NODES = 12
DEFAULT_BETA_NODES = 64


def beta_quadrature(
    a: float, b: float, nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and normalized weights for ``int_0^1 h(x) Beta(x; a, b) dx``.

    Gauss-Jacobi against the Beta kernel itself, so the rule is exact for
    polynomial ``h`` of degree below ``2 * nodes`` at every concentration.
    Gauss-Legendre would degrade badly as ``kappa`` grows and the density spikes.
    """

    if not (a > 0.0 and b > 0.0):
        raise ValueError("Beta parameters must be positive")
    # roots_jacobi weight on [-1,1] is (1-t)^alpha (1+t)^beta.
    t, w = roots_jacobi(int(nodes), b - 1.0, a - 1.0)
    x = 0.5 * (t + 1.0)
    log_scale = -(a + b - 1.0) * math.log(2.0) - betaln(a, b)
    weights = np.exp(np.log(np.maximum(w, np.finfo(float).tiny)) + log_scale)
    weights = np.where(w > 0.0, weights, 0.0)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"degenerate Beta quadrature at a={a}, b={b}")
    return x, weights / total


@dataclass(frozen=True)
class PooledDeficitPrior:
    """The ``psi`` grid: one tokenwise atom plus concentration layers."""

    kappas: tuple[float, ...] = DEFAULT_KAPPAS
    mu_nodes: int = DEFAULT_MU_NODES
    beta_nodes: int = DEFAULT_BETA_NODES

    def validate(self) -> None:
        if not self.kappas:
            raise ValueError("at least one concentration is required")
        if any(not (k > 0.0) for k in self.kappas):
            raise ValueError("concentrations must be positive (inf allowed)")
        if self.mu_nodes < 2 or self.beta_nodes < 2:
            raise ValueError("mu_nodes and beta_nodes must each be at least 2")

    def components(self) -> list[tuple[float, float, float]]:
        """``(mu, kappa, weight)`` triples, weights summing to one.

        The tokenwise atom is carried alone at ``mu = 1/2`` rather than swept
        over ``mu``: mixing ``Beta(1,1)`` over ``mu`` would not be the tokenwise
        rule, and reproducing that rule exactly is the point of including it.
        """

        self.validate()
        nodes, weights = np.polynomial.legendre.leggauss(self.mu_nodes)
        mus = 0.5 * (nodes + 1.0)
        mu_weights = 0.5 * weights
        mu_weights = mu_weights / mu_weights.sum()
        layers = len(self.kappas)
        out: list[tuple[float, float, float]] = []
        for kappa in self.kappas:
            share = 1.0 / layers
            if kappa == TOKENWISE_KAPPA:
                out.append((TOKENWISE_MU, kappa, share))
                continue
            for mu, w in zip(mus, mu_weights):
                out.append((float(mu), float(kappa), share * float(w)))
        total = sum(w for _, _, w in out)
        return [(mu, k, w / total) for mu, k, w in out]


class PooledDeficitGrid:
    """Per-token log likelihood ratio of each ``psi`` component, tabulated.

    ``component_log_ratio(r)`` returns ``log g_psi(r)`` with

        g_psi(r) = int_0^1 f(r | Delta(x)) Beta(x; kappa*mu, kappa*(1-mu)) dx,

    and ``f`` the equal-tail spike density.  Under the exact pivot null
    ``f_0 = 1``, so this is already the log ratio.
    """

    def __init__(
        self,
        *,
        vocabulary_size: int,
        delta_low: float,
        delta_high: float,
        prior: PooledDeficitPrior | None = None,
    ) -> None:
        if vocabulary_size < 2:
            raise ValueError("vocabulary_size must be at least 2")
        if not 0.0 <= delta_low < delta_high < 1.0:
            raise ValueError("require 0 <= delta_low < delta_high < 1")
        self.vocabulary_size = int(vocabulary_size)
        self.delta_low = float(delta_low)
        self.delta_high = float(delta_high)
        self.prior = prior or PooledDeficitPrior()
        self.triples = self.prior.components()
        self.log_component_weights = np.log(
            np.array([w for _, _, w in self.triples], dtype=float)
        )
        self._deltas: list[np.ndarray] = []
        self._weights: list[np.ndarray] = []
        for mu, kappa, _ in self.triples:
            if math.isinf(kappa):
                x = np.array([mu], dtype=float)
                w = np.array([1.0], dtype=float)
            else:
                x, w = beta_quadrature(
                    kappa * mu, kappa * (1.0 - mu), self.prior.beta_nodes
                )
            self._deltas.append(
                self.delta_low + (self.delta_high - self.delta_low) * x
            )
            self._weights.append(w)

    def n_components(self) -> int:
        return len(self.triples)

    def _spike_log_density(self, r: np.ndarray, deltas: np.ndarray) -> np.ndarray:
        """``log f(r | Delta)`` for the equal-tail spike, shape (r, Delta)."""

        log_r = np.log(np.maximum(r, np.finfo(float).tiny))[:, None]
        top = (deltas / (1.0 - deltas))[None, :] * log_r
        tail = (
            math.log(self.vocabulary_size - 1.0)
            + ((self.vocabulary_size - 1.0) / deltas - 1.0)[None, :] * log_r
        )
        return np.logaddexp(top, tail)

    def component_log_ratio(self, pivots: np.ndarray) -> np.ndarray:
        r = np.asarray(pivots, dtype=float).reshape(-1)
        out = np.empty((r.size, self.n_components()), dtype=float)
        for j, (deltas, weights) in enumerate(zip(self._deltas, self._weights)):
            dens = self._spike_log_density(r, deltas)
            out[:, j] = _logsumexp_rows(dens + np.log(weights)[None, :])
        return out

    def shared_paths(self, pivots: np.ndarray) -> np.ndarray:
        """Document log Bayes factor at every prefix, shape (rows, horizon).

        Token increments are accumulated per component and the components are
        mixed only afterwards.  That ordering is what shares information across
        tokens: a document whose early pivots favour a concentrated ``psi``
        reweights the components the later tokens are scored under.
        """

        values = np.asarray(pivots, dtype=float)
        rows, horizon = values.shape
        per_token = self.component_log_ratio(values.reshape(-1))
        per_token = per_token.reshape(rows, horizon, self.n_components())
        cumulative = np.cumsum(per_token, axis=1)
        return _logsumexp_rows(
            cumulative + self.log_component_weights[None, None, :]
        )

    def prior_fingerprint(self) -> dict[str, object]:
        return {
            "kappas": [
                "inf" if math.isinf(k) else float(k) for k in self.prior.kappas
            ],
            "mu_nodes": int(self.prior.mu_nodes),
            "beta_nodes": int(self.prior.beta_nodes),
            "n_components": self.n_components(),
            "delta_support": [self.delta_low, self.delta_high],
            "tokenwise_atom": [TOKENWISE_MU, TOKENWISE_KAPPA],
        }


def _logsumexp_rows(values: np.ndarray) -> np.ndarray:
    peak = np.max(values, axis=-1, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    return (peak + np.log(np.sum(np.exp(values - peak), axis=-1, keepdims=True)))[
        ..., 0
    ]
