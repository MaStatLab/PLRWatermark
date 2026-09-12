"""Partial pooling of the tail width.

The deficit analogue is in ``deficit_hierarchy``; this is the same construction
for the other coordinate.  The union tail of \\eqref{eq:union-tail} carries a
FIXED prior ``pi_J`` on the width ladder, so the width is either one number for
a whole document (the shared hierarchy) or redrawn at every token (the
tokenwise one).  A deployed tail is neither: the number of live coordinates
moves with the context but not independently between neighbouring tokens.

Give the width prior free parameters and integrate them at the document level.
With the frozen ladder ``J_1 < ... < J_L`` and ``phi = (c, lambda)``,

    Pr(J_t = J_k | phi) proportional to exp(-lambda * |k - c|),   i.i.d. over t,

so ``lambda`` is a persistence dial on the ladder index.  Note which end is
which -- an earlier version of this docstring had them the wrong way round:

  * ``lambda = inf`` pins every token of a document to the rung ``c``, so
    mixing that layer over the rungs with equal weights IS the frozen
    document-level ``pi_J`` of the union tail.  ``PooledWidthEndpointTests``
    checks the two agree to 1e-10.
  * ``lambda = 0`` is uniform on the ladder INDEPENDENTLY AT EVERY TOKEN.  It
    shares the frozen prior's marginal but not its hierarchy: the frozen rule
    draws one width per document, this one redraws per token.  It is a hybrid,
    since the deficit stays at the document level either way, so it is not the
    tokenwise union rule either.

As in the deficit case the tokens are conditionally independent given ``phi``,
so the per-token increment stays a scalar function of the pivot and the
coupling enters only when ``phi`` is integrated after the token likelihoods are
multiplied.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import dirichlet_detector as dd

DEFAULT_LAMBDAS: tuple[float, ...] = (0.0, 0.5, 1.5, 4.0, math.inf)


def ladder_weights(rungs: int, centre: int, lam: float) -> np.ndarray:
    """Normalized ``exp(-lambda |k - centre|)`` over the ladder."""

    k = np.arange(rungs, dtype=float)
    if math.isinf(lam):
        w = np.zeros(rungs, dtype=float)
        w[int(centre)] = 1.0
        return w
    logw = -float(lam) * np.abs(k - float(centre))
    w = np.exp(logw - logw.max())
    return w / w.sum()


@dataclass(frozen=True)
class PooledWidthPrior:
    lambdas: tuple[float, ...] = DEFAULT_LAMBDAS

    def validate(self) -> None:
        if not self.lambdas:
            raise ValueError("at least one concentration is required")
        if any(l < 0.0 for l in self.lambdas):
            raise ValueError("concentrations must be nonnegative (inf allowed)")

    def components(self, rungs: int) -> list[tuple[int, float, float]]:
        """``(centre, lambda, weight)`` triples, weights summing to one.

        ``lambda = 0`` is carried once, at a single centre: the distribution
        does not depend on the centre there, so sweeping it would multiply one
        model by the number of rungs and silently reweight the prior toward the
        uniform member.
        """

        self.validate()
        layers = len(self.lambdas)
        out: list[tuple[int, float, float]] = []
        for lam in self.lambdas:
            share = 1.0 / layers
            if lam == 0.0:
                out.append((0, 0.0, share))
                continue
            for centre in range(rungs):
                out.append((centre, float(lam), share / rungs))
        total = sum(w for _, _, w in out)
        return [(c, l, w / total) for c, l, w in out]


class PooledWidthGrid:
    """Document-level ``(Delta, phi)``; the width is redrawn inside a document.

    The deficit stays a document-level coordinate, exactly as in the shared
    hierarchy, and only ``J`` moves with ``phi``.  Marginalizing ``Delta`` per
    token instead would change a second coordinate at the same time and the
    ``lambda = inf`` limit would no longer reproduce the existing width block,
    which is how a first version of this class was caught: it disagreed with
    that block by 20 nats.

        log BF = logsumexp_{i,j} [ log w_i + log v_j
                                   + sum_t log g_{Delta_i, phi_j}(r_t) ],
        g_{Delta, phi}(r) = sum_k Pr(J_k | phi) f_{Delta, J_k}(r).
    """

    def __init__(
        self,
        *,
        delta_grid: np.ndarray,
        delta_weights: np.ndarray,
        tail_size: int,
        prior: PooledWidthPrior | None = None,
    ) -> None:
        # The ladder must be the UNION'S width block, which excludes the
        # full-width rung so the equal tail is not counted in both branches.
        # Building on the default six-rung ladder pooled a different prior from
        # the canonical one, and the endpoint then reproduced the wrong object.
        ladder = tuple(
            j for j in dd.dyadic_tail_width_grid(int(tail_size))
            if j != int(tail_size)
        )
        self.block = dd.TailWidthBayesGrid(
            delta_grid=delta_grid, delta_weights=delta_weights,
            tail_size=int(tail_size), tail_width_grid=ladder,
        )
        self.tail_widths = tuple(int(j) for j in self.block.tail_widths)
        self.deltas = np.asarray(delta_grid, dtype=float)
        self.delta_log_weights = np.log(
            np.asarray(delta_weights, dtype=float)
            / np.asarray(delta_weights, dtype=float).sum()
        )
        self.prior = prior or PooledWidthPrior()
        self.phi = self.prior.components(len(self.tail_widths))
        self.phi_log_weights = np.log(
            np.array([w for _, _, w in self.phi], dtype=float)
        )
        # (Delta, phi) components, Delta varying fastest.
        self.log_component_weights = (
            self.delta_log_weights[:, None] + self.phi_log_weights[None, :]
        ).reshape(-1)
        self._ladder = np.stack(
            [ladder_weights(len(self.tail_widths), c, l) for c, l, _ in self.phi]
        )  # (phi, J)

    def n_components(self) -> int:
        return self.deltas.size * len(self.phi)

    def component_log_ratio(self, pivots: np.ndarray) -> np.ndarray:
        """``log g_{Delta, phi}(r)``, columns ordered Delta-fastest.

        Materializes ``(r, phi, Delta)``.  Only safe for small ``r``; the
        document scorer below never calls it on a whole experiment.
        """

        r = np.asarray(pivots, dtype=float).reshape(-1)
        per = self.block.component_log_ratio(r).reshape(
            r.size, len(self.tail_widths), self.deltas.size
        )
        out = np.empty((r.size, len(self.phi), self.deltas.size), dtype=float)
        with np.errstate(divide="ignore"):
            log_ladder = np.log(self._ladder)
        for j, w in enumerate(log_ladder):
            live = np.isfinite(w)
            out[:, j, :] = dd._logsumexp_rows(
                np.moveaxis(per[:, live, :] + w[live][None, :, None], 1, -1)
            )
        return np.moveaxis(out, 1, 2).reshape(r.size, -1)

    def shared_paths(
        self, pivots: np.ndarray, *, horizons=None, batch_rows: int = 250
    ) -> np.ndarray:
        """Document log Bayes factor at ``horizons`` (default: every prefix).

        Streams over documents and over ``phi``.  The naive version built an
        ``(r, phi, J, Delta)`` array, which at 5,000 documents by 700 tokens is
        hundreds of gigabytes; it was killed by the OS with no traceback, which
        is what an exit code of zero and an empty log look like.  Here only one
        ``phi`` slab is resident at a time and the components are folded into a
        running log-sum-exp.
        """

        values = np.asarray(pivots, dtype=float)
        rows, horizon = values.shape
        want = tuple(range(1, horizon + 1)) if horizons is None else tuple(horizons)
        columns = np.asarray(want, dtype=int) - 1
        n_delta = self.deltas.size
        with np.errstate(divide="ignore"):
            log_ladder = np.log(self._ladder)
        out = np.full((rows, columns.size), -np.inf, dtype=float)
        for start in range(0, rows, batch_rows):
            stop = min(start + batch_rows, rows)
            block = values[start:stop]
            flat = block.reshape(-1)
            per = self.block.component_log_ratio(flat).reshape(
                flat.size, len(self.tail_widths), n_delta
            )
            acc = np.full((stop - start, columns.size), -np.inf, dtype=float)
            for j, (w, (_, _, weight)) in enumerate(zip(log_ladder, self.phi)):
                live = np.isfinite(w)
                ratio = dd._logsumexp_rows(
                    np.moveaxis(per[:, live, :] + w[live][None, :, None], 1, -1)
                ).reshape(stop - start, horizon, n_delta)
                cumulative = np.cumsum(ratio, axis=1)[:, columns, :]
                term = dd._logsumexp_rows(
                    cumulative + self.delta_log_weights[None, None, :]
                ) + math.log(weight)
                acc = np.logaddexp(acc, term)
            out[start:stop] = acc
        return out

    def prior_fingerprint(self) -> dict[str, object]:
        return {
            "lambdas": ["inf" if math.isinf(l) else float(l)
                        for l in self.prior.lambdas],
            "ladder": [int(j) for j in self.tail_widths],
            "n_delta_nodes": int(self.deltas.size),
            "n_phi": len(self.phi),
            "n_components": self.n_components(),
        }
