"""Truncated goodness-of-fit (Tr-GoF) detection of \\citet{li2024robust}.

A faithful reimplementation of the statistic in the authors' released code
(github.com/lx10077/TrGoF, ``compute_score``), kept here so the comparison in
this study runs inside the same calibration and pairing harness as every other
rule.  ``test_trgof.py`` asserts agreement with the reference implementation.

Given pivots with an exact uniform null, form one-sided p-values, sort them
ascending, and maximise a phi-divergence between the empirical and null
occupancy of each order statistic:

    S_n(s) = n * max_t K_s(t/n, p_(t)),

over the *truncated*, one-sided index set ``p_(t) >= 1/n`` and ``t/n >= p_(t)``.
The first condition is the truncation the method is named for and the second is
the one-sided restriction written ``K_s^+`` in the paper: only an excess of
small p-values over the null counts as evidence.  ``s`` indexes the Cressie--Read
family; ``s = 2`` is Higher Criticism.

Unlike every sum-based score, this statistic is not additive over tokens, so it
must be recomputed from the whole prefix rather than accumulated.

Scope note.  The released implementation applies Tr-GoF to Gumbel-max pivots
only, via ``p = 1 - Y``.  Applying it to the inverse-transform pivot requires a
p-value for that pivot, which the authors do not define; :func:`inverse_p_values`
supplies the natural one-sided choice, ``p = F_0(d)`` with the exact
finite-vocabulary null, because small ``d`` is the evidence direction there.
That extension is ours and is labelled as such wherever it is reported.
"""

from __future__ import annotations

import numpy as np

# The reference implementation's numerical guard, kept at its released value so
# the statistics agree term by term.
ACCURACY_EPS = 1e-10

# The Cressie--Read indices the authors sweep in their simulation code.
DEFAULT_S_VALUES: tuple[float, ...] = (2.0, 1.5, 1.0, 0.5, 0.0)
# s = 2 is Higher Criticism and is the value their figures lead with.
DEFAULT_S = 2.0


def _divergence(rank: np.ndarray, p: np.ndarray, s: float, eps: float) -> np.ndarray:
    """``K_s(rank, p)`` on the Cressie--Read scale, branch for branch as released."""

    if s == 1:
        return (
            rank * np.log(rank + eps)
            - rank * np.log(p + eps)
            + (1 - rank + eps) * np.log(1 - rank + eps)
            - (1 - rank) * np.log(1 - p + eps)
        )
    if s == 0:
        return (
            p * np.log(p + eps)
            - p * np.log(rank + eps)
            + (1 - p + eps) * np.log(1 - p + eps)
            - (1 - p) * np.log(1 - rank + eps)
        )
    if s == 2:
        return (rank - p) ** 2 / (p * (1 - p) + eps) / 2
    if s == 0.5:
        return 2 * (np.sqrt(rank) - np.sqrt(p)) ** 2 + 2 * (
            np.sqrt(1 - rank) - np.sqrt(1 - p)
        ) ** 2
    if s >= 0:
        return (
            1
            - (rank**s) * (p + eps) ** (1 - s)
            - ((1 - rank) ** s) * ((1 - p + eps) ** (1 - s))
        ) / (s * (1 - s))
    if s == -1:
        return (rank - p) ** 2 / (rank * (1 - rank) + eps) / 2
    return (
        1
        - p ** (1 - s) / (rank + eps) ** (-s)
        - (1 - p) ** (1 - s) / (1 - rank + eps) ** (-s)
    ) / (s * (1 - s))


def statistic(
    p_values: np.ndarray,
    *,
    s: float = DEFAULT_S,
    truncate: bool = True,
    eps: float = ACCURACY_EPS,
) -> np.ndarray:
    """Tr-GoF statistic for each row of ``p_values``.

    ``p_values`` has shape ``(rows, n)``; the statistic is computed per row.
    Rows whose truncated index set is empty score zero, which is the value the
    reference implementation's maximum degenerates to when nothing survives.
    """

    values = np.atleast_2d(np.asarray(p_values, dtype=float))
    rows, n = values.shape
    ordered = np.sort(values, axis=1)
    rank = (np.arange(1, n + 1) / n)[None, :]
    scores = _divergence(np.broadcast_to(rank, ordered.shape), ordered, s, eps)
    if truncate:
        keep = (ordered >= 1.0 / n) & (rank >= ordered)
        scores = np.where(keep, scores, -np.inf)
    best = scores.max(axis=1)
    return n * np.where(np.isfinite(best), best, 0.0)


def gumbel_p_values(pivots: np.ndarray) -> np.ndarray:
    """``p = 1 - Y`` for the Gumbel-max pivot, whose null is ``Unif(0,1)``."""

    return 1.0 - np.asarray(pivots, dtype=float)


def inverse_p_values(pivots: np.ndarray, vocabulary_size: int) -> np.ndarray:
    """``p = F_0(d)`` for the inverse-transform pivot at finite vocabulary.

    Small ``d`` is the evidence direction, so the one-sided p-value is the exact
    null CDF rather than its complement.  The exact null of ``|U - eta(I)|`` is
    piecewise linear with a closed-form CDF; using it keeps the p-values uniform
    under the null at finite ``M``.  Not part of the released implementation.
    """

    d = np.clip(np.asarray(pivots, dtype=float), 0.0, 1.0)
    m = int(vocabulary_size)
    # The exact null density is piecewise constant,
    #     f_0(x) = (2/M) (M - 1 - floor((M-1) x)),
    # so its integral is available in closed form.  With k = floor((M-1)d) whole
    # pieces plus a partial one, summing the arithmetic series gives
    #     F_0(d) = 2/(M(M-1)) [k(M-1) - k(k-1)/2] + (2/M)(M-1-k)(d - k/(M-1)).
    # Evaluating this per pivot avoids materialising an (n, M) grid, which at the
    # released vocabulary would be tens of gigabytes.
    k = np.minimum(np.floor((m - 1) * d), m - 1)
    whole = k * (m - 1) - k * (k - 1) / 2.0
    partial = (m - 1 - k) * (d - k / (m - 1))
    return np.clip(2.0 * whole / (m * (m - 1)) + 2.0 * partial / m, 0.0, 1.0)
