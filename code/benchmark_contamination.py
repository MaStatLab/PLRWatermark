#!/usr/bin/env python3
"""Robustness benchmark for iid null-like watermark contamination.

The experiment extends the shared-Delta, equal-tail sensitivity study in
``benchmark_paper_experiment.py``.  For every document it generates a clean
watermarked pivot path, an independent exact-null replacement path, and one
uniform mask path.  At true contamination rate rho, positions with mask <= rho
are replaced by null pivots.  Within each pivot scheme, reusing these three
arrays makes all methods and contamination levels paired.

Every fixed-horizon statistic is calibrated once using its scheme's independent
exact-null sample and uses that same cutoff at every contamination rate.  The primary
Bayesian comparison is between a clean shared model (rho=0) and a robust shared
model with the frozen spike-and-grid prior

    0.5 delta_0 + 0.125(delta_.1 + delta_.25 + delta_.4 + delta_.6).

The tokenwise robust score is retained only as a hierarchy-misspecification
sensitivity: integrating rho independently at each token collapses to E[rho]
and cannot learn a document-level contamination rate.

A second family of diagnostic detectors replaces the spike-and-grid prior by a
point mass ``pi_rho = delta_{rho_0}`` while holding everything else fixed: the
same Delta ~ Unif(delta_low, delta_high) with the same Gauss-Legendre nodes, the
same shared hierarchy, the same paired contamination paths, and the same
per-method 5% null calibration reused across the rho_true sweep.  They answer
whether the robust rule's advantage comes from *averaging* over pi_rho or merely
from assuming *some* positive contamination rate.

Two enlarged Gumbel component families are carried alongside the spike family,
each in a clean (rho=0) and a robust (spike-and-grid pi_rho) variant: the
Dirichlet tail-shape layer and the tail-width layer, which puts a prior on the
number of live tail coordinates J instead of fixing the tail equal over all V-1.
Both contain the equal-tail spike as a component, so neither can be a different
model of the same data by accident.

The non-Bayesian competitor is Tr-GoF (\\citet{li2024robust}, ``code/trgof.py``),
carried for both pivot schemes at the Higher Criticism index s=2.  It is a
robust goodness-of-fit test designed for precisely this contamination problem,
which makes it the sharpest comparator available here.  It is not a token sum --
it sorts the p-values of the whole prefix -- so it is recomputed at every
reported horizon and calibrated through the same 5% randomized-boundary path as
every other rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from experiment_metadata import union_tail_metadata

import benchmark_paper_experiment as paper
import dirichlet_detector as dirichlet
import trgof


DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "bayesian_paper_benchmark"
)
DEFAULT_QUICK_RESULTS_DIR = DEFAULT_RESULTS_DIR / "quick" / "contamination"

# The contamination benchmark uses the same randomized-boundary decision rule
# as the clean benchmark, but its addressable auxiliary RNG must also separate
# true-contamination strata.  A dedicated salt keeps these uniforms independent
# of every pivot stream (and of the clean benchmark's auxiliary uniforms).
BOUNDARY_RANDOMIZATION_RULE_VERSION = paper.BOUNDARY_RANDOMIZATION_RULE_VERSION
BOUNDARY_RANDOMIZATION_SEED_SALT = 0x434F4E54  # ASCII "CONT"
_BOUNDARY_SCHEME_CODES = {"gumbel": 0, "inverse": 1}


GUMBEL_PAPER_METHODS = (
    "h_ars",
    "h_log",
    "h_ind_1_over_e",
    "h_gum_star_0.1",
    "h_gum_star_0.01",
    "h_gum_star_0.005",
)
INVERSE_PAPER_METHODS = (
    "h_neg",
    "h_dif_star_0.1",
    "h_dif_star_0.01",
    "h_dif_star_0.001",
)
BAYES_METHODS = (
    "bayes_tokenwise_clean",
    "bayes_tokenwise_robust",
    "bayes_shared_clean",
    "bayes_shared_robust",
)

# The Dirichlet tail layer adds a third latent axis, the tail concentration
# alpha, to the shared-Delta hierarchy.  Gumbel only: for the inverse pivot the
# limiting alternative depends on Delta alone, so a prior on the tail shape is
# inert by construction and there is nothing to add.
DIRICHLET_CLEAN_METHOD = "bayes_shared_dirichlet_clean"
DIRICHLET_ROBUST_METHOD = "bayes_shared_dirichlet_robust"
DIRICHLET_METHODS = (DIRICHLET_CLEAN_METHOD, DIRICHLET_ROBUST_METHOD)
DIRICHLET_SCHEMES = ("gumbel",)

# The tail-width layer is the other way of enlarging the Gumbel component
# family: instead of fixing the tail equal over all V-1 coordinates it puts a
# prior on the number of live tail coordinates J.  J = V-1 is exactly the
# equal-tail spike every other Gumbel rule assumes, so the family *contains*
# them.  The component
#
#     f_{Delta,J}(r) = r**(Delta/(1-Delta)) + J r**(J/Delta - 1)
#
# is closed form, so unlike the Dirichlet tail-shape layer it needs no
# transform table and no interpolation diagnostic.  Shared hierarchy only: the
# tokenwise rule averages over J before multiplying and therefore cannot learn
# a document-level tail width.  Gumbel only, for the same reason as the
# Dirichlet layer -- the inverse limiting alternative depends on Delta alone.
UNIONTAIL_CLEAN_METHOD = "bayes_shared_uniontail_clean"
UNIONTAIL_ROBUST_METHOD = "bayes_shared_uniontail_robust"
UNIONTAIL_METHODS = (UNIONTAIL_CLEAN_METHOD, UNIONTAIL_ROBUST_METHOD)
UNIONTAIL_SCHEMES = ("gumbel",)

# Tr-GoF of \citet{li2024robust}: the truncated goodness-of-fit competitor built
# for precisely this question, a test designed to stay powerful when a fraction
# of the pivots is replaced by null-like draws.  It is the natural non-Bayesian
# comparator for this benchmark and is carried for both pivot schemes.  The
# statistic sorts the p-values of the whole prefix and maximises a phi-divergence
# over the truncated order statistics, so unlike every other rule here it is not
# a token sum and cannot be accumulated; each horizon is recomputed.  The single
# reported index is s = 2 (Higher Criticism), the value the authors' figures lead
# with; the remaining indices of their sweep are recorded as a sensitivity in the
# run metadata rather than added as separate rules.
TRGOF_METHOD = f"trgof_s{trgof.DEFAULT_S:g}"
TRGOF_METHODS = (TRGOF_METHOD,)
TRGOF_SCHEMES = ("gumbel", "inverse")

# Diagnostic-only shared-Bayes detectors with pi_rho = delta_{rho_0}.
POINT_MASS_PREFIX = "bayes_shared_rho0_"

# The grid originally asked for.  It stops at 0.4, below both the top of the
# robust prior's support (0.6) and the top of the contamination sweep, so on its
# own it understates what a fixed rho_0 can do at large rho_true.
REQUESTED_RHO_POINT_MASSES = (0.1, 0.15, 0.25, 0.4)

# The default extends that grid to span the prior's support, so that the
# "best fixed rho_0" comparison is not decided by where the grid was truncated.
DEFAULT_RHO_POINT_MASSES = REQUESTED_RHO_POINT_MASSES + (0.5, 0.6)

# Prespecified reference scores, frozen from the independent clean n=700 benchmark.
PRESPECIFIED_PAPER_SCORE = {
    "gumbel": "h_gum_star_0.005",
    "inverse": "h_dif_star_0.01",
}


def point_mass_method(rho_zero: float) -> str:
    """Stable method name for the pi_rho = delta_{rho_0} diagnostic detector."""

    return f"{POINT_MASS_PREFIX}{float(rho_zero):g}"


def is_point_mass_method(method: str) -> bool:
    return method.startswith(POINT_MASS_PREFIX)


def indicator_key(scheme: str, rho_true: float, horizon: int, method: str) -> str:
    """Stable flat key for persisted per-document rejection indicators."""

    return f"{scheme}|{float(rho_true):g}|{int(horizon)}|{method}"


def _rho_float64_words(rho_true: float) -> tuple[int, int]:
    """Return the low/high 32-bit words of rho's IEEE-754 representation."""

    rho = float(rho_true)
    if not math.isfinite(rho) or not 0.0 <= rho <= 1.0:
        raise ValueError("rho_true must lie in [0,1]")
    bits = int(np.asarray(rho, dtype=np.float64).view(np.uint64))
    return bits & 0xFFFFFFFF, bits >> 32


def _boundary_randomization_seed_entropy(
    seed: int, scheme: str, rho_true: float, horizon: int
) -> tuple[int, ...]:
    """Stable SeedSequence entropy for one contaminated paired cell.

    The method is deliberately absent from the address, so all methods use the
    same auxiliary uniform for a given scheme/rho/horizon/document.  Encoding
    rho by its two binary64 words avoids unstable string hashes or rounding.
    """

    if scheme not in _BOUNDARY_SCHEME_CODES:
        raise ValueError("scheme must be 'gumbel' or 'inverse'")
    if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(horizon, (int, np.integer)) or int(horizon) < 1:
        raise ValueError("horizon must be a positive integer")
    rho_low, rho_high = _rho_float64_words(rho_true)
    return (
        int(seed),
        BOUNDARY_RANDOMIZATION_SEED_SALT,
        _BOUNDARY_SCHEME_CODES[scheme],
        rho_low,
        rho_high,
        int(horizon),
    )


def boundary_randomization_uniforms(
    *,
    seed: int,
    scheme: str,
    rho_true: float,
    horizon: int,
    n_documents: int,
) -> np.ndarray:
    """Generate the paired auxiliary uniforms for one contaminated cell."""

    if not isinstance(n_documents, (int, np.integer)) or int(n_documents) < 0:
        raise ValueError("n_documents must be a non-negative integer")
    entropy = _boundary_randomization_seed_entropy(
        seed, scheme, rho_true, horizon
    )
    return np.random.default_rng(np.random.SeedSequence(entropy)).random(
        int(n_documents)
    )


@dataclass(frozen=True)
class ContaminationConfig:
    vocabulary_size: int = 1000
    horizons: tuple[int, ...] = (100, 300, 700)
    alpha: float = 0.05
    # Generating law, unchanged.
    delta_low: float = 0.001
    delta_high: float = 0.5
    # Detector prior support, kept separate from the generator.
    prior_low: float = 0.001
    prior_high: float = 0.5
    rho_true_grid: tuple[float, ...] = (0.0, 0.1, 0.25, 0.4, 0.6, 1.0)
    rho_prior_grid: tuple[float, ...] = (0.0, 0.1, 0.25, 0.4, 0.6)
    rho_prior_weights: tuple[float, ...] = (0.5, 0.125, 0.125, 0.125, 0.125)
    rho_point_masses: tuple[float, ...] = DEFAULT_RHO_POINT_MASSES
    n_calibration: int = 10_000
    n_evaluation_null: int = 5_000
    n_evaluation_alternative: int = 5_000
    batch_size: int = 250
    seed: int = 24_040_1246
    bayes_quadrature_nodes: int = 96
    gumbel_lookup_size: int = 80_001
    gumbel_lookup_logit_limit: float = 30.0
    dirichlet_c_nodes: int = dirichlet.DEFAULT_C_NODES
    dirichlet_alpha_grid: tuple[float, ...] = dirichlet.DEFAULT_ALPHA_GRID

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)

    def paper_config(self) -> paper.BenchmarkConfig:
        return paper.BenchmarkConfig(
            vocabulary_size=self.vocabulary_size,
            max_horizon=self.max_horizon,
            alpha=self.alpha,
            delta_low=self.delta_low,
            delta_high=self.delta_high,
            prior_low=self.prior_low,
            prior_high=self.prior_high,
            n_calibration=self.n_calibration,
            n_evaluation_null=self.n_evaluation_null,
            n_evaluation_alternative=self.n_evaluation_alternative,
            batch_size=self.batch_size,
            seed=self.seed,
            bayes_quadrature_nodes=self.bayes_quadrature_nodes,
            gumbel_lookup_size=self.gumbel_lookup_size,
            gumbel_lookup_logit_limit=self.gumbel_lookup_logit_limit,
            dirichlet_c_nodes=self.dirichlet_c_nodes,
        )

    def dirichlet_grid(self) -> dirichlet.DirichletBayesGrid:
        """The joint (Delta, alpha, rho) prior behind the Dirichlet rules.

        The Delta and rho factors are exactly the ones the spike-family shared
        rules use, so the Dirichlet rules differ from them only by carrying the
        tail-concentration axis.  The alpha grid includes the equal-tail limit,
        which makes the existing shared rules components of these mixtures.
        """

        deltas, weights = dirichlet.gauss_legendre_delta_grid(
            self.prior_low, self.prior_high, self.bayes_quadrature_nodes
        )
        return dirichlet.DirichletBayesGrid(
            delta_grid=deltas,
            delta_weights=weights,
            alpha_grid=self.dirichlet_alpha_grid,
            tail_size=self.vocabulary_size - 1,
            allow_large_delta=False,
            rho_grid=self.rho_prior_grid,
            rho_weights=self.rho_prior_weights,
            c_nodes=self.dirichlet_c_nodes,
        )

    def uniontail_grid(self) -> dirichlet.UnionTailBayesGrid:
        """The joint (Delta, J, rho) prior behind the tail-width rules.

        The Delta factor uses the same Gauss-Legendre call as
        :meth:`dirichlet_grid`, and the rho factor is the identical
        spike-and-grid prior, so these rules differ from the existing shared
        rules only by carrying a latent tail width.  The width factor is
        uniform on the default dyadic ladder ``1, 4, 16, ..., V-1``, whose last
        atom is the equal-tail spike; that makes the existing shared rules
        components of these mixtures.
        """

        deltas, weights = dirichlet.gauss_legendre_delta_grid(
            self.prior_low, self.prior_high, self.bayes_quadrature_nodes
        )
        return dirichlet.UnionTailBayesGrid(
            delta_grid=deltas,
            delta_weights=weights,
            tail_size=self.vocabulary_size - 1,
            allow_large_delta=False,
            alpha_grid=self.dirichlet_alpha_grid,
            tail_width_grid=None,
            rho_grid=self.rho_prior_grid,
            rho_weights=self.rho_prior_weights,
            c_nodes=self.dirichlet_c_nodes,
        )

    def validate(self) -> None:
        self.paper_config().validate()
        if self.dirichlet_c_nodes < 16:
            raise ValueError("dirichlet_c_nodes must be at least 16")
        if not self.dirichlet_alpha_grid:
            raise ValueError("dirichlet_alpha_grid must be nonempty")
        if not self.horizons or tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be strictly increasing")
        if self.horizons[0] < 1:
            raise ValueError("horizons must be positive")
        true_rhos = np.asarray(self.rho_true_grid, dtype=float)
        prior_rhos = np.asarray(self.rho_prior_grid, dtype=float)
        prior_weights = np.asarray(self.rho_prior_weights, dtype=float)
        if np.any((true_rhos < 0.0) | (true_rhos > 1.0)):
            raise ValueError("rho_true_grid must lie in [0,1]")
        if not np.any(np.isclose(true_rhos, 1.0)):
            raise ValueError("rho_true_grid must include 1.0 for the null-equivalence diagnostic")
        if np.any((prior_rhos < 0.0) | (prior_rhos >= 1.0)):
            raise ValueError("rho_prior_grid must lie in [0,1)")
        if prior_rhos.size != prior_weights.size:
            raise ValueError("rho prior grid and weights must have equal length")
        if np.any(prior_weights <= 0.0) or not np.isclose(prior_weights.sum(), 1.0):
            raise ValueError("rho prior weights must be positive and sum to one")
        point_masses = np.asarray(self.rho_point_masses, dtype=float)
        if point_masses.size:
            if np.any((point_masses < 0.0) | (point_masses >= 1.0)):
                raise ValueError("rho_point_masses must lie in [0,1)")
            if np.unique(point_masses).size != point_masses.size:
                raise ValueError("rho_point_masses must be distinct")
            names = {point_mass_method(value) for value in point_masses}
            if len(names) != point_masses.size:
                raise ValueError("rho_point_masses must have distinct method names")

    def point_mass_methods(self) -> tuple[str, ...]:
        return tuple(point_mass_method(value) for value in self.rho_point_masses)


def _json_safe_config(config: ContaminationConfig) -> dict[str, object]:
    """``asdict`` with the alpha grid rendered as strings.

    ``math.inf`` is a legitimate member of the tail-concentration grid -- it is
    the equal-tail limit -- but ``json.dumps`` writes it as the bare token
    ``Infinity``, which no strict JSON parser accepts.  Rendering the grid with
    :func:`dirichlet_detector.format_alpha` keeps the recorded config both
    faithful and valid JSON.
    """

    payload = asdict(config)
    payload["dirichlet_alpha_grid"] = [
        dirichlet.format_alpha(value) for value in config.dirichlet_alpha_grid
    ]
    return payload


def _logsumexp_rows(values: np.ndarray) -> np.ndarray:
    maximum = np.max(values, axis=1)
    result = np.full(values.shape[0], -np.inf)
    finite = np.isfinite(maximum)
    if np.any(finite):
        result[finite] = maximum[finite] + np.log(
            np.exp(values[finite] - maximum[finite, None]).sum(axis=1)
        )
    return result


def contaminated_pivots(
    clean: np.ndarray,
    null_replacements: np.ndarray,
    mask_uniforms: np.ndarray,
    rho_true: float,
) -> np.ndarray:
    """Apply a nested iid null-replacement mask to a clean pivot array."""

    if clean.shape != null_replacements.shape or clean.shape != mask_uniforms.shape:
        raise ValueError("clean, null replacement, and mask arrays must have equal shape")
    if not 0.0 <= rho_true <= 1.0:
        raise ValueError("rho_true must lie in [0,1]")
    if rho_true == 0.0:
        return clean.copy()
    if rho_true == 1.0:
        return null_replacements.copy()
    return np.where(mask_uniforms < rho_true, null_replacements, clean)


def _delta_quadrature(config: ContaminationConfig) -> tuple[np.ndarray, np.ndarray]:
    """Detector prior quadrature, clamped to what the vocabulary can represent."""
    nodes, weights = np.polynomial.legendre.leggauss(config.bayes_quadrature_nodes)
    low = config.prior_low
    high = min(config.prior_high, 1.0 - 1.0 / config.vocabulary_size)
    deltas = low + 0.5 * (nodes + 1.0) * (high - low)
    return deltas, 0.5 * weights


def _component_log_ratios(
    observations: np.ndarray,
    scheme: str,
    deltas: np.ndarray,
    config: ContaminationConfig,
) -> np.ndarray:
    observation = observations[:, None]
    if scheme == "gumbel":
        log_r = np.log(observation)
        top_exponents = deltas / (1.0 - deltas)
        tail_exponents = (config.vocabulary_size - 1.0) / deltas - 1.0
        return np.logaddexp(
            log_r * top_exponents[None, :],
            math.log(config.vocabulary_size - 1.0)
            + log_r * tail_exponents[None, :],
        )
    if scheme == "inverse":
        scales = 1.0 - deltas
        inside = 1.0 - observation / scales[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            log_density = np.where(
                inside > 0.0,
                np.log(2.0 / scales)[None, :] + np.log(inside),
                -np.inf,
            )
        log_f0 = np.log(
            paper.inverse_exact_null_density(observations, config.vocabulary_size)
        )
        return log_density - log_f0[:, None]
    raise ValueError("scheme must be 'gumbel' or 'inverse'")


def _point_mass_plan(
    config: ContaminationConfig, rhos: np.ndarray
) -> tuple[list[tuple[str, bool, int]], np.ndarray]:
    """Route each rho_0 point mass to an existing prior column or an extra one.

    Returns ``(plan, extra_rhos)`` where each plan entry is
    ``(method_name, uses_prior_column, column_index)``.  Point masses that
    coincide with a node of ``rho_prior_grid`` reuse that node's accumulator
    slice, which is bit-for-bit the accumulation a standalone point mass would
    produce; only genuinely new rho_0 values need extra work.
    """

    plan: list[tuple[str, bool, int]] = []
    extra: list[float] = []
    for value in config.rho_point_masses:
        matches = np.flatnonzero(np.isclose(rhos, float(value)))
        if matches.size:
            plan.append((point_mass_method(value), True, int(matches[0])))
        else:
            plan.append((point_mass_method(value), False, len(extra)))
            extra.append(float(value))
    return plan, np.asarray(extra, dtype=float)


def shared_bayes_selected_paths(
    pivots: np.ndarray,
    scheme: str,
    config: ContaminationConfig,
) -> dict[str, np.ndarray]:
    """Selected-horizon shared Bayes paths with rho fixed or marginalized.

    Produces the clean (rho=0) rule, the mixture-prior robust rule, and one
    diagnostic rule per entry of ``config.rho_point_masses`` that uses
    ``pi_rho = delta_{rho_0}``.  Every rule shares the same Delta quadrature and
    the same shared-Delta hierarchy, so they differ only in pi_rho.
    """

    deltas, delta_weights = _delta_quadrature(config)
    rhos = np.asarray(config.rho_prior_grid, dtype=float)
    rho_weights = np.asarray(config.rho_prior_weights, dtype=float)
    log_delta_weights = np.log(delta_weights)
    log_joint_weights = (
        log_delta_weights[:, None] + np.log(rho_weights)[None, :]
    ).reshape(-1)
    with np.errstate(divide="ignore"):
        log_rho = np.log(rhos)
        log_one_minus_rho = np.log1p(-rhos)

    plan, extra_rhos = _point_mass_plan(config, rhos)
    with np.errstate(divide="ignore"):
        log_extra = np.log(extra_rhos)
        log_one_minus_extra = np.log1p(-extra_rhos)

    selected_indices = {horizon - 1: index for index, horizon in enumerate(config.horizons)}
    clean_output = np.empty((pivots.shape[0], len(config.horizons)), dtype=float)
    robust_output = np.empty_like(clean_output)
    point_outputs = {name: np.empty_like(clean_output) for name, _, _ in plan}

    for start in range(0, pivots.shape[0], config.batch_size):
        stop = min(start + config.batch_size, pivots.shape[0])
        batch = pivots[start:stop]
        cumulative_clean = np.zeros((batch.shape[0], deltas.size), dtype=float)
        cumulative_robust = np.zeros(
            (batch.shape[0], deltas.size, rhos.size), dtype=float
        )
        cumulative_extra = np.zeros(
            (batch.shape[0], deltas.size, extra_rhos.size), dtype=float
        )
        for time_index in range(config.max_horizon):
            component = _component_log_ratios(
                batch[:, time_index], scheme, deltas, config
            )
            cumulative_clean += component
            cumulative_robust += np.logaddexp(
                log_rho[None, None, :],
                log_one_minus_rho[None, None, :] + component[:, :, None],
            )
            if extra_rhos.size:
                cumulative_extra += np.logaddexp(
                    log_extra[None, None, :],
                    log_one_minus_extra[None, None, :] + component[:, :, None],
                )
            selected_column = selected_indices.get(time_index)
            if selected_column is not None:
                clean_output[start:stop, selected_column] = _logsumexp_rows(
                    cumulative_clean + log_delta_weights[None, :]
                )
                robust_output[start:stop, selected_column] = _logsumexp_rows(
                    cumulative_robust.reshape(batch.shape[0], -1)
                    + log_joint_weights[None, :]
                )
                for name, uses_prior, column in plan:
                    source = cumulative_robust if uses_prior else cumulative_extra
                    point_outputs[name][start:stop, selected_column] = (
                        _logsumexp_rows(
                            source[:, :, column] + log_delta_weights[None, :]
                        )
                    )
    paths = {
        "bayes_shared_clean": clean_output,
        "bayes_shared_robust": robust_output,
    }
    paths.update(point_outputs)
    return paths


def token_and_paper_selected_paths(
    pivots: np.ndarray,
    scheme: str,
    config: ContaminationConfig,
    lookup: paper.GumbelBayesLookup,
) -> dict[str, np.ndarray]:
    """Reference-score and tokenwise-Bayes sums at selected horizons."""

    paper_config = config.paper_config()
    methods = GUMBEL_PAPER_METHODS if scheme == "gumbel" else INVERSE_PAPER_METHODS
    selected = np.asarray(config.horizons, dtype=int) - 1
    paths: dict[str, np.ndarray] = {}
    for method in methods:
        if scheme == "gumbel":
            increments = paper.gumbel_score(pivots, method, lookup)
        else:
            increments = paper.inverse_score(pivots, method, paper_config)
        paths[method] = np.cumsum(increments, axis=1)[:, selected]

    if scheme == "gumbel":
        clean_log_ratio = lookup(pivots)
    else:
        clean_log_ratio = paper.inverse_bayes_log_ratio(pivots, paper_config)
    paths["bayes_tokenwise_clean"] = np.cumsum(clean_log_ratio, axis=1)[:, selected]

    mean_rho = float(
        np.dot(
            np.asarray(config.rho_prior_grid, dtype=float),
            np.asarray(config.rho_prior_weights, dtype=float),
        )
    )
    if mean_rho == 0.0:
        robust_log_ratio = clean_log_ratio
    else:
        robust_log_ratio = np.logaddexp(
            math.log(mean_rho), math.log1p(-mean_rho) + clean_log_ratio
        )
    paths["bayes_tokenwise_robust"] = np.cumsum(robust_log_ratio, axis=1)[:, selected]
    return paths


def trgof_selected_paths(
    pivots: np.ndarray,
    scheme: str,
    config: ContaminationConfig,
    *,
    s: float = trgof.DEFAULT_S,
) -> dict[str, np.ndarray]:
    """Tr-GoF scores at the selected horizons.

    Every other rule in this file is a sum over tokens, so its whole path is a
    cumulative sum of one increment array.  Tr-GoF is not: it sorts the
    p-values of the entire prefix and maximises a divergence over the truncated
    order statistics, which is not additive.  The statistic is therefore
    recomputed from scratch at each reported horizon, exactly as the shared
    Bayes rules are re-marginalised at each horizon.

    The p-value map is the authors' ``p = 1 - Y`` for the Gumbel pivot.  For the
    inverse pivot the released code offers none, so this uses the exact
    finite-vocabulary null CDF ``p = F_0(d)`` at ``config.vocabulary_size``;
    that extension is ours and is labelled as such in the run metadata.

    The returned scores enter the shared calibration dictionary unchanged, so
    Tr-GoF is cut at the same nominal level, off the same common exact-null
    sample, under the same randomised-boundary rule as every other method.  The
    randomisation is not cosmetic here: the statistic has an atom at zero,
    reached whenever the truncated index set of a prefix is empty.
    """

    if scheme == "gumbel":

        def to_p_values(values: np.ndarray) -> np.ndarray:
            return trgof.gumbel_p_values(values)

    elif scheme == "inverse":

        def to_p_values(values: np.ndarray) -> np.ndarray:
            return trgof.inverse_p_values(values, config.vocabulary_size)

    else:
        raise ValueError("scheme must be 'gumbel' or 'inverse'")

    output = np.empty((pivots.shape[0], len(config.horizons)), dtype=float)
    for start in range(0, pivots.shape[0], config.batch_size):
        stop = min(start + config.batch_size, pivots.shape[0])
        p_values = to_p_values(pivots[start:stop])
        for column, horizon in enumerate(config.horizons):
            output[start:stop, column] = trgof.statistic(
                p_values[:, :horizon], s=s
            )
    return {TRGOF_METHOD: output}


def dirichlet_shared_selected_paths(
    pivots: np.ndarray,
    scheme: str,
    config: ContaminationConfig,
    grid: dirichlet.DirichletBayesGrid | dirichlet.UnionTailBayesGrid,
    *,
    clean_method: str = DIRICHLET_CLEAN_METHOD,
    robust_method: str = DIRICHLET_ROBUST_METHOD,
    schemes: tuple[str, ...] = DIRICHLET_SCHEMES,
) -> dict[str, np.ndarray]:
    """Selected-horizon shared Bayes paths for a tail layer.

    Drives either the Dirichlet tail-shape grid or the tail-width grid; both
    expose the same shared-latent component protocol, so the accumulation is
    identical and only the emitted method names differ.

    One accumulation over the joint ``(tail, Delta, rho)`` grid serves both
    rules: the robust rule marginalises every component, while the clean rule
    reads off the ``rho = 0`` slice with its weights renormalised.  Computing
    both from one pass keeps them exactly paired and costs no extra tokens.

    Returns an empty mapping for the inverse scheme, where the layer is inert.
    """

    if scheme not in schemes:
        return {}
    zero_rho = np.flatnonzero(grid.component_rho == 0.0)
    if zero_rho.size == 0:
        raise ValueError("the rho prior must contain a zero node for the clean rule")
    # Dropping the rho factor from the weights of the rho=0 slice leaves
    # log alpha_weight + log delta_weight, which sums to one over that slice.
    rho_zero_index = int(np.flatnonzero(np.asarray(grid.rhos) == 0.0)[0])
    log_rho_zero = math.log(float(grid.rho_weights[rho_zero_index]))
    clean_weights = grid.log_component_weights[zero_rho] - log_rho_zero

    robust_output = np.empty(
        (pivots.shape[0], len(config.horizons)), dtype=float
    )
    clean_output = np.empty_like(robust_output)
    selected = {horizon - 1: index for index, horizon in enumerate(config.horizons)}
    full_weights = grid.log_component_weights[None, :]

    for start in range(0, pivots.shape[0], config.batch_size):
        stop = min(start + config.batch_size, pivots.shape[0])
        batch = pivots[start:stop]
        accumulator = np.zeros((batch.shape[0], grid.n_components), dtype=float)
        for time_index in range(config.max_horizon):
            accumulator += grid.component_log_ratio(batch[:, time_index])
            column = selected.get(time_index)
            if column is None:
                continue
            robust_output[start:stop, column] = _logsumexp_rows(
                accumulator + full_weights
            )
            clean_output[start:stop, column] = _logsumexp_rows(
                accumulator[:, zero_rho] + clean_weights[None, :]
            )
    return {
        clean_method: clean_output,
        robust_method: robust_output,
    }


def uniontail_shared_selected_paths(
    pivots: np.ndarray,
    scheme: str,
    config: ContaminationConfig,
    grid: dirichlet.UnionTailBayesGrid,
) -> dict[str, np.ndarray]:
    """Selected-horizon shared Bayes paths for the tail-width layer.

    Same accumulation as :func:`dirichlet_shared_selected_paths`, over the
    joint ``(J, Delta, rho)`` grid instead of ``(alpha, Delta, rho)``.
    """

    return dirichlet_shared_selected_paths(
        pivots,
        scheme,
        config,
        grid,
        clean_method=UNIONTAIL_CLEAN_METHOD,
        robust_method=UNIONTAIL_ROBUST_METHOD,
        schemes=UNIONTAIL_SCHEMES,
    )


def all_selected_paths(
    pivots: np.ndarray,
    scheme: str,
    config: ContaminationConfig,
    lookup: paper.GumbelBayesLookup,
    dirichlet_grid: dirichlet.DirichletBayesGrid | None = None,
    uniontail_grid: dirichlet.UnionTailBayesGrid | None = None,
) -> dict[str, np.ndarray]:
    paths = token_and_paper_selected_paths(pivots, scheme, config, lookup)
    paths.update(shared_bayes_selected_paths(pivots, scheme, config))
    if scheme in TRGOF_SCHEMES:
        paths.update(trgof_selected_paths(pivots, scheme, config))
    if dirichlet_grid is not None:
        paths.update(
            dirichlet_shared_selected_paths(pivots, scheme, config, dirichlet_grid)
        )
    if uniontail_grid is not None:
        paths.update(
            uniontail_shared_selected_paths(pivots, scheme, config, uniontail_grid)
        )
    return paths


def _method_group(method: str) -> str:
    if method in TRGOF_METHODS:
        return "trgof"
    if is_point_mass_method(method):
        return "bayes_shared_point_mass_diagnostic"
    if method in DIRICHLET_METHODS:
        return "bayes_shared_dirichlet"
    if method in UNIONTAIL_METHODS:
        return "bayes_shared_uniontail"
    if method.startswith("bayes_shared"):
        return "bayes_shared"
    if method.startswith("bayes_tokenwise"):
        return "bayes_tokenwise_sensitivity"
    return "paper_score"


def _simulate(
    scheme: str,
    kind: str,
    rng: np.random.Generator,
    rows: int,
    config: ContaminationConfig,
) -> np.ndarray:
    paper_config = config.paper_config()
    if scheme == "gumbel" and kind == "null":
        return paper.simulate_gumbel_null(rng, rows, config.max_horizon, paper_config)
    if scheme == "gumbel" and kind == "alternative":
        return paper.simulate_gumbel_alternative_shared_delta(
            rng, rows, config.max_horizon, paper_config
        )
    if scheme == "inverse" and kind == "null":
        return paper.simulate_inverse_null(rng, rows, config.max_horizon, paper_config)
    if scheme == "inverse" and kind == "alternative":
        return paper.simulate_inverse_alternative_shared_delta(
            rng, rows, config.max_horizon, paper_config
        )
    raise ValueError("unknown scheme or simulation kind")


def run_contamination_benchmark(
    config: ContaminationConfig,
    indicator_sink: dict[str, np.ndarray] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Run the contamination sweep.

    If ``indicator_sink`` is given it is filled in place with per-document
    boolean rejection indicators on each contaminated alternative sample, keyed
    by :func:`indicator_key`.  These indicators realise the calibrated rule
    ``score > c`` or ``score == c and U_i < gamma``.  Every method in a
    scheme/rho/horizon cell shares the same addressable auxiliary ``U_i``; its
    stream is independent of the pivot streams.  Aggregate rows retain the
    Rao--Blackwellized expected rate ``greater + gamma * equal`` and therefore
    do not depend on whether the sink is supplied.
    """

    config.validate()
    started = time.perf_counter()
    lookup = paper.GumbelBayesLookup(config.paper_config())
    dirichlet_grid = config.dirichlet_grid()
    # Both tail layers are Gumbel-only and are evaluated on exactly the same
    # pivots as every other rule.  All selected-path collectors in this file are
    # pure functions of an already-drawn pivot array, and the auxiliary
    # boundary uniforms are addressed by scheme/rho_true/horizon with the method
    # deliberately absent, so adding a rule consumes no randomness from any
    # generator an existing rule draws from.
    uniontail_grid = config.uniontail_grid()
    scheme_seeds = np.random.SeedSequence(config.seed).spawn(2)
    rows: list[dict[str, object]] = []
    calibration_summary: dict[str, object] = {}
    tie_records: list[dict[str, object]] = []

    for scheme, scheme_seed in zip(("gumbel", "inverse"), scheme_seeds):
        rng_calibration, rng_null, rng_clean, rng_replacement, rng_mask = [
            np.random.default_rng(seed) for seed in scheme_seed.spawn(5)
        ]
        calibration_pivots = _simulate(
            scheme, "null", rng_calibration, config.n_calibration, config
        )
        calibration_paths = all_selected_paths(
            calibration_pivots, scheme, config, lookup, dirichlet_grid, uniontail_grid
        )
        cutoffs, gammas = paper.calibrate_randomized_boundary(
            calibration_paths, config.alpha
        )
        del calibration_pivots

        evaluation_null = _simulate(
            scheme, "null", rng_null, config.n_evaluation_null, config
        )
        null_paths = all_selected_paths(
            evaluation_null, scheme, config, lookup, dirichlet_grid, uniontail_grid
        )
        del evaluation_null
        type_i = {
            method: paper.expected_rejection_rate(values, cutoffs[method], gammas[method])
            for method, values in null_paths.items()
        }
        # The min/max range is held over the Bayes rules and reference scores it
        # has always covered, so that adding a competitor cannot silently widen
        # an already-reported interval.  Tr-GoF's own range is reported beside
        # it, at full precision, so nothing is hidden by the scoping.
        legacy_type_i = {
            method: values
            for method, values in type_i.items()
            if method not in TRGOF_METHODS
        }
        scheme_summary: dict[str, object] = {
            "evaluation_type_i_min": float(
                min(values.min() for values in legacy_type_i.values())
            ),
            "evaluation_type_i_max": float(
                max(values.max() for values in legacy_type_i.values())
            ),
            "evaluation_type_i_scope": (
                "Bayes rules, point-mass diagnostics and reference scores; the "
                "Tr-GoF competitor is summarised separately in the same block."
            ),
        }
        for method in TRGOF_METHODS:
            if method in type_i:
                scheme_summary[f"{method}_evaluation_type_i_min"] = float(
                    type_i[method].min()
                )
                scheme_summary[f"{method}_evaluation_type_i_max"] = float(
                    type_i[method].max()
                )
        calibration_summary[scheme] = scheme_summary
        del null_paths

        clean = _simulate(
            scheme, "alternative", rng_clean, config.n_evaluation_alternative, config
        )
        replacements = _simulate(
            scheme, "null", rng_replacement, config.n_evaluation_alternative, config
        )
        mask_uniforms = rng_mask.uniform(size=clean.shape)

        for rho_true in config.rho_true_grid:
            pivots = contaminated_pivots(clean, replacements, mask_uniforms, rho_true)
            alternative_paths = all_selected_paths(
                pivots, scheme, config, lookup, dirichlet_grid, uniontail_grid
            )
            boundary_uniforms = {
                horizon: boundary_randomization_uniforms(
                    seed=config.seed,
                    scheme=scheme,
                    rho_true=rho_true,
                    horizon=horizon,
                    n_documents=config.n_evaluation_alternative,
                )
                for horizon in config.horizons
            }
            for method, values in alternative_paths.items():
                rejection = paper.expected_rejection_rate(
                    values, cutoffs[method], gammas[method]
                )
                for horizon_index, horizon in enumerate(config.horizons):
                    type_i_value = float(type_i[method][horizon_index])
                    type_ii_value = float(1.0 - rejection[horizon_index])
                    column = values[:, horizon_index]
                    threshold = float(cutoffs[method][horizon_index])
                    gamma = float(gammas[method][horizon_index])
                    strict = column > threshold
                    at_boundary = column == threshold
                    rejected = paper.randomized_boundary_rejections(
                        column,
                        threshold,
                        gamma,
                        boundary_uniforms[horizon],
                    )
                    boundary_rejected = at_boundary & rejected
                    if indicator_sink is not None:
                        indicator_sink[
                            indicator_key(scheme, rho_true, horizon, method)
                        ] = rejected
                    tie_records.append(
                        {
                            "scheme": scheme,
                            "rho_true": float(rho_true),
                            "horizon": int(horizon),
                            "method": method,
                            "threshold": threshold,
                            "n_documents": int(column.size),
                            "n_at_atom": int(np.count_nonzero(at_boundary)),
                            "boundary_randomization_probability": gamma,
                            "n_rejected_strict": int(np.count_nonzero(strict)),
                            "n_rejected_at_atom": int(
                                np.count_nonzero(boundary_rejected)
                            ),
                            "n_rejected_randomized": int(
                                np.count_nonzero(rejected)
                            ),
                            "type_ii_error_randomized_realization": float(
                                1.0 - rejected.mean()
                            ),
                            "auxiliary_seed_entropy": list(
                                _boundary_randomization_seed_entropy(
                                    config.seed, scheme, rho_true, horizon
                                )
                            ),
                        }
                    )
                    if method in TRGOF_METHODS:
                        assumed_rho = (
                            "none (Tr-GoF competitor, s="
                            f"{trgof.DEFAULT_S:g}; robust to null-like edits by "
                            "construction rather than by a rho model)"
                        )
                        rho_structure = "not_applicable"
                        delta_structure = "not_applicable"
                    elif is_point_mass_method(method):
                        assumed_rho = (
                            "fixed rho="
                            f"{method[len(POINT_MASS_PREFIX):]} (point mass diagnostic)"
                        )
                        rho_structure = "fixed_point_mass"
                        delta_structure = "document_shared"
                    elif method == DIRICHLET_ROBUST_METHOD:
                        assumed_rho = "shared spike-and-grid prior"
                        rho_structure = "document_shared"
                        delta_structure = "document_shared_plus_dirichlet_tail_prior"
                    elif method == DIRICHLET_CLEAN_METHOD:
                        assumed_rho = "fixed rho=0"
                        rho_structure = "fixed_zero"
                        delta_structure = "document_shared_plus_dirichlet_tail_prior"
                    elif method == UNIONTAIL_ROBUST_METHOD:
                        assumed_rho = "shared spike-and-grid prior"
                        rho_structure = "document_shared"
                        delta_structure = "document_shared_plus_tail_width_prior"
                    elif method == UNIONTAIL_CLEAN_METHOD:
                        assumed_rho = "fixed rho=0"
                        rho_structure = "fixed_zero"
                        delta_structure = "document_shared_plus_tail_width_prior"
                    elif method == "bayes_shared_robust":
                        assumed_rho = "shared spike-and-grid prior"
                        rho_structure = "document_shared"
                        delta_structure = "document_shared"
                    elif method == "bayes_shared_clean":
                        assumed_rho = "fixed rho=0"
                        rho_structure = "fixed_zero"
                        delta_structure = "document_shared"
                    elif method == "bayes_tokenwise_robust":
                        assumed_rho = "tokenwise prior; equivalent fixed mean rho=0.16875"
                        rho_structure = "tokenwise_collapses_to_prior_mean"
                        delta_structure = "tokenwise"
                    elif method == "bayes_tokenwise_clean":
                        assumed_rho = "fixed rho=0"
                        rho_structure = "fixed_zero"
                        delta_structure = "tokenwise"
                    else:
                        assumed_rho = "none (reference score)"
                        rho_structure = "not_applicable"
                        delta_structure = "not_applicable"
                    rows.append(
                        {
                            "scheme": scheme,
                            "scenario": "shared_delta_iid_null_replacement",
                            "rho_true": float(rho_true),
                            "horizon": int(horizon),
                            "method": method,
                            "method_group": _method_group(method),
                            "decision_rule": "fixed_horizon_mc_calibrated",
                            "alpha": config.alpha,
                            "type_i_error": type_i_value,
                            "type_i_mc_se": math.sqrt(
                                type_i_value * (1.0 - type_i_value)
                                / config.n_evaluation_null
                            ),
                            "type_ii_error": type_ii_value,
                            "type_ii_mc_se": math.sqrt(
                                type_ii_value * (1.0 - type_ii_value)
                                / config.n_evaluation_alternative
                            ),
                            "power": float(rejection[horizon_index]),
                            "fixed_horizon_cutoff": float(
                                cutoffs[method][horizon_index]
                            ),
                            "boundary_randomization_probability": float(
                                gammas[method][horizon_index]
                            ),
                            "assumed_rho_model": assumed_rho,
                            "delta_structure": delta_structure,
                            "rho_structure": rho_structure,
                            "n_calibration": config.n_calibration,
                            "n_evaluation_null": config.n_evaluation_null,
                            "n_evaluation_alternative": config.n_evaluation_alternative,
                        }
                    )
            del alternative_paths
        del clean, replacements, mask_uniforms

    def _rho_one_diagnostic(candidates: list[dict[str, object]]) -> dict[str, object]:
        worst = max(
            candidates,
            key=lambda row: abs(float(row["power"]) - float(row["type_i_error"])),
        )
        gap = abs(float(worst["power"]) - float(worst["type_i_error"]))
        se = math.sqrt(
            float(worst["power"])
            * (1.0 - float(worst["power"]))
            / config.n_evaluation_alternative
            + float(worst["type_i_error"])
            * (1.0 - float(worst["type_i_error"]))
            / config.n_evaluation_null
        )
        return {"row": worst, "gap": gap, "se": se}

    rho_one = [row for row in rows if row["rho_true"] == 1.0]
    # The headline diagnostic is taken over the Bayes-versus-reference-score
    # methods this file has always carried, so that adding a detector cannot
    # move an already-reported number.  The point-mass diagnostics and the
    # Tr-GoF competitor are excluded here and get their own rho=1 gaps below;
    # excluding them costs nothing, because the diagnostic is a per-method
    # validity check and every method's own gap is recoverable from the rows.
    headline = _rho_one_diagnostic(
        [
            row
            for row in rho_one
            if not is_point_mass_method(str(row["method"]))
            and str(row["method"]) not in TRGOF_METHODS
        ]
    )
    diagnostic_row = headline["row"]
    diagnostic_gap = float(headline["gap"])
    diagnostic_se = float(headline["se"])
    point_mass_rho_one = [
        row for row in rho_one if is_point_mass_method(str(row["method"]))
    ]
    trgof_rho_one = [row for row in rho_one if str(row["method"]) in TRGOF_METHODS]
    metadata: dict[str, object] = {
        "config": _json_safe_config(config),
        "runtime_seconds": time.perf_counter() - started,
        "data_model": (
            "Within each contamination stratum, rho_true is fixed and shared across "
            "positions; C_it is iid Bernoulli(rho_true). One Delta is drawn per document, "
            "and C_it=1 replaces the clean watermarked pivot by an exact-null pivot."
        ),
        "pairing": (
            "Within each pivot scheme, all methods and rho levels share clean paths, "
            "exact-null replacement paths, and nested uniforms C_t(rho)=1{U_t<rho}."
        ),
        "robust_rho_prior": {
            "support": list(config.rho_prior_grid),
            "weights": list(config.rho_prior_weights),
            "mean": float(np.dot(config.rho_prior_grid, config.rho_prior_weights)),
            "frozen_before_simulation": True,
        },
        "calibration": (
            "Each pivot scheme has one exact-null calibration sample, common to all methods "
            "within that scheme and independent of evaluation; each method-specific cutoff "
            "is reused across rho_true."
        ),
        "calibration_summary": calibration_summary,
        "rho_one_diagnostic_max_abs_power_minus_type_i": diagnostic_gap,
        "rho_one_diagnostic_mc_se_at_max_gap": diagnostic_se,
        "rho_one_diagnostic_standardized_gap": diagnostic_gap / diagnostic_se,
        "rho_one_diagnostic_cell": {
            "scheme": diagnostic_row["scheme"],
            "method": diagnostic_row["method"],
            "horizon": diagnostic_row["horizon"],
        },
        "rho_one_diagnostic_scope": (
            "Maximized over the Bayes rules and the reference scores only; the "
            "pi_rho=delta_{rho_0} point-mass detectors and the Tr-GoF competitor "
            "are reported separately below, so that adding a detector cannot move "
            "this number.  Every method's own rho=1 gap is recoverable from the "
            "power and type_i_error columns of the result rows."
        ),
        "rho_point_mass_diagnostic": {
            "support": [float(value) for value in config.rho_point_masses],
            "methods": list(config.point_mass_methods()),
            "purpose": (
                "Replace the spike-and-grid prior by pi_rho = delta_{rho_0} in the "
                "spike-family (equal-tail) shared detector, holding the shared-Delta "
                "hierarchy, quadrature, paired paths and 5% null calibration fixed, to "
                "separate 'averaging over pi_rho' from 'assuming some rho>0'.  These "
                "point masses do not carry the Dirichlet tail layer; the layer's own "
                "rho treatments are the bayes_shared_dirichlet_* methods."
            ),
            "rho_one_max_abs_power_minus_type_i": (
                float(_rho_one_diagnostic(point_mass_rho_one)["gap"])
                if point_mass_rho_one
                else None
            ),
        },
        "bayes_uniontail": {
            **union_tail_metadata(uniontail_grid, methods=list(UNIONTAIL_METHODS), tokenwise=False),
            'rho_treatment': {
                "clean": "point mass at rho=0",
                "robust": (
                    "the identical frozen spike-and-grid prior the "
                    "bayes_shared_dirichlet_* and bayes_shared_* rules use"
                ),
                "support": list(config.rho_prior_grid),
                "weights": list(config.rho_prior_weights),
            },
            'containment_check': {
                "shape_equal_tail_atom_vs_closed_form_spike_max_abs_log_density_difference": (
                    uniontail_grid.spike_agreement(
                        np.random.default_rng(config.seed).uniform(size=20_000)
                    )
                ),
            },
        },
        "trgof": {
            "methods": list(TRGOF_METHODS),
            "schemes": list(TRGOF_SCHEMES),
            "source": (
                "Tr-GoF, the truncated goodness-of-fit test of Li et al. (2024) "
                "(github.com/lx10077/TrGoF).  Reimplemented in code/trgof.py and "
                "checked term by term against the authors' released compute_score, "
                "so the statistic evaluated here is theirs, not a paraphrase."
            ),
            "why_included": (
                "Tr-GoF is a robust test built for exactly this contamination "
                "question -- detection when a fraction of the pivots has been "
                "replaced by null-like draws -- so it is the sharpest available "
                "non-Bayesian comparator for this benchmark, and it is evaluated "
                "at every rho_true and every tabulated horizon."
            ),
            "statistic": (
                "S_n(s) = n * max_t K_s(t/n, p_(t)) over the truncated one-sided "
                "index set p_(t) >= 1/n and t/n >= p_(t), with K_s the "
                "Cressie-Read divergence."
            ),
            "s_reported": float(trgof.DEFAULT_S),
            "s_reported_name": "Higher Criticism (s=2)",
            "s_sensitivity_grid": [float(value) for value in trgof.DEFAULT_S_VALUES],
            "s_sensitivity_note": (
                "The authors' simulation code sweeps s over "
                f"{[float(v) for v in trgof.DEFAULT_S_VALUES]}.  Only s="
                f"{trgof.DEFAULT_S:g}, the index their figures lead with, is run as a "
                "rule here; the remaining indices are recorded as this sensitivity "
                "grid rather than added as separate rows, because the tables are "
                "already large.  Rerunning with a different s changes one argument "
                "of trgof.statistic and nothing else."
            ),
            "not_a_token_sum": (
                "The statistic sorts the p-values of the whole prefix, so it is not "
                "additive over tokens and cannot be accumulated by cumsum.  It is "
                "recomputed from scratch at every reported horizon, as the shared "
                "Bayes rules are."
            ),
            "p_value_map": {
                "gumbel": {
                    "definition": "p = 1 - Y",
                    "provenance": "the authors' own definition, as released",
                },
                "inverse": {
                    "definition": "p = F_0(d), the exact finite-vocabulary null CDF",
                    "provenance": (
                        "OUR EXTENSION, not the authors'.  The released "
                        "implementation applies Tr-GoF to Gumbel-max pivots only "
                        "and defines no p-value for the inverse-transform pivot.  "
                        "Small d is the evidence direction there, so the natural "
                        "one-sided p-value is the null CDF rather than its "
                        "complement; using the exact finite-V null keeps these "
                        "p-values uniform under the null at the vocabulary size "
                        "actually simulated.  Any inverse-scheme Tr-GoF number in "
                        "this study is therefore ours to defend, not a reproduction "
                        "of a published result."
                    ),
                    "vocabulary_size_for_exact_null": int(config.vocabulary_size),
                },
            },
            "calibration": (
                "Routed through the shared calibration path with no special case: "
                "the same per-scheme exact-null calibration sample as every other "
                "method, the same nominal level alpha, and the same randomized "
                "boundary rule score > c or (score == c and U_i < gamma) with the "
                "same addressable auxiliary uniforms."
            ),
            "atom_at_zero": (
                "The statistic is exactly zero whenever a prefix's truncated index "
                "set is empty, so its null distribution has an atom at zero and the "
                "randomized boundary treatment is load-bearing rather than cosmetic; "
                "the per-cell atom and boundary-rejection counts are in "
                "per_document_indicators.tie_and_randomized_rejection_counts."
            ),
            "rho_one_max_abs_power_minus_type_i": (
                float(_rho_one_diagnostic(trgof_rho_one)["gap"])
                if trgof_rho_one
                else None
            ),
        },
        "per_document_indicators": {
            "indicator_rule_version": BOUNDARY_RANDOMIZATION_RULE_VERSION,
            "horizons": list(config.horizons),
            "sample": "contaminated alternative evaluation sample",
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
                    "scheme_code",
                    "rho_true_binary64_low32",
                    "rho_true_binary64_high32",
                    "horizon",
                ],
                "scheme_codes": dict(_BOUNDARY_SCHEME_CODES),
                "rho_true_encoding": (
                    "The exact IEEE-754 binary64 bit pattern, split into low and "
                    "high 32-bit SeedSequence words."
                ),
                "coupling": (
                    "The same auxiliary U_i is reused for document i by every "
                    "method in a scheme/rho_true/horizon cell. Distinct cells have "
                    "separate addressable SeedSequence streams independent of all "
                    "pivot, replacement, and contamination-mask RNGs."
                ),
                "aggregate_rate_note": (
                    "CSV power is the Rao-Blackwellized conditional expectation "
                    "greater + gamma*equal. Persisted indicators are one reproducible "
                    "realization of that randomized-boundary procedure."
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
        "scope_warning": (
            "This models iid null-like dilution only, not burst edits, paraphrase "
            "dependence, insertion/deletion alignment, or key desynchronization."
        ),
        "tokenwise_warning": (
            "A tokenwise rho prior collapses to its mean and cannot learn a shared "
            "document contamination rate; tokenwise rows are misspecification sensitivity."
        ),
        "inverse_model_warning": (
            "Inverse data use the exact finite-V equal-tail generator. Inverse Bayes uses "
            "the large-V triangular alternative of Li et al. (2025) divided by the "
            "exact finite-V null."
        ),
    }
    return rows, metadata


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(
    rows: list[dict[str, object]], metadata: dict[str, object], path: Path
) -> None:
    payload = dict(metadata)
    payload["selected_results"] = rows
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _find(
    rows: list[dict[str, object]], scheme: str, method: str, horizon: int, rho: float
) -> dict[str, object]:
    matches = [
        row
        for row in rows
        if row["scheme"] == scheme
        and row["method"] == method
        and row["horizon"] == horizon
        and math.isclose(float(row["rho_true"]), rho)
    ]
    if len(matches) != 1:
        raise ValueError("result lookup is not unique")
    return matches[0]


def _sweep_rhos(config: ContaminationConfig) -> list[float]:
    """Contamination rates excluding the rho=1 null-equivalence diagnostic."""

    return [float(rho) for rho in config.rho_true_grid if float(rho) < 1.0]


def build_rho_diagnostic(
    rows: list[dict[str, object]],
    metadata: dict[str, object],
    config: ContaminationConfig,
    indicators: dict[str, np.ndarray] | None = None,
    mcnemar_rhos: tuple[float, ...] = (0.25, 0.4, 0.6),
) -> dict[str, object]:
    """Compare the mixture prior against pi_rho = delta_{rho_0} point masses.

    The comparison is reported two ways.  The *oracle* reading lets rho_0 be
    re-chosen at every rho_true, which no analyst can do; the *single fixed*
    reading picks one rho_0 for the whole sweep, which an analyst can.
    """

    point_methods = list(config.point_mass_methods())
    sweep = _sweep_rhos(config)
    all_rhos = [float(rho) for rho in config.rho_true_grid]

    def type_ii(scheme: str, method: str, horizon: int, rho: float) -> float:
        return float(_find(rows, scheme, method, horizon, rho)["type_ii_error"])

    tables: list[dict[str, object]] = []
    for scheme in ("gumbel", "inverse"):
        paper_method = PRESPECIFIED_PAPER_SCORE[scheme]
        # The Dirichlet rules exist for the Gumbel pivot only; asking _find for
        # them under the inverse scheme would raise after the whole run.
        layer_methods = (
            list(DIRICHLET_METHODS) if scheme in DIRICHLET_SCHEMES else []
        ) + (list(UNIONTAIL_METHODS) if scheme in UNIONTAIL_SCHEMES else [])
        listed = (
            ["bayes_shared_robust"]
            + point_methods
            + ["bayes_shared_clean", paper_method]
            + ([TRGOF_METHOD] if scheme in TRGOF_SCHEMES else [])
            + layer_methods
        )
        for horizon in config.horizons:
            table_rows = []
            for method in listed:
                if method == "bayes_shared_robust":
                    role = "mixture prior over rho (manuscript robust rule)"
                elif method == "bayes_shared_clean":
                    role = "point mass at rho=0 (manuscript clean rule)"
                elif method == DIRICHLET_ROBUST_METHOD:
                    role = (
                        "mixture prior over rho and over the Dirichlet tail "
                        "concentration alpha"
                    )
                elif method == DIRICHLET_CLEAN_METHOD:
                    role = (
                        "point mass at rho=0, mixture prior over the Dirichlet "
                        "tail concentration alpha"
                    )
                elif method == UNIONTAIL_ROBUST_METHOD:
                    role = (
                        "mixture prior over rho and over the number of live "
                        "tail coordinates J"
                    )
                elif method == UNIONTAIL_CLEAN_METHOD:
                    role = (
                        "point mass at rho=0, mixture prior over the number of "
                        "live tail coordinates J"
                    )
                elif method == TRGOF_METHOD:
                    role = (
                        "Tr-GoF robust competitor, Higher Criticism index s="
                        f"{trgof.DEFAULT_S:g}"
                    )
                elif method == paper_method:
                    role = "prespecified reference score"
                else:
                    role = "diagnostic point mass at rho_0"
                table_rows.append(
                    {
                        "method": method,
                        "role": role,
                        "type_ii_error": [
                            type_ii(scheme, method, horizon, rho) for rho in all_rhos
                        ],
                        "type_i_error": float(
                            _find(rows, scheme, method, horizon, all_rhos[0])["type_i_error"]
                        ),
                    }
                )
            tables.append(
                {
                    "scheme": scheme,
                    "horizon": int(horizon),
                    "rho_true_grid": all_rhos,
                    "rows": table_rows,
                }
            )

    def _readings(family: list[str]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        oracle_records: list[dict[str, object]] = []
        single_fixed_records: list[dict[str, object]] = []
        if not family:
            return oracle_records, single_fixed_records
        for scheme in ("gumbel", "inverse"):
            for horizon in config.horizons:
                robust = {
                    rho: type_ii(scheme, "bayes_shared_robust", horizon, rho)
                    for rho in all_rhos
                }
                # Oracle reading: rho_0 re-chosen with hindsight at each rho_true.
                for rho in all_rhos:
                    values = {m: type_ii(scheme, m, horizon, rho) for m in family}
                    best = min(values, key=lambda m: (values[m], m))
                    oracle_records.append(
                        {
                            "scheme": scheme,
                            "horizon": int(horizon),
                            "rho_true": rho,
                            "best_point_mass": best,
                            "best_point_mass_type_ii": values[best],
                            "robust_type_ii": robust[rho],
                            "robust_minus_oracle_point_mass": robust[rho] - values[best],
                        }
                    )
                # Single-fixed reading: one rho_0 held across the whole sweep.
                if not sweep:
                    continue
                summaries = {
                    m: {
                        "mean_type_ii_over_sweep": sum(
                            type_ii(scheme, m, horizon, rho) for rho in sweep
                        )
                        / len(sweep),
                        "max_type_ii_over_sweep": max(
                            type_ii(scheme, m, horizon, rho) for rho in sweep
                        ),
                    }
                    for m in family
                }
                best_mean = min(
                    summaries, key=lambda m: (summaries[m]["mean_type_ii_over_sweep"], m)
                )
                best_max = min(
                    summaries, key=lambda m: (summaries[m]["max_type_ii_over_sweep"], m)
                )
                robust_sweep = [robust[rho] for rho in sweep]
                single_fixed_records.append(
                    {
                        "scheme": scheme,
                        "horizon": int(horizon),
                        "sweep_rho_true": sweep,
                        "per_point_mass": summaries,
                        "best_by_mean": best_mean,
                        "best_by_max": best_max,
                        "robust_mean_type_ii_over_sweep": sum(robust_sweep)
                        / len(robust_sweep),
                        "robust_max_type_ii_over_sweep": max(robust_sweep),
                        "robust_minus_best_single_fixed_by_rho": {
                            f"{rho:g}": robust[rho]
                            - type_ii(scheme, best_mean, horizon, rho)
                            for rho in all_rhos
                        },
                    }
                )
        return oracle_records, single_fixed_records

    families: dict[str, list[str]] = (
        {"all_point_masses": point_methods} if point_methods else {}
    )
    requested = [
        point_mass_method(value)
        for value in REQUESTED_RHO_POINT_MASSES
        if point_mass_method(value) in point_methods
    ]
    if requested and requested != point_methods:
        families["requested_grid_only"] = requested
    oracle_records, single_fixed_records = _readings(point_methods)

    payload: dict[str, object] = {
        "question": (
            "Is the robust shared Bayes rule's advantage due to averaging over the "
            "frozen prior pi_rho, or would a single fixed rho_0 point mass do as well?"
        ),
        "design": (
            "Every detector shares the same Delta ~ Unif(delta_low, delta_high) with the "
            "same Gauss-Legendre nodes, the same shared-Delta hierarchy, the same paired "
            "clean/null-replacement/mask paths, and its own 5% null calibration reused "
            "across the rho_true sweep. Only pi_rho differs."
        ),
        "robust_rho_prior": metadata["robust_rho_prior"],
        "point_masses": [float(v) for v in config.rho_point_masses],
        "point_mass_methods": point_methods,
        "point_mass_diagnostic_status": (
            "evaluated" if point_methods else "disabled_empty_point_mass_menu"
        ),
        "prespecified_paper_score": dict(PRESPECIFIED_PAPER_SCORE),
        "monte_carlo": {
            "n_evaluation_alternative": config.n_evaluation_alternative,
            "n_evaluation_null": config.n_evaluation_null,
            "paired": (
                "All methods and rho_true levels within a scheme reuse the same documents, "
                "so differences are paired and their standard error is smaller than the "
                "independent-sample sqrt(p(1-p)/n)."
            ),
        },
        "type_ii_tables": tables,
        "oracle_reading": {
            "definition": (
                "rho_0 chosen with hindsight separately at each rho_true; an upper bound "
                "on what any fixed rho_0 could achieve."
            ),
            "records": oracle_records,
        },
        "single_fixed_reading": {
            "definition": (
                "One rho_0 held across the whole rho_true sweep, selected by lowest mean "
                "Type II over rho_true<1; this is what an analyst could actually do."
            ),
            "records": single_fixed_records,
        },
        "readings_by_point_mass_family": {
            "note": (
                "The requested grid stops at rho_0=0.4, below the top of both the prior's "
                "support and the contamination sweep, so it understates a fixed rho_0 at "
                "large rho_true. Both families are reported."
            ),
            "families": {
                name: {
                    "point_mass_methods": members,
                    "oracle_records": oracle,
                    "single_fixed_records": single_fixed,
                }
                for name, members in families.items()
                for oracle, single_fixed in [_readings(members)]
            },
        },
    }
    if indicators:
        payload["mcnemar_robust_vs_point_mass"] = _mcnemar_records(
            rows, config, indicators, mcnemar_rhos
        )
        payload["mcnemar_note"] = (
            "Indicators realize score > c or (score == c and U_i < gamma), with the "
            "same reproducible auxiliary U_i for every method in a "
            "scheme/rho_true/horizon/document cell. The aggregate Type-II rows remain "
            "the Rao-Blackwellized expectation greater + gamma*equal, whereas McNemar "
            "and Wilson summaries use this one randomized realization. b counts "
            "documents missed by the robust rule but caught by the point mass, c the "
            "reverse. p is the exact two-sided McNemar binomial p-value. When b+c=0, "
            "the directional effect is not estimable; p=1 is retained as the "
            "conservative exact-test bookkeeping value so the cell can be retained "
            "if these prespecified comparisons are multiplicity-adjusted."
        )
    return payload


def _mcnemar_records(
    rows: list[dict[str, object]],
    config: ContaminationConfig,
    indicators: dict[str, np.ndarray],
    mcnemar_rhos: tuple[float, ...],
) -> list[dict[str, object]]:
    import paired_comparisons as paired

    horizon = max(config.horizons)
    point_methods = list(config.point_mass_methods())
    records: list[dict[str, object]] = []
    if not point_methods:
        return records
    for scheme in ("gumbel", "inverse"):
        for rho in mcnemar_rhos:
            robust_key = indicator_key(scheme, rho, horizon, "bayes_shared_robust")
            if robust_key not in indicators:
                continue
            robust_miss = np.logical_not(indicators[robust_key])
            robust_expected_type_ii = float(
                _find(rows, scheme, "bayes_shared_robust", horizon, rho)[
                    "type_ii_error"
                ]
            )
            values = {
                method: float(
                    _find(rows, scheme, method, horizon, rho)["type_ii_error"]
                )
                for method in point_methods
            }
            best = min(values, key=lambda m: (values[m], m))
            requested = [
                point_mass_method(value)
                for value in REQUESTED_RHO_POINT_MASSES
                if point_mass_method(value) in values
            ]
            best_requested = (
                min(requested, key=lambda m: (values[m], m)) if requested else None
            )
            n_documents = int(robust_miss.size)
            robust_low, robust_high = paired.wilson_interval(
                int(np.count_nonzero(robust_miss)), n_documents
            )
            for method in point_methods:
                key = indicator_key(scheme, rho, horizon, method)
                if key not in indicators:
                    continue
                other_miss = np.logical_not(indicators[key])
                b = int(np.count_nonzero(robust_miss & ~other_miss))
                c = int(np.count_nonzero(other_miss & ~robust_miss))
                low, high = paired.wilson_interval(
                    int(np.count_nonzero(other_miss)), n_documents
                )
                records.append(
                    {
                        "scheme": scheme,
                        "horizon": int(horizon),
                        "rho_true": float(rho),
                        "reference": "bayes_shared_robust",
                        "comparator": method,
                        "comparator_is_best_point_mass_at_this_rho": method == best,
                        "comparator_is_best_requested_grid_point_mass_at_this_rho": (
                            method == best_requested
                        ),
                        "n_documents": n_documents,
                        "b_robust_only_miss": b,
                        "c_point_mass_only_miss": c,
                        "discordant_total": b + c,
                        "p_value": paired.exact_mcnemar_p_value(b, c),
                        "p_value_status": (
                            "no_discordant_pairs_p_equals_one"
                            if b + c == 0
                            else "defined"
                        ),
                        "robust_n_missed": int(np.count_nonzero(robust_miss)),
                        "robust_type_ii_randomized_realization": float(
                            np.mean(robust_miss)
                        ),
                        "robust_type_ii_rao_blackwellized": robust_expected_type_ii,
                        "robust_wilson95": [robust_low, robust_high],
                        "comparator_n_missed": int(np.count_nonzero(other_miss)),
                        "comparator_type_ii_randomized_realization": float(
                            np.mean(other_miss)
                        ),
                        "comparator_type_ii_rao_blackwellized": values[method],
                        "comparator_wilson95": [low, high],
                    }
                )
    return records


def write_rho_diagnostic(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_report(
    rows: list[dict[str, object]], metadata: dict[str, object], path: Path
) -> None:
    horizon = max(int(row["horizon"]) for row in rows)
    available_rhos = sorted({float(row["rho_true"]) for row in rows})
    headline_rho = 0.4 if 0.4 in available_rhos else max(
        rho for rho in available_rhos if rho < 1.0
    )
    lines = [
        "# Independent null-like contamination benchmark",
        "",
        "Within each contamination stratum, `rho_true` is fixed across positions. The "
        "study replaces each clean watermarked pivot by an independent exact-null pivot "
        "with probability `rho_true`; one Delta is drawn per document.",
        "",
        "The robust prior is frozen at `0.5 delta_0 + 0.125(delta_.1 + delta_.25 + "
        "delta_.4 + delta_.6)`. Each pivot scheme has one exact-null calibration sample, "
        "common to all methods within that scheme and independent of evaluation; the "
        "method-specific cutoffs are reused at every contamination rate.",
        "",
        f"## Headline results at n={horizon}, rho_true={headline_rho}",
        "",
        "| Scheme | Robust shared Bayes | Robust shared + Dirichlet tail | "
        "Clean shared Bayes | Prespecified reference score |",
        "|---|---:|---:|---:|---:|",
    ]
    references = {
        "gumbel": "h_gum_star_0.005",
        "inverse": "h_dif_star_0.01",
    }
    for scheme in ("gumbel", "inverse"):
        robust = _find(rows, scheme, "bayes_shared_robust", horizon, headline_rho)
        clean = _find(rows, scheme, "bayes_shared_clean", horizon, headline_rho)
        competitor = _find(rows, scheme, references[scheme], horizon, headline_rho)
        # The layer is Gumbel-only; _find raises rather than returning None, so
        # the inverse row gets an em-dash instead of a lookup.
        if scheme in DIRICHLET_SCHEMES:
            layer = _find(
                rows, scheme, DIRICHLET_ROBUST_METHOD, horizon, headline_rho
            )
            layer_cell = f"{float(layer['type_ii_error']):.4f}"
        else:
            layer_cell = "--"
        lines.append(
            f"| {scheme.title()} | {float(robust['type_ii_error']):.4f} | "
            f"{layer_cell} | "
            f"{float(clean['type_ii_error']):.4f} | "
            f"{float(competitor['type_ii_error']):.4f} |"
        )
    lines.extend(
        [
            "",
            "The prespecified reference scores are `h*_gum,.005` and `h*_dif,.01`; "
            "they were evaluated by Li et al. (2025), selected from the independent "
            "clean n=700 benchmark, and then frozen before "
            "the contamination sweep.",
        ]
    )
    lines.extend(
        [
            "",
            "All table entries are Type II error. At `rho_true=1`, the alternative equals "
            "the null. The maximum absolute difference between simulated power and the "
            "corresponding evaluation Type I rate is "
            f"{float(metadata['rho_one_diagnostic_max_abs_power_minus_type_i']):.4f} "
            f"({float(metadata['rho_one_diagnostic_standardized_gap']):.2f} Monte Carlo SE).",
            "",
            "Inverse data use the exact finite-V equal-tail generator, whereas inverse "
            "Bayes uses the large-V triangular alternative with the exact finite-V null. "
            "The result establishes robustness only to iid null-like replacement. It does "
            "not model burst edits, context-dependent paraphrase, insertion/deletion "
            "alignment, or key desynchronization.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_rows(
    rows: list[dict[str, object]], scheme: str, method: str, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    selected = sorted(
        (
            (float(row["rho_true"]), float(row["type_ii_error"]))
            for row in rows
            if row["scheme"] == scheme
            and row["method"] == method
            and int(row["horizon"]) == horizon
        ),
        key=lambda pair: pair[0],
    )
    return np.asarray([item[0] for item in selected]), np.asarray(
        [item[1] for item in selected]
    )


def _plot_style() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    labels = {
        "bayes_shared_robust": r"Bayes, shared $\Delta$; $\rho$ mixture",
        "bayes_shared_clean": r"Bayes, shared $\Delta$; $\rho=0$",
        "bayes_tokenwise_robust": r"Bayes, tokenwise $\Delta$; $\bar\rho=.16875$",
        DIRICHLET_ROBUST_METHOD: r"Bayes, shared $\Delta$ + tail shape; $\rho$ mixture",
        DIRICHLET_CLEAN_METHOD: r"Bayes, shared $\Delta$ + tail shape; $\rho=0$",
        UNIONTAIL_ROBUST_METHOD: r"Bayes, shared $\Delta$ + union tail; $\rho$ mixture",
        UNIONTAIL_CLEAN_METHOD: r"Bayes, shared $\Delta$ + union tail; $\rho=0$",
        "h_gum_star_0.005": r"$h^\star_{\mathrm{gum},.005}$",
        "h_dif_star_0.01": r"$h^\star_{\mathrm{dif},.01}$",
        TRGOF_METHOD: r"Tr-GoF, $s=2$ (Li et al. 2024)",
    }
    colors = {
        "bayes_shared_robust": "#542788",
        "bayes_shared_clean": "#542788",
        "bayes_tokenwise_robust": "#7a7a7a",
        DIRICHLET_ROBUST_METHOD: "#9B2226",
        DIRICHLET_CLEAN_METHOD: "#9B2226",
        UNIONTAIL_ROBUST_METHOD: "#0B6FA4",
        UNIONTAIL_CLEAN_METHOD: "#0B6FA4",
        "h_gum_star_0.005": "#009E73",
        "h_dif_star_0.01": "#009E73",
        TRGOF_METHOD: "#6b4c9a",
    }
    styles = {
        "bayes_shared_robust": "-",
        "bayes_shared_clean": "--",
        "bayes_tokenwise_robust": ":",
        # The layer curves coincide with their spike counterparts wherever the
        # tail prior is idle, which for Gumbel is most of the sweep.  Long
        # dashes keep the underlying curve visible instead of hiding it.
        DIRICHLET_ROBUST_METHOD: (0, (7, 2.5)),
        DIRICHLET_CLEAN_METHOD: (0, (2, 1.4)),
        # Same reasoning for the tail-width curves, which under an equal-tail
        # generator also sit almost on top of their spike counterparts.
        UNIONTAIL_ROBUST_METHOD: (0, (5, 1.6, 1, 1.6)),
        UNIONTAIL_CLEAN_METHOD: (0, (1, 1.6)),
        "h_gum_star_0.005": "-.",
        "h_dif_star_0.01": "-.",
        # Tr-GoF is the other competitor on the panel, so it gets a dash-dot
        # variant that reads as a competitor without colliding with the
        # reference score's line.
        TRGOF_METHOD: (0, (6, 1.6, 1, 1.6, 1, 1.6)),
    }
    return labels, colors, styles


def plot_error_estimates(axis, x, y, zero_bound, **style):
    """Keep positive estimates unchanged and display zero-count bounds separately."""
    values = np.asarray(y, dtype=float)
    positions = np.asarray(x)
    axis.plot(positions, np.where(values > 0, values, np.nan), **style)
    zero = values == 0
    if np.any(zero):
        axis.plot(
            positions[zero], np.full(int(zero.sum()), zero_bound),
            color=style.get("color"), linestyle="None", marker="v",
            markerfacecolor="none", markersize=style.get("markersize", 4),
            label="_nolegend_",
        )


def error_axis_floor(rows, zero_bound):
    positive = [float(row["type2_error"]) for row in rows if float(row["type2_error"]) > 0]
    return .8 * min([zero_bound] + positive)


def make_full_plot(
    rows: list[dict[str, object]], config: ContaminationConfig, output_stem: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, colors, styles = _plot_style()
    fig, axes = plt.subplots(2, len(config.horizons), figsize=(13.2, 7.6), sharex=True, sharey=True, squeeze=False)
    display_floor = 3.0 / config.n_evaluation_alternative
    for row_index, scheme in enumerate(("gumbel", "inverse")):
        paper_method = "h_gum_star_0.005" if scheme == "gumbel" else "h_dif_star_0.01"
        methods = (
            "bayes_shared_robust",
            "bayes_shared_clean",
            paper_method,
            "bayes_tokenwise_robust",
        ) + (TRGOF_METHODS if scheme in TRGOF_SCHEMES else ()) + (
            DIRICHLET_METHODS if scheme in DIRICHLET_SCHEMES else ()
        ) + (UNIONTAIL_METHODS if scheme in UNIONTAIL_SCHEMES else ())
        for column_index, horizon in enumerate(config.horizons):
            axis = axes[row_index, column_index]
            for method in methods:
                x, y = _plot_rows(rows, scheme, method, horizon)
                plot_error_estimates(
                    axis,
                    x,
                    y,
                    display_floor,
                    label=labels[method],
                    color=colors[method],
                    linestyle=styles[method],
                    marker="o",
                    linewidth=2.0 if method == "bayes_shared_robust" else 1.55,
                    markersize=4,
                )
            axis.set_yscale("log")
            axis.set_ylim(error_axis_floor(rows, display_floor), 1.15)
            axis.grid(alpha=0.2, which="both")
            axis.set_title(f"{scheme.title()}, n={horizon}")
            if row_index == 1:
                axis.set_xlabel(r"True null-like fraction $\rho_\star$")
            if column_index == 0:
                axis.set_ylabel("Type II error")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.052),
        ncol=4,
        frameon=False,
        fontsize=8.5,
    )
    fig.suptitle("Robustness to independent null-like pivot replacement", fontsize=15)
    fig.text(
        0.5,
        0.015,
        "All fixed-horizon rules are separately calibrated at 5%; rho=1 is the null-equivalence diagnostic."
        + ("\nOpen triangles: approximate 95% upper bounds (3/N) for zero observed errors."
           if any(float(row["type2_error"]) == 0 for row in rows) else ""),
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.95))
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def make_slide_plot(
    rows: list[dict[str, object]], config: ContaminationConfig, output_stem: Path,
    schemes: tuple[str, ...] = ("gumbel", "inverse"),
    exclude_methods: tuple[str, ...] = (),
) -> None:
    """``schemes`` and ``exclude_methods`` let a Gumbel-only variant reuse this
    figure rather than fork it; both default to the full two-panel plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, colors, styles = _plot_style()
    horizon = max(config.horizons)
    display_floor = 3.0 / config.n_evaluation_alternative
    width = 11.8 if len(schemes) > 1 else 6.4
    fig, axes = plt.subplots(1, len(schemes), figsize=(width, 4.25), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, scheme in zip(axes, schemes):
        paper_method = "h_gum_star_0.005" if scheme == "gumbel" else "h_dif_star_0.01"
        panel_methods = ("bayes_shared_robust", "bayes_shared_clean", paper_method) + (
            TRGOF_METHODS if scheme in TRGOF_SCHEMES else ()
        ) + ((DIRICHLET_ROBUST_METHOD,) if scheme in DIRICHLET_SCHEMES else ()) + (
            (UNIONTAIL_ROBUST_METHOD,) if scheme in UNIONTAIL_SCHEMES else ()
        )
        if exclude_methods:
            panel_methods = tuple(m for m in panel_methods if m not in exclude_methods)
        for method in panel_methods:
            x, y = _plot_rows(rows, scheme, method, horizon)
            plot_error_estimates(
                axis,
                x,
                y,
                display_floor,
                label=labels[method],
                color=colors[method],
                linestyle=styles[method],
                marker="o",
                linewidth=2.4 if method == "bayes_shared_robust" else 1.9,
                markersize=5,
            )
        axis.set_yscale("log")
        axis.set_ylim(error_axis_floor(rows, display_floor), 1.15)
        axis.set_xlabel(r"Null-like replacement probability $\rho_\star$", fontsize=9.5)
        axis.set_title(rf"{scheme.title()} pivot, $n={horizon}$", fontsize=10.5, pad=6)
        axis.set_axisbelow(True)
        axis.grid(alpha=0.28, linewidth=0.55, color="#9aa0a6", which="major")
        axis.grid(alpha=0.12, linewidth=0.35, color="#9aa0a6", which="minor")
        axis.tick_params(labelsize=8.5)
        axis.legend(fontsize=8.2, framealpha=0.92, loc="upper left", handlelength=2.8)
    axes[0].set_ylabel("Type II error (log scale)", fontsize=9.5)
    fig.text(
        0.5,
        0.01,
        r"Separate 5% null calibration; $\rho_\star=1$ makes the alternative identical to the null."
        + ("\nOpen triangles: approximate 95% upper bounds (3/N) for zero observed errors."
           if any(float(row["type2_error"]) == 0 for row in rows) else ""),
        ha="center",
        fontsize=8.8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.99))
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
            "isolated quick/contamination subdirectory with --quick)"
        ),
    )
    parser.add_argument("--n-calibration", type=int, default=10_000)
    parser.add_argument("--n-null", type=int, default=5_000)
    parser.add_argument("--n-alternative", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--quadrature-nodes", type=int, default=96)
    parser.add_argument("--seed", type=int, default=24_040_1246)
    parser.add_argument(
        "--rho-point-masses",
        type=float,
        nargs="*",
        default=list(DEFAULT_RHO_POINT_MASSES),
        help=(
            "rho_0 values for the diagnostic pi_rho = delta_{rho_0} detectors. "
            "Pass with no values to disable them."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke run with horizons 20, 40, 80 and small Monte Carlo samples.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--replot",
        action="store_true",
        help=(
            "Redraw the figures from the existing contamination_results.csv "
            "without rerunning the sweep."
        ),
    )
    return parser.parse_args(argv)


def read_rows(path: Path) -> list[dict[str, object]]:
    """Load a written contamination_results.csv back into plottable rows."""

    numeric = {
        "rho_true": float,
        "horizon": int,
        "alpha": float,
        "type_i_error": float,
        "type_i_mc_se": float,
        "type_ii_error": float,
        "type_ii_mc_se": float,
        "power": float,
        "fixed_horizon_cutoff": float,
        "boundary_randomization_probability": float,
        "n_calibration": int,
        "n_evaluation_null": int,
        "n_evaluation_alternative": int,
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
    horizons = (100, 300, 700)
    if args.quick:
        args.n_calibration = 300
        args.n_null = 200
        args.n_alternative = 200
        args.batch_size = min(args.batch_size, 100)
        args.quadrature_nodes = min(args.quadrature_nodes, 16)
        horizons = (20, 40, 80)
    config = ContaminationConfig(
        horizons=horizons,
        n_calibration=args.n_calibration,
        n_evaluation_null=args.n_null,
        n_evaluation_alternative=args.n_alternative,
        batch_size=args.batch_size,
        seed=args.seed,
        bayes_quadrature_nodes=args.quadrature_nodes,
        rho_point_masses=tuple(float(value) for value in args.rho_point_masses),
    )
    output_dir = resolve_output_dir(args.output_dir, quick=args.quick)
    if args.replot:
        if args.skip_plots:
            raise SystemExit("--replot and --skip-plots are mutually exclusive")
        rows = read_rows(output_dir / "contamination_results.csv")
        replot_config = replace(
            config, horizons=tuple(sorted({int(r["horizon"]) for r in rows}))
        )
        make_full_plot(rows, replot_config, output_dir / "contamination_robustness")
        make_slide_plot(rows, replot_config, output_dir / "slide_bayes_contamination")
        print(
            json.dumps(
                {"replotted_from": str(output_dir / "contamination_results.csv")},
                indent=2,
            )
        )
        return
    indicators: dict[str, np.ndarray] = {}
    rows, metadata = run_contamination_benchmark(config, indicator_sink=indicators)
    write_csv(rows, output_dir / "contamination_results.csv")
    np.savez_compressed(
        output_dir / "contamination_rejection_indicators.npz",
        **indicators,
    )
    write_json(rows, metadata, output_dir / "contamination_summary.json")
    write_report(rows, metadata, output_dir / "contamination_report.md")
    diagnostic = build_rho_diagnostic(rows, metadata, config, indicators)
    write_rho_diagnostic(diagnostic, output_dir / "contamination_rho_diagnostic.json")
    if not args.skip_plots:
        make_full_plot(rows, config, output_dir / "contamination_robustness")
        make_slide_plot(rows, config, output_dir / "slide_bayes_contamination")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "rows": len(rows),
                "runtime_seconds": metadata["runtime_seconds"],
                "rho_one_max_gap": metadata[
                    "rho_one_diagnostic_max_abs_power_minus_type_i"
                ],
                "point_mass_methods": list(config.point_mass_methods()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
