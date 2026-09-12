#!/usr/bin/env python3
"""Generating-regime sweep for the shared-Delta Gumbel detector.

Motivation
----------
The manuscript currently reports that a single fixed ``Delta_0`` evaluated under
the generating spike family reproduces the shared Bayes rule's rejections
exactly, and concludes that prior averaging adds nothing.  That conclusion rests
on ONE generating configuration, ``Delta ~ Uniform(0.001, 0.5)``, which is also
the prior.  A prior is not meant to buy power at the configuration it was
written for; it is meant to buy adaptivity across configurations the analyst
cannot foresee.  Measuring it at its own generating law is not a test of what it
buys.

This experiment therefore fixes every rule and varies only the law that
generates the per-document deficit ``Delta``.  Nine regimes are used: three
random laws and six point masses, one of which (``Delta = 0.70``) lies outside
the prior's support.  The manuscript displays the six informative regimes; the
three higher-deficit regimes are retained as saturation diagnostics.  Each rule
is calibrated once at nominal 5% on the exact null and is then applied unchanged
to every regime; the Bayes rules keep the frozen ``Uniform(0.001, 0.5)`` prior
throughout and are never retuned.  The null law does not depend on the regime,
so one calibration sample legitimately serves all of them; the evaluation null
is a fresh, separate sample as elsewhere in this benchmark.

The headline is maximum regret across regimes, regret being the excess Type II
error of a rule at a regime over the best rule at that regime.  That is what an
analyst pays for having to commit to a rule before seeing the document, which is
the situation the pivotal framework exists to address.

Scope: Gumbel scheme only, shared-Delta hierarchy, equal-tail spike NTP, exactly
the generator used by ``shared_delta_equal_tail_sensitivity`` in
``benchmark_paper_experiment``.  "Best rule" and regret refer only to the
finite, prespecified competitor menu below; this sweep does not optimize over
all point deficits or all possible priors.  Depends only on NumPy.

The generator is equal-tail throughout.  Two rules in the menu carry a strictly
larger component family than that generator -- a prior on the Dirichlet tail
concentration, and a prior on the number of live tail coordinates -- and both
contain the equal-tail spike as one atom, so they are enlargements of the
menu's other rules rather than rivals drawn from a different model.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterator

import numpy as np

from experiment_metadata import union_tail_metadata

import benchmark_paper_experiment as paper
import dirichlet_detector as dirichlet


DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "bayesian_paper_benchmark"
)
DEFAULT_QUICK_RESULTS_DIR = DEFAULT_RESULTS_DIR / "quick" / "regime_sweep"

# Distinct from 240401245 (clean benchmark), 240401246 (contamination) and
# 240401247 (contamination quadrature check).
SWEEP_SEED = 24_040_1248

# The maximum-regret functional selects both the best rule within a regime and
# the worst regime for each rule.  Its Monte Carlo standard error is therefore
# estimated by a paired document bootstrap rather than by a binomial formula.
# This salt addresses bootstrap streams without perturbing any simulation
# stream used for the reported point estimates.
MAX_REGRET_BOOTSTRAP_SEED_SALT = 91_730_041
DEFAULT_MAX_REGRET_BOOTSTRAP_REPLICATES = 2_000
DEFAULT_MAX_REGRET_BOOTSTRAP_BATCH_SIZE = 32

# Point-mass spike scores h^sp_{Delta_0} entered as competitors.
DEFAULT_SPIKE_DELTAS = (0.005, 0.01, 0.05, 0.2, 0.4)

# Supplementary horizons used only to locate where the easy high-deficit
# regimes stop being saturated.  They are not part of the reported design.
PROBE_HORIZONS = (5, 10, 20, 50)

# The reference score carried over from the clean benchmark.
# The three tuning constants li2025framework publishes.  Scoring only one of
# them left the sensitivity-to-tuning claim resting on the matched real data
# alone, with nothing behind it in the synthetic sweeps.
DEFAULT_PAPER_DELTAS: tuple[float, ...] = (0.1, 0.01, 0.005)
# The three reference scores the rest of the study compares against.  The
# sweeps scored only the least-favorable family and the spike point rules,
# so the regret tables could not say how the Bayes rules stand against the
# menu a practitioner actually chooses from.
REFERENCE_SCORES: tuple[str, ...] = ("h_ars", "h_log", "h_ind_1_over_e")

# The shared-Delta rule that additionally carries a hyperprior on the Dirichlet
# tail concentration.  Gumbel only, which is the whole of this sweep.
DIRICHLET_RULE = "bayes_shared_dirichlet"

# The shared-Delta rule that instead carries a prior on the number of live tail
# coordinates J, rather than fixing the tail to be equal over all V-1 of them.
# J = V-1 is exactly the equal-tail spike every other Gumbel rule in this sweep
# assumes, so the family contains those rules rather than rivalling them.  The
# component is closed form, so unlike the Dirichlet tail-shape layer it needs no
# transform table and no interpolation.  Shared hierarchy only: the tokenwise
# rule averages over J before multiplying and so cannot learn a document-level
# tail width.  Gumbel only, which is the whole of this sweep.
UNIONTAIL_RULE = "bayes_shared_uniontail"


def _format_delta(delta: float) -> str:
    return f"{float(delta):g}"


def spike_rule_name(delta: float) -> str:
    """Name of the fixed point-mass spike score at ``delta``."""

    return f"h_spike_{_format_delta(delta)}"


def paper_rule_name(delta: float) -> str:
    """Name of the least-favorable Gumbel score evaluated by Li et al. (2025)."""

    return f"h_gum_star_{_format_delta(delta)}"


@dataclass(frozen=True)
class Regime:
    """One data-generating law for the per-document deficit ``Delta``.

    ``kind='uniform'`` draws ``Delta ~ Uniform(low, high)`` once per document;
    ``kind='point'`` fixes ``Delta = low`` for every document.  In both cases the
    NTP is the equal-tail spike and the drawn ``Delta`` is shared by all tokens
    of the document, matching ``shared_delta_equal_tail_sensitivity``.
    """

    label: str
    kind: str
    low: float
    high: float
    description: str

    def validate(self) -> None:
        if self.kind not in ("uniform", "point"):
            raise ValueError("regime kind must be 'uniform' or 'point'")
        if not 0.0 < self.low < 1.0:
            raise ValueError("regime low must lie in (0, 1)")
        if not 0.0 < self.high < 1.0:
            raise ValueError("regime high must lie in (0, 1)")
        if self.kind == "point" and self.low != self.high:
            raise ValueError("a point regime must have low == high")
        if self.kind == "uniform" and not self.low < self.high:
            raise ValueError("a uniform regime must have low < high")

    @property
    def is_point(self) -> bool:
        return self.kind == "point"

    @property
    def mean_delta(self) -> float:
        return float(self.low) if self.is_point else 0.5 * (self.low + self.high)

    def sample(self, rng: np.random.Generator, rows: int) -> np.ndarray:
        """One ``Delta`` per document, returned with shape ``(rows, 1)``.

        A point regime consumes no randomness, so the pivot draws that follow it
        are unaffected by the degenerate branch.  A uniform regime draws exactly
        as ``simulate_gumbel_alternative_shared_delta`` does, which keeps regime
        A bit-for-bit identical to the existing shared-Delta generator.
        """

        if rows < 1:
            raise ValueError("rows must be positive")
        if self.is_point:
            return np.full((rows, 1), float(self.low), dtype=float)
        return rng.uniform(self.low, self.high, size=(rows, 1))

    def within(self, prior_low: float, prior_high: float) -> bool:
        """Whether the regime's whole support sits inside the prior support."""

        return prior_low <= self.low and self.high <= prior_high


DEFAULT_REGIMES = (
    Regime(
        "A",
        "uniform",
        0.001,
        0.5,
        "Delta ~ Uniform(0.001, 0.5); exactly the frozen prior, the manuscript's "
        "single generating configuration",
    ),
    Regime(
        "B",
        "uniform",
        0.001,
        0.05,
        "Delta ~ Uniform(0.001, 0.05); low deficit, near-deterministic NTP",
    ),
    Regime(
        "C",
        "uniform",
        0.2,
        0.5,
        "Delta ~ Uniform(0.2, 0.5); high deficit",
    ),
    Regime("D", "point", 0.005, 0.005, "Delta = 0.005 fixed; point mass, tiny deficit"),
    Regime("E", "point", 0.40, 0.40, "Delta = 0.40 fixed; point mass, large deficit"),
    Regime(
        "F",
        "point",
        0.70,
        0.70,
        "Delta = 0.70 fixed; point mass OUTSIDE the prior's support, which caps at 0.5",
    ),
    # Regime D coincides with a tested Delta_0 and with the reference score of
    # Li et al., which favours those rules there.  G, H and I are point deficits
    # that no competitor assumes: below the tested grid, strictly between its two
    # smallest atoms, and strictly between .01 and .05.
    Regime(
        "G",
        "point",
        0.002,
        0.002,
        "Delta = 0.002 fixed; point mass below every tested Delta_0",
    ),
    Regime(
        "H",
        "point",
        0.0075,
        0.0075,
        "Delta = 0.0075 fixed; point mass strictly between the tested Delta_0 "
        "atoms 0.005 and 0.01",
    ),
    Regime(
        "I",
        "point",
        0.02,
        0.02,
        "Delta = 0.02 fixed; point mass strictly between the tested Delta_0 "
        "atoms 0.01 and 0.05",
    ),
)


@dataclass(frozen=True)
class RegimeSweepConfig:
    vocabulary_size: int = 1000
    horizons: tuple[int, ...] = (100, 300, 700)
    alpha: float = 0.05
    # The frozen prior.  Never a function of the regime.
    # Detector prior only.  The generating regimes below are unchanged; the
    # detector is deliberately not told their range.
    prior_low: float = 0.001
    prior_high: float = 0.5
    allow_large_delta: bool = False
    spike_deltas: tuple[float, ...] = DEFAULT_SPIKE_DELTAS
    paper_deltas: tuple[float, ...] = DEFAULT_PAPER_DELTAS
    regimes: tuple[Regime, ...] = DEFAULT_REGIMES
    n_calibration: int = 10_000
    n_evaluation_null: int = 5_000
    n_evaluation_alternative: int = 5_000
    batch_size: int = 500
    seed: int = SWEEP_SEED
    bayes_quadrature_nodes: int = 96
    gumbel_lookup_size: int = 80_001
    gumbel_lookup_logit_limit: float = 30.0
    dirichlet_c_nodes: int = dirichlet.DEFAULT_C_NODES
    dirichlet_alpha_grid: tuple[float, ...] = dirichlet.DEFAULT_ALPHA_GRID
    max_regret_bootstrap_replicates: int = DEFAULT_MAX_REGRET_BOOTSTRAP_REPLICATES
    max_regret_bootstrap_batch_size: int = DEFAULT_MAX_REGRET_BOOTSTRAP_BATCH_SIZE

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)

    def paper_config(self) -> paper.BenchmarkConfig:
        """The clean-benchmark config carrying the frozen prior.

        Only ``delta_low``/``delta_high`` matter for the Bayes rules, and they
        are the prior's endpoints, never a regime's.
        """

        return paper.BenchmarkConfig(
            vocabulary_size=self.vocabulary_size,
            max_horizon=self.max_horizon,
            alpha=self.alpha,
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
        """The frozen joint (Delta, alpha) prior, built from the config alone.

        Takes no regime argument, so by construction it cannot be retuned to the
        deficit law that generated the data.
        """

        deltas, weights = frozen_delta_quadrature(self)
        return dirichlet.DirichletBayesGrid(
            delta_grid=deltas,
            delta_weights=weights,
            alpha_grid=self.dirichlet_alpha_grid,
            tail_size=self.vocabulary_size - 1,
            c_nodes=self.dirichlet_c_nodes,
            allow_large_delta=self.allow_large_delta,
        )

    def uniontail_grid(self) -> dirichlet.UnionTailBayesGrid:
        """The frozen joint (Delta, J) prior, built from the config alone.

        Takes no regime argument, so by construction it cannot be retuned to the
        deficit law that generated the data.  The deficit factor is the very
        same Gauss-Legendre grid every other Bayes score in this sweep uses; the
        width factor is uniform on the default dyadic ladder
        ``1, 4, 16, ..., V-1``, whose last atom is the equal-tail spike.
        """

        deltas, weights = frozen_delta_quadrature(self)
        return dirichlet.UnionTailBayesGrid(
            delta_grid=deltas,
            delta_weights=weights,
            tail_size=self.vocabulary_size - 1,
            alpha_grid=self.dirichlet_alpha_grid,
            tail_width_grid=None,
            c_nodes=self.dirichlet_c_nodes,
            allow_large_delta=self.allow_large_delta,
        )

    def rule_names(self) -> tuple[str, ...]:
        return (
            tuple(spike_rule_name(delta) for delta in self.spike_deltas)
            + (
                "bayes_shared",
                DIRICHLET_RULE,
                UNIONTAIL_RULE,
                "bayes_tokenwise",
            )
            + tuple(paper_rule_name(delta) for delta in self.paper_deltas)
            + REFERENCE_SCORES
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
        if not self.spike_deltas:
            raise ValueError("at least one spike Delta_0 is required")
        if len(set(self.spike_deltas)) != len(self.spike_deltas):
            raise ValueError("spike_deltas must be distinct")
        for delta in self.spike_deltas:
            if not 0.0 < float(delta) <= max_delta + 1e-12:
                raise ValueError(
                    "spike_deltas must lie in (0, 1 - 1/vocabulary_size]"
                )
        if not self.paper_deltas:
            raise ValueError("paper_deltas must not be empty")
        if len(set(self.paper_deltas)) != len(self.paper_deltas):
            raise ValueError("paper_deltas must be distinct")
        for delta in self.paper_deltas:
            if not 0.0 < float(delta) <= max_delta + 1e-12:
                raise ValueError(
                    "paper_deltas must lie in (0, 1 - 1/vocabulary_size]"
                )
        if not self.dirichlet_alpha_grid:
            raise ValueError("dirichlet_alpha_grid must be nonempty")
        if len(
            {dirichlet.format_alpha(a) for a in self.dirichlet_alpha_grid}
        ) != len(self.dirichlet_alpha_grid):
            raise ValueError("dirichlet_alpha_grid values must be distinct")
        for value in self.dirichlet_alpha_grid:
            if math.isnan(float(value)) or float(value) <= 0.0:
                raise ValueError(
                    "dirichlet_alpha_grid values must be positive (inf allowed)"
                )
        # Above one half the tail-width component's residual coordinates
        # outweigh the designated leading one, so Delta stops naming a
        # top-probability deficit.  That is a change of interpretation, not of
        # validity, and ``allow_large_delta`` is the explicit opt-in.
        if (float(np.max(frozen_delta_quadrature(self)[0])) > 0.5 + 1e-12
                and not self.allow_large_delta):
            raise ValueError(
                "prior_high above 0.5 requires allow_large_delta=True: the "
                "tail-width component's residual coordinates then outweigh the "
                "designated leading one"
            )
        # Containment is the point of the layer, so refuse a vocabulary whose
        # dyadic ladder would omit the equal-tail atom J = V-1.
        ladder = dirichlet.dyadic_tail_width_grid(self.vocabulary_size - 1)
        if not ladder or int(ladder[-1]) != self.vocabulary_size - 1:
            raise ValueError(
                "the dyadic tail-width ladder must end at the equal-tail atom "
                "J = vocabulary_size - 1"
            )
        if not self.regimes:
            raise ValueError("at least one regime is required")
        if len(set(self.regime_labels())) != len(self.regimes):
            raise ValueError("regime labels must be distinct")
        for regime in self.regimes:
            regime.validate()
            if regime.high > max_delta + 1e-12:
                raise ValueError(
                    "regime deficits must not exceed 1 - 1/vocabulary_size"
                )
        if len(set(self.rule_names())) != len(self.rule_names()):
            raise ValueError("rule names must be distinct")
        if self.max_regret_bootstrap_replicates < 2:
            raise ValueError("max_regret_bootstrap_replicates must be at least 2")
        if self.max_regret_bootstrap_batch_size < 1:
            raise ValueError("max_regret_bootstrap_batch_size must be positive")


def _json_safe_config(config: RegimeSweepConfig) -> dict[str, object]:
    """``asdict`` with the alpha grid rendered as strings.

    ``math.inf`` is the equal-tail limit and a legitimate grid member, but
    ``json.dumps`` writes it as the bare token ``Infinity``, which no strict
    JSON parser accepts.

    The tail-width ladder is not a configuration field -- the sweep always takes
    the package default -- so it is recorded here as a derived key so the
    artifact still pins down which prior was used.
    """

    payload = asdict(config)
    payload["dirichlet_alpha_grid"] = [
        dirichlet.format_alpha(value) for value in config.dirichlet_alpha_grid
    ]
    payload["tail_width_grid"] = [
        int(width)
        for width in dirichlet.dyadic_tail_width_grid(config.vocabulary_size - 1)
    ]
    payload["tail_width_grid_base"] = int(dirichlet.DEFAULT_TAIL_WIDTH_BASE)
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


def frozen_delta_quadrature(
    config: RegimeSweepConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights for the frozen ``Delta`` prior.

    Takes only the configuration, so by construction it cannot see a regime.
    Every Bayes score in this sweep is built from this one grid.
    """

    nodes, weights = np.polynomial.legendre.leggauss(config.bayes_quadrature_nodes)
    # No NTP on V tokens has a deficit above 1 - 1/V, so a whole-interval prior
    # is clamped there rather than placing nodes on unrepresentable vectors.
    prior_high = min(config.prior_high, 1.0 - 1.0 / config.vocabulary_size)
    deltas = config.prior_low + 0.5 * (nodes + 1.0) * (
        prior_high - config.prior_low
    )
    return deltas, 0.5 * weights


def prior_fingerprint(config: RegimeSweepConfig) -> dict[str, object]:
    """A small numeric signature of the frozen prior, recorded per regime."""

    deltas, weights = frozen_delta_quadrature(config)
    return {
        "delta": {
            "family": "uniform",
            "low": float(config.prior_low),
            "high": float(config.prior_high),
            "quadrature": "gauss_legendre",
            "nodes": int(config.bayes_quadrature_nodes),
            "node_sum": float(deltas.sum()),
            "weight_sum": float(weights.sum()),
            "node_min": float(deltas.min()),
            "node_max": float(deltas.max()),
        },
        "alpha": {
            "family": "uniform on a fixed grid",
            "grid": [
                dirichlet.format_alpha(value)
                for value in config.dirichlet_alpha_grid
            ],
            "used_by": [DIRICHLET_RULE],
            "note": (
                "inf is the equal-tail limit, so the spike-family rules are "
                "components of the Dirichlet mixture rather than rivals to it."
            ),
        },
        "tail_width": {
            "family": "uniform on a fixed scale-free ladder",
            "grid": [
                int(width)
                for width in dirichlet.dyadic_tail_width_grid(
                    config.vocabulary_size - 1
                )
            ],
            "base": int(dirichlet.DEFAULT_TAIL_WIDTH_BASE),
            "used_by": [UNIONTAIL_RULE],
            "note": (
                "J = vocabulary_size - 1 is the equal-tail spike, so the "
                "spike-family rules are components of the tail-width mixture "
                "rather than rivals to it; J = 1 is the sparsest tail at that "
                "deficit."
            ),
        },
        # Kept at the top level so existing readers of the flat keys still work.
        "family": "uniform",
        "low": float(config.prior_low),
        "high": float(config.prior_high),
        "quadrature": "gauss_legendre",
        "nodes": int(config.bayes_quadrature_nodes),
        "node_sum": float(deltas.sum()),
        "weight_sum": float(weights.sum()),
        "node_min": float(deltas.min()),
        "node_max": float(deltas.max()),
    }


def simulate_regime_pivots(
    rng: np.random.Generator,
    rows: int,
    horizon: int,
    regime: Regime,
    vocabulary_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Gumbel pivots for one shared-Delta equal-tail spike regime.

    Returns ``(pivots, deltas)`` where ``deltas`` has one entry per document.
    The body after the ``Delta`` draw is the body of
    ``paper.simulate_gumbel_alternative_shared_delta``; only the law of ``Delta``
    changes across regimes.
    """

    regime.validate()
    if vocabulary_size < 2:
        raise ValueError("vocabulary_size must be at least 2")
    column = regime.sample(rng, rows)
    deltas = np.repeat(column, horizon, axis=1)
    selected_top = rng.uniform(size=(rows, horizon)) >= deltas
    selected_probability = np.where(
        selected_top, 1.0 - deltas, deltas / (vocabulary_size - 1.0)
    )
    pivots = rng.uniform(size=(rows, horizon)) ** selected_probability
    return pivots, column[:, 0]


def _additive_increments(
    batch: np.ndarray,
    config: RegimeSweepConfig,
    lookup: paper.GumbelBayesLookup,
) -> Iterator[tuple[str, np.ndarray]]:
    """Per-token score increments for every rule that is a plain token sum."""

    for delta in config.spike_deltas:
        yield (
            spike_rule_name(delta),
            paper.spike_point_mass_score(
                batch, float(delta), config.vocabulary_size
            ),
        )
    for delta in config.paper_deltas:
        yield (
            paper_rule_name(delta),
            paper.paper_gumbel_optimal_score(batch, float(delta)),
        )
    for name in REFERENCE_SCORES:
        yield name, paper.gumbel_score(batch, name, None)
    # Frozen-prior tokenwise Bayes: the lookup is built once from the prior.
    yield "bayes_tokenwise", lookup(batch)


def _shared_bayes_batch(
    batch: np.ndarray,
    deltas: np.ndarray,
    log_weights: np.ndarray,
    config: RegimeSweepConfig,
) -> np.ndarray:
    """Shared-Delta log Bayes factor at the selected horizons for one batch."""

    vocabulary_size = config.vocabulary_size
    top_exponents = deltas / (1.0 - deltas)
    tail_exponents = (vocabulary_size - 1.0) / deltas - 1.0
    log_tail_count = math.log(vocabulary_size - 1.0)
    record = {horizon - 1: index for index, horizon in enumerate(config.horizons)}

    log_pivots = np.log(batch)
    accumulator = np.zeros((batch.shape[0], deltas.size), dtype=float)
    output = np.empty((batch.shape[0], len(config.horizons)), dtype=float)
    for time_index in range(config.max_horizon):
        log_r = log_pivots[:, time_index, None]
        accumulator += np.logaddexp(
            log_r * top_exponents[None, :],
            log_tail_count + log_r * tail_exponents[None, :],
        )
        column = record.get(time_index)
        if column is not None:
            output[:, column] = _logsumexp_rows(
                accumulator + log_weights[None, :]
            )
    return output


def all_rule_paths(
    pivots: np.ndarray,
    config: RegimeSweepConfig,
    lookup: paper.GumbelBayesLookup,
    dirichlet_grid: dirichlet.DirichletBayesGrid | None = None,
    uniontail_grid: dirichlet.UnionTailBayesGrid | None = None,
) -> dict[str, np.ndarray]:
    """Every rule's statistic at the selected horizons.

    There is deliberately no ``regime`` parameter.  The frozen priors enter only
    through ``config``, ``dirichlet_grid`` and ``uniontail_grid``, so no rule in
    this sweep can be retuned to the regime that produced ``pivots``.
    """

    if pivots.ndim != 2 or pivots.shape[1] < config.max_horizon:
        raise ValueError("pivots must be (rows, >= max_horizon)")
    rows = pivots.shape[0]
    selected = np.asarray(config.horizons, dtype=int) - 1
    deltas, weights = frozen_delta_quadrature(config)
    log_weights = np.log(weights)
    if dirichlet_grid is None:
        dirichlet_grid = config.dirichlet_grid()
    if uniontail_grid is None:
        uniontail_grid = config.uniontail_grid()

    output = {
        name: np.empty((rows, len(config.horizons)), dtype=float)
        for name in config.rule_names()
    }
    for start in range(0, rows, config.batch_size):
        stop = min(start + config.batch_size, rows)
        batch = pivots[start:stop, : config.max_horizon]
        for name, increments in _additive_increments(batch, config, lookup):
            output[name][start:stop] = np.cumsum(increments, axis=1)[:, selected]
        output["bayes_shared"][start:stop] = _shared_bayes_batch(
            batch, deltas, log_weights, config
        )
        values, _ = dirichlet_grid.shared_paths(batch, horizons=config.horizons)
        output[DIRICHLET_RULE][start:stop] = values
        # Same shared-hierarchy collector, closed-form component, no table.
        widths, _ = uniontail_grid.shared_paths(batch, horizons=config.horizons)
        output[UNIONTAIL_RULE][start:stop] = widths
    return output


def is_benchmark_rule(rule: str) -> bool:
    """Whether a rule may set the regret benchmark.

    The fixed-Delta_0 equal-tail scores are diagnostic devices, not detectors
    anyone would deploy: each is handed the generating family and one true
    deficit.  Letting them set the bar measures every real rule against an
    oracle, so they are scored and reported but excluded from the minimum.
    """

    return not rule.startswith("h_spike_")


def regret_table(
    type2_by_rule: dict[str, dict[str, float]],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Regret of each rule at each regime, and the per-regime best Type II.

    ``regret[rule][regime] = type2[rule][regime] - min_rule type2[rule][regime]``
    over the benchmark rules only; see :func:`is_benchmark_rule`.  The
    subtrahend is one of the very floats being subtracted, so the result is
    exactly non-negative and exactly zero for a benchmark rule attaining the
    minimum.  A diagnostic that beats every benchmark rule therefore has
    NEGATIVE regret, which is the honest way to show it.
    """

    if not type2_by_rule:
        raise ValueError("type2_by_rule must be non-empty")
    rules = list(type2_by_rule)
    regimes = list(type2_by_rule[rules[0]])
    for rule in rules:
        if list(type2_by_rule[rule]) != regimes:
            raise ValueError("every rule must be scored on the same regimes")
    benchmark = [rule for rule in rules if is_benchmark_rule(rule)]
    if not benchmark:
        raise ValueError("no benchmark rule: every rule was excluded")
    best = {
        regime: min(type2_by_rule[rule][regime] for rule in benchmark)
        for regime in regimes
    }
    regret = {
        rule: {
            regime: type2_by_rule[rule][regime] - best[regime] for regime in regimes
        }
        for rule in rules
    }
    return regret, best


def max_regret(
    regret: dict[str, dict[str, float]],
) -> dict[str, dict[str, object]]:
    """Worst-case regret across regimes, with the regime attaining it.

    Ties are broken by the regime order of ``regret``.  A rule whose maximum
    regret is exactly zero is on the lower envelope everywhere, so no regime
    "attains" its worst case in any meaningful sense and ``argmax_regime`` is
    reported as ``None``.
    """

    summary: dict[str, dict[str, object]] = {}
    for rule, row in regret.items():
        if not row:
            raise ValueError("each rule must be scored on at least one regime")
        worst_regime = max(row, key=row.__getitem__)
        worst_value = float(row[worst_regime])
        summary[rule] = {
            "max_regret": worst_value,
            "argmax_regime": None if worst_value == 0.0 else worst_regime,
        }
    return summary


def pooled_max_regret(
    regret_by_horizon: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, object]]:
    """Worst-case regret over every (horizon, regime) cell, per rule.

    This is the summary for an analyst who must commit to one rule without
    knowing either the deficit regime or the document length.
    """

    if not regret_by_horizon:
        raise ValueError("regret_by_horizon must be non-empty")
    horizons = list(regret_by_horizon)
    rules = list(regret_by_horizon[horizons[0]])
    summary: dict[str, dict[str, object]] = {}
    for rule in rules:
        candidates = [
            (float(regret_by_horizon[horizon][rule][regime]), horizon, regime)
            for horizon in horizons
            for regime in regret_by_horizon[horizon][rule]
        ]
        worst_value = max(value for value, _, _ in candidates)
        worst = next(item for item in candidates if item[0] == worst_value)
        summary[rule] = {
            "max_regret": worst_value,
            "argmax_horizon": None if worst_value == 0.0 else int(worst[1]),
            "argmax_regime": None if worst_value == 0.0 else worst[2],
        }
    return summary


def _binomial_se(rate: float, trials: int) -> float:
    return math.sqrt(max(rate * (1.0 - rate), 0.0) / trials)


def conditional_miss_contributions(
    values: np.ndarray,
    cutoff: np.ndarray,
    boundary_probability: np.ndarray,
) -> np.ndarray:
    """Per-document miss contributions for a randomized-boundary rule.

    The aggregate benchmark reports the Rao--Blackwellized expectation over
    the boundary coin.  A strict miss contributes one, a strict rejection zero,
    and a score exactly at the cutoff contributes ``1 - gamma``.  Retaining
    these document-level contributions preserves pairing across rules and
    horizons for the regret bootstrap.
    """

    scores = np.asarray(values, dtype=float)
    thresholds = np.asarray(cutoff, dtype=float)
    gammas = np.asarray(boundary_probability, dtype=float)
    if scores.ndim != 2:
        raise ValueError("values must have shape (documents, horizons)")
    if thresholds.shape != (scores.shape[1],):
        raise ValueError("cutoff must have one value per horizon")
    if gammas.shape != thresholds.shape:
        raise ValueError("boundary_probability must match cutoff")
    if np.any((gammas < 0.0) | (gammas > 1.0)):
        raise ValueError("boundary probabilities must lie in [0, 1]")
    return (scores < thresholds[None, :]).astype(float) + (
        scores == thresholds[None, :]
    ) * (1.0 - gammas[None, :])


def paired_bootstrap_means(
    losses: np.ndarray,
    rng: np.random.Generator,
    *,
    replicates: int,
    batch_size: int,
) -> np.ndarray:
    """Bootstrap mean loss while pairing rules and horizons by document."""

    values = np.asarray(losses, dtype=float)
    if values.ndim != 3:
        raise ValueError("losses must have shape (documents, rules, horizons)")
    documents, rules, horizons = values.shape
    if documents < 1 or rules < 1 or horizons < 1:
        raise ValueError("every losses dimension must be positive")
    if replicates < 2:
        raise ValueError("replicates must be at least 2")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    means = np.empty((replicates, rules, horizons), dtype=float)
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        indices = rng.integers(
            0, documents, size=(stop - start, documents), dtype=np.intp
        )
        means[start:stop] = values[indices].mean(axis=1)
    return means


def bootstrap_max_regret_mc_se(
    bootstrap_type2: np.ndarray,
    benchmark_mask: "np.ndarray | None" = None,
) -> tuple[np.ndarray, np.ndarray]:
    """MC SEs after recomputing the lower envelope and maxima per draw.

    ``bootstrap_type2`` has shape ``(replicates, regimes, rules, horizons)``.
    The first result has shape ``(rules, horizons)`` for the horizon-specific
    maximum regret; the second has shape ``(rules,)`` for the maximum pooled
    across regimes and horizons.
    """

    values = np.asarray(bootstrap_type2, dtype=float)
    if values.ndim != 4:
        raise ValueError(
            "bootstrap_type2 must have shape (replicates, regimes, rules, horizons)"
        )
    if values.shape[0] < 2 or any(size < 1 for size in values.shape[1:]):
        raise ValueError("bootstrap_type2 has an empty or singleton dimension")
    # The envelope must be taken over the SAME rules as the point estimate, or
    # the standard error describes a benchmark the table never reports.
    if benchmark_mask is None:
        envelope = values
    else:
        mask = np.asarray(benchmark_mask, dtype=bool)
        if mask.shape != (values.shape[2],):
            raise ValueError("benchmark_mask must have one entry per rule")
        if not mask.any():
            raise ValueError("benchmark_mask excludes every rule")
        envelope = values[:, :, mask, :]
    best = envelope.min(axis=2, keepdims=True)
    regret = values - best
    per_horizon = regret.max(axis=1)
    pooled = regret.max(axis=(1, 3))
    return (
        per_horizon.std(axis=0, ddof=1),
        pooled.std(axis=0, ddof=1),
    )


def format_table(
    title: str,
    row_labels: list[str],
    column_labels: list[str],
    values: dict[str, dict[str, float]],
    extra_columns: list[tuple[str, dict[str, float]]] | None = None,
    decimals: int = 4,
) -> str:
    """Render a small numeric table as GitHub-flavoured Markdown."""

    columns = list(column_labels) + [name for name, _ in (extra_columns or [])]
    header = "| " + " | ".join(["rule"] + columns) + " |"
    divider = "|" + "---|" * (len(columns) + 1)
    lines = [f"### {title}", "", header, divider]
    for row in row_labels:
        cells = [f"{values[row][column]:.{decimals}f}" for column in column_labels]
        for _, mapping in extra_columns or []:
            cells.append(f"{mapping[row]:.{decimals}f}")
        lines.append("| " + " | ".join([row] + cells) + " |")
    return "\n".join(lines)


def run_regime_sweep(
    config: RegimeSweepConfig,
    probe_horizons: tuple[int, ...] | None = None,
) -> dict[str, object]:
    """Run the sweep and return the complete JSON-serializable payload.

    ``probe_horizons`` adds a supplementary short-horizon rerun whose only job is
    to locate the horizon below which the easy high-deficit regimes stop being
    saturated.  It uses its own seed and its own samples and never touches the
    reported design.
    """

    config.validate()
    started = time.perf_counter()
    paper_config = config.paper_config()
    # Built once, from the prior alone; reused for every regime.
    lookup = paper.GumbelBayesLookup(paper_config)
    dirichlet_grid = config.dirichlet_grid()
    uniontail_grid = config.uniontail_grid()

    seeds = np.random.SeedSequence(config.seed).spawn(2 + len(config.regimes))
    rng_calibration = np.random.default_rng(seeds[0])
    rng_evaluation_null = np.random.default_rng(seeds[1])
    regime_rngs = [np.random.default_rng(seed) for seed in seeds[2:]]

    rules = list(config.rule_names())
    horizons = list(config.horizons)

    # One exact-null calibration sample serves every regime: the null law of the
    # Gumbel pivot is Uniform(0,1) and does not depend on the alternative.
    calibration_pivots = paper.simulate_gumbel_null(
        rng_calibration, config.n_calibration, config.max_horizon, paper_config
    )
    calibration_paths = all_rule_paths(
        calibration_pivots, config, lookup, dirichlet_grid, uniontail_grid
    )
    cutoffs, gammas = paper.calibrate_randomized_boundary(
        calibration_paths, config.alpha
    )
    del calibration_pivots, calibration_paths

    # A fresh, separate null sample for realized size, as elsewhere.
    evaluation_null_pivots = paper.simulate_gumbel_null(
        rng_evaluation_null, config.n_evaluation_null, config.max_horizon, paper_config
    )
    null_paths = all_rule_paths(
        evaluation_null_pivots, config, lookup, dirichlet_grid, uniontail_grid
    )
    type1 = {
        rule: paper.expected_rejection_rate(
            null_paths[rule], cutoffs[rule], gammas[rule]
        )
        for rule in rules
    }
    del evaluation_null_pivots, null_paths

    cells: list[dict[str, object]] = []
    delta_diagnostics: dict[str, dict[str, object]] = {}
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
        pivots, deltas = simulate_regime_pivots(
            rng,
            config.n_evaluation_alternative,
            config.max_horizon,
            regime,
            config.vocabulary_size,
        )
        delta_diagnostics[regime.label] = {
            "kind": regime.kind,
            "low": float(regime.low),
            "high": float(regime.high),
            "description": regime.description,
            "nominal_mean_delta": regime.mean_delta,
            "realized_mean_delta": float(deltas.mean()),
            "realized_min_delta": float(deltas.min()),
            "realized_max_delta": float(deltas.max()),
            "support_inside_prior": bool(
                regime.within(config.prior_low, config.prior_high)
            ),
            "fraction_outside_prior_support": float(
                np.mean(
                    (deltas < config.prior_low) | (deltas > config.prior_high)
                )
            ),
            # Recomputed inside the regime loop from the config alone: it is the
            # same object for every regime, which is the point.
            "frozen_prior_used": prior_fingerprint(config),
        }
        alternative_paths = all_rule_paths(
            pivots, config, lookup, dirichlet_grid, uniontail_grid
        )
        del pivots
        loss_cube = np.stack(
            [
                conditional_miss_contributions(
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
        bootstrap_type2[:, regime_index] = paired_bootstrap_means(
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
                        "alpha": float(config.alpha),
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

    max_regret_mc_se, pooled_max_regret_mc_se = bootstrap_max_regret_mc_se(
        bootstrap_type2,
        np.array([is_benchmark_rule(rule) for rule in rules], dtype=bool),
    )
    del bootstrap_type2

    regret: dict[str, dict[str, dict[str, float]]] = {}
    best_by_regime: dict[str, dict[str, object]] = {}
    max_regret_by_horizon: dict[str, dict[str, dict[str, object]]] = {}
    ranking: dict[str, list[dict[str, object]]] = {}
    saturated: dict[str, list[str]] = {}

    for horizon_index, horizon in enumerate(horizons):
        key = str(horizon)
        horizon_regret, best = regret_table(type2[key])
        regret[key] = horizon_regret
        best_by_regime[key] = {
            regime: {
                "type2_error": best[regime],
                "rules": sorted(
                    rule
                    for rule in rules
                    if is_benchmark_rule(rule)
                    and type2[key][rule][regime] == best[regime]
                ),
            }
            for regime in config.regime_labels()
        }
        max_regret_by_horizon[key] = max_regret(horizon_regret)
        for rule_index, rule in enumerate(rules):
            max_regret_by_horizon[key][rule]["max_regret_mc_se"] = float(
                max_regret_mc_se[rule_index, horizon_index]
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
                rules, key=lambda name: (max_regret_by_horizon[key][name]["max_regret"], name)
            )
        ]
        saturated[key] = [
            regime
            for regime in config.regime_labels()
            if all(type2[key][rule][regime] == 0.0 for rule in rules)
        ]

    # Pooled over all horizons, the same rule facing every (horizon, regime) cell.
    pooled = pooled_max_regret(regret)
    for rule_index, rule in enumerate(rules):
        pooled[rule]["max_regret_mc_se"] = float(
            pooled_max_regret_mc_se[rule_index]
        )

    lower_envelope = {
        str(horizon): {
            "rules_attaining_every_regime_minimum": sorted(
                rule
                for rule in rules
                if all(
                    regret[str(horizon)][rule][regime] == 0.0
                    for regime in config.regime_labels()
                )
            )
        }
        for horizon in horizons
    }

    tables: dict[str, dict[str, str]] = {}
    for horizon in horizons:
        key = str(horizon)
        labels = list(config.regime_labels())
        tables[key] = {
            "type2": format_table(
                f"Type II error at n={horizon} (nominal 5%, 5,000 documents per cell)",
                rules,
                labels,
                type2[key],
            ),
            "regret": format_table(
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

    probe: dict[str, object] | None = None
    if probe_horizons:
        probe_config = replace(
            config, horizons=tuple(probe_horizons), seed=config.seed + 1
        )
        inner = run_regime_sweep(probe_config)
        first_unsaturated = {
            regime: next(
                (
                    int(horizon)
                    for horizon in probe_config.horizons
                    if regime not in inner["saturated_regimes"][str(horizon)]
                ),
                None,
            )
            for regime in probe_config.regime_labels()
        }
        probe = {
            "purpose": (
                "Supplementary only.  Locates the horizon below which the "
                "high-deficit regimes stop being saturated, so that 'Type II is "
                "zero for everyone' can be reported as a property of the "
                "horizon rather than of the rules."
            ),
            "horizons": [int(horizon) for horizon in probe_config.horizons],
            "seed": int(probe_config.seed),
            "independent_of_the_main_sweep": True,
            "type2_error": inner["type2_error"],
            "regret": inner["regret"],
            "max_regret": inner["max_regret"],
            "pooled_max_regret_across_probe_horizons": inner[
                "pooled_max_regret_across_horizons"
            ],
            "saturated_regimes": inner["saturated_regimes"],
            "shortest_horizon_at_which_a_regime_separates_rules": first_unsaturated,
        }

    elapsed = time.perf_counter() - started
    payload: dict[str, object] = {
        "description": (
            "Generating-regime sweep for the Gumbel scheme, shared-Delta "
            "hierarchy and equal-tail spike NTP.  Every rule is calibrated once "
            "at nominal 5% on the exact null and then applied unchanged to the "
            "configured "
            "generating regimes for the per-document deficit Delta.  The Bayes "
            "rules keep a frozen Uniform(0.001, 0.5) prior and are never "
            "retuned.  The headline is maximum regret across regimes: what an "
            "analyst pays for committing to a rule before seeing the document."
        ),
        "motivation": (
            "The manuscript's family diagnostic evaluates a fixed Delta_0 at the "
            "single generating configuration Delta ~ Uniform(0.001, 0.5), which "
            "is also the prior, and concludes that prior averaging adds nothing. "
            "A prior's value is adaptivity across configurations the analyst "
            "cannot foresee, so that comparison cannot settle the question.  "
            "This sweep varies only the generating law of Delta."
        ),
        "scheme": "gumbel",
        "hierarchy": "shared_delta_equal_tail_spike",
        "generator": (
            "One Delta per document, NTP (1-Delta, Delta/(M-1), ..., "
            "Delta/(M-1)), pivot R = U**P_selected; identical to "
            "benchmark_paper_experiment.simulate_gumbel_alternative_shared_delta "
            "except for the law of Delta."
        ),
        "seed": int(config.seed),
        "seed_note": (
            "Fresh seed, distinct from 240401245 (clean benchmark), 240401246 "
            "(contamination) and 240401247 (contamination quadrature check).  "
            "Spawned by SeedSequence into one stream for calibration, one for "
            "the evaluation null, and one per regime."
        ),
        "config": _json_safe_config(config),
        "frozen_prior": prior_fingerprint(config),
        "frozen_prior_note": (
            "Uniform(0.001, 0.5) with 96 Gauss-Legendre nodes, fixed before the "
            "sweep and shared by bayes_shared, bayes_tokenwise, "
            "bayes_shared_dirichlet and bayes_shared_uniontail at every regime.  "
            "The Dirichlet rule "
            "additionally carries a frozen uniform prior on the tail "
            "concentration, so its joint grid has 96 x 6 = 576 components; the "
            "generator here is equal-tail throughout, which is the alpha = inf "
            "atom of that prior. The union combines this shape branch with a "
            "uniform width prior on the ladder strictly below V-1; its actual "
            "branch weights and component counts are in union_tail_layer. "
            "The equal-tail atom lies in the shape branch. "
            "Regime F generates Delta = 0.70, outside the "
            "deficit support.  Regimes G, H and I are point deficits that "
            "match no tested Delta_0."
        ),
        "union_tail_layer": {
            **union_tail_metadata(uniontail_grid, methods=[UNIONTAIL_RULE], tokenwise=False),
            'containment_check': {
                "shape_equal_tail_atom_vs_closed_form_spike_max_abs_log_density_difference": (
                    uniontail_grid.spike_agreement(
                        np.random.default_rng(config.seed).uniform(size=20_000)
                    )
                ),
                "note": (
                    "Evaluated on a throwaway generator seeded from config.seed, "
                    "which consumes no randomness from any stream used by the "
                    "reported simulation."
                ),
            },
            'effect_on_pre_existing_numbers': "The rule is additive: it consumes no randomness and does not "
                "touch any other rule's statistic, so every per-rule quantity "
                "(cutoffs, boundary randomization, Type I, power, Type II) is "
                "unchanged.  Regret is min-relative over the menu by definition, "
                "so regret, best_rule_by_regime, max_regret and the rankings do "
                "move wherever the new rule attains a new per-regime minimum.",
        },
        "calibration": {
            "n_calibration": int(config.n_calibration),
            "n_evaluation_null": int(config.n_evaluation_null),
            "n_evaluation_alternative": int(config.n_evaluation_alternative),
            "rule": (
                "score > c plus a randomized score == c boundary, targeting "
                "alpha in the calibration sample; the reported rate is the "
                "Rao-Blackwellized expectation over the boundary coin."
            ),
            "shared_across_regimes": (
                "The Gumbel pivot null is Uniform(0,1) and does not depend on "
                "the regime, so one calibration sample serves all configured "
                "regimes.  "
                "The evaluation null is a fresh, separate sample."
            ),
            "cutoffs": {
                rule: {
                    str(horizon): float(cutoffs[rule][index])
                    for index, horizon in enumerate(horizons)
                }
                for rule in rules
            },
            "boundary_randomization": {
                rule: {
                    str(horizon): float(gammas[rule][index])
                    for index, horizon in enumerate(horizons)
                }
                for rule in rules
            },
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
                "regime and jointly across all rules and horizons.  Each draw "
                "recomputes the best rule within every regime and then the "
                "maximum regret across regimes."
            ),
            "conditional_on_realized_calibration": True,
            "excluded_uncertainty": (
                "Calibration-sample, quadrature, finite-menu selection and "
                "model uncertainty are not included."
            ),
        },
        "horizons": [int(horizon) for horizon in horizons],
        "rules": [
            {
                "name": rule,
                "kind": (
                    "fixed_point_mass_spike"
                    if rule.startswith("h_spike_")
                    else "paper_least_favorable"
                    if rule.startswith("h_gum_star_")
                    else "bayes_frozen_prior_dirichlet_tail"
                    if rule == DIRICHLET_RULE
                    else "bayes_frozen_prior_union_tail"
                    if rule == UNIONTAIL_RULE
                    else "bayes_frozen_prior_spike_tail"
                ),
                "component_family": (
                    "dirichlet_tail_mixture"
                    if rule == DIRICHLET_RULE
                    else "union_tail_mixture"
                    if rule == UNIONTAIL_RULE
                    else "equal_tail_spike"
                ),
                "retuned_per_regime": False,
            }
            for rule in rules
        ],
        "competitor_scope": (
            "Best-rule and regret summaries are relative only to the configured "
            "finite benchmark rule menu, excluding the h_spike fixed-deficit "
            "diagnostics in config.spike_deltas. Diagnostics retain Type II and "
            "regret entries for comparison but neither set the benchmark minimum "
            "nor appear in best_rule_by_regime. These summaries do not optimize "
            "over a continuum of deficits or over alternative priors."
        ),
        "regimes": [
            {
                "label": regime.label,
                "kind": regime.kind,
                "low": float(regime.low),
                "high": float(regime.high),
                "description": regime.description,
                "support_inside_prior": bool(
                    regime.within(config.prior_low, config.prior_high)
                ),
            }
            for regime in config.regimes
        ],
        "regime_delta_diagnostics": delta_diagnostics,
        "type1_error": {
            rule: {
                str(horizon): float(type1[rule][index])
                for index, horizon in enumerate(horizons)
            }
            for rule in rules
        },
        "type2_error": type2,
        "regret": regret,
        "best_rule_by_regime": best_by_regime,
        "max_regret": max_regret_by_horizon,
        "max_regret_ranking": ranking,
        "pooled_max_regret_across_horizons": pooled,
        "rules_attaining_the_lower_envelope": lower_envelope,
        "saturated_regimes": saturated,
        "saturation_note": (
            "A regime is listed as saturated at a horizon when every rule has "
            "Type II error exactly zero there, so the cell cannot separate rules "
            "and contributes no regret to anyone."
        ),
        "short_horizon_saturation_probe": probe,
        "cells": cells,
        "markdown_tables": tables,
        "runtime_seconds": elapsed,
    }
    return payload


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
            "isolated quick/regime_sweep subdirectory with --quick)"
        ),
    )
    parser.add_argument("--n-calibration", type=int, default=10_000)
    parser.add_argument("--n-null", type=int, default=5_000)
    parser.add_argument("--n-alternative", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=SWEEP_SEED)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="tiny smoke run: 400/200/200 documents and horizons 20/60/140",
    )
    parser.add_argument(
        "--skip-probe",
        action="store_true",
        help="skip the supplementary short-horizon saturation probe",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.quick:
        config = RegimeSweepConfig(
            horizons=(20, 60, 140),
            n_calibration=400,
            n_evaluation_null=200,
            n_evaluation_alternative=200,
            batch_size=100,
            seed=args.seed,
            max_regret_bootstrap_replicates=200,
        )
    else:
        config = RegimeSweepConfig(
            n_calibration=args.n_calibration,
            n_evaluation_null=args.n_null,
            n_evaluation_alternative=args.n_alternative,
            batch_size=args.batch_size,
            seed=args.seed,
        )

    payload = run_regime_sweep(
        config, probe_horizons=None if args.skip_probe else PROBE_HORIZONS
    )
    results_dir = resolve_results_dir(args.results_dir, quick=args.quick)
    output = results_dir / "regime_sweep.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    tables = payload["markdown_tables"]
    for horizon in payload["horizons"]:
        key = str(horizon)
        print(tables[key]["type2"])
        print()
        print(tables[key]["regret"])
        print()
        print(f"max-regret ranking at n={horizon} (best first):")
        for entry in payload["max_regret_ranking"][key]:
            where = (
                "on the lower envelope everywhere"
                if entry["argmax_regime"] is None
                else f"worst at regime {entry['argmax_regime']}"
            )
            print(f"  {entry['rule']:<20s} {entry['max_regret']:.4f}  ({where})")
        saturated = payload["saturated_regimes"][key]
        print(
            "  saturated regimes (Type II exactly zero for every rule): "
            + (", ".join(saturated) if saturated else "none")
        )
        print()

    probe = payload["short_horizon_saturation_probe"]
    if probe is not None:
        print(
            "supplementary saturation probe at horizons "
            f"{probe['horizons']} (seed {probe['seed']}):"
        )
        for regime, horizon in probe[
            "shortest_horizon_at_which_a_regime_separates_rules"
        ].items():
            where = (
                "saturated at every probed horizon"
                if horizon is None
                else f"first separates rules at n={horizon}"
            )
            print(f"  regime {regime}: {where}")
        print()
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
