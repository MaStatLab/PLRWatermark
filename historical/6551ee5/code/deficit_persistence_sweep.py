"""Does partial pooling of the deficit buy anything?

Every deficit design in the study is all-or-nothing.  The shared experiment
holds ``Delta`` fixed for a whole document; the tokenwise experiment redraws it
independently at every position.  A rule that pools partially therefore has
nothing to detect in either: one endpoint rewards full pooling and the other
rewards none, and the pooled rule can only pay the dilution cost of covering
both.  This is the same gap the tail-width sweep closed for the tail blocks.

This sweep supplies the missing axis.  For a document with hyper-mean ``mu``,

    X_t | mu, kappa ~ Beta(kappa*mu, kappa*(1-mu)) i.i.d.,
    Delta_t = low + (high - low) * X_t,

so ``kappa`` is exactly a persistence dial: large ``kappa`` pins every token's
deficit near ``Delta(mu)`` and small ``kappa`` scatters it.  The two existing
designs are the endpoints, reproduced exactly rather than approximately:

  P1  kappa = inf,  mu ~ Unif(0,1)   Delta constant within a document
                                      == the shared design
  P6  kappa = 2,    mu = 1/2         Beta(1,1), so Delta_t i.i.d. uniform
                                      == the tokenwise design

with P2-P5 at kappa = 32, 8, 2, 0.5 in between.  The regimes are reported with
the realized persistence -- the share of total deficit variance that is
between-document -- so the axis is read off the data rather than off kappa.
Any gain the pooled rule shows at P2-P5 is the value of partial pooling; what
it gives up at P1 and P6 is what that flexibility costs against a correctly
specified endpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import benchmark_paper_experiment as bpe
import deficit_hierarchy as dh
import dirichlet_detector as dd
import tail_family as tf

RESULTS = Path(__file__).resolve().parents[1] / "results" / "bayesian_paper_benchmark"
SWEEP_SEED = 20260902


@dataclass(frozen=True)
class PersistenceRegime:
    label: str
    kappa: float
    mu: float | None          # None -> draw mu ~ Unif(0,1) per document
    description: str

    def deltas(
        self, rng: np.random.Generator, rows: int, horizon: int,
        low: float, high: float,
    ) -> np.ndarray:
        if self.mu is None:
            mu = rng.uniform(size=(rows, 1))
        else:
            mu = np.full((rows, 1), float(self.mu))
        if math.isinf(self.kappa):
            x = np.repeat(mu, horizon, axis=1)
        else:
            a = self.kappa * mu
            b = self.kappa * (1.0 - mu)
            x = rng.beta(a, b, size=(rows, horizon))
        return low + (high - low) * x


DEFAULT_REGIMES: tuple[PersistenceRegime, ...] = (
    PersistenceRegime("P1", math.inf, None,
                      "kappa = inf; Delta constant within a document -- exactly "
                      "the shared-deficit design"),
    PersistenceRegime("P2", 32.0, None, "kappa = 32; strongly persistent"),
    PersistenceRegime("P3", 8.0, None, "kappa = 8; moderately persistent"),
    PersistenceRegime("P4", 2.0, None, "kappa = 2; weakly persistent"),
    PersistenceRegime("P5", 0.5, None,
                      "kappa = 0.5; barely persistent, and the Beta is U-shaped "
                      "so the within-document spread is near its maximum"),
    PersistenceRegime("P6", 2.0, 0.5,
                      "kappa = 2 at mu = 1/2, i.e. Beta(1,1); Delta_t i.i.d. "
                      "uniform with no document-level component at all -- "
                      "exactly the tokenwise design"),
)


@dataclass(frozen=True)
class PersistenceConfig:
    vocabulary_size: int = 1_000
    max_horizon: int = 700
    horizons: tuple[int, ...] = (100, 300, 700)
    delta_low: float = 0.001
    delta_high: float = 0.5
    alpha_level: float = 0.05
    n_calibration: int = 10_000
    n_evaluation_null: int = 5_000
    n_evaluation_alternative: int = 5_000
    seed: int = SWEEP_SEED
    regimes: tuple[PersistenceRegime, ...] = DEFAULT_REGIMES
    pooled_prior: dh.PooledDeficitPrior = field(
        default_factory=dh.PooledDeficitPrior
    )

    def benchmark_config(self) -> bpe.BenchmarkConfig:
        return bpe.BenchmarkConfig(
            vocabulary_size=self.vocabulary_size,
            max_horizon=self.max_horizon,
            delta_low=self.delta_low,
            delta_high=self.delta_high,
        )


def build_rules(config: PersistenceConfig) -> dict[str, object]:
    """The three hierarchies plus the reference score, all frozen in advance."""

    base = config.benchmark_config()
    deltas, weights = dd.gauss_legendre_delta_grid(
        config.delta_low, config.delta_high, 96
    )
    shared = dd.DirichletBayesGrid(
        delta_grid=deltas, delta_weights=weights,
        alpha_grid=(float("inf"),), tail_size=config.vocabulary_size - 1,
    )
    pooled = dh.PooledDeficitGrid(
        vocabulary_size=config.vocabulary_size,
        delta_low=config.delta_low, delta_high=config.delta_high,
        prior=config.pooled_prior,
    )
    tokenwise = bpe.GumbelBayesLookup(base)
    return {"shared": shared, "pooled": pooled, "tokenwise": tokenwise}


RULES = ("bayes_shared", "bayes_tokenwise", "bayes_pooled", "h_gum_star_0.005")


def score(rule: str, pivots: np.ndarray, built: dict[str, object]) -> np.ndarray:
    if rule == "bayes_shared":
        values, _ = built["shared"].shared_paths(pivots)   # type: ignore[union-attr]
        return values
    if rule == "bayes_pooled":
        return built["pooled"].shared_paths(pivots)        # type: ignore[union-attr]
    if rule == "bayes_tokenwise":
        return np.cumsum(built["tokenwise"](pivots), axis=1)
    if rule == "h_gum_star_0.005":
        return np.cumsum(bpe.paper_gumbel_optimal_score(pivots, 0.005), axis=1)
    raise KeyError(rule)


def run(config: PersistenceConfig) -> dict[str, object]:
    started = time.perf_counter()
    built = build_rules(config)
    seeds = np.random.SeedSequence(config.seed).spawn(2 + len(config.regimes))
    v = config.vocabulary_size

    # One null arm, shared by every regime: the null does not depend on Delta,
    # so calibrating once puts every regime on the same cutoffs.
    # Under the exact pivot null the Gumbel pivot is Uniform(0,1) whatever the
    # NTP is, so the null arms need no generator and no Delta.
    cal_rng = np.random.default_rng(seeds[0])
    cal = cal_rng.uniform(size=(config.n_calibration, config.max_horizon))
    null_rng = np.random.default_rng(seeds[1])
    null = null_rng.uniform(size=(config.n_evaluation_null, config.max_horizon))

    cutoffs: dict[tuple[str, int], float] = {}
    type_i: dict[str, dict[str, float]] = {}
    for rule in RULES:
        cal_paths = score(rule, cal, built)
        null_paths = score(rule, null, built)
        for horizon in config.horizons:
            q = float(np.quantile(
                cal_paths[:, horizon - 1], 1.0 - config.alpha_level, method="higher"
            ))
            cutoffs[(rule, horizon)] = q
            type_i.setdefault(str(horizon), {})[rule] = float(
                np.mean(null_paths[:, horizon - 1] > q)
            )

    type_ii: dict[str, dict[str, dict[str, float]]] = {
        str(h): {rule: {} for rule in RULES} for h in config.horizons
    }
    realized: dict[str, dict[str, float]] = {}
    for index, regime in enumerate(config.regimes):
        rng = np.random.default_rng(seeds[2 + index])
        deltas = regime.deltas(
            rng, config.n_evaluation_alternative, config.max_horizon,
            config.delta_low, config.delta_high,
        )
        pivots = tf.simulate_narrow_width_pivots(rng, deltas, v - 1)
        within = deltas.std(axis=1).mean()
        between = deltas.mean(axis=1).std()
        realized[regime.label] = {
            "mean_delta": float(deltas.mean()),
            "mean_within_document_sd": float(within),
            "between_document_sd_of_the_document_mean": float(between),
            # 1 = every document has its own constant deficit, 0 = no document
            # level signal at all.  This is the axis the sweep varies.
            "persistence": float(between**2 / (between**2 + within**2))
            if (between**2 + within**2) > 0 else float("nan"),
        }
        for rule in RULES:
            paths = score(rule, pivots, built)
            for horizon in config.horizons:
                type_ii[str(horizon)][rule][regime.label] = float(
                    np.mean(paths[:, horizon - 1] <= cutoffs[(rule, horizon)])
                )

    return {
        "description": __doc__.strip().splitlines()[0],
        "seed": config.seed,
        "config": {
            "vocabulary_size": v, "horizons": list(config.horizons),
            "delta_support": [config.delta_low, config.delta_high],
            "alpha_level": config.alpha_level,
            "n_calibration": config.n_calibration,
            "n_evaluation_null": config.n_evaluation_null,
            "n_evaluation_alternative": config.n_evaluation_alternative,
        },
        "pooled_prior": built["pooled"].prior_fingerprint(),  # type: ignore[union-attr]
        "regimes": {
            r.label: {
                "kappa": "inf" if math.isinf(r.kappa) else r.kappa,
                "mu": r.mu, "description": r.description,
                **realized[r.label],
            }
            for r in config.regimes
        },
        "type_i_error": type_i,
        "type2_error": type_ii,
        "runtime_seconds": time.perf_counter() - started,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    config = PersistenceConfig()
    if args.quick:
        config = PersistenceConfig(
            max_horizon=100, horizons=(50, 100), n_calibration=800,
            n_evaluation_null=400, n_evaluation_alternative=400,
        )
    payload = run(config)
    out = args.results_dir / "deficit_persistence_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"runtime: {payload['runtime_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
