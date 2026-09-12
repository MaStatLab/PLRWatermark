"""Does partial pooling of the deficit buy anything?

Every deficit design in the study is all-or-nothing.  The shared experiment
holds ``Delta`` fixed for a whole document; the tokenwise experiment redraws it
independently at every position.  A rule that pools partially therefore has
nothing to detect in either, and can only pay the dilution cost of covering
both.  This is the gap the tail-width sweep closed for the tail.

The generator is a GAUSSIAN COPULA, so persistence moves on its own:

    W ~ N(0,1) per document,   Z_t = rho W + sqrt(1-rho^2) eps_t,
    X_t = Phi(Z_t),            Delta_t = low + (high-low) X_t.

``X_t`` is exactly Uniform(0,1) for every ``rho``, so the marginal law of the
deficit is identical across the sweep and the intraclass correlation is
``(6/pi) arcsin(rho^2/2)`` by construction.  P1 (``rho=1``) is the shared design and P6
(``rho=0``) the tokenwise one, both reproduced exactly.

An earlier version drew ``X_t | mu, kappa ~ Beta(kappa mu, kappa(1-mu))`` with
``mu ~ U(0,1)``.  That has ``Var(X_t) = 1/12 + 1/(6(kappa+1))``: lowering kappa
made the marginal more dispersed and more U-shaped while it lowered
persistence, so the sweep moved two things at once and no gain could be
attributed to persistence alone.  Its reported "persistence" was also not a
variance share -- it squared the mean within-document standard deviation
instead of averaging within-document variances, and left sampling noise in the
between term.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.special import ndtr

import benchmark_paper_experiment as bpe
import deficit_hierarchy as dh
import dirichlet_detector as dd
import paired_comparisons as paired_tests
import tail_family as tf

RESULTS = Path(__file__).resolve().parents[1] / "results" / "bayesian_paper_benchmark"
QUICK_RESULTS = RESULTS / "quick" / "deficit_persistence_sweep"
SWEEP_SEED = 20260902


@dataclass(frozen=True)
class PersistenceRegime:
    """One generating law, indexed by the within-document correlation ``rho``.

    A Gaussian copula, NOT a Beta hierarchy.  With
    ``Z_t = rho * W + sqrt(1 - rho^2) * eps_t`` and ``X_t = Phi(Z_t)``, every
    regime has X_t EXACTLY Uniform(0,1) marginally, so the deficit's marginal
    law is identical across the sweep and only the dependence changes.  The
    intraclass correlation is ``(6/pi) arcsin(rho^2/2)`` by construction.

    The first version used ``X_t | mu, kappa ~ Beta(kappa mu, kappa(1-mu))``
    with ``mu ~ U(0,1)``.  That has ``Var(X_t) = 1/12 + 1/(6(kappa+1))``, so
    lowering kappa made the marginal more dispersed and more U-shaped at the
    same time as it lowered persistence: the sweep moved two things at once and
    could not attribute a gain to persistence alone.
    """

    label: str
    rho: float
    description: str

    @property
    def icc(self) -> float:
        """Population intraclass correlation of ``Delta_t``.

        NOT ``rho^2``.  ``rho^2`` is the correlation of the latent Gaussians
        ``Z_s, Z_t``; the probability-integral transform ``X_t = Phi(Z_t)``
        maps it to the Spearman correlation, and because ``X`` is uniform that
        is also its Pearson correlation:

            ICC = (6 / pi) * arcsin(rho^2 / 2).

        The two agree only at 0 and 1, and differ by up to .015 in between,
        which is larger than several of the effects this sweep reports.
        """

        return (6.0 / math.pi) * math.asin(float(self.rho) ** 2 / 2.0)

    def deltas(
        self, rng: np.random.Generator, rows: int, horizon: int,
        low: float, high: float,
    ) -> np.ndarray:
        r = float(self.rho)
        if not 0.0 <= r <= 1.0:
            raise ValueError("rho must lie in [0, 1]")
        w = rng.standard_normal(size=(rows, 1))
        if r >= 1.0:
            z = np.repeat(w, horizon, axis=1)
        else:
            eps = rng.standard_normal(size=(rows, horizon))
            z = r * w + math.sqrt(1.0 - r * r) * eps
        x = ndtr(z)          # Phi, vectorized; math.erf is scalar-only
        return low + (high - low) * x


DEFAULT_REGIMES: tuple[PersistenceRegime, ...] = (
    PersistenceRegime("P1", 1.0,
                      "rho = 1; the deficit is constant within a document -- "
                      "exactly the shared-deficit design"),
    PersistenceRegime("P2", 0.95, "rho = .95; ICC .894, strongly persistent"),
    PersistenceRegime("P3", 0.80, "rho = .80; ICC .622, moderately persistent"),
    PersistenceRegime("P4", 0.60, "rho = .60; ICC .346, weakly persistent"),
    PersistenceRegime("P5", 0.30, "rho = .30; ICC .086, barely persistent"),
    PersistenceRegime("P6", 0.0,
                      "rho = 0; Delta_t i.i.d. uniform with no document-level "
                      "component -- exactly the tokenwise design"),
)


@dataclass(frozen=True)
class PersistenceConfig:
    vocabulary_size: int = 1_000
    max_horizon: int = 60
    # Short horizons on purpose.  With Delta_t marginally Unif(.001,.5) the
    # task saturates fast: at n=100 every regime below ICC .64 already has
    # zero observed misses, and a table of zeros cannot separate the rules.
    horizons: tuple[int, ...] = (15, 30, 60)
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
            prior_low=self.delta_low,
            prior_high=self.delta_high,
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


PAIRED_CONTRASTS = (("bayes_pooled", "bayes_shared"),
                    ("bayes_pooled", "bayes_tokenwise"))
RULES = ("bayes_shared", "bayes_tokenwise", "bayes_pooled", "h_gum_star_0.1", "h_gum_star_0.01",
         "h_gum_star_0.005",
         "h_ars", "h_log", "h_ind_1_over_e")


def score(rule: str, pivots: np.ndarray, built: dict[str, object]) -> np.ndarray:
    if rule == "bayes_shared":
        values, _ = built["shared"].shared_paths(pivots)   # type: ignore[union-attr]
        return values
    if rule == "bayes_pooled":
        return built["pooled"].shared_paths(pivots)        # type: ignore[union-attr]
    if rule == "bayes_tokenwise":
        return np.cumsum(built["tokenwise"](pivots), axis=1)
    if rule in ("h_ars", "h_log", "h_ind_1_over_e"):
        return np.cumsum(bpe.gumbel_score(pivots, rule, None), axis=1)
    if rule.startswith("h_gum_star_"):
        # The tuning constant is in the name, so the three published values
        # share one branch instead of a hard-coded .005.
        delta = float(rule.rsplit("_", 1)[1])
        return np.cumsum(bpe.paper_gumbel_optimal_score(pivots, delta), axis=1)
    raise KeyError(rule)


def holm_adjust(pvalues: "list[float]") -> "list[float]":
    """Holm step-down, returning adjusted values in the input order.

    The family is every cell of a contrast in this sweep, declared here rather
    than chosen after seeing which cells were small.
    """

    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar, the convention used elsewhere in the study.

    ``b`` and ``c`` are discordant counts.  With ``b + c == 0`` the conditional
    reference distribution is a point mass and no nontrivial test exists, so
    this returns 1 rather than pretending to evidence.
    """

    return paired_tests.exact_mcnemar_p_value(b, c)


def run(config: PersistenceConfig) -> dict[str, object]:
    started = time.perf_counter()
    indicators: dict[str, np.ndarray] = {}
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

    paired: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
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
        # Average the within-document VARIANCES; squaring the mean of the
        # standard deviations is a different and smaller quantity by Jensen.
        within_var = deltas.var(axis=1, ddof=1).mean()
        between_var = deltas.mean(axis=1).var(ddof=1)
        # The between term above still contains within-document sampling noise;
        # subtract it so the realized figure estimates the same population
        # quantity the copula sets exactly.
        between_var_corrected = max(
            between_var - within_var / deltas.shape[1], 0.0
        )
        realized[regime.label] = {
            "mean_delta": float(deltas.mean()),
            "sd_delta": float(deltas.std(ddof=1)),
            "mean_within_document_variance": float(within_var),
            "between_document_variance_raw": float(between_var),
            "between_document_variance_corrected": float(between_var_corrected),
            "population_icc": regime.icc,
            "realized_icc": float(
                between_var_corrected / (between_var_corrected + within_var)
            ) if (between_var_corrected + within_var) > 0 else 1.0,
        }
        misses: dict[tuple[str, int], np.ndarray] = {}
        for rule in RULES:
            paths = score(rule, pivots, built)
            for horizon in config.horizons:
                missed = paths[:, horizon - 1] <= cutoffs[(rule, horizon)]
                misses[(rule, horizon)] = missed
                # Keep the per-document decision, not just its mean.  Only the
                # discordant counts of the contrasts chosen here survive into
                # the JSON, so without this no other paired statistic can be
                # recomputed from the artifact.
                indicators[f"{regime.label}|{rule}|{horizon}"] = missed
                type_ii[str(horizon)][rule][regime.label] = float(missed.mean())
        # Both rules score the SAME documents, so the comparison is paired and a
        # marginal binomial standard error is the wrong uncertainty for their
        # difference.  Record the discordant counts and an exact McNemar test.
        for better, worse in PAIRED_CONTRASTS:
            for horizon in config.horizons:
                b = int(np.sum(misses[(worse, horizon)] & ~misses[(better, horizon)]))
                c = int(np.sum(misses[(better, horizon)] & ~misses[(worse, horizon)]))
                paired.setdefault(str(horizon), {}).setdefault(
                    f"{better}_vs_{worse}", {}
                )[regime.label] = {
                    "b_worse_only_miss": b,
                    "c_better_only_miss": c,
                    "difference": float(
                        type_ii[str(horizon)][better][regime.label]
                        - type_ii[str(horizon)][worse][regime.label]
                    ),
                    "mcnemar_p": mcnemar_exact(b, c),
                }

    # ONE Holm family over every (contrast, horizon, regime) cell, not one per
    # contrast.  The caption claims a comprehensive family and a reader scans
    # both contrasts for significance, so the family is their union; adjusting
    # each contrast separately would understate the multiplicity.  The family
    # is comprehensive but not prospective: the Type II cells were computed and
    # written before any paired test existed here.
    cells = [(h, name, g)
             for h in paired for name in paired[h] for g in paired[h][name]]
    raw = [paired[h][name][g]["mcnemar_p"] for h, name, g in cells]
    for (h, name, g), value in zip(cells, holm_adjust(raw)):
        paired[h][name][g]["mcnemar_p_holm"] = value

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
                "rho": float(r.rho), "description": r.description,
                **realized[r.label],
            }
            for r in config.regimes
        },
        "type_i_error": type_i,
        "type2_error": type_ii,
        "paired": paired,
        "paired_holm_family_size": sum(
            len(body) for h in paired for body in paired[h].values()
        ),
        "document_indicators": indicators,
        "paired_holm_family": (
            "One Holm step-down over the union of every (contrast, horizon, "
            "regime) cell reported here, not one family per contrast.  Post "
            "hoc: the Type II cells were computed and written before any "
            "paired test was added, so the family was not declared in "
            "advance.  It is comprehensive rather than prospective -- every "
            "cell of every contrast enters, none was chosen after seeing the "
            "estimates -- which controls the familywise rate over the "
            "reported set but does not make the analysis confirmatory."
        ),
        "paired_note": (
            "Both rules score the same documents, so differences are paired. "
            "b and c are discordant counts and mcnemar_p is the two-sided "
            "exact test; marginal binomial standard errors do not apply to "
            "these differences."
        ),
        "runtime_seconds": time.perf_counter() - started,
    }


def resolve_results_dir(requested: Path | None, *, quick: bool) -> Path:
    """Keep smoke-run artifacts separate unless a directory is explicit."""

    return requested if requested is not None else QUICK_RESULTS if quick else RESULTS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = PersistenceConfig()
    if args.quick:
        config = PersistenceConfig(
            max_horizon=100, horizons=(50, 100), n_calibration=800,
            n_evaluation_null=400, n_evaluation_alternative=400,
        )
    payload = run(config)
    out = resolve_results_dir(args.results_dir, quick=args.quick) / "deficit_persistence_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Per-document decisions go beside the JSON.  The JSON keeps only the
    # discordant counts of the contrasts chosen here, so a reader who wants a
    # different paired statistic -- or who wants to check these -- needs the
    # indicators themselves.
    indicators = payload.pop("document_indicators")
    npz = out.with_name("deficit_persistence_indicators.npz")
    np.savez_compressed(npz, **indicators)
    payload["document_indicators_file"] = npz.name
    payload["document_indicators_note"] = (
        "Boolean miss indicator per evaluation document, keyed "
        "'<regime>|<rule>|<horizon>'.  True means the rule failed to reject."
    )
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} and {npz}")
    print(f"runtime: {payload['runtime_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
