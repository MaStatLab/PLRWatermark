"""Does partial pooling of the TAIL WIDTH buy anything?

The deficit analogue is ``deficit_persistence_sweep``.  Section
sec:tail-width-sweep varies the width BETWEEN documents but holds it fixed
within one, so the width block is only ever asked to find a constant, and a
rule that pools the width partially has nothing to detect there.  A deployed
tail is not like that: the number of live coordinates moves with the context.

The generating law uses Z_t = rho*W + sqrt(1-rho**2)*eps_t with independent
standard normal W and eps_t. Quantizing Phi(Z_t) onto the ladder leaves the
marginal width distribution uniform in every regime while changing its
within-document persistence. Delta is shared throughout each document.

  Q1  rho = 1   J constant within a document
  Q5  rho = 0   J independent uniform on the ladder at each token

The fitted hierarchical detector uses an exponential distance kernel around a
latent ladder centre. That working alternative is not the generating copula.

Any gain the pooled rule shows at Q2-Q4 is the value of pooling the width; what
it gives up at the endpoints is what that flexibility costs.
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
import dirichlet_detector as dd
import paired_comparisons as paired_tests
import width_hierarchy as wh

RESULTS = Path(__file__).resolve().parents[1] / "results" / "bayesian_paper_benchmark"
QUICK_RESULTS = RESULTS / "quick" / "width_persistence_sweep"
SWEEP_SEED = 20260903


@dataclass(frozen=True)
class WidthRegime:
    """One generating law, indexed by the latent Gaussian correlation ``rho``.

    A Gaussian copula on the LADDER INDEX, not a bounded exponential kernel.
    ``Z_t = rho W + sqrt(1-rho^2) eps_t`` and the rung is the ``Phi(Z_t)``
    quantile of the uniform distribution over rungs, so every regime has
    exactly the same marginal width distribution and only the dependence
    changes.

    The kernel this replaced, ``Pr(J_t = J_k) ~ exp(-lambda |k - c|)``, is
    truncated at the ends of the ladder, so its marginal shifts with
    ``lambda``: the mean width moved by roughly eight per cent between regimes.
    The sweep therefore varied persistence and the marginal together and could
    not attribute anything to persistence alone -- the same defect the deficit
    sweep had.
    """

    label: str
    rho: float
    description: str

    @property
    def continuous_copula_spearman(self) -> float:
        """Rank correlation before quantization onto the width ladder.

        This is the correlation of the continuous uniforms Phi(Z_t), not
        that of the discrete ladder indices with ties, nor the latent rho^2.
        """

        return (6.0 / math.pi) * math.asin(float(self.rho) ** 2 / 2.0)


DEFAULT_REGIMES: tuple[WidthRegime, ...] = (
    WidthRegime("Q1", 1.0, "rho = 1; the width is constant within a document"),
    WidthRegime("Q2", 0.95, "rho = .95; the width rarely leaves its rung"),
    WidthRegime("Q3", 0.80, "rho = .80; the width wanders one or two rungs"),
    WidthRegime("Q4", 0.60, "rho = .60; the width wanders widely"),
    WidthRegime("Q5", 0.0, "rho = 0; the width is i.i.d. uniform on the ladder "
                           "at every token"),
)


@dataclass(frozen=True)
class WidthSweepConfig:
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
    regimes: tuple[WidthRegime, ...] = DEFAULT_REGIMES


PAIRED_CONTRASTS = (("bayes_pooled_width", "bayes_width"),)
RULES = ("bayes_shared", "bayes_width", "bayes_pooled_width", "h_gum_star_0.1", "h_gum_star_0.01",
         "h_gum_star_0.005",
         "h_ars", "h_log", "h_ind_1_over_e")
# Set per run so the pooled scorer can return just these columns.
HORIZON_REQUEST: tuple[int, ...] = ()


def build(config: WidthSweepConfig) -> dict[str, object]:
    delta, weights = dd.gauss_legendre_delta_grid(
        config.delta_low, config.delta_high, 96
    )
    k = config.vocabulary_size - 1
    return {
        "shared": dd.DirichletBayesGrid(
            delta_grid=delta, delta_weights=weights,
            alpha_grid=(float("inf"),), tail_size=k),
        # The comparator MUST use the same width support as the pooled rule.
        # Left on the default six-rung ladder it silently carried the extra
        # full-width rung the pooled grid excludes, so the experiment compared
        # two supports as well as two hierarchies and the lambda=inf endpoint
        # no longer reproduced it.  This is the union tail's width block, which
        # is what "the frozen ladder" means in the text.
        "width": dd.TailWidthBayesGrid(
            delta_grid=delta, delta_weights=weights, tail_size=k,
            tail_width_grid=tuple(
                j for j in dd.dyadic_tail_width_grid(k) if j != k
            ),
        ),
        "pooled": wh.PooledWidthGrid(
            delta_grid=delta, delta_weights=weights, tail_size=k),
        "ladder": None,
    }


def score(rule: str, pivots: np.ndarray, built: dict) -> np.ndarray:
    if rule == "bayes_shared":
        return built["shared"].shared_paths(pivots)[0]
    if rule == "bayes_width":
        out = built["width"].shared_paths(pivots)
        return out[0] if isinstance(out, tuple) else out
    if rule == "bayes_pooled_width":
        # Ask only for the reported horizons: the pooled width grid streams,
        # and every prefix would be 2400 components by the full pivot count.
        return built["pooled"].shared_paths(pivots, horizons=HORIZON_REQUEST)
    if rule in ("h_ars", "h_log", "h_ind_1_over_e"):
        return np.cumsum(bpe.gumbel_score(pivots, rule, None), axis=1)
    if rule.startswith("h_gum_star_"):
        # The tuning constant is in the name, so the three published values
        # share one branch instead of a hard-coded .005.
        delta = float(rule.rsplit("_", 1)[1])
        return np.cumsum(bpe.paper_gumbel_optimal_score(pivots, delta), axis=1)
    raise KeyError(rule)


def simulate(
    rng: np.random.Generator, regime: WidthRegime, rows: int, horizon: int,
    ladder: tuple[int, ...], config: WidthSweepConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Pivots and the realized per-token widths."""

    rungs = len(ladder)
    r = float(regime.rho)
    w_doc = rng.standard_normal(size=(rows, 1))
    if r >= 1.0:
        z = np.repeat(w_doc, horizon, axis=1)
    else:
        eps = rng.standard_normal(size=(rows, horizon))
        z = r * w_doc + math.sqrt(1.0 - r * r) * eps
    # Uniform over rungs at EVERY rho: the marginal cannot move with rho.
    idx = np.minimum((ndtr(z) * rungs).astype(int), rungs - 1)
    widths = np.asarray(ladder, dtype=float)[idx]
    # One shared deficit per document, as in the width sweep.
    deltas = rng.uniform(config.delta_low, config.delta_high, size=(rows, 1))
    deltas = np.repeat(deltas, horizon, axis=1)
    on_top = rng.uniform(size=(rows, horizon)) >= deltas
    selected = np.where(on_top, 1.0 - deltas, deltas / widths)
    return rng.uniform(size=(rows, horizon)) ** selected, widths


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


def run(config: WidthSweepConfig) -> dict[str, object]:
    global HORIZON_REQUEST
    HORIZON_REQUEST = tuple(config.horizons)
    started = time.perf_counter()
    indicators: dict[str, np.ndarray] = {}
    built = build(config)
    ladder = built["pooled"].tail_widths
    seeds = np.random.SeedSequence(config.seed).spawn(2 + len(config.regimes))
    cal = np.random.default_rng(seeds[0]).uniform(
        size=(config.n_calibration, config.max_horizon))
    null = np.random.default_rng(seeds[1]).uniform(
        size=(config.n_evaluation_null, config.max_horizon))

    cutoffs, type_i = {}, {}
    for rule in RULES:
        cp, npth = score(rule, cal, built), score(rule, null, built)
        for k, h in enumerate(config.horizons):
            column = k if rule == "bayes_pooled_width" else h - 1
            q = float(np.quantile(cp[:, column], 1 - config.alpha_level,
                                  method="higher"))
            cutoffs[(rule, h)] = q
            type_i.setdefault(str(h), {})[rule] = float(np.mean(npth[:, column] > q))

    paired: dict = {}
    type_ii = {str(h): {r: {} for r in RULES} for h in config.horizons}
    realized = {}
    for i, regime in enumerate(config.regimes):
        rng = np.random.default_rng(seeds[2 + i])
        piv, widths = simulate(
            rng, regime, config.n_evaluation_alternative, config.max_horizon,
            ladder, config)
        lw = np.log(widths)
        # Average within-document VARIANCES.  Squaring the mean of the standard
        # deviations is smaller by Jensen and is not a variance component; the
        # between term also carries finite-sequence noise, removed here.
        within = float(lw.var(axis=1, ddof=1).mean())
        between_raw = float(lw.mean(axis=1).var(ddof=1))
        between = max(between_raw - within / lw.shape[1], 0.0)
        realized[regime.label] = {
            "rho": float(regime.rho),
            "continuous_copula_spearman": regime.continuous_copula_spearman,
            "description": regime.description,
            "mean_width": float(widths.mean()),
            "mean_within_document_variance_log_width": within,
            "between_document_variance_raw": between_raw,
            "between_document_variance_corrected": between,
            "width_persistence": float(between / (between + within))
            if (between + within) > 0 else 1.0,
        }
        misses: dict[tuple[str, int], np.ndarray] = {}
        for rule in RULES:
            paths = score(rule, piv, built)
            for k, h in enumerate(config.horizons):
                column = k if rule == "bayes_pooled_width" else h - 1
                missed = paths[:, column] <= cutoffs[(rule, h)]
                misses[(rule, h)] = missed
                # Keep the per-document decision, not just its mean: the JSON
                # records only the discordant counts of the contrasts chosen
                # here, so no other paired statistic could be recomputed.
                indicators[f"{regime.label}|{rule}|{h}"] = missed
                type_ii[str(h)][rule][regime.label] = float(missed.mean())
        # Paired design: both rules score the SAME documents, so a marginal
        # binomial standard error is not the uncertainty of their difference.
        for better, worse in PAIRED_CONTRASTS:
            for h in config.horizons:
                b = int(np.sum(misses[(worse, h)] & ~misses[(better, h)]))
                c = int(np.sum(misses[(better, h)] & ~misses[(worse, h)]))
                paired.setdefault(str(h), {}).setdefault(
                    f"{better}_vs_{worse}", {}
                )[regime.label] = {
                    "b_worse_only_miss": b,
                    "c_better_only_miss": c,
                    "difference": float(
                        type_ii[str(h)][better][regime.label]
                        - type_ii[str(h)][worse][regime.label]
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
        "ladder": [int(j) for j in ladder],
        "pooled_prior": built["pooled"].prior_fingerprint(),
        "config": {
            "vocabulary_size": config.vocabulary_size,
            "horizons": list(config.horizons),
            "delta_support": [config.delta_low, config.delta_high],
            "n_calibration": config.n_calibration,
            "n_evaluation_null": config.n_evaluation_null,
            "n_evaluation_alternative": config.n_evaluation_alternative,
        },
        "regimes": realized,
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
            "exact test; marginal binomial standard errors do not apply."
        ),
        "runtime_seconds": time.perf_counter() - started,
    }


def resolve_results_dir(requested: Path | None, *, quick: bool) -> Path:
    """Keep smoke-run artifacts separate unless a directory is explicit."""

    return requested if requested is not None else QUICK_RESULTS if quick else RESULTS


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--quick", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = WidthSweepConfig()
    if args.quick:
        config = WidthSweepConfig(
            max_horizon=100, horizons=(50, 100), n_calibration=600,
            n_evaluation_null=300, n_evaluation_alternative=300)
    payload = run(config)
    out = resolve_results_dir(args.results_dir, quick=args.quick) / "width_persistence_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Per-document decisions go beside the JSON; see the note stored with them.
    indicators = payload.pop("document_indicators")
    npz = out.with_name("width_persistence_indicators.npz")
    np.savez_compressed(npz, **indicators)
    payload["document_indicators_file"] = npz.name
    payload["document_indicators_note"] = (
        "Boolean miss indicator per evaluation document, keyed "
        "'<regime>|<rule>|<horizon>'.  True means the rule failed to reject."
    )
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} and {npz}\nruntime: {payload['runtime_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
