"""AUC contrasts on the temperature-matched data, with a prompt-cluster bootstrap.

Why this exists.  Each model's four temperature cells use the same model-specific
prompt table, and within a cell the watermarked and unwatermarked arms are
aligned by prompt: row i of both arms continues the same C4 prefix.  The two
models' prompt sets need not be identical.  Two kinds of dependence follow.

  * Between detectors.  Several rules score the same documents, so their AUCs
    are correlated.  DeLong's variance handles exactly this.
  * Between the two class samples.  The positive and negative arms are paired
    by prompt, so they are not independent samples.  DeLong's variance does
    NOT handle this, and neither does any test that treats the arms as
    independent.

The independent-arm DeLong records are retained only in separately labeled
historical artifacts. Current inference uses a joint prompt-cluster bootstrap:
one resample of prompt indices per replicate, applied to BOTH arms and ALL rules
at once.  That preserves the pairing and the between-rule correlation
simultaneously.

An earlier version of this analysis also ran an exact sign test over the eight
cells.  That was invalid because each model's temperature cells share prompts,
so the eight signs are not independent, and it has been removed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

import plot_temperature_prefix_curves as ptc

WINDOW = ("0.2", "0.3", "0.4", "0.5")
# Every rule whose AUC is displayed in the matched-data tables, so each printed
# estimate can carry a clustered standard error rather than appear bare.
RULES = (
    ("bayes_shared",)
    + tuple(f"bayes_uniontail_w{w:g}" for w in
            (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
    # Every reference score and every published tuning, so the AUC tables can
    # show the same menu as the synthetic sweeps rather than a selection.
    + ("h_ars", "h_log", "h_ind_1_over_e")
    + ("h_gum_star_0.1", "h_gum_star_0.01", "h_gum_star_0.005")
    # tab:pooled-real and tab:deficit-support printed their areas bare.  They
    # are here only for the per-rule clustered standard error: none of them
    # enters CONTRASTS, so the reported family size does not change.
    + ("bayes_pooled",)
    + ("bayes_shared_wide", "bayes_uniontail_w0_wide", "bayes_uniontail_w0.5_wide")
)
# (better, worse) -- the contrasts the results section actually makes
CONTRASTS = (
    # The canonical rule the abstract and conclusion talk about.  It was
    # missing, so "significantly improves the shared rule in all eight cells"
    # was resting on the w=0 contrast, a different rule.
    ("bayes_uniontail_w0.5", "bayes_shared"),
    ("bayes_uniontail_w0", "bayes_shared"),
    ("bayes_uniontail_w1", "bayes_shared"),
    ("bayes_uniontail_w0", "bayes_uniontail_w1"),
    ("h_ars", "h_gum_star_0.1"),
    ("h_ars", "bayes_uniontail_w0.5"),
)
# Differences that are REPORTED but not tested: they get a paired bootstrap
# standard error, because a difference of two AUCs on the same documents has no
# marginal standard error, but they stay out of the Holm family so the reported
# multiplicity is unchanged.  The pooled-deficit table's difference row
# (tab:pooled-real) and the wide-versus-narrow prior arms (tab:deficit-support)
# are both of this kind.  Referred to by label because table numbers move.
DESCRIPTIVE_CONTRASTS = (
    ("bayes_pooled", "bayes_shared"),
    ("bayes_shared_wide", "bayes_shared"),
    ("bayes_uniontail_w0_wide", "bayes_uniontail_w0"),
    ("bayes_uniontail_w0.5_wide", "bayes_uniontail_w0.5"),
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "results/bayesian_paper_benchmark/real_model/temperature_matched"
    / "delong_tests_bootstrap.json"
)


def auc_matrix(wm: np.ndarray, null: np.ndarray) -> float:
    """P(score_wm > score_null) with ties at one half, via rank sums.

    rankdata does the tie averaging in C.  The equivalent Python tie loop costs
    5,000 interpreter steps per call, which at 2,000 bootstrap replicates times
    eight cells times six rules is hours rather than minutes.
    """
    n_w, n_0 = wm.shape[0], null.shape[0]
    ranks = rankdata(np.concatenate([wm, null]))
    return float((ranks[:n_w].sum() - n_w * (n_w + 1) / 2.0) / (n_w * n_0))


def cell_scores(watermarked: np.ndarray, null: np.ndarray,
                grids: Mapping[str, object], horizon: int) -> dict[str, tuple]:
    """Per-document score of each rule at one prefix, for both arms."""
    if (watermarked.ndim != 2 or null.shape != watermarked.shape or
            watermarked.shape[0] < 2 or not 1 <= horizon <= watermarked.shape[1] or
            not np.isfinite(watermarked).all() or not np.isfinite(null).all()):
        raise ValueError("paired finite pivot matrices and an available positive horizon are required")
    out: dict[str, tuple] = {}
    for rule in RULES:
        w = ptc.score_paths(rule, watermarked, grids)[:, horizon - 1]
        z = ptc.score_paths(rule, null, grids)[:, horizon - 1]
        out[rule] = (np.asarray(w, dtype=np.float64), np.asarray(z, dtype=np.float64))
    return out


def bootstrap_cell(scores: Mapping[str, tuple], replicates: int,
                   seed: int) -> dict[str, dict[str, float]]:
    """Joint prompt-cluster bootstrap of every contrast in one cell.

    One index draw per replicate is shared by both arms and every rule, so the
    resample respects the prompt pairing and the between-rule correlation.
    """
    if type(replicates) is not int or replicates < 2:
        raise ValueError("bootstrap requires at least two replicates")
    required = {rule for pair in CONTRASTS + DESCRIPTIVE_CONTRASTS for rule in pair}
    if not scores or not required <= scores.keys():
        raise ValueError("scores must include all contrast rules")
    if any(not isinstance(pair, (tuple, list)) or len(pair) != 2 for pair in scores.values()):
        raise ValueError("each rule needs exactly two paired score arrays")
    vectors = [np.asarray(values) for pair in scores.values() for values in pair]
    n = vectors[0].size
    if n < 2 or any(values.shape != (n,) or not np.issubdtype(values.dtype, np.number)
                    or np.issubdtype(values.dtype, np.complexfloating) or not np.isfinite(values).all()
                    for values in vectors):
        raise ValueError("all scores must be finite paired one-dimensional arrays of equal length >=2")
    rng = np.random.default_rng(seed)
    every = CONTRASTS + DESCRIPTIVE_CONTRASTS
    draws = {c: np.empty(replicates) for c in every}
    for b in range(replicates):
        idx = rng.integers(0, n, size=n)
        auc = {r: auc_matrix(w[idx], z[idx]) for r, (w, z) in scores.items()}
        for better, worse in every:
            draws[(better, worse)][b] = auc[better] - auc[worse]
    point = {r: auc_matrix(w, z) for r, (w, z) in scores.items()}
    out: dict[str, dict[str, float]] = {}
    descriptive: dict[str, dict[str, float]] = {}
    for better, worse in every:
        d = point[better] - point[worse]
        boot = draws[(better, worse)]
        se = float(boot.std(ddof=1))
        z_stat = d / se if se > 0 else math.copysign(math.inf, d) if d else 0.0
        target = out if (better, worse) in CONTRASTS else descriptive
        target[f"{better}_vs_{worse}"] = {
            "dauc": d,
            "bootstrap_se": se,
            "z": z_stat,
            # Wald: a normal tail using the BOOTSTRAP standard error, not an
            # empirical bootstrap-tail probability.  The percentile p is also
            # recorded, and is floored at 1/replicates because the bootstrap
            # cannot resolve anything smaller.
            "p": math.erfc(abs(z_stat) / math.sqrt(2.0)),
            "p_type": "wald_normal_on_bootstrap_se",
            # Capped at one: for an identically zero contrast both tails are
            # the whole sample, so twice the smaller is 2 and the record used
            # to report an impossible probability.
            "p_percentile_two_sided": min(
                1.0,
                max(
                    1.0 / replicates,
                    2.0 * min(
                        float(np.mean(boot <= 0.0)),
                        float(np.mean(boot >= 0.0)),
                    ),
                ),
            ),
            "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5)),
        }
    return {
        "auc": point,
        "tests": out,
        # Reported with a standard error, deliberately outside the Holm family.
        "descriptive_differences": descriptive,
        "n": n,
    }


def holm(pairs: Sequence[tuple[str, float]]) -> dict[str, float]:
    """Holm step-down over the whole reported family."""
    ordered = sorted(pairs, key=lambda kv: kv[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    return adjusted


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--models", nargs="+", default=["1p3B", "2p7B"])
    ap.add_argument("--scratch", type=Path, default=ptc.SCRATCH)
    ap.add_argument("--allow-legacy-archives", action="store_true",
                    help="allow only the eight SHA-allowlisted historical NPZ archives")
    # The first 500 prompts of the 2500 are the released ones, and the
    # tail-width family and its mixing weight were argued for partly from the
    # released outputs.  --prompt-start 500 drops exactly those documents, so
    # the remaining 2000 are a holdout with respect to that model-building
    # step.  Rows are in prompt order: the base arms hold prompts 0-499 and the
    # extension arms 500-2499, concatenated in that order.
    ap.add_argument("--prompt-start", type=int, default=0,
                    help="drop the leading documents of every arm before scoring")
    args = ap.parse_args(argv)
    if args.replicates < 2 or args.prompt_start < 0:
        ap.error("replicates must be at least two and prompt-start must be nonnegative")
    if len(set(args.models)) != len(args.models) or any(model not in ptc.MODEL_VOCAB for model in args.models):
        ap.error("models must be distinct supported model names")

    captured: list[tuple] = []
    input_provenance = {}
    original = ptc.curve_rows

    def capture(model, temperature, watermarked, null, grids, **kw):
        captured.append((model, str(temperature), watermarked, null, grids))
        return []

    ptc.curve_rows = capture
    try:
        ptc.collect(args.models, args.scratch, scheme="gumbel",
                    allow_legacy=args.allow_legacy_archives, provenance=input_provenance)
    finally:
        ptc.curve_rows = original

    cells: dict[str, dict] = {}
    for i, (model, temperature, wm, null, grids) in enumerate(captured):
        if temperature not in WINDOW:
            continue
        key = f"{model}_T{temperature}"
        if args.prompt_start:
            if wm.shape[0] <= args.prompt_start:
                raise SystemExit(
                    f"{key}: {wm.shape[0]} documents cannot spare the first "
                    f"{args.prompt_start}"
                )
            # Both arms are dropped at the same rows, which is what keeps the
            # prompt pairing intact: document i of each arm continues prompt i.
            wm, null = wm[args.prompt_start:], null[args.prompt_start:]
        scores = cell_scores(wm, null, grids, ptc.SUMMARY_HORIZON)
        cells[key] = bootstrap_cell(scores, args.replicates, args.seed + i)
        # Per-AUC standard errors, from the same joint resample.  The AUC
        # tables carried bare point estimates against the manuscript's own
        # convention; a binomial SE does not apply to an AUC and DeLong's does
        # not cover the prompt pairing, so the SE has to come from here.
        rng = np.random.default_rng(args.seed + i)
        n_docs = next(iter(scores.values()))[0].shape[0]
        draws = {name: np.empty(args.replicates) for name in scores}
        for b in range(args.replicates):
            idx = rng.integers(0, n_docs, size=n_docs)
            for name, (w, u) in scores.items():
                draws[name][b] = auc_matrix(w[idx], u[idx])
        cells[key]["auc_cluster_se"] = {
            name: float(d.std(ddof=1)) for name, d in draws.items()
        }
        print(f"  {key}: n={cells[key]['n']}", flush=True)

    if not cells:
        raise ValueError("no paired cells in the requested inference window")
    flat = [(f"{cell}|{name}", t["p"])
            for cell, body in cells.items() for name, t in body["tests"].items()]
    adjusted = holm(flat)
    for cell, body in cells.items():
        for name, t in body["tests"].items():
            t["p_holm"] = adjusted[f"{cell}|{name}"]

    payload = {
        "method": (
            "Joint prompt-cluster bootstrap. One resample of prompt indices per "
            "replicate is applied to both arms and every rule, so it respects both "
            "the prompt pairing between the watermarked and unwatermarked arms and "
            "the correlation between rules scored on the same documents. DeLong's "
            "analytic variance handles only the second."
        ),
        "replicates": args.replicates,
        "seed": args.seed,
        "prompt_start": args.prompt_start,
        "input_provenance": input_provenance,
        "prompt_note": (
            "documents 0-499 continue the released prompts and also informed "
            "the tail-width family; prompt_start=500 excludes them"
            if args.prompt_start else
            "all documents, including the 500 released prompts that also "
            "informed the tail-width family"
        ),
        "family_size": len(flat),
        "multiplicity": "Holm step-down over all reported contrasts and cells.",
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
