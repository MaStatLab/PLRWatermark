#!/usr/bin/env python3
"""Paired document-level comparisons for the clean benchmark.

The clean benchmark reports aggregate Type II error rates.  Those rates alone
cannot support a paired claim: two detectors with the same marginal miss rate
can agree on every document or disagree on all of them.  This module consumes
the per-document rejection indicators persisted by
``benchmark_paper_experiment.py`` (``rejection_indicators.npz``) and produces

* Wilson 95% intervals for every method's Type II error,
* exact two-sided McNemar p-values for the Bayes rule against each comparator,
* Holm-adjusted p-values across the families of comparisons reported.

The indicators realise each method's calibrated randomized-boundary rule:
``score > c`` or ``score == c`` and ``U_i < gamma``.  Within a
scenario/scheme/horizon cell the benchmark reuses the same reproducible
auxiliary ``U_i`` for document ``i`` across all methods, preserving the paired
design without injecting avoidable randomization noise.

Everything here is exact integer arithmetic plus ``math``; there is no SciPy
dependency.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import benchmark_paper_experiment as benchmark


DEFAULT_RESULTS_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "bayesian_paper_benchmark"
)

# The scenario determines which Bayes rule the manuscript reports.  The shared
# scenario has one latent Delta per document, which is what bayes_shared
# models; the tokenwise scenario redraws Delta at every token, which is what
# bayes_tokenwise models.
REFERENCE_BY_SCENARIO = {
    "shared_delta_equal_tail_sensitivity": "bayes_shared",
    "paper_text_iid_delta_equal_tail": "bayes_tokenwise",
}

BAYES_METHODS = (
    "bayes_tokenwise",
    "bayes_shared",
    "bayes_tokenwise_dirichlet",
    "bayes_shared_dirichlet",
    "bayes_shared_uniontail",
    "bayes_tokenwise_uniontail",
)

# Authoritative eligibility for both best-reference selection and the Holm
# family.  Tr-GoF and our own Bayesian/diagnostic rules are not paper scores.
# The current manuscript is Gumbel-only; retain the published inverse menu for
# the archived two-pivot analyses rather than classifying it as diagnostic.
PAPER_REFERENCE_BY_SCHEME = {
    "gumbel": frozenset(benchmark.PAPER_REFERENCE_SCORES),
    "inverse": frozenset((
        "h_neg", "h_dif_star_0.1", "h_dif_star_0.01", "h_dif_star_0.001",
    )),
}

# Each Dirichlet-layer rule and the spike-family rule it generalises.  The layer
# contains the spike family at alpha=inf, so these pairs differ only by carrying
# a hyperprior on the tail concentration; comparing them isolates what that
# hyperprior costs or buys on the same documents.
LAYER_PAIRS = {
    "bayes_shared_dirichlet": "bayes_shared",
    "bayes_tokenwise_dirichlet": "bayes_tokenwise",
}


def normal_quantile(probability: float) -> float:
    """Inverse standard normal CDF by bisection on ``math.erf``.

    Avoids a SciPy dependency while staying accurate to full double precision
    for the confidence levels used here.
    """

    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie in (0, 1)")

    def cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    low, high = -40.0, 40.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if cdf(middle) < probability:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def wilson_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""

    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie in [0, trials]")
    z = normal_quantile(0.5 * (1.0 + confidence))
    z2 = z * z
    denominator = trials + z2
    center = (successes + 0.5 * z2) / denominator
    half_width = (
        z
        / denominator
        * math.sqrt(successes * (trials - successes) / trials + 0.25 * z2)
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def binomial_cdf_half(quantile: int, trials: int) -> float:
    """P(X <= quantile) for X ~ Binomial(trials, 1/2), exact integer arithmetic."""

    if trials < 0:
        raise ValueError("trials must be non-negative")
    if quantile < 0:
        return 0.0
    if quantile >= trials:
        return 1.0
    numerator = sum(math.comb(trials, k) for k in range(quantile + 1))
    return numerator / (1 << trials)


def exact_mcnemar_p_value(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value, binomial, no continuity correction.

    ``p = min(1, 2 P(X <= min(b, c)))`` with ``X ~ Binomial(b + c, 1/2)``.

    At ``b=c=0`` the conditional reference distribution is the point mass at
    zero, so the exact tail probability is one.  The caller separately records
    perfect concordance as a status; retaining ``p=1`` also keeps this
    prespecified comparison in its Holm family instead of shrinking the family
    after observing the data.
    """

    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    total = b + c
    if total == 0:
        return 1.0
    return min(1.0, 2.0 * binomial_cdf_half(min(b, c), total))


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, preserving input order.

    The input is the full prespecified family.  Perfect-concordance McNemar
    cells enter as ``p=1`` and therefore remain in the multiplicity count.
    """

    values = [float(value) for value in p_values]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("p-values must be finite and lie in [0, 1]")
    adjusted = [0.0] * len(values)
    count = len(values)
    if count == 0:
        return adjusted
    order = sorted(range(count), key=values.__getitem__)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (count - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def load_indicators(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as handle:
        return {key: np.asarray(handle[key], dtype=bool) for key in handle.files}


def load_indicator_metadata(path: Path) -> dict[str, object]:
    """Load and validate the sidecar metadata for persisted indicators.

    Legacy benchmark artifacts stored strict-boundary decisions.  Refusing
    those artifacts prevents a newly generated paired-comparison JSON from
    silently claiming that old indicators realise the randomized test.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("per_document_indicators")
    if not isinstance(metadata, dict):
        raise ValueError("benchmark summary has no per_document_indicators metadata")
    version = metadata.get("indicator_rule_version")
    if version != benchmark.BOUNDARY_RANDOMIZATION_RULE_VERSION:
        raise ValueError(
            "rejection_indicators.npz predates reproducible boundary randomization; "
            "rerun benchmark_paper_experiment.py before paired_comparisons.py"
        )
    return metadata


def _parse_key(key: str) -> tuple[str, str, int, str]:
    scenario, scheme, horizon, method = key.split("|")
    return scenario, scheme, int(horizon), method


def build_comparisons(
    indicators: dict[str, np.ndarray],
    confidence: float = 0.95,
    *,
    indicator_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Assemble Wilson intervals, McNemar tests and Holm adjustments."""

    cells: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
    for key, value in indicators.items():
        scenario, scheme, horizon, method = _parse_key(key)
        cells.setdefault((scenario, scheme, horizon), {})[method] = value

    type2_records: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []

    for (scenario, scheme, horizon), methods in sorted(cells.items()):
        if scheme not in PAPER_REFERENCE_BY_SCHEME:
            raise ValueError(f"unknown pivot scheme: {scheme}")
        paper_scores = PAPER_REFERENCE_BY_SCHEME[scheme]
        n_documents = int(next(iter(methods.values())).size)
        misses = {
            method: np.logical_not(rejected) for method, rejected in methods.items()
        }
        for method in sorted(methods):
            miss_count = int(np.count_nonzero(misses[method]))
            low, high = wilson_interval(miss_count, n_documents, confidence)
            type2_records.append(
                {
                    "scenario": scenario,
                    "scheme": scheme,
                    "horizon": horizon,
                    "method": method,
                    "n_documents": n_documents,
                    "n_missed": miss_count,
                    "type2_error": miss_count / n_documents,
                    "wilson95_low": low,
                    "wilson95_high": high,
                }
            )

        reference = REFERENCE_BY_SCENARIO[scenario]
        if reference not in methods:
            continue
        reference_miss = misses[reference]
        reference_count = int(np.count_nonzero(reference_miss))
        reference_low, reference_high = wilson_interval(
            reference_count, n_documents, confidence
        )

        comparators = [
            method
            for method in sorted(methods)
            if method != reference and method not in BAYES_METHODS
        ]
        paper_comparators = [
            method
            for method in comparators
            if method in paper_scores
        ]
        best_paper = min(
            paper_comparators,
            key=lambda method: (int(np.count_nonzero(misses[method])), method),
            default=None,
        )

        for comparator in comparators:
            comparator_miss = misses[comparator]
            comparator_count = int(np.count_nonzero(comparator_miss))
            comparator_low, comparator_high = wilson_interval(
                comparator_count, n_documents, confidence
            )
            b = int(np.count_nonzero(reference_miss & ~comparator_miss))
            c = int(np.count_nonzero(comparator_miss & ~reference_miss))
            # A reference score is one of the cited paper's, not merely
            # anything that is not a diagnostic: our own Bayes rules must
            # not enter the family we correct them against.
            is_paper = comparator in paper_scores
            comparisons.append(
                {
                    "scenario": scenario,
                    "scheme": scheme,
                    "horizon": horizon,
                    "reference": reference,
                    "comparator": comparator,
                    "comparator_is_paper_score": is_paper,
                    "comparator_is_best_paper_score": comparator == best_paper,
                    "n_documents": n_documents,
                    "b_reference_only_miss": b,
                    "c_comparator_only_miss": c,
                    "discordant_total": b + c,
                    "p_value": exact_mcnemar_p_value(b, c),
                    "p_value_status": (
                        "no_discordant_pairs_p_equals_one"
                        if b + c == 0
                        else "defined"
                    ),
                    "reference_n_missed": reference_count,
                    "reference_type2_error": reference_count / n_documents,
                    "reference_wilson95": [reference_low, reference_high],
                    "comparator_n_missed": comparator_count,
                    "comparator_type2_error": comparator_count / n_documents,
                    "comparator_wilson95": [comparator_low, comparator_high],
                    # Key by scheme: the current Gumbel paper family has six
                    # scores at three horizons, while the legacy inverse
                    # family has four scores at those horizons.  Tr-GoF is a
                    # separate baseline, not one of the reference scores the
                    # manuscript corrects across, so it gets its own family.
                    "holm_family": (
                        f"{scenario}|{scheme}|"
                        + (
                            "trgof"
                            if comparator == benchmark.TRGOF_METHOD
                            else ("paper_scores" if is_paper else "diagnostic_scores")
                        )
                    ),
                }
            )

        # The layer against the family it generalises, on the same documents.
        for layer_rule, spike_rule in sorted(LAYER_PAIRS.items()):
            if layer_rule not in methods or spike_rule not in methods:
                continue
            layer_miss = misses[layer_rule]
            spike_miss = misses[spike_rule]
            layer_count = int(np.count_nonzero(layer_miss))
            spike_count = int(np.count_nonzero(spike_miss))
            layer_low, layer_high = wilson_interval(
                layer_count, n_documents, confidence
            )
            spike_low, spike_high = wilson_interval(
                spike_count, n_documents, confidence
            )
            b = int(np.count_nonzero(layer_miss & ~spike_miss))
            c = int(np.count_nonzero(spike_miss & ~layer_miss))
            comparisons.append(
                {
                    "scenario": scenario,
                    "scheme": scheme,
                    "horizon": horizon,
                    "reference": layer_rule,
                    "comparator": spike_rule,
                    "comparator_is_paper_score": False,
                    "comparator_is_best_paper_score": False,
                    "n_documents": n_documents,
                    "b_reference_only_miss": b,
                    "c_comparator_only_miss": c,
                    "discordant_total": b + c,
                    "p_value": exact_mcnemar_p_value(b, c),
                    "p_value_status": (
                        "no_discordant_pairs_p_equals_one"
                        if b + c == 0
                        else "defined"
                    ),
                    "reference_n_missed": layer_count,
                    "reference_type2_error": layer_count / n_documents,
                    "reference_wilson95": [layer_low, layer_high],
                    "comparator_n_missed": spike_count,
                    "comparator_type2_error": spike_count / n_documents,
                    "comparator_wilson95": [spike_low, spike_high],
                    "holm_family": f"{scenario}|dirichlet_layer",
                }
            )

    # Holm within each reported family.
    families: dict[str, list[int]] = {}
    for index, record in enumerate(comparisons):
        families.setdefault(str(record["holm_family"]), []).append(index)
    for family, members in families.items():
        raw = [comparisons[i]["p_value"] for i in members]
        adjusted = holm_adjust(raw)  # type: ignore[arg-type]
        zero_discordance = sum(
            int(comparisons[index]["discordant_total"]) == 0 for index in members
        )
        for index, value in zip(members, adjusted):
            comparisons[index]["holm_adjusted_p"] = value
            comparisons[index]["holm_family_size"] = len(members)
            comparisons[index]["holm_family_zero_discordance_retained"] = (
                zero_discordance
            )

    # A narrower family than the one the manuscript reports: shared scenario,
    # both schemes, three horizons, Bayes against only the single best reference
    # score in each cell.  The manuscript instead corrects over all 18 Gumbel
    # Bayes-versus-reference-score comparisons, whose adjusted p-values are in the
    # ``comparisons`` block above; this six-member sensitivity family has a
    # smaller multiplicity penalty and its numbers are deliberately different.
    manuscript_members = [
        record
        for record in comparisons
        if record["scenario"] == "shared_delta_equal_tail_sensitivity"
        and record["comparator_is_best_paper_score"]
    ]
    manuscript_adjusted = holm_adjust(
        [record["p_value"] for record in manuscript_members]  # type: ignore[misc]
    )
    manuscript_family = [
        {
            "scheme": record["scheme"],
            "horizon": record["horizon"],
            "reference": record["reference"],
            "comparator": record["comparator"],
            "b_reference_only_miss": record["b_reference_only_miss"],
            "c_comparator_only_miss": record["c_comparator_only_miss"],
            "p_value": record["p_value"],
            "p_value_status": record["p_value_status"],
            "holm_adjusted_p": value,
            "significant_at_0.05_after_holm": value < 0.05,
        }
        for record, value in zip(manuscript_members, manuscript_adjusted)
    ]

    return {
        "confidence_level": confidence,
        "rejection_rule": (
            "score > c or (score == c and U_i < gamma), using the benchmark's "
            "reproducible paired auxiliary uniforms"
        ),
        "indicator_metadata": indicator_metadata,
        "mcnemar": (
            "exact two-sided binomial, no continuity correction: "
            "p = min(1, 2 P(X <= min(b, c))), X ~ Binomial(b + c, 1/2). "
            "When b + c = 0 the conditional reference distribution is a point "
            "mass, so p=1. Such cells carry p_value_status = "
            "'no_discordant_pairs_p_equals_one' and remain in the full "
            "prespecified Holm family while also being reported as perfect "
            "concordance."
        ),
        "b_definition": "documents missed by the Bayes reference but not by the comparator",
        "c_definition": "documents missed by the comparator but not by the Bayes reference",
        "reference_by_scenario": dict(REFERENCE_BY_SCENARIO),
        "paper_reference_scores_by_scheme": {
            scheme: sorted(methods)
            for scheme, methods in PAPER_REFERENCE_BY_SCHEME.items()
        },
        "diagnostic_scores_excluded_from_best_paper_score": list(
            benchmark.DIAGNOSTIC_METHODS
        ),
        "type2_wilson_intervals": type2_records,
        "comparisons": comparisons,
        "best_comparator_only_holm_family": {
            "description": (
                "Six shared-scenario scheme-by-horizon comparisons of the shared "
                "Bayes rule against the best reference score in each cell.  This is not the "
                "family the manuscript reports: the manuscript corrects over all "
                "18 Bayes-versus-reference-score comparisons, whose adjusted p-values "
                "are the holm_adjusted_p entries of the comparisons block.  This "
                "narrower family is a lower-multiplicity sensitivity analysis, "
                "so its numbers differ by design."
            ),
            "size": len(manuscript_family),
            "members": manuscript_family,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir: Path = args.results_dir
    indicators = load_indicators(results_dir / "rejection_indicators.npz")
    indicator_metadata = load_indicator_metadata(
        results_dir / "benchmark_summary.json"
    )
    payload = build_comparisons(
        indicators, indicator_metadata=indicator_metadata
    )
    output = results_dir / "paired_comparisons.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "n_cells": len(payload["type2_wilson_intervals"]),
                "n_comparisons": len(payload["comparisons"]),
                "best_comparator_only_family_size": (
                    payload["best_comparator_only_holm_family"]["size"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
