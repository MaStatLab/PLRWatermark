#!/usr/bin/env python3
"""Generate full-prefix released-output curves under two distinct analyses.

This command deliberately keeps two constructions separate:

* ``common_calibration`` recomputes scores from the six SHA256-allowlisted
  released pickle files and calibrates every horizon on fixed-seeded,
  independently simulated exact-pivot null paths through
  :mod:`real_data_experiment`.
* ``upstream_recorded`` reads the eight commit-pinned ``*-null.json`` and
  ``*-result.json`` files verbatim.  Those arrays are stored rejection rates,
  not values digitized from a figure.  In particular, the upstream inverse
  optimal-score cutoffs were produced by unseeded simulations, were regenerated
  separately for the null and watermarked calls, and use a pointwise minimum
  over 13 cutoff curves for the 2.7B model.

The output CSV contains every upstream curve and every common-calibration
curve for horizons 1 through 200.  The common-calibration figure shows the
complete reference-score and Bayesian menus used in the released-output
analysis.  A separate upstream figure shows the overlapping reference menu;
the remaining upstream methods stay available in the CSV.

No model weights, tokenizer, C4 data, or GPU are used.  PyTorch is needed only
to deserialize the verified pickles and replay the release's CPU PRF, as
documented by :mod:`real_data_experiment`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import real_data_experiment as real


MAX_HORIZON = 200
N_RELEASED_DOCUMENTS = 500

DEFAULT_RESULTS_DIR = real.DEFAULT_RESULTS_DIR / "prefix_curves"

UPSTREAM_JSON_SHA256: dict[str, str] = {
    "1p3B-gumbel-c4-m200-T500-skipgram_prf-15485863-temp0.1-null.json":
        "69a30754b789b09e9eea62de371d4bf3122ef09050feaa43e4d7efdedccc9ee6",
    "1p3B-gumbel-c4-m200-T500-skipgram_prf-15485863-temp0.1-result.json":
        "ac8fe621e94a749e4a1846363e9e87577e33fe7ce9ad6696cd56f94b6dac78c4",
    "1p3B-transform-c4-m200-T500-skipgram_prf-15485863-temp0.1-null.json":
        "dcdd9b27329bab2f8744ce5329c37f95843f98ea3b30cf8a1c239124aafac570",
    "1p3B-transform-c4-m200-T500-skipgram_prf-15485863-temp0.1-result.json":
        "60a476d27f413b438d5bcbe4a5369f16de7876ea402ee8949c38a4bc007f97a3",
    "2p7B-gumbel-c4-m200-T500-skipgram_prf-15485863-temp0.1-null.json":
        "333854585eee5b35a5aa3e243471defc4d3473d18520d1921517d965638184a6",
    "2p7B-gumbel-c4-m200-T500-skipgram_prf-15485863-temp0.1-result.json":
        "fc84392a7b6072c2e07a4ceee54c9d3fabf63e1ef939827a167d143c8d2dd4fa",
    "2p7B-transform-c4-m200-T500-skipgram_prf-15485863-temp0.1-null.json":
        "5f8263e37954fde3d55e4667b13e349e0e212f0636669d9c9c32df1e81f22da2",
    "2p7B-transform-c4-m200-T500-skipgram_prf-15485863-temp0.1-result.json":
        "b3cc78c7fd0db667bb190c2b955eb56092b864df6e10f83c0a4e3f769236a5f1",
}


# The names on the left are the keys in the released JSON files.  The names on
# the right agree with real_data_experiment wherever the score is recomputed
# there; additional upstream scores receive unambiguous canonical names.
UPSTREAM_METHODS: dict[str, dict[str, tuple[str, str]]] = {
    "gumbel": {
        "ars": ("h_ars", "h_ars"),
        "log": ("h_log", "h_log"),
        "ind-08": ("h_ind_0.8", "h_ind,.8"),
        "ind-09": ("h_ind_0.9", "h_ind,.9"),
        "ind-1/e": ("h_ind_1_over_e", "h_ind,1/e"),
        "opt-02": ("h_gum_star_0.2", "h*_gum,.2"),
        "opt-015": ("h_gum_star_0.15", "h*_gum,.15"),
        "opt-01": ("h_gum_star_0.1", "h*_gum,.1"),
        "opt-005": ("h_gum_star_0.05", "h*_gum,.05"),
        "opt-001": ("h_gum_star_0.01", "h*_gum,.01"),
        "opt-0005": ("h_gum_star_0.005", "h*_gum,.005"),
        "opt-0001": ("h_gum_star_0.001", "h*_gum,.001"),
    },
    "inverse": {
        "dif": ("h_neg", "h_neg"),
        "dif-ind-01": ("h_dif_ind_0.1", "h_ind,dif,.1"),
        "dif-ind-02": ("h_dif_ind_0.2", "h_ind,dif,.2"),
        "dif-ind-05": ("h_dif_ind_0.5", "h_ind,dif,.5"),
        "dif-opt-02": ("h_dif_star_0.2", "h*_dif,.2"),
        "dif-opt-01": ("h_dif_star_0.1", "h*_dif,.1"),
        "dif-opt-005": ("h_dif_star_0.05", "h*_dif,.05"),
        "dif-opt-001": ("h_dif_star_0.01", "h*_dif,.01"),
        "dif-opt-0005": ("h_dif_star_0.005", "h*_dif,.005"),
        "dif-opt-0001": ("h_dif_star_0.001", "h*_dif,.001"),
    },
}


PLOTTED_REFERENCE_METHODS: dict[str, tuple[str, ...]] = {
    "gumbel": (
        "h_ars",
        "h_log",
        "h_ind_1_over_e",
        "h_gum_star_0.1",
        "h_gum_star_0.01",
        "h_gum_star_0.005",
    ),
    "inverse": (
        "h_neg",
        "h_dif_star_0.1",
        "h_dif_star_0.01",
        "h_dif_star_0.001",
    ),
}


CURVE_FIELDS = (
    "analysis",
    "model",
    "model_id",
    "scheme",
    "pivot_convention",
    "sample",
    "error_type",
    "method",
    "method_label",
    "method_origin",
    "horizon",
    "n_documents",
    "rejection_rate",
    "error_rate",
    "mcse",
    "cutoff",
    "boundary_probability",
    "source_file",
    "source_sha256",
    "calibration",
    "bayesian_delta_prior",
    "bayesian_delta_prior_low",
    "bayesian_delta_prior_high",
)


def upstream_json_names(model_name: str) -> dict[tuple[str, str], str]:
    """Return the four pinned curve-array names for one released model."""

    if model_name not in real.MODEL_SPECS:
        raise KeyError(model_name)
    stem = "c4-m200-T500-skipgram_prf-15485863-temp0.1"
    return {
        ("gumbel", "null"): f"{model_name}-gumbel-{stem}-null.json",
        ("gumbel", "result"): f"{model_name}-gumbel-{stem}-result.json",
        ("inverse", "null"): f"{model_name}-transform-{stem}-null.json",
        ("inverse", "result"): f"{model_name}-transform-{stem}-result.json",
    }


def verify_upstream_json(path: Path) -> dict[str, Any]:
    """Verify a released JSON array file against the fixed digest allowlist."""

    expected = UPSTREAM_JSON_SHA256.get(path.name)
    if expected is None:
        raise ValueError(f"{path.name!r} is not an allowlisted upstream JSON file")
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = real.file_sha256(path)
    if observed != expected:
        raise ValueError(
            f"SHA256 mismatch for {path}: expected {expected}, observed {observed}"
        )
    return {
        "filename": path.name,
        "sha256": observed,
        "size_bytes": path.stat().st_size,
        "upstream_url": f"{real.OFFICIAL_RAW_BASE}/{path.name}",
    }


def ensure_upstream_json(
    directory: Path,
    *,
    models: Sequence[str],
    download_missing: bool,
) -> list[dict[str, Any]]:
    """Fetch missing pinned JSON files and verify all bytes before use."""

    directory.mkdir(parents=True, exist_ok=True)
    requested = [
        filename
        for model in models
        for filename in upstream_json_names(model).values()
    ]
    records: list[dict[str, Any]] = []
    for filename in requested:
        path = directory / filename
        if not path.exists():
            if not download_missing:
                raise FileNotFoundError(
                    f"missing {path}; omit --no-download to fetch the pinned JSON"
                )
            temporary = path.with_name(f".{path.name}.part")
            try:
                with urllib.request.urlopen(
                    f"{real.OFFICIAL_RAW_BASE}/{filename}", timeout=120
                ) as response, temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
                observed = real.file_sha256(temporary)
                expected = UPSTREAM_JSON_SHA256[filename]
                if observed != expected:
                    raise ValueError(
                        f"downloaded SHA256 mismatch for {filename}: "
                        f"expected {expected}, observed {observed}"
                    )
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
            os.replace(temporary, path)
        records.append(verify_upstream_json(path))
    return records


def _binary_mcse(rate: float, n: int = N_RELEASED_DOCUMENTS) -> float:
    """Sample-standard-error convention for an observed binary rate."""

    if n < 2:
        return 0.0
    return math.sqrt(max(rate * (1.0 - rate), 0.0) / (n - 1))


def _validate_rate_curve(
    value: Any, *, filename: str, method: str
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (MAX_HORIZON,):
        raise ValueError(
            f"{filename}:{method} has shape {array.shape}, expected ({MAX_HORIZON},)"
        )
    if np.any(~np.isfinite(array)) or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{filename}:{method} contains a rate outside [0,1]")
    counts = array * N_RELEASED_DOCUMENTS
    if not np.allclose(counts, np.rint(counts), atol=1e-8, rtol=0.0):
        raise ValueError(
            f"{filename}:{method} is not a sequence of rates from 500 documents"
        )
    return array


def _upstream_calibration(model: str, scheme: str, upstream_method: str) -> str:
    if scheme == "gumbel":
        if upstream_method in {"ars", "log"}:
            return "upstream_analytic_gamma_quantile"
        if upstream_method.startswith("ind-"):
            return "upstream_binomial_quantile_with_greater_equal_rule"
        return "upstream_normal_approximation"
    if upstream_method == "dif":
        return "upstream_normal_approximation"
    if upstream_method.startswith("dif-ind-"):
        return "upstream_binomial_quantile_with_greater_equal_rule"
    if model == "2p7B":
        return "upstream_unseeded_simulation_pointwise_minimum_of_13_cutoff_curves"
    return "upstream_unseeded_simulation_mean_of_10_cutoff_curves"


def parse_upstream_payloads(
    *,
    model: str,
    scheme: str,
    null_payload: Mapping[str, Any],
    result_payload: Mapping[str, Any],
    null_record: Mapping[str, Any],
    result_record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Convert one verified null/result pair into the common long schema.

    The upstream ``result`` arrays contain rejection rates (power), so the
    reported Type-II error is one minus the stored value.
    """

    menu = UPSTREAM_METHODS[scheme]
    available = set(null_payload) & set(result_payload) - {"top_probs"}
    unknown = available - set(menu)
    if unknown:
        raise ValueError(f"unmapped upstream methods: {sorted(unknown)}")
    missing = set(menu) - available
    if missing:
        raise ValueError(
            f"{model} {scheme} JSON pair is missing methods: {sorted(missing)}"
        )

    convention = (
        real.GUMBEL_CONVENTION
        if scheme == "gumbel"
        else real.SHIFTED_INVERSE_CONVENTION
    )
    rows: list[dict[str, Any]] = []
    for upstream_method, (method, label) in menu.items():
        null_rates = _validate_rate_curve(
            null_payload[upstream_method],
            filename=str(null_record["filename"]),
            method=upstream_method,
        )
        result_rates = _validate_rate_curve(
            result_payload[upstream_method],
            filename=str(result_record["filename"]),
            method=upstream_method,
        )
        calibration = _upstream_calibration(model, scheme, upstream_method)
        for sample, rates, record in (
            ("released_raw_empirical_null", null_rates, null_record),
            ("released_watermarked", result_rates, result_record),
        ):
            error_type = "type_i_error" if sample.endswith("null") else "type_ii_error"
            for horizon, rejection_rate in enumerate(rates, start=1):
                rate = float(rejection_rate)
                error_rate = rate if error_type == "type_i_error" else 1.0 - rate
                rows.append(
                    {
                        "analysis": "upstream_recorded",
                        "model": model,
                        "model_id": real.MODEL_SPECS[model]["model_id"],
                        "scheme": scheme,
                        "pivot_convention": convention,
                        "sample": sample,
                        "error_type": error_type,
                        "method": method,
                        "method_label": label,
                        "method_origin": "reference_score",
                        "horizon": horizon,
                        "n_documents": N_RELEASED_DOCUMENTS,
                        "rejection_rate": rate,
                        "error_rate": error_rate,
                        "mcse": _binary_mcse(rate),
                        "cutoff": "",
                        "boundary_probability": "",
                        "source_file": record["filename"],
                        "source_sha256": record["sha256"],
                        "calibration": calibration,
                        "bayesian_delta_prior": "not_applicable",
                        "bayesian_delta_prior_low": "",
                        "bayesian_delta_prior_high": "",
                    }
                )
    return rows


def load_upstream_rows(
    directory: Path,
    *,
    models: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load all requested, already verified upstream curve arrays."""

    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for model in models:
        names = upstream_json_names(model)
        for scheme in ("gumbel", "inverse"):
            pair: dict[str, tuple[Mapping[str, Any], dict[str, Any]]] = {}
            for kind in ("null", "result"):
                path = directory / names[(scheme, kind)]
                record = verify_upstream_json(path)
                with path.open(encoding="utf-8") as handle:
                    payload = json.load(handle)
                if not isinstance(payload, Mapping):
                    raise ValueError(f"{path} does not contain a JSON object")
                record.update({"model": model, "scheme": scheme, "sample": kind})
                records.append(record)
                pair[kind] = (payload, record)
            rows.extend(
                parse_upstream_payloads(
                    model=model,
                    scheme=scheme,
                    null_payload=pair["null"][0],
                    result_payload=pair["result"][0],
                    null_record=pair["null"][1],
                    result_record=pair["result"][1],
                )
            )
    return rows, records


def common_rows_from_experiment(
    experiment_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert real_data_experiment rows into the prefix-curve schema."""

    output: list[dict[str, Any]] = []
    for source in experiment_rows:
        sample = str(source["sample"])
        if sample not in {
            "released_raw_empirical_null",
            "released_watermarked",
        }:
            continue
        rejection_rate = float(source["rejection_rate"])
        error_type = "type_i_error" if sample.endswith("null") else "type_ii_error"
        output.append(
            {
                "analysis": "common_calibration",
                "model": source["model"],
                "model_id": source["model_id"],
                "scheme": source["scheme"],
                "pivot_convention": source["pivot_convention"],
                "sample": sample,
                "error_type": error_type,
                "method": source["method"],
                "method_label": source["method_label"],
                "method_origin": source["method_origin"],
                "horizon": int(source["horizon"]),
                "n_documents": int(source["n_documents"]),
                "rejection_rate": rejection_rate,
                "error_rate": (
                    rejection_rate
                    if error_type == "type_i_error"
                    else 1.0 - rejection_rate
                ),
                "mcse": float(source["mcse"]),
                "cutoff": source["cutoff"],
                "boundary_probability": source["boundary_probability"],
                "source_file": "six_verified_released_pickles",
                "source_sha256": "see_metadata_upstream_pickle_assets",
                "calibration": (
                    "one_fixed_seeded_exact_pivot_null_sample_with_"
                    "randomized_boundary_per_model_scheme_convention"
                ),
                "bayesian_delta_prior": source["bayesian_delta_prior"],
                "bayesian_delta_prior_low": source["bayesian_delta_prior_low"],
                "bayesian_delta_prior_high": source["bayesian_delta_prior_high"],
            }
        )
    return output


def _primary_convention(scheme: str) -> str:
    return (
        real.GUMBEL_CONVENTION
        if scheme == "gumbel"
        else real.PRIMARY_INVERSE_CONVENTION
    )


def _index_rows(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    index: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            row["analysis"],
            row["model"],
            row["scheme"],
            row["pivot_convention"],
            row["error_type"],
            row["method"],
            int(row["horizon"]),
        )
        if key in index:
            raise ValueError(f"duplicate curve row key: {key}")
        index[key] = row
    return index


def comparison_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    horizons: Sequence[int] = (50, 100, 200),
) -> dict[str, Any]:
    """Compare overlapping reference curves without claiming a paired MCSE."""

    index = _index_rows(rows)
    comparisons: list[dict[str, Any]] = []
    for model in real.MODEL_SPECS:
        for scheme, methods in PLOTTED_REFERENCE_METHODS.items():
            upstream_convention = (
                real.GUMBEL_CONVENTION
                if scheme == "gumbel"
                else real.SHIFTED_INVERSE_CONVENTION
            )
            primary_convention = _primary_convention(scheme)
            for error_type in ("type_i_error", "type_ii_error"):
                for method in methods:
                    for horizon in horizons:
                        upstream_key = (
                            "upstream_recorded", model, scheme,
                            upstream_convention, error_type, method, horizon,
                        )
                        primary_key = (
                            "common_calibration", model, scheme,
                            primary_convention, error_type, method, horizon,
                        )
                        if upstream_key not in index or primary_key not in index:
                            continue
                        upstream = index[upstream_key]
                        primary = index[primary_key]
                        item: dict[str, Any] = {
                            "model": model,
                            "scheme": scheme,
                            "error_type": error_type,
                            "method": method,
                            "horizon": horizon,
                            "upstream_recorded": float(upstream["error_rate"]),
                            "common_primary": float(primary["error_rate"]),
                            "common_primary_minus_upstream": (
                                float(primary["error_rate"])
                                - float(upstream["error_rate"])
                            ),
                            "upstream_mcse": float(upstream["mcse"]),
                            "common_primary_mcse": float(primary["mcse"]),
                        }
                        if scheme == "inverse":
                            matched_key = (
                                "common_calibration", model, scheme,
                                real.SHIFTED_INVERSE_CONVENTION,
                                error_type, method, horizon,
                            )
                            if matched_key in index:
                                matched = index[matched_key]
                                item.update(
                                    {
                                        "common_matched_shifted": float(
                                            matched["error_rate"]
                                        ),
                                        "common_matched_shifted_minus_upstream": (
                                            float(matched["error_rate"])
                                            - float(upstream["error_rate"])
                                        ),
                                        "common_matched_shifted_mcse": float(
                                            matched["mcse"]
                                        ),
                                    }
                                )
                        comparisons.append(item)

    def largest(field: str) -> Mapping[str, Any] | None:
        eligible = [row for row in comparisons if field in row]
        return max(eligible, key=lambda row: abs(float(row[field]))) if eligible else None

    return {
        "selected_horizons": list(horizons),
        "comparisons": comparisons,
        "largest_absolute_primary_difference": largest(
            "common_primary_minus_upstream"
        ),
        "largest_absolute_matched_shifted_difference": largest(
            "common_matched_shifted_minus_upstream"
        ),
        "mcse_warning": (
            "The two rates use the same 500 released documents, but the upstream "
            "JSON contains only aggregate rates. Their covariance and a paired "
            "MCSE for the difference therefore cannot be recovered from these "
            "files. The marginal MCSEs are reported separately."
        ),
    }


def write_curve_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CURVE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# Method identities are explicit and stable: hue identifies a scientific family
# and dash identifies the member within that family.  The Gumbel colours and
# dash patterns match the temperature-matched figures.  This encoding remains
# legible without relying on the pale half of ``tab20`` and does not change when
# another method is appended to a result file.
METHOD_PLOT_STYLE: dict[str, tuple[str, tuple[float, ...]]] = {
    # Classical reference scores.
    "h_ars": ("#D55E00", ()),
    "h_log": ("#D55E00", (2.5, 1.5)),
    "h_ind_1_over_e": ("#D55E00", (1.0, 1.4)),
    "h_neg": ("#D55E00", ()),
    # Least-favourable reference-score families.
    "h_gum_star_0.1": ("#009E73", ()),
    "h_gum_star_0.01": ("#009E73", (3.5, 1.2)),
    "h_gum_star_0.005": ("#009E73", (1.2, 1.2)),
    "h_dif_star_0.1": ("#009E73", ()),
    "h_dif_star_0.01": ("#009E73", (3.5, 1.2)),
    "h_dif_star_0.001": ("#009E73", (1.2, 1.2)),
    # Equal-tail, tail-shape and union-tail Bayesian families.
    "bayes_shared": ("#542788", (5.0, 1.5)),
    "bayes_tokenwise": ("#542788", (1.5, 1.2)),
    "bayes_shared_dirichlet": ("#9B2226", (6.0, 1.5, 1.0, 1.5)),
    "bayes_tokenwise_dirichlet": ("#9B2226", (1.0, 1.4)),
    "bayes_shared_uniontail": ("#0B6FA4", ()),
    "bayes_shared_tailwidth": ("#0B6FA4", (3.5, 1.2)),
}

# Dark fallbacks keep diagnostic or future figures readable, while every method
# in the manuscript figures is covered explicitly above.
FALLBACK_PLOT_STYLES: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("#333333", ()),
    ("#0072B2", (4.0, 1.5)),
    ("#CC79A7", (1.2, 1.2)),
)


def _method_plot_style(method: str) -> tuple[str, tuple[float, ...]]:
    """Return a deterministic dark colour and line pattern for ``method``."""

    if method in METHOD_PLOT_STYLE:
        return METHOD_PLOT_STYLE[method]
    fallback = sum(ord(character) for character in method) % len(FALLBACK_PLOT_STYLES)
    return FALLBACK_PLOT_STYLES[fallback]


def _plotted_common_methods(rows: Sequence[Mapping[str, Any]], scheme: str) -> tuple[str, ...]:
    reference = list(PLOTTED_REFERENCE_METHODS[scheme])
    bayesian = sorted(
        {
            str(row["method"])
            for row in rows
            if row["analysis"] == "common_calibration"
            and row["scheme"] == scheme
            and str(row["method_origin"]).startswith("bayesian")
        }
    )
    return tuple(reference + bayesian)


def _curve_lookup(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str, str, str, str], list[Mapping[str, Any]]]:
    grouped: dict[
        tuple[str, str, str, str, str, str], list[Mapping[str, Any]]
    ] = {}
    for row in rows:
        key = (
            str(row["analysis"]),
            str(row["model"]),
            str(row["scheme"]),
            str(row["pivot_convention"]),
            str(row["error_type"]),
            str(row["method"]),
        )
        grouped.setdefault(key, []).append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["horizon"]))
    return grouped


def _plot_grid(
    rows: Sequence[Mapping[str, Any]],
    *,
    analysis: str,
    methods_by_scheme: Mapping[str, Sequence[str]],
    output_pdf: Path,
    output_png: Path,
    title: str,
    subtitle: str,
    schemes: tuple[str, ...] = ("gumbel", "inverse"),
) -> None:
    """Write a readable 4-by-2 full-prefix grid in vector and raster form."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    lookup = _curve_lookup(rows)
    model_order = tuple(real.MODEL_SPECS)
    row_specs = [(model, scheme) for model in model_order for scheme in schemes]
    all_methods = tuple(dict.fromkeys(
        method for scheme in schemes for method in methods_by_scheme[scheme]
    ))
    styles = {method: _method_plot_style(method) for method in all_methods}
    labels: dict[str, str] = {}
    for row in rows:
        labels.setdefault(str(row["method"]), str(row["method_label"]))
    labels.update(
        {
            "h_ars": r"$h_{\mathrm{ars}}$",
            "h_log": r"$h_{\mathrm{log}}$",
            "h_ind_1_over_e": r"$h_{\mathrm{ind},1/e}$",
            "h_gum_star_0.1": r"$h^\star_{\mathrm{gum},.1}$",
            "h_gum_star_0.01": r"$h^\star_{\mathrm{gum},.01}$",
            "h_gum_star_0.005": r"$h^\star_{\mathrm{gum},.005}$",
            "h_neg": r"$h_{\mathrm{neg}}$",
            "h_dif_star_0.1": r"$h^*_{\mathrm{dif},.1}$",
            "h_dif_star_0.01": r"$h^*_{\mathrm{dif},.01}$",
            "h_dif_star_0.001": r"$h^*_{\mathrm{dif},.001}$",
            "bayes_tokenwise": r"Bayes, tokenwise $\Delta$",
            "bayes_shared": r"Bayes, shared $\Delta$",
            "bayes_tokenwise_dirichlet":
                r"Bayes, tokenwise $\Delta$ + tail shape",
            "bayes_shared_dirichlet":
                r"Bayes, shared $\Delta$ + tail shape",
            "bayes_shared_uniontail":
                r"Bayes, shared $\Delta$ + union tail",
            "bayes_shared_tailwidth":
                r"Bayes, shared $\Delta$ + tail width $J$",
        }
    )
    display_models = {
        "1p3B": "OPT-1.3B",
        "2p7B": "Sheared-LLaMA-2.7B",
    }

    present_methods = {
        str(row["method"])
        for row in rows
        if str(row["analysis"]) == analysis and str(row["scheme"]) in schemes
    }
    legend_methods = tuple(
        method for method in all_methods if method in present_methods
    )
    # The manuscript scales this 7.2-inch source to about 6.5 inches.  Eight-
    # point source text therefore remains above seven points in print.
    # The manuscript's two-row Gumbel figures use two columns: references form
    # the first column and Bayesian rules the second in the common-calibration
    # panel; the upstream panel likewise separates its two reference families.
    legend_columns = 2 if len(row_specs) <= 2 else 3
    legend_columns = max(1, min(legend_columns, len(legend_methods)))
    legend_rows = math.ceil(len(legend_methods) / legend_columns)
    top_in = 0.82
    bottom_in = 0.78 + 0.21 * legend_rows
    plot_height = 2.05 * len(row_specs)
    row_gaps = 0.34 * max(0, len(row_specs) - 1)
    figure_height = top_in + bottom_in + plot_height + row_gaps
    fig, axes = plt.subplots(
        len(row_specs), 2, figsize=(7.2, figure_height), sharex=True
    )
    height = fig.get_figheight()
    fig.subplots_adjust(
        top=1.0 - top_in / height,
        bottom=bottom_in / height,
        left=0.105,
        right=0.985,
        hspace=0.40,
        wspace=0.25,
    )
    fig.suptitle(title, fontsize=11.2, y=1.0 - 0.10 / height)
    fig.text(
        0.5,
        1.0 - 0.38 / height,
        subtitle,
        ha="center",
        va="top",
        fontsize=8.0,
        linespacing=1.2,
    )

    for row_index, (model, scheme) in enumerate(row_specs):
        convention = (
            _primary_convention(scheme)
            if analysis == "common_calibration"
            else (
                real.GUMBEL_CONVENTION
                if scheme == "gumbel"
                else real.SHIFTED_INVERSE_CONVENTION
            )
        )
        methods = tuple(methods_by_scheme[scheme])
        for column, error_type in enumerate(("type_i_error", "type_ii_error")):
            ax = axes[row_index, column]
            observed_max = 0.0
            for method in methods:
                key = (analysis, model, scheme, convention, error_type, method)
                curve = lookup.get(key)
                if not curve:
                    continue
                x = np.asarray([int(item["horizon"]) for item in curve])
                y = np.asarray([float(item["error_rate"]) for item in curve])
                observed_max = max(observed_max, float(y.max(initial=0.0)))
                color, dashes = styles[method]
                ax.plot(
                    x,
                    y,
                    color=color,
                    linestyle=(0, dashes) if dashes else "-",
                    linewidth=1.45,
                    solid_capstyle="round",
                    dash_capstyle="round",
                    label=labels.get(method, method),
                )
            if error_type == "type_i_error":
                ax.axhline(0.05, color="0.25", linestyle=":", linewidth=1.0)
                # Exact inclusion of n=1 creates large discrete-test spikes in
                # some upstream curves.  A symmetric-log scale keeps those
                # visible without compressing the behavior around alpha=.05.
                ax.set_yscale("symlog", linthresh=0.01, linscale=1.0)
                ax.set_ylim(0.0, 1.0)
                ax.set_yticks((0.0, 0.01, 0.05, 0.1, 1.0))
                ax.set_yticklabels(("0", ".01", ".05", ".1", "1"))
                ax.set_title("Empirical Type I error", fontsize=9.0)
            else:
                ax.set_ylim(0.0, 1.0)
                ax.set_title("Type II error", fontsize=9.0)
            ax.set_xlim(1, MAX_HORIZON)
            ax.grid(True, color="0.90", linewidth=0.6)
            ax.set_xlabel(r"Scored prefix length $n$", fontsize=8.0)
            ax.tick_params(axis="both", labelsize=8.0)
            if column == 0:
                scheme_label = "Gumbel" if scheme == "gumbel" else "Inverse transform"
                ax.set_ylabel(
                    f"{display_models[model]}\n{scheme_label}\nError rate", fontsize=8.0
                )

    legend_handles = [
        Line2D(
            [0], [0],
            color=styles[method][0],
            linewidth=1.8,
            linestyle=(0, styles[method][1]) if styles[method][1] else "-",
            solid_capstyle="round",
            dash_capstyle="round",
            label=labels.get(method, method),
        )
        for method in legend_methods
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=legend_columns,
        fontsize=8.0,
        frameon=False,
        handlelength=3.0,
        columnspacing=1.4,
        labelspacing=0.55,
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, format="pdf")
    fig.savefig(output_png, format="png", dpi=200)
    plt.close(fig)


def write_figures(
    rows: Sequence[Mapping[str, Any]], output_dir: Path
) -> list[Path]:
    common_methods = {
        scheme: _plotted_common_methods(rows, scheme)
        for scheme in ("gumbel", "inverse")
    }
    common_pdf = output_dir / "real_data_common_calibration_prefix_curves.pdf"
    common_png = output_dir / "real_data_common_calibration_prefix_curves.png"
    upstream_pdf = output_dir / "real_data_upstream_recorded_prefix_curves.pdf"
    upstream_png = output_dir / "real_data_upstream_recorded_prefix_curves.png"

    _plot_grid(
        rows,
        analysis="common_calibration",
        methods_by_scheme=common_methods,
        output_pdf=common_pdf,
        output_png=common_png,
        title="WatermarkFramework benchmark: common calibration",
        subtitle=(
            "Recomputed common calibration at each prefix; one cutoff per method "
            "is applied unchanged to both samples.\n"
            "Family colours and line patterns identify methods; inverse panels "
            "use formal ranks."
        ),
    )
    _plot_grid(
        rows,
        analysis="upstream_recorded",
        methods_by_scheme=PLOTTED_REFERENCE_METHODS,
        output_pdf=upstream_pdf,
        output_png=upstream_png,
        title="WatermarkFramework: archived reference-score curves",
        subtitle=(
            "Li et al. benchmark; Type II error = 1 - power. "
            "Reference scores only.\n"
            "The original cutoff construction is distinct from the common "
            "reanalysis."
        ),
    )
    return [common_pdf, common_png, upstream_pdf, upstream_png]


def _sort_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    order = {"upstream_recorded": 0, "common_calibration": 1}
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            order[str(row["analysis"])],
            str(row["model"]),
            str(row["scheme"]),
            str(row["pivot_convention"]),
            str(row["error_type"]),
            str(row["method"]),
            int(row["horizon"]),
        ),
    )


def build_metadata(
    *,
    config: real.RealDataConfig,
    common_metadata: Mapping[str, Any],
    upstream_records: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    artifacts: Sequence[Path],
) -> dict[str, Any]:
    common_methods = {
        scheme: list(_plotted_common_methods(rows, scheme))
        for scheme in ("gumbel", "inverse")
    }
    return {
        "experiment": "released_output_full_prefix_comparison",
        "horizons": list(range(1, MAX_HORIZON + 1)),
        "configuration": asdict(config),
        "analysis_definitions": {
            "common_calibration": {
                "description": (
                    "Scores recomputed from the six verified pickle inputs; "
                    "fixed-seeded exact-pivot-null calibration with randomized "
                    "cutoff boundaries; the same cutoff is used on the released "
                    "raw and watermarked documents."
                ),
                "primary_inverse_convention": real.PRIMARY_INVERSE_CONVENTION,
                "inverse_sensitivity_convention": real.SHIFTED_INVERSE_CONVENTION,
                "metadata": common_metadata,
            },
            "upstream_recorded": {
                "description": (
                    "Stored rejection-rate arrays read verbatim from the eight "
                    "commit-pinned JSON files; result-array power is converted "
                    "to Type-II error as one minus the stored rate."
                ),
                "values_digitized": False,
                "commit": real.OFFICIAL_COMMIT,
                "tree_url": real.OFFICIAL_TREE_URL,
                "json_assets": list(upstream_records),
                "inverse_cutoff_caveat": (
                    "The inverse optimal-score cutoff simulations are unseeded "
                    "and are rerun separately in the upstream null and "
                    "watermarked calls. For 2p7B, each horizon uses the pointwise "
                    "minimum of 13 simulated cutoff curves. The upstream inverse "
                    "pivots also use the shifted (rank-1)/(V-1) convention."
                ),
            },
        },
        "curve_schema": {
            "csv_fields": list(CURVE_FIELDS),
            "upstream_result_arrays": "stored rejection rate (power)",
            "upstream_null_arrays": "stored empirical rejection rate (Type-I error)",
            "error_rate": (
                "rejection_rate for released raw documents; 1-rejection_rate "
                "for released watermarked documents"
            ),
            "mcse": (
                "sample standard deviation divided by sqrt(500); upstream "
                "binary rates are reconstructed from their exact multiples of 1/500"
            ),
        },
        "figure_policy": {
            "common_calibration_plotted_methods": common_methods,
            "upstream_recorded_plotted_methods": {
                scheme: list(methods)
                for scheme, methods in PLOTTED_REFERENCE_METHODS.items()
            },
            "upstream_unplotted_methods": (
                "The additional pinned upstream reference-score curves remain "
                "in the CSV; the figure uses the overlapping displayed menu to "
                "keep its 4-by-2 panels legible."
            ),
            "figures_are_separate": True,
            "type_i_axis_scale": (
                "symlog with linear threshold .01, retaining the discrete "
                "short-prefix spikes at n=1 without compressing values near .05"
            ),
        },
        "comparison": comparison_summary(rows),
        "row_counts": {
            analysis: sum(row["analysis"] == analysis for row in rows)
            for analysis in ("common_calibration", "upstream_recorded")
        },
        "interpretation_limits": [
            (
                "The two analyses are not Monte Carlo replicates of one fixed "
                "test because their cutoff constructions differ."
            ),
            (
                "The same 500 documents underlie both sets of rates, but the "
                "upstream JSON files do not retain per-document decisions; a "
                "paired MCSE for their difference is therefore unavailable."
            ),
            (
                "The released watermarked continuations end at 200 scored "
                "tokens, so these files cannot support matched power curves "
                "beyond n=200."
            ),
        ],
        "artifacts": [
            {
                "filename": path.name,
                "sha256": real.file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in artifacts
        ],
    }


def _uniform_real_config(args: argparse.Namespace) -> real.RealDataConfig:
    base = real.RealDataConfig()
    config = real.RealDataConfig(
        alpha=args.alpha,
        horizons=tuple(range(1, MAX_HORIZON + 1)),
        n_calibration=args.n_calibration,
        calibration_seed=args.calibration_seed,
        delta_prior=base.delta_prior,
        delta_low=base.delta_low,
        delta_high=base.delta_high,
        bayes_quadrature_nodes=args.bayes_quadrature_nodes,
        gumbel_lookup_size=args.gumbel_lookup_size,
        gumbel_lookup_logit_limit=base.gumbel_lookup_logit_limit,
        dirichlet_c_nodes=args.dirichlet_c_nodes,
        score_batch_size=args.score_batch_size,
    )
    config.validate()
    if not (
        config.delta_prior == base.delta_prior
        and math.isclose(config.delta_low, base.delta_low)
        and math.isclose(config.delta_high, base.delta_high)
    ):
        raise RuntimeError(
            "prefix curves require the canonical real-data prior "
            f"Delta ~ Uniform({base.delta_low:g}, {base.delta_high:g})"
        )
    return config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = real.RealDataConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=real.DEFAULT_DATA_DIR)
    parser.add_argument(
        "--upstream-json-dir",
        type=Path,
        default=None,
        help="directory containing the eight JSON arrays (default: --data-dir)",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--models", nargs="+", choices=tuple(real.MODEL_SPECS),
        default=list(real.MODEL_SPECS),
    )
    parser.add_argument("--alpha", type=float, default=defaults.alpha)
    parser.add_argument(
        "--n-calibration", type=int, default=defaults.n_calibration
    )
    parser.add_argument(
        "--calibration-seed", type=int, default=defaults.calibration_seed
    )
    parser.add_argument(
        "--bayes-quadrature-nodes",
        type=int,
        default=defaults.bayes_quadrature_nodes,
    )
    parser.add_argument(
        "--gumbel-lookup-size", type=int, default=defaults.gumbel_lookup_size
    )
    parser.add_argument(
        "--dirichlet-c-nodes", type=int, default=defaults.dirichlet_c_nodes
    )
    parser.add_argument(
        "--score-batch-size", type=int, default=defaults.score_batch_size
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="require all six pickles and eight JSON files to exist locally",
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="write CSV/JSON only (useful for a non-graphical smoke test)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if len(set(args.models)) != len(args.models):
        raise ValueError("models must not contain duplicates")
    config = _uniform_real_config(args)
    download_missing = not args.no_download
    upstream_json_dir = (
        args.data_dir if args.upstream_json_dir is None else args.upstream_json_dir
    )

    # Validate/download JSON first.  run_real_data_experiment independently
    # validates/downloads the six pickle inputs before unpickling anything.
    ensure_upstream_json(
        upstream_json_dir,
        models=args.models,
        download_missing=download_missing,
    )
    upstream_rows, upstream_records = load_upstream_rows(
        upstream_json_dir, models=args.models
    )
    experiment_rows, common_metadata, _ = real.run_real_data_experiment(
        config,
        data_dir=args.data_dir,
        models=args.models,
        download_missing=download_missing,
    )
    common_rows = common_rows_from_experiment(experiment_rows)
    rows = _sort_rows([*upstream_rows, *common_rows])
    _index_rows(rows)  # Fail before writing if any curve key is duplicated.

    output_dir = args.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "real_data_prefix_curves.csv"
    write_curve_csv(rows, csv_path)
    artifacts: list[Path] = [csv_path]
    if not args.skip_plots:
        artifacts.extend(write_figures(rows, output_dir))

    metadata = build_metadata(
        config=config,
        common_metadata=common_metadata,
        upstream_records=upstream_records,
        rows=rows,
        artifacts=artifacts,
    )
    json_path = output_dir / "real_data_prefix_curves.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)

    print(f"wrote {len(rows)} curve rows to {csv_path}")
    print(f"wrote metadata and selected-horizon comparisons to {json_path}")
    if not args.skip_plots:
        print("wrote separate common-calibration and upstream-recorded PDF/PNG figures")


if __name__ == "__main__":
    main()
