#!/usr/bin/env python3
"""Update manuscript result-table estimates and parenthetical MC standard errors.

The manuscript tables are intentionally compact selections from larger result
artifacts.  This script performs the mechanical formatting step; the separate
``check_manuscript_tables.py`` command verifies every rendered estimate--SE
pair without writing files.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import check_manuscript_tables as audit


CELL_TOKEN = re.compile(
    r"\\(?P<macro>bestmcse|mcse)\{(?P<macro_est>\.[0-9]{4})\}"
    r"\{(?P<macro_se>\.[0-9]{4})\}"
    r"|\\textbf\{(?P<bold>\.[0-9]{4})\}"
    r"|(?P<plain>(?<![0-9])\.[0-9]{4})"
)


def update_row(
    block: str,
    marker: str,
    estimates: Iterable[float],
    standard_errors: Iterable[float],
    *,
    occurrence: int = 0,
) -> str:
    """Replace one table row, preserving which estimates are bold."""

    estimate_text = [audit.formatted(value) for value in estimates]
    se_text = [audit.formatted(value) for value in standard_errors]
    if len(estimate_text) != len(se_text):
        raise AssertionError(f"estimate/SE length mismatch for {marker!r}")

    position = -1
    for _ in range(occurrence + 1):
        position = block.find(marker, position + 1)
        if position < 0:
            raise AssertionError(f"missing table row marker: {marker!r}")
    row_end = block.index(r"\\", position)
    first_separator = block.index("&", position)
    start = first_separator + 1
    segment = block[start:row_end]
    index = 0

    def replacement(match: re.Match[str]) -> str:
        nonlocal index
        if index >= len(estimate_text):
            raise AssertionError(f"too many numeric cells in row {marker!r}")
        observed = match.group("macro_est") or match.group("bold") or match.group(
            "plain"
        )
        wanted = estimate_text[index]
        if observed != wanted:
            raise AssertionError(
                f"row {marker!r} drifted before update: {observed} != {wanted}"
            )
        bold = match.group("macro") == "bestmcse" or match.group("bold") is not None
        macro = "bestmcse" if bold else "mcse"
        rendered = rf"\{macro}{{{wanted}}}{{{se_text[index]}}}"
        index += 1
        return rendered

    updated = CELL_TOKEN.sub(replacement, segment)
    if index != len(estimate_text):
        raise AssertionError(
            f"row {marker!r} has {index} numeric cells; expected {len(estimate_text)}"
        )
    return block[:start] + updated + block[row_end:]


def update_table(
    tex: str,
    label: str,
    rows: Iterable[tuple[str, list[float], list[float], int]],
    *,
    body_only: bool = False,
) -> str:
    original = audit.raw_table_block(tex, label)
    normalized = audit.canonical_method_labels(original)
    prefix = ""
    updated = normalized
    if body_only:
        body_start = normalized.index(r"\midrule")
        prefix, updated = normalized[:body_start], normalized[body_start:]
    for marker, estimates, standard_errors, occurrence in rows:
        updated = update_row(
            updated,
            marker,
            estimates,
            standard_errors,
            occurrence=occurrence,
        )
    updated = prefix + updated
    old_cells = list(CELL_TOKEN.finditer(normalized))
    new_cells = list(CELL_TOKEN.finditer(updated))
    if len(old_cells) != len(new_cells):
        raise AssertionError(f"numeric cell count changed while updating tab:{label}")
    edits = []
    for old, new in zip(old_cells, new_cells):
        if old.group() != new.group():
            start, stop = audit.source_cell_span(
                normalized, original, old.start(), old.end()
            )
            edits.append((start, stop, new.group()))
    raw_updated = original
    for start, stop, replacement in reversed(edits):
        raw_updated = raw_updated[:start] + replacement + raw_updated[stop:]
    return tex.replace(original, raw_updated, 1)


def clean_values(
    rows: list[dict[str, str]], method: str, scheme: str, field: str
) -> list[float]:
    return audit.values_at_horizons(rows, method, scheme, field=field)


def clean_standard_errors(
    rows: list[dict[str, str]], summary: dict[str, object], method: str, scheme: str
) -> list[float]:
    """The clean shared/family tables use the same boundary-rate uncertainty."""

    if method == "h_ind_1_over_e":
        return [
            audit.conditional_weight_mc_se(
                summary,
                scenario="shared_delta_equal_tail_sensitivity",
                scheme=scheme,
                method=method,
                horizon=horizon,
            )
            for horizon in (100, 300, 700)
        ]
    return clean_values(rows, method, scheme, "type2_mc_se")


def main() -> None:
    if audit.GUMBEL_ONLY:
        # The current manuscript's row set and table layout are maintained by
        # the checker-backed synchronizer.  Reuse that mapping instead of the
        # legacy two-pivot row lists below, which include removed diagnostics
        # and inverse-transform rows.  The synchronizer preserves current
        # labels, uses the corrected conditional MCSEs, and verifies the result.
        from sync_manuscript_tables import main as synchronize

        synchronize([])
        return

    tex = audit.TEX_PATH.read_text(encoding="utf-8")
    clean = audit.read_csv("benchmark_results.csv")
    clean_summary = audit.read_json("benchmark_summary.json")
    contamination = audit.read_csv("contamination_results.csv")
    regime = audit.read_json("regime_sweep.json")
    tails = audit.read_json("tail_regime_sweep.json")
    released = audit.read_csv("real_model/real_data_results.csv")

    shared_mapping = (
        (r"$h_{\mathrm{ars}}$", "h_ars", "gumbel", 0),
        (r"$h_{\log}$", "h_log", "gumbel", 0),
        (r"$h_{\mathrm{ind},1/e}$", "h_ind_1_over_e", "gumbel", 0),
        (r"$h^\star_{\mathrm{gum},.01}$", "h_gum_star_0.01", "gumbel", 0),
        (r"$h^\star_{\mathrm{gum},.005}$", "h_gum_star_0.005", "gumbel", 0),
        ("Bayes, tokenwise prior", "bayes_tokenwise", "gumbel", 0),
        ("Bayes, tokenwise $+$ tail shape", "bayes_tokenwise_dirichlet", "gumbel", 0),
        (r"Bayes, shared $\Delta$ &", "bayes_shared", "gumbel", 0),
        (r"Bayes, shared $\Delta$ $+$ tail shape", "bayes_shared_dirichlet", "gumbel", 0),
        (r"Bayes, shared $\Delta$ $+$ union tail", "bayes_shared_uniontail", "gumbel", 0),
        (r"$h_{\mathrm{neg}}$", "h_neg", "inverse", 0),
        (r"$h^\star_{\mathrm{dif},.1}$", "h_dif_star_0.1", "inverse", 0),
        (r"$h^\star_{\mathrm{dif},.01}$", "h_dif_star_0.01", "inverse", 0),
        (r"$h^\star_{\mathrm{dif},.001}$", "h_dif_star_0.001", "inverse", 0),
        ("Bayes, tokenwise prior", "bayes_tokenwise", "inverse", 1),
        (r"Bayes, shared $\Delta$ &", "bayes_shared", "inverse", 1),
    )
    shared_rows: list[tuple[str, list[float], list[float], int]] = []
    for marker, method, scheme, occurrence in shared_mapping:
        standard_errors = clean_standard_errors(clean, clean_summary, method, scheme)
        shared_rows.append(
            (
                marker,
                clean_values(clean, method, scheme, "type2_error"),
                standard_errors,
                occurrence,
            )
        )
    tex = update_table(tex, "shared", shared_rows)

    family_mapping = (
        (r"$h_{\mathrm{ind},1/e}$", "h_ind_1_over_e"),
        (r"$h^\star_{\mathrm{gum},.005}$", "h_gum_star_0.005"),
        (r"$h^{\mathrm{sp}}_{.01}$", "h_spike_0.01"),
        (r"$h^{\mathrm{sp}}_{.05}$", "h_spike_0.05"),
        ("Bayes, tokenwise prior (equal", "bayes_tokenwise"),
        (r"Bayes, shared $\Delta$ (equal", "bayes_shared"),
        ("Bayes, tokenwise prior (tail shape", "bayes_tokenwise_dirichlet"),
        (r"Bayes, shared $\Delta$ $+$ tail shape", "bayes_shared_dirichlet"),
        (r"Bayes, shared $\Delta$ $+$ union tail", "bayes_shared_uniontail"),
    )
    tex = update_table(
        tex,
        "family",
        [
            (
                marker,
                clean_values(clean, method, "gumbel", "type2_error"),
                clean_standard_errors(clean, clean_summary, method, "gumbel"),
                0,
            )
            for marker, method in family_mapping
        ],
    )

    regime_cells = {
        (str(cell["horizon"]), str(cell["rule"]), str(cell["regime"])): cell
        for cell in regime["cells"]
    }
    regime_rows = []
    for occurrence, horizon in enumerate(("100", "300", "700")):
        type2 = regime["type2_error"][horizon]
        for marker, method in audit.REGRET_ROWS:
            estimates = audit.expected_row(type2, method)
            standard_errors = [
                float(regime_cells[(horizon, method, name)]["type2_mc_se"])
                for name in audit.REGRET_REGIMES
            ]
            standard_errors.append(
                float(regime["max_regret"][horizon][method]["max_regret_mc_se"])
            )
            anchored = marker + " &" if method == "bayes_shared" else marker
            regime_rows.append(
                (anchored, estimates, standard_errors, occurrence)
            )
    tex = update_table(tex, "regret", regime_rows, body_only=True)

    tail_mapping = (
        (r"Tail shape, $\alpha=\infty$", "bayes_shared_spike"),
        (r"Tail shape, $\alpha_0=1000$", "bayes_shared_alpha_1000"),
        (r"Tail shape, $\alpha_0=100$", "bayes_shared_alpha_100"),
        (r"Tail shape, $\alpha_0=10$", "bayes_shared_alpha_10"),
        (r"Tail shape, $\alpha_0=1$", "bayes_shared_alpha_1"),
        (r"Tail shape, $\alpha_0=0.1$", "bayes_shared_alpha_0.1"),
        (r"Bayes, shared $\Delta$ $+$ tail shape", "bayes_shared_dirichlet_mixture"),
        (r"Bayes, shared $\Delta$ $+$ union tail", "bayes_shared_uniontail"),
        (r"$h^{\mathrm{sp}}_{.01}$", "h_spike_0.01"),
        (r"$h^\star_{\mathrm{gum},.005}$", "h_gum_star_0.005"),
    )
    tail_cells = {
        (str(cell["horizon"]), str(cell["rule"]), str(cell["regime"])): cell
        for cell in tails["cells"]
    }
    laws = ("T1", "T2", "T3", "T4", "T5", "T6")
    tail_rows = []
    for occurrence, horizon in enumerate(("100", "300", "700")):
        for marker, method in tail_mapping:
            estimates = [
                float(tails["type2_error"][horizon][method][law]) for law in laws
            ]
            estimates.append(
                float(tails["max_regret"][horizon][method]["max_regret"])
            )
            standard_errors = [
                float(tail_cells[(horizon, method, law)]["type2_mc_se"])
                for law in laws
            ]
            standard_errors.append(
                float(tails["max_regret"][horizon][method]["max_regret_mc_se"])
            )
            tail_rows.append((marker, estimates, standard_errors, occurrence))
    tex = update_table(tex, "tails", tail_rows, body_only=True)

    contamination_mapping = (
        ("Bayes, robust shared &", "gumbel", "bayes_shared_robust", 0),
        ("Bayes, robust shared $+$ tail shape", "gumbel", "bayes_shared_dirichlet_robust", 0),
        ("Bayes, robust shared $+$ union tail", "gumbel", "bayes_shared_uniontail_robust", 0),
        (r"Bayes, assumes $\rho=0$ &", "gumbel", "bayes_shared_clean", 0),
        (r"Bayes, shared $\Delta$ $+$ tail shape ($\rho=0$)", "gumbel", "bayes_shared_dirichlet_clean", 0),
        (r"Bayes, shared $\Delta$ $+$ union tail ($\rho=0$)", "gumbel", "bayes_shared_uniontail_clean", 0),
        (r"$h^\star_{\mathrm{gum},.005}$", "gumbel", "h_gum_star_0.005", 1),
        ("Bayes, robust shared &", "inverse", "bayes_shared_robust", 1),
        (r"Bayes, assumes $\rho=0$ &", "inverse", "bayes_shared_clean", 1),
        (r"$h^\star_{\mathrm{dif},.01}$", "inverse", "h_dif_star_0.01", 1),
    )
    rhos = (0.0, 0.25, 0.4, 0.6)
    contamination_rows = []
    for marker, scheme, method, occurrence in contamination_mapping:
        contamination_rows.append(
            (
                marker,
                [
                    audit.contamination_cell(
                        contamination,
                        scheme=scheme,
                        method=method,
                        rho_true=rho,
                    )
                    for rho in rhos
                ],
                [
                    audit.contamination_cell(
                        contamination,
                        scheme=scheme,
                        method=method,
                        rho_true=rho,
                        field="type_ii_mc_se",
                    )
                    for rho in rhos
                ],
                occurrence,
            )
        )
    tex = update_table(tex, "contamination", contamination_rows)

    anytime_mapping = (
        ("Gumbel & tokenwise, spike", "gumbel", "bayes_tokenwise"),
        ("Gumbel & tokenwise $+$ tail shape", "gumbel", "bayes_tokenwise_dirichlet"),
        (r"Gumbel & shared $\Delta$, spike", "gumbel", "bayes_shared"),
        (r"Gumbel & shared $\Delta$ $+$ tail shape", "gumbel", "bayes_shared_dirichlet"),
        (r"Gumbel & shared $\Delta$ $+$ union tail", "gumbel", "bayes_shared_uniontail"),
        (r"Inverse & tokenwise $\Delta$", "inverse", "bayes_tokenwise"),
        (r"Inverse & shared $\Delta$", "inverse", "bayes_shared"),
    )
    anytime_rows = []
    for marker, scheme, method in anytime_mapping:
        estimates = []
        standard_errors = []
        for field, se_field in (
            ("type1_error", "type1_mc_se"),
            ("type2_error", "type2_mc_se"),
        ):
            estimates.append(
                audit.clean_cell(
                    clean,
                    scenario="shared_delta_equal_tail_sensitivity",
                    scheme=scheme,
                    method=method,
                    horizon=700,
                    field=field,
                    decision_rule="anytime_bf_ge_1_over_alpha",
                )
            )
            standard_errors.append(
                audit.clean_cell(
                    clean,
                    scenario="shared_delta_equal_tail_sensitivity",
                    scheme=scheme,
                    method=method,
                    horizon=700,
                    field=se_field,
                    decision_rule="anytime_bf_ge_1_over_alpha",
                )
            )
        anytime_rows.append((marker, estimates, standard_errors, 0))
    tex = update_table(tex, "anytime", anytime_rows)

    released_mapping = (
        (r"$h_{\mathrm{ars}}$", "gumbel", "h_ars", 0),
        (r"$h_{\log}$", "gumbel", "h_log", 0),
        (r"$h_{\mathrm{ind},1/e}$", "gumbel", "h_ind_1_over_e", 0),
        (r"$h^\star_{\mathrm{gum},.1}$", "gumbel", "h_gum_star_0.1", 0),
        ("Bayes, tokenwise prior &", "gumbel", "bayes_tokenwise", 0),
        (r"Bayes, shared $\Delta$ &", "gumbel", "bayes_shared", 0),
        ("Bayes, tokenwise $+$ tail shape", "gumbel", "bayes_tokenwise_dirichlet", 0),
        (r"Bayes, shared $\Delta$ $+$ tail shape", "gumbel", "bayes_shared_dirichlet", 0),
        (r"Bayes, shared $\Delta$ $+$ union tail", "gumbel", "bayes_shared_uniontail", 0),
        (r"$h_{\mathrm{neg}}$", "inverse", "h_neg", 0),
        (r"$h^\star_{\mathrm{dif},.1}$", "inverse", "h_dif_star_0.1", 0),
        (r"$h^\star_{\mathrm{dif},.01}$", "inverse", "h_dif_star_0.01", 0),
        (r"$h^\star_{\mathrm{dif},.001}$", "inverse", "h_dif_star_0.001", 0),
        ("Bayes, tokenwise prior &", "inverse", "bayes_tokenwise", 1),
        (r"Bayes, shared $\Delta$ &", "inverse", "bayes_shared", 1),
    )
    released_rows = []
    for marker, scheme, method, occurrence in released_mapping:
        estimates = []
        standard_errors = []
        for model in ("1p3B", "2p7B"):
            for sample in (
                "released_raw_empirical_null",
                "released_watermarked",
            ):
                estimates.append(
                    audit.real_data_cell(
                        released,
                        model=model,
                        scheme=scheme,
                        method=method,
                        sample=sample,
                    )
                )
                standard_errors.append(
                    audit.real_data_cell(
                        released,
                        model=model,
                        scheme=scheme,
                        method=method,
                        sample=sample,
                        field="mcse",
                    )
                )
        released_rows.append((marker, estimates, standard_errors, occurrence))
    tex = update_table(tex, "released-output", released_rows)

    audit.TEX_PATH.write_text(tex, encoding="utf-8")
    print(f"updated {audit.TEX_PATH}")


if __name__ == "__main__":
    main()
