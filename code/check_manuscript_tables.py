#!/usr/bin/env python3
"""Fail if manuscript result tables drift from generated artifacts.

The manuscript deliberately presents compact selected tables rather than
including the full CSVs.  This check keeps those hand-formatted rows tied to
the machine-readable outputs.  It uses only the standard library and performs
no writes.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "bayesian_paper_benchmark"
# The working manuscript is the Gumbel-only version; the two-pivot original is
# kept under manuscript/Old/.  ``--tex PATH`` overrides, and ``GUMBEL_ONLY``
# makes the checker skip the inverse rows that variant does not display.
TEX_PATH = ROOT / "manuscript" / "bayesian_pivot_watermark_gumbel.tex"
LEGACY_TEX_PATH = ROOT / "manuscript" / "Old" / "bayesian_pivot_watermark.tex"
GUMBEL_ONLY = True
CELL_NUMBER = r"(?:\.[0-9]{4}|1\.0000)"
ESTIMATE_WITH_MC_SE = re.compile(
    r"\\(?:best)?mcse\{(" + CELL_NUMBER + r")\}\{(" + CELL_NUMBER + r")\}"
)


def visible_tex(tex: str) -> str:
    """Remove real TeX comments and reject likely unescaped percentages.

    Checking the raw source previously let numerical prose pass even when an
    unescaped percentage hid the rest of its line in the compiled manuscript.
    Escaped percent signs survive; text already inside a comment is ignored.
    """

    lines = []
    for line_number, line in enumerate(tex.splitlines(), 1):
        for position, character in enumerate(line):
            if character != "%":
                continue
            backslashes = 0
            cursor = position - 1
            while cursor >= 0 and line[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if backslashes % 2:
                continue
            if position and line[position - 1].isdigit():
                raise AssertionError(
                    f"unescaped percentage at TeX line {line_number}; "
                    "the rest of the line would not render"
                )
            line = line[:position]
            break
        lines.append(line)
    return "\n".join(lines)


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(name: str) -> dict[str, object]:
    with (RESULTS / name).open(encoding="utf-8") as stream:
        return json.load(stream)


def canonical_method_labels(tex: str) -> str:
    """Map current display names to the legacy labels used by artifact checks.

    Only method-label text is normalized. Numeric cells and stored method keys
    are untouched, and the archived manuscript remains checkable as before.
    Longer labels precede their prefixes to keep distinct rules distinct.
    """

    aliases = (
        (r"Bayes, tokenwise $\Delta$ $+$ tail shape", "Bayes, tokenwise $+$ tail shape"),
        (r"Bayes, tokenwise $\Delta$", "Bayes, tokenwise prior"),
        (r"Bayes, hierarchical $\Delta$", r"Bayes, pooled $\Delta$"),
        (r"\quad hierarchical $\Delta$ (equal tail)", r"\quad pooled $\Delta$ (equal tail)"),
        (r"\quad $+$ shared tail width", r"\quad $+$ tail width, frozen ladder"),
        (r"\quad $+$ hierarchical tail width", r"\quad $+$ tail width, pooled"),
        (r"Bayes, shared $\Delta$ $+$ tail shape ($\rho$ mixture)",
         "Bayes, robust shared $+$ tail shape"),
        (r"Bayes, shared $\Delta$ $+$ union tail ($\rho$ mixture)",
         "Bayes, robust shared $+$ union tail"),
        (r"Bayes, shared $\Delta$ ($\rho$ mixture)", "Bayes, robust shared"),
        (r"Bayes, shared $\Delta$ ($\rho=0$)", r"Bayes, assumes $\rho=0$"),
    )
    for current, legacy in aliases:
        tex = tex.replace(current, legacy)
    return tex


def raw_table_block(tex: str, label: str) -> str:
    """Return the source table without changing current display labels."""

    marker = rf"\label{{tab:{label}}}"
    position = tex.index(marker)
    start = tex.rfind(r"\begin{table}", 0, position)
    stop = tex.index(r"\end{table}", position)
    if start < 0:
        raise AssertionError(f"could not locate table {label}")
    return tex[start:stop]


def table_block(tex: str, label: str) -> str:
    return canonical_method_labels(raw_table_block(tex, label))


def source_cell_span(block: str, raw: str, start: int, stop: int) -> tuple[int, int]:
    """Map a numeric cell in a label-normalized block back to its raw source.

    Method-label aliases change character offsets but not source lines or table
    separators.  Comments may have been stripped by visible_tex.  Anchor the
    cell to its line's preceding separator and verify its original bytes before
    allowing an editor to replace it.
    """

    line_index = block.count("\n", 0, start)
    line_start = block.rfind("\n", 0, start) + 1
    prefix = block[line_start:start]
    raw_lines = raw.splitlines(keepends=True)
    if line_index >= len(raw_lines):
        raise AssertionError("normalized table has no corresponding source line")
    raw_line = raw_lines[line_index]
    separators = prefix.count("&")
    column = len(prefix)
    if separators:
        raw_separators = [m.start() for m in re.finditer("&", raw_line)]
        if len(raw_separators) < separators:
            raise AssertionError("normalized table separators differ from source")
        column = raw_separators[separators - 1] + len(prefix) - prefix.rfind("&")
    raw_start = sum(len(line) for line in raw_lines[:line_index]) + column
    raw_stop = raw_start + stop - start
    if raw[raw_start:raw_stop] != block[start:stop]:
        raise AssertionError("numeric cell does not match its mapped source span")
    return raw_start, raw_stop


def table_body(tex: str, label: str) -> str:
    """The tabular rows only.

    Captions now quote rule names such as ``$h^\\star_{\\mathrm{gum},.005}$``,
    so row markers must be searched below the header rule, not from the top of
    the float.
    """

    block = table_block(tex, label)
    return block[block.index(r"\midrule"):]


def formatted(value: float) -> str:
    text = f"{float(value):.4f}"
    return text[1:] if text.startswith("0") else text


def assert_mixed_row(
    block: str,
    marker: str,
    expected: "Iterable[float]",
    decimals: "Sequence[int]",
    *,
    occurrence: int = 0,
) -> None:
    """A bare-number row whose columns are printed to differing precision.

    tab:deficit-regime prints four-decimal deficits beside three-decimal
    proportions, so a single format would mismatch every second column.  Rows
    after the first are continuations that do not repeat the model name.
    """

    position = block.find(marker)
    if position < 0:
        raise AssertionError(f"missing table row marker: {marker!r}")
    row_end = block.index(r"\\", position)
    first_separator = block.index("&", position)
    for _ in range(occurrence):
        first_separator = block.index("&", row_end)
        row_end = block.index(r"\\", first_separator)
    segment = block[first_separator + 1 : row_end]
    # Take the LAST len(expected) cells rather than dropping a fixed prefix:
    # the temperature label is "$.1$" in most rows but "$1$" at temperature one,
    # which a value regex does not capture, shifting every later column by one.
    cells = [c.strip() for c in segment.split("&")]
    observed = [re.sub(r"[^\d.]", "", c) for c in cells[-len(expected):]]
    wanted = []
    for value, places in zip(expected, decimals):
        text = f"{value:.{places}f}"
        wanted.append(text[1:] if text.startswith("0.") else text)
    if observed != wanted:
        raise AssertionError(
            f"row {marker!r} occurrence {occurrence} drifted: "
            f"manuscript {observed}, artifacts {wanted}"
        )


def assert_bare_row(
    block: str,
    marker: str,
    expected: Iterable[float],
    *,
    occurrence: int = 0,
    decimals: int = 4,
) -> None:
    """Like assert_row, for tables whose cells carry no \\mcse pair.

    tab:deficit-support reports bare point estimates by design, so the
    mcse-pair regex finds nothing there and any row would pass vacuously.
    A second prior arm is a CONTINUATION row that does not repeat the label,
    so ``occurrence`` steps forward whole rows rather than re-finding a marker.
    """

    position = block.find(marker)
    if position < 0:
        raise AssertionError(f"missing table row marker: {marker!r}")
    row_end = block.index(r"\\", position)
    first_separator = block.index("&", position)
    for _ in range(occurrence):
        first_separator = block.index("&", row_end)
        row_end = block.index(r"\\", first_separator)
    segment = block[first_separator + 1 : row_end]
    # Drop the prior-support column, e.g. "$(.001,.5)$": those digits label the
    # row and would shift every comparison by one.
    segment = re.sub(r"\$\([^)]*\)\$", "", segment)
    observed = re.findall(r"(?<![\d.])(?:1\.000|\.\d{3,4})(?![\d])", segment)
    def render(value: float) -> str:
        text = f"{value:.{decimals}f}"
        return text[1:] if text.startswith("0.") else text
    wanted = [render(value) for value in expected]
    if observed != wanted:
        raise AssertionError(
            f"row {marker!r} drifted: manuscript {observed}, artifacts {wanted}"
        )


def assert_row(
    block: str,
    marker: str,
    expected: Iterable[float],
    expected_mc_se: Iterable[float],
    *,
    occurrence: int = 0,
    continuation: bool = False,
    signed: bool = False,
) -> None:
    """``continuation`` steps forward whole rows instead of re-finding ``marker``.

    A second prior arm in tab:deficit-support does not repeat its label, so
    searching for the marker again finds nothing.
    """

    if continuation:
        position = block.find(marker)
        if position < 0:
            raise AssertionError(f"missing table row marker: {marker!r}")
        row_end = block.index(r"\\", position)
        first_separator = block.index("&", position)
        for _ in range(occurrence):
            first_separator = block.index("&", row_end)
            row_end = block.index(r"\\", first_separator)
    else:
        position = -1
        for _ in range(occurrence + 1):
            position = block.find(marker, position + 1)
            if position < 0:
                raise AssertionError(f"missing table row marker: {marker!r}")
        row_end = block.index(r"\\", position)
        first_separator = block.index("&", position)
    segment = block[first_separator + 1 : row_end]
    observed = ESTIMATE_WITH_MC_SE.findall(segment)
    expected = list(expected)
    expected_mc_se = list(expected_mc_se)
    if len(expected) != len(expected_mc_se):
        raise AssertionError(f"row {marker!r}: estimate/MCSE counts differ")
    wanted = list(
        zip(
            (formatted(abs(value) if signed else value) for value in expected),
            (formatted(value) for value in expected_mc_se),
        )
    )
    if observed != wanted:
        raise AssertionError(
            f"row {marker!r} drifted: manuscript {observed}, artifacts {wanted}"
        )
    if signed:
        # Signs are outside the macro in the difference row.  Checking only
        # abs(dauc) would accept a reversed scientific conclusion.
        signs = []
        for hit in ESTIMATE_WITH_MC_SE.finditer(segment):
            prefix = segment[:hit.start()]
            match = re.search(r"\$([+-])\$\s*$", prefix)
            signs.append(match.group(1) if match else None)
        wanted_signs = ["-" if value < 0 else "+" for value in expected]
        if signs != wanted_signs:
            raise AssertionError(
                f"row {marker!r} signs drifted: manuscript {signs}, "
                f"artifacts {wanted_signs}"
            )


def clean_cell(
    rows: list[dict[str, str]],
    *,
    scenario: str,
    scheme: str,
    method: str,
    horizon: int,
    field: str = "type2_error",
    decision_rule: str = "fixed_horizon_mc_calibrated",
) -> float:
    matches = [
        row
        for row in rows
        if row["scenario"] == scenario
        and row["scheme"] == scheme
        and row["method"] == method
        and row["decision_rule"] == decision_rule
        and int(row["horizon"]) == horizon
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one clean cell, found {len(matches)} for "
            f"{scenario}/{scheme}/{method}/{horizon}/{decision_rule}"
        )
    return float(matches[0][field])


def contamination_cell(
    rows: list[dict[str, str]],
    *,
    scheme: str,
    method: str,
    rho_true: float,
    horizon: int = 700,
    field: str = "type_ii_error",
) -> float:
    matches = [
        row
        for row in rows
        if row["scheme"] == scheme
        and row["method"] == method
        and float(row["rho_true"]) == float(rho_true)
        and int(row["horizon"]) == horizon
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one contamination cell, found {len(matches)} for "
            f"{scheme}/{method}/{rho_true}/{horizon}"
        )
    return float(matches[0][field])


def real_data_cell(
    rows: list[dict[str, str]],
    *,
    model: str,
    scheme: str,
    method: str,
    sample: str,
    horizon: int = 200,
    field: str = "rejection_rate",
) -> float:
    convention = (
        "stored_gumbel_pivot"
        if scheme == "gumbel"
        else "formal_eta_j_over_v_minus_1"
    )
    matches = [
        row
        for row in rows
        if row["model"] == model
        and row["scheme"] == scheme
        and row["pivot_convention"] == convention
        and row["sample"] == sample
        and row["method"] == method
        and int(row["horizon"]) == horizon
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one released-output cell, found {len(matches)} for "
            f"{model}/{scheme}/{method}/{sample}/{horizon}"
        )
    value = float(matches[0][field])
    if sample == "released_watermarked" and field == "rejection_rate":
        return 1.0 - value
    return value


def values_at_horizons(
    rows: list[dict[str, str]],
    method: str,
    scheme: str,
    *,
    decision_rule: str = "fixed_horizon_mc_calibrated",
    field: str = "type2_error",
) -> list[float]:
    return [
        clean_cell(
            rows,
            scenario="shared_delta_equal_tail_sensitivity",
            scheme=scheme,
            method=method,
            horizon=horizon,
            field=field,
            decision_rule=decision_rule,
        )
        for horizon in (100, 300, 700)
    ]


def conditional_weight_mc_se(
    summary: dict[str, object],
    *,
    scenario: str,
    scheme: str,
    method: str,
    horizon: int,
) -> float:
    """MC SE of a Rao--Blackwellized randomized-boundary miss rate."""

    indicator = summary["per_document_indicators"]
    assert isinstance(indicator, dict)
    records = indicator["tie_and_randomized_rejection_counts"]
    assert isinstance(records, list)
    matches = [
        record
        for record in records
        if record["scenario"] == scenario
        and record["scheme"] == scheme
        and record["method"] == method
        and int(record["horizon"]) == horizon
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one boundary record, found {len(matches)} for "
            f"{scenario}/{scheme}/{method}/{horizon}"
        )
    record = matches[0]
    n = int(record["n_documents"])
    equal = int(record["n_at_atom"])
    strict_reject = int(record["n_rejected_strict"])
    gamma = float(record["boundary_randomization_probability"])
    strict_miss = n - strict_reject - equal
    boundary_miss = 1.0 - gamma
    total = strict_miss + equal * boundary_miss
    total_squares = strict_miss + equal * boundary_miss**2
    sample_variance = (total_squares - total**2 / n) / (n - 1)
    return math.sqrt(max(sample_variance, 0.0) / n)


def check_shared_and_family(
    tex: str,
    rows: list[dict[str, str]],
    summary: dict[str, object],
) -> int:
    shared = table_block(tex, "shared")
    shared_rows = (
        (r"$h_{\mathrm{ars}}$", "h_ars", "gumbel", 0),
        (r"$h_{\log}$", "h_log", "gumbel", 0),
        (r"$h_{\mathrm{ind},1/e}$", "h_ind_1_over_e", "gumbel", 0),
        (r"$h^\star_{\mathrm{gum},.1}$", "h_gum_star_0.1", "gumbel", 0),
        (r"$h^\star_{\mathrm{gum},.01}$", "h_gum_star_0.01", "gumbel", 0),
        (r"$h^\star_{\mathrm{gum},.005}$", "h_gum_star_0.005", "gumbel", 0),
        ("Bayes, tokenwise prior", "bayes_tokenwise", "gumbel", 0),
        ("Bayes, tokenwise $+$ tail shape", "bayes_tokenwise_dirichlet", "gumbel", 0),
        (r"Bayes, shared $\Delta$ &", "bayes_shared", "gumbel", 0),
        (r"Bayes, shared $\Delta$ $+$ tail shape", "bayes_shared_dirichlet", "gumbel", 0),
        # Left unregistered when the union tail was added, so both of its rows
        # kept a stale .0190 against the artifact's .0192 through several
        # regenerations.  A row the checker does not name is a row that drifts.
        (r"Bayes, shared $\Delta$ $+$ union tail", "bayes_shared_uniontail", "gumbel", 0),
        (r"$h_{\mathrm{neg}}$", "h_neg", "inverse", 0),
        (r"$h^\star_{\mathrm{dif},.1}$", "h_dif_star_0.1", "inverse", 0),
        (r"$h^\star_{\mathrm{dif},.01}$", "h_dif_star_0.01", "inverse", 0),
        (r"$h^\star_{\mathrm{dif},.001}$", "h_dif_star_0.001", "inverse", 0),
        ("Bayes, tokenwise prior", "bayes_tokenwise", "inverse", 1),
        (r"Bayes, shared $\Delta$ &", "bayes_shared", "inverse", 1),
    )
    for marker, method, scheme, occurrence in shared_rows:
        if GUMBEL_ONLY and scheme == "inverse":
            continue
        standard_errors = values_at_horizons(
            rows, method, scheme, field="type2_mc_se"
        )
        if method == "h_ind_1_over_e":
            standard_errors = [
                conditional_weight_mc_se(
                    summary,
                    scenario="shared_delta_equal_tail_sensitivity",
                    scheme=scheme,
                    method=method,
                    horizon=horizon,
                )
                for horizon in (100, 300, 700)
            ]
        assert_row(
            shared,
            marker,
            values_at_horizons(rows, method, scheme),
            standard_errors,
            occurrence=occurrence,
        )

    family = table_block(tex, "family")
    family_rows = (
        (r"$h_{\mathrm{ars}}$", "h_ars"),
        (r"$h_{\log}$", "h_log"),
        (r"$h_{\mathrm{ind},1/e}$", "h_ind_1_over_e"),
        # All three published tunings: the table's whole point is that the
        # constant matters, which one row cannot show.
        (r"$h^\star_{\mathrm{gum},.1}$", "h_gum_star_0.1"),
        (r"$h^\star_{\mathrm{gum},.01}$", "h_gum_star_0.01"),
        (r"$h^\star_{\mathrm{gum},.005}$", "h_gum_star_0.005"),
        (r"$h^{\mathrm{sp}}_{.01}$", "h_spike_0.01"),
        (r"$h^{\mathrm{sp}}_{.05}$", "h_spike_0.05"),
        ("Bayes, tokenwise prior (equal", "bayes_tokenwise"),
        (r"Bayes, shared $\Delta$ (equal", "bayes_shared"),
        ("Bayes, tokenwise prior (tail shape", "bayes_tokenwise_dirichlet"),
        (r"Bayes, shared $\Delta$ $+$ tail shape", "bayes_shared_dirichlet"),
        # Same omission as in tab:shared -- this row carried .0190 against the
        # artifact's .0192 because nothing named it.
        (r"Bayes, shared $\Delta$ $+$ union tail", "bayes_shared_uniontail"),
        ("Bayes, tokenwise prior (union", "bayes_tokenwise_uniontail"),
    )
    for marker, method in family_rows:
        standard_errors = values_at_horizons(
            rows, method, "gumbel", field="type2_mc_se"
        )
        if method == "h_ind_1_over_e":
            standard_errors = [
                conditional_weight_mc_se(
                    summary,
                    scenario="shared_delta_equal_tail_sensitivity",
                    scheme="gumbel",
                    method=method,
                    horizon=horizon,
                )
                for horizon in (100, 300, 700)
            ]
        assert_row(
            family,
            marker,
            values_at_horizons(rows, method, "gumbel"),
            standard_errors,
        )
    return len(shared_rows) + len(family_rows)


REGRET_ROWS = (
    # All three tuning constants li2025framework publishes.  Showing one made
    # the sensitivity-to-tuning claim unfalsifiable from these tables.
    (r"$h^\star_{\mathrm{gum},.005}$", "h_gum_star_0.005"),
    (r"$h^\star_{\mathrm{gum},.1}$", "h_gum_star_0.1"),
    (r"$h^\star_{\mathrm{gum},.01}$", "h_gum_star_0.01"),
    (r"$h_{\mathrm{ars}}$", "h_ars"),
    (r"$h_{\log}$", "h_log"),
    (r"$h_{\mathrm{ind},1/e}$", "h_ind_1_over_e"),
    ("Bayes, tokenwise prior", "bayes_tokenwise"),
    (r"Bayes, shared $\Delta$", "bayes_shared"),
    (r"Bayes, shared $\Delta$ $+$ tail shape", "bayes_shared_dirichlet"),
    (r"Bayes, shared $\Delta$ $+$ union tail", "bayes_shared_uniontail"),
)

# The manuscript displays six of the nine generating regimes the sweep runs, in
# this order: the two uniform laws, then the four point deficits in increasing
# order.  Regret is defined over the displayed six, while the artifact's own
# max_regret block is taken over all nine, so recompute rather than read it.
# The tail sweep is arm-split the same way: its shape table reads
# max_regret_shape and its width table max_regret_width, never the pooled
# max_regret, so neither arm's regret column moves when the other gains a
# regime.
REGRET_REGIMES = ("A", "B", "G", "D", "H", "I")


def displayed_max_regret(cell: dict, method: str) -> float:
    best = {
        regime: min(float(cell[rule][regime]) for _, rule in REGRET_ROWS)
        for regime in REGRET_REGIMES
    }
    return max(float(cell[method][regime]) - best[regime] for regime in REGRET_REGIMES)


def expected_row(cell: dict, method: str, recorded: float) -> list[float]:
    """The six displayed regimes plus the regret recorded in the artifact.

    The regret benchmark is the best rule SCORED in the sweep, which is not the
    same as the best rule printed: the equal-tail diagnostics are scored and no
    longer shown.  Taking it from the artifact keeps the number independent of
    which rows fit on the page.
    """

    values = [float(cell[method][regime]) for regime in REGRET_REGIMES]
    values.append(recorded)
    return values


def check_regret(tex: str, payload: dict[str, object]) -> int:
    """The main sweep: one panel per horizon over the displayed regimes."""

    block = table_body(tex, "regret")
    type2 = payload["type2_error"]
    stored = payload["max_regret"]
    assert isinstance(type2, dict) and isinstance(stored, dict)
    cell_mc_se = {
        (str(cell["horizon"]), str(cell["rule"]), str(cell["regime"])): float(
            cell["type2_mc_se"]
        )
        for cell in payload["cells"]  # type: ignore[index]
    }

    checked = 0
    for occurrence, key in enumerate(("100", "300", "700")):
        cell = type2[key]  # type: ignore[index]
        for marker, method in REGRET_ROWS:
            shown = displayed_max_regret(cell, method)
            recorded = float(stored[key][method]["max_regret"])  # type: ignore[index]
            # The benchmark is the best rule SCORED in the sweep, not the best
            # one printed: which rows fit on a page should not move a reported
            # number.  The equal-tail diagnostics are scored but no longer
            # shown, so the printed subset can only have a weaker best rule and
            # therefore a smaller regret.  Equality is no longer required; a
            # displayed maximum that EXCEEDS the recorded one would mean the
            # recorded benchmark missed a rule, which is a real error.
            if shown - recorded > 5e-9:
                raise AssertionError(
                    f"max regret over the displayed regimes ({shown:.4f}) exceeds the "
                    f"all-rule value ({recorded:.4f}) for {method} at n={key}"
                )
            anchored = marker + " &" if method == "bayes_shared" else marker
            expected_se = [
                cell_mc_se[(key, method, regime)] for regime in REGRET_REGIMES
            ]
            expected_se.append(
                float(stored[key][method]["max_regret_mc_se"])  # type: ignore[index]
            )
            assert_row(
                block,
                anchored,
                expected_row(cell, method, recorded),
                expected_se,
                occurrence=occurrence,
            )
            checked += 1
    return checked


def check_tails(
    tex: str,
    payload: dict[str, object],
    *,
    table: str = "tails",
    laws: tuple[str, ...] = ("T1", "T2", "T3", "T4", "T5", "T6"),
    arm: str = "shape",
) -> int:
    """Check a tail sweep table.  ``laws`` selects the shape or width arm.

    ``arm`` picks the matching regret column.  Each table's regret is a maximum
    over the regimes it displays, so the shape table must not read the width
    arm's figure or the pooled one; they are three different claims.
    """

    block = table_body(tex, table)
    mapping = (
        (r"Tail shape, $\alpha=\infty$", "bayes_shared_spike"),
        (r"Tail shape, $\alpha_0=1000$", "bayes_shared_alpha_1000"),
        (r"Tail shape, $\alpha_0=100$", "bayes_shared_alpha_100"),
        (r"Tail shape, $\alpha_0=10$", "bayes_shared_alpha_10"),
        (r"Tail shape, $\alpha_0=1$", "bayes_shared_alpha_1"),
        (r"Tail shape, $\alpha_0=0.1$", "bayes_shared_alpha_0.1"),
        (r"Bayes, shared $\Delta$ $+$ tail shape", "bayes_shared_dirichlet_mixture"),
        (r"Bayes, shared $\Delta$ $+$ union tail", "bayes_shared_uniontail"),
        (r"$h^\star_{\mathrm{gum},.005}$", "h_gum_star_0.005"),
        (r"$h^\star_{\mathrm{gum},.1}$", "h_gum_star_0.1"),
        (r"$h^\star_{\mathrm{gum},.01}$", "h_gum_star_0.01"),
        (r"$h_{\mathrm{ars}}$", "h_ars"),
        (r"$h_{\log}$", "h_log"),
        (r"$h_{\mathrm{ind},1/e}$", "h_ind_1_over_e"),
    )
    type2 = payload["type2_error"]
    regret = payload["max_regret"]
    assert isinstance(type2, dict) and isinstance(regret, dict)
    cell_mc_se = {
        (str(cell["horizon"]), str(cell["rule"]), str(cell["regime"])): float(
            cell["type2_mc_se"]
        )
        for cell in payload["cells"]  # type: ignore[index]
    }

    checked = 0
    for occurrence, horizon in enumerate(("100", "300", "700")):
        for marker, method in mapping:
            expected = [
                float(type2[horizon][method][law]) for law in laws  # type: ignore[index]
            ]
            # The shape table's regret column must be taken over the SHAPE arm
            # only.  Reading the pooled "max_regret" here made adding the width
            # regimes silently rewrite this column (.0046 -> .0962 for the
            # alpha=inf layer), which is a different claim about a table whose
            # own rows never moved.
            expected.append(
                float(regret[horizon][method][f"max_regret_{arm}"])  # type: ignore[index]
            )
            expected_se = [
                cell_mc_se[(horizon, method, law)] for law in laws
            ]
            expected_se.append(
                float(regret[horizon][method][f"max_regret_{arm}_mc_se"])  # type: ignore[index]
            )
            assert_row(
                block,
                marker,
                expected,
                expected_se,
                occurrence=occurrence,
            )
            checked += 1
    return checked


# The reference scores as they appear in the three AUC tables, with their
# clustered standard errors from the prompt bootstrap.
AUC_REFERENCE_ROWS = (
    (r"$h_{\mathrm{ars}}$", "h_ars"),
    (r"$h_{\log}$", "h_log"),
    (r"$h_{\mathrm{ind},1/e}$", "h_ind_1_over_e"),
    (r"$h^\star_{\mathrm{gum},.1}$", "h_gum_star_0.1"),
    (r"$h^\star_{\mathrm{gum},.01}$", "h_gum_star_0.01"),
    (r"$h^\star_{\mathrm{gum},.005}$", "h_gum_star_0.005"),
)


def check_pooled_real(
    tex: str, curves: list[dict[str, str]], bootstrap: dict[str, object]
) -> int:
    """tab:pooled-real: the pooled deficit rule on the matched arms.

    Added in the same pass that introduced the table, after a negative control
    perturbed one of its cells and nothing objected.
    """

    block = table_body(tex, "pooled-real")
    pairs = [(m, t) for m in ("1p3B", "2p7B")
             for t in ("0.2", "0.3", "0.4", "0.5")]
    auc = {
        (r["model"], r["temperature"], r["rule"]): float(r["auc"])
        for r in curves if r["prefix"] == "100"
    }
    checked = 0
    for marker, rule in ((r"Bayes, shared $\Delta$", "bayes_shared"),
                         (r"Bayes, pooled $\Delta$", "bayes_pooled")):
        expected = [auc[(m, t, rule)] for m, t in pairs]
        # The clustered standard errors come from the bootstrap artifact, not
        # from the curve CSV, which carries no uncertainty for an AUC.
        expected_se = [
            float(bootstrap[f"{m}_T{t}"]["auc_cluster_se"][rule]) for m, t in pairs
        ]
        assert_row(block, marker, expected, expected_se)
        checked += 1
    # The difference row carries a PAIRED standard error, which is not
    # recoverable from the two marginal ones because the rules score the same
    # documents.  It had none at all until a referee asked.
    diffs = [
        bootstrap[f"{m}_T{t}"]["descriptive_differences"][
            "bayes_pooled_vs_bayes_shared"
        ]
        for m, t in pairs
    ]
    assert_row(
        block, "Difference &",
        [float(d["dauc"]) for d in diffs],
        [float(d["bootstrap_se"]) for d in diffs],
        signed=True,
    )
    checked += 1
    for marker, rule in AUC_REFERENCE_ROWS:
        assert_row(
            block, marker,
            [float(bootstrap[f"{m}_T{t}"]["auc"][rule]) for m, t in pairs],
            [float(bootstrap[f"{m}_T{t}"]["auc_cluster_se"][rule]) for m, t in pairs],
        )
        checked += 1
    return checked


def check_auc_reference_block(tex: str, label: str, bootstrap: dict, cells: list,
                              continuation: bool = False) -> int:
    """The reference-score block appended to an AUC table.

    These rows sit in their own panel so the sweep block keeps marking its own
    best: on the matched arms h_ars beats every mixing weight, and a single
    panel would have moved tab:union-weight's boldface off the weight sweep.
    """

    block = table_body(tex, label)
    checked = 0
    for marker, rule in AUC_REFERENCE_ROWS:
        assert_row(
            block, marker,
            [float(bootstrap[c]["auc"][rule]) for c in cells],
            [float(bootstrap[c]["auc_cluster_se"][rule]) for c in cells],
            continuation=continuation,
        )
        checked += 1
    return checked


def check_pooled_auc_mean(tex: str, bootstrap: dict[str, object]) -> int:
    """The union-tail mean comparison in the hierarchical-deficit discussion.

    The largest cell's improvement was reported as the eight-cell mean.  Read
    the same two AUCs in every specified cell and check their mean in the
    relevant section, so an unrelated occurrence cannot satisfy the check.
    """

    tex = visible_tex(tex)
    marker = r"\label{sec:deficit-pooling}"
    if marker not in tex:
        raise AssertionError("missing hierarchical-deficit section for AUC mean")
    section = tex.split(marker, 1)[1]
    section = re.split(r"\\(?:sub)*section\{", section, maxsplit=1)[0]
    section = re.sub(r"\s+", " ", section)
    cells = [f"{model}_T{temperature}"
             for model in ("1p3B", "2p7B")
             for temperature in ("0.2", "0.3", "0.4", "0.5")]
    differences = [
        float(bootstrap[cell]["auc"]["bayes_uniontail_w0.5"])
        - float(bootstrap[cell]["auc"]["bayes_shared"])
        for cell in cells
    ]
    mean = math.fsum(differences) / len(differences)
    wanted = f"mean AUC improvement of ${formatted(mean)}$"
    if wanted not in section:
        raise AssertionError(
            f"the hierarchical-deficit section should say {wanted!r}"
        )
    return 1


def check_union_weight(tex: str, payload: dict[str, object]) -> int:
    """tab:union-weight against the prompt-cluster bootstrap artifact.

    Unchecked until a negative control showed that perturbing a cell in this
    family went undetected: nothing read these 96 numbers.
    """

    block = table_body(tex, "union-weight")
    cells = [f"{m}_T{t}" for m in ("1p3B", "2p7B")
             for t in ("0.2", "0.3", "0.4", "0.5")]
    body = payload["cells"]  # type: ignore[index]
    checked = 0
    for weight in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        rule = f"bayes_uniontail_w{weight:g}"
        if weight == 0.0:
            marker = "$0$ (width only)"
        elif weight == 1.0:
            marker = "$1$ (shape only)"
        else:
            marker = f"${weight:g}$".replace("$0.", "$.")
        expected = [float(body[c]["auc"][rule]) for c in cells]
        expected_se = [float(body[c]["auc_cluster_se"][rule]) for c in cells]
        assert_row(block, marker, expected, expected_se)
        checked += 1
    expected = [float(body[c]["auc"]["bayes_shared"]) for c in cells]
    expected_se = [float(body[c]["auc_cluster_se"]["bayes_shared"]) for c in cells]
    baseline = "equal-tail baseline" if "equal-tail baseline" in block else "no tail layer"
    assert_row(block, baseline, expected, expected_se)
    return checked + 1


def check_deficit_support(
    tex: str,
    payload: dict[str, object],
    curves: list[dict[str, str]],
    bootstrap: dict[str, object],
) -> int:
    """tab:deficit-support against delong_tests_wideprior.json.

    The shared-Delta rows come from the curve CSV: the wide-prior artifact does
    not carry that rule, and leaving them out let a perturbation of .7828 pass
    unnoticed in a negative control.  Both of its prior arms are checked
    against the SAME number, which is the table's actual claim -- the equal-tail
    rule is invariant to widening the deficit prior.
    """

    block = table_body(tex, "deficit-support")
    shared = {
        (r["model"], r["temperature"]): float(r["auc"])
        for r in curves
        if r["prefix"] == "100" and r["rule"] == "bayes_shared"
    }
    cells = [f"{m}_T{t}" for m in ("1p3B", "2p7B")
             for t in ("0.2", "0.3", "0.4", "0.5")]
    rows = (
        (r"\quad $+$ tail width only", "bayes_uniontail_w0", 0),
        (r"\quad $+$ tail width only", "bayes_uniontail_w0_wide", 1),
        (r"\quad $+$ union tail, $w=.5$", "bayes_uniontail_w0.5", 0),
        (r"\quad $+$ union tail, $w=.5$", "bayes_uniontail_w0.5_wide", 1),
    )
    checked = 0
    pairs = [(m, t) for m in ("1p3B", "2p7B")
             for t in ("0.2", "0.3", "0.4", "0.5")]
    shared_expected = [shared[k] for k in pairs]
    for occurrence, rule in enumerate(("bayes_shared", "bayes_shared_wide")):
        assert_row(
            block, r"Bayes, shared $\Delta$ (equal tail)",
            shared_expected,
            [float(bootstrap[c]["auc_cluster_se"][rule]) for c in cells],
            occurrence=occurrence, continuation=True,
        )
        checked += 1
    for marker, rule, occurrence in rows:
        expected = [float(payload[c]["auc"][rule]) for c in cells]  # type: ignore[index]
        assert_row(
            block, marker, expected,
            [float(bootstrap[c]["auc_cluster_se"][rule]) for c in cells],
            occurrence=occurrence, continuation=True,
        )
        checked += 1
    return checked


def check_temperature_matched(tex: str, rows: list[dict[str, str]]) -> int:
    """tab:temperature-matched against the matched prefix-curve CSV.

    This table was unchecked, which is how a hand-added pooled row could enter
    it with nothing verifying the numbers.
    """

    block = table_body(tex, "temperature-matched")
    mapping = (
        (r"$h_{\mathrm{ars}}$", "h_ars"),
        (r"$h^\star_{\mathrm{gum},.1}$", "h_gum_star_0.1"),
        (r"$h_{\log}$", "h_log"),
        (r"Bayes, shared $\Delta$ (equal tail)", "bayes_shared"),
        (r"\quad pooled $\Delta$ (equal tail)", "bayes_pooled"),
        (r"\quad $+$ union tail, $w=.5$", "bayes_uniontail_w0.5"),
    )
    cells = [(m, t) for m in ("1p3B", "2p7B")
             for t in ("0.2", "0.3", "0.4", "0.5")]
    lookup = {
        (r["model"], r["temperature"], r["rule"]): r
        for r in rows if r["prefix"] == "100"
    }
    checked = 0
    for marker, rule in mapping:
        expected, expected_se = [], []
        for model, temperature in cells:
            row = lookup[(model, temperature, rule)]
            expected.append(float(row["type2"]))
            expected_se.append(float(row["type2_mcse"]))
        assert_row(block, marker, expected, expected_se)
        checked += 1
    return checked


def check_deficit_regime(tex: str, payload: dict[str, object]) -> int:
    """tab:deficit-regime against empirical_deficits.json.

    The last table nothing read.  Its entries summarize recorded deficits, so
    they cannot drift from a rerun, but they can drift from an edit -- which is
    what happened to the intraclass row of a sibling table.  Writing the
    artifact also caught a subtler thing: summarizing only the 500-document
    base rather than the pooled 2500 moves the quantiles visibly, so the
    artifact and the table now agree on the sample as well as the statistic.
    """

    block = table_body(tex, "deficit-regime")
    stats = payload["empirical_deficit_by_temperature"]  # type: ignore[index]
    fields = ("median", "mean", "q90", "below_support", "in_support", "above_half")
    checked = 0
    for model, marker in (("1p3B", "OPT-1.3B"), ("2p7B", "Sheared-LLaMA-2.7B")):
        rows = stats[model]  # type: ignore[index]
        for index, temperature in enumerate(sorted(rows, key=float)):
            cell = rows[temperature]
            expected = [float(cell[f]) for f in fields]
            decimals = [4, 4, 4, 3, 3, 3]
            assert_mixed_row(block, marker, expected, decimals, occurrence=index)
            checked += 1
    return checked


def check_deficit_pooling(tex: str, payload: dict[str, object]) -> int:
    """tab:deficit-pooling against deficit_persistence_sweep.json."""

    block = table_body(tex, "deficit-pooling")
    mapping = (
        (r"Bayes, shared $\Delta$", "bayes_shared"),
        (r"Bayes, tokenwise prior", "bayes_tokenwise"),
        (r"Bayes, pooled $\Delta$", "bayes_pooled"),
        (r"$h^\star_{\mathrm{gum},.005}$", "h_gum_star_0.005"),
        (r"$h^\star_{\mathrm{gum},.1}$", "h_gum_star_0.1"),
        (r"$h^\star_{\mathrm{gum},.01}$", "h_gum_star_0.01"),
        (r"$h_{\mathrm{ars}}$", "h_ars"),
        (r"$h_{\log}$", "h_log"),
        (r"$h_{\mathrm{ind},1/e}$", "h_ind_1_over_e"),
    )
    type2 = payload["type2_error"]
    regimes = list(payload["regimes"])  # type: ignore[arg-type]
    horizons = [str(h) for h in payload["config"]["horizons"]]  # type: ignore[index]
    n = int(payload["config"]["n_evaluation_alternative"])  # type: ignore[index]
    # The intraclass-correlation row is the axis the table is read against and
    # nothing verified it: a negative control changed .894 to .794 and the
    # checker passed.  It is (6/pi) arcsin(rho^2/2), not rho^2, and the table
    # carried rho^2 until that was corrected.
    icc = [float(payload["regimes"][g]["population_icc"]) for g in regimes]  # type: ignore[index]
    # The ICC row sits above \midrule, which table_body excludes, so read it
    # from the whole table block.
    assert_bare_row(
        table_block(tex, "deficit-pooling"),
        "Intraclass correlation", icc, decimals=3,
    )
    checked = 1
    for occurrence, horizon in enumerate(horizons):
        for marker, method in mapping:
            expected = [
                float(type2[horizon][method][g]) for g in regimes  # type: ignore[index]
            ]
            expected_se = [
                math.sqrt(max(v * (1.0 - v), 1e-12) / n) for v in expected
            ]
            # The max-regret cell is a maximum of differences, so its error is
            # the stored paired bootstrap rather than a binomial plug-in.  It
            # is appended, not checked separately, because assert_row compares
            # every estimate in the row: a column added to the table without a
            # number behind it would fail here rather than pass silently.
            regret = payload["max_regret"]["by_horizon"][horizon][method]  # type: ignore[index]
            expected.append(float(regret["max_regret"]))
            expected_se.append(float(regret["mc_se"]))
            assert_row(block, marker, expected, expected_se, occurrence=occurrence)
            checked += 1
    return checked

def check_width_pooling(tex: str, payload: dict[str, object]) -> int:
    """tab:width-pooling against width_persistence_sweep.json.

    Check the generator description as well as the rows.  The detailed
    description lives in the containing subsection, with a short caption.
    Keeping the check local to that subsection prevents an unrelated number
    elsewhere in the manuscript from hiding a changed persistence value.
    """

    section_start = tex.index(r"\label{sec:width-pooling}")
    table_start = tex.index(r"\begin{table}", section_start)
    description = visible_tex(
        tex[section_start:table_start] + table_block(tex, "width-pooling")
    )
    regimes_full = payload["regimes"]
    for key, decimals in (("width_persistence", 3), ("rho", 2)):
        for name, cell in regimes_full.items():  # type: ignore[union-attr]
            value = float(cell[key])
            text = f"{value:.{decimals}f}".rstrip("0").rstrip(".") or "0"
            rendered = text[1:] if text.startswith("0.") else text
            if f"${rendered}$" not in description:
                raise AssertionError(
                    f"sec:width-pooling description is missing {key} {rendered} "
                    f"for regime {name}"
                )
    for name, cell in regimes_full.items():  # type: ignore[union-attr]
        latent = float(cell["rho"]) ** 2
        text = f"{latent:.4f}".rstrip("0").rstrip(".") or "0"
        rendered = text[1:] if text.startswith("0.") else text
        if f"${rendered}$" not in description:
            raise AssertionError(
                f"sec:width-pooling description is missing the latent correlation "
                f"{rendered} (rho^2, not rho) for regime {name}"
            )

    block = table_body(tex, "width-pooling")
    mapping = (
        (r"Bayes, shared $\Delta$ (equal tail)", "bayes_shared"),
        (r"\quad $+$ tail width, frozen ladder", "bayes_width"),
        (r"\quad $+$ tail width, pooled", "bayes_pooled_width"),
        (r"$h^\star_{\mathrm{gum},.005}$", "h_gum_star_0.005"),
        (r"$h^\star_{\mathrm{gum},.1}$", "h_gum_star_0.1"),
        (r"$h^\star_{\mathrm{gum},.01}$", "h_gum_star_0.01"),
        (r"$h_{\mathrm{ars}}$", "h_ars"),
        (r"$h_{\log}$", "h_log"),
        (r"$h_{\mathrm{ind},1/e}$", "h_ind_1_over_e"),
    )
    type2 = payload["type2_error"]
    regimes = list(payload["regimes"])  # type: ignore[arg-type]
    n = int(payload["config"]["n_evaluation_alternative"])  # type: ignore[index]
    checked = 0
    for occurrence, horizon in enumerate(("100", "300", "700")):
        for marker, method in mapping:
            expected = [
                float(type2[horizon][method][g]) for g in regimes  # type: ignore[index]
            ]
            expected_se = [
                math.sqrt(max(v * (1.0 - v), 1e-12) / n) for v in expected
            ]
            # The max-regret cell is a maximum of differences, so its error is
            # the stored paired bootstrap rather than a binomial plug-in.  It
            # is appended, not checked separately, because assert_row compares
            # every estimate in the row: a column added to the table without a
            # number behind it would fail here rather than pass silently.
            regret = payload["max_regret"]["by_horizon"][horizon][method]  # type: ignore[index]
            expected.append(float(regret["max_regret"]))
            expected_se.append(float(regret["mc_se"]))
            assert_row(block, marker, expected, expected_se, occurrence=occurrence)
            checked += 1
    return checked + 1

def check_contamination(tex: str, rows: list[dict[str, str]]) -> int:
    block = table_block(tex, "contamination")
    rhos = (0.0, 0.25, 0.4, 0.6)
    mapping = (
        ("Bayes, robust shared &", "gumbel", "bayes_shared_robust", 0),
        ("Bayes, robust shared $+$ tail shape", "gumbel", "bayes_shared_dirichlet_robust", 0),
        ("Bayes, robust shared $+$ union tail", "gumbel", "bayes_shared_uniontail_robust", 0),
        (r"Bayes, shared $\Delta$ $+$ union tail ($\rho=0$)", "gumbel", "bayes_shared_uniontail_clean", 0),
        (r"Bayes, assumes $\rho=0$ &", "gumbel", "bayes_shared_clean", 0),
        (r"Bayes, shared $\Delta$ $+$ tail shape ($\rho=0$)", "gumbel", "bayes_shared_dirichlet_clean", 0),
        (r"$h^\star_{\mathrm{gum},.1}$", "gumbel", "h_gum_star_0.1", 0),
        (r"$h^\star_{\mathrm{gum},.01}$", "gumbel", "h_gum_star_0.01", 0),
        (r"$h^\star_{\mathrm{gum},.005}$", "gumbel", "h_gum_star_0.005", 1),
        (r"$h_{\mathrm{ars}}$", "gumbel", "h_ars", 0),
        (r"$h_{\log}$", "gumbel", "h_log", 0),
        (r"$h_{\mathrm{ind},1/e}$", "gumbel", "h_ind_1_over_e", 0),
        ("Bayes, robust shared &", "inverse", "bayes_shared_robust", 1),
        (r"Bayes, assumes $\rho=0$ &", "inverse", "bayes_shared_clean", 1),
        (r"$h^\star_{\mathrm{dif},.01}$", "inverse", "h_dif_star_0.01", 1),
    )
    for marker, scheme, method, occurrence in mapping:
        if GUMBEL_ONLY and scheme == "inverse":
            continue
        expected = [
            contamination_cell(rows, scheme=scheme, method=method, rho_true=rho)
            for rho in rhos
        ]
        standard_errors = [
            contamination_cell(
                rows,
                scheme=scheme,
                method=method,
                rho_true=rho,
                field="type_ii_mc_se",
            )
            for rho in rhos
        ]
        assert_row(
            block,
            marker,
            expected,
            standard_errors,
            occurrence=occurrence,
        )
    return len(mapping)


def check_anytime(tex: str, rows: list[dict[str, str]]) -> int:
    block = table_block(tex, "anytime")
    mapping = (
        ("Gumbel & tokenwise, spike", "gumbel", "bayes_tokenwise"),
        ("Gumbel & tokenwise $+$ tail shape", "gumbel", "bayes_tokenwise_dirichlet"),
        (r"Gumbel & shared $\Delta$, spike", "gumbel", "bayes_shared"),
        (r"Gumbel & shared $\Delta$ $+$ tail shape", "gumbel", "bayes_shared_dirichlet"),
        # Absent until it was found to be printing .0108 against the artifact's
        # .0154: the only row of this table nothing read.
        (r"Gumbel & shared $\Delta$ $+$ union tail", "gumbel", "bayes_shared_uniontail"),
        (r"Inverse & tokenwise $\Delta$", "inverse", "bayes_tokenwise"),
        (r"Inverse & shared $\Delta$", "inverse", "bayes_shared"),
    )
    for marker, scheme, method in mapping:
        if GUMBEL_ONLY and scheme == "inverse":
            continue
        if GUMBEL_ONLY:
            # The single-scheme anytime table omits its redundant scheme column.
            marker = marker.removeprefix("Gumbel & ")
        expected = [
            clean_cell(
                rows,
                scenario="shared_delta_equal_tail_sensitivity",
                scheme=scheme,
                method=method,
                horizon=700,
                field=field,
                decision_rule="anytime_bf_ge_1_over_alpha",
            )
            for field in ("type1_error", "type2_error")
        ]
        standard_errors = [
            clean_cell(
                rows,
                scenario="shared_delta_equal_tail_sensitivity",
                scheme=scheme,
                method=method,
                horizon=700,
                field=field,
                decision_rule="anytime_bf_ge_1_over_alpha",
            )
            for field in ("type1_mc_se", "type2_mc_se")
        ]
        assert_row(block, marker, expected, standard_errors)
    return len(mapping)


BOLD_CELL = re.compile(r"\\(best)?mcse\{(" + CELL_NUMBER + r")\}\{([^}]*)\}")

# Which end of a column each table's boldface marks, from its caption.  Kept
# here as well as in sync_manuscript_tables so that a hand edit is caught even
# when the synchroniser is not run.
BOLD_DIRECTION: dict[str, str] = {
    "shared": "min",
    "regret": "min",
    "tails": "min",
    "tail-widths": "min",
    "deficit-pooling": "min",
    "width-pooling": "min",
    "released-output": "min",
    "temperature-matched": "min",
    "union-weight": "max",
    "pooled-real": "max",
    "deficit-support": "max",
}

# Columns a table deliberately leaves unemphasised, by index among its
# \mcse cells.  Declared rather than inferred: "this column has no boldface,
# so skip it" is vacuous -- deleting the one correct marker would then pass.
BOLD_SKIP: dict[str, frozenset[int]] = {
    # Only the Type II columns are marked; the Type I columns report size.
    "released-output": frozenset({0, 2}),
}


def check_tail_type_i_claim(tex: str, payload: dict[str, object]) -> int:
    """The realized Type I range and cell count in the tail-sweep prose."""

    cells = [c for c in payload["cells"] if "type1_error" in c]  # type: ignore
    rules = {c["rule"] for c in cells}
    horizons = {c["horizon"] for c in cells}
    n = len(rules) * len(horizons)
    lo = min(float(c["type1_error"]) for c in cells)
    hi = max(float(c["type1_error"]) for c in cells)
    claim = (f"Realized Type~I error over the {n} rule-by-horizon cells lies in "
             f"$[{lo:.4f}".lstrip("0").replace("$[0", "$[") + f",{hi:.4f}$".lstrip("0"))
    want = (f"Realized Type~I error over the {n} rule-by-horizon cells lies in "
            f"$[{f'{lo:.4f}'[1:]},{f'{hi:.4f}'[1:]}]$.")
    if want not in tex:
        raise AssertionError(f"the tail-sweep prose should say {want!r}")
    return 1


def check_baseline_sensitivity_claim(tex: str, payload: dict[str, object]) -> int:
    """The with/without-tail-prior regret readings against the sweep.

    This paragraph quoted .0012 at n=300 where the table two lines above said
    .0004, and .0018 at n=700 where excluding the tail-prior rule leaves the
    shared rule minimal in every regime, i.e. zero.  Nothing read it.
    """

    import regime_sweep as rs

    t2 = payload["type2_error"]  # type: ignore[index]
    labels = [g["label"] for g in payload["regimes"]]  # type: ignore[index]
    bench = [r["name"] for r in payload["rules"]  # type: ignore[index]
             if rs.is_benchmark_rule(r["name"])]

    def max_regret(horizon: int, rule: str, drop: tuple[str, ...] = ()) -> float:
        pool = [n for n in bench if n not in drop]
        return max(t2[str(horizon)][rule][g] - min(t2[str(horizon)][n][g] for n in pool)
                   for g in labels)

    dirichlet = ("bayes_shared_dirichlet",)
    with_tail = max_regret(700, "bayes_shared")
    without = max_regret(700, "bayes_shared", dirichlet)
    if abs(without) > 1e-12:
        raise AssertionError(
            "excluding the tail-prior rule no longer leaves the shared rule "
            f"minimal at n=700 (regret {without:.4f}); the prose says zero"
        )
    shown = f"maximum regret is ${with_tail:.4f}".replace("$0.", "$.")
    if shown not in tex:
        raise AssertionError(f"the baseline-sensitivity prose should say {shown!r}")
    for horizon in (100, 300):
        a = max_regret(horizon, "bayes_shared")
        b = max_regret(horizon, "bayes_shared", dirichlet)
        if abs(a - b) > 1e-12:
            raise AssertionError(
                f"the two readings no longer agree at n={horizon}"
            )
    pair = (f"at ${max_regret(100, 'bayes_shared'):.4f}$ and "
            f"${max_regret(300, 'bayes_shared'):.4f}$.").replace("$0.", "$.")
    if pair not in tex:
        raise AssertionError(f"the baseline-sensitivity prose should say {pair!r}")
    return 3


def check_holdout_claims(tex: str) -> int:
    """The holdout paragraph against the holdout artifact.

    The holdout exists to answer a selection worry, so its numbers are the
    ones a reader will check hardest.  Every count and range the paragraph
    states is recomputed here from the stored tests rather than trusted.
    """

    # Source-line wrapping is editorial; only visible wording is evidence.
    tex = re.sub(r"\s+", " ", visible_tex(tex))
    path = (ROOT / "results/bayesian_paper_benchmark/real_model"
            / "temperature_matched" / "delong_tests_holdout.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("prompt_start") != 500:
        raise AssertionError("the holdout artifact must exclude 500 prompts")
    if {b["n"] for b in payload["cells"].values()} != {2000}:
        raise AssertionError("every holdout cell should carry 2000 documents")

    cells = sorted(payload["cells"])
    words = {6: "six", 7: "seven", 8: "eight", 2: "two", 3: "three", 5: "five"}

    def summarise(name):
        tests = [payload["cells"][c]["tests"][name] for c in cells]
        d = [t["dauc"] for t in tests]
        return (sum(v > 0 for v in d), sum(t["p_holm"] < .05 for t in tests),
                min(d), max(d), max(t["p_holm"] for t in tests))

    def rng(lo, hi):
        return f"${lo:.4f}$ to ${hi:.4f}$".replace("$0.", "$.")

    checked = 0
    width = ("bayes_uniontail_w0.5_vs_bayes_shared",
             "bayes_uniontail_w0_vs_bayes_shared",
             "bayes_uniontail_w0_vs_bayes_uniontail_w1")
    for name, (phrase, suffix) in zip(width, (
            (r"On the holdout, the union tail has higher AUC than the shared-$\Delta$ "
             "rule in all eight cells, with differences from ",
             ", and each contrast is significant."),
            ("The width-only prior has higher AUC in all eight cells, "
             "with differences from ", ", and $w=0$"),
            ("$w=0$ has higher AUC than $w=1$ in all eight cells, "
             "with differences from ", "; all sixteen contrasts are significant."))):
        pos, sig, lo, hi, _ = summarise(name)
        if (pos, sig) != (8, 8):
            raise AssertionError(f"{name}: holdout is {pos}/8 positive, {sig}/8 significant")
        want = phrase + rng(lo, hi) + suffix
        if want not in tex:
            raise AssertionError(f"the holdout paragraph should say {want!r}")
        checked += 1

    worst = max(summarise(n)[4] for n in width)
    # The prose states an upper bound, so this rounds the largest adjusted p
    # UP to two significant figures.  Rounding to nearest could put the bound
    # below a p-value it claims to cover.
    exponent = math.floor(math.log10(worst))
    mantissa = math.ceil(worst / 10.0 ** exponent * 10.0) / 10.0
    bound = f"${mantissa:.1f}\\times10^{{{exponent}}}$"
    if f"below {bound}" not in tex:
        raise AssertionError(f"the holdout paragraph should bound p by {bound!r}")
    checked += 1

    full = json.loads(
        path.with_name("delong_tests_bootstrap.json").read_text(encoding="utf-8")
    )
    paired = [
        (full["cells"][cell]["tests"][name], payload["cells"][cell]["tests"][name])
        for cell in cells for name in width
    ]
    increases = sum(holdout["dauc"] > original["dauc"] for original, holdout in paired)
    decreases = sum(holdout["dauc"] < original["dauc"] for original, holdout in paired)
    if (increases, decreases) != (20, 4):
        raise AssertionError(
            f"holdout effect changes are {increases} increases and {decreases} decreases; "
            "update the narrative counts"
        )
    want = "twenty point estimates increase and four decrease slightly"
    if want not in tex or "does not shrink the effect" in tex:
        raise AssertionError(f"the holdout effect-change prose should say {want!r}")
    checked += 1

    if not all(
        holdout["ci_hi"] - holdout["ci_lo"] > original["ci_hi"] - original["ci_lo"]
        for original, holdout in paired
    ) or "all twenty-four contrasts remain significant despite wider intervals" not in tex:
        raise AssertionError("the holdout interval-width prose must match all 24 contrasts")
    checked += 1

    pos, sig, _, _, _ = summarise("bayes_uniontail_w1_vs_bayes_shared")
    want = (f"shape-only prior has a positive contrast in {words[pos]} rather than "
            f"seven cells and a significant contrast in {words[sig]} rather than three")
    if want not in tex:
        raise AssertionError(f"the holdout paragraph should say {want!r}")
    checked += 1

    pos, sig, lo, hi, _ = summarise("h_ars_vs_bayes_uniontail_w0.5")
    want = ("versus union-tail contrast is positive in all "
            f"{words[pos]} holdout cells rather than seven, remains significant "
            f"in {words[sig]}, and ranges from {rng(lo, hi)}")
    if want not in tex:
        raise AssertionError(f"the holdout paragraph should say {want!r}")
    checked += 1
    return checked


def check_displayed_rule_counts(tex: str) -> int:
    """Prose counts of displayed rules against the tables themselves.

    "fifteen displayed rules" outlived two edits: adding the reference scores
    and then removing the diagnostics.  Counting the rows is cheap.
    """

    words = {10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
             14: "fourteen", 15: "fifteen", 16: "sixteen"}
    checked = 0
    for label, phrase in (("regret", "of the %s displayed rules at $n=100$"),
                          ("tails", "Among the %s displayed rules")):
        block = table_body(tex, label)
        rows = [l for l in block.split("\n") if BOLD_CELL.search(l)]
        per_panel = len(rows) // 3
        want = phrase % words[per_panel]
        if want not in tex:
            raise AssertionError(
                f"tab:{label} displays {per_panel} rules per panel; "
                f"the prose should say {want!r}"
            )
        checked += 1
    released_rows = sum(
        bool(ESTIMATE_WITH_MC_SE.search(line))
        for line in table_body(tex, "released-output").splitlines()
    )
    want = f"Across the {words[released_rows]} Gumbel methods tabulated here"
    if want not in tex:
        raise AssertionError(
            f"tab:released-output displays {released_rows} rules; "
            f"the prose should say {want!r}"
        )
    checked += 1
    return checked


def check_paper_family_size(tex: str, payload: dict[str, object]) -> int:
    """The declared Holm family matches the comparators the tables display.

    The family listed five reference scores while the tables showed six: the
    ".1" tuning went untabulated when the family was written and was corrected
    across as though it belonged to a different one.  A family smaller than
    what a reader scans understates the multiplicity.
    """

    import benchmark_paper_experiment as benchmark

    fam = [
        c for c in payload["comparisons"]  # type: ignore[index]
        if "paper" in str(c.get("holm_family", ""))
        and c.get("scenario") == "shared_delta_equal_tail_sensitivity"
        and c.get("scheme") == "gumbel"
    ]
    scores = {c["comparator"] for c in fam}
    if scores != set(benchmark.PAPER_REFERENCE_SCORES):
        raise AssertionError(
            f"artifact family {sorted(scores)} differs from the declared "
            f"{sorted(benchmark.PAPER_REFERENCE_SCORES)}"
        )
    words = {15: "15", 18: "18", 21: "21"}
    n = len(fam)
    claim = f"Holm correction over all {words.get(n, n)} Bayes-versus-reference-score"
    if claim not in tex:
        raise AssertionError(f"the manuscript should say {claim!r}")
    return 1


def check_tail_width_fit(tex: str, payload: dict[str, object]) -> int:
    """The released-width prose against the reduction it says it uses.

    The estimator moved to one position per PRF address but the
    goodness-of-fit sentence kept quoting the pivot-deduplicated run, which
    reverses which model is nominally below .05.  Reported numbers are checked
    against the address-level block only.
    """

    tex = re.sub(r"\s+", " ", visible_tex(tex))
    # Scope the search to the sentence that makes the claim.  Searching the
    # whole document lets an unrelated ".025" elsewhere satisfy a check for
    # ".0246", which a negative control caught.
    anchor = "On the same first-use sample"
    if anchor not in tex:
        raise AssertionError(
            "the released-width paragraph no longer says which reduction it uses"
        )
    sentence = tex[tex.index(anchor):tex.index(". Both fitted-width")]

    models = payload["models"]  # type: ignore[index]
    checked = 0
    for spec, label in (("1p3B", "OPT-1.3B"), ("2p7B", "Sheared-LLaMA-2.7B")):
        block = models[spec]["distinct_addresses"]  # type: ignore[index]
        fit = block["goodness_of_fit"]["fitted"]
        wide = block["goodness_of_fit"]["assumed_full_width"]
        for value in (fit["ks_statistic"], wide["ks_statistic"]):
            # The prose rounds to three or four decimals and writes the value
            # after "D=", so match the numeral itself with digit boundaries.
            shown = {f"{float(value):.{d}f}".lstrip("0") for d in (3, 4)}
            if not any(re.search(re.escape(f) + r"(?![0-9])", sentence)
                       for f in shown):
                raise AssertionError(
                    f"tail-width KS statistic {float(value):.4f} for {label} missing "
                    f"(looked for {sorted(shown)}); the prose may still quote the "
                    "pivot-deduplicated reduction"
                )
            checked += 1
        if int(block["estimate"]) not in (1, 2):
            raise AssertionError("address-level width left the one-or-two range")
        checked += 1
        # The share quoted in the prose must come from a bootstrap that
        # resamples DOCUMENTS.  Resampling first-use positions independently
        # prices the shared-pseudorandom-vector dependence but not the
        # shared-continuation dependence, and understates the spread.
        boot = models[spec]["address_cluster_bootstrap"]  # type: ignore[index]
        resample = str(boot.get("resample", ""))
        if "document-cluster" not in resample or "first-use positions" not in resample:
            raise AssertionError(
                "the reported bootstrap is no longer a document-cluster resample "
                "of the first-use positions"
            )
        if not isinstance(boot.get("n_documents"), int) or boot["n_documents"] < 2:
            raise AssertionError("the bootstrap records no document count")
        checked += 1
        share = boot["argmax_share"][str(int(block["estimate"]))]
        pct = f"{round(share * 100)}\\%"
        if pct not in tex:
            raise AssertionError(
                f"bootstrap share {pct} for {label} missing from the text"
            )
        checked += 1

    counts = [
        int(models[spec]["address_cluster_bootstrap"]["n_documents"])
        for spec in ("1p3B", "2p7B")
    ]
    count_claim = (
        f"resamples {counts[0]} OPT-1.3B documents and "
        f"{counts[1]} Sheared-LLaMA-2.7B documents"
    )
    if count_claim not in tex:
        raise AssertionError(
            "bootstrap document counts must be attached to the correct models: "
            + count_claim
        )
    checked += 1

    # The identifiability check must be reported at the sample sizes the data
    # actually supply.  It used to run only at 8,000 simulated positions, where
    # recovery is uniform, and the prose read that as identification here.
    recovery = payload["identifiability_check"]["by_sample_size"]  # type: ignore[index]
    observed = {str(int(models[m]["distinct_addresses"]["n"]))  # type: ignore[index]
                for m in ("1p3B", "2p7B")}
    missing = observed - set(recovery)
    if missing:
        raise AssertionError(
            f"no identifiability check at the observed sample size(s) {sorted(missing)}"
        )
    for size in observed:
        block = recovery[size]
        if block["replicates"] < 100:
            raise AssertionError(
                f"identifiability at n={size} rests on {block['replicates']} "
                "replicates; a single draw cannot support a reliability claim"
            )
        if block["recovered_always"]:
            raise AssertionError(
                f"recovery at n={size} is now uniform; the softened wording in "
                "the manuscript would understate what the check supports"
            )
        checked += 1
    if "recovered the generating width in every replicate at all five widths" not in tex:
        raise AssertionError(
            "the manuscript should still say the 8,000-position check is the "
            "uniform one, so the contrast with the observed sizes is explicit"
        )
    checked += 1
    model_sizes = [
        str(int(models[spec]["distinct_addresses"]["n"]))
        for spec in ("1p3B", "2p7B")
    ]
    rates = {
        size: {int(row["true_live_tail"]): float(row["recovery_rate"])
               for row in recovery[size]["by_truth"]}
        for size in model_sizes
    }
    percentage = lambda value: f"{100 * value:g}" + r"\%"
    recovery_claim = (
        r"$J=2$ in $" + percentage(rates[model_sizes[0]][2])
        + r"$ and $" + percentage(rates[model_sizes[1]][2])
        + "$, respectively"
    )
    if recovery_claim not in tex:
        raise AssertionError(
            "observed-size J=2 recovery is missing or rounded incorrectly: "
            + recovery_claim
        )
    checked += 1
    for width in (4, 16):
        rate = min(rates[size][width] for size in model_sizes)
        forms = {percentage(rate), f"{100 * rate:.1f}" + r"\%"}
        if not any(
            "$" + form + f"$ at $J={width}$" in tex for form in forms
        ):
            raise AssertionError(
                f"worst observed-size recovery for J={width} is missing"
            )
        checked += 1
    return checked


def check_test_count(tex: str) -> int:
    """The manuscript's test count against the suite it describes.

    It has been stale in three different places at once: 396 in the
    manuscript, 398 in both READMEs, 406 actually running.  Counting is
    cheap, so it is counted rather than remembered.
    """

    import unittest

    root = Path(__file__).resolve().parent
    suite = unittest.defaultTestLoader.discover(str(root), pattern="test_*.py")
    total = suite.countTestCases()
    if f"The {total} automated tests cover" not in tex:
        raise AssertionError(
            f"manuscript should say {total} automated tests"
        )
    for readme in (root / "README.md", root.parent / "README.md"):
        text = readme.read_text(encoding="utf-8")
        if f"comprises {total} tests" not in text:
            raise AssertionError(f"{readme.name} should say {total} tests")
    return 1


def check_figure_rule_counts(tex: str) -> int:
    """Figure 6's caption counts the rules it draws, against the plot spec.

    The caption said five, including three reference scores; six are plotted,
    two Bayesian and four reference.  A caption that counts curves is exactly
    the sort of claim that drifts when a curve is added.
    """

    import plot_temperature_prefix_curves as ptc

    # The manuscript is the Gumbel-only build, which drops the Tr-GoF curve
    # (make_gumbel_only_figures), so the caption counts what that build draws.
    drawn = [name for name, _, _, _ in ptc.RULE_STYLE if name != "trgof_s2"]
    bayesian = [name for name in drawn if name.startswith("bayes")]
    words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
             8: "eight", 9: "nine", 10: "ten",
             20: "twenty", 21: "twenty-one"}
    wanted = (
        f"{words[len(drawn)].capitalize()} of the "
        f"{words[len(ptc.COMPUTED_RULES)]} scored rules are drawn"
    )
    if wanted not in tex:
        raise AssertionError(
            f"Figure 6 caption should say {wanted!r}"
        )
    reference = f"and {words[len(drawn) - len(bayesian)]} reference scores"
    if reference not in tex:
        raise AssertionError(f"Figure 6 caption should say {reference!r}")
    return 2


def boldface_plan(block: str, name: str) -> tuple[list[tuple[re.Match[str], bool, str]], int]:
    """Expected emphasis with source-relative spans, shared with the editor.

    The deficit-support table compares reference scores only; its prior pairs
    are descriptive.  The pooled-real difference row is likewise not an AUC
    candidate.  Policies do not depend on whether any marker currently exists.
    """
    pick = min if BOLD_DIRECTION[name] == "min" else max
    targets: list[tuple[re.Match[str], bool, str]] = []
    current: list[list[re.Match[str]]] = []
    checked = 0
    active = name != "deficit-support"
    seen_reference = False

    def flush() -> None:
        nonlocal checked
        if not current:
            return
        for column in range(max(len(row) for row in current)):
            hits = [row[column] for row in current if len(row) > column]
            excluded = column in BOLD_SKIP.get(name, frozenset())
            target = pick(float(hit.group(2)) for hit in hits)
            for hit in hits:
                want = (not excluded and len(hits) > 1
                        and abs(float(hit.group(2)) - target) < 1e-12)
                targets.append((hit, want, f"column {column} boldface"))
            if not excluded and len(hits) > 1:
                checked += 1
        current.clear()

    offset = block.index(r"\midrule")
    for raw_line in block[offset:].splitlines(keepends=True):
        # Removing an inline comment preserves every preceding source offset.
        line = visible_tex(raw_line)
        hits = list(BOLD_CELL.finditer(block, offset, offset + len(line.rstrip("\n"))))
        boundary = r"\midrule" in line or (
            "\\multicolumn" in line and ("\\textit" in line or "\\emph" in line)
        )
        if boundary:
            flush()
            if name == "deficit-support":
                active = "Reference scores" in line
                seen_reference = seen_reference or active
        elif name == "pooled-real" and re.match(r"\s*Difference\s*&", line):
            targets.extend((hit, False, "Difference row is descriptive, not an AUC maximum")
                           for hit in hits)
        elif not active:
            targets.extend((hit, False, "prior-sensitivity row is deliberately unemphasised")
                           for hit in hits)
        elif hits:
            current.append(hits)
        offset += len(raw_line)
    flush()
    if name == "deficit-support" and not seen_reference:
        raise AssertionError("tab:deficit-support is missing its Reference scores block")
    return targets, checked


def check_boldface(tex: str) -> int:
    """Every \\bestmcse marks its panel column's extremum, ties included.

    Nothing checked this, and the 224 row checks could not: they compare cell
    values, and boldface is not a value.  Table~14 spent the whole revision
    bolding the *worst* mixing weight in all eight columns, because the
    synchroniser bolds a minimum and that table reports areas under the curve.
    This reads the printed table only -- which is the right instrument for a
    presentation error -- and is deliberately blind to the artifacts.
    """

    checked = 0
    for match in re.finditer(r"\\label\{tab:([a-z-]+)\}", tex):
        start = tex.rfind(r"\begin{table}", 0, match.start())
        stop = tex.index(r"\end{table}", match.start())
        block = tex[start:stop]
        name = match.group(1)
        # Every declared current policy remains required when all its markers
        # disappear.  Unbold archived tables retain their historical contract.
        required = GUMBEL_ONLY and name in BOLD_DIRECTION
        if "bestmcse" not in block and not required:
            continue
        if name not in BOLD_DIRECTION:
            raise AssertionError(
                f"tab:{name} carries boldface but declares no direction"
            )
        targets, count = boldface_plan(block, name)
        for hit, want, reason in targets:
            if bool(hit.group(1)) != want:
                raise AssertionError(
                    f"tab:{name} {reason}: {hit.group(0)} should be "
                    f"{'bold' if want else 'unemphasised'}"
                )
        checked += count
    return checked


def check_pivot_repetition(tex: str, payload: dict[str, object]) -> int:
    """The released-output mechanism sentence against pivot_repetition.json.

    These were prose numbers with no artifact behind them, and one of them was
    wrong: the distinct-pivot counts are 76 and 86, not "76--87".  The group
    counts are here because the sentence now makes a mechanism claim -- the
    pair determines the pivot, the context token alone does not -- and a claim
    that is checkable should be checked.
    """

    models = payload["models"]  # type: ignore[index]
    checked = 0
    for spec in ("1p3B", "2p7B"):
        cell = models[spec]  # type: ignore[index]
        for value in (cell["pair_groups"], cell["seed_groups"],
                      cell["seed_groups_with_unequal_pivots"]):
            rendered = f"{int(value):,}".replace(",", "{,}")
            if f"${rendered}$" not in tex:
                raise AssertionError(
                    f"pivot repetition group count {rendered} missing from the text"
                )
            checked += 1
        if int(cell["pair_groups_with_unequal_pivots"]) != 0:
            raise AssertionError(
                "the pair no longer determines the pivot; the mechanism "
                "sentence in the released-output subsection is now false"
            )
        checked += 1
        distinct = round(float(cell["watermarked_distinct_pivots_mean"]))
        if f"${distinct}$" not in tex:
            raise AssertionError(
                f"distinct watermarked pivot count {distinct} missing from the text"
            )
        checked += 1
    null = {round(float(models[s]["unwatermarked_distinct_pivots_mean"]))  # type: ignore[index]
            for s in ("1p3B", "2p7B")}
    if len(null) != 1:
        raise AssertionError(f"unwatermarked counts disagree across models: {null}")
    if f"${null.pop()}$" not in tex:
        raise AssertionError("unwatermarked distinct pivot count missing from the text")
    return checked + 1


def check_released_output(tex: str, rows: list[dict[str, str]]) -> int:
    block = table_block(tex, "released-output")
    mapping = (
        (r"$h_{\mathrm{ars}}$", "gumbel", "h_ars", 0),
        (r"$h_{\log}$", "gumbel", "h_log", 0),
        (r"$h_{\mathrm{ind},1/e}$", "gumbel", "h_ind_1_over_e", 0),
        (r"$h^\star_{\mathrm{gum},.1}$", "gumbel", "h_gum_star_0.1", 0),
        (r"$h^\star_{\mathrm{gum},.01}$", "gumbel", "h_gum_star_0.01", 0),
        (r"$h^\star_{\mathrm{gum},.005}$", "gumbel", "h_gum_star_0.005", 0),
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
    for marker, scheme, method, occurrence in mapping:
        if GUMBEL_ONLY and scheme == "inverse":
            continue
        expected: list[float] = []
        standard_errors: list[float] = []
        for model in ("1p3B", "2p7B"):
            for sample in (
                "released_raw_empirical_null",
                "released_watermarked",
            ):
                expected.append(
                    real_data_cell(
                        rows,
                        model=model,
                        scheme=scheme,
                        method=method,
                        sample=sample,
                    )
                )
                standard_errors.append(
                    real_data_cell(
                        rows,
                        model=model,
                        scheme=scheme,
                        method=method,
                        sample=sample,
                        field="mcse",
                    )
                )
        assert_row(block, marker, expected, standard_errors, occurrence=occurrence)
    return len(mapping)


def _select_tex() -> None:
    """``--tex legacy`` checks the archived two-pivot manuscript instead."""
    global TEX_PATH, GUMBEL_ONLY
    import sys
    if "--tex" in sys.argv:
        which = sys.argv[sys.argv.index("--tex") + 1]
        if which == "legacy":
            TEX_PATH, GUMBEL_ONLY = LEGACY_TEX_PATH, False
        else:
            TEX_PATH, GUMBEL_ONLY = Path(which), False


def check_manuscript(tex: str) -> int:
    """Validate an in-memory candidate without changing any file or CLI state."""
    tex = visible_tex(tex)
    clean = read_csv("benchmark_results.csv")
    contamination = read_csv("contamination_results.csv")
    checked = 0
    checked += check_shared_and_family(tex, clean, read_json("benchmark_summary.json"))
    checked += check_regret(tex, read_json("regime_sweep.json"))
    checked += check_tails(tex, read_json("tail_regime_sweep.json"))
    checked += check_tails(
        tex, read_json("tail_regime_sweep.json"),
        table="tail-widths", laws=("W1", "W2", "W3", "W4"), arm="width",
    )
    bootstrap_cells = read_json(
        "real_model/temperature_matched/delong_tests_bootstrap.json"
    )["cells"]
    checked += check_pooled_real(
        tex,
        read_csv("real_model/temperature_matched/temperature_prefix_curves.csv"),
        bootstrap_cells,
    )
    if GUMBEL_ONLY:
        checked += check_pooled_auc_mean(tex, bootstrap_cells)
    checked += check_union_weight(
        tex,
        read_json(
            "real_model/temperature_matched/delong_tests_bootstrap.json"
        ),
    )
    checked += check_deficit_support(
        tex,
        read_json(
            "real_model/temperature_matched/delong_tests_wideprior.json"
        ),
        read_csv("real_model/temperature_matched/temperature_prefix_curves.csv"),
        bootstrap_cells,
    )
    checked += check_temperature_matched(
        tex,
        read_csv("real_model/temperature_matched/temperature_prefix_curves.csv"),
    )
    checked += check_deficit_regime(
        tex,
        read_json(
            "real_model/temperature_matched/empirical_deficits.json"
        ),
    )
    checked += check_deficit_pooling(
        tex, read_json("deficit_persistence_sweep.json")
    )
    checked += check_width_pooling(
        tex, read_json("width_persistence_sweep.json")
    )
    checked += check_contamination(tex, contamination)
    checked += check_anytime(tex, clean)
    checked += check_released_output(
        tex, read_csv("real_model/real_data_results.csv")
    )
    checked += check_pivot_repetition(
        tex, read_json("real_model/pivot_repetition.json")
    )
    matched_cells = [f"{m}_T{t}" for m in ("1p3B", "2p7B")
                     for t in ("0.2", "0.3", "0.4", "0.5")]
    for label in ("union-weight", "deficit-support"):
        checked += check_auc_reference_block(
            tex, label, bootstrap_cells, matched_cells
        )
    checked += check_tail_type_i_claim(
        tex, read_json("tail_regime_sweep.json")
    )
    checked += check_displayed_rule_counts(tex)
    checked += check_holdout_claims(tex)
    checked += check_baseline_sensitivity_claim(tex, read_json("regime_sweep.json"))
    checked += check_paper_family_size(
        tex, read_json("paired_comparisons.json")
    )
    checked += check_boldface(tex)
    checked += check_figure_rule_counts(tex)
    checked += check_test_count(tex)
    checked += check_tail_width_fit(
        tex, read_json("tail_width_estimate.json")
    )
    return checked


def main() -> None:
    _select_tex()
    checked = check_manuscript(TEX_PATH.read_text(encoding="utf-8"))
    print(f"manuscript table check passed: {checked} rows match generated artifacts")


if __name__ == "__main__":
    main()
