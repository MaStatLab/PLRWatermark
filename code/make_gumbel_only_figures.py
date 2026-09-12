#!/usr/bin/env python3
"""Regenerate the manuscript figures for the Gumbel-only version.

The Gumbel-only manuscript reuses every artifact of the full study, so only the
figures that draw an inverse-transform panel or a Tr-GoF curve need replacing.
Those variants are written to ``gumbel_only/`` and the Gumbel-only ``.tex``
lists that directory first in its ``\\graphicspath``, so it picks up the
overrides while every other figure falls through to the shared directory.  The
originals are never touched.

No experiment is re-run: this replots stored result rows.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "bayesian_paper_benchmark"
OUT = RESULTS / "gumbel_only"


def slide_figures() -> list[str]:
    import benchmark_paper_experiment as benchmark

    def load(path):
        with path.open(encoding="utf-8") as handle:
            out = []
            for row in csv.DictReader(handle):
                row["horizon"] = int(row["horizon"])
                for key in ("type2_error", "type1_error"):
                    if row.get(key) not in (None, ""):
                        row[key] = float(row[key])
                out.append(row)
            return out

    rows = load(RESULTS / "benchmark_results.csv")
    contamination = RESULTS / "contamination_results.csv"
    if contamination.exists():
        extra = load(contamination)
        # the contamination artifact names its scenario differently
        # the contamination artifact names its scenario and error columns
        # differently from the clean benchmark
        for row in extra:
            row["scenario"] = "independent_null_contamination"
            if "type2_error" not in row and "type_ii_error" in row:
                row["type2_error"] = float(row["type_ii_error"])
            if "type1_error" not in row and "type_i_error" in row:
                row["type1_error"] = float(row["type_i_error"])
        rows = rows + extra
    config = benchmark.BenchmarkConfig()
    written = []
    for scenario, stem in (
        ("shared_delta_equal_tail_sensitivity", "slide_bayes_shared_delta"),
        ("paper_text_iid_delta_equal_tail", "slide_bayes_tokenwise_delta"),
    ):
        subset = [r for r in rows if r.get("scenario") == scenario]
        if not subset:
            print(f"  skip {stem}: no rows for {scenario}")
            continue
        benchmark.make_slide_plot(
            subset, OUT / stem, scenario, config,
            schemes=("gumbel",),
            exclude_methods=(benchmark.TRGOF_METHOD,),
        )
        written.append(stem)
    return written


def contamination_figure() -> list[str]:
    """The contamination slide comes from its own module, not the clean
    benchmark: it plots Type II against the contamination rate rather than
    against text length, so it needs that module's plotting function."""
    import benchmark_contamination as contam

    csv_path = RESULTS / "contamination_results.csv"
    rows = []
    with csv_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["horizon"] = int(row["horizon"])
            row["rho_true"] = float(row["rho_true"])
            for key in ("type_i_error", "type_ii_error"):
                if row.get(key) not in (None, ""):
                    row[key] = float(row[key])
            rows.append(row)
    config = contam.ContaminationConfig()
    stem = OUT / "slide_bayes_contamination"
    contam.make_slide_plot(
        rows, config, stem,
        schemes=("gumbel",),
        exclude_methods=tuple(contam.TRGOF_METHODS),
    )
    return [stem.name]


def released_prefix_figures() -> list[str]:
    import real_data_prefix_curves as rdp

    csv_path = RESULTS / "real_model" / "prefix_curves" / "real_data_prefix_curves.csv"
    with csv_path.open(encoding="utf-8") as handle:
        rows = [dict(r) for r in csv.DictReader(handle)]
    written = []
    for analysis, stem, title, subtitle in (
        ("common_calibration", "real_data_common_calibration_prefix_curves",
         "WatermarkFramework benchmark: common calibration",
         "Recomputed common calibration at each prefix; one cutoff per method "
         "is applied unchanged to both samples."),
        ("upstream_recorded", "real_data_upstream_recorded_prefix_curves",
         "WatermarkFramework: archived reference-score curves",
         "Li et al. benchmark; Type II error = 1 - power. "
         "Reference scores only."),
    ):
        subset = [r for r in rows if r.get("analysis") == analysis]
        if not subset:
            print(f"  skip {stem}: no rows for {analysis}")
            continue
        methods = {"gumbel": rdp._plotted_common_methods(subset, "gumbel")}
        target = OUT / "real_model" / "prefix_curves" / stem
        target.parent.mkdir(parents=True, exist_ok=True)
        rdp._plot_grid(
            subset, analysis=analysis, methods_by_scheme=methods,
            output_pdf=target.with_suffix(".pdf"),
            output_png=target.with_suffix(".png"),
            title=title,
            subtitle=subtitle,
            schemes=("gumbel",),
        )
        written.append(stem)
    return written


def matched_figures() -> list[str]:
    """Redraw the temperature-matched curves without the Tr-GoF rule."""
    import plot_temperature_prefix_curves as ptc

    csv_path = RESULTS / "real_model" / "temperature_matched" / "temperature_prefix_curves.csv"
    with csv_path.open(encoding="utf-8") as handle:
        rows = [dict(r) for r in csv.DictReader(handle)]
    keep = tuple(r for r in ptc.RULE_ORDER if r != "trgof_s2")
    original = ptc.RULE_ORDER
    ptc.RULE_ORDER = keep
    written = []
    try:
        target_dir = OUT / "real_model" / "temperature_matched"
        target_dir.mkdir(parents=True, exist_ok=True)
        for model in ("1p3B", "2p7B"):
            stem = target_dir / f"prefix_curves_matched_{model}"
            ptc.plot_prefix_grid(rows, model, stem)
            written.append(stem.name)
    finally:
        ptc.RULE_ORDER = original
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    written = (slide_figures() + contamination_figure()
               + released_prefix_figures() + matched_figures())
    print(f"wrote {len(written)} Gumbel-only figures under {OUT}")
    for w in written:
        print(f"  {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
