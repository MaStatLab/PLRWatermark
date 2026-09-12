"""Legacy base-only temperature-matched diagnostic with split-half calibration.

Both arms are generated here at a common temperature, so the entropy confound
in the released pair cannot operate.  The null arm is unwatermarked text at the
same temperature; its pivots are obtained by replaying the released key/hash
functions over the generated tokens, exactly as the released-output analysis
does for the released raw continuations. This is not the canonical combined
base/extension prefix or bootstrap analysis. Historical RNG independence is
not certified by the stored arrays.
"""
import argparse
import json
from pathlib import Path
import os

# The upstream clone supplying the released key/sampler/pivot functions, and a
# scratch directory for the generated arrays.  Both are overridable so this runs
# on a machine other than the one it was written on.
SCRATCH = Path(os.environ.get("WM_SCRATCH", str(
    Path(__file__).resolve().parents[1] /
    "results/bayesian_paper_benchmark/real_model/temperature_matched"
)))
import numpy as np
import trgof, dirichlet_detector as dd
import matched_data_loader as matched
from wrong_key_null import replay_with_key

SPECS = {"1p3B": 50272, "2p7B": 32000}
HORIZONS = (25, 50, 100, 200)


def rules_for(vocab):
    d, w = dd.gauss_legendre_delta_grid(0.001, 0.5, 96)
    union = dd.UnionTailBayesGrid(delta_grid=d, delta_weights=w, tail_size=vocab - 1)
    spike = dd.DirichletBayesGrid(delta_grid=d, delta_weights=w,
                                  alpha_grid=(float("inf"),), tail_size=vocab - 1)
    def bayes(grid):
        return lambda x, h: grid.shared_paths(np.clip(x[:, :h], 1e-12, 1 - 1e-12),
                                              horizons=(h,))[0][:, 0]
    return {
        "h_ars":       lambda x, h: (-np.log1p(-np.clip(x[:, :h], 0, 1 - 1e-12))).sum(1),
        "h_log":       lambda x, h: np.log(np.clip(x[:, :h], 1e-12, 1)).sum(1),
        "trgof_s2":    lambda x, h: trgof.statistic(trgof.gumbel_p_values(x[:, :h])),
        "bayes_shared":    bayes(spike),
        "bayes_uniontail": bayes(union),
    }


def load_base_cells(scratch, model, *, allow_legacy=False):
    parts = [Path(scratch) / f"matched_{model}_{band}.npz" for band in ("lo", "hi")]
    if not all(part.is_file() for part in parts):
        raise ValueError(f"{model}: both base temperature archives are required")
    return matched.paired_cells(
        parts, matched.load_prompt_table(scratch, model), model,
        allow_legacy=allow_legacy,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, default=SCRATCH)
    parser.add_argument("--output", type=Path, help="default: scratch/temperature_matched_analysis.json (legacy base-only)")
    parser.add_argument("--allow-legacy-archives", action="store_true")
    args = parser.parse_args(argv)
    out = {}
    for model, vocab in SPECS.items():
        cells = load_base_cells(args.scratch, model, allow_legacy=args.allow_legacy_archives)
        rules = rules_for(vocab)
        for cell in cells:
            t = cell.temperature
            wm = cell.watermarked["Y"].astype(float)
            # null pivots: replay the released PRF over our unwatermarked tokens
            nullpiv, _ = replay_with_key(cell.prompts, cell.raw["tokens"], vocabulary_size=vocab,
                                         key=15_485_863, horizon=wm.shape[1])
            dw = np.mean([np.unique(np.round(r, 12)).size for r in wm])
            dn = np.mean([np.unique(np.round(r, 12)).size for r in nullpiv])
            half = nullpiv.shape[0] // 2
            cal, emp = nullpiv[:half], nullpiv[half:]
            block = {"distinct_watermarked": dw, "distinct_null": dn,
                     "mean_pivot_watermarked": float(wm.mean()),
                     "mean_pivot_null": float(nullpiv.mean()), "rules": {}}
            # the confound check that failed on the released pair
            cutrep = np.quantile(-np.array([np.unique(np.round(r,12)).size for r in nullpiv]), .95)
            block["repetition_only_type2"] = float(
                1 - np.mean(-np.array([np.unique(np.round(r,12)).size for r in wm]) > cutrep))
            for name, f in rules.items():
                block["rules"][name] = {}
                for h in HORIZONS:
                    if h > wm.shape[1]: continue
                    c = f(cal, h); cut = float(np.quantile(c, 0.95))
                    block["rules"][name][str(h)] = {
                        "type1": float(np.mean(f(emp, h) > cut)),
                        "type2": float(1 - np.mean(f(wm, h) > cut)),
                    }
            out[f"{model}_t{t}"] = block
            print(f"{model} temp {t}: distinct wm {dw:.0f} null {dn:.0f}; "
                  f"repetition-only TypeII {block['repetition_only_type2']:.3f}", flush=True)
    output = args.output or args.scratch / "temperature_matched_analysis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"cells": len(out)}))


if __name__ == "__main__":
    main()
