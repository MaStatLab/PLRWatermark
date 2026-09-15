# Predictive likelihood ratios for language model watermark detection

**Paper: [arXiv:2609.15657](https://arxiv.org/abs/2609.15657)**, submitted September 14, 2026,
by Li Ma, Department of Statistics
and Data Science Institute, University of Chicago; email: li.ma@uchicago.edu.

[Read the paper and supplement on arXiv](https://arxiv.org/abs/2609.15657).
The manuscript source, bibliography, and compiled article are not included
in this repository.

This repository contains numerical summaries, stored generation arrays, prompt tables, analysis code,
regression tests, and figure assets. The author-supplied
repository URL is [MaStatLab/PLRWatermark](https://github.com/MaStatLab/PLRWatermark/).
The original supporting code, data,
documentation, and figure assets are covered by the [MIT License](LICENSE),
subject to the scope and third-party conditions in
[LICENSING.md](LICENSING.md). This license does not apply to the paper on arXiv.

## Start here

The report's supplementary computational-materials index explains which
records support the reported analyses. The groups at the end of this README
give complete paths and the matching report's section numbers. Manuscript-specific
checks require a separately obtained source copy, as documented below.

Verify the supplied files before loading arrays or running code:

```sh
python3 scripts/verify_files.py
```

The verifier uses only Python's standard library; it checks the file hashes,
sizes, and inventory without loading pickles or NumPy objects. An independent
checksum check is `shasum -a 256 -c SHA256SUMS` on macOS or
`sha256sum -c SHA256SUMS` on Linux. Keep the verified snapshot unchanged and
use a working copy for new analyses.

## Layout and scope

- `code/`: all current Python implementation and scientific test modules, with
  pinned dependency files and the original implementation guide.
- `results/bayesian_paper_benchmark/`: production numerical summaries, CSV
  tables, rejection indicators, all eight stored temperature-matched generation
  archives, both extension-prompt tables and their metadata, and the seven
  external PDF figures used by the article. Additional production results and
  figure variants are included as well, including inverse-transform diagnostics,
  alternate union weights, and prior-sensitivity outputs; those records do not
  expand the main Gumbel-only claims. The three graphical-model figures are
  drawn directly in the manuscript source hosted on arXiv.
- `historical/6551ee5/`: the earlier Beta-generator script and result summary,
  extracted from commit `6551ee573698eb197930399b8b97c5bbf039be85`. They are not
  the current Gaussian-copula experiment or a complete historical runtime.
- `contexts/drift_report.py` and the selected
  `results/bayesian_paper_benchmark_delta_half/` JSON files: supporting
  regression-test inputs. The alternate-prior files and inverse-only inference
  summary are explicitly historical and non-inferential, not additional
  current results.
- `FILE_INDEX.json`, `SHA256SUMS`, and `UPSTREAM_SOURCES.json`: public-file
  inventory, integrity checks, and pinned third-party input attribution.
- `LICENSE` and `LICENSING.md`: MIT license and third-party scope clarification.

There are no model weights, tokenizers, C4 source records, external
WatermarkFramework checkout or original benchmark tensors, private editorial
notes, parent Git history, quick-test outputs, or unrelated experimental
directories. The current implementation retains inverse-transform methods;
including that code and its regression tests does not expand the article's
Gumbel-only claims.

## Check the reported numbers

Create a virtual environment, then install the pinned scientific dependencies:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r code/requirements.txt
python -m pip install -r code/requirements-real-data.txt
```

The following **52 self-contained tests** run without manuscript sources or
external benchmark inputs:

```sh
cd code
python -m unittest test_matched_data_loader.MatchedLoaderTests \
  test_matched_data_loader.LegacyArchiveTests.test_all_eight_actual_archives_require_opt_in_and_retain_unknown_history \
  test_generate_temperature_matched
cd ..
```

The source project's scientific suite comprised 604 tests, and its separate
manuscript audit comprised 498 checks. Those counts describe the earlier
verification, not a fresh run of this manuscript-free repository. Manuscript
checks require a separately obtained source copy; see
[REPRODUCING.md](REPRODUCING.md) for the required layout and limitations.

The full suite also requires the external benchmark tensors described below;
several stored-array tests fail if they are absent. Reference-comparison
tests are skipped if the keyed reference implementation is unavailable.
The suite does not download model weights or generate new language-model
continuations. Some tests use small synthetic fixtures; they are not
production reruns.

## External benchmark inputs

The original benchmark and keyed reference implementation belong to Xiang Li
et al.: [WatermarkFramework at the pinned commit](https://github.com/lx10077/WatermarkFramework/tree/05b7ffda9279fc9e645f38807e4a0e2dbcff4330).
`UPSTREAM_SOURCES.json` lists 14 original input files, their exact source URLs,
and SHA-256 hashes. None are silently substituted by a moving upstream branch.

To enable the external benchmark and prompt-alignment checks, download the
six commit-pinned benchmark tensors and verify them using the supplied code:

```sh
cd code
python -c "import real_data_experiment as r; r.ensure_assets(r.DEFAULT_DATA_DIR, download_missing=True)"
cd ..
```

These external files are placed in
`results/bayesian_paper_benchmark/real_model/upstream_assets/`, which is ignored
by Git. The code checks their hashes before deserialization. The stored-array
loader checks the first 500 rows of each prompt table against those original
benchmark prompts. The other eight upstream JSON records are the archived
reference-score curves and are identified separately in the input manifest.

For row-by-row comparisons with the reference implementation, obtain a clean
WatermarkFramework checkout at commit
`05b7ffda9279fc9e645f38807e4a0e2dbcff4330` and place it at
`third_party/WatermarkFramework`, or set `WATERMARK_FRAMEWORK_DIR` to its path.
Keep that checkout clean and use `PYTHONDONTWRITEBYTECODE=1` to avoid creating
Python caches in it. Install the required dependencies before running the
full suite again; reference tests must not be interpreted as passing when
they are skipped for missing inputs.

The original project's simulation comparisons also refer to five files at
that same pinned commit. They are not redistributed here and are separate
from the 14 real-data inputs in `UPSTREAM_SOURCES.json`. Their SHA-256 records
are preserved below; the result-array paths are also recorded in
`benchmark_summary.json`.

- `simulation/results_data/K1000N5000c5key23333T700Delta0.5-alpha0.05-max-result.json`:
  `4bda65fa7642877aa8ccf3ac88e9d338065ed16ceb447c172880dd367b9cb785`.
- `simulation/results_data/K1000N5000c5key23333T700Delta0.5-alpha0.05-inv-result.json`:
  `6f59b645d2d4560e7857fed571b8c46c2e6b48312d96b18cb4bab2689bfd4d86`.
- `simulation/simulation_max.py`:
  `87fbca63db1cfc234b3ce2cf7e853f123cd626f970d7ee1c65887d229a420742`.
- `simulation/simulation_inv.py`:
  `2c90fbebdc0318c0a450c9cff2517d04944325c35c03f5c4faecbefe86b0cfd4`.
- `simulation/plot.py`:
  `5fb0fd7ba0d35ecfeafb9e1f125f0a0b761f37194a0f39ea7764f8e58402cddb`.

After preparing both sets of external inputs, run from the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m unittest discover -s code -p 'test_*.py'
```

## Interpret the stored records carefully

JSON files preserve configurations, seeds, cutoffs, estimates, Monte Carlo
uncertainties, and diagnostics where recorded. CSV headers identify the
corresponding methods, regimes, horizons, and error measures. In prefix-curve
files, `common_calibration` denotes this study's recalibrated comparisons and
`upstream_recorded` denotes Li et al.'s archived reference-score curves.
Legacy names containing `released` refer to the latter benchmark, not this
study's temperature-matched generation experiment.

NPZ files contain named arrays; rejection indicators retain document-level
pairing. The article distinguishes fractional boundary-integrated rejection
rates from realized randomized decisions used in paired comparisons. All
supplied NPZ/NPY arrays are numeric and can be loaded with
`allow_pickle=False`; verify their hashes first. The external benchmark
pickle files must be loaded only through the checksum-verifying allowlist
code described above. Never deserialize an unverified pickle.

The names `delong_tests_bootstrap.json` and `delong_tests_holdout.json` are
historical. Current inference uses the paired prompt-cluster bootstrap
described in the article. The reported p-values use normal tails based on
bootstrap standard errors. Recorded empirical bootstrap-tail probabilities
are distinct quantities. Files marked `historical_non_inferential` are
preserved to document and test that distinction, not to support new claims.

The historical generation arrays do not contain enough metadata to certify
all original generation settings or independent raw random streams. The
current source code is not evidence that it generated those historical
arrays. Existing hash checks establish file identity, not missing provenance.
Analysis entry points require an explicit `--allow-legacy-archives` opt-in
for the allowlisted legacy archives; this option does not strengthen their
historical-generation record.

The copied `provenance.json` describes the larger source-project snapshot,
not this curated repository's membership. Use `FILE_INDEX.json` and
`SHA256SUMS` for the export. Exactly three allowlisted machine-local metadata
paths may be omitted in public copies; the index records the actual fields
and both original and public hashes. Numerical records are otherwise
byte-for-byte copies. Manuscript paths in the original provenance record
do not imply that those files are included in this repository.

## Rerun analyses

`code/README.md` documents experiment and plotting entry points. Paths are
relative to this repository's `code/` and `results/` layout; there is no `anc/`
dependency. The original implementation guide also describes author workflows
that require separately obtained manuscript sources. Commands
for results outside the explicit file inventory require additional inputs
or a fresh computation. Use new output directories for reruns and preserve
the supplied observations and hashes.

[REPRODUCING.md](REPRODUCING.md) gives the principal artifact-generation
order and the complete current fresh-generation recipe, including the
commit-pinned sampler checkout, base and extension temperatures, prompt
construction, separate output directories, and replotting the Gumbel-only
figure assets. These are workflows to execute deliberately, not commands
run by repository preparation.

Synthetic experiments need no model weights. Analyses of the stored matched
arrays need the external prompt-alignment tensors but no new language-model
generation. Fresh generation additionally needs the pinned models, tokenizers,
and C4 inputs identified in `code/generate_temperature_matched.py` and
`code/build_prompts.py`, appropriate access, and substantial computation.
Their respective access and reuse terms remain separate.

## Citing the paper

Li Ma (2026). *Predictive Likelihood Ratios for Language Model Watermark
Detection*. arXiv:2609.15657. https://arxiv.org/abs/2609.15657

The preferred citation in `CITATION.cff` links to the paper. Cite the repository
commit used for an analysis as well, so the supporting-file snapshot is identifiable.

## Synthetic comparisons and posterior summaries

Manuscript sections: 3.2, A.2, A.3, A.8.

- [results/bayesian_paper_benchmark/benchmark_results.csv](results/bayesian_paper_benchmark/benchmark_results.csv) (6,396,748 bytes).
- [results/bayesian_paper_benchmark/benchmark_selected_horizons.csv](results/bayesian_paper_benchmark/benchmark_selected_horizons.csv) (39,550 bytes).
- [results/bayesian_paper_benchmark/rejection_indicators.npz](results/bayesian_paper_benchmark/rejection_indicators.npz) (61,014 bytes).
- [results/bayesian_paper_benchmark/paired_comparisons.json](results/bayesian_paper_benchmark/paired_comparisons.json) (206,648 bytes).
- [results/bayesian_paper_benchmark/regime_sweep.json](results/bayesian_paper_benchmark/regime_sweep.json) (272,575 bytes).
- [results/bayesian_paper_benchmark/tail_regime_sweep.json](results/bayesian_paper_benchmark/tail_regime_sweep.json) (226,695 bytes).
- [results/bayesian_paper_benchmark/delta_learning.json](results/bayesian_paper_benchmark/delta_learning.json) (4,997 bytes).
- [results/bayesian_paper_benchmark/block_posterior.json](results/bayesian_paper_benchmark/block_posterior.json) (533 bytes).

## Contamination

Manuscript sections: A.1.

- [results/bayesian_paper_benchmark/contamination_summary.json](results/bayesian_paper_benchmark/contamination_summary.json) (948,260 bytes).
- [results/bayesian_paper_benchmark/contamination_results.csv](results/bayesian_paper_benchmark/contamination_results.csv) (198,034 bytes).
- [results/bayesian_paper_benchmark/contamination_rejection_indicators.npz](results/bayesian_paper_benchmark/contamination_rejection_indicators.npz) (422,431 bytes).
- [results/bayesian_paper_benchmark/contamination_asymmetry.json](results/bayesian_paper_benchmark/contamination_asymmetry.json) (10,230 bytes).
- [results/bayesian_paper_benchmark/contamination_rho_diagnostic.json](results/bayesian_paper_benchmark/contamination_rho_diagnostic.json) (134,458 bytes).
- [results/bayesian_paper_benchmark/contamination_quadrature_sensitivity.json](results/bayesian_paper_benchmark/contamination_quadrature_sensitivity.json) (11,765 bytes).
- [results/bayesian_paper_benchmark/contamination_union_quadrature_sensitivity.json](results/bayesian_paper_benchmark/contamination_union_quadrature_sensitivity.json) (17,044 bytes).

## Numerical validation

Manuscript sections: A.5, A.6.

- [results/bayesian_paper_benchmark/benchmark_summary.json](results/bayesian_paper_benchmark/benchmark_summary.json) (189,171 bytes).
- [results/bayesian_paper_benchmark/clean_quadrature_sensitivity.json](results/bayesian_paper_benchmark/clean_quadrature_sensitivity.json) (28,589 bytes).
- [results/bayesian_paper_benchmark/clean_union_quadrature_sensitivity.json](results/bayesian_paper_benchmark/clean_union_quadrature_sensitivity.json) (47,500 bytes).
- [results/bayesian_paper_benchmark/union_outer_lookup_validation.json](results/bayesian_paper_benchmark/union_outer_lookup_validation.json) (2,664 bytes).

## Archived benchmark reanalysis

Manuscript sections: A.7.

- [results/bayesian_paper_benchmark/tail_width_estimate.json](results/bayesian_paper_benchmark/tail_width_estimate.json) (13,308 bytes).
- [results/bayesian_paper_benchmark/real_model/pivot_repetition.json](results/bayesian_paper_benchmark/real_model/pivot_repetition.json) (1,389 bytes).
- [results/bayesian_paper_benchmark/real_model/real_data_summary.json](results/bayesian_paper_benchmark/real_model/real_data_summary.json) (463,463 bytes).
- [results/bayesian_paper_benchmark/real_model/real_data_results.csv](results/bayesian_paper_benchmark/real_model/real_data_results.csv) (138,052 bytes).
- [results/bayesian_paper_benchmark/real_model/real_data_indicators.npz](results/bayesian_paper_benchmark/real_model/real_data_indicators.npz) (311,648 bytes).
- [results/bayesian_paper_benchmark/real_model/prefix_curves/real_data_prefix_curves.csv](results/bayesian_paper_benchmark/real_model/prefix_curves/real_data_prefix_curves.csv) (16,446,584 bytes).
- [results/bayesian_paper_benchmark/real_model/prefix_curves/real_data_prefix_curves.json](results/bayesian_paper_benchmark/real_model/prefix_curves/real_data_prefix_curves.json) (109,033 bytes).

## Temperature-matched comparisons

Manuscript sections: 3.7, A.9, A.3, A.4.2.

- [results/bayesian_paper_benchmark/real_model/temperature_matched/delong_tests.json](results/bayesian_paper_benchmark/real_model/temperature_matched/delong_tests.json) (9,615 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/delong_tests_bootstrap.json](results/bayesian_paper_benchmark/real_model/temperature_matched/delong_tests_bootstrap.json) (50,881 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/delong_tests_holdout.json](results/bayesian_paper_benchmark/real_model/temperature_matched/delong_tests_holdout.json) (51,027 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/delong_tests_weights.json](results/bayesian_paper_benchmark/real_model/temperature_matched/delong_tests_weights.json) (9,989 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/delong_tests_wideprior.json](results/bayesian_paper_benchmark/real_model/temperature_matched/delong_tests_wideprior.json) (8,604 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/empirical_deficits.json](results/bayesian_paper_benchmark/real_model/temperature_matched/empirical_deficits.json) (4,270 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/temperature_matched_analysis.json](results/bayesian_paper_benchmark/real_model/temperature_matched/temperature_matched_analysis.json) (18,295 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/temperature_prefix_curves.csv](results/bayesian_paper_benchmark/real_model/temperature_matched/temperature_prefix_curves.csv) (7,665,561 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/temperature_prefix_curves_summary.json](results/bayesian_paper_benchmark/real_model/temperature_matched/temperature_prefix_curves_summary.json) (369 bytes).

## Hierarchical extensions

Manuscript sections: A.10.

- [results/bayesian_paper_benchmark/deficit_persistence_sweep.json](results/bayesian_paper_benchmark/deficit_persistence_sweep.json) (22,610 bytes).
- [results/bayesian_paper_benchmark/deficit_persistence_indicators.npz](results/bayesian_paper_benchmark/deficit_persistence_indicators.npz) (98,181 bytes).
- [results/bayesian_paper_benchmark/width_persistence_sweep.json](results/bayesian_paper_benchmark/width_persistence_sweep.json) (17,182 bytes).
- [results/bayesian_paper_benchmark/width_persistence_indicators.npz](results/bayesian_paper_benchmark/width_persistence_indicators.npz) (79,197 bytes).

## Current source/artifact snapshot

Manuscript sections: A.11.

- [results/bayesian_paper_benchmark/provenance.json](results/bayesian_paper_benchmark/provenance.json) (28,056 bytes).

## Implementation and tests

- [code/analyse_temperature_matched.py](code/analyse_temperature_matched.py) (5,156 bytes).
- [code/bayesian_watermark.py](code/bayesian_watermark.py) (26,982 bytes).
- [code/benchmark_contamination.py](code/benchmark_contamination.py) (95,182 bytes).
- [code/benchmark_paper_experiment.py](code/benchmark_paper_experiment.py) (105,337 bytes).
- [code/build_prompts.py](code/build_prompts.py) (8,105 bytes).
- [code/check_block_posterior.py](code/check_block_posterior.py) (3,425 bytes).
- [code/check_clean_quadrature.py](code/check_clean_quadrature.py) (9,228 bytes).
- [code/check_contamination_quadrature.py](code/check_contamination_quadrature.py) (7,401 bytes).
- [code/check_delta_learning.py](code/check_delta_learning.py) (6,927 bytes).
- [code/check_figure_geometry.py](code/check_figure_geometry.py) (8,583 bytes).
- [code/check_manuscript_tables.py](code/check_manuscript_tables.py) (81,430 bytes).
- [code/contamination_asymmetry.py](code/contamination_asymmetry.py) (16,482 bytes).
- [code/deficit_hierarchy.py](code/deficit_hierarchy.py) (9,229 bytes).
- [code/deficit_persistence_sweep.py](code/deficit_persistence_sweep.py) (18,465 bytes).
- [code/delong_prompt_bootstrap.py](code/delong_prompt_bootstrap.py) (14,588 bytes).
- [code/demo_bayesian_watermark.py](code/demo_bayesian_watermark.py) (3,264 bytes).
- [code/diff_artifacts.py](code/diff_artifacts.py) (4,610 bytes).
- [code/dirichlet_detector.py](code/dirichlet_detector.py) (55,096 bytes).
- [code/estimate_tail_width.py](code/estimate_tail_width.py) (18,122 bytes).
- [code/experiment_metadata.py](code/experiment_metadata.py) (3,949 bytes).
- [code/generate_temperature_matched.py](code/generate_temperature_matched.py) (28,672 bytes).
- [code/inverse_batched_sampling.py](code/inverse_batched_sampling.py) (5,785 bytes).
- [code/inverse_support_edge.py](code/inverse_support_edge.py) (14,709 bytes).
- [code/make_gumbel_only_figures.py](code/make_gumbel_only_figures.py) (6,628 bytes).
- [code/matched_data_loader.py](code/matched_data_loader.py) (14,632 bytes).
- [code/paired_comparisons.py](code/paired_comparisons.py) (20,505 bytes).
- [code/pivot_repetition.py](code/pivot_repetition.py) (6,033 bytes).
- [code/plot_temperature_prefix_curves.py](code/plot_temperature_prefix_curves.py) (37,746 bytes).
- [code/pooled_width_real_data.py](code/pooled_width_real_data.py) (4,547 bytes).
- [code/pooling_max_regret.py](code/pooling_max_regret.py) (7,087 bytes).
- [code/real_data_experiment.py](code/real_data_experiment.py) (69,190 bytes).
- [code/real_data_prefix_curves.py](code/real_data_prefix_curves.py) (44,528 bytes).
- [code/regime_sweep.py](code/regime_sweep.py) (60,800 bytes).
- [code/summarise_empirical_deficits.py](code/summarise_empirical_deficits.py) (2,903 bytes).
- [code/sync_manuscript_tables.py](code/sync_manuscript_tables.py) (11,171 bytes).
- [code/tail_family.py](code/tail_family.py) (27,079 bytes).
- [code/tail_regime_sweep.py](code/tail_regime_sweep.py) (56,869 bytes).
- [code/test_bayesian_watermark.py](code/test_bayesian_watermark.py) (8,843 bytes).
- [code/test_benchmark_contamination.py](code/test_benchmark_contamination.py) (37,513 bytes).
- [code/test_benchmark_paper_experiment.py](code/test_benchmark_paper_experiment.py) (45,804 bytes).
- [code/test_build_prompts.py](code/test_build_prompts.py) (7,819 bytes).
- [code/test_check_manuscript_tables.py](code/test_check_manuscript_tables.py) (15,372 bytes).
- [code/test_contamination_asymmetry.py](code/test_contamination_asymmetry.py) (5,762 bytes).
- [code/test_core_numerical_regressions.py](code/test_core_numerical_regressions.py) (9,326 bytes).
- [code/test_deficit_hierarchy.py](code/test_deficit_hierarchy.py) (7,716 bytes).
- [code/test_deficit_persistence_sweep.py](code/test_deficit_persistence_sweep.py) (6,430 bytes).
- [code/test_delong_prompt_bootstrap.py](code/test_delong_prompt_bootstrap.py) (9,028 bytes).
- [code/test_dirichlet_detector.py](code/test_dirichlet_detector.py) (37,639 bytes).
- [code/test_estimate_tail_width.py](code/test_estimate_tail_width.py) (9,361 bytes).
- [code/test_experiment_metadata.py](code/test_experiment_metadata.py) (2,618 bytes).
- [code/test_generate_temperature_matched.py](code/test_generate_temperature_matched.py) (21,960 bytes).
- [code/test_inverse_batched_sampling.py](code/test_inverse_batched_sampling.py) (10,660 bytes).
- [code/test_inverse_support_edge.py](code/test_inverse_support_edge.py) (6,030 bytes).
- [code/test_legacy_analysis_consumers.py](code/test_legacy_analysis_consumers.py) (1,664 bytes).
- [code/test_matched_data_loader.py](code/test_matched_data_loader.py) (14,011 bytes).
- [code/test_pivot_repetition.py](code/test_pivot_repetition.py) (2,434 bytes).
- [code/test_plot_temperature_prefix_curves.py](code/test_plot_temperature_prefix_curves.py) (21,295 bytes).
- [code/test_pooled_width_real_data.py](code/test_pooled_width_real_data.py) (2,495 bytes).
- [code/test_pooling_max_regret.py](code/test_pooling_max_regret.py) (6,163 bytes).
- [code/test_real_data_experiment.py](code/test_real_data_experiment.py) (16,310 bytes).
- [code/test_real_data_prefix_curves.py](code/test_real_data_prefix_curves.py) (11,857 bytes).
- [code/test_regime_sweep.py](code/test_regime_sweep.py) (41,112 bytes).
- [code/test_reporting_tools.py](code/test_reporting_tools.py) (18,724 bytes).
- [code/test_tail_regime_sweep.py](code/test_tail_regime_sweep.py) (30,607 bytes).
- [code/test_trgof.py](code/test_trgof.py) (24,587 bytes).
- [code/trgof.py](code/trgof.py) (5,569 bytes).
- [code/update_manuscript_table_mcse.py](code/update_manuscript_table_mcse.py) (16,960 bytes).
- [code/width_hierarchy.py](code/width_hierarchy.py) (9,431 bytes).
- [code/width_persistence_sweep.py](code/width_persistence_sweep.py) (16,622 bytes).
- [code/write_provenance.py](code/write_provenance.py) (6,465 bytes).
- [code/wrong_key_null.py](code/wrong_key_null.py) (1,675 bytes).
- [code/requirements.txt](code/requirements.txt) (418 bytes).
- [code/requirements-real-data.txt](code/requirements-real-data.txt) (179 bytes).
- [code/README.md](code/README.md) (39,003 bytes).

## Additional regression-test support; not current inferential results

- [contexts/drift_report.py](contexts/drift_report.py) (2,652 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/delong_tests_inverse.json](results/bayesian_paper_benchmark/real_model/temperature_matched/delong_tests_inverse.json) (9,648 bytes).
- [results/bayesian_paper_benchmark_delta_half/real_model/temperature_matched/delong_tests.json](results/bayesian_paper_benchmark_delta_half/real_model/temperature_matched/delong_tests.json) (9,816 bytes).
- [results/bayesian_paper_benchmark_delta_half/real_model/temperature_matched/delong_tests_weights.json](results/bayesian_paper_benchmark_delta_half/real_model/temperature_matched/delong_tests_weights.json) (10,190 bytes).
- [results/bayesian_paper_benchmark_delta_half/real_model/temperature_matched/delong_tests_wideprior.json](results/bayesian_paper_benchmark_delta_half/real_model/temperature_matched/delong_tests_wideprior.json) (8,733 bytes).
- [results/bayesian_paper_benchmark_delta_half/real_model/temperature_matched/delong_tests_inverse.json](results/bayesian_paper_benchmark_delta_half/real_model/temperature_matched/delong_tests_inverse.json) (9,648 bytes).

## Stored generations and prompt tables

- [results/bayesian_paper_benchmark/real_model/temperature_matched/matched_1p3B_hi.npz](results/bayesian_paper_benchmark/real_model/temperature_matched/matched_1p3B_hi.npz) (3,295,738 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/matched_1p3B_lo.npz](results/bayesian_paper_benchmark/real_model/temperature_matched/matched_1p3B_lo.npz) (2,405,377 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/matched_2p7B_hi.npz](results/bayesian_paper_benchmark/real_model/temperature_matched/matched_2p7B_hi.npz) (3,322,704 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/matched_2p7B_lo.npz](results/bayesian_paper_benchmark/real_model/temperature_matched/matched_2p7B_lo.npz) (2,587,521 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/extension/ext_1p3B_hi.npz](results/bayesian_paper_benchmark/real_model/temperature_matched/extension/ext_1p3B_hi.npz) (4,979,925 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/extension/ext_1p3B_lo.npz](results/bayesian_paper_benchmark/real_model/temperature_matched/extension/ext_1p3B_lo.npz) (6,979,832 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/extension/ext_2p7B_hi.npz](results/bayesian_paper_benchmark/real_model/temperature_matched/extension/ext_2p7B_hi.npz) (5,039,757 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/extension/ext_2p7B_lo.npz](results/bayesian_paper_benchmark/real_model/temperature_matched/extension/ext_2p7B_lo.npz) (7,383,127 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/extension/prompts_1p3B.npy](results/bayesian_paper_benchmark/real_model/temperature_matched/extension/prompts_1p3B.npy) (1,000,128 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/extension/prompts_2p7B.npy](results/bayesian_paper_benchmark/real_model/temperature_matched/extension/prompts_2p7B.npy) (1,000,128 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/extension/prompts_1p3B.json](results/bayesian_paper_benchmark/real_model/temperature_matched/extension/prompts_1p3B.json) (678 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/extension/prompts_2p7B.json](results/bayesian_paper_benchmark/real_model/temperature_matched/extension/prompts_2p7B.json) (693 bytes).

## Historical Beta-generator experiment; not current Gaussian-copula results

- [historical/6551ee5/code/deficit_persistence_sweep.py](historical/6551ee5/code/deficit_persistence_sweep.py) (10,426 bytes).
- [historical/6551ee5/results/bayesian_paper_benchmark/deficit_persistence_sweep.json](historical/6551ee5/results/bayesian_paper_benchmark/deficit_persistence_sweep.json) (5,315 bytes).

## External figure assets used by the technical report

- [results/bayesian_paper_benchmark/gumbel_only/real_model/prefix_curves/real_data_common_calibration_prefix_curves.pdf](results/bayesian_paper_benchmark/gumbel_only/real_model/prefix_curves/real_data_common_calibration_prefix_curves.pdf) (62,100 bytes).
- [results/bayesian_paper_benchmark/gumbel_only/real_model/prefix_curves/real_data_upstream_recorded_prefix_curves.pdf](results/bayesian_paper_benchmark/gumbel_only/real_model/prefix_curves/real_data_upstream_recorded_prefix_curves.pdf) (44,893 bytes).
- [results/bayesian_paper_benchmark/gumbel_only/real_model/temperature_matched/prefix_curves_matched_1p3B.pdf](results/bayesian_paper_benchmark/gumbel_only/real_model/temperature_matched/prefix_curves_matched_1p3B.pdf) (70,278 bytes).
- [results/bayesian_paper_benchmark/gumbel_only/real_model/temperature_matched/prefix_curves_matched_2p7B.pdf](results/bayesian_paper_benchmark/gumbel_only/real_model/temperature_matched/prefix_curves_matched_2p7B.pdf) (70,014 bytes).
- [results/bayesian_paper_benchmark/gumbel_only/slide_bayes_contamination.pdf](results/bayesian_paper_benchmark/gumbel_only/slide_bayes_contamination.pdf) (25,456 bytes).
- [results/bayesian_paper_benchmark/gumbel_only/slide_bayes_shared_delta.pdf](results/bayesian_paper_benchmark/gumbel_only/slide_bayes_shared_delta.pdf) (49,779 bytes).
- [results/bayesian_paper_benchmark/gumbel_only/slide_bayes_tokenwise_delta.pdf](results/bayesian_paper_benchmark/gumbel_only/slide_bayes_tokenwise_delta.pdf) (34,177 bytes).

## Additional production outputs and figure variants; exploratory inverse and alternate-prior results are outside the main Gumbel claims

- [results/bayesian_paper_benchmark/bayesian_vs_paper_shared_delta.pdf](results/bayesian_paper_benchmark/bayesian_vs_paper_shared_delta.pdf) (168,181 bytes).
- [results/bayesian_paper_benchmark/bayesian_vs_paper_shared_delta.png](results/bayesian_paper_benchmark/bayesian_vs_paper_shared_delta.png) (749,779 bytes).
- [results/bayesian_paper_benchmark/bayesian_vs_paper_tokenwise_delta.pdf](results/bayesian_paper_benchmark/bayesian_vs_paper_tokenwise_delta.pdf) (106,130 bytes).
- [results/bayesian_paper_benchmark/bayesian_vs_paper_tokenwise_delta.png](results/bayesian_paper_benchmark/bayesian_vs_paper_tokenwise_delta.png) (607,826 bytes).
- [results/bayesian_paper_benchmark/benchmark_report.md](results/bayesian_paper_benchmark/benchmark_report.md) (951 bytes).
- [results/bayesian_paper_benchmark/contamination_report.md](results/bayesian_paper_benchmark/contamination_report.md) (1,614 bytes).
- [results/bayesian_paper_benchmark/contamination_robustness.pdf](results/bayesian_paper_benchmark/contamination_robustness.pdf) (47,310 bytes).
- [results/bayesian_paper_benchmark/contamination_robustness.png](results/bayesian_paper_benchmark/contamination_robustness.png) (397,807 bytes).
- [results/bayesian_paper_benchmark/gumbel_only/real_model/prefix_curves/real_data_common_calibration_prefix_curves.png](results/bayesian_paper_benchmark/gumbel_only/real_model/prefix_curves/real_data_common_calibration_prefix_curves.png) (307,249 bytes).
- [results/bayesian_paper_benchmark/gumbel_only/real_model/prefix_curves/real_data_upstream_recorded_prefix_curves.png](results/bayesian_paper_benchmark/gumbel_only/real_model/prefix_curves/real_data_upstream_recorded_prefix_curves.png) (267,347 bytes).
- [results/bayesian_paper_benchmark/gumbel_only/real_model/temperature_matched/prefix_curves_matched_1p3B.png](results/bayesian_paper_benchmark/gumbel_only/real_model/temperature_matched/prefix_curves_matched_1p3B.png) (353,223 bytes).
- [results/bayesian_paper_benchmark/gumbel_only/real_model/temperature_matched/prefix_curves_matched_2p7B.png](results/bayesian_paper_benchmark/gumbel_only/real_model/temperature_matched/prefix_curves_matched_2p7B.png) (358,990 bytes).
- [results/bayesian_paper_benchmark/gumbel_only/slide_bayes_contamination.png](results/bayesian_paper_benchmark/gumbel_only/slide_bayes_contamination.png) (120,170 bytes).
- [results/bayesian_paper_benchmark/gumbel_only/slide_bayes_shared_delta.png](results/bayesian_paper_benchmark/gumbel_only/slide_bayes_shared_delta.png) (217,585 bytes).
- [results/bayesian_paper_benchmark/gumbel_only/slide_bayes_tokenwise_delta.png](results/bayesian_paper_benchmark/gumbel_only/slide_bayes_tokenwise_delta.png) (222,425 bytes).
- [results/bayesian_paper_benchmark/inverse_support_edge.json](results/bayesian_paper_benchmark/inverse_support_edge.json) (9,511 bytes).
- [results/bayesian_paper_benchmark/real_model/prefix_curves/real_data_common_calibration_prefix_curves.pdf](results/bayesian_paper_benchmark/real_model/prefix_curves/real_data_common_calibration_prefix_curves.pdf) (83,383 bytes).
- [results/bayesian_paper_benchmark/real_model/prefix_curves/real_data_common_calibration_prefix_curves.png](results/bayesian_paper_benchmark/real_model/prefix_curves/real_data_common_calibration_prefix_curves.png) (433,481 bytes).
- [results/bayesian_paper_benchmark/real_model/prefix_curves/real_data_upstream_recorded_prefix_curves.pdf](results/bayesian_paper_benchmark/real_model/prefix_curves/real_data_upstream_recorded_prefix_curves.pdf) (59,591 bytes).
- [results/bayesian_paper_benchmark/real_model/prefix_curves/real_data_upstream_recorded_prefix_curves.png](results/bayesian_paper_benchmark/real_model/prefix_curves/real_data_upstream_recorded_prefix_curves.png) (372,206 bytes).
- [results/bayesian_paper_benchmark/real_model/sensitivity_uniform_0_0p1/comparison_report.md](results/bayesian_paper_benchmark/real_model/sensitivity_uniform_0_0p1/comparison_report.md) (4,161 bytes).
- [results/bayesian_paper_benchmark/real_model/sensitivity_uniform_0_0p1/real_data_indicators.npz](results/bayesian_paper_benchmark/real_model/sensitivity_uniform_0_0p1/real_data_indicators.npz) (267,208 bytes).
- [results/bayesian_paper_benchmark/real_model/sensitivity_uniform_0_0p1/real_data_results.csv](results/bayesian_paper_benchmark/real_model/sensitivity_uniform_0_0p1/real_data_results.csv) (117,486 bytes).
- [results/bayesian_paper_benchmark/real_model/sensitivity_uniform_0_0p1/real_data_summary.json](results/bayesian_paper_benchmark/real_model/sensitivity_uniform_0_0p1/real_data_summary.json) (385,060 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/prefix_curves_matched_1p3B.pdf](results/bayesian_paper_benchmark/real_model/temperature_matched/prefix_curves_matched_1p3B.pdf) (58,627 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/prefix_curves_matched_1p3B.png](results/bayesian_paper_benchmark/real_model/temperature_matched/prefix_curves_matched_1p3B.png) (237,491 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/prefix_curves_matched_1p3B_inverse.pdf](results/bayesian_paper_benchmark/real_model/temperature_matched/prefix_curves_matched_1p3B_inverse.pdf) (57,748 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/prefix_curves_matched_1p3B_inverse.png](results/bayesian_paper_benchmark/real_model/temperature_matched/prefix_curves_matched_1p3B_inverse.png) (262,740 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/prefix_curves_matched_1p3B_w0p9.pdf](results/bayesian_paper_benchmark/real_model/temperature_matched/prefix_curves_matched_1p3B_w0p9.pdf) (58,637 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/prefix_curves_matched_1p3B_w0p9.png](results/bayesian_paper_benchmark/real_model/temperature_matched/prefix_curves_matched_1p3B_w0p9.png) (238,595 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/prefix_curves_matched_2p7B.pdf](results/bayesian_paper_benchmark/real_model/temperature_matched/prefix_curves_matched_2p7B.pdf) (58,609 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/prefix_curves_matched_2p7B.png](results/bayesian_paper_benchmark/real_model/temperature_matched/prefix_curves_matched_2p7B.png) (244,113 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/prefix_curves_matched_2p7B_inverse.pdf](results/bayesian_paper_benchmark/real_model/temperature_matched/prefix_curves_matched_2p7B_inverse.pdf) (59,415 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/prefix_curves_matched_2p7B_inverse.png](results/bayesian_paper_benchmark/real_model/temperature_matched/prefix_curves_matched_2p7B_inverse.png) (273,111 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/prefix_curves_matched_2p7B_w0p9.pdf](results/bayesian_paper_benchmark/real_model/temperature_matched/prefix_curves_matched_2p7B_w0p9.pdf) (58,759 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/prefix_curves_matched_2p7B_w0p9.png](results/bayesian_paper_benchmark/real_model/temperature_matched/prefix_curves_matched_2p7B_w0p9.png) (245,597 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/temperature_prefix_curves_inverse.csv](results/bayesian_paper_benchmark/real_model/temperature_matched/temperature_prefix_curves_inverse.csv) (2,303,042 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/temperature_prefix_curves_summary_inverse.json](results/bayesian_paper_benchmark/real_model/temperature_matched/temperature_prefix_curves_summary_inverse.json) (384 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/temperature_prefix_curves_summary_w0p9.json](results/bayesian_paper_benchmark/real_model/temperature_matched/temperature_prefix_curves_summary_w0p9.json) (364 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/temperature_prefix_curves_w0p9.csv](results/bayesian_paper_benchmark/real_model/temperature_matched/temperature_prefix_curves_w0p9.csv) (3,373,757 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/temperature_sweep_matched.pdf](results/bayesian_paper_benchmark/real_model/temperature_matched/temperature_sweep_matched.pdf) (27,664 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/temperature_sweep_matched.png](results/bayesian_paper_benchmark/real_model/temperature_matched/temperature_sweep_matched.png) (109,335 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/temperature_sweep_matched_inverse.pdf](results/bayesian_paper_benchmark/real_model/temperature_matched/temperature_sweep_matched_inverse.pdf) (27,669 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/temperature_sweep_matched_inverse.png](results/bayesian_paper_benchmark/real_model/temperature_matched/temperature_sweep_matched_inverse.png) (134,872 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/temperature_sweep_matched_w0p9.pdf](results/bayesian_paper_benchmark/real_model/temperature_matched/temperature_sweep_matched_w0p9.pdf) (27,622 bytes).
- [results/bayesian_paper_benchmark/real_model/temperature_matched/temperature_sweep_matched_w0p9.png](results/bayesian_paper_benchmark/real_model/temperature_matched/temperature_sweep_matched_w0p9.png) (111,300 bytes).
- [results/bayesian_paper_benchmark/slide_bayes_contamination.pdf](results/bayesian_paper_benchmark/slide_bayes_contamination.pdf) (30,181 bytes).
- [results/bayesian_paper_benchmark/slide_bayes_contamination.png](results/bayesian_paper_benchmark/slide_bayes_contamination.png) (204,953 bytes).
- [results/bayesian_paper_benchmark/slide_bayes_shared_delta.pdf](results/bayesian_paper_benchmark/slide_bayes_shared_delta.pdf) (77,952 bytes).
- [results/bayesian_paper_benchmark/slide_bayes_shared_delta.png](results/bayesian_paper_benchmark/slide_bayes_shared_delta.png) (292,596 bytes).
- [results/bayesian_paper_benchmark/slide_bayes_shared_delta_preview.png](results/bayesian_paper_benchmark/slide_bayes_shared_delta_preview.png) (154,266 bytes).
- [results/bayesian_paper_benchmark/slide_bayes_tokenwise_delta.pdf](results/bayesian_paper_benchmark/slide_bayes_tokenwise_delta.pdf) (39,438 bytes).
- [results/bayesian_paper_benchmark/slide_bayes_tokenwise_delta.png](results/bayesian_paper_benchmark/slide_bayes_tokenwise_delta.png) (252,834 bytes).
- [results/bayesian_paper_benchmark/slide_bayes_tokenwise_delta_preview.png](results/bayesian_paper_benchmark/slide_bayes_tokenwise_delta_preview.png) (64,796 bytes).
