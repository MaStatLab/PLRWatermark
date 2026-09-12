# Bayesian pivot-watermark detector

This directory is the executable companion to the Bayesian pivot-watermark
manuscript. The core detector uses NumPy; analysis and figures also use SciPy
and Matplotlib. Released-tensor replay uses PyTorch. Prompt construction and
model regeneration additionally use Transformers and the pinned upstream
sampler's IPython dependency.

Repository-wide setup, provenance, full artifact-generation order, legacy-file
warnings, and manuscript compilation are documented in the top-level
`README.md`.

`python3 code/write_provenance.py` writes a machine-readable manifest containing
the interpreter and package versions, Git revision when available, an aggregate
input snapshot hash, per-file manuscript/code hashes, and a separate aggregate
hash plus per-file hashes for generated results and the definitive PDF.
Its runtime describes the manifest writer, not historical generation. Current
hashes identify stored files but cannot recover missing generation settings.
`python3 code/check_manuscript_tables.py` independently checks every selected
result row in the manuscript against the generated clean, regime, tail, and
contamination artifacts, preventing hand-formatted tables from silently
drifting. It also verifies the hierarchical comparison's mean AUC gain,
consistent conditional-contribution standard errors for the clean benchmark,
and within-block boldface extrema, including both AUC tables. `python3
code/update_manuscript_table_mcse.py` performs the preceding mechanical refresh
of each displayed estimate and its parenthetical standard error, and
`python3 code/sync_manuscript_tables.py` re-marks the extremum of each panel
column.

The detector keeps the pivotal null laws of Li et al. (2025) and integrates over uncertain
alternative parameters:

\[
\mathrm{BF}_n
=\frac{\int \prod_{t=1}^n f_1(y_t\mid\Delta,\rho)\,
              \pi(d\Delta,d\rho)}
       {\prod_{t=1}^n f_0(y_t)},
\qquad
f_1(y\mid\Delta,\rho)
=(1-\rho)f_{\mathrm{signal}}(y\mid\Delta)+\rho f_0(y).
\]

Implemented likelihoods:

- Gumbel-max: \(f_0(r)=1\) and
  \(f_1(r\mid\mathbf P)=\sum_{w:P_w>0}r^{1/P_w-1}\).
- Inverse transform: the exact finite-vocabulary pivot null is available by
  setting `inverse_null_vocabulary_size`; its large-vocabulary limit is
  \(f_0(d)=2(1-d)\). The alternative uses the large-vocabulary density of
  Li et al. (2025),
  \(f_1(d\mid\Delta)=\frac{2}{1-\Delta}
  (1-d/(1-\Delta))_+\).

`SequentialBayesDetector` supports two prior structures:

- `shared`: one latent `(Delta, rho)` pair for the whole document, updated
  sequentially;
- `tokenwise`: integrate a fresh pair at every token, for heterogeneous NTP
  regimes.

The Bayes factor can be converted to a posterior probability, a loss-optimal
decision, or an anytime evidence rule. With an exact simple null, the
idealized independent-key assumptions of Li et al. (2025), and a coherent alternative mixture
fixed before observing the document, stopping when `BF >= 1/alpha` controls the
probability of ever crossing under the null by `alpha`. Gumbel's uniform null is
exact. For inverse pivots at finite vocabulary, set
`inverse_null_vocabulary_size`; otherwise the triangular limiting null provides
only asymptotic calibration.

Run the reproducible examples:

```bash
# From the repository root.
python3 code/demo_bayesian_watermark.py
```

Run the test suite:

```bash
cd code
python3 -m unittest -v test_bayesian_watermark
```

The complete suite currently comprises 604 tests in 28 modules;
11 upstream-comparison tests skip when the pinned sampler source is unavailable:

```bash
cd code
python3 -m unittest discover -p 'test_*.py'
cd ..
```

## Clean synthetic benchmark adapted from Li et al. (2025)

`benchmark_paper_experiment.py` compares the Bayesian rules with the baseline
and minimax scores evaluated in the clean synthetic experiment of Li et al. (2025):

- Gumbel: `h_ars`, `h_log`, `h_ind,1/e`, and the optimal scores of Li et al. at
  `Delta=0.1`, `0.01`, and `0.005`;
- inverse transform: `h_neg` and the optimal scores at `Delta=0.1`, `0.01`,
  and `0.001`;
- tokenwise and shared-Delta Bayesian mixtures, each as a fixed-horizon test
  and as a separate anytime `BF >= 20` test.

The default run uses `V=1000`, all prefixes from 1 through 700, 10,000
independent null calibration sequences, 5,000 independent null evaluation
sequences, and 5,000 watermarked sequences. Fixed-horizon thresholds are
calibrated fairly on the same exact-null pivots; the indicator test uses a
randomized boundary to target 5% despite its discrete score.

For the exact tested versions, install `code/requirements.txt` in an
isolated Python environment. If Matplotlib is unavailable, add `--skip-plots`;
the complete CSV and JSON results are still produced.

The pinned environment contains NumPy 2.0.2, SciPy 1.13.1, Matplotlib 3.9.4,
PyTorch 2.8.0, Transformers 4.56.1, and IPython 8.18.1. Preserve the interpreter
version, dependency versions, code revision or file hashes, and the
configuration/seed/runtime metadata written to JSON when reporting a rerun.

```bash
MPLCONFIGDIR=/tmp/bayesian-watermark-mpl python3 code/benchmark_paper_experiment.py
```

The run writes long-form results, a compact JSON summary, and both raster and
vector plots to `results/bayesian_paper_benchmark/`. A fast smoke run is:

```bash
MPLCONFIGDIR=/tmp/bayesian-watermark-mpl python3 \
    code/benchmark_paper_experiment.py --quick
```

All five quick entry points, including the clean benchmark, isolate their
default outputs automatically.  An explicit output path still takes priority.

`benchmark_report.md` is a legacy hand-written snapshot and is **not an
authoritative or generated artifact**.  It can lag the CSV/JSON and manuscript;
do not use it to validate a run.  The machine-readable outputs and their embedded
configuration, seed, and runtime fields are authoritative.

The benchmark reports two deliberately labeled data-generating scenarios.
`paper_text_iid_delta_equal_tail` is the stable machine identifier for the
written tokenwise specification of Li et al. (2025):
`Delta_t` is redrawn at each token and the spike tail is equal. The accompanying
[commit-pinned simulation snapshot](https://github.com/lx10077/WatermarkFramework/tree/05b7ffda9279fc9e645f38807e4a0e2dbcff4330)
instead draws one `Delta` per document and uses randomly normalized tail
probabilities. `shared_delta_equal_tail_sensitivity` isolates that shared-Delta
dependence while retaining the stated equal-tail spike. The exact upstream
commit and result-array paths are also stored in `benchmark_summary.json`; the
top-level README records the file hashes.
Legacy `paper`, `paper_*`, and `*_paper_*` keys, enum values, paths, and Python
identifiers are retained only for machine-schema compatibility; prose and
figures use author--date citations and the term "reference score."

Benchmark tests, including numerical normalization of both Bayesian marginal
likelihoods, run with:

```bash
cd code
python3 -m unittest -v test_benchmark_paper_experiment
```

## Released real-model output reanalysis

`real_data_experiment.py` reanalyses, for each of OPT-1.3B and
Sheared-LLaMA-2.7B, 500 Gumbel-max-watermarked, 500 inverse-transform-watermarked,
and 500 unwatermarked continuations released in the
[commit-pinned real-data directory](https://github.com/lx10077/WatermarkFramework/tree/05b7ffda9279fc9e645f38807e4a0e2dbcff4330/real%20data).
The release stores the watermarked pivots and the raw-model token sequences.
Consequently this stage needs no language-model weights, tokenizer, C4 access,
or GPU. PyTorch is used only to deserialize six SHA256-allowlisted pickles and
to replay the released CPU hash/key operations for the raw sequences.
Python pickle deserialization can execute code; the script refuses every file
outside the fixed name-and-hash allowlist.  Run this stage in an isolated
environment and do not relax that check for untrusted inputs.

The primary inverse analysis reconstructs the integer zero-based rank from the
stored shifted rank, validates that reconstruction, and uses the formal
`rank/(V-1)` pivot. A separately labelled coordinate sensitivity applies the
primary score formulas to shifted `(rank-1)/(V-1)` pivots and recalibrates them
on simulated shifted-null paths; it is not interpreted as a Bayes factor under
that shifted pivot law. Every fixed-horizon method is calibrated on
the same independent exact-pivot null paths; the released raw continuations are
used only for an empirical Type-I check. Endpoint results and per-document
randomized decisions at horizons 50, 100, and 200 are written to
`results/bayesian_paper_benchmark/real_model/`.

The released-output Bayesian rows retain
`Delta ~ Uniform(0.001, 0.5)`, as in the synthetic experiments. The code uses
96-node Gauss--Legendre quadrature in `Delta`; the six-atom tail prior remains
independent and uniform on `{0.1, 1, 10, 100, 1000, inf}`, and `rho` is fixed
at zero.

The canonical bounds remain the default.  An isolated real-output sensitivity
can set `--delta-low` and `--delta-high` together with a separate
`--results-dir`; for example, `--delta-low 0 --delta-high 0.1` evaluates
`Delta ~ Uniform(0,0.1)` without overwriting the canonical artifacts.  The
all-prefix script intentionally continues to use the canonical prior.

```bash
# On Linux, an official CPU-only PyTorch wheel may be used instead of a
# platform default that bundles CUDA libraries.
python3 -m pip install -r code/requirements-real-data.txt
python3 code/real_data_experiment.py
MPLCONFIGDIR=/tmp/bayesian-watermark-mpl \
    python3 code/real_data_prefix_curves.py
cd code && python3 -m unittest -v \
    test_real_data_experiment test_real_data_prefix_curves \
    test_inverse_support_edge test_estimate_tail_width \
    test_plot_temperature_prefix_curves
```

`real_data_prefix_curves.py` evaluates every prefix from 1 through 200 under
the fixed-seed common calibration and writes CSV/JSON plus vector and raster
figures under `real_model/prefix_curves/`. A separate figure reads the eight
commit-pinned upstream `*-null.json` and `*-result.json` arrays directly; those
rates are not digitized from a plot and retain the upstream cutoff construction.
The common-calibration and upstream-recorded curves are deliberately not
treated as repeated runs of one test.

## Temperature-matched regeneration

The released pair cannot be used for a power comparison as it stands.  The
watermarked continuations are generated at temperature `0.1`; the released
unwatermarked generator is never passed the temperature argument and samples at
`1.0`.  This is upstream, not a choice made here -- `generation.py` divides the
logits by `temperature` in `generate_gum` and `generate_inv` and not in
`generate_rnd`, and `generating_samples.py` carries the comment
`## We don't adjust the temperature parameter here`.

The consequence is measurable rather than hypothetical.  A "detector" that
discards every pivot value and counts only how many of them are *distinct*
separates the two released samples with AUC `.997`-`.998` and Type II
`.004`-`.012`, beating every method in the study.  Watermarked documents carry
76 and 86 distinct pivots out of 200 and unwatermarked documents carry 197, so a
rule can score well on the released pair by detecting low-temperature
repetition and nothing else.

`generate_temperature_matched.py` removes the confound by generating *both*
arms at a common temperature, from the released prompt tensors, importing the
released key, sampler and pivot functions unchanged. The unwatermarked generator
divides by the same temperature as the watermarked one. Generation is
checkpointed after each complete arm. Resume first checks the model, prompt
hashes and offsets, dimensions, batch size, seed, device, software/code identity,
and every stored arm's companion shapes and hashes. Schema-v3 raw sampling uses
one stream per global prompt index and prompt-row digest, preventing stream
restarts across separately generated base/extension blocks. Compatible runs can add temperatures
or methods; mismatches and historical files without a run record require a new
output path.

The downstream loader checks the manifests against the actual selected prompt
rows and rejects overlapping offsets, duplicate arms, incompatible runs and
unpaired cells. Historical replay requires `--allow-legacy-archives` and an exact
match to one of eight SHA-256 allowlisted files. This opt-in does not certify
their missing generation history or independent raw streams across prompts.

Against these arms the repetition-only detector reads Type II `.948` at
temperature `0.1` -- chance.  A mild residual survives at high temperature
(`.69` at `1.0`) because keyed sampling is deterministic given the context, so
watermarked text genuinely is a little more repetitive; that is a property of
the scheme, not a protocol mismatch.

`wrong_key_null.py` provides a matched-text diagnostic helper: it re-keys the
watermarked text while holding temperature, prompts, model and repetition fixed.
An independent key does not remove repeated PRF addresses or token pairs, so
the resulting paths are not certified iid or conditionally uniform. This is
not an anytime-valid repair. The retained diagnostic summaries are descriptive
(mean pivot `.5013`, 76.2 distinct pivots, repetition-only Type II `.950`).

`plot_temperature_prefix_curves.py` recomputes every rule at every prefix on
the matched arms and writes the curve CSV, one temperature-panel figure per
model, and a summary of Type II against temperature at a fixed horizon.

Calibration deserves care here, because an earlier version of this command got
it wrong in a way that changed the conclusion.  Taking the 5% cutoff from one
half of the unwatermarked arm and the Type I check from the other is
leakage-free, but with 250 calibration documents the *realised* level is a
random variable with standard deviation about `.02`, which induces a Type II
standard deviation of `.020`-`.026` -- larger than the gaps between the leading
rules.  A ranking read off one split is therefore not evidence: on
Sheared-2.7B at temperature `0.3` the reported ordering reversed when the two
halves were swapped.

So Type II now uses a cutoff taken from the *whole* 500-document null arm.  That
is still leakage-free -- the cutoff never sees a watermarked document -- and it
puts every rule at the same realised size, which the split does not.  Type I is
five-fold cross-fitted, and comes out at `.052` for every continuous rule;
`h_ind,1/e` is conservative at `.039` because its statistic is a count and the
tie rule is a strict `>`.  The CSV also carries `auc`, which needs no cutoff at
all, and `type2_split_sd`, which records the spread the old design was drawing
from.

Two features of the curves are worth reading carefully.  Detection is only
informative between temperature `0.2` and `0.5`: everything is near chance at
`0.1` and everything saturates by `0.7`.

And Tr-GoF's power is **not monotone in the token count** at low temperature.
Repeated skipgram contexts reuse one PRF seed, so a repeated n-gram reproduces
its pivot exactly; at temperature `0.3` the first fifty tokens contribute 45.3
distinct pivots while tokens 150-200 contribute 10.0, and only half of the 200
positions are distinct.  The tempting explanation -- that the evidence
saturates -- is wrong, and the curve CSV carries `distinct_watermarked` and
`distinct_null` at every prefix so it can be checked.  Tr-GoF's *watermarked*
statistic keeps growing (median 11.1 at 100 tokens to 20.5 at 200 on
Sheared-2.7B at temperature `0.3`), and threshold-free separation barely moves
(AUC `.920` to `.899`).  What explodes is the empirical **null**: a k-fold tied
p-value is exactly the `t/n` jump Higher Criticism is built to flag, so the
null's 95th percentile inflates superlinearly -- `9.2` at 100 tokens to `23.6`
at 200, against a flat `~4.8` under i.i.d. uniform pivots -- until the cutoff
overtakes the watermarked median and Type II climbs from `.306` to `.570`.

Scoring only the distinct pivots restores Tr-GoF's monotonicity
(`.572/.408/.314/.292/.226` at 20/50/100/150/200).  It is a repair specific to
Tr-GoF: the eight cumulative-sum rules do not normalise by `n` at all, and
deduplicating them destroys power -- `h_ars` at 200 tokens goes from `.192` to
`.772` -- because for a sum a repeated pivot carries repeated signal.

Generation needs model weights; `--device auto` selects CUDA, then MPS, then
CPU. The analysis and figures need no weights or accelerator. Preserve the
device and `--batch` with the seed and software versions: batching changes the
assignment of random draws and can change generated text. Generation uses
fp32; at temperature
`0.1` the logits are scaled by ten and `1 - max p` is the quantity of interest,
and it already underflows to zero in 67% of the released float32 values.

```bash
# Replay stored arrays (no accelerator or weights).
export WM_SCRATCH="$PWD/results/bayesian_paper_benchmark/real_model/temperature_matched"
python3 code/analyse_temperature_matched.py --allow-legacy-archives
MPLCONFIGDIR=/tmp/bayesian-watermark-mpl \
    python3 code/plot_temperature_prefix_curves.py --allow-legacy-archives
cd code && python3 -m unittest -v test_plot_temperature_prefix_curves
```

The top-level README supplies the complete fresh-generation recipe, including
the clone and detached checkout of upstream commit
`05b7ffda9279fc9e645f38807e4a0e2dbcff4330`, prompt construction, and analysis in
an isolated output directory. The generator rejects a dirty or different
upstream checkout. Keep each prompt table's JSON sidecar; its file hash and the
selected-row hash are validated before generation. The historical NPZ arrays
remain usable for replay but predate the metadata required for resume.

## Generating-regime sweep

`regime_sweep.py` evaluates adaptation across generating regimes.  The family diagnostic in
the clean benchmark shows that a single fixed `Delta_0` under the generating
spike family reproduces the shared Bayes rule, but it measures that at one
generating configuration, `Delta ~ Uniform(0.001, 0.5)`, which is also the
prior.  A prior is supposed to buy adaptivity across configurations the analyst
cannot foresee, so this experiment holds every rule fixed and varies only the
law of the per-document deficit: `Uniform(.001,.5)`, `Uniform(.001,.05)`,
`Uniform(.2,.5)`, and point masses at `.005`, `.40`, `.70`, `.002`, `.0075` and
`.02`.  `.70` lies outside the prior's support.  Of the point masses only `.005`
coincides with a tested `Delta_0`, which would let the rule assuming it set the
baseline unopposed, so `.002`, `.0075` and `.02` were added below the tested grid
and inside its two smallest gaps.  The three high-deficit regimes
(`Uniform(.2,.5)`, `.40`, `.70`) saturate at every main horizon -- no rule misses
a document -- so the manuscript displays the other six; the sweep runs and
records all nine.

Gumbel scheme, shared-`Delta` hierarchy, equal-tail spike, `n` in
`{100, 300, 700}`, `M=1000`, seed `240401248`. The recorded rules include fixed
point-mass spike scores at `Delta_0` in `{.005, .01, .05, .2, .4}`, shared and
tokenwise Bayes with the *frozen* `Uniform(0.001, 0.5)` prior on 96
Gauss--Legendre nodes, and the Li et al. comparator `h*_gum,.005`.  The Gumbel pivot
null does not depend on the regime, so one 10,000-sequence calibration sample
serves all nine; the evaluation null is a fresh, separate sample.  The reported
summary is maximum regret across regimes, defined as a rule's excess Type II
error over the best eligible displayed rule in that regime. Point-deficit
diagnostics are excluded from this regret baseline. Its Monte Carlo standard
error uses 2,000 paired document-bootstrap draws that recompute the within-regime
minimum and across-regime maximum on every draw; the bootstrap is conditional
on the realized calibration cutoffs.

```bash
python3 code/regime_sweep.py
python3 code/regime_sweep.py --quick
cd code && python3 -m unittest -v test_regime_sweep
```

Results go to `results/bayesian_paper_benchmark/regime_sweep.json`, including
the Type II and regret tables, the max-regret ranking and standard errors, the
regime definitions and realized deficits, and a supplementary short-horizon probe recording where
the high-deficit regimes stop being saturated, at `n` in `{5, 10, 20, 50}`
under seed `240401249`.  The probe is recorded but not reported in the
manuscript.  No figures are produced.
Quick results instead go to
`results/bayesian_paper_benchmark/quick/regime_sweep/regime_sweep.json` unless
`--results-dir` is supplied.  “Best” and regret are relative only to the finite
eligible competitor set; they are not an optimization over every fixed
`Delta_0` or every possible prior.

## The Dirichlet tail layer as a standard competitor

`dirichlet_detector.py` is the single owner of the layer detector and is used by
every experiment here.  `DirichletBayesGrid` carries a joint
`(alpha, Delta, rho)` component grid (alpha slowest, rho fastest) and provides
the shared-latent log Bayes factor at **all** prefixes or at selected horizons,
with a running maximum for the anytime rule, plus a tabulated tokenwise
prior-predictive lookup.  `alpha = math.inf` is the equal-tail spike family, so
the layer *contains* the spike detector reported everywhere else; a test asserts
that collapsing the alpha prior to `{inf}` reproduces the published spike numbers
bit for bit at every prefix, for both hierarchies and both decision rules.

The layer therefore appears as a competitor in every Gumbel comparison:

| script | layer rules added |
|---|---|
| `benchmark_paper_experiment.py` | `bayes_shared_dirichlet`, `bayes_tokenwise_dirichlet` (fixed-horizon and anytime) |
| `regime_sweep.py` | `bayes_shared_dirichlet` |
| `benchmark_contamination.py` | `bayes_shared_dirichlet_robust`, `bayes_shared_dirichlet_clean` |
| `tail_regime_sweep.py` | the tail-shape sweep below |

It is **Gumbel-only** by construction: for the inverse pivot the limiting
alternative depends on `Delta` alone, so a prior on the tail shape is inert.
That is enforced structurally (`DIRICHLET_SCHEMES = ("gumbel",)`, a `None` grid in
the inverse scheme spec, and `inverse_score` raising `KeyError`) rather than by
convention.

Three operational notes.  The `psi_alpha` transform table defaults to 200,001
nodes.  On the manuscript's separate deterministic validation grid, direct
non-tabulated quadrature gives largest component log-density discrepancies of
9.5e-6 for the 20,001-node `tail_family` default and 9.5e-8 for 200,001 nodes.
These are empirical finite-grid diagnostics, not uniform error bounds; the
coarser discrepancy can accumulate to about .0067 over 700 tokens.  The
tokenwise rule adds an outer 80,001-node logit lookup after mixing the
components.  Checking every lookup-interval midpoint against the same frozen
mixture evaluated without this outer interpolation gives a maximum additional
log-ratio error of 2.12e-6; numerical integration gives numerator mass
`1 - 7.67e-9`.  Both values are diagnostics rather than certified bounds and
are serialized in `benchmark_summary.json`.  Finally,
`benchmark_paper_experiment.py --replot` redraws every figure from the existing
`benchmark_results.csv`, so a figure fix costs seconds rather than a full run.

```bash
MPLCONFIGDIR=/tmp/bayesian-watermark-mpl python3 code/benchmark_paper_experiment.py --replot
```

## The tail-width layer as a standard competitor

`dirichlet_detector.TailWidthBayesGrid` is the second enlargement of the Gumbel
spike family.  Where the Dirichlet layer keeps the tail `K = V - 1` coordinates
wide and puts a prior on how evenly the residual mass is spread, this layer
keeps the tail equal and puts a prior on **how many coordinates are live**:

\[
f_{\Delta,J}(r) = r^{\Delta/(1-\Delta)} + J\, r^{J/\Delta - 1},
\]

the exact Gumbel pivot density of `(1-Delta, Delta/J, ..., Delta/J)`.  `J = V-1`
is exactly the equal-tail spike, so the family contains that Bayesian component
and the finite-grid evidence bound applies unchanged; `J = 1`
is the sparsest tail at that deficit.

Two properties distinguish it from the Dirichlet layer.  It is **closed form**,
so there is no transform table, no interpolation, and no quadrature
normalisation diagnostic: every component integrates to one identically, and the
test-martingale statement applies to the density actually evaluated rather than
to a numerical approximation of it.  And the prior grid is **scale free** --
`dyadic_tail_width_grid` returns `1, 4, 16, ..., V-1`, whose last atom is the
equal tail whatever the vocabulary is.  That matters because the tail width the
detector assumes is the one quantity in the component family that depends on the
deployed vocabulary: at `V = 1000` the equal-tail term is active over a visible
range of `r`, while at the released `V = 50272` it is a numerical point mass at
`r = 1`.

This paragraph describes the closed-form `TailWidthBayesGrid` component family,
not the current union-tail detector. The latter combines a full-width Dirichlet
shape branch with width atoms strictly below `V-1`; the equal-tail atom is in
the shape branch. Its shape transform is interpolated, and tokenwise scoring
adds outer lookups. Numerical checks do not certify one-sided mass error or
an e-process guarantee. `experiment_metadata.py` records this distinction
consistently. The clean benchmark evaluates both shared and tokenwise union
rules, with the branch state held at document level. Other experiments may
evaluate only the shared rule; no general power inferiority of tokenwise
mixtures is implied. The inverse working alternative depends on `Delta` alone.

| script | rule added |
|---|---|
| `benchmark_paper_experiment.py` | `bayes_shared_uniontail`, `bayes_tokenwise_uniontail` |
| `regime_sweep.py` | `bayes_shared_uniontail` |
| `tail_regime_sweep.py` | `bayes_shared_uniontail` |
| `benchmark_contamination.py` | `bayes_shared_uniontail_clean`, `bayes_shared_uniontail_robust` |
| `real_data_experiment.py` | `bayes_shared_uniontail` |

Adding the rule leaves every pre-existing Type I and Type II number bit
identical, which is asserted by the existing per-experiment invariance tests.
The regret columns of `regime_sweep.json` and `tail_regime_sweep.json` are the
exception by construction: regret is measured against the best eligible rule in
each regime, excluding point-deficit diagnostics, so enlarging the menu can move
the regret of rules that did not themselves change.

## Tail-prior (Dirichlet layer) sweep

`tail_family.py` implements the Dirichlet tail layer for the Gumbel pivot: the
leading NTP coordinate is `1 - Delta` and the remaining `K = M - 1` carry
`Delta * q` with `q ~ Dirichlet(alpha, ..., alpha)`.  Because the pivot density
is a sum over coordinates, the prior passes inside by linearity and
exchangeability leaves a single univariate transform per `alpha`,
`psi_alpha(c) = E_q[exp(-c/q)]`, tabulated once and reused for every
`(Delta, r)`.  The tail quadrature nodes are rescaled so that `K E[q] = 1`
exactly, which keeps each *non-interpolated* component a probability density and
its Bayes factor a null martingale.  The fast experiment code interpolates a
tabulated transform; its normalization is checked numerically but not enclosed
by a certified one-sided error bound.  Exact Ville validity therefore belongs
to the analytic/non-interpolated construction, while the recorded interpolated
paths are a high-accuracy numerical approximation.  `alpha = math.inf` is the
equal-tail spike family, so the layer contains the detector used everywhere
else as a special case.

`tail_regime_sweep.py` is the companion experiment.  It holds every rule fixed
and varies only the generating tail law: Dirichlet at `alpha` in
`{inf, 10, 3, 0.5, 0.1}` plus the normalized-uniform tail used by the released
simulator of Li et al. (2025), which is outside the Dirichlet family at every
`alpha`.  One
`Delta ~ Uniform(0.001, 0.5)` per document, shared by its tokens; the Gumbel
null is `Uniform(0,1)` whatever the tail looks like, so one 10,000-sequence
calibration sample serves all configured regimes and the evaluation null is a fresh,
separate sample.  Competitors are the layer at a single assumed `alpha_0` in
`{0.1, 1, 10, 100, 1000}`, the layer at `alpha = inf`, the point-`Delta_0` spike
diagnostic `h^sp_.01`, and the Li et al. comparator `h*_gum,.005`.  The new rule
is the layer under the frozen product prior
`Uniform(0.001, 0.5) x Uniform{0.1, 1, 10, 100, 1000, inf}` on 96
Gauss--Legendre deficit nodes, i.e. 576 joint components.  The reported summary
is maximum Type II regret across the six tail laws.  Seed `240401250`; `M=1000`;
`n` in `{100, 300, 700}`.  Its Monte Carlo standard error uses 2,000 paired
document-bootstrap draws that recompute the best rule within each law and the
maximum across laws; it is conditional on the realized calibration cutoffs.

The current output also includes four restricted-width laws (W1--W4), the
union-tail method, and the full reference-score menu. It reports shape-only,
width-only, and combined regret summaries; the six-law description above
specifies the original shape block.

```bash
python3 code/tail_regime_sweep.py
python3 code/tail_regime_sweep.py --quick
cd code && python3 -m unittest -v test_tail_regime_sweep
```

Results go to `results/bayesian_paper_benchmark/tail_regime_sweep.json`,
including the Type II and regret tables, the max-regret ranking and standard
errors, per-regime size-biased tail moments, the frozen-prior fingerprint, and a validation block
recording that the `alpha = inf` member reproduces the closed-form spike density
and giving a numerical (not certified one-sided) normalization diagnostic for
the interpolated components.  No figures are produced.  A fast smoke run is
`python3 code/tail_regime_sweep.py --quick`; it writes under
`results/bayesian_paper_benchmark/quick/tail_regime_sweep/` unless
`--results-dir` is supplied.  “Best” and regret are relative only to the listed
point-alpha rules (`0.1`, `1`, `10`, `100`, `1000`, and `inf`) plus the other
listed methods.  Every atom of the frozen alpha prior appears as a point-rule
competitor; the sweep does not optimize over all positive `alpha` or all priors.

## Contamination robustness study

`benchmark_contamination.py` evaluates the contamination layer directly.  It
draws one `Delta` per document and independently replaces each clean
watermarked pivot by an exact-null pivot with probability `rho_true` in
`{0, .1, .25, .4, .6, 1}`.  Within each pivot scheme, clean paths, null
replacement paths, and nested masks are shared across methods and
contamination levels.

The primary comparison includes the reference scores evaluated by Li et al. (2025), shared Bayes with `rho=0`,
and shared Bayes with the frozen prior
`0.5 delta_0 + 0.125(delta_.1 + delta_.25 + delta_.4 + delta_.6)`.  Every
statistic is calibrated separately on one independent null sample per pivot
scheme, common to all methods within that scheme, and keeps that cutoff across
the contamination sweep.  The `rho_true=1` cell is a
diagnostic: the alternative is then exactly the null, so power should equal
the evaluation Type I rate.

```bash
MPLCONFIGDIR=/tmp/bayesian-watermark-mpl python3 code/benchmark_contamination.py
MPLCONFIGDIR=/tmp/bayesian-watermark-mpl python3 \
    code/benchmark_contamination.py --quick
cd code && python3 -m unittest -v test_benchmark_contamination
```

Outputs are written beside the clean benchmark results as
`contamination_results.csv`, `contamination_summary.json`,
`contamination_rejection_indicators.npz`,
`contamination_rho_diagnostic.json`, `contamination_report.md`, and
publication/slide PDF and PNG figures.  Aggregate errors use the expected
randomized-boundary rate; the NPZ stores one reproducible, commonly coupled
realization for paired Wilson/McNemar summaries.  This is
a robustness study of iid null-like dilution only; it does not simulate burst
edits, paraphrase dependence, insertion/deletion alignment, or key
desynchronization.  Quick artifacts go under
`results/bayesian_paper_benchmark/quick/contamination/` unless `--output-dir`
is supplied, so a smoke test cannot silently replace the recorded full run.

### Why the rho model matters for inverse pivots and not for Gumbel

`contamination_asymmetry.py` explains the flat Gumbel column of the
contamination table.  Replacement does degrade the Gumbel rules, but modelling
`rho` produces little improvement in this design, and the script measures why.
The Gumbel component has integrable log-density endpoints: its null expected
log loss is finite, although its pointwise log loss is unbounded near zero.
With `D1` the component-versus-null
Kullback--Leibler divergence and `D0` its reverse, a fixed-`Delta` clean rule
still drifts upward while `rho_true < D1/(D1+D0)`, a break-even rate of `0.73`
to `0.84` at `M=1000`.  The limiting inverse alternative instead lives on
`(0, 1-Delta)` while its null lives on `(0, 1)`, so a single null-like pivot `d`
annihilates every component with `Delta > 1-d` and contamination becomes a hard
constraint on the deficit posterior rather than a finite expected null cost.

The script reports the break-even rates by quadrature, the terminal log Bayes
factor of the shared clean-`rho` mixture against `rho_true`, the realized
deficit ceiling and surviving prior mass for inverse pivots, an exact McNemar
replay of clean against robust from the stored indicators, and a comparison of
each contaminated error with the same rule on `(1-rho_true) n` clean tokens in
the independent benchmark.  Seed `240401251`; `M=1000`; `n=700`.  It consumes
`benchmark_results.csv`, `contamination_results.csv`, and
`contamination_rejection_indicators.npz`, so run it after both benchmarks.

```bash
python3 code/contamination_asymmetry.py
cd code && python3 -m unittest -v test_contamination_asymmetry
```

Results go to `results/bayesian_paper_benchmark/contamination_asymmetry.json`.
The break-even identity is a per-token drift statement for a fixed `Delta`; it
bounds neither calibrated power nor the mixture, and the conclusion is specific
to independent null-like replacement.  No figures are produced.

### How wide is the released tail?

`estimate_tail_width.py` examines the equal-tail Bayesian Gumbel model's
assumption that residual mass is spread over all `V-1` non-leading coordinates.
The released `top_probs` pin `Delta_t` exactly, so conditional on it the exact Gumbel density
`f_{Delta,J}(r) = r^{Delta/(1-Delta)} + J r^{J/Delta - 1}` has `J` as its only
free parameter and it can be profiled out.

The primary analysis selects the first use of each PRF vector address
`W_{t-4}` on the full sequence, then retains positions with
`.05 <= Delta <= .5`. Selecting by the observed pivot value instead is not a
valid independence correction: the pair `(W_{t-4}, W_t)` identifies the selected
pivot coordinate, not the PRF vector address. This leaves 1,083 positions in
385 contributing OPT documents and 989 positions in 368 Sheared documents. Dependence within a
document is retained in 2,000 document-bootstrap replicates.

The effective width estimates are `J=2` for OPT and `J=1` for Sheared. Their
bootstrap selection frequencies are 65.75% and 94.25%, respectively; these are
descriptive stability measures, not calibrated significance tests. The fitted
probability-integral-transform KS statistics are `.034` and `.045`, compared
with `.163` and `.152` under full width. The corresponding full-width nominal
KS p-values are about `1.3e-25` and `2.7e-20`; within-document dependence and
parameter fitting prevent interpreting ordinary KS p-values as exact tests.

The recovery check uses 200 replicates at each actual sample size. It recovers
`J=1` in every replicate; `J=2` is recovered in 98.5% of OPT-sized and 96% of
Sheared-sized samples. Across tested truths `{1, 2, 4, 16, 256}`, recovery is
89.5%-100% for OPT and 93%-100% for Sheared. These checks support the narrow-tail
family while allowing uncertainty about a particular width. A real tail is not
exactly equal, so the fitted `J` describes the residual mass's effective spread,
not a literal count of nonzero probabilities.

Seed `240401261`; writes
`results/bayesian_paper_benchmark/tail_width_estimate.json`.  No figures.

```bash
python3 code/estimate_tail_width.py
cd code && python3 -m unittest -v test_estimate_tail_width
```

### Why tail sparsity strains the inverse working alternative

`inverse_support_edge.py` reaches the contamination asymmetry by a second route.
The limiting inverse alternative is supported on `(0, 1-Delta)` and is derived
under `(log M) p_(2) -> 0`.  A tail with `J` live coordinates has
`p_(2) = Delta/J`, so that quantity is `Delta (log M)/J` and does **not**
vanish: at `M=50272`, `Delta=.1`, `J=1` it is `1.08`. This historical diagnostic
uses independent continuous rank locations, not distinct finite-vocabulary
ranks. Its large-vocabulary approximation error has not been bounded. It
compares mean pivots with `(1-Delta)/3` and records mass at or beyond
`1-Delta`, where the working
numerator is exactly zero: `.0008`, `.0177` and `.0847` at `Delta = .1, .3, .5`
for `J=1`, against at most `1e-4` once `J>=16`.

The decisive part is the `rho` comparison.  Because `rho + (1-rho) L >= rho`, a
positive `rho` bounds the component ratio away from zero.  Generating from a
sparse tail with `rho_true = 0`, so that no contamination exists anywhere, the
frozen `rho` prior *lowers* Type II error from `.1267` to `.1140` at `J=1`
(discordance 37 versus 18, exact McNemar `p=.014`) and reverts to a cost once
the edge mass vanishes, `.0920` against `.1033` at `J=256` (`p=.002`).  The sign
tracks the edge mass, not any contamination rate, so averaging over `rho` is
better read as insurance against alternative mass falling where the working
component assigns none than as a model of editing alone.

Seed `240401260`; `M=50272`; `n=200`; 1,500 documents; 3,000 calibration paths.
About 20 s; writes `results/bayesian_paper_benchmark/inverse_support_edge.json`.
No figures.

```bash
python3 code/inverse_support_edge.py
python3 code/inverse_support_edge.py --quick
cd code && python3 -m unittest -v test_inverse_support_edge
```

Minimal usage:

```python
import numpy as np
from bayesian_watermark import SequentialBayesDetector, beta_grid_prior

delta, weight = beta_grid_prior(2, 3, low=0.01, high=0.6, size=101)
detector = SequentialBayesDetector(
    "gumbel",
    delta,
    weight,
    rho_grid=(0.0, 0.1),
    rho_weights=(0.8, 0.2),
    gumbel_family="least_favorable",
    vocabulary_size=50,
    prior_watermark=0.1,
)
detector.update_many(np.array([0.83, 0.91, 0.74]))
print(detector.log_bayes_factor)
print(detector.posterior_probability())
```

For a production deployment, estimate priors on held-out data and freeze them
before evaluating a document. Reusing the evaluated text to tune the prior
breaks the stated anytime calibration.
