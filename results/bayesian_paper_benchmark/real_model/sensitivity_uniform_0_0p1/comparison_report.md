# Real-output sensitivity: Uniform(0, 0.1) deficit prior

This exploratory run changes only the Bayesian deficit prior from the
canonical `Delta ~ Uniform(0.001, 0.5)` to `Delta ~ Uniform(0, 0.1)` in the
released-output analysis.  It uses the same two models, released documents,
seed 240401253, 10,000 exact-null calibration paths, 96 Gauss--Legendre nodes,
Gumbel tail prior, and `rho=0`.  Every Bayesian cutoff is recalibrated.  The
canonical CSV, JSON, and indicator archive are not overwritten.

The endpoints are measure-equivalent to an open interval, and every quadrature
node lies strictly inside `(0,0.1)`.  This prior has mean and median 0.05,
compared with 0.2505 under the canonical prior.  It has no atom at zero.

## Results at n=200

Entries are empirical Type I or Type II error with the across-document Monte
Carlo standard error in parentheses.  The standard errors treat the 500
released-document contributions as independent and are conditional on the
calibrated cutoffs.  They do not include cutoff-calibration, prompt-selection,
generation, key-to-key, or model uncertainty.

| Model | Scheme | Bayesian rule | Type I, canonical | Type I, trial | Type II, canonical | Type II, trial |
|---|---|---|---:|---:|---:|---:|
| OPT-1.3B | Gumbel | tokenwise | .046 (.0094) | .042 (.0090) | .552 (.0223) | .570 (.0222) |
| OPT-1.3B | Gumbel | shared | .042 (.0090) | .042 (.0090) | .566 (.0222) | .566 (.0222) |
| OPT-1.3B | Gumbel | tokenwise + tail | .060 (.0106) | .050 (.0098) | .574 (.0221) | .590 (.0220) |
| OPT-1.3B | Gumbel | shared + tail | .044 (.0092) | .044 (.0092) | .568 (.0222) | .568 (.0222) |
| Sheared-LLaMA-2.7B | Gumbel | tokenwise | .042 (.0090) | .044 (.0092) | .590 (.0220) | .618 (.0218) |
| Sheared-LLaMA-2.7B | Gumbel | shared | .048 (.0096) | .048 (.0096) | .616 (.0218) | .616 (.0218) |
| Sheared-LLaMA-2.7B | Gumbel | tokenwise + tail | .044 (.0092) | .046 (.0094) | .622 (.0217) | .654 (.0213) |
| Sheared-LLaMA-2.7B | Gumbel | shared + tail | .036 (.0083) | .036 (.0083) | .624 (.0217) | .624 (.0217) |
| OPT-1.3B | inverse | tokenwise | .064 (.0110) | .050 (.0098) | .604 (.0219) | .608 (.0219) |
| OPT-1.3B | inverse | shared | .052 (.0099) | .054 (.0101) | .616 (.0218) | .614 (.0218) |
| Sheared-LLaMA-2.7B | inverse | tokenwise | .028 (.0074) | .028 (.0074) | .626 (.0217) | .652 (.0213) |
| Sheared-LLaMA-2.7B | inverse | shared | .032 (.0079) | .030 (.0076) | .664 (.0211) | .664 (.0211) |

The lowest Bayesian Type II errors under the canonical and trial priors are,
respectively, .552 versus .566 for OPT Gumbel, .590 versus .616 for Sheared
Gumbel, .604 versus .608 for OPT inverse, and .626 versus .652 for Sheared
inverse.  Thus the trial prior does not improve the best observed Bayesian
result in any of the four model--scheme cells.

For the Sheared Gumbel tokenwise rule, Type II error rises by .028: 20 released
documents are detected only by the canonical-prior rule and 6 only by the trial
rule (two-sided exact McNemar p=.00936).  For its tokenwise tail-layer version,
the rise is .032, with discordances 23 versus 7 (p=.00522).  These p-values are
descriptive because the prior was selected after inspecting earlier analyses;
no multiplicity correction can make this adaptive comparison confirmatory.

Every reference-score row and randomized indicator is identical across the two
runs.  The boundary-integrated calibration rejection rate is exactly .05 in
every trial Bayesian cell.  A separate 192-node run changes no rate, boundary
count, randomized rejection count, or released-document indicator; its largest
cutoff change relative to 96 nodes is 7.9e-14.

The narrower prior includes the recorded zero deficits in its support closure,
but it assigns no probability atom to zero.  Across the four released
model--scheme arrays, 66.8%--69.9% of the float32 deficits are recorded as zero,
while 6.7%--7.3% exceed 0.1.  The deterioration of the tokenwise rules is
consistent with repeatedly averaging over alternatives close to the null;
the unchanged shared-Gumbel decisions indicate that document-level likelihood
accumulation is much less sensitive to this prior change on these samples.
