# Independent null-like contamination benchmark

Within each contamination stratum, `rho_true` is fixed across positions. The study replaces each clean watermarked pivot by an independent exact-null pivot with probability `rho_true`; one Delta is drawn per document.

The robust prior is frozen at `0.5 delta_0 + 0.125(delta_.1 + delta_.25 + delta_.4 + delta_.6)`. Each pivot scheme has one exact-null calibration sample, common to all methods within that scheme and independent of evaluation; the method-specific cutoffs are reused at every contamination rate.

## Headline results at n=700, rho_true=0.4

| Scheme | Robust shared Bayes | Robust shared + Dirichlet tail | Clean shared Bayes | Prespecified reference score |
|---|---:|---:|---:|---:|
| Gumbel | 0.0028 | 0.0028 | 0.0028 | 0.0130 |
| Inverse | 0.0986 | -- | 0.2220 | 0.1526 |

The prespecified reference scores are `h*_gum,.005` and `h*_dif,.01`; they were evaluated by Li et al. (2025), selected from the independent clean n=700 benchmark, and then frozen before the contamination sweep.

All table entries are Type II error. At `rho_true=1`, the alternative equals the null. The maximum absolute difference between simulated power and the corresponding evaluation Type I rate is 0.0076 (1.70 Monte Carlo SE).

Inverse data use the exact finite-V equal-tail generator, whereas inverse Bayes uses the large-V triangular alternative with the exact finite-V null. The result establishes robustness only to iid null-like replacement. It does not model burst edits, context-dependent paraphrase, insertion/deletion alignment, or key desynchronization.
