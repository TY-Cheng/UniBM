# Validation

<p class="unibm-case-intro">
Synthetic benchmarks test whether the finite-sample workflow recovers known severity and
persistence targets under controlled serial dependence. They support method comparison inside
the declared grid; they do not guarantee performance for every process or real-data regime.
</p>

## Severity branch (EVI)

The EVI suite varies true tail severity, extremal dependence, and process family at a short
record length. The figure compares median absolute percentage error and 95% Winkler interval
score across the predeclared grid. Lower is better, but coverage and interval width must be read
together: a narrow interval is not useful if it systematically misses the target.

<figure class="unibm-figure">
  <img src="../assets/validation/evi_benchmark.png" alt="EVI synthetic benchmark panels comparing estimation error and interval score across tail severity, dependence, and process families.">
  <figcaption>Core EVI benchmark summary generated from the frozen benchmark tables. Error bars summarize Monte Carlo variation across the scenario grid.</figcaption>
</figure>

## Persistence branch (EI)

The EI suite varies true persistence, tail severity, and process family. It compares the pooled
block-maxima estimators on a common `θ` scale. Threshold and native estimators remain useful
comparators, but the chart below focuses on the internal pooled-BM family used by the workflow.

<figure class="unibm-figure">
  <img src="../assets/validation/ei_benchmark.png" alt="EI synthetic benchmark panels comparing estimation error and interval score across persistence, tail severity, and process families.">
  <figcaption>Core EI benchmark summary generated from the frozen benchmark tables.</figcaption>
</figure>

## What is and is not validated

| Checked here | Boundary |
|---|---|
| Recovery of known `ξ` and `θ` targets | Only the simulated families, parameter grid, and record lengths |
| Error and interval-score comparisons | Finite Monte Carlo design, not an asymptotic proof |
| Sensitivity to dependence and tail severity | Not robustness to arbitrary nonstationarity or model misspecification |
| Fixed workflow defaults | No claim that data-driven selection uncertainty is fully absorbed |

The case-study intervals should therefore be read as finite-sample, benchmark-supported
intervals conditional on the selected window and fixed regularization defaults.
