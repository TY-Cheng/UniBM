# Validation

<p class="unibm-case-intro">
Stationary synthetic benchmarks compare severity and extremal-clustering estimates against
known targets under serial dependence. The clearest gains from covariance-aware FGLS are in
interval performance, with smaller changes in point accuracy.
</p>

## Design and scoring

The main production grid uses **N=365 observations and M=100 independent
simulated records per scenario**, with master seed `20260401`. M counts repeated
datasets whose true parameter is known. R instead counts bootstrap draws within
one of those datasets; increasing R does not increase M.

| Branch | Process families | True parameters | Scenarios |
|---|---|---|---|
| EVI | Fréchet max-AR, moving maxima (q=99), Pareto additive AR(1) | `xi = 0.01, 0.03, 0.10, 0.30, 1, 3, 10`; `theta = 0.01, 0.10, 0.50, 1` | 84 |
| EI | Same three families | `xi = 0.01, 0.50, 1, 5`; `theta = 0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 1` | 84 |

All internal FGLS methods use **fixed diagonal shrinkage 0.37**, the declared
plateau/stable-window rules, and **adaptive R at 128/256/512/768/1024**. Shrinkage
0.37 is a pre-specified computational default, not an optimum selected from the
sensitivity analysis. External baselines use pre-specified, method-specific
tuning rules consistently across scenarios. OLS and the external estimators
retain their own interval constructions.

The primary interval metric is the **arithmetic mean 95% Winkler score** on the
parameter's original scale. For an interval `[l, u]` and known true value `t`,
each score is:

```text
W = (u - l) + 40 * max(l - t, 0) + 40 * max(t - u, 0)
```

Lower is better. The **grid-average score** is the equally weighted mean of the
scenario-level means. Scores are **not normalized**: scenarios with larger
interval widths and noncoverage penalties contribute more to this average.
Scenario wins and grid-average scores therefore describe different aspects of
performance. Empirical coverage complements the score by showing how often
intervals contain the true parameter.

At each plotted scenario, score bars show the mean **plus or minus one outer
Monte Carlo standard error (MCSE)**, `sd(W) / sqrt(n_score)`. These bars are not 95% confidence
intervals and are not the internal MCSE used to select R. The secondary point
estimation metric remains median absolute percentage error, with its 25th–75th
percentile range. APE is displayed as a fraction, so `1.0` means 100% error.

## Severity branch (EVI)

The EVI suite varies the true tail index, extremal dependence, and process family at a short
record length. The full comparison contains 12 internal methods (three summaries,
two block schemes, two regressions) and four external reference estimators. The
figure shows the six core methods; the CSV outputs retain all methods, including
the mean/mode and external comparisons.

Within the BM framework, **median-sliding-FGLS has the lowest grid-average
Winkler score** and wins 27 of 84 scenarios; all within-BM scenario winners use
FGLS. For Fréchet max-AR at `theta=0.10`, its score averaged over the `xi` grid
falls from 39.55 under OLS to 8.32 under FGLS, while median APE remains
0.55.

<figure class="unibm-figure">
  <a href="../assets/validation/evi_benchmark.png">
  <img src="../assets/validation/evi_benchmark.png" alt="EVI synthetic benchmark panels comparing estimation error and interval score across tail severity, dependence, and process families.">
  </a>
  <figcaption>Six core EVI methods. Mean Winkler score uses ±1 outer MCSE; median absolute percentage error uses its IQR. Each point summarizes M=100 records in one scenario. Click the figure for full size.</figcaption>
</figure>

[Download all EVI method/scenario summaries (CSV)](assets/validation/evi_benchmark.csv).
The external comparison considers seven methods: median-, mean-, and mode-based
sliding-FGLS, Hill, max-spectrum, Pickands, and DEdH moment. Within this set,
**median-sliding-FGLS is the only method that never ranks last by mean Winkler
score across the 84 scenarios**. Hill wins 22 scenarios and median-sliding-FGLS
wins 19, but their grid-average scores are 25.81 and 9.49, respectively.

Hill and max-spectrum often have lower APE, particularly under weaker
dependence. Among the sliding-FGLS variants, Winkler scores for mean and mode
targets rise more steeply at large `xi`. At `xi=0.01`, the near-zero denominator
can magnify APE. The EVI grid also exhibits undercoverage; read `coverage` and
`n_score` alongside `interval_score_mean`.

## Persistence branch (EI)

The EI suite varies the true extremal index, tail severity, and process family.
The chart below compares eight pooled-BM estimators on the dimensionless `θ`
scale.

**BB-sliding-FGLS has lower mean Winkler scores than its pooled OLS counterpart
in all 84 scenarios.** This improvement also holds for the other combinations
of EI path and block scheme. Median APE changes little between sliding-OLS and
sliding-FGLS. For BB-sliding, the mean interval width across the grid increases
from 0.0093 under OLS to 0.1352 under FGLS, alleviating overly narrow intervals.
BB-sliding-FGLS and Northrop-sliding-FGLS have the two lowest within-BM
grid-average scores and together win 54 of 84 within-BM scenarios.

<figure class="unibm-figure">
  <a href="../assets/validation/ei_benchmark.png">
  <img src="../assets/validation/ei_benchmark.png" alt="EI synthetic benchmark panels comparing estimation error and interval score across persistence, tail severity, and process families.">
  </a>
  <figcaption>Eight internal pooled EI methods: BB/Northrop × sliding/disjoint × OLS/FGLS. Mean Winkler score uses ±1 outer MCSE; median absolute percentage error uses its IQR. Click the figure for full size.</figcaption>
</figure>

The eight-method external comparison combines the four pooled-BM FGLS variants
with Ferro–Segers, K-gaps, native Northrop sliding, and native BB sliding on the
same simulated records. **BB-sliding-FGLS and native BB are the only methods in
this comparison that never rank last by mean Winkler score.** BB-sliding-FGLS
outperforms native BB in 81 of 84 scenarios, with grid-average scores of 1.61
and 2.08, respectively.

K-gaps has the lowest grid-average score, 1.00, and wins 59 of 84 scenarios.
It ranks last in all 12 scenarios with `theta=1`. These results distinguish
BB-sliding-FGLS's consistent within-BM improvement from the advantages of
threshold-based estimation in much of the examined grid.

[Download all EI method/scenario summaries (CSV)](assets/validation/ei_benchmark.csv).
`interval_score_mean` is the primary score; `n_rep` counts outer simulated
records, not bootstrap R. The grid exhibits substantial undercoverage, so a
nominal 95% interval must not be read as having achieved 95% empirical coverage.

## Numerical precision and failed fits

An adaptive fit that reaches R=1024 without meeting its numerical precision
tolerance is **retained and scored**, with `bootstrap_precision_met=False`.
This is different from an unusable covariance: strict FGLS does not
fall back to OLS. Recorded failed estimates count as misses in coverage and have
no finite interval score. The mean score is consequently conditional on valid
scored fits; read `n_score` and failure counts alongside it.

The detailed internal CSVs record actual R, the precision flag, maximum MCSE
ratio, and shrinkage. A missing precision flag means it was not assessed for
that fit, not that it passed. Adaptive stopping is conditional on the observed
fit window and does not establish CI coverage or design-life-level precision.

In this run, 3,010 of 50,400 EVI FGLS fits and 2,100 of 33,585 successful EI
FGLS fits reached the cap without meeting the tolerance. Fifteen EI FGLS
attempts failed because of degenerate covariance (seven BB-sliding and eight
Northrop-sliding, all at true `theta=1`). The EVI DEdH-moment reference has 421
unscored attempts out of 8,400. These are disclosed separately from adaptive
precision-unmet fits.

## What is and is not validated

| Checked here | Boundary |
|---|---|
| Known-target estimation error and parameter-interval coverage | Only the simulated families, parameter grid, and record length N=365 |
| Error and interval-score comparisons | Finite Monte Carlo design, not an asymptotic proof |
| Sensitivity to dependence and tail severity | Not robustness to arbitrary nonstationarity or model misspecification |
| Current fixed-shrinkage/adaptive-R workflow | No claim that data-driven selection uncertainty is fully absorbed |
| EVI/EI parameter intervals | Coverage of derived design-life intervals has not been evaluated |

The [case-study pages](cases/index.md) have also been recomputed from their
unchanged frozen provider inputs using fixed shrinkage `0.37` and adaptive R.
Those real-data illustrations are separate from the synthetic validation and
do not supply known-truth coverage checks for extrapolated dollar/discharge
levels.

## Reproduction and scope

After source installation with development dependencies, force the two main
grids from the repository root:

```bash
UNIBM_FORCE_BENCHMARK=1 uv run python scripts/benchmark/evi_benchmark.py
UNIBM_FORCE_BENCHMARK=1 uv run python scripts/benchmark/ei_benchmark.py
```

`UNIBM_BENCHMARK_WORKERS` optionally limits parallel workers. These commands
reuse the cached raw simulated records but recompute the estimates. Adaptive
fitting bypasses old fixed-R covariance caches. The canonical outputs are
`out/benchmark/{detail,summary,external_detail,external_summary}.csv` and the
matching `ei_` files. Report functions in `scripts/benchmark/evi_report.py` and
`ei_report.py` generate the static plots from those summaries.

This page covers the two complete **main grids**, including their external
comparators. Neither a site build nor these two commands downloads provider
data or reruns applications.

## Shrinkage sensitivity

The shrinkage sensitivity analysis varies
`delta = 0, 0.15, 0.37, 0.55, 0.75, 1`, using mean Winkler score as the primary
metric. It covers median-sliding-FGLS EVI and BB/Northrop sliding-FGLS EI on
the same main grids. Median coverage and median APE remain secondary
diagnostics. Family summaries give equal weight to scenario-level mean scores,
with no normalization.

For each simulated record, the original selected window and the bootstrap draws
obtained by adaptive stopping at `delta=0.37` are reused across the grid.
R is not independently retuned at each delta, and the default-fit precision
flag does not certify the other delta fits. Failed EI covariances are retained
as noncoverage, with no finite score, at every delta.

Diagonal shrinkage reduces the grid-average EVI score from 27.466 at `delta=0`
to 9.485 at the fixed default. The lowest grid-average score is 8.897 at 0.75,
while family-specific minima occur at 0.75 for Fréchet max-AR, 0.37 for moving
maxima, and 1.00 for Pareto additive AR(1). Median APE changes little across
positive shrinkage values.

The two EI scores are 1.607 (BB) and 1.648 (Northrop) at the default, with minima
of 1.561 and 1.601 at 0.15. Full diagonal shrinkage worsens these scores to
2.141 and 2.179 and reduces median coverage. **The common default remains
0.37.** These descriptive comparisons assess sensitivity; they neither select
a new default nor establish nominal coverage.

The existing report entrypoints generate the appendix CSVs when absent and
refresh the manuscript figures:

```bash
uv run python scripts/benchmark/evi_report.py
uv run python scripts/benchmark/ei_report.py
```

The corresponding `build_evi_shrinkage_sensitivity_summary` and
`build_ei_shrinkage_sensitivity_summary` functions accept `force=True` to
recompute an existing sensitivity CSV. Outputs remain under `out/benchmark/`;
no separate validation suite or intermediate storage format is introduced.
