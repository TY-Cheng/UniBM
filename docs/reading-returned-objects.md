# Reading Returned Objects

The main UniBM entrypoints return lightweight dataclasses rather than raw
tuples. Start with the fits created in [Worked Examples](worked-examples.md);
this page explains their fields and inference limits.

## EVI fits

Read the result in this order:

- `fit.slope` is the headline UniBM `xi` estimate
- `fit.confidence_interval` gives the uncertainty interval for `xi`
- `fit.regression_policy` records the requested `OLS`, `FGLS`, or `AUTO` policy
- `fit.regression` records the `OLS` or `FGLS` estimator actually used
- `fit.ci_variant` is `hc0` for OLS or `bootstrap_cov` for FGLS
- `fit.plateau_bounds` shows which block-size window supported the fit
- `fit.bootstrap` stores the bootstrap metadata and covariance inputs used by
  FGLS fitting
- `design_life` contains the design-life-level estimates on the original data
  scale

The remaining fields such as `curve` and `plateau` are mainly for plotting,
diagnostics, and workflow-side reuse.

`OLS` does not bootstrap. `FGLS` and `AUTO` use adaptive bootstrap repetitions by
default unless a labeled `bootstrap_result` is supplied. `bootstrap_reps="adaptive"`
is explicit; an integer such as `bootstrap_reps=480` uses exactly that fixed budget.
The checkpoints are 128, 256, 512, 768, and 1024, with fixed diagonal shrinkage 0.37.
FGLS fails closed when
covariance is unavailable or invalid; only `AUTO` can fall back to OLS, and
malformed caller-supplied covariance is always an error.

## EI fits

Read the result in this order:

- `fit.theta_hat` is the headline extremal-index estimate
- `fit.confidence_interval` gives the uncertainty interval for `theta`
- `fit.stable_window` shows which block-size region was pooled
- `fit.base_path` and `fit.regression` record which BM path and pooling rule
  produced the estimate
- `fit.ci_variant` identifies the interval construction actually used
- `fit.standard_error` is on the `theta` scale; `fit.z_standard_error` is the
  pooled regression-scale SE, before transforming back from `z = log(1 / theta)`

The path-level fields are supporting diagnostics:

- `fit.path_level` records the observed block sizes retained on the finite path
- `fit.path_theta` and `fit.path_eir` retain the observed path values for
  plotting and method audits

## FGLS versus OLS

Pooled EI fits always pool the observed stable-window path. If you switch from
`regression="OLS"` to `regression="FGLS"`, the observed path is still what
gets pooled. The bootstrap result only contributes the cross-block covariance
matrix used for FGLS weighting.

`bootstrap_bm_ei_path` likewise defaults to `reps="adaptive"`; passing an integer
retains fixed-R sampling. It checks pooled `theta` and CI endpoints, plus the
unconstrained `z` fit and endpoints so clipping at `theta=1` cannot conceal
Monte Carlo error. Supply the same `covariance_shrinkage` to bootstrap and fit
when overriding the default 0.37. The pooled estimator still requires explicit
usable covariance for FGLS and never silently falls back to OLS.

## Reading adaptive precision

The following fields are present on both EVI and EI fit objects:

| Field | Meaning |
|---|---|
| `bootstrap_reps_policy` | `adaptive`, `fixed`, or no bootstrap policy |
| `bootstrap_reps_used` | Number of bootstrap path draws used for covariance |
| `bootstrap_precision_met` | `True`: tolerance met; `False`: cap reached without meeting it; `None`: not assessed for this fit |
| `bootstrap_mcse_targets`, `bootstrap_mcse` | Matching target names and estimated Monte Carlo standard errors |
| `bootstrap_mcse_max_ratio` | Largest MCSE divided by its statistical-SE denominator |
| `covariance_shrinkage` | Shrinkage actually used for FGLS |

Adaptive sampling uses eight equal delete groups and two random partitions,
refitting the **same observed window** after deleting bootstrap rows. It stops
when every target's estimated MCSE is at most 10% of its statistical-SE
denominator. The endpoint checks use the corresponding parameter's statistical
SE as the denominator, not a separately estimated endpoint SE.

EVI checks `xi` and its two CI endpoints, **not** extrapolated design-life levels.
EI checks `theta` and its endpoints as well as the unconstrained `z` fit and its
endpoints. At `R=1024`, an unmet tolerance produces a warning and retains the fit
with `bootstrap_precision_met=False`. Fixed-R fits are not automatically assessed.
Changing a reused fit's window or shrinkage clears its old precision claim.

This controls conditional numerical error from a finite bootstrap budget. It
does not guarantee CI coverage, account for selection uncertainty, or ensure
small MCSE for a 50-year design-life level. Increasing R does not add observed
extreme events and need not narrow a statistical CI.

New application CSV/JSON exports include these diagnostics. The
[case-study snapshots](cases/index.md) have not been rerun with the current
defaults; rebuilding this site alone does not update their statistical results.
