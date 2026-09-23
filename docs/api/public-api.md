# API Overview

Choose an entrypoint by the quantity you want to estimate. EVI and EI estimators
are public through `unibm.evi` and `unibm.ei`. The current source also exposes
six core workflow functions directly from `unibm`. These convenience imports
refer to the same function objects; their placement does not select a default
estimator for every task. See the [version note](#top-level-convenience-imports)
below when using PyPI 0.1.0.

## Estimator map

| Task | Public entrypoints |
|---|---|
| EVI from block-maxima quantiles, including the median | [`estimate_evi_quantile`][unibm.evi.estimate_evi_quantile] |
| Scaling from block-maxima means or modes | [`estimate_target_scaling`][unibm.evi.estimate_target_scaling] with `target="mean"` or `"mode"` |
| EVI comparator estimators | [`estimate_hill_evi`][unibm.evi.estimate_hill_evi], [`estimate_pickands_evi`][unibm.evi.estimate_pickands_evi], [`estimate_dedh_moment_evi`][unibm.evi.estimate_dedh_moment_evi], [`estimate_max_spectrum_evi`][unibm.evi.estimate_max_spectrum_evi] |
| EI pooled across a stable block-size window | [`prepare_ei_bundle`][unibm.ei.prepare_ei_bundle], then [`estimate_pooled_bm_ei`][unibm.ei.estimate_pooled_bm_ei]; FGLS also requires [`bootstrap_bm_ei_path`][unibm.ei.bootstrap_bm_ei_path] |
| Native Northrop or BB EI at a selected block size | [`estimate_native_bm_ei`][unibm.ei.estimate_native_bm_ei] |
| EI from threshold exceedances | [`estimate_ferro_segers`][unibm.ei.estimate_ferro_segers], [`estimate_k_gaps`][unibm.ei.estimate_k_gaps] |
| Design-life levels and their intervals | [`estimate_design_life_level`][unibm.evi.estimate_design_life_level], [`estimate_design_life_level_interval`][unibm.evi.estimate_design_life_level_interval] |
| Quantile at a specified block size | [`predict_block_quantile`][unibm.evi.predict_block_quantile] |
| Empirical ranks | [`empirical_cdf`][unibm.cdf.empirical_cdf] |

The [EVI namespace](evi.md) and [EI namespace](ei.md) contain the full signatures,
result types, bootstrap, selection, and plotting helpers. Estimator-specific
assumptions and CI methods are documented there; bootstrap is not a common
default for all estimators. Mean/mode scaling fits require their own scaling
assumptions and cannot be passed to the quantile design-life helpers.

## Defaults and explicit choices

### EVI

`estimate_evi_quantile` defaults to `quantile=0.5` (median) and `sliding=True`.
**`regression` has no default:** choose `"OLS"`, `"FGLS"`, or `"AUTO"` explicitly.

- With `"FGLS"` or `"AUTO"`, omitting `bootstrap_reps` uses adaptive bootstrap
  when no `bootstrap_result` is supplied. An integer requests a fixed budget.
- `"OLS"` does not bootstrap. `"FGLS"` requires usable covariance; `"AUTO"`
  permits missing internally generated covariance to fall back to OLS.
- Covariance shrinkage defaults to the fixed value `0.37`; `random_state`
  defaults to `0`.

Thus, `estimate_evi_quantile(sample, regression="FGLS")` selects
**median + sliding + adaptive FGLS**. The lower-level bootstrap backbone and
multi-target helpers instead take fixed integer repetition budgets. See the
[EVI example](../worked-examples.md#example-1-median-sliding-block-evi-fit) and
[covariance reuse example](../worked-examples.md#example-2-bootstrap-covariance-backbone).

### EI

There is **no single default EI estimator**. Native BM, pooled BM, Ferro--Segers,
and K-gaps are separate entrypoints. The pooled workflow is:

1. Call `prepare_ei_bundle` with an explicit `allow_zeros` choice appropriate to
   the observation clock. It prepares Northrop and BB paths with both block
   schemes. Its threshold candidates default to `(0.90, 0.95)`.
2. For FGLS, call `bootstrap_bm_ei_path` with explicit `base_path`, `sliding`,
   `block_sizes`, and `allow_zeros`. Its `reps` defaults to `"adaptive"`.
3. Call `estimate_pooled_bm_ei` with explicit `base_path="northrop"` or `"bb"`,
   `sliding=True` or `False`, and `regression="OLS"` or `"FGLS"`. FGLS requires
   the matching bootstrap result; OLS rejects one.

Both the bootstrap and pooled fit default to covariance shrinkage `0.37`.
The bootstrap seed defaults to `0`. Native BM and threshold estimators do not
automatically run this covariance bootstrap. See the complete
[pooled EI example](../worked-examples.md#example-3-pooled-extremal-index-fit).

The repository application workflow selects quantile-sliding adaptive FGLS for
EVI and both Northrop-sliding and BB-sliding adaptive FGLS for pooled EI. These
are application configurations, not additional defaults imposed by the package.

### Design-life levels

`estimate_design_life_level` takes an existing **quantile-based fit** and an
explicit horizon `years`. It does not refit the data, bootstrap, or select an EI
estimator. It maps the fitted scaling law to the horizon's observation count:

- `observations_per_year=365.25` is a daily-clock default. Set it to match the
  fitted series' actual observation clock.
- `tau=None` inherits `fit.quantile`; an explicit `tau` must match it. A median
  fit yields the median of the horizon maximum, not a return level whose return
  period equals `years`.
- `estimate_design_life_level_interval` uses the fit's coefficient covariance
  for a log-scale delta-method interval, nominally 95% with `z_crit=1.96`.
  This is a conditional interval for the fitted quantile, not a prediction
  interval for a future maximum.

Adaptive EVI and EI bootstrap check precision at 128, 256, 512, 768, and 1024
draws. Reaching the cap can leave `bootstrap_precision_met=False`. EVI's
precision diagnostic monitors the EVI estimate and its CI endpoints, **not the
extrapolated design-life levels**. See
[Reading Returned Objects](../reading-returned-objects.md#reading-adaptive-precision).

## Top-level convenience imports

The current source supports:

```python
from unibm import (
    estimate_evi_quantile,
    prepare_ei_bundle,
    bootstrap_bm_ei_path,
    estimate_pooled_bm_ei,
    estimate_design_life_level,
    estimate_design_life_level_interval,
)
```

The EVI and pooled EI estimators return fit objects containing both the point
estimate and its CI. Design-life point estimates and intervals remain separate
calls. All six imports load lazily and resolve to the original subpackage
functions. Comparator estimators and other helpers remain in the grouped
namespaces above.

**Version note:** the four additional root imports are unreleased. PyPI 0.1.0
exports only `estimate_evi_quantile` and `estimate_design_life_level` at the
root. On that release, import the other four functions as follows; these paths
also remain supported in the current source:

```python
from unibm.ei import prepare_ei_bundle, bootstrap_bm_ei_path, estimate_pooled_bm_ei
from unibm.evi import estimate_design_life_level_interval
```

::: unibm
    options:
      members:
        - estimate_evi_quantile
        - prepare_ei_bundle
        - bootstrap_bm_ei_path
        - estimate_pooled_bm_ei
        - estimate_design_life_level
        - estimate_design_life_level_interval
      show_root_heading: true
      show_source: false
      members_order: source
      filters:
        - "!^_"
