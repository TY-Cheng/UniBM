# API Overview

Choose an entrypoint by the quantity you want to estimate. EVI and EI estimators
are public through `unibm.evi` and `unibm.ei`. UniBM also exposes
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
- Omitted `covariance_shrinkage=None` resolves to `0.73` for FGLS/AUTO. OLS
  rejects an explicit shrinkage value because it does not use covariance weights.
  `random_state` defaults to `0`.
- Supply only one grid source: `min_block_size/max_block_size/num_step`, an
  explicit `block_sizes`, or a reused `curve`. Reused curves reject additional
  grid arguments; explicit grids reject generation controls.
- Automatic EVI bounds are `max(5, ceil(n**(1/3)))` through
  `min(floor(n**(1 - 1/e)), floor(n/17))`. An infeasible range raises;
  a feasible range with fewer than five positive summaries cannot select a window.
- Explicit grid bounds are valid integer sizes with `max_block_size` strictly
  greater than `min_block_size`. They are not silently expanded. Explicit EVI
  `super_block_size` is honored unchanged when valid; see
  [bootstrap length rules](../concepts.md#bootstrap-lengths-repetitions-and-regularization).

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
   schemes by default. `path_keys=(("bb", True),)` prepares only BB-sliding;
   `path_keys=()` skips BM preparation for threshold-only use. Its threshold
   candidates default to `(0.90, 0.95)`. Its automatic BM grid uses
   `max(5, ceil(n**(1/3)))` through `min(floor(sqrt(n)), floor(n/17))`;
   EVI retains the upper exponent `1 - 1/e`. Bounds are never widened to
   satisfy an estimator's minimum window size.
2. For FGLS, call `bootstrap_bm_ei_path` with explicit `base_path`, `sliding`,
   `block_sizes`, and `allow_zeros`. Its `reps` defaults to `"adaptive"`.
3. Call `estimate_pooled_bm_ei` with explicit `base_path="northrop"` or `"bb"`,
   `sliding=True` or `False`, and `regression="OLS"` or `"FGLS"`. FGLS requires
   the matching bootstrap result; OLS rejects one and also rejects an explicit
   `covariance_shrinkage`.

The adaptive bootstrap monitors a pooled fit with shrinkage `0.37` by default
(`covariance_shrinkage=None`). An explicit value only configures that adaptive
monitor and is rejected with fixed integer `reps`. Both bootstrap modes return
the raw sample covariance; `estimate_pooled_bm_ei` applies shrinkage when fitting
FGLS, also defaulting to `0.37`.
The bootstrap seed defaults to `0`. Native BM and threshold estimators do not
automatically run this covariance bootstrap. See the complete
[pooled EI example](../worked-examples.md#example-3-pooled-extremal-index-fit).

For fixed-b native inference, prepare a single `block_sizes=[b]` and the required
`path_keys`. No stable-window selection is performed, and `stable_window` is
`None`. Multilevel native fits retain the smallest size in their selected window.
`use_adjusted_chandwich=True` is supported only for native Northrop; BB rejects it.

The repository application workflow selects quantile-sliding adaptive FGLS for
EVI and both Northrop-sliding and BB-sliding adaptive FGLS for pooled EI. These
are application configurations, not additional defaults imposed by the package.

### Design-life levels

`estimate_design_life_level` takes an existing **quantile-based fit** and an
explicit horizon `years`. It does not refit the data, bootstrap, or select an EI
estimator. It maps the fitted scaling law to the horizon's observation count:

- `observations_per_year=365.25` is a daily-clock default. Set it to match the
  fitted series' actual observation clock.
- `fit.quantile` determines the probability; there is no separate post-fit `tau`
  argument. A median fit yields the median of the horizon maximum, not a return
  level whose return period equals `years`.
- `estimate_design_life_level_interval` uses the fit's coefficient covariance
  for a log-scale delta-method interval, nominally 95% with `z_crit=1.96`.
  This is a conditional interval for the fitted quantile, not a prediction
  interval for a future maximum.

Adaptive EVI and EI bootstrap check precision at 128, 256, 512, 768, and 1024
draws. Reaching the cap can leave `bootstrap_precision_met=False`. EVI's
precision diagnostic monitors the EVI estimate and its CI endpoints, **not the
extrapolated design-life levels**. See
[Reading Returned Objects](../reading-returned-objects.md#reading-adaptive-precision).

## API migration

The 0.3.1 source checkout preserves the 0.3.0 public API and inference defaults.
Version 0.3.0 introduced the following API changes from 0.2.0.
Review the [current defaults](#defaults-and-explicit-choices) above as well:

| Earlier call | Current call or behavior |
|---|---|
| Design-life helpers with `tau=fit.quantile` | Omit `tau`; choose the quantile when fitting. |
| Plot helpers with `save=True, file_path=path` | Pass `file_path=path`; saving follows the path. |
| Plot helpers with `save=False` | Omit both `save` and `file_path` to avoid saving. |
| OLS with `covariance_shrinkage=...` | Omit shrinkage; it only applies to covariance-weighted fits. |
| Native BB with `use_adjusted_chandwich=True` | Omit this Northrop-only adjustment. |
| Conflicting bounds, duplicated grid sources, or infeasible explicit bootstrap lengths | Correct the arguments; these now raise `ValueError`. |
| Max-spectrum with `min_scale_count < 3`, a boolean, or a fractional value | Supply an integer of at least 3. |

## Top-level convenience imports

Since 0.2.0, EVI estimation and EVI/EI bootstrap calls accept `n_threads`.
Optional native kernels preserve the same public API. See
[thread budgets](../getting-started.md#bootstrap-threads) and
[native acceleration](../getting-started.md#optional-native-acceleration)
for defaults, nested parallelism, and the NumPy fallback. These additions are
not in PyPI 0.1.0.

UniBM 0.3.x supports:

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

**Version note:** the four additional root imports were added in 0.2.0. Version 0.1.0
exports only `estimate_evi_quantile` and `estimate_design_life_level` at the
root. On that release, import the other four functions as follows; these paths
also remain supported in current versions:

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
