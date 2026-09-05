# Worked Examples

## Example 1: Median sliding-block EVI fit

```python
import numpy as np
from unibm import estimate_evi_quantile, estimate_design_life_level
from unibm.evi import estimate_design_life_level_interval

sample = np.random.default_rng(7).pareto(2.0, 4096) + 1.0
fit = estimate_evi_quantile(
    sample,
    regression="FGLS",
    quantile=0.5,
    sliding=True,
    bootstrap_reps="adaptive",
    random_state=7,
)
design_life = estimate_design_life_level(
    fit,
    years=np.array([10.0, 50.0]),
    observations_per_year=365.25,
)
design_life_interval = estimate_design_life_level_interval(
    fit,
    years=np.array([10.0, 50.0]),
    observations_per_year=365.25,
)
```

Read `fit.slope` as the headline `xi` estimate,
`fit.confidence_interval` for uncertainty, and `fit.plateau_bounds` for the
selected regression window. `design_life` contains the resulting
design-life-level estimates on the original data scale, while
`design_life_interval` returns the matching conditional interval bounds from
the fitted coefficient covariance. The observation rate is an explicit daily-clock
assumption for this synthetic example. Adaptive stopping monitors `xi` and its
CI endpoints, **not** these extrapolated 10- and 50-year levels.

## Example 2: Bootstrap covariance backbone

```python
import numpy as np
from unibm.evi import generate_block_sizes
from unibm.evi.bootstrap import (
    build_block_summary_bootstrap_backbone,
    evaluate_block_summary_bootstrap_backbone,
)
from unibm.evi.estimation import estimate_target_scaling

sample = np.random.default_rng(13).pareto(2.0, 4096) + 1.0
block_sizes = generate_block_sizes(sample.size)
backbone = build_block_summary_bootstrap_backbone(
    sample, block_sizes, sliding=True, reps=480, random_state=13
)
quantile_boot = evaluate_block_summary_bootstrap_backbone(
    backbone, target="quantile", quantile=0.5
)
reused_fit = estimate_target_scaling(
    sample,
    regression="FGLS",
    target="quantile",
    quantile=0.5,
    sliding=True,
    block_sizes=block_sizes,
    bootstrap_result=quantile_boot,
)
```

The most useful outputs here are `quantile_boot["block_sizes"]`,
`quantile_boot["samples"]`, and `quantile_boot["covariance"]`. They feed
covariance-aware EVI fits without redoing the resampling step. This low-level
backbone takes a fixed integer budget: it has no fitted-window precision target
on which to base adaptive stopping. Reuse requires matching target, quantile,
and block scheme; covariance is aligned by block-size labels. It does not verify
that two arrays came from the same dataset, so callers must preserve that link.
These objects stay in memory; the package does not automatically write NPZ or
Parquet files.

## Example 3: Pooled extremal-index fit

```python
import numpy as np
from unibm.ei.preparation import prepare_ei_bundle
from unibm.ei.bm import estimate_pooled_bm_ei
from unibm.ei.bootstrap import bootstrap_bm_ei_path

sample = np.random.default_rng(21).pareto(2.0, 4096) + 1.0
bundle = prepare_ei_bundle(sample, allow_zeros=False)
bootstrap = bootstrap_bm_ei_path(
    bundle.values,
    allow_zeros=False,
    base_path="bb",
    sliding=True,
    block_sizes=bundle.block_sizes,
    reps="adaptive",  # Or a fixed integer, for example 480.
    random_state=21,
)
fit = estimate_pooled_bm_ei(
    bundle,
    base_path="bb",
    sliding=True,
    regression="FGLS",
    bootstrap_result=bootstrap,
)
```

Read `fit.theta_hat` as the headline EI estimate and
`fit.confidence_interval` for its uncertainty. `fit.stable_window` shows which
block-size region was pooled. The observed path is what gets pooled; the
bootstrap supplies covariance weights and uncertainty, not a replacement path.
Both calls use fixed shrinkage `0.37` by default. If you override it, pass the
same value to both calls to retain a matching adaptive precision diagnostic.
Strict FGLS errors on unusable or mismatched covariance; it never silently
switches to OLS.

Inspect `fit.bootstrap_reps_used`, `fit.bootstrap_precision_met`, and
`dict(zip(fit.bootstrap_mcse_targets, fit.bootstrap_mcse))` before interpreting
numerical precision. A cap warning retains the fit but marks the diagnostic
unmet. It is not a reason to silently drop the fit from a comparison.
