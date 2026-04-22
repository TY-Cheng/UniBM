# Worked Examples

## Example 1: Median sliding-block EVI fit

```python
import numpy as np
from unibm import estimate_evi_quantile, estimate_design_life_level
from unibm.evi import estimate_design_life_level_interval

sample = np.random.default_rng(7).pareto(2.0, 4096) + 1.0
fit = estimate_evi_quantile(sample, quantile=0.5, sliding=True, bootstrap_reps=120)
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
the fitted coefficient covariance.

## Example 2: Bootstrap covariance backbone

```python
import numpy as np
from unibm.evi import generate_block_sizes
from unibm.evi.bootstrap import (
    build_block_summary_bootstrap_backbone,
    evaluate_block_summary_bootstrap_backbone,
)

sample = np.random.default_rng(13).pareto(2.0, 4096) + 1.0
block_sizes = generate_block_sizes(sample.size)
backbone = build_block_summary_bootstrap_backbone(sample, block_sizes, sliding=True, reps=64)
quantile_boot = evaluate_block_summary_bootstrap_backbone(backbone, target="quantile")
```

The most useful outputs here are `quantile_boot["block_sizes"]`,
`quantile_boot["samples"]`, and `quantile_boot["covariance"]`. They feed
covariance-aware EVI fits without redoing the resampling step.

## Example 3: Pooled extremal-index fit

```python
import numpy as np
from unibm.ei.preparation import prepare_ei_bundle
from unibm.ei.bm import estimate_pooled_bm_ei

sample = np.random.default_rng(21).pareto(2.0, 4096) + 1.0
bundle = prepare_ei_bundle(sample)
fit = estimate_pooled_bm_ei(bundle, base_path="bb", sliding=True, regression="OLS")
```

Read `fit.theta_hat` as the headline EI estimate and
`fit.confidence_interval` for its uncertainty. `fit.stable_window` shows which
block-size region was pooled. If you later supply a bootstrap covariance
result and switch to `regression="FGLS"`, the observed path is still what gets
pooled; the bootstrap only supplies the covariance weights.
