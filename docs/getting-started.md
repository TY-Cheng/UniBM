# Getting Started

The reusable statistical package lives in `src/unibm` and is importable as
`unibm` once the repository environment has been synced. This site focuses on
the reusable package layer, not on the full repo orchestration under
`scripts/benchmark` and `scripts/application`.

## Environment setup

```bash
cp .env.example .env
just verify
```

Top-level `just` tasks load `.env` automatically and sync the development
environment before they run. If you prefer raw `uv` commands instead, load
`.env` into your shell first and then sync:

```bash
set -a; source .env; set +a
uv sync --dev
```

The repo-level workflow details stay in the repository `README.md` and
`justfile`. Use this site when you want the reusable `unibm` package API
itself.

## Package usage

```python
import numpy as np
from unibm import estimate_evi_quantile, estimate_design_life_level
from unibm.evi import estimate_design_life_level_interval

sample = np.random.default_rng(7).pareto(2.0, 4096) + 1.0
fit = estimate_evi_quantile(sample, quantile=0.5, sliding=True, bootstrap_reps=120)
design_life = estimate_design_life_level(
    fit,
    years=np.array([10.0]),
    observations_per_year=365.25,
)
design_life_interval = estimate_design_life_level_interval(
    fit,
    years=np.array([10.0]),
    observations_per_year=365.25,
)
```

The shortest EI package workflow is:

```python
from unibm.ei.preparation import prepare_ei_bundle
from unibm.ei.bm import estimate_pooled_bm_ei

bundle = prepare_ei_bundle(sample)
ei_fit = estimate_pooled_bm_ei(bundle, base_path="bb", sliding=True, regression="OLS")
```

The scalar/vector outputs from `estimate_design_life_level` are point
estimates on the original response scale.
`estimate_design_life_level_interval` adds the matching conditional interval
summary from the fitted coefficient covariance.

For a quick guide to which returned fields matter most, see
[Reading Returned Objects](reading-returned-objects.md).

## Package boundaries

- `unibm.evi` owns the severity-side workflow, design-life mapping, and related
  plotting/bootstrap helpers.
- `unibm.ei` owns the persistence-side workflow, BM-path preparation, and
  threshold/BM EI estimators.
- `unibm.cdf` contains the public empirical CDF helper used by EI path
  preparation.
