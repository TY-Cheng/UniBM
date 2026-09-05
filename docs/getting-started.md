# Getting Started

The installable package lives in `src/unibm` and is importable as
`unibm` once the repository environment has been synced. This site focuses on
the package layer, not on the full repo orchestration under
`scripts/benchmark` and `scripts/application`.

## Source installation

UniBM is **not yet released on PyPI**. Until the first reviewed release, install
from a local source checkout:

```bash
git clone https://github.com/TY-Cheng/UniBM.git
cd UniBM
uv sync --locked
```

For repository development, include the development dependencies and run the
lightweight checks:

```bash
uv sync --locked --dev
just check
```

No `.env` is required. Copy `.env.example` only to override the sibling
manuscript checkout or uv environment location. Top-level `just` tasks load it
automatically when present and sync the development environment before they run.
The repo-level workflow details stay in the repository `README.md` and
`justfile`. Use this site when you want the `unibm` package API itself.

## Package usage

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
    random_state=7,
)
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

FGLS uses adaptive repetitions by default. To request a fixed budget instead,
add `bootstrap_reps=480`. In this synthetic example, `365.25` is an illustrative
daily observation rate chosen by the caller, not inferred from the sample.

The shortest EI package workflow uses OLS and does not bootstrap:

```python
from unibm.ei.preparation import prepare_ei_bundle
from unibm.ei.bm import estimate_pooled_bm_ei

bundle = prepare_ei_bundle(sample, allow_zeros=False)
ei_fit = estimate_pooled_bm_ei(bundle, base_path="bb", sliding=True, regression="OLS")
```

Set `allow_zeros=True` only for a regularly spaced series whose observed zeros
must remain part of the calendar-day clock. With `False`, the input must already
be a strictly positive series on the caller's chosen clock. Missing or non-finite
observations are rejected in both modes rather than silently removed.
For the corresponding covariance-aware EI workflow, see the complete
[FGLS example](worked-examples.md#example-3-pooled-extremal-index-fit).

The scalar/vector outputs from `estimate_design_life_level` are point
estimates on the original response scale.
`estimate_design_life_level_interval` adds the matching conditional interval
summary from the fitted coefficient covariance.

EVI callers must choose `regression="OLS"`, `"FGLS"`, or `"AUTO"` explicitly.
Strict FGLS fails when usable bootstrap covariance is unavailable. `AUTO` may
fall back to OLS only when an internally generated bootstrap cannot supply
covariance; the returned fit records both the requested policy and the actual
regression.

For a quick guide to which returned fields matter most, see
[Reading Returned Objects](reading-returned-objects.md).

## Package boundaries

- `unibm.evi` owns the severity-side workflow, design-life-level helpers, and
  related plotting/bootstrap helpers.
- `unibm.ei` owns the persistence-side workflow, BM-path preparation, and
  threshold/BM EI estimators.
- `unibm.cdf` contains the public empirical CDF helper used by EI path
  preparation.
