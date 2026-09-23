# Getting Started

The installable package lives in `src/unibm` and is importable as
`unibm` after installation. This site focuses on
the package layer, not on the full repo orchestration under
`scripts/benchmark` and `scripts/application`.

## Installation

Install [UniBM 0.2.0 from PyPI](https://pypi.org/project/unibm/0.2.0/) with Python
3.11 or later:

```bash
python -m pip install unibm==0.2.0
```

## Source checkout and development

To work from a local source checkout:

```bash
git clone https://github.com/TY-Cheng/UniBM.git
cd UniBM
uv sync --locked
```

For repository development, include the development dependencies and run the
lightweight checks:

```bash
just check
```

No `.env` or external report project is required. Reports default to `out/reports/`.
Use `.env.example` to set `UNIBM_REPORT_DIR`, the uv environment location, or
the optional native-extension switch described below.
Top-level `just` tasks load it automatically when present and sync the development
environment before they run. For ad hoc commands with these overrides, use
`just --command uv run ...` or `just --command uv sync --locked --dev`.
Plain uv commands do not automatically load `.env` before selecting an environment.
The `just` recipes require zsh; on Windows without zsh, use `uv sync --locked --dev`
and direct uv commands with environment overrides exported in your shell. See
[Development setup](https://github.com/TY-Cheng/UniBM/blob/main/CONTRIBUTING.md#development-setup).
The repo-level workflow details stay in the repository `README.md` and
`justfile`. Use this site when you want the `unibm` package API itself.

## Package usage

These examples work with UniBM 0.1.0 and 0.2.0. Version 0.2.0
also exposes all six core workflow functions directly from `unibm`; see
[Top-level convenience imports](api/public-api.md#top-level-convenience-imports).

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
from unibm.ei import prepare_ei_bundle, estimate_pooled_bm_ei

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

## Bootstrap threads

Version 0.2.0 adds `n_threads` to EVI estimation and EVI/EI bootstrap
functions. This option is not part of the published 0.1.0 release.

- `None` (default) selects a small pool from the workload and available CPUs;
  fewer than 2,048 observations stay serial. Automatic selection uses at most
  eight threads. This is a size heuristic, not a runtime speed measurement.
- A positive integer sets an upper limit; `1` runs the bootstrap serially.
  Fewer threads may be used when there are fewer independent tasks or CPUs.
- This controls UniBM's bootstrap pool. NumPy/SciPy BLAS settings remain under
  the caller's control. For an outer process pool or concurrent fits, explicitly
  allocate the inner budget, usually `n_threads=1`.

```python
fit = estimate_evi_quantile(
    sample, regression="FGLS", quantile=0.95, random_state=7, n_threads=1,
)
```

For EI, pass the same option to `bootstrap_bm_ei_path`, then reuse that result
in `estimate_pooled_bm_ei`. Design-life point estimates and intervals reuse
the EVI fit and require no additional bootstrap.

The repository's benchmark, sensitivity, and application process pools assign
one internal bootstrap thread to each worker. Standalone calls retain automatic
selection. A fixed seed retains the same draws and adaptive stopping regardless
of thread count. Working arrays are processed in batches; retained inputs,
count tables, output samples, and concurrent workers still contribute to memory
use, so the batch budget is not a total process memory limit.

## Optional native acceleration

Version 0.2.0 accelerates EVI mode KDE and bootstrap quantile
rank searches with optional Cython kernels. The same APIs and `n_threads` setting
work with or without the extension. Mode uses repeated-maxima multiplicities;
tail estimators, FGLS regression and EI profile intervals also reuse computations
within NumPy/SciPy. Mean bootstrap retains its original reduction order.

Source installation attempts to build the extension and retains NumPy execution
if a C compiler is unavailable. Set `UNIBM_NO_EXTENSIONS=1` before building for a
pure Python distribution, or before starting Python to disable native execution.
The switch is read once at import; restart Python after changing it. It disables
UniBM's extension only, not NumPy/SciPy's own compiled code or BLAS threads.
This does not reduce bootstrap replicates or relax adaptive precision tolerances.
Native wheels are specific to their Python/platform tags; a pure Python wheel
provides the fallback wherever the runtime dependencies are supported. See the
[build and platform notes](https://github.com/TY-Cheng/UniBM/blob/main/CONTRIBUTING.md#native-acceleration).
These changes are not part of PyPI version 0.1.0.

## Plotting

The public plotting helpers return `(fig, ax)` and keep the figure open by default:

```python
import matplotlib.pyplot as plt
from unibm.evi import plot_scaling_fit

fig, ax = plot_scaling_fit(fit)
ax.set_title("My scaling fit")
fig.savefig("scaling.pdf")
plt.close(fig)
```

`plot_ei_path` and `plot_ei_fit` follow the same convention. Use `save=True` with
`file_path` for direct saving and `close=True` for batch jobs. The default figure
DPI is 150; repository report scripts retain their explicit report settings.
Plotting does not infer a repository or external report destination.

## Package boundaries

- `unibm.evi` owns the severity-side workflow, design-life-level helpers, and
  related plotting/bootstrap helpers.
- `unibm.ei` owns the persistence-side workflow, BM-path preparation, and
  threshold/BM EI estimators.
- `unibm.cdf` contains the public empirical CDF helper used by EI path
  preparation.

UniBM is developed and maintained by Tuoyuan Cheng under the project supervision
of Kan Chen. Repository experiments and applications remain available on
[GitHub](https://github.com/TY-Cheng/UniBM/); they are not installed with the package.
