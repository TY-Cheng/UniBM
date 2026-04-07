# UniBM

UniBM packages reusable statistical code for block-maxima inference under
serial dependence, with benchmark and application workflows for both the
extreme value index (`xi`) and the extremal index (`theta`).

## Quick Start

From the repository root, the standard end-to-end workflow is:

```bash
uv sync --dev
just full
uvx ruff format ./**/*.py ./**/*.ipynb
```

This covers:

- benchmark rebuilds;
- benchmark report generation;
- USGS site freezing plus application rebuild;
- vignette regeneration;
- unit tests, `scripts/unibm` coverage, and `ruff check`;
- Sphinx docs build for the reusable statistical core.

If you only want the scientific outputs and not the QA/docs pass, use:

```bash
uv sync --dev
just rebuild
```

Optional: install [`just`](https://github.com/casey/just) if you want short
task aliases. Without `just`, all commands below can still be run directly with
`uv run python ...` and `uv run jupytext ...`.

## Setup

1. Install dependencies with `uv`.
2. Copy the local environment template:

```bash
cp .env.example .env
```

3. Edit `.env` and set `DIR_WORK` to your local clone path.
4. Sync the environment:

```bash
uv sync --dev
```

## Recommended Commands

The main task entrypoints are:

```bash
just rebuild
just qa
just docs
```

`just full` expands to `rebuild + qa + docs`.

Current defaults:

- `workers="6"` for benchmark and application multiprocessing;
- `screening_bootstrap="20"` for USGS screening.

These defaults are intentionally conservative. They are fast enough for routine
use without being overly aggressive on CPU or memory.

Examples with explicit overrides:

```bash
just rebuild 8 40
just applications 6 40
just freeze-usgs 40
```

## Canonical Rebuild Order

Your standard full workflow is:

```bash
uv sync --dev
just full
uvx ruff format ./**/*.py ./**/*.ipynb
```

If you prefer the raw commands instead of `just`, the workflow behind `just full`
plus the final formatting pass is:

```bash
uv sync --dev

UNIBM_BENCHMARK_WORKERS=6 uv run python scripts/benchmark/evi_benchmark.py
UNIBM_BENCHMARK_WORKERS=6 uv run python scripts/benchmark/ei_benchmark.py

uv run python scripts/benchmark/evi_report.py
uv run python scripts/benchmark/ei_report.py

UNIBM_SCREENING_BOOTSTRAP_REPS=20 uv run python scripts/application/freeze_usgs.py
UNIBM_APPLICATION_WORKERS=6 uv run python scripts/application/build.py

uv run jupytext --sync notebooks/vignette.py
uv run python -m unittest discover -s tests -p 'test_*.py'
uv run coverage run -m unittest discover -s tests -p 'test_*.py'
uv run coverage report -m
uvx ruff check scripts tests notebooks
uv run sphinx-build -b html docs docs/_build/html
uvx ruff format ./**/*.py ./**/*.ipynb
```

Notes:

- Prefer `uv sync --dev` over `uv sync -U` for reproducible rebuilds. `-U`
  upgrades dependencies and is better treated as an explicit maintenance step.
- `just full` expands to `rebuild + qa + docs`; the final
  `uvx ruff format ./**/*.py ./**/*.ipynb` remains a separate explicit
  formatting pass in your standard workflow.
- `uv run python scripts/shared/profile_sliding_windows.py` is optional
  profiling/diagnostics and not required for benchmark, application, vignette,
  or docs outputs.

## Workflow-Specific Commands

### Benchmarks

Rebuild the raw benchmark summaries:

```bash
UNIBM_BENCHMARK_WORKERS=6 uv run python scripts/benchmark/evi_benchmark.py
UNIBM_BENCHMARK_WORKERS=6 uv run python scripts/benchmark/ei_benchmark.py
```

`UNIBM_BENCHMARK_WORKERS` controls scenario-level multiprocessing.

Build manuscript-facing benchmark figures and tables:

```bash
uv run python scripts/benchmark/evi_report.py
uv run python scripts/benchmark/ei_report.py
```

### Applications

Run the application workflow:

```bash
UNIBM_SCREENING_BOOTSTRAP_REPS=20 uv run python scripts/application/freeze_usgs.py
UNIBM_APPLICATION_WORKERS=6 uv run python scripts/application/build.py
```

The current `SERRA`-oriented application package uses six application-facing
series across three environmental-risk layers:

- Houston wet-season precipitation as a secondary EVI-only weather case;
- Phoenix hot-dry severity as a secondary EVI-only compound-hazard case;
- Texas streamflow;
- Florida streamflow;
- Texas NFIP daily building payouts;
- Florida NFIP daily building payouts.

`scripts/application/freeze_usgs.py` screens a curated Texas/Florida gauge pool
and freezes one flagship USGS site per state into
`data/metadata/application/usgs_frozen_sites.json`. The main application
workflow then downloads any missing raw inputs and writes manuscript-facing CSVs
and figures.

Provider-specific notes:

- `GHCN-Daily` is used for Houston and Phoenix weather-side EVI cases.
- `USGS daily discharge` is used for Texas and Florida streamflow.
- `OpenFEMA NFIP claims` is used for Texas and Florida impact series.
- NFIP uses `positive-payout-day` totals for EVI and `zero-filled daily` totals
  for EI so claim-wave timing is preserved.

Cached application downloads are reused by default. The workflow only refreshes
files when they are missing, obviously broken, or explicitly force-refreshed.

- USGS raw extracts are automatically refreshed when the cached file is too
  short or unreadable.
- GHCN station files and NFIP state extracts are reused unless their cached
  file fails a basic integrity check.
- Set `UNIBM_FORCE_REFRESH_APPLICATION_DATA=1` to force fresh downloads across
  application inputs.

Main application outputs are written to `out/applications/`:

- `application_series_registry.csv`
- `application_screening.csv`
- `application_summary.csv`
- `application_return_levels.csv`
- `application_methods.csv`
- `application_ei_methods.csv`
- `application_ei_seasonal_methods.csv`
- `application_usgs_site_screening.csv`

Application method defaults are now intentionally asymmetric:

- `application_methods.csv` records only the headline EVI fit
  `sliding_median_fgls`;
- `application_ei_methods.csv` records the four-method EI comparison set
  `bb_sliding_fgls`, `northrop_sliding_fgls`, `k_gaps`, and `ferro_segers`
  only for the formal EI applications (`tx_streamflow`, `fl_streamflow`,
  `tx_nfip_claims`, and `fl_nfip_claims`);
- `application_ei_seasonal_methods.csv` stores the appendix-only monthly
  empirical-PIT to unit-Frechet seasonal sensitivity for those same four EI
  methods and the same formal EI applications.

Manuscript-facing application tables are written to `UniBM_manuscript/Table/`:

- `application_summary_main.tex`
- `application_return_levels_main.tex`
- `application_ei_main.tex`

Manuscript-facing application figures are written to
`UniBM_manuscript/Figure/`, including:

- `application_ts_<stem>.pdf`
- `application_evi_<stem>.pdf`
- `application_target_<stem>.pdf`
- `application_ei_<stem>.pdf` for the formal EI applications only
- `application_rl_<stem>.pdf`
- `application_composite_<stem>.pdf`
- `application_overview.pdf`

The composite figure is now the default notebook-facing visual. For streamflow
and NFIP it combines target stability, the headline median-sliding-FGLS scaling
fit, the four-method EI comparison, and the return-level panel in one 2x2
layout. Houston and Phoenix use an EVI-only composite variant where the raw
daily series replaces the EI panel. The older single-purpose PDFs remain
available as secondary/debug outputs.

Return-level plotting uses a mixed scale convention:

- Houston precipitation and Phoenix hot-dry severity keep a linear `y` axis.
- Texas/Florida streamflow and Texas/Florida NFIP return-level plots use a log
  `y` axis so the multi-order-of-magnitude spread remains readable.

### Vignette

Sync the notebook artifact from the Jupytext source of truth:

```bash
just vignette
# or
uv run jupytext --sync notebooks/vignette.py
```

The old generator entrypoint
`uv run python scripts/rebuild_vignette.py`
no longer exists after the Jupytext migration.

The source of truth now lives at `notebooks/vignette.py` in Jupytext
`py:percent` format, and the committed paired notebook lives at
`notebooks/vignette.ipynb`.

The vignette presents the application section in the same style as the
benchmark sections:

- benchmark results are shown from cached benchmark summaries plus inline
  plotting helpers;
- application results are re-fit inside the notebook via
  `build_application_bundles(...)` and rendered inline with
  `plot_application_*` helpers rather than embedding external PDFs.

Application sections now use `plot_application_composite(...)` as the main
visual. The notebook still shows the raw time series separately, but the
headline formal-EI application comparison is carried by the composite figure
plus the CSV/LaTeX tables for streamflow and NFIP. Houston and Phoenix appear
later in the notebook only as secondary EVI-only weather plots.

The application notebook also includes a dedicated **Data Provenance and Source
Records** section summarizing:

- the NOAA GHCN-Daily station ids and source URLs used for Houston and Phoenix;
- the frozen USGS gauge ids used for Texas and Florida streamflow;
- the OpenFEMA NFIP endpoint and state-level claim filters used for Texas and
  Florida building-payout series.

It also reports an appendix-only **seasonal-adjusted EI sensitivity** based on
a monthly empirical PIT -> unit-Frechet transform of each prepared EI series
for the formal EI applications only. Those rows are a robustness check and are
not used in the main return-level adjustment or in the headline application
summary tables.

`out/benchmark/preview/` is no longer part of the formal workflow. It was used
only for temporary benchmark figure previews while tuning display limits and has
been removed.

## API Documentation

The reusable statistical library under `scripts/unibm/` has a lightweight
Sphinx documentation site under `docs/`.

Build the HTML docs with:

```bash
uv run sphinx-build -b html docs docs/_build/html
```

Or:

```bash
just docs
```

The generated site will be written to `docs/_build/html/index.html`.

The docs include:

- API reference pages for the `scripts/unibm` public modules;
- an `EVI and EI Workflow Guide` overview page;
- a `Worked Examples` page with small runnable examples for EVI, diagnostic
  reciprocal-EI, and formal EI estimation.

Sphinx is used here because it remains the standard choice for scientific
Python libraries and works well with the existing type hints and docstrings
without forcing a larger repo restructure.

## Testing and Coverage

Run the unit test suite:

```bash
uv run python -m unittest discover -s tests -p 'test_*.py'
```

Run the same suite with branch coverage:

```bash
uv run coverage run -m unittest discover -s tests -p 'test_*.py'
uv run coverage report -m
```

Generate the HTML coverage report:

```bash
uv run coverage html
```

Coverage is intentionally scoped to the reusable statistical core under
`scripts/unibm/`, not to raw-data downloads, workflow glue, or manuscript
artifacts. The current coverage gate is `90%` for `scripts/unibm/`.

## Profiling Sliding Windows

The guarded sliding-window fast path is benchmarked separately before any
future replacement of the stride-based baseline:

```bash
uv run python scripts/shared/profile_sliding_windows.py
```

This writes `out/sliding_window_profile.csv` and prints runtime plus peak-memory
comparisons for:

- `unibm.core.block_maxima`
- `unibm.bootstrap._sliding_block_maxima`
- `unibm.extremal_index._rolling_window_minima`

## Repository Layout

- `scripts/unibm/` contains the reusable statistical core.
- `scripts/benchmark/` contains synthetic benchmark compute/report pipelines.
- `scripts/application/` contains real-data application build, screening,
  metadata, and export code.
- `scripts/shared/` contains shared CLI bootstrap, runtime, and profiling
  helpers.
- `scripts/vignette/` contains the notebook-facing helper API used by the
  Jupytext vignette.
- `scripts/data_prep/` contains application-specific preprocessing helpers.
- `data/metadata/application/` contains frozen USGS site selections and the CPI
  deflator table used by the NFIP workflow.
- `docs/` contains Sphinx docs for the reusable statistical layer.
- `notebooks/vignette.py` is the Jupytext source of truth for the research
  notebook, and `uv run jupytext --sync notebooks/vignette.py` regenerates the
  paired `notebooks/vignette.ipynb`.
