# UniBM

UniBM packages reusable statistical code for block-maxima inference under
serial dependence, with benchmark and application workflows for both the
extreme value index (`xi`) and the extremal index (`theta`).

## Quick Start

From the repository root, the standard end-to-end entrypoint is:

```bash
uv sync --dev
just full
uvx ruff format ./**/*.py ./**/*.ipynb
```

This runs:

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
`uv run python ...`.

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

The `justfile` is organized around three levels of use:

```bash
just rebuild
just qa
just docs
```

And one full entrypoint:

```bash
just full
```

Current defaults:

- `workers="6"` for benchmark and application multiprocessing;
- `screening_bootstrap="20"` for USGS screening.

Those defaults are intentionally conservative. They are fast enough for routine
use without being overly aggressive on memory or CPU contention.

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

If you prefer the raw commands instead of `just`, the canonical rebuild order
behind that workflow is:

```bash
uv sync --dev

UNIBM_BENCHMARK_WORKERS=6 uv run python scripts/workflows/evi_benchmark.py
UNIBM_BENCHMARK_WORKERS=6 uv run python scripts/workflows/ei_benchmark.py

uv run python scripts/workflows/evi_report.py
uv run python scripts/workflows/ei_report.py

UNIBM_SCREENING_BOOTSTRAP_REPS=20 uv run python scripts/workflows/freeze_usgs_station_selection.py
UNIBM_APPLICATION_WORKERS=6 uv run python scripts/workflows/application.py

uv run python scripts/rebuild_vignette.py
uv run python -m unittest discover -s tests -p 'test_*.py'
uv run coverage run -m unittest discover -s tests -p 'test_*.py'
uv run coverage report -m
uvx ruff check .
uv run sphinx-build -b html docs docs/_build/html
uvx ruff format ./**/*.py ./**/*.ipynb
```

Notes:

- Prefer `uv sync --dev` over `uv sync -U` for reproducible rebuilds. `-U`
  upgrades dependencies and is better treated as an explicit maintenance step.
- `just full` expands to `rebuild + qa + docs`; the final `uvx ruff format`
  remains a separate explicit formatting pass in your standard workflow.
- `uv run python scripts/workflows/profile_sliding_windows.py` is optional
  profiling/diagnostics and not required for benchmark, application, vignette,
  or docs outputs.

## Workflow-Specific Commands

### Benchmarks

Rebuild the raw benchmark summaries:

```bash
UNIBM_BENCHMARK_WORKERS=6 uv run python scripts/workflows/evi_benchmark.py
UNIBM_BENCHMARK_WORKERS=6 uv run python scripts/workflows/ei_benchmark.py
```

`UNIBM_BENCHMARK_WORKERS` controls scenario-level multiprocessing.

Build manuscript-facing benchmark figures and tables:

```bash
uv run python scripts/workflows/evi_report.py
uv run python scripts/workflows/ei_report.py
```

### Applications

Run the application workflow:

```bash
UNIBM_SCREENING_BOOTSTRAP_REPS=20 uv run python scripts/workflows/freeze_usgs_station_selection.py
UNIBM_APPLICATION_WORKERS=6 uv run python scripts/workflows/application.py
```

The current `SERRA`-oriented application package materializes six series across
three environmental-risk layers:

- Houston wet-season precipitation;
- Phoenix hot-dry severity as a secondary compound-hazard case;
- Texas streamflow;
- Florida streamflow;
- Texas NFIP daily building payouts;
- Florida NFIP daily building payouts.

`freeze_usgs_station_selection.py` screens a curated Texas/Florida gauge pool
and freezes one flagship USGS site per state into
`data/metadata/application/usgs_frozen_sites.json`. The main application
workflow then downloads any missing raw inputs and writes manuscript-facing CSVs
and figures.

Provider-specific notes:

- `GHCN-Daily` is used for Houston and Phoenix.
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
- `application_usgs_site_screening.csv`

Manuscript-facing application tables are written to `UniBM_manuscript/Table/`:

- `application_summary_main.tex`
- `application_return_levels_main.tex`
- `application_ei_main.tex`

Manuscript-facing application figures are written to
`UniBM_manuscript/Figure/`, including:

- `application_ts_<stem>.pdf`
- `application_evi_<stem>.pdf`
- `application_ei_<stem>.pdf`
- `application_rl_<stem>.pdf`
- `application_overview.pdf`

Return-level plotting uses a mixed scale convention:

- Houston precipitation and Phoenix hot-dry severity keep a linear `y` axis.
- Texas/Florida streamflow and Texas/Florida NFIP return-level plots use a log
  `y` axis so the multi-order-of-magnitude spread remains readable.

### Vignette

Rebuild the notebook:

```bash
uv run python scripts/rebuild_vignette.py
```

The vignette presents the application section in the same style as the
benchmark sections:

- benchmark results are shown from cached benchmark summaries plus inline
  plotting helpers;
- application results are re-fit inside the notebook via
  `build_application_bundles(...)` and rendered inline with
  `plot_application_*` helpers rather than embedding external PDFs.

The application notebook also includes a dedicated **Data Provenance and Source
Records** section summarizing:

- the NOAA GHCN-Daily station ids and source URLs used for Houston and Phoenix;
- the frozen USGS gauge ids used for Texas and Florida streamflow;
- the OpenFEMA NFIP endpoint and state-level claim filters used for Texas and
  Florida building-payout series.

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
uv run python scripts/workflows/profile_sliding_windows.py
```

This writes `out/sliding_window_profile.csv` and prints runtime plus peak-memory
comparisons for:

- `unibm.core.block_maxima`
- `unibm.bootstrap._sliding_block_maxima`
- `unibm.extremal_index._rolling_window_minima`

## Repository Layout

- `scripts/unibm/` contains the reusable statistical core.
- `scripts/workflows/` contains benchmark, manuscript, and application
  pipelines.
- `scripts/data_prep/` contains application-specific preprocessing helpers.
- `data/metadata/application/` contains frozen USGS site selections and the CPI
  deflator table used by the NFIP workflow.
- `docs/` contains Sphinx docs for the reusable statistical layer.
- `scripts/rebuild_vignette.py` regenerates `scripts/vignette.ipynb`.
