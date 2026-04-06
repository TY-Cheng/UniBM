# UniBM

UniBM packages reusable statistical code for block-maxima inference under serial dependence, with benchmark and application workflows for both the extreme value index (`xi`) and the extremal index (`theta`).

## Getting Started

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

## Common Commands

Run all commands from the repository root.

### Rebuild raw benchmarks

```bash
UNIBM_BENCHMARK_WORKERS=6 uv run python scripts/workflows/evi_benchmark.py
UNIBM_BENCHMARK_WORKERS=6 uv run python scripts/workflows/ei_benchmark.py
```

`UNIBM_BENCHMARK_WORKERS` controls scenario-level multiprocessing. Omit it to use
the package default.

### Rebuild manuscript benchmark figures and tables

```bash
uv run python scripts/workflows/evi_report.py
uv run python scripts/workflows/ei_report.py
```

### Run the application workflow

```bash
uv run python scripts/workflows/freeze_usgs_station_selection.py
uv run python scripts/workflows/application.py
```

The application package is `SERRA`-oriented and currently materializes six
series across three environmental-risk layers:

- Houston wet-season precipitation
- Phoenix hot-dry severity (secondary compound-hazard case)
- Texas streamflow
- Florida streamflow
- Texas NFIP daily building payouts
- Florida NFIP daily building payouts

`freeze_usgs_station_selection.py` screens a curated Texas/Florida gauge pool and
freezes one flagship USGS site per state into
`data/metadata/application/usgs_frozen_sites.json`. The main application
workflow then downloads any missing raw inputs and writes manuscript-facing CSVs
and figures.

Provider-specific notes:

- `GHCN-Daily` is used for Houston and Phoenix.
- `USGS daily discharge` is used for Texas and Florida streamflow.
- `OpenFEMA NFIP claims` is used for Texas and Florida impact series.
- NFIP uses `positive-payout-day` totals for EVI and `zero-filled daily` totals
  for EI so claim-wave timing is preserved.

If stale one-day USGS extracts are present locally, the workflow automatically
refreshes them before fitting.

The main application outputs are written to `out/applications/`:

- `application_series_registry.csv`
- `application_screening.csv`
- `application_summary.csv`
- `application_return_levels.csv`
- `application_methods.csv`
- `application_ei_methods.csv`
- `application_usgs_site_screening.csv`

Manuscript-facing application figures are written to
`UniBM_manuscript/Figure/`, including:

- `application_ts_<stem>.pdf`
- `application_evi_<stem>.pdf`
- `application_ei_<stem>.pdf`
- `application_rl_<stem>.pdf`
- `application_overview.pdf`

### Rebuild the vignette notebook

```bash
uv run python scripts/rebuild_vignette.py
```

## Repository Layout

- `scripts/unibm/` contains the reusable statistical core.
- `scripts/workflows/` contains benchmark, manuscript, and application pipelines.
- `scripts/data_prep/` contains application-specific preprocessing helpers.
- `data/metadata/application/` contains frozen USGS site selections and the CPI
  deflator table used by the NFIP workflow.
- `scripts/rebuild_vignette.py` regenerates `scripts/vignette.ipynb`.
