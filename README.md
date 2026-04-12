# UniBM

UniBM packages reusable statistical code for block-maxima inference under
serial dependence, with benchmark and application workflows for both the
extreme value index (`xi`) and the extremal index (`theta`).

The main methodological selling point is not the term *design-life level* by
itself, but the paired UniBM workflow that starts from dependent block-maxima
quantile scaling and then reports:

- one **severity branch** for `xi` estimation on short records;
- one **persistence branch** for `theta` estimation for clustering and episode
  persistence;
- one decision-facing **design-life** output derived from the severity branch,
  i.e. `T`-year block-maximum `tau`-quantiles on the original physical scale.

## Quick Start

From the repository root, the standard end-to-end workflow is:

```bash
just full
```

This is the standard **code-repo full rebuild**. It covers:

- environment sync into the external uv environment declared in `.env`;
- benchmark rebuilds;
- benchmark report generation;
- USGS site freezing plus application rebuild;
- vignette regeneration plus in-place notebook execution;
- manuscript-facing figure/table refresh plus `paper_subset_manifest.json`;
- unit tests, `src/unibm` coverage, `ruff check`, Sphinx docs, and formatting.

`just full` already refreshes the sibling manuscript repo's paper-facing
`Figure/`, `Table/`, and `paper_subset_manifest.json`. If you only want the
building blocks individually, run the specific top-level tasks below.

If you only want the main workflow blocks individually, use:

```bash
just verify
just benchmark
just application
just vignette
just format
```

Optional: install [`just`](https://github.com/casey/just) if you want short
task aliases. Without `just`, all commands below can still be run directly with
`uv run python ...` and `uv run python -m jupytext ...`.

## Docs

The package documentation source lives under `docs/`.

Build the local HTML site with:

```bash
uv run sphinx-build -b html docs docs/_build/html
```

Use `just docs` for the same build through the repo task runner. `just full`
also includes a docs rebuild before the cold workflow refresh.

After the build finishes, open:

```text
docs/_build/html/index.html
```

For the docs source on GitHub, browse
[`docs/`](https://github.com/TY-Cheng/UniBM/tree/main/docs).

## Setup

1. Install dependencies with `uv`.
2. Copy the local environment template:

```bash
cp .env.example .env
```

3. Edit `.env` and set:
   - `DIR_WORK` to your local code-repo clone path
   - `DIR_MANUSCRIPT` if the manuscript lives in a separate repo
   - `UV_PROJECT_ENVIRONMENT` to a dedicated external uv environment path, e.g.
     `/Users/yourname/.venvs/unibm`
4. Run one of the top-level `just` tasks. `just verify` is the lightest first
   run and will create or update the environment automatically:

```bash
just verify
# or, for the full rebuild:
just full
# or, if you prefer raw uv commands:
set -a; source .env; set +a
uv sync --dev
```

## Recommended Commands

The main task entrypoints are:

```bash
just full
just verify
just docs
just benchmark
just application
just vignette
just format
just clean-generated
```

`just full` expands to `verify + docs + clean-generated + benchmark +
application + vignette + paper_subset_manifest.json`. Each top-level `just`
task loads `.env` and runs `uv sync --dev` before its main work. The commands
you mainly need to remember are:

- `just full`: fail-fast verify, then cold rebuild of the main code-repo outputs plus manuscript manifest refresh
- `just verify`: `uv sync --dev` + tests + coverage + lint
- `just docs`: `uv sync --dev` + Sphinx HTML build into `docs/_build/html`
- `just benchmark`: benchmark rebuild + benchmark reports
- `just application`: USGS freeze + application rebuild
- `just vignette`: sync the paired Jupytext notebook, execute it in place, and format outputs
- `just format`: `uv sync --dev` + `ruff format` on tracked `.py` and `.ipynb`

Current defaults:

- `workers="6"` for benchmark and application multiprocessing;
- `screening_bootstrap="20"` for USGS screening.

These defaults are intentionally conservative. They are fast enough for routine
use without being overly aggressive on CPU or memory.

Examples with explicit overrides:

```bash
just full 8 40
just application 6 40
just benchmark 8
just format
```

## Canonical Rebuild Order

Your standard code-repo full workflow is:

```bash
just full
```

If you prefer the raw commands instead of `just`, the workflow behind
`just full` is:

```bash
set -a; source .env; set +a
uv sync --dev

uv run python -m unittest discover -s tests -p 'test_*.py'
uv run coverage run -m unittest discover -s tests -p 'test_*.py'
uv run coverage report -m
uv run coverage xml
uv run coverage html
uv run ruff check scripts tests notebooks
rm -rf docs/_build
uv run sphinx-build -b html docs docs/_build/html
uv run ruff format ./**/*.py ./**/*.ipynb

mkdir -p out/benchmark/cache
find out -mindepth 1 -maxdepth 1 ! -name benchmark -exec rm -rf {} +
find out/benchmark -mindepth 1 -maxdepth 1 ! -name cache -exec rm -rf {} +
rm -rf "${DIR_MANUSCRIPT:-../UniBM_manuscript}/Figure" "${DIR_MANUSCRIPT:-../UniBM_manuscript}/Table"
rm -f "${DIR_MANUSCRIPT:-../UniBM_manuscript}/paper_subset_manifest.json"

UNIBM_BENCHMARK_WORKERS=6 uv run python scripts/benchmark/evi_benchmark.py
UNIBM_BENCHMARK_WORKERS=6 uv run python scripts/benchmark/ei_benchmark.py

uv run python scripts/benchmark/evi_report.py
uv run python scripts/benchmark/ei_report.py

UNIBM_SCREENING_BOOTSTRAP_REPS=20 uv run python scripts/application/freeze_usgs.py
UNIBM_APPLICATION_WORKERS=6 uv run python scripts/application/build.py

uv run python -m jupytext --sync notebooks/vignette.py
uv run python -m nbconvert --to notebook --execute --inplace notebooks/vignette.ipynb
uv run ruff format ./**/*.py ./**/*.ipynb
uv run python scripts/manuscript/artifact_manifest.py
```

Notes:

- Prefer `uv sync --dev` over `uv sync -U` for reproducible rebuilds. `-U`
  upgrades dependencies and is better treated as an explicit maintenance step.
- Top-level `just` tasks load `.env` automatically and sync the development
  environment before running. If you run `uv ...` commands directly, load `.env`
  into your shell first so `DIR_WORK`, `DIR_MANUSCRIPT`, and
  `UV_PROJECT_ENVIRONMENT` are respected.
- `just vignette` syncs `notebooks/vignette.py` into `notebooks/vignette.ipynb`,
  executes the paired notebook in place, and then formats tracked `.py` and
  `.ipynb` files.
- `just verify` expands to `uv sync --dev + tests + coverage + ruff check`.
- `just docs` expands to `uv sync --dev + sphinx-build -b html docs docs/_build/html`.
- `just full` expands to `verify + docs + clean-generated + benchmark +
  application + vignette`.
- `just clean-generated` removes generated outputs under `out/` while preserving
  `out/benchmark/cache`, and also removes the manuscript repo's `Figure/` plus
  `Table/` directories. Use it when you want a cold rebuild of all rendered
  artifacts without deleting the benchmark cache.

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

The current manuscript-facing application package uses six application-facing
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
- `application_design_life_levels.csv`
  design-life-level curves over the application tau grid;
- `application_methods.csv`
- `application_ei_methods.csv`
- `application_ei_seasonal_methods.csv`
- `application_usgs_site_screening.csv`

Application method defaults are now intentionally asymmetric:

- `application_methods.csv` records only the headline EVI fit
  `sliding_median_fgls`, but expands it over the application tau grid
  `0.50 / 0.90 / 0.95 / 0.99` by reusing the same plateau and `xi` while
  estimating only tau-specific intercept shifts;
- `application_ei_methods.csv` records the four-method EI comparison set
  `bb_sliding_fgls`, `northrop_sliding_fgls`, `k_gaps`, and `ferro_segers`
  only for the formal EI applications (`tx_streamflow`, `fl_streamflow`,
  `tx_nfip_claims`, and `fl_nfip_claims`);
- `application_ei_seasonal_methods.csv` stores the appendix-only monthly
  empirical-PIT to unit-Frechet seasonal sensitivity for those same four EI
  methods and the same formal EI applications.

Interpreting the streamflow/NFIP application diagnostics:

- the `quantile scaling` panel is the fitted UniBM log-log block-summary curve;
- the `design-life level` panel is not a separate GEV fit, but the same fitted
  scaling law evaluated at larger block sizes and then mapped to longer
  design-life spans;
- the literature term closest to this output is a **design-life level**, i.e.
  a `T`-year block-maximum `tau`-quantile;
- the current manuscript/application default is `tau = 0.50`, so the headline
  curve is a median design-life level;
- the application plots and exports now also show `tau = 0.90 / 0.95 / 0.99`
  as increasingly conservative shared-`xi` companion curves;
- the EVI plateau and the EI stable window are selected from different
  statistical paths, so they do not need to match;
- different `tau` values are conceptually valid and should share the same
  asymptotic slope `xi` while differing mainly in intercept; in the
  application workflow those higher-`tau` curves are derived by holding the
  headline plateau and slope fixed and re-estimating only the intercept;
- in this direct block-maxima framework, serial dependence is already
  internalized in the fitted block-maximum law, so there is no second BM-side
  `theta` adjustment on the design-life-level curve;
- for NFIP, active-day design-life levels and calendar-day EI estimates are
  kept separate on purpose because they live on different clocks.

Manuscript-facing application tables are written to the manuscript repo's
`Table/` directory (typically `../UniBM_manuscript/Table/`):

- `application_summary_main.tex`
- `application_design_life_levels_main.tex`
- `application_ei_main.tex`
- `application_case_audit_main.tex`
- `application_selection_sensitivity_main.tex`

Manuscript-facing application figures are written to the manuscript repo's
`Figure/` directory (typically `../UniBM_manuscript/Figure/`), including:

- `application_ts_<stem>.pdf`
- `application_evi_<stem>.pdf`
- `application_target_<stem>.pdf`
- `application_ei_<stem>.pdf` for the formal EI applications only
- `application_design_life_<stem>.pdf`
- `application_composite_<stem>.pdf`
- `application_overview.pdf`

The submission-facing manuscript subset is intentionally narrower than the full
application workflow. The benchmark/application report steps plus
`scripts/manuscript/artifact_manifest.py` refresh the manuscript repo's
`paper_subset_manifest.json` (typically `../UniBM_manuscript/paper_subset_manifest.json`)
for that curated figure/table subset, and the main `just full` workflow now
does this automatically after the rebuild.

The composite figure is now the default notebook-facing visual. For streamflow
and NFIP it combines target stability, the headline median-sliding-FGLS scaling
fit, the four-method EI comparison, and the design-life-level panel in one 2x2
layout. The scaling and design-life-level panels now show the application tau grid
`0.50 / 0.90 / 0.95 / 0.99`, with `tau = 0.50` as the headline design-life
median and the higher curves as shared-`xi` upper companions. Houston and
Phoenix use an EVI-only composite variant where the raw daily series replaces
the EI panel. The older single-purpose PDFs remain
available as secondary/debug outputs.

Design-life-level plotting uses a mixed scale convention:

- Houston precipitation and Phoenix hot-dry severity keep a linear `y` axis.
- Texas/Florida streamflow and Texas/Florida NFIP design-life-level plots use a log
  `y` axis so the multi-order-of-magnitude spread remains readable.

### Vignette

Sync the notebook artifact from the Jupytext source of truth and execute the
paired notebook in place:

```bash
just vignette
# or
uv run python -m jupytext --sync notebooks/vignette.py
uv run python -m nbconvert --to notebook --execute --inplace notebooks/vignette.ipynb
```

`just vignette` finishes with a repo-wide `ruff format` pass so the paired
`.py` and `.ipynb` stay normalized after execution.

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
not used in the main median design-life-level summaries or in the
headline application summary tables.

`out/benchmark/preview/` is no longer part of the formal workflow. It was used
only for temporary benchmark figure previews while tuning display limits and has
been removed.

## API Documentation

For the quick local-docs pointer, see `Docs` near the top of this README. The
reusable statistical library under `src/unibm/` has a lightweight Sphinx
site under `docs/`, and the generated HTML entrypoint is
`docs/_build/html/index.html`.

The docs include:

- API reference pages for the slim root facade plus the canonical `unibm.evi`
  and `unibm.ei` packages;
- an `EVI and EI Workflow Guide` overview page;
- a `Worked Examples` page with small runnable examples for EVI, bootstrap
  backbones, and formal EI estimation;
- a `Reading Returned Objects` page showing which result fields to inspect
  first for EVI and formal EI fits.

## Testing and Coverage

For the combined repo-level verification pass, use:

```bash
just verify
```

Run the unit test suite:

```bash
uv run python -m unittest discover -s tests -p 'test_*.py'
```

Run the same suite with branch coverage:

```bash
uv run coverage run -m unittest discover -s tests -p 'test_*.py'
uv run coverage report -m
uv run coverage xml
uv run coverage html
```

Coverage is intentionally scoped to the reusable statistical core under
`src/unibm/`, not to raw-data downloads, workflow glue, or manuscript
artifacts. The current coverage gate is `90%` for `src/unibm/`.

## Repository Layout

- `src/unibm/` contains the installable reusable statistical core.
- `scripts/benchmark/` contains synthetic benchmark compute/report pipelines.
- `scripts/application/` contains real-data application build, screening,
  metadata, and export code.
- `scripts/shared/` contains shared CLI bootstrap and runtime helpers.
- `scripts/notebook_api.py` contains the notebook-facing helper API used by the
  Jupytext vignette.
- `scripts/data_prep/` contains application-specific preprocessing helpers.
- `data/metadata/application/` contains frozen USGS site selections and the CPI
  deflator table used by the NFIP workflow.
- `docs/` contains Sphinx docs for the reusable statistical layer.
- `notebooks/vignette.py` is the Jupytext source of truth for the research
  notebook, and `just vignette` syncs plus executes the paired
  `notebooks/vignette.ipynb` in place.
