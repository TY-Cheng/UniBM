# UniBM

UniBM packages reusable statistical code for block-maxima inference under serial dependence.

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
uv run python scripts/workflows/application.py
```

This materializes the application CSV outputs and manuscript-facing figures. If
the required GHCN raw files are missing locally, the workflow downloads them.

### Rebuild the vignette notebook

```bash
uv run python scripts/rebuild_vignette.py
```

## Repository Layout

- `scripts/unibm/` contains the reusable statistical core.
- `scripts/workflows/` contains benchmark, manuscript, and application pipelines.
- `scripts/data_prep/` contains application-specific preprocessing helpers.
- `scripts/rebuild_vignette.py` regenerates `scripts/vignette.ipynb`.
