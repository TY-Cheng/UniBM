set shell := ["zsh", "-cu"]
set dotenv-load
export UV_LOCKED := "1"

default:
    @just --list

# Environment Guard
[private]
_require-manuscript-dir:
    @manuscript_path="${DIR_MANUSCRIPT:-../UniBM_manuscript}"; \
    manuscript_abs="${manuscript_path:A}"; \
    if [[ ! -f "${manuscript_abs}/0_manuscript.tex" ]]; then \
        echo "DIR_MANUSCRIPT does not point to a manuscript repo with 0_manuscript.tex: ${manuscript_abs}"; \
        exit 1; \
    fi

[private]
_require-workflow-env: _require-manuscript-dir

# Main Entrypoints
full workers="8" screening_bootstrap="20": _require-workflow-env
    just check-full
    just clean-generated
    just benchmark "{{ workers }}"
    just application "{{ workers }}" "{{ screening_bootstrap }}"
    uv run python scripts/manuscript/artifact_manifest.py
    uv run mkdocs build --strict

benchmark workers="8": _require-workflow-env
    uv sync --dev
    UNIBM_BENCHMARK_WORKERS={{ workers }} uv run python scripts/benchmark/evi_benchmark.py
    UNIBM_BENCHMARK_WORKERS={{ workers }} uv run python scripts/benchmark/ei_benchmark.py
    UNIBM_BENCHMARK_WORKERS={{ workers }} uv run python scripts/benchmark/evi_report.py
    UNIBM_BENCHMARK_WORKERS={{ workers }} uv run python scripts/benchmark/ei_report.py

manuscript workers="8" screening_bootstrap="20": _require-workflow-env
    uv sync --dev
    UNIBM_BENCHMARK_WORKERS={{ workers }} uv run python scripts/benchmark/evi_report.py
    UNIBM_BENCHMARK_WORKERS={{ workers }} uv run python scripts/benchmark/ei_report.py
    UNIBM_SCREENING_BOOTSTRAP_REPS={{ screening_bootstrap }} uv run python scripts/application/freeze_usgs.py
    UNIBM_APPLICATION_WORKERS={{ workers }} uv run python scripts/application/build.py
    uv run python scripts/manuscript/artifact_manifest.py

data screening_bootstrap="20":
    uv sync --dev
    UNIBM_SCREENING_BOOTSTRAP_REPS={{ screening_bootstrap }} uv run python scripts/application/freeze_usgs.py
    PYTHONPATH=scripts uv run python -c 'from application.inputs import build_application_inputs; from config import resolve_repo_dirs; build_application_inputs(resolve_repo_dirs("."))'

refresh-data:
    uv sync --dev
    uv run python scripts/application/refresh_data.py

application workers="8" screening_bootstrap="20": _require-workflow-env
    uv sync --dev
    UNIBM_SCREENING_BOOTSTRAP_REPS={{ screening_bootstrap }} uv run python scripts/application/freeze_usgs.py
    UNIBM_APPLICATION_WORKERS={{ workers }} uv run python scripts/application/build.py

check:
    uv sync --dev
    just --fmt --check
    mkdir -p .cache
    TESTMON_DATAFILE=.cache/testmondata uv run pytest --testmon -n auto
    uv run pytest -q tests/test_justfile.py
    uv run ruff format --check .
    uv run ruff check .

check-full:
    uv sync --dev
    just --fmt --check
    uv run pytest -n auto --cov=src/unibm --cov-report=term-missing --cov-fail-under=89
    uv run ruff format --check .
    uv run ruff check .

format:
    just --fmt
    uv run ruff format .

# Utilities
clean-generated: _require-manuscript-dir
    mkdir -p out/benchmark/cache
    find out -mindepth 1 -maxdepth 1 ! -name benchmark -exec rm -rf {} +
    find out/benchmark -mindepth 1 -maxdepth 1 ! -name cache -exec rm -rf {} +
    rm -rf "${DIR_MANUSCRIPT:-../UniBM_manuscript}/Figure" "${DIR_MANUSCRIPT:-../UniBM_manuscript}/Table"
    rm -f "${DIR_MANUSCRIPT:-../UniBM_manuscript}/paper_subset_manifest.json"
