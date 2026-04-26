set shell := ["zsh", "-cu"]
set dotenv-load

default:
    @just --list

# Environment Guard
[private]
_require-external-uv-env:
    @repo_dir="$(cd "{{justfile_directory()}}" && pwd -P)"; \
    env_path="${UV_PROJECT_ENVIRONMENT:-}"; \
    if [[ -z "${env_path}" ]]; then \
        echo "UV_PROJECT_ENVIRONMENT is missing. Set it in .env, for example: UV_PROJECT_ENVIRONMENT=/Users/yourname/.venvs/unibm"; \
        exit 1; \
    fi; \
    if [[ "${env_path}" != /* ]]; then \
        echo "UV_PROJECT_ENVIRONMENT must be an absolute path: ${env_path}"; \
        exit 1; \
    fi; \
    case "${env_path}" in \
        "${repo_dir}"|"${repo_dir}"/*) \
            echo "UV_PROJECT_ENVIRONMENT must point outside the repo: ${env_path}"; \
            exit 1; \
            ;; \
    esac

# Main Entrypoints
full workers="6" screening_bootstrap="20": _require-external-uv-env
    just verify
    just docs
    just clean-generated
    just benchmark "{{ workers }}"
    just application "{{ workers }}" "{{ screening_bootstrap }}"
    just vignette
    test -f "${DIR_MANUSCRIPT:-../UniBM_manuscript}/0_manuscript.tex" || { echo "DIR_MANUSCRIPT does not point to a manuscript repo with 0_manuscript.tex"; exit 1; }
    uv run python scripts/manuscript/artifact_manifest.py

benchmark workers="6": _require-external-uv-env
    just sync-env
    UNIBM_BENCHMARK_WORKERS={{ workers }} uv run python scripts/benchmark/evi_benchmark.py
    UNIBM_BENCHMARK_WORKERS={{ workers }} uv run python scripts/benchmark/ei_benchmark.py
    UNIBM_BENCHMARK_WORKERS={{ workers }} uv run python scripts/benchmark/evi_report.py
    UNIBM_BENCHMARK_WORKERS={{ workers }} uv run python scripts/benchmark/ei_report.py

manuscript workers="6" screening_bootstrap="20": _require-external-uv-env
    just sync-env
    test -f "${DIR_MANUSCRIPT:-../UniBM_manuscript}/0_manuscript.tex" || { echo "DIR_MANUSCRIPT does not point to a manuscript repo with 0_manuscript.tex"; exit 1; }
    UNIBM_BENCHMARK_WORKERS={{ workers }} uv run python scripts/benchmark/evi_report.py
    UNIBM_BENCHMARK_WORKERS={{ workers }} uv run python scripts/benchmark/ei_report.py
    UNIBM_SCREENING_BOOTSTRAP_REPS={{ screening_bootstrap }} uv run python scripts/application/freeze_usgs.py
    UNIBM_APPLICATION_WORKERS={{ workers }} uv run python scripts/application/build.py
    uv run python scripts/manuscript/artifact_manifest.py

application workers="6" screening_bootstrap="20": _require-external-uv-env
    just sync-env
    UNIBM_SCREENING_BOOTSTRAP_REPS={{ screening_bootstrap }} uv run python scripts/application/freeze_usgs.py
    UNIBM_APPLICATION_WORKERS={{ workers }} uv run python scripts/application/build.py

vignette: _require-external-uv-env
    just sync-env
    uv run python -m jupytext --sync notebooks/vignette.py
    uv run python -m nbconvert --to notebook --execute --inplace notebooks/vignette.ipynb
    uv run ruff format .

verify: _require-external-uv-env
    just sync-env
    rm -f .coverage
    uv run coverage run -m unittest discover -s tests -p 'test_*.py'
    uv run coverage report -m
    uv run coverage xml
    uv run coverage html
    uv run ruff format .

docs: _require-external-uv-env
    just sync-env
    rm -rf site
    uv run mkdocs build --strict

docs-serve: _require-external-uv-env
    just sync-env
    uv run mkdocs serve

# Setup
[private]
sync-env: _require-external-uv-env
    uv sync --dev

# Utilities
clean-generated: _require-external-uv-env
    just sync-env
    test -f "${DIR_MANUSCRIPT:-../UniBM_manuscript}/0_manuscript.tex" || { echo "DIR_MANUSCRIPT does not point to a manuscript repo with 0_manuscript.tex"; exit 1; }
    mkdir -p out/benchmark/cache
    find out -mindepth 1 -maxdepth 1 ! -name benchmark -exec rm -rf {} +
    find out/benchmark -mindepth 1 -maxdepth 1 ! -name cache -exec rm -rf {} +
    rm -rf "${DIR_MANUSCRIPT:-../UniBM_manuscript}/Figure" "${DIR_MANUSCRIPT:-../UniBM_manuscript}/Table"
    rm -f "${DIR_MANUSCRIPT:-../UniBM_manuscript}/paper_subset_manifest.json"
