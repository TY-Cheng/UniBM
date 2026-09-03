set shell := ["zsh", "-cu"]
set dotenv-load
export UV_LOCKED := "1"

default:
    @just --list

# Environment Guard
[private]
_require-external-uv-env:
    @repo_dir="$(cd "{{ justfile_directory() }}" && pwd -P)"; \
    env_path="${UV_PROJECT_ENVIRONMENT:-}"; \
    if [[ -z "${env_path}" ]]; then \
        echo "UV_PROJECT_ENVIRONMENT is missing. Set it in .env, for example: UV_PROJECT_ENVIRONMENT=/Users/yourname/.venvs/unibm"; \
        exit 1; \
    fi; \
    if [[ "${env_path}" != /* ]]; then \
        echo "UV_PROJECT_ENVIRONMENT must be an absolute path: ${env_path}"; \
        exit 1; \
    fi; \
    env_abs="${env_path:A}"; \
    case "${env_abs}" in \
        "${repo_dir}"|"${repo_dir}"/*) \
            echo "UV_PROJECT_ENVIRONMENT must point outside the repo: ${env_abs}"; \
            exit 1; \
            ;; \
    esac

[private]
_require-external-data-dir:
    @repo_dir="$(cd "{{ justfile_directory() }}" && pwd -P)"; \
    data_path="${DIR_DATA:-}"; \
    if [[ -z "${data_path}" ]]; then \
        echo "DIR_DATA is missing. Set it in .env, for example: DIR_DATA=/Volumes/ExternalSSD/data/unibm"; \
        exit 1; \
    fi; \
    if [[ "${data_path}" != /* ]]; then \
        echo "DIR_DATA must be an absolute path: ${data_path}"; \
        exit 1; \
    fi; \
    data_abs="${data_path:A}"; \
    case "${data_abs}" in \
        "${repo_dir}"|"${repo_dir}"/*) \
            echo "DIR_DATA must point outside the repo: ${data_abs}"; \
            exit 1; \
            ;; \
    esac; \
    if [[ -e "${repo_dir}/data" || -L "${repo_dir}/data" ]]; then \
        echo "Repo-local data path is not allowed in OneDrive: ${repo_dir}/data"; \
        echo "Remove the directory/symlink and use DIR_DATA=${data_abs}"; \
        exit 1; \
    fi

[private]
_require-manuscript-dir:
    @manuscript_path="${DIR_MANUSCRIPT:-../UniBM_manuscript}"; \
    manuscript_abs="${manuscript_path:A}"; \
    if [[ ! -f "${manuscript_abs}/0_manuscript.tex" ]]; then \
        echo "DIR_MANUSCRIPT does not point to a manuscript repo with 0_manuscript.tex: ${manuscript_abs}"; \
        exit 1; \
    fi

[private]
_require-data-env: _require-external-uv-env _require-external-data-dir

[private]
_require-workflow-env: _require-data-env _require-manuscript-dir

# Main Entrypoints
full workers="6" screening_bootstrap="20": _require-workflow-env
    just check-full
    just _docs-build
    just clean-generated
    just benchmark "{{ workers }}"
    just application "{{ workers }}" "{{ screening_bootstrap }}"
    just vignette
    uv run python scripts/manuscript/artifact_manifest.py

benchmark workers="6": _require-workflow-env
    uv sync --dev
    UNIBM_BENCHMARK_WORKERS={{ workers }} uv run python scripts/benchmark/evi_benchmark.py
    UNIBM_BENCHMARK_WORKERS={{ workers }} uv run python scripts/benchmark/ei_benchmark.py
    UNIBM_BENCHMARK_WORKERS={{ workers }} uv run python scripts/benchmark/evi_report.py
    UNIBM_BENCHMARK_WORKERS={{ workers }} uv run python scripts/benchmark/ei_report.py

manuscript workers="6" screening_bootstrap="20": _require-workflow-env
    uv sync --dev
    UNIBM_BENCHMARK_WORKERS={{ workers }} uv run python scripts/benchmark/evi_report.py
    UNIBM_BENCHMARK_WORKERS={{ workers }} uv run python scripts/benchmark/ei_report.py
    UNIBM_SCREENING_BOOTSTRAP_REPS={{ screening_bootstrap }} uv run python scripts/application/freeze_usgs.py
    UNIBM_APPLICATION_WORKERS={{ workers }} uv run python scripts/application/build.py
    uv run python scripts/manuscript/artifact_manifest.py

data screening_bootstrap="20": _require-data-env
    uv sync --dev
    UNIBM_SCREENING_BOOTSTRAP_REPS={{ screening_bootstrap }} uv run python scripts/application/freeze_usgs.py
    PYTHONPATH=scripts uv run python -c 'from application.inputs import build_application_inputs; from config import resolve_repo_dirs; build_application_inputs(resolve_repo_dirs("."))'

application workers="6" screening_bootstrap="20": _require-workflow-env
    uv sync --dev
    UNIBM_SCREENING_BOOTSTRAP_REPS={{ screening_bootstrap }} uv run python scripts/application/freeze_usgs.py
    UNIBM_APPLICATION_WORKERS={{ workers }} uv run python scripts/application/build.py

vignette: _require-workflow-env
    uv sync --dev
    uv run ruff format notebooks/vignette.py
    uv run python -m jupytext --sync notebooks/vignette.py
    uv run python -m nbconvert --to notebook --execute --inplace notebooks/vignette.ipynb

check: _require-external-uv-env
    uv sync --dev
    just --fmt --check
    mkdir -p .cache
    TESTMON_DATAFILE=.cache/testmondata uv run pytest --testmon -n auto
    uv run pytest -q tests/test_justfile.py
    uv run ruff format --check .
    uv run ruff check .

check-full: _require-external-uv-env
    uv sync --dev
    just --fmt --check
    uv run pytest -n auto --cov=src/unibm --cov-report=term-missing --cov-fail-under=89
    uv run ruff format --check .
    uv run ruff check .

format: _require-external-uv-env
    just --fmt
    uv run ruff format .

[private]
_docs-build: _require-external-uv-env
    uv sync --dev
    uv run mkdocs build --strict

docs: _require-external-uv-env
    just _docs-build
    uv run mkdocs serve

# Utilities
clean-generated: _require-manuscript-dir
    mkdir -p out/benchmark/cache
    find out -mindepth 1 -maxdepth 1 ! -name benchmark -exec rm -rf {} +
    find out/benchmark -mindepth 1 -maxdepth 1 ! -name cache -exec rm -rf {} +
    rm -rf "${DIR_MANUSCRIPT:-../UniBM_manuscript}/Figure" "${DIR_MANUSCRIPT:-../UniBM_manuscript}/Table"
    rm -f "${DIR_MANUSCRIPT:-../UniBM_manuscript}/paper_subset_manifest.json"
