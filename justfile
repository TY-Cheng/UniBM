set shell := ["zsh", "-cu"]
set dotenv-load := true

default:
    @just --list

# Main Entrypoints
full workers="6" screening_bootstrap="20":
    just clean-generated
    just sync
    just rebuild "{{workers}}" "{{screening_bootstrap}}"
    just vignette-execute
    just qa
    just docs
    just format

rebuild workers="6" screening_bootstrap="20":
    just benchmark "{{workers}}"
    just application "{{workers}}" "{{screening_bootstrap}}"
    just vignette

benchmark workers="6":
    just benchmark-sim "{{workers}}"
    just benchmark-reports

application workers="6" screening_bootstrap="20":
    just freeze-usgs "{{screening_bootstrap}}"
    just application-build "{{workers}}"

# Setup
sync:
    uv sync --dev

# Benchmark
benchmark-sim workers="6":
    just evi-benchmark "{{workers}}"
    just ei-benchmark "{{workers}}"

benchmark-reports:
    uv run python scripts/benchmark/evi_report.py
    uv run python scripts/benchmark/ei_report.py

evi-benchmark workers="6":
    UNIBM_BENCHMARK_WORKERS={{workers}} uv run python scripts/benchmark/evi_benchmark.py

ei-benchmark workers="6":
    UNIBM_BENCHMARK_WORKERS={{workers}} uv run python scripts/benchmark/ei_benchmark.py

# Application
freeze-usgs screening_bootstrap="20":
    UNIBM_SCREENING_BOOTSTRAP_REPS={{screening_bootstrap}} uv run python scripts/application/freeze_usgs.py

application-build workers="6":
    UNIBM_APPLICATION_WORKERS={{workers}} uv run python scripts/application/build.py

# QA
test:
    uv run python -m unittest discover -s tests -p 'test_*.py'

coverage:
    rm -f .coverage
    uv run coverage run -m unittest discover -s tests -p 'test_*.py'
    uv run coverage report -m

coverage-html:
    rm -f .coverage
    uv run coverage run -m unittest discover -s tests -p 'test_*.py'
    uv run coverage html

check:
    uvx ruff check scripts tests notebooks

qa:
    just test
    just coverage
    just check

# Docs
docs:
    uv run sphinx-build -b html docs docs/_build/html

docs-clean:
    rm -rf docs/_build

# Notebook
vignette:
    uv run jupytext --sync notebooks/vignette.py
    just format

vignette-execute:
    uv run --with nbconvert --with ipykernel jupyter-nbconvert --to notebook --execute --inplace notebooks/vignette.ipynb
    just format

# Utilities
clean-generated:
    mkdir -p out/benchmark/cache
    find out -mindepth 1 -maxdepth 1 ! -name benchmark -exec rm -rf {} +
    find out/benchmark -mindepth 1 -maxdepth 1 ! -name cache -exec rm -rf {} +
    rm -rf UniBM_manuscript/Figure UniBM_manuscript/Table

format:
    uvx ruff format ./**/*.py ./**/*.ipynb

profile-sliding:
    uv run python scripts/shared/profile_sliding_windows.py
