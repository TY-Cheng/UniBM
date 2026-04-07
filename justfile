set shell := ["zsh", "-cu"]
set dotenv-load := true

default:
    @just --list

# Setup
sync:
    uv sync --dev

# Rebuild
freeze-usgs screening_bootstrap="20":
    UNIBM_SCREENING_BOOTSTRAP_REPS={{screening_bootstrap}} uv run python scripts/application/freeze_usgs.py

application workers="6":
    UNIBM_APPLICATION_WORKERS={{workers}} uv run python scripts/application/build.py

applications workers="6" screening_bootstrap="20":
    just freeze-usgs "{{screening_bootstrap}}"
    just application "{{workers}}"

evi-benchmark workers="6":
    UNIBM_BENCHMARK_WORKERS={{workers}} uv run python scripts/benchmark/evi_benchmark.py

ei-benchmark workers="6":
    UNIBM_BENCHMARK_WORKERS={{workers}} uv run python scripts/benchmark/ei_benchmark.py

benchmarks workers="6":
    just evi-benchmark "{{workers}}"
    just ei-benchmark "{{workers}}"

reports:
    uv run python scripts/benchmark/evi_report.py
    uv run python scripts/benchmark/ei_report.py

vignette:
    uv run jupytext --sync notebooks/vignette.py

rebuild workers="6" screening_bootstrap="20":
    just benchmarks "{{workers}}"
    just reports
    just applications "{{workers}}" "{{screening_bootstrap}}"
    just vignette

full workers="6" screening_bootstrap="20":
    just rebuild "{{workers}}" "{{screening_bootstrap}}"
    just qa
    just docs

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

# Utilities
format:
    uvx ruff format ./**/*.py ./**/*.ipynb

profile-sliding:
    uv run python scripts/shared/profile_sliding_windows.py
