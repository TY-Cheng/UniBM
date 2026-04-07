set shell := ["zsh", "-cu"]
set dotenv-load := true

default:
    @just --list

sync:
    uv sync --dev

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

freeze-usgs screening_bootstrap="20":
    UNIBM_SCREENING_BOOTSTRAP_REPS={{screening_bootstrap}} uv run python scripts/workflows/freeze_usgs_station_selection.py

application workers="6":
    UNIBM_APPLICATION_WORKERS={{workers}} uv run python scripts/workflows/application.py

rebuild-applications workers="6" screening_bootstrap="20":
    just applications "{{workers}}" "{{screening_bootstrap}}"

applications workers="6" screening_bootstrap="20":
    UNIBM_SCREENING_BOOTSTRAP_REPS={{screening_bootstrap}} uv run python scripts/workflows/freeze_usgs_station_selection.py
    UNIBM_APPLICATION_WORKERS={{workers}} uv run python scripts/workflows/application.py

evi-benchmark workers="6":
    UNIBM_BENCHMARK_WORKERS={{workers}} uv run python scripts/workflows/evi_benchmark.py

ei-benchmark workers="6":
    UNIBM_BENCHMARK_WORKERS={{workers}} uv run python scripts/workflows/ei_benchmark.py

rebuild-benchmarks workers="6":
    just benchmarks "{{workers}}"

benchmarks workers="6":
    UNIBM_BENCHMARK_WORKERS={{workers}} uv run python scripts/workflows/evi_benchmark.py
    UNIBM_BENCHMARK_WORKERS={{workers}} uv run python scripts/workflows/ei_benchmark.py

reports:
    uv run python scripts/workflows/evi_report.py
    uv run python scripts/workflows/ei_report.py

rebuild-all workers="6" screening_bootstrap="20":
    uv sync --dev
    UNIBM_BENCHMARK_WORKERS={{workers}} uv run python scripts/workflows/evi_benchmark.py
    UNIBM_BENCHMARK_WORKERS={{workers}} uv run python scripts/workflows/ei_benchmark.py
    uv run python scripts/workflows/evi_report.py
    uv run python scripts/workflows/ei_report.py
    UNIBM_SCREENING_BOOTSTRAP_REPS={{screening_bootstrap}} uv run python scripts/workflows/freeze_usgs_station_selection.py
    UNIBM_APPLICATION_WORKERS={{workers}} uv run python scripts/workflows/application.py
    uv run python scripts/rebuild_vignette.py

rebuild workers="6" screening_bootstrap="20":
    just rebuild-all "{{workers}}" "{{screening_bootstrap}}"

full workers="6" screening_bootstrap="20":
    just rebuild "{{workers}}" "{{screening_bootstrap}}"
    just qa
    just docs

vignette:
    uv run python scripts/rebuild_vignette.py

docs:
    uv run sphinx-build -b html docs docs/_build/html

docs-clean:
    rm -rf docs/_build

profile-sliding:
    uv run python scripts/workflows/profile_sliding_windows.py

format:
    uvx ruff format ./**/*.py ./**/*.ipynb

check:
    uvx ruff check scripts tests

qa:
    uv run python -m unittest discover -s tests -p 'test_*.py'
    rm -f .coverage
    uv run coverage run -m unittest discover -s tests -p 'test_*.py'
    uv run coverage report -m
    uvx ruff check scripts tests
