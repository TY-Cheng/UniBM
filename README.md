# UniBM

UniBM is a Python package for dependence-aware block-maxima inference in
environmental extremes.

It exposes two complementary inferential targets:

- severity via the extreme value index (EVI) and design-life levels
- persistence via the extremal index (EI)

The installable package lives under `src/unibm`. Repository-level benchmark,
application, report, and static-site workflows are orchestrated through the
root `justfile`.

UniBM is developed and maintained by Tuoyuan Cheng under the project supervision
of Kan Chen. It is distributed under the MIT License.

## Package surface

The public package is organized around four entrypoints:

- `unibm` for the headline EVI and design-life-level calls
- `unibm.evi` for the severity-side workflow
- `unibm.ei` for the persistence-side workflow
- `unibm.cdf` for the public empirical CDF helper

## Quick start

Install the package from a source checkout with Python 3.11 or later:

```bash
git clone https://github.com/TY-Cheng/UniBM.git
python -m pip install ./UniBM
```

This installs `unibm`, including its estimators, interval helpers, design-life
levels, and plotting helpers. Research scripts, datasets, and `just full` belong
to the GitHub checkout and are not included in the wheel or source distribution.

For repository development:

```bash
cd UniBM
just check
```

No `.env` or external project is required. Defaults are `data/`, `out/reports/`,
and uv's normal project environment. Copy `.env.example` only to override the
report destination or environment location. Top-level `just` tasks load `.env`
and sync the development environment automatically.

For ad hoc uv commands that should use `.env`, use `just --command`, for example
`just --command uv run pytest -q tests/test_unibm_cdf.py`. A plain `uv sync` or
`uv run` does not automatically load `.env` before choosing its project environment;
without an exported override it uses `.venv/`.

## Results and reports

Calculation results stay in the code repository:

- `out/benchmark/`: benchmark CSVs and sensitivity summaries
- `out/benchmark/cache/`: reusable simulation caches
- `out/applications/`: application CSVs and JSON

Final PDF figures and LaTeX tables go to `Figure/` and `Table/` inside a single
report destination. `UNIBM_REPORT_DIR` unset or blank means `out/reports/`.
An explicit value selects that destination directly, without an additional local
copy. Relative paths are resolved against the code repository root:

```bash
UNIBM_REPORT_DIR="/path/to/your/report-project" just reports
```

An invalid destination raises an error; it never silently falls back. Report
output directories and files must not redirect writes through symlinks.
Table filenames describe their contents, for example `application_summary.tex`;
direct JoH updates use these names and their matching LaTeX labels.
Benchmark figures and tables use paired `benchmark_evi_*` / `benchmark_ei_*`
names; raw benchmark CSVs use `evi_*` / `ei_*`. Application method comparisons
use `application_evi_methods.csv` / `application_ei_methods.csv`. Combined
application summaries retain the general `application_summary` name.
`report_subset_manifest.json` indexes expected paths, labels, producers, and
placements for the curated four-case report subset. It is not a complete
inventory or a verification of generated files. `manifest_code_commit` and
`manifest_code_worktree_dirty` describe the code checkout when the index was
written, not the version used to calculate the listed results.

The cleanup step in `just full` removes only explicitly named workflow outputs for the
selected benchmark sample size and report destination. Those names are reserved
for generated files. It preserves caches, other sample-size runs, research notes,
historical report folders, and unrecognized files. It never deletes an entire
output directory. Web snapshots under `docs/assets/` are refreshed by the report
workflows separately.

## Documentation

Package documentation is available at:

- [https://ty-cheng.github.io/UniBM/](https://ty-cheng.github.io/UniBM/)

Useful local docs command:

```bash
just --command uv run mkdocs serve
```

This builds the static site under `site/` and launches the local preview server.

## Main repo entrypoints

The stable top-level entrypoints are:

- `just check`
- `just check-full`
- `just data`
- `just refresh-data`
- `just benchmark`
- `just application`
- `just reports`
- `just full`

`just check` runs tests affected by local changes in parallel, then checks all
formatting and lint rules. `just check-full` runs the complete parallel test
suite with the coverage gate.
`just data` validates the tracked canonical inputs and prepares the four report
cases without network access. `just refresh-data` is the only networked data
entrypoint; it refreshes the fixed-cutoff provider snapshots and leaves their
Git diff for review. `just reports` reuses valid benchmark summaries, computes
missing ones, and reruns application fits to refresh reports and web snapshots.
`just full` checks the project, cleans named outputs, and rebuilds the benchmark,
application, report, and static-site outputs offline. These research workflows
can be expensive; they are not required to use or install the package.

## Minimal package example

```python
import numpy as np
from unibm import estimate_design_life_level, estimate_evi_quantile

sample = np.random.default_rng(7).pareto(2.0, 4096) + 1.0
fit = estimate_evi_quantile(
    sample,
    regression="FGLS",
    quantile=0.5,
    sliding=True,
    bootstrap_reps="adaptive",
    random_state=7,
)
design_life = estimate_design_life_level(fit, years=np.array([10.0, 50.0]))
```

See the docs site for API details, returned objects, and worked examples.

The example uses the default adaptive policy with checkpoints 128, 256, 512,
768, and 1024. An explicit integer such as `bootstrap_reps=480` instead fixes R. Inspect
`bootstrap_reps_used` and `bootstrap_precision_met`: reaching the cap does not
imply precision was met. Adaptive R controls numerical Monte Carlo error, not
statistical CI width or coverage. Both EVI/EI FGLS defaults use fixed shrinkage 0.37.
