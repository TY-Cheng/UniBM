# UniBM

UniBM is a Python package for dependence-aware block-maxima inference in
environmental extremes.

It exposes two complementary inferential targets:

- severity via the extreme value index (EVI) and design-life levels
- persistence via the extremal index (EI)

The installable package lives under `src/unibm`. Repository-level benchmark,
application, manuscript, and static-site workflows are orchestrated through the
root `justfile`.

## Package surface

The public package is organized around four entrypoints:

- `unibm` for the headline EVI and design-life-level calls
- `unibm.evi` for the severity-side workflow
- `unibm.ei` for the persistence-side workflow
- `unibm.cdf` for the public empirical CDF helper

## Quick start

1. Run the lightest repo entrypoint:

```bash
just check
```

The repo defaults to `data/`, the sibling `../UniBM_manuscript` checkout, and
uv's normal project environment. Copy `.env.example` only when overriding the
last two locations:

```bash
cp .env.example .env
```

Top-level `just` tasks load `.env` automatically when present and sync the
development environment before they run. Raw `uv` commands need no environment
setup unless you use an override:

```bash
uv sync --locked --dev
```

## Documentation

Package documentation is available at:

- [https://ty-cheng.github.io/UniBM/](https://ty-cheng.github.io/UniBM/)

Useful local docs command:

```bash
uv run mkdocs serve
```

This builds the static site under `site/` and launches the local preview server.

UniBM is not yet released on PyPI. The documentation therefore describes source
installation until a separately reviewed package release is approved.

## Main repo entrypoints

The stable top-level entrypoints are:

- `just check`
- `just check-full`
- `just data`
- `just refresh-data`
- `just full`

`just check` runs tests affected by local changes in parallel, then checks all
formatting and lint rules. `just check-full` runs the complete parallel test
suite with the coverage gate.
`just data` validates the tracked canonical inputs and prepares the four paper
cases without network access. `just refresh-data` is the only networked data
entrypoint; it refreshes the fixed-cutoff provider snapshots and leaves their
Git diff for review. `just full` rebuilds the manuscript artifacts and static
website snapshots offline. For benchmark, application, and manuscript-artifact
details, see the root `justfile`.

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
)
design_life = estimate_design_life_level(fit, years=np.array([10.0, 50.0]))
```

See the docs site for API details, returned objects, and worked examples.

The example uses the default adaptive policy with checkpoints 128, 256, 512,
768, and 1024. An explicit integer such as `bootstrap_reps=480` instead fixes R. Inspect
`bootstrap_reps_used` and `bootstrap_precision_met`: reaching the cap does not
imply precision was met. Adaptive R controls numerical Monte Carlo error, not
statistical CI width or coverage. Both EVI/EI FGLS defaults use fixed shrinkage 0.37.
