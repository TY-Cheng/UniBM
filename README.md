# UniBM

UniBM is a Python package for dependence-aware block-maxima inference in
environmental extremes.

It exposes two complementary inferential targets:

- severity via the extreme value index (EVI) and design-life levels
- persistence via the extremal index (EI)

The reusable package lives under `src/unibm`. Repository-level benchmark,
application, and vignette workflows are orchestrated through the root
`justfile`.

## Package surface

The public package is organized around four entrypoints:

- `unibm` for the headline EVI and design-life-level calls
- `unibm.evi` for the severity-side workflow
- `unibm.ei` for the persistence-side workflow
- `unibm.cdf` for the public empirical CDF helper

## Quick start

1. Copy the local environment template:

```bash
cp .env.example .env
```

2. Edit `.env` and set:

- `DIR_WORK` to your local clone of this repository
- `UV_PROJECT_ENVIRONMENT` to a dedicated external uv environment path

3. Run the lightest repo entrypoint:

```bash
just verify
```

Top-level `just` tasks load `.env` automatically and sync the development
environment before they run. If you prefer raw `uv` commands, load `.env`
first and then run:

```bash
set -a; source .env; set +a
uv sync --dev
```

## Documentation

Package documentation is available at:

- [https://ty-cheng.github.io/UniBM/](https://ty-cheng.github.io/UniBM/)

Useful local docs commands:

```bash
just docs
just docs-serve
```

`just docs` builds the static site under `site/`.
`just docs-serve` launches the local preview server.

## Main repo entrypoints

The stable top-level entrypoints are:

- `just verify`
- `just docs`
- `just docs-serve`
- `just full`

`just full` is the full repo rebuild.
For benchmark, application, vignette, and manuscript-artifact details, see the
root `justfile`.

## Minimal package example

```python
import numpy as np
from unibm import estimate_design_life_level, estimate_evi_quantile

sample = np.random.default_rng(7).pareto(2.0, 4096) + 1.0
fit = estimate_evi_quantile(sample, quantile=0.5, sliding=True, bootstrap_reps=120)
design_life = estimate_design_life_level(fit, years=np.array([10.0, 50.0]))
```

See the docs site for API details, returned objects, and worked examples.
