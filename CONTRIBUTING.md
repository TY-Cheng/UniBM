# Contributing to UniBM

Bug reports, tests, documentation corrections, and focused improvements are welcome.
Please follow the [Code of Conduct](CODE_OF_CONDUCT.md). Report security vulnerabilities
privately as described in [SECURITY.md](SECURITY.md), not in a public issue.

## Before changing code

Check existing [issues](https://github.com/TY-Cheng/UniBM/issues) for related work.
For a substantial API or statistical-method change, discuss the intended behavior in
an issue before implementing it. Small fixes can go directly to a pull request.

The project is under active development; APIs may change before a stable release.
The repository separates the library (`src/unibm/`), research workflows (`scripts/`),
tests (`tests/`), and documentation (`docs/`). See [Concepts](docs/concepts.md) and
[Reading Returned Objects](docs/reading-returned-objects.md) for workflows and inference contracts.

## Development setup

Use Python 3.11 or later, [uv](https://docs.astral.sh/uv/), and
[just](https://just.systems/). From your clone of the repository:

```sh
just --command uv sync --locked --dev
```

The `just` recipes require zsh. On Windows without zsh, use uv directly:

```sh
uv sync --locked --dev
uv run pytest -n auto --cov=src/unibm --cov-report=term-missing --cov-fail-under=89
uv run ruff format --check .
uv run ruff check .
```

Export any environment overrides in your shell first; these direct commands do
not load `.env`. CI likewise runs uv directly on each operating system.

Library development, tests, and documentation builds do not require provider credentials,
a report checkout, or a custom `.env` file.

## Checks

Start with the tests relevant to your change, for example:

```sh
just --command uv run pytest -q tests/test_unibm_cdf.py
```

Use `just check` for incremental tests and formatting/lint checks, or `just check-full`
for the full test suite and coverage check. `just format` applies the existing formatters.
For documentation changes, run:

```sh
just --command uv run mkdocs build --strict
```

CI is configured to test native and NumPy fallback execution on Python 3.11 and
3.14 across Linux, macOS, and Windows, and to build the documentation. A separate
wheel workflow targets CPython 3.11–3.14 on Linux glibc x86_64/aarch64, macOS
Intel/ARM64, and Windows AMD64. Confirm the jobs passed for the exact commit;
the matrix is not evidence that a particular build succeeded. Those wheels are
CI artifacts; the workflow does not publish. In the pull request, state the
checks you actually ran and any relevant checks you did not run.

## Research results and data

For numerical changes, include a small reproducible example or regression test. Record
the random seed and relevant settings, and explain changes to estimates or uncertainty.
Prefer synthetic examples; do not post personal data, credentials, or large raw datasets.

`just full` is a research-artifact rebuild, not a routine contribution check: it removes
and regenerates named outputs in the code repository and selected report destination.
The latter defaults to `out/reports/`; set `UNIBM_REPORT_DIR` explicitly for an external
project. Caches, research notes, and unrecognized files are preserved. Coordinate
benchmark, application, data-refresh, and report reruns with the maintainers. Do not
hand-edit generated results or silently replace frozen provider data.

## Dependencies and pull requests

Dependency maintenance uses Dependabot alerts only. Maintainers review alerts and update
dependencies manually; automatic security-update PRs and scheduled version-update PRs are
not enabled. When changing dependencies, update `uv.lock` as needed and test the resulting
environment. Keep changes to dependency constraints deliberate and reviewable.

Keep each pull request focused, describe its purpose, and update tests and documentation
when behavior changes. Contributors are responsible for reviewing and understanding all
submitted code, including tool-assisted changes. Code contributions are made under the
project's [MIT License](LICENSE).

## Native acceleration

The source checkout optionally compiles two EVI kernels: repeated-value KDE and
integer quantile rank search. Statistical APIs, fitting and CI construction stay
in Python/NumPy/SciPy. Cython and setuptools are build dependencies only; the
extension uses Python buffers without the NumPy C API, OpenMP, fast-math or
machine-specific CPU flags. Existing per-call thread pools run the GIL-free work.

Source builds use the host C compiler (Xcode Command Line Tools on macOS, a C
compiler and Python development headers on Linux, or Visual Studio C++ Build Tools
on Windows). If compilation is unavailable, installation retains the NumPy
implementation. Set `UNIBM_NO_EXTENSIONS=1` before building to explicitly build a
pure Python wheel, or before starting Python to disable an installed extension.
The setting is read once at import; changing it inside a running process has no
effect. All public methods remain available in both modes.

For a local fallback check without rebuilding the installed extension (zsh/bash):

```sh
UNIBM_NO_EXTENSIONS=1 just --command uv run --no-sync pytest -q tests/test_acceleration.py tests/test_bootstrap_parallel.py
```

In PowerShell, set `$env:UNIBM_NO_EXTENSIONS = "1"` before the equivalent direct
`uv run --no-sync pytest ...` command, then remove it with
`Remove-Item Env:UNIBM_NO_EXTENSIONS` when finished.

After editing `.pyx` sources, rebuild the editable install:

```sh
just --command uv sync --locked --dev --reinstall-package unibm
just --command uv run python -c "from unibm.evi._accelerator import kernels; assert kernels is not None"
```

## Local release preparation

Check [PyPI](https://pypi.org/project/unibm/) and
[GitHub releases](https://github.com/TY-Cheng/UniBM/releases) for published versions.
Set the next version in `src/unibm/__about__.py` before building; preserve old artifacts.

```sh
just --command uv run pytest -q tests/test_distribution_artifacts.py
just --command uv build --out-dir dist/release-next
just --command uvx twine check --strict dist/release-next/*
```

Build the portable fallback separately (zsh/bash):

```sh
UNIBM_NO_EXTENSIONS=1 just --command uv build --out-dir dist/release-next-pure
just --command uvx twine check --strict dist/release-next-pure/*
```

On PowerShell, use the environment setting above with the direct uv commands.
Take the `py3-none-any.whl` from the pure build and one source distribution from
the reviewed build; do not upload two copies of the same-version sdist.

The artifact test builds a native-capable wheel, a source distribution and an
explicit pure Python wheel. It installs each outside the checkout and exercises
EVI/EI estimation, mode and quantile bootstrap, design-life intervals, and PNG
export using declared runtime dependencies. `uv build` builds its wheel from the
source distribution by default. A local native wheel covers only its tagged
Python/platform combination; use the wheel CI artifacts for other targets.
The current wheel matrix excludes musl Linux, Windows ARM64, PyPy, and
free-threaded CPython. The pure wheel provides the NumPy implementation where
compatible runtime dependencies are available; it does not establish support
for every interpreter or platform.

Run `just check-full` and the strict documentation build before release. Review
the exact files and SHA256 hashes to upload, including the tested platform wheels
and a pure Python wheel for platforms without a native build. The wheel contains
only `unibm` and distribution metadata. The source distribution also contains
build configuration, the `.pyx` source, README, license and package metadata;
research scripts, datasets, local configuration and generated reports are excluded.

### GitHub Trusted Publishing

The `release` workflow runs **only through Actions → release → Run workflow** on
`main`. Commits, pushes, pull requests, tags and draft releases do not trigger it.
Provide an existing version tag, such as `v0.2.0`, whose draft release contains
the verified wheels, sdist, `SHA256SUMS` and `release-manifest.json`.

The manifest records `version`, the full source `commit`, `artifacts` (filename
to SHA256), and `validation_runs` (the successful `ci`, `wheels` and `docs` run IDs
for that commit). The workflow checks the tag, hashes and matching CI, uploads the
same files to TestPyPI, verifies registry hashes and installation, repeats for
PyPI, then publishes the GitHub Release. It never rebuilds the distributions.
An interrupted upload can be retried with the same tag and identical files;
never replace already published distribution contents or move the version tag.

Configure a Trusted Publisher separately on PyPI and TestPyPI with owner
`TY-Cheng`, repository `UniBM`, workflow filename `release.yml`, and environment
`release`. The GitHub `release` environment permits deployments from `main`.
GitHub supplies short-lived OIDC credentials; no saved API token is required.
See [PyPI's configuration guide](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

### Manual local upload

Building does not publish. As an alternative, upload the reviewed files first with `uv publish
--index testpypi` for rehearsal, then with `uv publish` for PyPI as a separate
maintainer action. Supply file paths explicitly rather than a wildcard of old
builds. Supply local tokens via `UV_PUBLISH_TOKEN`, never repository files or
command history; trusted publishing requires a separately configured publisher.
Verify installation outside the checkout before updating release installation
instructions. Git tags, GitHub releases and documentation publication remain
separate maintainer actions. See the [uv publishing guide](https://docs.astral.sh/uv/guides/package/).
