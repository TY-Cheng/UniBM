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

CI tests Python 3.11–3.14 and builds the documentation. In the pull request, state the
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

## Local release preparation

[Version 0.1.0 is published on PyPI](https://pypi.org/project/unibm/0.1.0/).
The commands below illustrate that release's workflow. For a new release, update
`src/unibm/__about__.py` and use matching new versioned paths throughout; preserve
the existing release artifacts. Build and inspect both distributions locally:

```sh
just --command uv run pytest -q tests/test_distribution_artifacts.py
just --command uv build --out-dir dist/release-0.1.0
just --command uvx twine check --strict dist/release-0.1.0/unibm-0.1.0-py3-none-any.whl dist/release-0.1.0/unibm-0.1.0.tar.gz
```

The artifact test checks archive contents and installs both the wheel and source
distribution into isolated environments outside the checkout. It runs EVI/EI
estimation, design-life intervals, and PNG export using only declared runtime
dependencies. `uv build` also builds its wheel from the source distribution by
default. CI runs the artifact test on each supported Python version.

Run `just check-full` and the strict documentation build before release. Review
the exact two files being uploaded; do not publish a wildcard covering old builds.
The wheel contains only `unibm` and distribution metadata; the source distribution
adds the build configuration, README, and license. Neither contains research
scripts, datasets, local configuration, or generated reports.

Building does not publish anything. When the maintainer is ready to upload, the
existing `testpypi` index can be used for a rehearsal:

```sh
just --command uv publish --index testpypi dist/release-0.1.0/unibm-0.1.0-py3-none-any.whl dist/release-0.1.0/unibm-0.1.0.tar.gz
```

For local uploads, supply the appropriate index's API token through the process
environment variable `UV_PUBLISH_TOKEN`; do not put token values in repository
files or command history. Trusted publishing instead requires a configured CI
publisher. After verifying the TestPyPI installation, use PyPI credentials to
upload the same reviewed files to PyPI as a separate action:

```sh
just --command uv publish dist/release-0.1.0/unibm-0.1.0-py3-none-any.whl dist/release-0.1.0/unibm-0.1.0.tar.gz
```

Confirm the uploaded version installs outside the checkout before updating the
installation instructions in the README and getting-started guide.
Creating Git tags, GitHub releases, and publishing the documentation are separate
maintainer actions. See the [uv publishing guide](https://docs.astral.sh/uv/guides/package/)
for authentication and installation details.
