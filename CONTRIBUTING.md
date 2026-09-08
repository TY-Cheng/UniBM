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
uv sync --locked --dev
```

Library development, tests, and documentation builds do not require provider credentials,
a manuscript checkout, or a custom `.env` file.

## Checks

Start with the tests relevant to your change, for example:

```sh
uv run pytest -q tests/test_unibm_cdf.py
```

Use `just check` for incremental tests and formatting/lint checks, or `just check-full`
for the full test suite and coverage check. `just format` applies the existing formatters.
For documentation changes, run:

```sh
uv run mkdocs build --strict
```

CI tests Python 3.11–3.14 and builds the documentation. In the pull request, state the
checks you actually ran and any relevant checks you did not run.

## Research results and data

For numerical changes, include a small reproducible example or regression test. Record
the random seed and relevant settings, and explain changes to estimates or uncertainty.
Prefer synthetic examples; do not post personal data, credentials, or large raw datasets.

`just full` is a research-artifact rebuild, not a routine contribution check: it removes
and regenerates outputs, including files in the separate manuscript repository. Coordinate
benchmark, application, data-refresh, and manuscript reruns with the maintainers. Do not
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
