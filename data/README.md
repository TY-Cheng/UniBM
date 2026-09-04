# Canonical application data

This directory contains the small provider snapshots needed to reproduce the UniBM
applications without network access. All providers are bounded by the shared analysis cutoff
`2025-12-31`.

Tracked inputs are the six curated USGS candidate extracts, two GHCN station extracts, two
OpenFEMA NFIP state extracts, monthly NSA CPI-U (`CUUR0000SA0`), the USGS site registries, and
`metadata/sources.json`. Generated series and OpenFEMA yearly download chunks are ignored.

The climate case studies use complete consecutive seasonal suffixes: June--November for Houston
precipitation and April--October for Phoenix hot-dry severity (including the preceding 29 days
needed by its 30-day precipitation window). Streamflow uses the longest consecutive daily
suffix ending at the cutoff; its comparator maxima use complete October--September water years.
NFIP payouts are adjusted by loss-month CPI-U to the official 2025 annual-average base of
`321.943`. Because BLS does not report October 2025, that month is explicitly imputed as the
geometric mean of September and November and marked in the CPI file.

Run `just refresh-data` to replace the snapshots from their providers. The command refuses to
run when `data/` already has a Git diff, writes gzip files deterministically and atomically, and
leaves the resulting data changes for review.
