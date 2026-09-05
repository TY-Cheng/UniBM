# Canonical application data

This directory contains the small provider snapshots needed to reproduce the UniBM
applications without network access. All providers are bounded by the shared analysis cutoff
`2025-12-31`.

Tracked inputs are the six curated USGS candidate extracts, two GHCN station extracts, two
OpenFEMA NFIP state extracts, monthly NSA CPI-U (`CUUR0000SA0`), the USGS site registries, and
`metadata/sources.json`. Generated series and OpenFEMA yearly download chunks are ignored.

The climate case studies use consecutive seasonal suffixes whose analysis-ready daily coverage
is at least 97% in every retained season: June--November for Houston precipitation and
April--October for Phoenix hot-dry severity. A Phoenix day is analysis-ready only when TMAX is
finite and its 30-day precipitation window is fully observed. Streamflow uses the longest
consecutive daily suffix ending at the cutoff; its comparator maxima use January--December years
meeting the same 97% daily-coverage gate. Missing sensor observations are not imputed.
NFIP payouts are adjusted by loss-month CPI-U to the official 2025 annual-average base of
`321.943`. Because BLS does not report October 2025, that month is explicitly imputed as the
geometric mean of September and November and marked in the CPI file.

The NFIP calendar-day series spans the provider acquisition window from 1978-01-01 through the
cutoff. A zero means that the event ledger has no recorded building payout for that day; it is
not an imputation for a missing sensor observation. Severity uses only positive claim-active
days, while persistence uses the calendar-day series.

Run `just refresh-data` to replace the snapshots from their providers. The command refuses to
run when `data/` already has a Git diff, writes gzip files deterministically and atomically, and
leaves the resulting data changes for review.
