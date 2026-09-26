# Canonical application data

## Directory layout

```text
data/
├── raw/                     # Frozen provider snapshots and acquisition records
├── processed/
│   ├── inputs/              # Prepared local case inputs and provenance JSON
│   └── applications/        # Generated display/EVI/EI series and their distinct clocks
└── metadata/                # Provider registry and USGS site selection
```

`processed/applications/` replaces `derived/applications/`;
`processed/inputs/` replaces `processed/pilots/`. Raw records are grouped directly
by provider under `raw/`, including `raw/goes/` and `raw/finance/`.
Generated and optional local data remain Git-ignored.

The optional GOES and SPY/QQQ inputs support the published docs cases. Their
raw acquisition records live under `raw/goes/` and `raw/finance/`.
GOES retains both its quality-screened minute archive and hourly maxima: the
minute data are required to rebuild the hourly series. SPY/QQQ retain adjusted
signed returns and zero-clamped losses; EWMA normalization is applied when
loading the cases. See the case pages for sampling and normalization details.

Houston, Phoenix and streamflow are reconstructed directly from the tracked raw
snapshots, without intermediate pilot copies. PM2.5, sunspot and GOES daily pilots
are retired. Historical experiment outputs are separate from current inputs.

## Tracked snapshots and main application series

This directory contains the small provider snapshots needed to reproduce the UniBM
applications without network access. All providers are bounded by the shared analysis cutoff
`2025-12-31`.

Tracked inputs are the six curated USGS candidate extracts, two GHCN station extracts, two
OpenFEMA NFIP state extracts, monthly NSA CPI-U (`CUUR0000SA0`), the USGS site registries, and
`metadata/sources.json`. Generated series and OpenFEMA yearly download chunks are ignored.

The application and docs workflows use the same full-calendar Houston and Phoenix
series: select the longest continuous qualified daily segment, then divide by a
lagged EWMA level with 180-day half-life and 30-day warmup. True zeros remain.
A Phoenix day is qualified only when TMAX is finite and its 30-day precipitation
window is fully observed. Its severity construction is retrospective, not a
real-time forecasting signal. Streamflow uses the longest
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

`just data` checks the frozen core inputs and refreshes USGS screening; it does not
fetch optional provider histories or rebuild figures. `just application` writes
the selected display/EVI/EI series to `processed/applications/`, indexes them in
`out/applications/application_series_registry.csv`, and generates the local report
and docs assets from the same fitted bundles. GOES and SPY/QQQ join those outputs
when their local CSV/provenance pairs are available. Missing optional pairs are
reported and frozen docs assets are kept; incomplete or invalid pairs fail.
