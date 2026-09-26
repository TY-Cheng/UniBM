# GOES soft X-rays

This case estimates block-maximum scaling in **EWMA-normalized hourly soft
X-ray irradiance**. It retains the hourly calendar and uses complete sliding
windows across the available history. It reports **OLS EVI point estimates
and design-life curves only: no EI and no CI**.

## Source and hourly aggregation

The source is NOAA's science-quality GOES XRS-B **0.1–0.8 nm** one-minute mean
irradiance archive. The reconstructed calendar covers **1995-01-03–2025-12-31**.
It is a retrospective science product, not the finest native sensor sampling
and not total solar irradiance.

The snapshot builder follows the documented satellite priority schedule, using
secondary observations for gaps and treating GOES-13 as fallback. It retains
finite positive unmasked flux, excludes nonzero flux quality flags and unresolved
saturation, and requires valid electron correction where provided. It neither
selects the largest reading across satellites nor adds an empirical rescaling.

For each UTC hour, take the maximum of the accepted minute means only if
**at least 57 of 60 minutes** are valid and no unresolved saturation remains.
Otherwise retain a missing hour. This is a maximum of minute averages, not an
instantaneous irradiance peak; even three missing minutes could conceal a peak.

## Normalize without compressing gaps

For each uninterrupted qualified hourly run:

```text
X_t = hourly maximum of qualified minute-mean irradiance
Y_t = X_t / m_t
m_t = ρ m_(t−1) + (1−ρ) X_(t−1),  ρ = 2^(−1/4320)
```

Initialize with the mean of its first **720 hours (30 days)**, then analyze from
the next hour. The half-life is **4,320 hours (180 days)**. Restart after every
gap. Runs no longer than the warmup contribute no normalized observations.
Missing and warmup positions remain missing; no interpolation or time compression
is performed. `Y_t` is a dimensionless irradiance-to-past-level ratio, not W/m².

The 271,704-hour calendar contains 266,000 qualified hours before normalization.
Restarted warmups leave **94,226 normalized hours across 64 runs**, spanning
**1998-08-21 19:00–2025-12-31 23:00 UTC**. These endpoints do not imply complete
coverage between them. The displayed time series uses these same eligible hours.

## EVI point diagnostic

The fitted EVI is **0.885**, with selected block sizes **65–504 hours**.
The grid is based on the complete calendar length, including missing positions;
only fully observed windows contribute to a block summary. Consequently the
`N/17` cap does not guarantee 17 usable disjoint blocks after missingness.

<figure class="unibm-figure">
  <a href="../../assets/cases/goes_normalized.png"><img src="../../assets/cases/goes_normalized.png" alt="GOES normalized hourly irradiance: complete-window summary stability, OLS EVI scaling, gapped time series, and relative design-life levels."></a>
  <figcaption>The time-series panel replaces EI. No parameter or design-life confidence intervals are reported. Click the figure for full size.</figcaption>
</figure>

[Download numerical results and preparation settings (JSON)](../assets/cases/goes_normalized.json).

Complete-window counts are provided in the JSON record. Overlapping windows
are dependent and are not independent replications. EWMA restarts and complete-window
selection change the contribution of different years and solar-cycle phases at
each block size; missingness need not be random.

The existing FGLS bootstrap has not been validated for this gapped construction.
The API's OLS HC0 interval is therefore suppressed in the published results.
Design-life curves use 8,766 hours per year and express maxima of the relative
series under a stationary working model. They do not convert to future W/m²
without modeling future scales. Solar cycles, residual calibration differences,
and instrument limits remain relevant after normalization.

## Reproduce

Using the local quality-screened hourly input and its hash-bearing JSON:

```bash
PYTHONPATH=scripts uv run python -m application.docs_cases --keys goes
```

To rebuild that input from NOAA files, the optional downloader requires `netCDF4`;
it is not a package runtime dependency. Downloading the historical archive can
be substantial, so this step is separate from the normal application workflow:

```bash
uv run --with netCDF4 python scripts/data_prep/goes_snapshot.py
uv run python scripts/data_prep/goes_hourly.py
```

`goes_snapshot.py --offline` reuses the previously saved listings and NetCDF files.
Inputs remain local under `data/raw/goes` and `data/processed/inputs`.
The scripts record source URLs, quality rules, satellite priorities, and SHA-256
hashes. Available archive versions may change, so a new retrieval need not recreate
the frozen result exactly.

Sources: [NOAA legacy science archive](https://data.ngdc.noaa.gov/instruments/solar-space-observing/particle-detectors/sem/goes/access/science/xrs/)
and [GOES-R archive](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/).
