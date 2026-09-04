# Climate extremes

<p class="unibm-case-intro">
How does block-maximum severity scale in a direct precipitation record and in a derived
compound hot–dry index? These GHCN cases exercise the severity branch only; neither is used for
formal extremal-index inference.
</p>

## Houston precipitation

**Question.** How quickly does the upper tail of wet-season daily precipitation grow across
longer block sizes at one long-record station?

**Data.** GHCN-Daily station `USW00012918` (Houston William P. Hobby Airport), restricted to
complete June–November seasons from 1984 through 2025. The retained series has 7,686 daily
observations in millimetres and ends on 2025-11-30.

**Estimand.** The EVI `ξ` from the median sliding-block maximum scaling path. Design-life curves
use the seasonal-day clock, mapped at 183 observations per year.

<div class="unibm-stat-grid">
  <div class="unibm-stat"><strong>0.407</strong><span>fitted EVI ξ</span></div>
  <div class="unibm-stat"><strong>[0.281, 0.534]</strong><span>conditional 95% interval</span></div>
  <div class="unibm-stat"><strong>37–75 days</strong><span>selected block-size plateau</span></div>
</div>

<figure class="unibm-figure">
  <img src="../../assets/cases/houston_precipitation.png" alt="Four-panel diagnostic for Houston wet-season daily precipitation, showing target stability, block-maximum scaling, the observed series, and design-life levels.">
  <figcaption>Houston precipitation diagnostics. The lower-left panel replaces EI with the observed seasonal series because this case has no formal EI estimand.</figcaption>
</figure>

## Phoenix compound hot–dry severity

**Question.** Can the same severity workflow describe the upper tail of a compound hot–dry
signal without pretending that the signal is a directly observed physical variable?

**Data.** GHCN-Daily station `USW00023183` (Phoenix Airport), restricted to complete
April–October seasons from 1948 through 2025. The retained series has 16,692 daily observations
and ends on 2025-10-31.

<div class="unibm-method-box">
  <strong>Derived-index definition</strong>
  <p>
    The dimensionless daily index adds the positive standardized <code>TMAX</code> anomaly to the
    positive standardized 30-day precipitation deficit. Standardization is by retained
    day-of-year. It is constructed from GHCN <code>TMAX</code> and <code>PRCP</code>; it is not a
    native GHCN element or an operational drought index.
  </p>
</div>

**Estimand.** The EVI `ξ` from the median sliding-block maximum scaling path. Design-life curves
use the warm-season-day clock, mapped at 214 observations per year.

<div class="unibm-stat-grid">
  <div class="unibm-stat"><strong>0.292</strong><span>fitted EVI ξ</span></div>
  <div class="unibm-stat"><strong>[0.181, 0.404]</strong><span>conditional 95% interval</span></div>
  <div class="unibm-stat"><strong>14–21 days</strong><span>selected block-size plateau</span></div>
</div>

<figure class="unibm-figure">
  <img src="../../assets/cases/phoenix_hotdry.png" alt="Four-panel diagnostic for Phoenix compound hot-dry severity, showing target stability, block-maximum scaling, the derived daily series, and design-life levels.">
  <figcaption>Phoenix compound hot–dry diagnostics. The index is useful here as a worked statistical target, not as a validated hazard scale.</figcaption>
</figure>

## Limits and reproduction

Both results are station-specific, conditional on approximate stationarity over the retained
seasonal windows, and intended as research illustrations rather than attribution, forecasting,
or operational guidance. The source inputs are fixed at the shared 2025-12-31 cutoff.

- Preparation: [`scripts/data_prep/ghcn.py`](https://github.com/TY-Cheng/UniBM/blob/main/scripts/data_prep/ghcn.py)
- Case registry: [`scripts/application/specs.py`](https://github.com/TY-Cheng/UniBM/blob/main/scripts/application/specs.py)
- Provider files: [NOAA GHCN by-station archive](https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/)
