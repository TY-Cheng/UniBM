# Streamflow

<p class="unibm-case-intro">
The paired streamflow cases separate two questions: how daily-discharge severity grows over a
planning horizon, and how strongly extreme daily flows cluster within flood waves.
</p>

These fits use [fixed shrinkage 0.37 and adaptive R](index.md), with unchanged
stations and analysis windows.

## Texas — Trinity River at Romayor

**Data.** USGS site `08066500`, using the longest gap-free daily-discharge suffix through
2025-12-31: 37,135 observations from 1924-05-01. The severity and persistence branches both use
the calendar-day clock.

The separate GEV scale check groups observations by January–December calendar year and retains
only years meeting the predeclared 97% finite-daily-coverage gate. The selected Texas and
Florida windows contribute 101 and 41 maxima, respectively; all retained years happen to have
100% coverage.

**Estimands.** EVI `ξ` describes fitted tail growth. EI `θ` describes extremal clustering and is
reported separately; it is not inserted into the design-life formula.

<div class="unibm-stat-grid">
  <div class="unibm-stat"><strong>0.644</strong><span>EVI ξ · 95% CI [0.591, 0.697]</span></div>
  <div class="unibm-stat"><strong>0.0488</strong><span>EI θ · 95% CI [0.0447, 0.0532]</span></div>
  <div class="unibm-stat"><strong>232k cfs</strong><span>median 10-year level · CI [193k, 280k]</span></div>
</div>

<figure class="unibm-figure">
  <img src="../../assets/cases/tx_streamflow.png" alt="Four-panel diagnostic for Texas streamflow, showing target stability, block-maximum scaling, extremal-index estimates, and design-life levels.">
  <figcaption>Texas streamflow diagnostics. The fitted EI corresponds to a limiting mean of about 20.5 extreme daily observations per cluster, not an elapsed flood duration.</figcaption>
</figure>

## Florida — Choctawhatchee River near Bruce

**Data.** USGS site `02366500`, using the longest gap-free daily-discharge suffix through
2025-12-31: 15,189 observations from 1984-06-01.

<div class="unibm-stat-grid">
  <div class="unibm-stat"><strong>0.325</strong><span>EVI ξ · 95% CI [0.059, 0.591]</span></div>
  <div class="unibm-stat"><strong>0.0551</strong><span>EI θ · 95% CI [0.0484, 0.0628]</span></div>
  <div class="unibm-stat"><strong>38.5k cfs</strong><span>median 10-year level · CI [9.64k, 154k]</span></div>
</div>

<figure class="unibm-figure">
  <img src="../../assets/cases/fl_streamflow.png" alt="Four-panel diagnostic for Florida streamflow, showing target stability, block-maximum scaling, extremal-index estimates, and design-life levels.">
  <figcaption>Florida streamflow diagnostics. This snapshot has wider conditional severity uncertainty than Texas; the comparison does not isolate record length as the cause.</figcaption>
</figure>

## Interpretation limits

These are conditional stationary extrapolations for the selected gauges. The design-life
levels are daily-clock horizon-maximum quantiles, not calendar-year GEV return levels and not
replacements for Bulletin 17C or a GEV flood-frequency analysis. Regime shifts, regulation,
land-use change, and future climate change are outside the fitted information set.

- Preparation: [`scripts/data_prep/usgs.py`](https://github.com/TY-Cheng/UniBM/blob/main/scripts/data_prep/usgs.py)
- Selection disclosure: [`data/metadata/application/usgs_frozen_sites.json`](https://github.com/TY-Cheng/UniBM/blob/main/data/metadata/application/usgs_frozen_sites.json)
- Provider: [USGS Water Services](https://waterservices.usgs.gov/nwis/dv/)
