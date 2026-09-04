# Streamflow

<p class="unibm-case-intro">
The paired streamflow cases separate two questions: how daily-discharge severity grows over a
planning horizon, and how strongly extreme daily flows cluster within flood waves.
</p>

## Texas — Trinity River at Romayor

**Data.** USGS site `08066500`, using the longest gap-free daily-discharge suffix through
2025-12-31: 37,135 observations from 1924-05-01. The severity and persistence branches both use
the calendar-day clock.

**Estimands.** EVI `ξ` describes fitted tail growth. EI `θ` describes extremal clustering and is
reported separately; it is not inserted into the design-life formula.

<div class="unibm-stat-grid">
  <div class="unibm-stat"><strong>0.647</strong><span>EVI ξ · 95% CI [0.591, 0.702]</span></div>
  <div class="unibm-stat"><strong>0.0488</strong><span>EI θ · 95% CI [0.0444, 0.0536]</span></div>
  <div class="unibm-stat"><strong>234k cfs</strong><span>median 10-year level · CI [193k, 284k]</span></div>
</div>

<figure class="unibm-figure">
  <img src="../../assets/cases/tx_streamflow.png" alt="Four-panel diagnostic for Texas streamflow, showing target stability, block-maximum scaling, extremal-index estimates, and design-life levels.">
  <figcaption>Texas streamflow diagnostics. The fitted EI corresponds to a mean-cluster-size interpretation of about 20.5 daily observations.</figcaption>
</figure>

## Florida — Choctawhatchee River near Bruce

**Data.** USGS site `02366500`, using the longest gap-free daily-discharge suffix through
2025-12-31: 15,189 observations from 1984-06-01.

<div class="unibm-stat-grid">
  <div class="unibm-stat"><strong>0.325</strong><span>EVI ξ · 95% CI [0.048, 0.603]</span></div>
  <div class="unibm-stat"><strong>0.0552</strong><span>EI θ · 95% CI [0.0486, 0.0625]</span></div>
  <div class="unibm-stat"><strong>38.5k cfs</strong><span>median 10-year level · CI [9.1k, 163k]</span></div>
</div>

<figure class="unibm-figure">
  <img src="../../assets/cases/fl_streamflow.png" alt="Four-panel diagnostic for Florida streamflow, showing target stability, block-maximum scaling, extremal-index estimates, and design-life levels.">
  <figcaption>Florida streamflow diagnostics. Its shorter record produces substantially wider conditional severity uncertainty than the Texas case.</figcaption>
</figure>

## Interpretation limits

These are conditional stationary extrapolations for the selected gauges. The design-life
levels are daily-clock horizon-maximum quantiles, not water-year return levels and not
replacements for Bulletin 17C or a GEV flood-frequency analysis. Regime shifts, regulation,
land-use change, and future climate change are outside the fitted information set.

- Preparation: [`scripts/data_prep/usgs.py`](https://github.com/TY-Cheng/UniBM/blob/main/scripts/data_prep/usgs.py)
- Selection disclosure: [`data/metadata/application/usgs_frozen_sites.json`](https://github.com/TY-Cheng/UniBM/blob/main/data/metadata/application/usgs_frozen_sites.json)
- Provider: [USGS Water Services](https://waterservices.usgs.gov/nwis/dv/)
