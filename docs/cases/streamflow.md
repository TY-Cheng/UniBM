# Streamflow

<p class="unibm-case-intro">
The paired streamflow cases describe how the maximum daily mean discharge scales with the
design life and how strongly extreme daily flows cluster in time.
</p>

These fits use [fixed shrinkage 0.37 and adaptive R](index.md), with unchanged
stations and analysis windows.

## Texas — Trinity River at Romayor

**Data.** USGS site `08066500`: 37,135 daily mean discharge observations from
1924-05-01 to 2025-12-31, spanning 101.7 years. After removing repeated dates
and invalid discharge values, the analysis retains the longest gap-free daily
suffix ending at the fixed cutoff. Missing days are not imputed or compressed.
Both branches use this continuous calendar-day record.

The separate GEV scale check groups observations by January–December calendar year and retains
only years meeting the predeclared 97% finite-daily-coverage gate. The selected Texas and
Florida windows contribute 101 maxima from 1925–2025 and 41 from 1985–2025,
respectively. All retained years have complete daily coverage; the incomplete
initial years, 1924 and 1984, are excluded.

**Estimands.** EVI `ξ` describes the fitted severity-scaling slope. EI `θ` is
dimensionless and describes extremal clustering. It is reported separately and
is not inserted into the design-life formula. Discharge levels are in cubic
feet per second (cfs).

<div class="unibm-stat-grid">
  <div class="unibm-stat"><strong>0.644</strong><span>EVI ξ · 95% CI [0.591, 0.697]</span></div>
  <div class="unibm-stat"><strong>0.0488</strong><span>EI θ · 95% CI [0.0447, 0.0532]</span></div>
  <div class="unibm-stat"><strong>232k cfs</strong><span>median 10-year level · CI [193k, 280k]</span></div>
</div>

<figure class="unibm-figure">
  <img src="../../assets/cases/tx_streamflow.png" alt="Four-panel diagnostic for Texas streamflow, showing target stability, block-maximum scaling, extremal-index estimates, and design-life levels.">
  <figcaption>Texas streamflow diagnostics. The fitted EI implies a limiting mean of about 20.5 upper-tail exceedances of daily discharge per cluster; this count is not an elapsed flood duration.</figcaption>
</figure>

## Florida — Choctawhatchee River near Bruce

**Data.** USGS site `02366500`: 15,189 daily mean discharge observations from
1984-06-01 to 2025-12-31, spanning 41.6 years. Although the archive begins on
1930-10-01, this is the longest gap-free suffix ending at the cutoff. Both
branches use the retained continuous calendar-day record.

<div class="unibm-stat-grid">
  <div class="unibm-stat"><strong>0.325</strong><span>EVI ξ · 95% CI [0.059, 0.591]</span></div>
  <div class="unibm-stat"><strong>0.0551</strong><span>EI θ · 95% CI [0.0484, 0.0628]</span></div>
  <div class="unibm-stat"><strong>38.5k cfs</strong><span>median 10-year level · CI [9.64k, 154k]</span></div>
</div>

<figure class="unibm-figure">
  <img src="../../assets/cases/fl_streamflow.png" alt="Four-panel diagnostic for Florida streamflow, showing target stability, block-maximum scaling, extremal-index estimates, and design-life levels.">
  <figcaption>Florida streamflow diagnostics. The fitted EI implies a limiting mean of about 18.1 upper-tail exceedances of daily discharge per cluster. Its proximity to the Texas EI contrasts with the different severity-scaling slopes.</figcaption>
</figure>

## Severity scaling and the GEV comparison

The selected EVI windows span 14–183 calendar days in Texas and 17–26 in
Florida. Their slopes, 0.644 and 0.325, imply faster growth of horizon-maximum
discharge in Texas under the fitted scaling model. Both 50-year targets use
`b_50=18,263` calendar days, about 99.8 and 702 times the respective largest
fitted block size. These ratios compare block sizes, not historical record
lengths.

L-moment GEV fits to calendar-year maxima provide a separate hydrologic scale
check. Discharge levels below are in **thousands of cfs**:

| Site | GEV shape | GEV 50-year return level | UniBM median 50-year design-life level [95% CI] |
|---|---:|---:|---:|
| Texas | −0.03 | 116 | 655 [503, 854] |
| Florida | 0.29 | 133 | 65.0 [10.6, 398] |

The GEV return level is an upper quantile of annual maxima; the UniBM level is
the median maximum over 50 years. These are different probability targets, so
the table is not a comparison of predictive accuracy. Under stationary
regular variation and a positive EI, the GEV shape and limiting block-quantile
slope nevertheless target the same tail index. The Texas discrepancy calls for
assessment of tail-index stability across block sizes and estimation methods.
Its slope is interpreted over the selected finite-block range, and its 50-year
level is illustrative.

## Interpretation and use

The two sites have similar EI estimates but different EVI estimates. Clustering
of upper-tail exceedances and growth of maximum discharge are complementary
features; EI does not measure elapsed or consecutive-day flood duration.

The confidence intervals quantify estimation uncertainty in the median
design-life levels under the stationary working model. The fits do not
explicitly model seasonal or long-term changes, regulation, or land-use
effects. Standard flood-frequency estimates can be supplemented by these
severity and clustering summaries when periods, discharge definitions, and
probability targets are aligned.

- Preparation: [`scripts/data_prep/usgs.py`](https://github.com/TY-Cheng/UniBM/blob/main/scripts/data_prep/usgs.py)
- Selection disclosure: [`data/metadata/application/usgs_frozen_sites.json`](https://github.com/TY-Cheng/UniBM/blob/main/data/metadata/application/usgs_frozen_sites.json)
- Provider: [USGS Water Services](https://waterservices.usgs.gov/nwis/dv/)
