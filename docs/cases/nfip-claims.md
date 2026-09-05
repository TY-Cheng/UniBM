# NFIP claims

<p class="unibm-case-intro">
The NFIP cases move from physical hazard to insured impact. Severity is fitted on positive
claim-active days; persistence is fitted on the calendar-day process. A zero means no recorded
building payout in the NFIP event ledger, not an imputed sensor observation. Keeping those
clocks separate is part of the estimand, not a display choice.
</p>

These fits use [fixed shrinkage 0.37 and adaptive R](index.md), with unchanged
claim extracts, analysis windows, and CPI inputs.

## Texas building payouts

**Data.** OpenFEMA NFIP building-claim payouts over the fixed 1978-01-01 through 2025-12-31
acquisition window, aggregated daily and adjusted by loss-month CPI-U to 2025 dollars. The
display and EI series contain 17,532 calendar days; the EVI series contains 5,807 positive
claim-active days, or 120.979 per calendar year over the 48-year window.

For each claim, the payout multiplier is the **2025 annual-average CPI-U divided
by the CPI-U for its loss month**, using the not-seasonally-adjusted series.
Thus all months are expressed in the same 2025-average dollar units; this is not
a daily inflation adjustment or a December-2025 price base. The tracked CPI
input explicitly flags October 2025 as a geometric interpolation between
September and November. See the [CPI input](https://github.com/TY-Cheng/UniBM/blob/main/data/raw/cpi/cpi_u_monthly.csv)
and preparation code below for that data treatment.

<div class="unibm-stat-grid">
  <div class="unibm-stat"><strong>1.395</strong><span>EVI ξ · 95% CI [0.719, 2.071]</span></div>
  <div class="unibm-stat"><strong>0.312</strong><span>EI θ · 95% CI [0.278, 0.349]</span></div>
  <div class="unibm-stat"><strong>$478m</strong><span>median 10-year active-day level · CI [$24.1m, $9.47b]</span></div>
</div>

<figure class="unibm-figure">
  <img src="../../assets/cases/tx_nfip_claims.png" alt="Four-panel diagnostic for Texas NFIP building payouts, showing target stability, active-day block-maximum scaling, calendar-day extremal-index estimates, and design-life levels.">
  <figcaption>Texas NFIP diagnostics. The severity and persistence branches deliberately retain different observation clocks.</figcaption>
</figure>

## Florida building payouts

**Data.** The corresponding Florida series uses the same fixed 1978-01-01 through 2025-12-31
acquisition window. It contains 17,532 calendar days and 5,440 positive claim-active days, or
113.333 per calendar year.

<div class="unibm-stat-grid">
  <div class="unibm-stat"><strong>1.383</strong><span>EVI ξ · 95% CI [1.051, 1.716]</span></div>
  <div class="unibm-stat"><strong>0.309</strong><span>EI θ · 95% CI [0.271, 0.352]</span></div>
  <div class="unibm-stat"><strong>$224m</strong><span>median 10-year active-day level · CI [$47.3m, $1.06b]</span></div>
</div>

<figure class="unibm-figure">
  <img src="../../assets/cases/fl_nfip_claims.png" alt="Four-panel diagnostic for Florida NFIP building payouts, showing target stability, active-day block-maximum scaling, calendar-day extremal-index estimates, and design-life levels.">
  <figcaption>Florida NFIP diagnostics. The fitted EI gives a limiting mean-cluster-size interpretation of about 3.23 extreme daily observations per cluster, not 3.23 elapsed or consecutive days.</figcaption>
</figure>

## Interpretation limits

The fitted EVI values above one indicate extremely heavy active-day payout tails. The dollar
design-life levels are therefore descriptive stress measures under the fitted stationary
scaling relation—not expected annual loss, reserves, prices, portfolio loss, or operational
risk guidance. The records reflect program exposure, reporting, policy, inflation adjustment,
and claim processes as well as physical flooding.

The statewide NFIP ledger and an individual streamgage are complementary case
studies, not matched event-level outcomes. Their common calendar-year convention
does not make streamflow a causal explanation of the NFIP payout series.

- Preparation: [`scripts/data_prep/fema.py`](https://github.com/TY-Cheng/UniBM/blob/main/scripts/data_prep/fema.py)
- CPI treatment: [`scripts/data_prep/cpi.py`](https://github.com/TY-Cheng/UniBM/blob/main/scripts/data_prep/cpi.py)
- Provider: [OpenFEMA NFIP claims API](https://www.fema.gov/api/open/v2/FimaNfipClaims)
