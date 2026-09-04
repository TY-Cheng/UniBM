# NFIP claims

<p class="unibm-case-intro">
The NFIP cases move from physical hazard to insured impact. Severity is fitted on positive
claim-active days; persistence is fitted on the zero-filled calendar-day process. Keeping those
clocks separate is part of the estimand, not a display choice.
</p>

## Texas building payouts

**Data.** OpenFEMA NFIP building-claim payouts from 1978-01-09 through 2025-12-31, aggregated
daily and adjusted by loss-month CPI-U to 2025 dollars. The display and EI series contain 17,524
calendar days; the EVI series contains 5,807 positive claim-active days.

<div class="unibm-stat-grid">
  <div class="unibm-stat"><strong>1.395</strong><span>EVI ξ · 95% CI [0.735, 2.055]</span></div>
  <div class="unibm-stat"><strong>0.312</strong><span>EI θ · 95% CI [0.277, 0.351]</span></div>
  <div class="unibm-stat"><strong>$477m</strong><span>median 10-year active-day level · CI [$25.8m, $8.84b]</span></div>
</div>

<figure class="unibm-figure">
  <img src="../../assets/cases/tx_nfip_claims.png" alt="Four-panel diagnostic for Texas NFIP building payouts, showing target stability, active-day block-maximum scaling, calendar-day extremal-index estimates, and design-life levels.">
  <figcaption>Texas NFIP diagnostics. The severity and persistence branches deliberately retain different observation clocks.</figcaption>
</figure>

## Florida building payouts

**Data.** The corresponding Florida series runs from 1978-01-08 through 2025-12-31. It contains
17,525 calendar days and 5,440 positive claim-active days.

<div class="unibm-stat-grid">
  <div class="unibm-stat"><strong>1.385</strong><span>EVI ξ · 95% CI [1.058, 1.712]</span></div>
  <div class="unibm-stat"><strong>0.309</strong><span>EI θ · 95% CI [0.270, 0.353]</span></div>
  <div class="unibm-stat"><strong>$226m</strong><span>median 10-year active-day level · CI [$49.4m, $1.03b]</span></div>
</div>

<figure class="unibm-figure">
  <img src="../../assets/cases/fl_nfip_claims.png" alt="Four-panel diagnostic for Florida NFIP building payouts, showing target stability, active-day block-maximum scaling, calendar-day extremal-index estimates, and design-life levels.">
  <figcaption>Florida NFIP diagnostics. The fitted EI gives a mean-cluster-size interpretation of about 3.24 calendar days.</figcaption>
</figure>

## Interpretation limits

The fitted EVI values above one indicate extremely heavy active-day payout tails. The dollar
design-life levels are therefore descriptive stress measures under the fitted stationary
scaling relation—not expected annual loss, reserves, prices, portfolio loss, or operational
risk guidance. The records reflect program exposure, reporting, policy, inflation adjustment,
and claim processes as well as physical flooding.

- Preparation: [`scripts/data_prep/fema.py`](https://github.com/TY-Cheng/UniBM/blob/main/scripts/data_prep/fema.py)
- CPI treatment: [`scripts/data_prep/cpi.py`](https://github.com/TY-Cheng/UniBM/blob/main/scripts/data_prep/cpi.py)
- Provider: [OpenFEMA NFIP claims API](https://www.fema.gov/api/open/v2/FimaNfipClaims)
