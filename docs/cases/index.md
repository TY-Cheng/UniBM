# Case studies

<p class="unibm-case-intro">
Six case studies illustrate severity scaling and extremal clustering in environmental
records. The figures and numerical summaries use the repository's archived provider inputs.
</p>

The paper's four main cases are Texas and Florida streamflow and NFIP claims.
The two GHCN cases are broader severity-only illustrations.

**Analysis settings.** These results use fixed shrinkage `0.37`, the declared
window-selection rules, and adaptive R at `128/256/512/768/1024`. Shrinkage is a
pre-specified computational setting, not an optimum selected from sensitivity
results.

| Case | EVI R | BB-sliding EI R | Northrop-sliding EI R |
|---|---:|---:|---:|
| Texas streamflow | 256 | 256 | 256 |
| Florida streamflow | 256 | 256 | 512 |
| Texas NFIP | 128 | 256 | 256 |
| Florida NFIP | 512 | 256 | 256 |
| Houston precipitation | 256 | — | — |
| Phoenix hot–dry severity | 128 | — | — |

All listed fits met the 10%-of-statistical-SE MCSE criterion for the parameter
estimate and its interval endpoints. This controls finite-bootstrap numerical
precision conditional on the selected window; it does **not** establish 95%
empirical coverage or certify the numerical precision of extrapolated
design-life levels. Explicit fixed integer R remains supported by the API.

<div class="unibm-domain-grid">
  <a class="unibm-domain-card unibm-domain-climate" href="climate-extremes/">
    <span class="unibm-domain-index">GHCN · 2 cases</span>
    <h3>Climate extremes</h3>
    <p>June–November precipitation in Houston and April–October hot–dry severity in Phoenix.</p>
    <span class="unibm-domain-meta">Severity branch · no formal EI</span>
  </a>
  <a class="unibm-domain-card unibm-domain-streamflow" href="streamflow/">
    <span class="unibm-domain-index">USGS · 2 cases</span>
    <h3>Streamflow</h3>
    <p>Long daily-discharge records at selected Texas and Florida streamgages.</p>
    <span class="unibm-domain-meta">Severity + persistence</span>
  </a>
  <a class="unibm-domain-card unibm-domain-nfip" href="nfip-claims/">
    <span class="unibm-domain-index">OpenFEMA · 2 cases</span>
    <h3>NFIP claims</h3>
    <p>Daily building-claim totals, grouped by date of loss, on active-day and calendar-day clocks.</p>
    <span class="unibm-domain-meta">Severity + persistence</span>
  </a>
</div>

## Reading the evidence

| Quantity | What it describes | What it does not establish |
|---|---|---|
| EVI `ξ` | Slope of log block-maximum quantiles against log block size | A causal mechanism or event forecast |
| EI `θ` | Dimensionless measure of clustering among upper-tail exceedances | Elapsed flood duration or the number of all active days |
| Design-life level `D_tau(T)` | The non-exceedance quantile `tau` of the maximum over design life `T`, on the specified clock | An annual-maxima return-level label |

The reported headline design-life levels use `tau=0.5`: they estimate the median
horizon maximum. Their 95% confidence intervals describe uncertainty in that
estimated median, not the range of future maxima. Parameter intervals and
design-life intervals are conditional on the selected window, fixed
regularization, and stationary scaling model. NFIP's calendar-year conversion
also assumes the historical active-day rate remains applicable.

## Results across the four main applications

The fitted EVI values are 0.33–0.64 for streamflow and 1.38–1.40 for NFIP
building-claim totals on active days. Calendar-time EI estimates are about
0.05 and 0.31, respectively. Streamflow therefore exhibits stronger extremal
clustering in these records, while NFIP severity fits imply faster growth of
maximum daily building-claim totals on the active-day clock as the design life
increases.

The streamflow records have similar EI estimates despite different EVI
estimates: marginal tail behavior and extremal dependence describe different
features of a record. NFIP estimates are close between states, but similar EVI
and EI values can coexist with different loss magnitudes and regional
dependence structures. A statewide claim ledger and an individual streamgage
provide a descriptive comparison, not matched event-level outcomes.

## Window sensitivity and interpretation

Refitting over the three best-ranked admissible windows gives the following
min–max ranges under the fixed selection rules:

| Application | EVI range | BB-sliding-FGLS EI range |
|---|---:|---:|
| Texas streamflow | [0.59, 0.64] | [0.0488, 0.0495] |
| Florida streamflow | [0.32, 0.33] | [0.0551, 0.0569] |
| Texas NFIP claims | [1.31, 1.49] | [0.3117, 0.3129] |
| Florida NFIP claims | [1.37, 1.43] | [0.3087, 0.3092] |

EI estimates vary little across these windows; EVI sensitivity is greatest for
Texas NFIP. These are window-sensitivity ranges, not post-selection confidence
intervals. All four 50-year design-life levels extrapolate beyond the fitted
block-size range, as detailed on the streamflow and NFIP pages.

The applications use a stationary working model for each retained record.
Block-size diagnostics assess scaling and EI-path stability, not temporal
stationarity. Clear trends or regime shifts require separate treatment before
future-design extrapolation; the reported intervals do not include uncertainty
from temporal change or data-driven period selection.

UniBM combines information across block sizes within one record. It can
supplement standard flood-frequency analysis with severity and clustering
summaries. Conventional annual-peak design and regional or ungauged-site
estimation should follow established methods. Quantitative comparisons require
aligned periods, observation scales, and probability definitions.

The shared analysis cutoff is **2025-12-31**. See [Validation](../validation.md)
for synthetic benchmark evidence, with its separate data and scope.
