# Case studies

Nine records across six pages illustrate EVI, EI, and design-life estimation on
explicit observation scales. Streamflow and NFIP retain their physical or
monetary scales; the other cases use the normalization defined below.
GOES is an EVI point-estimation example. The other eight records include EI.

<div class="unibm-domain-grid">
  <a class="unibm-domain-card unibm-domain-streamflow" href="streamflow/">
    <span class="unibm-domain-index">01</span>
    <h3>Streamflow</h3>
    <p>Texas and Florida daily mean discharge.</p>
    <span class="unibm-domain-meta">USGS · Raw · EVI + EI</span>
  </a>
  <a class="unibm-domain-card unibm-domain-nfip" href="nfip-claims/">
    <span class="unibm-domain-index">02</span>
    <h3>NFIP claims</h3>
    <p>Texas and Florida daily building-claim totals in 2025 dollars.</p>
    <span class="unibm-domain-meta">OpenFEMA · CPI-adjusted · EVI + EI</span>
  </a>
  <a class="unibm-domain-card unibm-domain-climate" href="houston-precipitation/">
    <span class="unibm-domain-index">03</span>
    <h3>Houston precipitation</h3>
    <p>Full-calendar daily rainfall divided by its past EWMA level.</p>
    <span class="unibm-domain-meta">GHCN · Normalized · EVI + EI</span>
  </a>
  <a class="unibm-domain-card unibm-domain-climate" href="phoenix-hot-dry/">
    <span class="unibm-domain-index">04</span>
    <h3>Phoenix hot–dry severity</h3>
    <p>A derived daily severity index divided by its past EWMA level.</p>
    <span class="unibm-domain-meta">GHCN · Normalized · EVI + EI</span>
  </a>
  <a class="unibm-domain-card unibm-domain-streamflow" href="goes-xray/">
    <span class="unibm-domain-index">05</span>
    <h3>GOES soft X-rays</h3>
    <p>Hourly maxima divided by their past EWMA level, with gaps retained.</p>
    <span class="unibm-domain-meta">NOAA · Normalized · EVI only</span>
  </a>
  <a class="unibm-domain-card unibm-domain-nfip" href="spy-qqq/">
    <span class="unibm-domain-index">06</span>
    <h3>SPY / QQQ losses</h3>
    <p>Left-tail log losses divided by past EWMA return volatility.</p>
    <span class="unibm-domain-meta">Massive · Normalized · EVI + EI</span>
  </a>
</div>

## What “normalized” means

Normalization divides an observation by a positive scale estimated from earlier
observations. It does not subtract a mean, produce z-scores, or establish stationarity.
The resulting EVI and EI describe the **transformed series**, not automatically
the raw process. A time-varying denominator can change both tail behavior and clustering.

| Cases | Numerator | Denominator | Initialization and update | Clock |
|---|---|---|---|---|
| Streamflow | Daily mean discharge | None | No EWMA or warmup | Calendar days |
| NFIP claims | Daily building-claim totals in 2025 USD | None | CPI adjustment retained; no EWMA | Active days for EVI; calendar days for EI |
| Houston | Daily precipitation, including zeros | Past EWMA precipitation level | First 30 days' mean; 180-day half-life | Calendar days |
| Phoenix | Derived hot–dry severity, including zeros | Past EWMA severity level | First 30 days' mean; 180-day half-life | Calendar days |
| GOES | Qualified hourly maximum of minute-mean irradiance | Past EWMA irradiance level | First 720 hours' mean; 4,320-hour half-life; restart after each gap | UTC hours |
| SPY / QQQ | `max(−r_t, 0)` from split- and dividend-adjusted log returns | Past EWMA root mean square of signed returns | First 252 returns' mean square; decay 0.94 | Trading sessions |

For Houston, Phoenix, and GOES, the level-normalized series is:

```text
Y_t = X_t / m_t
m_w = mean(X_0, ..., X_(w−1))
m_t = ρ m_(t−1) + (1−ρ) X_(t−1),  t > w
ρ = 2^(−1/h)
```

Here `w` is the warmup length and `h` is the half-life, both in observations.
The first analyzed observation is `t=w`; its denominator uses only the warmup.
For daily cases, `w=30, h=180`; for GOES, `w=720, h=4320` in each uninterrupted run.
The zero-based convention makes explicit that the current observation does not
enter its own denominator. No epsilon floor, interpolation, or zero deletion is applied.

SPY/QQQ use a different denominator:

```text
Y_t = max(−r_t, 0) / sqrt(v_t)
v_252 = mean(r_0², ..., r_251²)
v_t = 0.94 v_(t−1) + 0.06 r_(t−1)²,  t > 252
```

The scale uses **all signed returns**, including gains; it is a zero-mean EWMA
volatility estimate. Gains remain as zero observations in `Y_t`. These are neither
loss-only samples nor cumulative portfolio losses.

The EWMA steps are past-only. Phoenix's preceding climatological standardization
uses the full retained period, and the provider archives are retrospective data
snapshots. Neither feature should be described as point-in-time forecasting data.

## Inference and units

The eight continuous-record cases use median-sliding FGLS for EVI and
BB/Northrop-sliding FGLS for EI, with fixed shrinkage 0.73 and 0.37 respectively.
K-gaps and Ferro–Segers provide EI comparisons. The random seed is 7, with
adaptive bootstrap stages 128/256/512/768/1024 and model-based Wald intervals.
There is no additional CI scale calibration.

The shared grid starts at `max(5, ceil(N**(1/3)))`. Its upper bound is
`min(floor(N**(1−1/e)), floor(N/17))` for EVI and
`min(floor(sqrt(N)), floor(N/17))` for EI. Selection uses the full admissible
grid without edge trimming. EVI uses `L=max(2B, floor(sqrt(N)))`;
see [Concepts](../concepts.md). GOES instead uses OLS point estimates with
complete windows on the uncompressed hourly calendar; it has no EI or reported CI.

A reported 95% CI is conditional on the preprocessing, selected window,
regularization, and stationary working model. EWMA is not re-estimated in each
bootstrap draw. These intervals omit preprocessing and selection uncertainty;
bootstrap numerical precision does not establish empirical coverage.
See the separate [Benchmark](../benchmark.md) for coverage results with known truth.

Design-life levels describe quantiles of a **maximum observation** over the
stated horizon. The median curve uses `tau=0.5`; its CI is not a prediction
interval for future maxima. Higher-quantile curves reuse the median fit's slope
with quantile-specific intercepts. Normalized levels are dimensionless relative
scales: they cannot be converted to future millimetres, irradiance, or loss
percentages without a model for future denominators.

## Data and reproduction

The archive cutoff is 2025-12-31, but actual fitted dates vary after quality
screening and warmup; Houston's longest complete record ends in 2022. No common
end date or common sample size is implied. Provider, observation clock, preparation,
fit dates, and numerical results are stated on each page and in its JSON record.

From the repository root, rebuild only the documented figures and summaries:

```bash
PYTHONPATH=scripts uv run python -m application.docs_cases
uv run mkdocs build --strict
```

This command uses existing local inputs and does not export to an external report
directory. `--keys` selects case IDs; `--available` retains frozen optional
GOES/finance assets when their local input pair is absent, and reports each skip.
Malformed or hash-mismatched inputs still fail. The ordinary `just application`
workflow includes available extra cases in its combined CSVs, prepared-series
registry, and `out/applications/report.html`. The figures and JSON under
`out/applications/cases/` are copied identically to `docs/assets/cases/` from the
same fits. The local report lists only cases rebuilt in that run, with skipped
optional inputs identified. The four-case streamflow/NFIP report export retains
its configured destination.

GHCN, USGS, and NFIP use the repository's archived inputs. GOES and finance
preparation commands and local-input requirements are on their respective pages.
Building the documentation alone uses tracked static figures and requires no
provider account, download, or model fit.
