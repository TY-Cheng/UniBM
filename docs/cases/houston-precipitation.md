# Houston precipitation

This case estimates severity and extremal clustering in daily precipitation
relative to its recent level. It uses **EWMA-normalized, full-calendar observations**,
including dry-day zeros.

## Record and normalization

The source is NOAA GHCN-Daily station `USW00012918`, Houston William P. Hobby
Airport. Daily PRCP is converted from tenths of a millimetre to millimetres;
missing values and nonblank quality flags are excluded. The full daily calendar
is retained while identifying the longest uninterrupted valid run, with the
latest run selected on a length tie. No months are concatenated across seasons.

After 30 warmup days, the fitted record is **1990-03-03–2022-05-30**, with
**11,777 consecutive days**. The archive cutoff remains 2025-12-31; later gaps
prevent the chosen continuous run from extending to that cutoff.
This sample differs from the former June–November illustration.

```text
X_t = daily precipitation in mm, including genuine zeros
Y_t = X_t / m_t
m_t = ρ m_(t−1) + (1−ρ) X_(t−1),  ρ = 2^(−1/180)
```

The initial denominator is the mean of the preceding 30 days. `Y_t` is a
**dimensionless precipitation-to-past-level ratio**, not rainfall in millimetres
or a standardized anomaly. Only earlier observations enter the denominator.
See the [shared normalization definitions](index.md#what-normalized-means).

## EVI, EI, and design-life levels

| Quantity | Estimate | Conditional 95% CI |
|---|---:|---|
| EVI ξ | 0.520 | [0.380, 0.660] |
| BB-sliding-FGLS EI θ | 0.844 | [0.793, 0.898] |
| Northrop-sliding-FGLS EI θ | 0.842 | [0.790, 0.897] |

<figure class="unibm-figure">
  <a href="../../assets/cases/houston_precipitation.png"><img src="../../assets/cases/houston_precipitation.png" alt="Houston normalized precipitation: summary stability, EVI quantile scaling, EI comparison, and relative design-life levels."></a>
  <figcaption>All four panels use the same continuous normalized record. The EI panel retains dry-day zeros. Click the figure for full size.</figcaption>
</figure>

[Download numerical results and preparation settings (JSON)](../assets/cases/houston_precipitation.json).

The selected EVI window is **23–39 days**. The EI describes clustering of large
relative rainfall observations; it is not the persistence of all wet days.
Design-life curves use 365.25 observations per year and describe the maximum
**normalized daily rainfall** over the horizon, rather than accumulated rainfall.

The nominal CIs use the common FGLS/Wald settings in the [case overview](index.md).
They do not include uncertainty from estimating the EWMA level or selecting the
record and fit window. EWMA does not establish removal of rainfall seasonality
or long-term change. A future millimetre-scale level would additionally require
a model for the future denominator.

## Reproduce

```bash
PYTHONPATH=scripts uv run python -m application.docs_cases --keys houston_hobby_precipitation
```

Preparation uses [the tracked normalization code](https://github.com/TY-Cheng/UniBM/blob/main/scripts/application/normalization.py)
and the archived [GHCN station input](https://github.com/TY-Cheng/UniBM/blob/main/data/raw/ghcn/USW00012918.csv.gz).
