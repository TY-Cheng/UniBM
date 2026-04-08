# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # UniBM manuscript workflow
#
# This vignette demonstrates the revised `UniBM` workflow for **sliding-block, quantile-first inference** under the environmental trilemma of scarcity, dependence, and slow penultimate convergence.
#
# The notebook mirrors the revised manuscript:
# 1. synthetic benchmark evidence;
# 2. dataset screening;
# 3. Texas/Florida streamflow as the hydrologic-response applications;
# 4. Texas/Florida NFIP building payouts as the hazard-to-impact applications;
# 5. Houston precipitation and Phoenix hot-dry severity as secondary EVI-only weather cases.
#

# %% [markdown]
# ## Setup
#
# The first code cell is intentionally lightweight: it only resolves paths and imports the plotting/statistical helpers used below.
# The second code cell is an explicit rebuild switch for the derived series, benchmark summaries, and manuscript figures.
#

# %%
# ruff: noqa: E402
from pathlib import Path
import importlib.util
import json


def _load_import_bootstrap(start: Path):
    for _candidate in (start, *start.parents):
        _helper_path = _candidate / "scripts" / "shared" / "import_bootstrap.py"
        if _helper_path.exists():
            _spec = importlib.util.spec_from_file_location(
                "_shared_import_bootstrap",
                _helper_path,
            )
            if _spec is None or _spec.loader is None:
                raise ImportError(f"Could not load import bootstrap helper from {_helper_path}")
            _module = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_module)
            return _module
    raise FileNotFoundError(
        "Could not locate scripts/shared/import_bootstrap.py from the current notebook session."
    )


_bootstrap = _load_import_bootstrap(Path.cwd().resolve())
_scripts_dir = _bootstrap.bootstrap_notebook_scripts_dir(Path.cwd().resolve())


def _load_notebook_api(scripts_dir: Path):
    api_path = scripts_dir / "notebook_api" / "api.py"
    spec = importlib.util.spec_from_file_location("_unibm_notebook_api", api_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load notebook API helper from {api_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


from unibm._runtime import prepare_matplotlib_env

prepare_matplotlib_env("unibm-notebook")
import matplotlib.pyplot as plt
import pandas as pd
from config import resolve_repo_dirs
from IPython.display import Markdown, display
from data_prep.fema import OPENFEMA_NFIP_CLAIMS_ENDPOINT
from data_prep.usgs import USGS_DV_ENDPOINT

_vignette_api = _load_notebook_api(_scripts_dir)
CORE_METHODS = _vignette_api.CORE_METHODS
UNIVERSAL_BENCHMARK_SET = _vignette_api.UNIVERSAL_BENCHMARK_SET
benchmark_story_latex = _vignette_api.benchmark_story_latex
benchmark_story_table = _vignette_api.benchmark_story_table
benchmark_table = _vignette_api.benchmark_table
build_application_bundles = _vignette_api.build_application_bundles
build_application_outputs = _vignette_api.build_application_outputs
build_ei_benchmark_manuscript_outputs = _vignette_api.build_ei_benchmark_manuscript_outputs
build_evi_benchmark_manuscript_outputs = _vignette_api.build_evi_benchmark_manuscript_outputs
ei_core_story_table = _vignette_api.ei_core_story_table
ei_interval_story_table = _vignette_api.ei_interval_story_table
ei_story_latex = _vignette_api.ei_story_latex
ei_targets_story_table = _vignette_api.ei_targets_story_table
interval_sharpness_story_latex = _vignette_api.interval_sharpness_story_latex
interval_sharpness_story_table = _vignette_api.interval_sharpness_story_table
plot_application_composite = _vignette_api.plot_application_composite
plot_application_overview = _vignette_api.plot_application_overview
plot_application_time_series = _vignette_api.plot_application_time_series
plot_benchmark_panels = _vignette_api.plot_benchmark_panels
plot_ei_core_panels = _vignette_api.plot_ei_core_panels
plot_ei_interval_sharpness_scatter = _vignette_api.plot_ei_interval_sharpness_scatter
plot_ei_targets_panels = _vignette_api.plot_ei_targets_panels
plot_interval_sharpness_scatter = _vignette_api.plot_interval_sharpness_scatter
plot_target_plus_external_panels = _vignette_api.plot_target_plus_external_panels
target_plus_external_story_latex = _vignette_api.target_plus_external_story_latex
target_plus_external_story_table = _vignette_api.target_plus_external_story_table

DIRS = resolve_repo_dirs()
ROOT = DIRS["DIR_WORK"]
OUT = DIRS["DIR_OUT_APPLICATIONS"]
BENCH = DIRS["DIR_OUT_BENCHMARK"]
FIG_DIR = DIRS["DIR_MANUSCRIPT_FIGURE"]
TABLE_DIR = DIRS["DIR_MANUSCRIPT_TABLE"]
BENCHMARK_SET = UNIVERSAL_BENCHMARK_SET


# %%
REBUILD_OUTPUTS = False
REQUIRED_OUTPUTS = [
    OUT / "application_series_registry.csv",
    OUT / "application_screening.csv",
    OUT / "application_summary.csv",
    OUT / "application_design_life_levels.csv",
    OUT / "application_methods.csv",
    OUT / "application_ei_methods.csv",
    OUT / "application_ei_seasonal_methods.csv",
    OUT / "application_usgs_site_screening.csv",
    TABLE_DIR / "application_summary_main.tex",
    TABLE_DIR / "application_design_life_levels_main.tex",
    TABLE_DIR / "application_ei_main.tex",
    FIG_DIR / "application_composite_houston_precipitation.pdf",
    FIG_DIR / "application_composite_phoenix_hotdry.pdf",
    FIG_DIR / "application_composite_tx_streamflow.pdf",
    FIG_DIR / "application_composite_fl_streamflow.pdf",
    FIG_DIR / "application_composite_tx_nfip_claims.pdf",
    FIG_DIR / "application_composite_fl_nfip_claims.pdf",
    BENCH / "summary.csv",
    BENCH / "external_summary.csv",
    BENCH / "ei_summary.csv",
    BENCH / "ei_external_summary.csv",
    FIG_DIR / "benchmark_overview.pdf",
    FIG_DIR / "benchmark_summary.pdf",
    FIG_DIR / "benchmark_targets.pdf",
    FIG_DIR / "benchmark_interval_sharpness.pdf",
    FIG_DIR / "benchmark_ei_summary.pdf",
    FIG_DIR / "benchmark_ei_targets.pdf",
    FIG_DIR / "benchmark_ei_interval_sharpness.pdf",
]
missing_outputs = [path for path in REQUIRED_OUTPUTS if not path.exists()]

if REBUILD_OUTPUTS:
    print("Rebuilding application outputs...")
    _ = build_application_outputs(ROOT)
    print("Rebuilding benchmark manuscript figures/tables from current benchmark CSVs...")
    _ = build_evi_benchmark_manuscript_outputs(ROOT)
    print("Rebuilding EI benchmark manuscript figures/tables from current benchmark CSVs...")
    _ = build_ei_benchmark_manuscript_outputs(ROOT)
elif missing_outputs:
    print("Missing outputs detected. Set REBUILD_OUTPUTS = True and rerun this cell.")
    for path in missing_outputs:
        print(" -", path)
else:
    print("Using existing derived outputs and benchmark summaries.")


# %%
application_summary = pd.read_csv(OUT / "application_summary.csv")
application_design_life_levels = pd.read_csv(OUT / "application_design_life_levels.csv")
application_ei_methods = pd.read_csv(OUT / "application_ei_methods.csv")
application_ei_seasonal_methods = pd.read_csv(OUT / "application_ei_seasonal_methods.csv")
application_methods = pd.read_csv(OUT / "application_methods.csv")
application_screening = pd.read_csv(OUT / "application_screening.csv")
application_series_registry = pd.read_csv(OUT / "application_series_registry.csv")


# %% [markdown]
# ## 1. Synthetic Benchmark
#
# The benchmark has two linked but distinct estimation paths.
#
# **Internal UniBM path**
# 1. start from the raw time series;
# 2. choose a block-size grid;
# 3. compute sliding or disjoint block maxima for each block size;
# 4. summarize those block maxima by the median (main target), mean, or mode;
# 5. regress `log(T_b)` on `log(b)` over a selected plateau;
# 6. for FGLS, estimate the covariance of the full log-summary curve by a dependence-preserving super-block bootstrap;
# 7. fit the final regression on the original full-data curve and report a bootstrap-assisted Wald/FGLS confidence interval.
#
# **External published-estimator path**
# 1. start from the same raw time series;
# 2. fit published xi estimators such as Hill, max-spectrum, Pickands, and DEdH-moment;
# 3. select the native tuning level for each estimator;
# 4. compute each estimator's native asymptotic standard error;
# 5. report an estimator-specific Gaussian/Wald confidence interval.
#
# The key distinction is that the internal UniBM bootstrap estimates the covariance structure of the regression inputs across block sizes, whereas the external baselines default to their own asymptotic Gaussian/Wald inference. A raw-series bootstrap percentile interval remains available only as an optional sensitivity analysis for those external estimators.
#
# %% [markdown]
# ## Definitions: Return Period, Design-Life Level, and `T`-Year Block-Maximum `\tau`-Quantiles
#
# Before turning to the benchmark and application results, it helps to separate three related but non-identical ideas.
#
# 1. **Classical return period / annual exceedance probability (AEP).**
#    A “100-year flood” in the classical engineering sense is usually shorthand for a level with annual exceedance probability `0.01`.
#    That does **not** mean “it will happen only once in 100 years.” Under the usual stationary/independent yearly-block approximation, the probability of seeing at least one exceedance over 100 years is `1 - (1 - 0.01)^100 ≈ 63.4%`.
# 2. **Design-life level.**
#    In the water-resources literature, a design-life level is a quantile of the maximum over a design-life span rather than an annual exceedance threshold.
# 3. **UniBM design-life level.**
#    The current UniBM application workflow estimates `Q_tau(M_T)`, the `tau`-quantile of the block maximum over a `T`-year design-life span. The headline application fit uses `tau = 0.50`, and the displayed companion curves `0.90 / 0.95 / 0.99` reuse the same plateau and `xi` while shifting only the intercept.
#
# In that sense, the application-side design-life-level panel is a direct estimate of a **design-life level**, i.e. a `T`-year block-maximum `tau`-quantile, rather than a classical return-period level.
# The quantile-scaling panel shows the fitted relationship on the `block size` axis, while the design-life-level panel simply evaluates the same fitted law at larger block sizes corresponding to longer design-life spans.
# Higher-`tau` application curves are not separate EVI estimators: they are **shared-`xi` derived quantiles** that keep the headline slope and plateau fixed.
#
# This quantity is often easier to communicate in planning settings:
# `Q_0.5(M_100y)` means that, under the fitted model, there is a 50% chance that the maximum over the next 100 years stays below that threshold.
# By contrast, `Q_0.99(M_100y)` is a much more stress-oriented upper design-life quantile and is naturally the least stable of the displayed application curves.
# That complements rather than replaces the classical AEP language used in engineering standards.
#
# Official USGS references for the classical hydrologic terminology:
#
# - [100-Year Flood—It's All About Chance](https://pubs.usgs.gov/gip/106/)
# - [Guidelines for determining flood flow frequency — Bulletin 17C](https://www.usgs.gov/publications/guidelines-determining-flood-flow-frequency-bulletin-17c)
#
# ```mermaid
# flowchart TD
#     A["Simulated or observed univariate time series"] --> B["Point-estimation layer"]
#
#     B --> C["Internal UniBM path"]
#     B --> D["External published-estimator path"]
#
#     C --> C1["Choose block-size grid<br/>generate_block_sizes()"]
#     C1 --> C2["For each block size b:<br/>compute block maxima<br/>sliding or disjoint"]
#     C2 --> C3["For each b:<br/>compute block summary T_b<br/>median (main), mean, or mode"]
#     C3 --> C4["Build scaling curve<br/>log(T_b) vs log(b)"]
#     C4 --> C5["Select plateau window<br/>linear fit error + curvature penalty"]
#     C5 --> C6["Estimate xi from slope<br/>OLS or FGLS"]
#
#     D --> D1["Fit published xi estimators<br/>Hill / max-spectrum / Pickands / DEdH-moment"]
#
#     C6 --> E["Internal uncertainty layer"]
#     D1 --> F["External uncertainty layer"]
#
#     E --> E1["Split original series into long super-block segments"]
#     E1 --> E2["For each segment and each b:<br/>precompute segment-level block maxima"]
#     E2 --> E3["Bootstrap replicate:<br/>resample segments with replacement"]
#     E3 --> E4["For each b:<br/>pool maxima from resampled segments"]
#     E4 --> E5["Recompute summary vector across b<br/>(log T_b1*, ..., log T_bK*)"]
#     E5 --> E6["Estimate covariance matrix of the log-summary curve"]
#     E6 --> E7["Fit final FGLS on the original full-data curve"]
#     E7 --> E8["Report bootstrap-assisted Wald/FGLS CI"]
#
#     F --> F1["Use estimator-specific asymptotic variance formulas"]
#     F1 --> F2["Compute standard error at the selected tuning level"]
#     F2 --> F3["Apply Gaussian/Wald approximation"]
#     F3 --> F4["Report native asymptotic CI"]
#
#     E8 --> G["Compare xi estimation and uncertainty"]
#     F4 --> G
#
#     C6 --> H["Design-life-level extrapolation<br/>estimate_design_life_level()"]
#     G --> I["Simulation metrics:<br/>APE, coverage, Winkler interval score"]
# ```
#
# The benchmark is now organized around the truth pair `(xi, theta)`, with `phi` retained only as a derived construction parameter for the simulation families.
# The notebook reads the currently materialized benchmark CSVs and reports the active universal grid directly from those files, so the vignette stays in sync if the benchmark design changes.
# The raw synthetic families are:
# `frechet_max_ar`,
# `moving_maxima_q99`,
# and `pareto_additive_ar1`.
# Each family has a closed-form map from `(xi, theta)` back to its construction parameter `phi`, so the benchmark can compare methods on the same truth grid even though the underlying dependence mechanisms differ.
# Each synthetic series in the main benchmark has length `n_obs = 365`, so the benchmark is explicitly framed as a short-record regime rather than a long-record asymptotic exercise.
# A longer `n_obs = 1000` run can still be generated separately as an appendix sensitivity without changing the main manuscript-facing cache.
#
# The full benchmark still spans the factorial overview:
# sliding versus disjoint blocks,
# median versus mean versus mode summaries,
# and OLS versus FGLS regression.
# For the main story, the first figure combines six methods:
# `mean-disjoint-OLS`, `mode-disjoint-OLS`, `median-disjoint-OLS`, `median-disjoint-FGLS`, `median-sliding-OLS`, and `median-sliding-FGLS`.
# That single chart shows the baseline target comparison under the simplest disjoint-OLS setup, then isolates what is gained by covariance-aware regression and by sliding blocks for the median target.
# The second figure fixes the proposed sliding/FGLS setup and compares `median`, `mean`, and `mode` targets together with published xi baselines `Hill`, `max-spectrum`, `Pickands`, and `DEdH moment`.
# The internal benchmark tables below report `median APE (IQR) / median Winkler interval score (IQR)` from bootstrap-assisted Wald CI for each internal method.
# The mixed internal-versus-external tables also report `median APE (IQR) / median Winkler interval score (IQR)`, with the understanding that internal methods use bootstrap-assisted Wald intervals while the external baselines use their native asymptotic Gaussian/Wald intervals.
# All interval metrics below use 95% confidence intervals, i.e. `alpha = 0.05` in the Winkler score.
# The main internal figure therefore returns to the simple two-row view:
# absolute percentage error and Winkler interval score.
# Appendix diagnostics below unpack interval quality into width and coverage.
#

# %%
benchmark = pd.read_csv(BENCH / "summary.csv")
external_benchmark = pd.read_csv(BENCH / "external_summary.csv")
ei_benchmark = pd.read_csv(BENCH / "ei_summary.csv")
ei_external_benchmark = pd.read_csv(BENCH / "ei_external_summary.csv")


def _grid_summary(frame, primary, secondary):
    primary_values = ", ".join(str(value) for value in sorted(frame[primary].dropna().unique()))
    secondary_values = ", ".join(
        str(value) for value in sorted(frame[secondary].dropna().unique())
    )
    return primary_values, secondary_values


evi_xi_grid, evi_theta_grid = _grid_summary(benchmark, "xi_true", "theta_true")
ei_xi_grid, ei_theta_grid = _grid_summary(ei_benchmark, "xi_true", "theta_true")

print(f"EVI benchmark_set = {BENCHMARK_SET}")
print(f"EVI xi grid = {evi_xi_grid}")
print(f"EVI theta grid = {evi_theta_grid}")
print(f"EI benchmark_set = {BENCHMARK_SET}")
print(f"EI xi grid = {ei_xi_grid}")
print(f"EI theta grid = {ei_theta_grid}")

core_table = benchmark_story_table(
    benchmark,
    methods=CORE_METHODS,
    benchmark_set=BENCHMARK_SET,
)
target_table = target_plus_external_story_table(
    benchmark,
    external_benchmark,
    benchmark_set=BENCHMARK_SET,
)

display(core_table)
print(
    benchmark_story_latex(
        benchmark,
        methods=CORE_METHODS,
        benchmark_set=BENCHMARK_SET,
        caption=f"Core EVI benchmark comparison across the xi grid {{{evi_xi_grid}}} at fixed theta in {{{evi_theta_grid}}}. Cells report median APE (IQR) / median Winkler interval score (IQR). All interval metrics use 95% CI (alpha = 0.05).",
        label="tab:vignette-benchmark-core",
    )
)

display(target_table)
print(
    target_plus_external_story_latex(
        benchmark,
        external_benchmark,
        benchmark_set=BENCHMARK_SET,
        caption=f"Target-comparison EVI benchmark across the xi grid {{{evi_xi_grid}}} at fixed theta in {{{evi_theta_grid}}}. Cells report median APE (IQR) / median Winkler interval score (IQR). All interval metrics use 95% CI (alpha = 0.05).",
        label="tab:vignette-benchmark-targets",
    )
)

plot_benchmark_panels(
    benchmark,
    benchmark_set=BENCHMARK_SET,
    methods=CORE_METHODS,
    title="Necessary components: from disjoint OLS baselines to sliding-median FGLS",
    legend_mode="explicit",
    interval_style="errorbar",
)
plt.show()

plot_target_plus_external_panels(
    benchmark,
    external_benchmark,
    benchmark_set=BENCHMARK_SET,
    title="Target comparison under sliding-block FGLS",
)
plt.show()


# %% [markdown]
# ### Appendix: Full Benchmark Overview
#
# The full twelve-method grid is retained here for completeness, but the main notebook story focuses on the condensed tables and the two comparison figures above.
#

# %%
display(benchmark_table(benchmark, benchmark_set=BENCHMARK_SET))


# %% [markdown]
# ### Appendix: 95% Interval Sharpness Versus Calibration
#
# The main benchmark figures emphasize point-estimation error and Winkler score. This appendix view adds interval sharpness:
# median 95% interval width, median coverage, and median Winkler interval score across the same `xi` grid.
# Lower width is sharper, lower interval score is better overall, and ideal coverage stays close to `0.95`.
#

# %%
sharpness_table = interval_sharpness_story_table(
    benchmark,
    external_benchmark,
    benchmark_set=BENCHMARK_SET,
)
display(sharpness_table)
print(
    interval_sharpness_story_latex(
        benchmark,
        external_benchmark,
        benchmark_set=BENCHMARK_SET,
        caption=f"Appendix EVI interval sharpness-versus-calibration summary across the xi grid {{{evi_xi_grid}}} at fixed theta in {{{evi_theta_grid}}}. All interval metrics use 95% CI (alpha = 0.05).",
        label="tab:vignette-benchmark-interval",
    )
)

plot_interval_sharpness_scatter(
    benchmark,
    external_benchmark,
    benchmark_set=BENCHMARK_SET,
    title="Appendix: 95% interval sharpness versus calibration",
)
plt.show()


# %% [markdown]
# ## 1B. Native EI Benchmark
#
# The new EI suite mirrors the EVI benchmark structurally, but it swaps the target of comparison:
# `xi` is fixed over the currently materialized EI grid read from the cached benchmark CSVs, while
# `theta` is swept over the corresponding EI dependence grid from the same cache.
# The internal UniBM EI estimators are pooled block-maxima methods built from Northrop or BB reciprocal-EI paths:
# disjoint or sliding blocks,
# OLS or FGLS pooling on `log(1 / theta_hat(b))`,
# and log-scale Wald intervals.
# The external EI baselines are:
# `Ferro-Segers` with Wald CI,
# `K-gaps` with profile-likelihood CI,
# native sliding `Northrop` with adjusted chandwich profile likelihood,
# and native sliding `BB` with Wald CI.
#
# The targets panel below keeps the selected pooled BM finalists
# `BB-disjoint-FGLS` and `BB-sliding-FGLS`
# and compares them against the four external baselines on the truth scale that matters for extremal clustering.
# As in the EVI benchmark, all interval metrics use 95% confidence intervals with `alpha = 0.05`.
#

# %%
ei_core_table = ei_core_story_table(ei_benchmark, benchmark_set=BENCHMARK_SET)
ei_target_table = ei_targets_story_table(
    ei_benchmark,
    ei_external_benchmark,
    benchmark_set=BENCHMARK_SET,
)
ei_interval_table = ei_interval_story_table(
    ei_benchmark,
    ei_external_benchmark,
    benchmark_set=BENCHMARK_SET,
)
display(ei_core_table)
print(
    ei_story_latex(
        ei_core_table,
        caption=f"EI core benchmark across the theta grid {{{ei_theta_grid}}} at fixed xi in {{{ei_xi_grid}}}. Cells report median APE (IQR) / median Winkler interval score (IQR). All interval metrics use 95% CI (alpha = 0.05).",
        label="tab:vignette-benchmark-ei-core",
    )
)

display(ei_target_table)
print(
    ei_story_latex(
        ei_target_table,
        caption=f"EI target benchmark across the theta grid {{{ei_theta_grid}}} at fixed xi in {{{ei_xi_grid}}}. Cells report median APE (IQR) / median Winkler interval score (IQR). All interval metrics use 95% CI (alpha = 0.05).",
        label="tab:vignette-benchmark-ei-targets",
    )
)

display(ei_interval_table)


# %%
plot_ei_core_panels(
    ei_benchmark,
)
plt.show()

plot_ei_targets_panels(
    ei_benchmark,
    ei_external_benchmark,
)
plt.show()

plot_ei_interval_sharpness_scatter(
    ei_benchmark,
    ei_external_benchmark,
)
plt.show()


# %% [markdown]
# ## 2. Dataset Screening
#
# Candidate applications are screened for record length, defensible preprocessing, stable intermediate-range scaling, and whether sliding block maxima remain informative after preprocessing.
#

# %%
screening = pd.read_csv(OUT / "application_screening.csv")
series_registry = pd.read_csv(OUT / "application_series_registry.csv")
application_summary = pd.read_csv(OUT / "application_summary.csv")
display(screening)
display(series_registry[["application", "provider", "role", "series_name", "series_basis"]])
display(application_summary)

with open(ROOT / "data" / "metadata" / "sources.json") as fh:
    sources = json.load(fh)
sources


# %% [markdown]
# ### What Each Application Actually Does
#
# The application workflow is intentionally not “one raw series, one estimator” for every case.
#
# - **Houston precipitation** uses the wet-season daily precipitation series for display and EVI only; it is retained as a weather-side tail example rather than a formal EI case.
# - **Texas and Florida streamflow** use the same full-year daily discharge series for display, EVI, and EI, so design-life levels and extremal-index summaries refer to the same hydrologic process.
# - **Texas and Florida NFIP claims** deliberately split the series by task:
#   the display and EI series are zero-filled daily state payout totals on the calendar-day axis,
#   while the EVI series keeps only positive-payout days so tail extrapolation is done on the claim-active-day scale.
# - **Phoenix hot-dry severity** is a derived compound-hazard index retained as a secondary EVI-only contrast case rather than a formal EI application.
#
# This split matters because the application chapter now combines three layers of environmental risk:
# physical hazard occurrence,
# hydrologic response,
# and downstream socio-economic impact.
#

# %% [markdown]
# ### Data Provenance and Source Records
#
# The application chapter is built from three public data systems:
#
# - **NOAA GHCN-Daily** station files for Houston precipitation and Phoenix temperature/precipitation inputs.
# - **USGS NWIS daily discharge** for the frozen Texas and Florida streamflow gauges.
# - **OpenFEMA NFIP Redacted Claims v2** for Texas and Florida building-claim payouts.
#
# The table below exposes the exact station ids, gauge ids, state filters, and source URLs used by the workflow so the case studies can be checked independently.
#

# %%
with open(ROOT / "data" / "metadata" / "sources.json") as fh:
    ghcn_sources = json.load(fh)
with open(ROOT / "data" / "metadata" / "application" / "usgs_frozen_sites.json") as fh:
    usgs_frozen_sites = json.load(fh)

provenance_rows = [
    {
        "application": "houston_hobby_precipitation",
        "provider": "NOAA GHCN-Daily",
        "source_reference": f"{ghcn_sources['houston_hobby_precipitation']['station_id']} ({ghcn_sources['houston_hobby_precipitation']['station_name']})",
        "source_url": ghcn_sources["houston_hobby_precipitation"]["source_url"],
        "notes": "Wet-season daily precipitation (Jun-Nov).",
    },
    {
        "application": "phoenix_hot_dry_severity",
        "provider": "NOAA GHCN-Daily",
        "source_reference": f"{ghcn_sources['phoenix_hot_dry_severity']['station_id']} ({ghcn_sources['phoenix_hot_dry_severity']['station_name']})",
        "source_url": ghcn_sources["phoenix_hot_dry_severity"]["source_url"],
        "notes": "Warm-season temperature and precipitation inputs used to build the 30-day hot-dry severity index.",
    },
    {
        "application": "tx_streamflow",
        "provider": "USGS NWIS daily discharge",
        "source_reference": f"{usgs_frozen_sites['TX']['site_no']} ({usgs_frozen_sites['TX']['station_name']})",
        "source_url": f"https://waterdata.usgs.gov/monitoring-location/{usgs_frozen_sites['TX']['site_no']}/",
        "notes": "Daily discharge (parameter 00060); frozen flagship Texas streamgage.",
    },
    {
        "application": "fl_streamflow",
        "provider": "USGS NWIS daily discharge",
        "source_reference": f"{usgs_frozen_sites['FL']['site_no']} ({usgs_frozen_sites['FL']['station_name']})",
        "source_url": f"https://waterdata.usgs.gov/monitoring-location/{usgs_frozen_sites['FL']['site_no']}/",
        "notes": "Daily discharge (parameter 00060); frozen flagship Florida streamgage.",
    },
    {
        "application": "tx_nfip_claims",
        "provider": "OpenFEMA NFIP Redacted Claims v2",
        "source_reference": "Texas state filter on dateOfLoss and amountPaidOnBuildingClaim",
        "source_url": OPENFEMA_NFIP_CLAIMS_ENDPOINT,
        "notes": "Building-claim payouts aggregated to daily state totals; EVI uses positive-payout days, EI keeps zero-filled calendar days.",
    },
    {
        "application": "fl_nfip_claims",
        "provider": "OpenFEMA NFIP Redacted Claims v2",
        "source_reference": "Florida state filter on dateOfLoss and amountPaidOnBuildingClaim",
        "source_url": OPENFEMA_NFIP_CLAIMS_ENDPOINT,
        "notes": "Building-claim payouts aggregated to daily state totals; EVI uses positive-payout days, EI keeps zero-filled calendar days.",
    },
]

provenance = pd.DataFrame(provenance_rows)
display(provenance)
display(Markdown(f"USGS API endpoint used by the downloader: `{USGS_DV_ENDPOINT}`"))


# %% [markdown]
# ### Manuscript-Facing Application Tables
#
# The application workflow now emits a matched LaTeX table set for manuscript assembly:
# a cross-application summary table, a design-life-level table, and an EI-focused comparison table.
# This keeps the application chapter structurally parallel to the benchmark/report pipelines rather than relying on figures alone.
# By design, the default EVI method table now reports only the headline `sliding_median_fgls` fit,
# expanded over the application tau grid `0.50 / 0.90 / 0.95 / 0.99` by reusing the same plateau and `xi`
# while estimating only tau-specific intercept shifts.
# The EI comparison table carries the four-method set
# `BB-sliding-FGLS`, `Northrop-sliding-FGLS`, `K-gaps`, and `Ferro-Segers`
# only for the formal EI applications (streamflow and NFIP).
#

# %% [markdown]
# ### Inline Application Plots
#
# Unlike the benchmark sections, the application plots require fitted bundle objects rather than only summary CSVs.
# The next cell reuses the cached raw inputs and re-fits the six application bundles once so the notebook can call the plotting helpers directly and display the figures inline.
# The main per-application visual is now a single **composite diagnostic figure**.
# For streamflow and NFIP it aligns target stability, the headline median-sliding-FGLS scaling fit, the four-method EI comparison, and the design-life-level panel.
# For Houston and Phoenix it becomes an EVI-only diagnostic figure with the raw daily series in place of the EI panel.
#

# %%
application_bundles = build_application_bundles(DIRS)
application_bundle_map = {bundle.spec.key: bundle for bundle in application_bundles}
plot_application_overview(application_bundles)
plt.show()


# %%
application_design_life_levels = pd.read_csv(OUT / "application_design_life_levels.csv")
application_ei_methods = pd.read_csv(OUT / "application_ei_methods.csv")
application_ei_seasonal_methods = pd.read_csv(OUT / "application_ei_seasonal_methods.csv")

display(application_summary)
display(application_design_life_levels.head(12))
display(application_ei_methods)
display(application_ei_seasonal_methods)

print((TABLE_DIR / "application_summary_main.tex").read_text())
print((TABLE_DIR / "application_design_life_levels_main.tex").read_text())
print((TABLE_DIR / "application_ei_main.tex").read_text())


# %%
def _fmt_value(value, *, digits=2, sci_at=1e6):
    if pd.isna(value):
        return "NA"
    value = float(value)
    if abs(value) >= sci_at:
        return f"{value:.2e}"
    if abs(value) >= 1_000:
        return f"{value:,.0f}"
    return f"{value:.{digits}f}"


def _fmt_interval(center, lo, hi, *, digits=2, sci_at=1e6):
    return f"{_fmt_value(center, digits=digits, sci_at=sci_at)} [{_fmt_value(lo, digits=digits, sci_at=sci_at)}, {_fmt_value(hi, digits=digits, sci_at=sci_at)}]"


def _summary_row(app_key):
    return application_summary.loc[application_summary["application"] == app_key].iloc[0]


def _screening_row(app_key, analysis_type):
    return application_screening.loc[
        (application_screening["name"] == app_key)
        & (application_screening["analysis_type"] == analysis_type)
    ].iloc[0]


def _design_life_row(app_key, years):
    return application_design_life_levels.loc[
        (application_design_life_levels["application"] == app_key)
        & (application_design_life_levels["tau"] == 0.5)
        & (application_design_life_levels["design_life_years"] == float(years))
    ].iloc[0]


def _design_life_row_tau(app_key, years, tau):
    return application_design_life_levels.loc[
        (application_design_life_levels["application"] == app_key)
        & (application_design_life_levels["tau"] == float(tau))
        & (application_design_life_levels["design_life_years"] == float(years))
    ].iloc[0]


def _seasonal_row(app_key, method):
    return application_ei_seasonal_methods.loc[
        (application_ei_seasonal_methods["application"] == app_key)
        & (application_ei_seasonal_methods["method"] == method)
    ].iloc[0]


# %%
tx_stream_row = _summary_row("tx_streamflow")
fl_stream_row = _summary_row("fl_streamflow")
tx_nfip_row = _summary_row("tx_nfip_claims")
fl_nfip_row = _summary_row("fl_nfip_claims")

display(
    Markdown(
        f"""
**Cross-application reading guide**

- **Texas and Florida streamflow** are the strongest clustering cases: BB-sliding-FGLS gives `theta = {_fmt_value(tx_stream_row["theta_hat_bb_sliding_fgls"], digits=3)}` in Texas and `{_fmt_value(fl_stream_row["theta_hat_bb_sliding_fgls"], digits=3)}` in Florida, implying flood-wave cluster sizes around `{_fmt_value(tx_stream_row["mean_cluster_size"])}` and `{_fmt_value(fl_stream_row["mean_cluster_size"])}` days.
- **Texas and Florida NFIP claims** have the heaviest tails: `xi = {_fmt_value(tx_nfip_row["xi_hat"])}` in Texas and `{_fmt_value(fl_nfip_row["xi_hat"])}` in Florida, with both states showing multi-day claim waves (`theta ~ 0.31`).
- **Houston precipitation and Phoenix hot-dry severity** remain useful weather-side EVI contrasts, but formal EI interpretation is intentionally reserved for the streamflow and NFIP applications.
        """
    )
)


# %% [markdown]
# **Seasonal-adjusted EI sensitivity.**
#
# The main EI workflow is still run on the original prepared EI series for streamflow and NFIP.
# As an appendix sensitivity, the notebook also reports a monthly empirical-PIT to unit-Frechet transform that strips month-specific marginal seasonality while preserving the daily ordering.
# Those seasonal-adjusted EI rows are shown below each application and should be read as a robustness check, not as the headline application estimator.
#
# **How the scaling, design-life-level, and EI panels fit together.**
#
# The design-life-level panel is **not** a separate annual-maxima or GEV fit.
# It is the same UniBM scaling law shown in the quantile-scaling panel, simply evaluated at larger block sizes corresponding to longer design-life spans.
# In other words, the quantile-scaling panel works on `log(block size)` versus `log block summary`, while the design-life-level panel converts those larger block sizes into design-life lengths on the original physical scale.
# That is why the two panels are mathematically linked but still use different x-axes: one is a fitting axis (`block size`), the other is an interpretation axis (`design-life length in years`).
#
# The EVI plateau and the EI stable window also need not coincide.
# The EVI fit asks where the block-quantile curve is approximately linear on the log-log scale, whereas the EI fit asks where the estimated clustering path is stable enough to support a formal `theta` estimate.
# Those are different statistical objects, so it is normal for the selected block-size ranges to overlap only partially or even to sit in different parts of the admissible grid.
#
# In the current direct BM-quantile workflow, dependence is already built into the fitted block-maximum law because the block maxima are computed from the original dependent series.
# We therefore read the design-life-level curve directly as a design-life severity curve rather than applying a second BM-side `theta` adjustment.
# Within the application outputs, `tau = 0.50` is the headline design-life median and the higher curves `0.90 / 0.95 / 0.99` are shared-`xi` companions rather than separately re-estimated `xi_tau` fits.
#
# For risk management, the two outputs answer different questions.
# Use the design-life-level panel when the decision is about **severity** on the original scale, such as discharge or payout magnitude at a chosen design-life span.
# Use the EI panel when the decision is about **persistence and recovery**, such as whether extremes arrive as isolated shocks or as multi-day flood waves or claim waves.
# Streamflow therefore combines design-life levels and EI on the same calendar-day process, while NFIP intentionally keeps active-day design-life levels and calendar-day EI on separate clocks.
#
# %% [markdown]
# ## 3. Hydrologic Response Applications: Texas and Florida Streamflow
#
# The streamflow applications move one step downstream from weather into hydrologic response.
# Here the same daily discharge series is used for display, EVI, and EI, so the estimated `xi`, `theta`, and design-life levels all describe the same calendar-day river process.
# This makes the streamflow cases the cleanest bridge from the meteorological Houston case to impact-facing insurance data.
#

# %%
streamflow_bundles = [application_bundle_map[key] for key in ["tx_streamflow", "fl_streamflow"]]
streamflow_screening = application_screening[
    application_screening["name"].isin(["tx_streamflow", "fl_streamflow"])
].copy()
streamflow_summary = application_summary[
    application_summary["application"].isin(["tx_streamflow", "fl_streamflow"])
].copy()
streamflow_design_life_levels = application_design_life_levels[
    application_design_life_levels["application"].isin(["tx_streamflow", "fl_streamflow"])
].copy()
streamflow_ei = application_ei_methods[
    application_ei_methods["application"].isin(["tx_streamflow", "fl_streamflow"])
].copy()
streamflow_seasonal_ei = application_ei_seasonal_methods[
    application_ei_seasonal_methods["application"].isin(["tx_streamflow", "fl_streamflow"])
].copy()
streamflow_methods = application_methods[
    application_methods["application"].isin(["tx_streamflow", "fl_streamflow"])
].copy()
display(streamflow_screening)
display(streamflow_summary)
display(streamflow_design_life_levels)
display(streamflow_ei)
display(streamflow_seasonal_ei)
display(streamflow_methods)

plot_application_time_series(streamflow_bundles[0])
plt.show()
plot_application_composite(streamflow_bundles[0])
plt.show()
plot_application_time_series(streamflow_bundles[1])
plt.show()
plot_application_composite(streamflow_bundles[1])
plt.show()


# %%
tx_stream_design_life_10 = _design_life_row("tx_streamflow", 10.0)
fl_stream_design_life_10 = _design_life_row("fl_streamflow", 10.0)
tx_stream_design_life_50 = _design_life_row("tx_streamflow", 50.0)
fl_stream_design_life_50 = _design_life_row("fl_streamflow", 50.0)
tx_stream_design_life_10_p95 = _design_life_row_tau("tx_streamflow", 10.0, 0.95)
fl_stream_design_life_10_p95 = _design_life_row_tau("fl_streamflow", 10.0, 0.95)
tx_stream_design_life_50_p99 = _design_life_row_tau("tx_streamflow", 50.0, 0.99)
fl_stream_design_life_50_p99 = _design_life_row_tau("fl_streamflow", 50.0, 0.99)
tx_stream_northrop = streamflow_ei.loc[
    (streamflow_ei["application"] == "tx_streamflow")
    & (streamflow_ei["method"] == "northrop_sliding_fgls")
].iloc[0]
fl_stream_northrop = streamflow_ei.loc[
    (streamflow_ei["application"] == "fl_streamflow")
    & (streamflow_ei["method"] == "northrop_sliding_fgls")
].iloc[0]
tx_stream_ferro = streamflow_ei.loc[
    (streamflow_ei["application"] == "tx_streamflow") & (streamflow_ei["method"] == "ferro_segers")
].iloc[0]
fl_stream_ferro = streamflow_ei.loc[
    (streamflow_ei["application"] == "fl_streamflow") & (streamflow_ei["method"] == "ferro_segers")
].iloc[0]
tx_stream_seasonal_bb = _seasonal_row("tx_streamflow", "bb_sliding_fgls")
fl_stream_seasonal_bb = _seasonal_row("fl_streamflow", "bb_sliding_fgls")

display(
    Markdown(
        f"""
**Interpretation.** The two river gauges are the most dependence-dominated applications in the package. BB-sliding-FGLS gives `theta = {_fmt_interval(streamflow_summary.loc[streamflow_summary["application"] == "tx_streamflow", "theta_hat_bb_sliding_fgls"].iloc[0], streamflow_summary.loc[streamflow_summary["application"] == "tx_streamflow", "theta_lo_bb_sliding_fgls"].iloc[0], streamflow_summary.loc[streamflow_summary["application"] == "tx_streamflow", "theta_hi_bb_sliding_fgls"].iloc[0], digits=3)}` in Texas and `{_fmt_interval(streamflow_summary.loc[streamflow_summary["application"] == "fl_streamflow", "theta_hat_bb_sliding_fgls"].iloc[0], streamflow_summary.loc[streamflow_summary["application"] == "fl_streamflow", "theta_lo_bb_sliding_fgls"].iloc[0], streamflow_summary.loc[streamflow_summary["application"] == "fl_streamflow", "theta_hi_bb_sliding_fgls"].iloc[0], digits=3)}` in Florida, corresponding to average flood-wave cluster sizes of roughly `{_fmt_value(streamflow_summary.loc[streamflow_summary["application"] == "tx_streamflow", "mean_cluster_size"].iloc[0])}` and `{_fmt_value(streamflow_summary.loc[streamflow_summary["application"] == "fl_streamflow", "mean_cluster_size"].iloc[0])}` days.

That main pooled-BM story is reinforced rather than contradicted by the other EI estimators: the Northrop pooled-BM fits are `{_fmt_interval(tx_stream_northrop["theta_hat"], tx_stream_northrop["theta_lo"], tx_stream_northrop["theta_hi"], digits=3)}` in Texas and `{_fmt_interval(fl_stream_northrop["theta_hat"], fl_stream_northrop["theta_lo"], fl_stream_northrop["theta_hi"], digits=3)}` in Florida, while Ferro-Segers remains in the same low-theta regime at `{_fmt_interval(tx_stream_ferro["theta_hat"], tx_stream_ferro["theta_lo"], tx_stream_ferro["theta_hi"], digits=3)}` and `{_fmt_interval(fl_stream_ferro["theta_hat"], fl_stream_ferro["theta_lo"], fl_stream_ferro["theta_hi"], digits=3)}`.

The design-life severity scale is still substantial. The headline median 10-year design-life levels are `{_fmt_value(tx_stream_design_life_10["design_life_level"])}` in Texas and `{_fmt_value(fl_stream_design_life_10["design_life_level"])}` in Florida, while the shared-`xi` `tau = 0.95` 10-year curves rise to `{_fmt_value(tx_stream_design_life_10_p95["design_life_level"])}` and `{_fmt_value(fl_stream_design_life_10_p95["design_life_level"])}`. At the longer and more stress-oriented end, the `tau = 0.99` 50-year design-life levels reach `{_fmt_value(tx_stream_design_life_50_p99["design_life_level"])}` and `{_fmt_value(fl_stream_design_life_50_p99["design_life_level"])}`. Those numbers all describe peak flood severity on the calendar-day discharge scale; the EI results above should be read alongside them to quantify how long one flood episode tends to persist once it starts.

Seasonal adjustment does not erase that conclusion. The monthly-PIT BB sensitivity remains very small at `{_fmt_interval(tx_stream_seasonal_bb["theta_hat"], tx_stream_seasonal_bb["theta_lo"], tx_stream_seasonal_bb["theta_hi"], digits=3)}` in Texas and `{_fmt_interval(fl_stream_seasonal_bb["theta_hat"], fl_stream_seasonal_bb["theta_lo"], fl_stream_seasonal_bb["theta_hi"], digits=3)}` in Florida, so the clustering story is not just a marginal seasonal artifact.
        """
    )
)


# %% [markdown]
# ## 4. Hazard-to-Impact Applications: Texas and Florida NFIP Building Payout Waves
#
# The NFIP cases are intentionally constructed differently from the raw physical-hazard series.
# We keep the **calendar-day zero-filled daily payout totals** for display and EI so the clustering analysis preserves claim-wave timing,
# but we fit EVI and design-life levels on the **positive-payout-day** series so the tail extrapolation is not diluted by long runs of structural zeros.
# The design-life levels reported here should therefore be read as **claim-active-day design-life levels**, not calendar-day design-life levels.
#

# %%
nfip_bundles = [application_bundle_map[key] for key in ["tx_nfip_claims", "fl_nfip_claims"]]
nfip_screening = application_screening[
    application_screening["name"].isin(["tx_nfip_claims", "fl_nfip_claims"])
].copy()
nfip_summary = application_summary[
    application_summary["application"].isin(["tx_nfip_claims", "fl_nfip_claims"])
].copy()
nfip_design_life_levels = application_design_life_levels[
    application_design_life_levels["application"].isin(["tx_nfip_claims", "fl_nfip_claims"])
].copy()
nfip_ei = application_ei_methods[
    application_ei_methods["application"].isin(["tx_nfip_claims", "fl_nfip_claims"])
].copy()
nfip_seasonal_ei = application_ei_seasonal_methods[
    application_ei_seasonal_methods["application"].isin(["tx_nfip_claims", "fl_nfip_claims"])
].copy()
nfip_methods = application_methods[
    application_methods["application"].isin(["tx_nfip_claims", "fl_nfip_claims"])
].copy()
nfip_registry = series_registry[
    series_registry["application"].isin(["tx_nfip_claims", "fl_nfip_claims"])
].copy()
display(nfip_registry[["application", "role", "series_name", "series_basis", "n_obs"]])
display(nfip_screening)
display(nfip_summary)
display(nfip_design_life_levels)
display(nfip_ei)
display(nfip_seasonal_ei)
display(nfip_methods)

plot_application_time_series(nfip_bundles[0])
plt.show()
plot_application_composite(nfip_bundles[0])
plt.show()
plot_application_time_series(nfip_bundles[1])
plt.show()
plot_application_composite(nfip_bundles[1])
plt.show()


# %%
tx_nfip_evi_screen = _screening_row("tx_nfip_claims", "evi")
fl_nfip_evi_screen = _screening_row("fl_nfip_claims", "evi")
tx_nfip_ei_screen = _screening_row("tx_nfip_claims", "ei")
fl_nfip_ei_screen = _screening_row("fl_nfip_claims", "ei")
tx_nfip_design_life_10 = _design_life_row("tx_nfip_claims", 10.0)
fl_nfip_design_life_10 = _design_life_row("fl_nfip_claims", 10.0)
tx_nfip_design_life_10_p95 = _design_life_row_tau("tx_nfip_claims", 10.0, 0.95)
fl_nfip_design_life_10_p95 = _design_life_row_tau("fl_nfip_claims", 10.0, 0.95)
tx_nfip_design_life_50_p99 = _design_life_row_tau("tx_nfip_claims", 50.0, 0.99)
fl_nfip_design_life_50_p99 = _design_life_row_tau("fl_nfip_claims", 50.0, 0.99)
tx_nfip_northrop = nfip_ei.loc[
    (nfip_ei["application"] == "tx_nfip_claims") & (nfip_ei["method"] == "northrop_sliding_fgls")
].iloc[0]
fl_nfip_northrop = nfip_ei.loc[
    (nfip_ei["application"] == "fl_nfip_claims") & (nfip_ei["method"] == "northrop_sliding_fgls")
].iloc[0]
tx_nfip_ferro = nfip_ei.loc[
    (nfip_ei["application"] == "tx_nfip_claims") & (nfip_ei["method"] == "ferro_segers")
].iloc[0]
fl_nfip_ferro = nfip_ei.loc[
    (nfip_ei["application"] == "fl_nfip_claims") & (nfip_ei["method"] == "ferro_segers")
].iloc[0]
tx_nfip_seasonal_bb = _seasonal_row("tx_nfip_claims", "bb_sliding_fgls")
fl_nfip_seasonal_bb = _seasonal_row("fl_nfip_claims", "bb_sliding_fgls")

display(
    Markdown(
        f"""
**Interpretation.** The NFIP applications are the clearest heavy-tail examples in the package. Texas has `xi = {_fmt_interval(nfip_summary.loc[nfip_summary["application"] == "tx_nfip_claims", "xi_hat"].iloc[0], nfip_summary.loc[nfip_summary["application"] == "tx_nfip_claims", "xi_lo"].iloc[0], nfip_summary.loc[nfip_summary["application"] == "tx_nfip_claims", "xi_hi"].iloc[0])}` and Florida has `{_fmt_interval(nfip_summary.loc[nfip_summary["application"] == "fl_nfip_claims", "xi_hat"].iloc[0], nfip_summary.loc[nfip_summary["application"] == "fl_nfip_claims", "xi_lo"].iloc[0], nfip_summary.loc[nfip_summary["application"] == "fl_nfip_claims", "xi_hi"].iloc[0])}`, so the claim-active-day tail is much heavier than in the physical-hazard series.

The split-series design is also justified by the raw timing structure: only `{100 * tx_nfip_ei_screen["daily_positive_share"]:.1f}%` of Texas calendar days and `{100 * fl_nfip_ei_screen["daily_positive_share"]:.1f}%` of Florida calendar days carry positive payouts, so EI needs the zero-filled daily axis to preserve claim-wave timing. On that calendar scale, BB-sliding-FGLS gives `theta ~ {_fmt_value(nfip_summary.loc[nfip_summary["application"] == "tx_nfip_claims", "theta_hat_bb_sliding_fgls"].iloc[0])}` in Texas and `{_fmt_value(nfip_summary.loc[nfip_summary["application"] == "fl_nfip_claims", "theta_hat_bb_sliding_fgls"].iloc[0])}` in Florida, with Northrop pooled-BM at `{_fmt_interval(tx_nfip_northrop["theta_hat"], tx_nfip_northrop["theta_lo"], tx_nfip_northrop["theta_hi"])}` and `{_fmt_interval(fl_nfip_northrop["theta_hat"], fl_nfip_northrop["theta_lo"], fl_nfip_northrop["theta_hi"])}` and Ferro-Segers at `{_fmt_interval(tx_nfip_ferro["theta_hat"], tx_nfip_ferro["theta_lo"], tx_nfip_ferro["theta_hi"])}` and `{_fmt_interval(fl_nfip_ferro["theta_hat"], fl_nfip_ferro["theta_lo"], fl_nfip_ferro["theta_hi"])}`. Those estimates all point to multi-day claim waves rather than isolated one-day impacts.

The headline 10-year claim-active-day design-life levels, `{_fmt_value(tx_nfip_design_life_10["design_life_level"])}` in Texas and `{_fmt_value(fl_nfip_design_life_10["design_life_level"])}` in Florida, should therefore be read together with the EI evidence for multi-day claim waves. The upper shared-`xi` curves are materially higher: `tau = 0.95` gives `{_fmt_value(tx_nfip_design_life_10_p95["design_life_level"])}` in Texas and `{_fmt_value(fl_nfip_design_life_10_p95["design_life_level"])}` in Florida, while the 50-year `tau = 0.99` stress curves reach `{_fmt_value(tx_nfip_design_life_50_p99["design_life_level"])}` and `{_fmt_value(fl_nfip_design_life_50_p99["design_life_level"])}`. The design-life-level curve is fit on **positive claim-active days**, whereas the EI analysis is defined on the **zero-filled calendar-day process**. Those are different clocks, so they are reported side by side rather than forced onto one axis as if they were the same estimand.

Seasonal adjustment also leaves the qualitative claim-wave story intact. The monthly-PIT BB sensitivity lands at `{_fmt_interval(tx_nfip_seasonal_bb["theta_hat"], tx_nfip_seasonal_bb["theta_lo"], tx_nfip_seasonal_bb["theta_hi"])}` in Texas and `{_fmt_interval(fl_nfip_seasonal_bb["theta_hat"], fl_nfip_seasonal_bb["theta_lo"], fl_nfip_seasonal_bb["theta_hi"])}` in Florida, so the clustering evidence is not an artifact of month-specific payout levels alone.
        """
    )
)


# %% [markdown]
# ## 5. Secondary Weather Applications: Houston Precipitation and Phoenix Hot-Dry Severity
#
# Houston and Phoenix are retained as weather-side contrasts, but they are intentionally demoted relative to the streamflow and NFIP case studies.
# In this revised package they serve only as EVI-focused visual examples, so the notebook keeps them brief: a time-series view plus the EVI-oriented composite diagnostic for each series.
#

# %%
houston_bundle = application_bundle_map["houston_hobby_precipitation"]
phoenix_bundle = application_bundle_map["phoenix_hot_dry_severity"]

plot_application_time_series(houston_bundle)
plt.show()
plot_application_composite(houston_bundle)
plt.show()

plot_application_time_series(phoenix_bundle)
plt.show()
plot_application_composite(phoenix_bundle)
plt.show()


# %%
houston_summary = application_summary[
    application_summary["application"] == "houston_hobby_precipitation"
].copy()
phoenix_summary = application_summary[
    application_summary["application"] == "phoenix_hot_dry_severity"
].copy()
houston_design_life_10 = _design_life_row("houston_hobby_precipitation", 10.0)
phoenix_design_life_10 = _design_life_row("phoenix_hot_dry_severity", 10.0)
houston_design_life_10_p95 = _design_life_row_tau("houston_hobby_precipitation", 10.0, 0.95)
phoenix_design_life_10_p95 = _design_life_row_tau("phoenix_hot_dry_severity", 10.0, 0.95)

display(
    Markdown(
        f"""
**Interpretation.** These two weather-driven series remain useful for illustrating the EVI side of the workflow, but they are not treated as formal EI applications in this notebook.
Houston retains the heavier weather-side tail (`xi = {_fmt_interval(houston_summary.iloc[0]["xi_hat"], houston_summary.iloc[0]["xi_lo"], houston_summary.iloc[0]["xi_hi"])}`), while Phoenix provides a milder-tail compound-hazard contrast (`xi = {_fmt_interval(phoenix_summary.iloc[0]["xi_hat"], phoenix_summary.iloc[0]["xi_lo"], phoenix_summary.iloc[0]["xi_hi"])}`).
The headline 10-year UniBM design-life levels are `{_fmt_value(houston_design_life_10["design_life_level"])}` for Houston wet-season precipitation and `{_fmt_value(phoenix_design_life_10["design_life_level"])}` for Phoenix hot-dry severity. The shared-`xi` `tau = 0.95` companion curves push those 10-year values up to `{_fmt_value(houston_design_life_10_p95["design_life_level"])}` and `{_fmt_value(phoenix_design_life_10_p95["design_life_level"])}`.
        """
    )
)
