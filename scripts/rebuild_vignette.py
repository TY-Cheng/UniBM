"""Programmatically rebuild the research vignette notebook.

The notebook is treated as a generated artifact so the public story stays in
sync with the current module structure, benchmark figures, and manuscript
exports. Keeping the notebook source here also makes large content changes
reviewable in plain Python rather than raw notebook JSON.
"""

from __future__ import annotations

import json

try:  # pragma: no cover - exercised through script execution
    from config import resolve_repo_dirs
    from vignette_cells import code_cell, md_cell
    from vignette_sections import group_application_section, single_application_section
except ImportError:  # pragma: no cover - exercised through module import
    from .config import resolve_repo_dirs
    from .vignette_cells import code_cell, md_cell
    from .vignette_sections import group_application_section, single_application_section


def build_notebook() -> dict:
    """Construct the vignette notebook as a plain nbformat-compatible dict."""
    cells = [
        md_cell(
            """
            # UniBM manuscript workflow

            This vignette demonstrates the revised `UniBM` workflow for **sliding-block, quantile-first inference** under the environmental trilemma of scarcity, dependence, and slow penultimate convergence.

            The notebook mirrors the revised manuscript:
            1. synthetic benchmark evidence;
            2. dataset screening;
            3. Houston precipitation plus Texas/Florida streamflow as the physical-hazard applications;
            4. Texas/Florida NFIP building payouts as the hazard-to-impact applications;
            5. Phoenix hot-dry severity as a secondary compound-hazard case.
            """
        ),
        md_cell(
            """
            ## Setup

            The first code cell is intentionally lightweight: it only resolves paths and imports the plotting/statistical helpers used below.
            The second code cell is an explicit rebuild switch for the derived series, benchmark summaries, and manuscript figures.
            """
        ),
        code_cell(
            """
            # ruff: noqa: E402
            from pathlib import Path
            import importlib.util
            import json

            def _load_import_bootstrap(start: Path):
                for _candidate in (start, *start.parents):
                    _helper_path = _candidate / "scripts" / "workflows" / "import_bootstrap.py"
                    if _helper_path.exists():
                        _spec = importlib.util.spec_from_file_location(
                            "_workflow_import_bootstrap",
                            _helper_path,
                        )
                        if _spec is None or _spec.loader is None:
                            raise ImportError(f"Could not load import bootstrap helper from {_helper_path}")
                        _module = importlib.util.module_from_spec(_spec)
                        _spec.loader.exec_module(_module)
                        return _module
                raise FileNotFoundError(
                    "Could not locate scripts/workflows/import_bootstrap.py from the current notebook session."
                )

            _bootstrap = _load_import_bootstrap(Path.cwd().resolve())
            _scripts_dir = _bootstrap.bootstrap_notebook_scripts_dir(Path.cwd().resolve())

            from unibm._runtime import prepare_matplotlib_env

            prepare_matplotlib_env("unibm-notebook")
            import matplotlib.pyplot as plt
            import pandas as pd
            from config import resolve_repo_dirs
            from IPython.display import Markdown, display
            from data_prep.fema import OPENFEMA_NFIP_CLAIMS_ENDPOINT
            from data_prep.usgs import USGS_DV_ENDPOINT
            from workflows.notebook_api import (
                CORE_METHODS,
                UNIVERSAL_BENCHMARK_SET,
                benchmark_story_latex,
                benchmark_story_table,
                benchmark_table,
                build_application_bundles,
                build_application_outputs,
                build_ei_benchmark_manuscript_outputs,
                build_evi_benchmark_manuscript_outputs,
                ei_core_story_table,
                ei_interval_story_table,
                ei_story_latex,
                ei_targets_story_table,
                interval_sharpness_story_latex,
                interval_sharpness_story_table,
                plot_application_ei,
                plot_application_overview,
                plot_application_return_levels,
                plot_application_scaling,
                plot_application_target_stability,
                plot_application_time_series,
                plot_benchmark_panels,
                plot_ei_core_panels,
                plot_ei_interval_sharpness_scatter,
                plot_ei_targets_panels,
                plot_interval_sharpness_scatter,
                plot_target_plus_external_panels,
                target_plus_external_story_latex,
                target_plus_external_story_table,
            )

            DIRS = resolve_repo_dirs()
            ROOT = DIRS["DIR_WORK"]
            OUT = DIRS["DIR_OUT_APPLICATIONS"]
            BENCH = DIRS["DIR_OUT_BENCHMARK"]
            FIG_DIR = DIRS["DIR_MANUSCRIPT_FIGURE"]
            TABLE_DIR = DIRS["DIR_MANUSCRIPT_TABLE"]
            BENCHMARK_SET = UNIVERSAL_BENCHMARK_SET
            """
        ),
        code_cell(
            """
            REBUILD_OUTPUTS = False
            REQUIRED_OUTPUTS = [
                OUT / "application_series_registry.csv",
                OUT / "application_screening.csv",
                OUT / "application_summary.csv",
                OUT / "application_return_levels.csv",
                OUT / "application_methods.csv",
                OUT / "application_ei_methods.csv",
                OUT / "application_usgs_site_screening.csv",
                TABLE_DIR / "application_summary_main.tex",
                TABLE_DIR / "application_return_levels_main.tex",
                TABLE_DIR / "application_ei_main.tex",
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
            """
        ),
        code_cell(
            """
            application_summary = pd.read_csv(OUT / "application_summary.csv")
            application_return_levels = pd.read_csv(OUT / "application_return_levels.csv")
            application_ei_methods = pd.read_csv(OUT / "application_ei_methods.csv")
            application_methods = pd.read_csv(OUT / "application_methods.csv")
            application_screening = pd.read_csv(OUT / "application_screening.csv")
            application_series_registry = pd.read_csv(OUT / "application_series_registry.csv")
            """
        ),
        md_cell(
            """
            ## 1. Synthetic Benchmark

            The benchmark has two linked but distinct estimation paths.

            **Internal UniBM path**
            1. start from the raw time series;
            2. choose a block-size grid;
            3. compute sliding or disjoint block maxima for each block size;
            4. summarize those block maxima by the median (main target), mean, or mode;
            5. regress `log(T_b)` on `log(b)` over a selected plateau;
            6. for FGLS, estimate the covariance of the full log-summary curve by a dependence-preserving super-block bootstrap;
            7. fit the final regression on the original full-data curve and report a bootstrap-assisted Wald/FGLS confidence interval.

            **External published-estimator path**
            1. start from the same raw time series;
            2. fit published xi estimators such as Hill, max-spectrum, Pickands, and DEdH-moment;
            3. select the native tuning level for each estimator;
            4. compute each estimator's native asymptotic standard error;
            5. report an estimator-specific Gaussian/Wald confidence interval.

            The key distinction is that the internal UniBM bootstrap estimates the covariance structure of the regression inputs across block sizes, whereas the external baselines default to their own asymptotic Gaussian/Wald inference. A raw-series bootstrap percentile interval remains available only as an optional sensitivity analysis for those external estimators.

            ```mermaid
            flowchart TD
                A["Simulated or observed univariate time series"] --> B["Point-estimation layer"]

                B --> C["Internal UniBM path"]
                B --> D["External published-estimator path"]

                C --> C1["Choose block-size grid<br/>generate_block_sizes()"]
                C1 --> C2["For each block size b:<br/>compute block maxima<br/>sliding or disjoint"]
                C2 --> C3["For each b:<br/>compute block summary T_b<br/>median (main), mean, or mode"]
                C3 --> C4["Build scaling curve<br/>log(T_b) vs log(b)"]
                C4 --> C5["Select plateau window<br/>linear fit error + curvature penalty"]
                C5 --> C6["Estimate xi from slope<br/>OLS or FGLS"]

                D --> D1["Fit published xi estimators<br/>Hill / max-spectrum / Pickands / DEdH-moment"]

                C6 --> E["Internal uncertainty layer"]
                D1 --> F["External uncertainty layer"]

                E --> E1["Split original series into long super-block segments"]
                E1 --> E2["For each segment and each b:<br/>precompute segment-level block maxima"]
                E2 --> E3["Bootstrap replicate:<br/>resample segments with replacement"]
                E3 --> E4["For each b:<br/>pool maxima from resampled segments"]
                E4 --> E5["Recompute summary vector across b<br/>(log T_b1*, ..., log T_bK*)"]
                E5 --> E6["Estimate covariance matrix of the log-summary curve"]
                E6 --> E7["Fit final FGLS on the original full-data curve"]
                E7 --> E8["Report bootstrap-assisted Wald/FGLS CI"]

                F --> F1["Use estimator-specific asymptotic variance formulas"]
                F1 --> F2["Compute standard error at the selected tuning level"]
                F2 --> F3["Apply Gaussian/Wald approximation"]
                F3 --> F4["Report native asymptotic CI"]

                E8 --> G["Compare xi estimation and uncertainty"]
                F4 --> G

                C6 --> H["Return-level extrapolation<br/>estimate_return_level()"]
                G --> I["Simulation metrics:<br/>APE, coverage, Winkler interval score"]
            ```

            The benchmark is now organized around the truth pair `(xi, theta)`, with `phi` retained only as a derived construction parameter for the simulation families.
            The notebook reads the currently materialized benchmark CSVs and reports the active universal grid directly from those files, so the vignette stays in sync if the benchmark design changes.
            The raw synthetic families are:
            `frechet_max_ar`,
            `moving_maxima_q99`,
            and `pareto_additive_ar1`.
            Each family has a closed-form map from `(xi, theta)` back to its construction parameter `phi`, so the benchmark can compare methods on the same truth grid even though the underlying dependence mechanisms differ.
            Each synthetic series in the main benchmark has length `n_obs = 365`, so the benchmark is explicitly framed as a short-record regime rather than a long-record asymptotic exercise.
            A longer `n_obs = 1000` run can still be generated separately as an appendix sensitivity without changing the main manuscript-facing cache.

            The full benchmark still spans the factorial overview:
            sliding versus disjoint blocks,
            median versus mean versus mode summaries,
            and OLS versus FGLS regression.
            For the main story, the first figure combines six methods:
            `mean-disjoint-OLS`, `mode-disjoint-OLS`, `median-disjoint-OLS`, `median-disjoint-FGLS`, `median-sliding-OLS`, and `median-sliding-FGLS`.
            That single chart shows the baseline target comparison under the simplest disjoint-OLS setup, then isolates what is gained by covariance-aware regression and by sliding blocks for the median target.
            The second figure fixes the proposed sliding/FGLS setup and compares `median`, `mean`, and `mode` targets together with published xi baselines `Hill`, `max-spectrum`, `Pickands`, and `DEdH moment`.
            The internal benchmark tables below report `median APE (IQR) / median Winkler interval score (IQR)` from bootstrap-assisted Wald CI for each internal method.
            The mixed internal-versus-external tables also report `median APE (IQR) / median Winkler interval score (IQR)`, with the understanding that internal methods use bootstrap-assisted Wald intervals while the external baselines use their native asymptotic Gaussian/Wald intervals.
            All interval metrics below use 95% confidence intervals, i.e. `alpha = 0.05` in the Winkler score.
            The main internal figure therefore returns to the simple two-row view:
            absolute percentage error and Winkler interval score.
            Appendix diagnostics below unpack interval quality into width and coverage.
            """
        ),
        code_cell(
            """
            benchmark = pd.read_csv(BENCH / "summary.csv")
            external_benchmark = pd.read_csv(BENCH / "external_summary.csv")
            ei_benchmark = pd.read_csv(BENCH / "ei_summary.csv")
            ei_external_benchmark = pd.read_csv(BENCH / "ei_external_summary.csv")

            def _grid_summary(frame, primary, secondary):
                primary_values = ", ".join(str(value) for value in sorted(frame[primary].dropna().unique()))
                secondary_values = ", ".join(str(value) for value in sorted(frame[secondary].dropna().unique()))
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
            """
        ),
        md_cell(
            """
            ### Appendix: Full Benchmark Overview

            The full twelve-method grid is retained here for completeness, but the main notebook story focuses on the condensed tables and the two comparison figures above.
            """
        ),
        code_cell(
            """
            display(benchmark_table(benchmark, benchmark_set=BENCHMARK_SET))
            """
        ),
        md_cell(
            """
            ### Appendix: 95% Interval Sharpness Versus Calibration

            The main benchmark figures emphasize point-estimation error and Winkler score. This appendix view adds interval sharpness:
            median 95% interval width, median coverage, and median Winkler interval score across the same `xi` grid.
            Lower width is sharper, lower interval score is better overall, and ideal coverage stays close to `0.95`.
            """
        ),
        code_cell(
            """
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
            """
        ),
        md_cell(
            """
            ## 1B. Native EI Benchmark

            The new EI suite mirrors the EVI benchmark structurally, but it swaps the target of comparison:
            `xi` is fixed over the currently materialized EI grid read from the cached benchmark CSVs, while
            `theta` is swept over the corresponding EI dependence grid from the same cache.
            The internal UniBM EI estimators are pooled block-maxima methods built from Northrop or BB reciprocal-EI paths:
            disjoint or sliding blocks,
            OLS or FGLS pooling on `log(1 / theta_hat(b))`,
            and log-scale Wald intervals.
            The external EI baselines are:
            `Ferro-Segers` with Wald CI,
            `K-gaps` with profile-likelihood CI,
            native sliding `Northrop` with adjusted chandwich profile likelihood,
            and native sliding `BB` with Wald CI.

            The targets panel below keeps the selected pooled BM finalists
            `BB-disjoint-FGLS` and `BB-sliding-FGLS`
            and compares them against the four external baselines on the truth scale that matters for extremal clustering.
            As in the EVI benchmark, all interval metrics use 95% confidence intervals with `alpha = 0.05`.
            """
        ),
        code_cell(
            """
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
            """
        ),
        code_cell(
            """
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
            """
        ),
        md_cell(
            """
            ## 2. Dataset Screening

            Candidate applications are screened for record length, defensible preprocessing, stable intermediate-range scaling, and whether sliding block maxima remain informative after preprocessing.
            """
        ),
        code_cell(
            """
            screening = pd.read_csv(OUT / "application_screening.csv")
            series_registry = pd.read_csv(OUT / "application_series_registry.csv")
            application_summary = pd.read_csv(OUT / "application_summary.csv")
            display(screening)
            display(series_registry[["application", "provider", "role", "series_name", "series_basis"]])
            display(application_summary)

            with open(ROOT / "data" / "metadata" / "sources.json") as fh:
                sources = json.load(fh)
            sources
            """
        ),
        md_cell(
            """
            ### What Each Application Actually Does

            The application workflow is intentionally not “one raw series, one estimator” for every case.

            - **Houston precipitation** uses the same wet-season daily precipitation series for display, EVI, and EI, so both tail severity and clustering are interpreted on the daily rainfall calendar.
            - **Texas and Florida streamflow** use the same full-year daily discharge series for display, EVI, and EI, so return levels and extremal-index summaries refer to the same hydrologic process.
            - **Texas and Florida NFIP claims** deliberately split the series by task:
              the display and EI series are zero-filled daily state payout totals on the calendar-day axis,
              while the EVI series keeps only positive-payout days so tail extrapolation is done on the claim-active-day scale.
            - **Phoenix hot-dry severity** is a derived compound-hazard index, retained as a secondary case rather than the main validation dataset.

            This split matters because the application chapter now combines three layers of environmental risk:
            physical hazard occurrence,
            hydrologic response,
            and downstream socio-economic impact.
            """
        ),
        md_cell(
            """
            ### Data Provenance and Source Records

            The application chapter is built from three public data systems:

            - **NOAA GHCN-Daily** station files for Houston precipitation and Phoenix temperature/precipitation inputs.
            - **USGS NWIS daily discharge** for the frozen Texas and Florida streamflow gauges.
            - **OpenFEMA NFIP Redacted Claims v2** for Texas and Florida building-claim payouts.

            The table below exposes the exact station ids, gauge ids, state filters, and source URLs used by the workflow so the case studies can be checked independently.
            """
        ),
        code_cell(
            """
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
            """
        ),
        md_cell(
            """
            ### Manuscript-Facing Application Tables

            The application workflow now emits a matched LaTeX table set for manuscript assembly:
            a cross-application summary table, a return-level table, and an EI-focused comparison table.
            This keeps the application chapter structurally parallel to the benchmark/report pipelines rather than relying on figures alone.
            """
        ),
        md_cell(
            """
            ### Inline Application Plots

            Unlike the benchmark sections, the application plots require fitted bundle objects rather than only summary CSVs.
            The next cell reuses the cached raw inputs and re-fits the six application bundles once so the notebook can call the plotting helpers directly and display the figures inline.
            """
        ),
        code_cell(
            """
            application_bundles = build_application_bundles(DIRS)
            application_bundle_map = {bundle.spec.key: bundle for bundle in application_bundles}
            plot_application_overview(application_bundles)
            plt.show()
            """
        ),
        code_cell(
            """
            application_return_levels = pd.read_csv(OUT / "application_return_levels.csv")
            application_ei_methods = pd.read_csv(OUT / "application_ei_methods.csv")

            display(application_summary)
            display(application_return_levels.head(12))
            display(application_ei_methods)

            print((TABLE_DIR / "application_summary_main.tex").read_text())
            print((TABLE_DIR / "application_return_levels_main.tex").read_text())
            print((TABLE_DIR / "application_ei_main.tex").read_text())
            """
        ),
        code_cell(
            """
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


            def _return_row(app_key, horizon):
                return application_return_levels.loc[
                    (application_return_levels["application"] == app_key)
                    & (application_return_levels["horizon_years"] == float(horizon))
                ].iloc[0]
            """
        ),
        code_cell(
            """
            houston_row = _summary_row("houston_hobby_precipitation")
            phoenix_row = _summary_row("phoenix_hot_dry_severity")
            tx_stream_row = _summary_row("tx_streamflow")
            fl_stream_row = _summary_row("fl_streamflow")
            tx_nfip_row = _summary_row("tx_nfip_claims")
            fl_nfip_row = _summary_row("fl_nfip_claims")

            display(
                Markdown(
                    f'''
            **Cross-application reading guide**

            - **Houston precipitation** is the mildest dependence case: `theta = {_fmt_value(houston_row['theta_hat_bb_sliding_fgls'])}` implies an average cluster size of about `{_fmt_value(houston_row['mean_cluster_size'])}` wet days.
            - **Texas and Florida streamflow** are the strongest clustering cases: BB-sliding-FGLS gives `theta = {_fmt_value(tx_stream_row['theta_hat_bb_sliding_fgls'], digits=3)}` in Texas and `{_fmt_value(fl_stream_row['theta_hat_bb_sliding_fgls'], digits=3)}` in Florida, implying flood-wave cluster sizes around `{_fmt_value(tx_stream_row['mean_cluster_size'])}` and `{_fmt_value(fl_stream_row['mean_cluster_size'])}` days.
            - **Texas and Florida NFIP claims** have the heaviest tails: `xi = {_fmt_value(tx_nfip_row['xi_hat'])}` in Texas and `{_fmt_value(fl_nfip_row['xi_hat'])}` in Florida, with both states showing multi-day claim waves (`theta ~ 0.31`).
            - **Phoenix hot-dry severity** has a lighter tail than Houston (`xi = {_fmt_value(phoenix_row['xi_hat'])}`) but much stronger clustering (`theta = {_fmt_value(phoenix_row['theta_hat_bb_sliding_fgls'])}`), so persistence matters even when the tail is milder.
                    '''
                )
            )
            """
        ),
        *single_application_section(
            heading="""
            ## 3. Flagship Application: Houston Wet-Season Daily Precipitation

            This is the main manuscript-facing application because block maxima are native, return levels are interpretable for flood-related risk, and the record is long enough to study finite-block behavior under serial dependence.
            """,
            application_key="houston_hobby_precipitation",
            variable_prefix="houston",
            include_target=True,
            interpretation_code="""
            houston_evi_screen = _screening_row("houston_hobby_precipitation", "evi")
            houston_ei_screen = _screening_row("houston_hobby_precipitation", "ei")
            houston_rl10 = _return_row("houston_hobby_precipitation", 10.0)
            houston_rl50 = _return_row("houston_hobby_precipitation", 50.0)
            houston_k_gaps = houston_ei.loc[houston_ei["method"] == "k_gaps"].iloc[0]

            display(
                Markdown(
                    f'''
            **Interpretation.** Houston combines a moderately heavy wet-day tail (`xi = {_fmt_interval(houston_summary.iloc[0]['xi_hat'], houston_summary.iloc[0]['xi_lo'], houston_summary.iloc[0]['xi_hi'])}`) with only mild clustering. The EI estimate `theta = {_fmt_interval(houston_summary.iloc[0]['theta_hat_bb_sliding_fgls'], houston_summary.iloc[0]['theta_lo_bb_sliding_fgls'], houston_summary.iloc[0]['theta_hi_bb_sliding_fgls'])}` is close to the K-gaps reference `{_fmt_interval(houston_summary.iloc[0]['theta_hat_k_gaps'], houston_k_gaps['theta_lo'], houston_k_gaps['theta_hi'])}`, implying an average extreme-rainfall cluster size of about `{_fmt_value(houston_summary.iloc[0]['mean_cluster_size'])}` days.

            The screening output also shows why keeping zeros in the EI time axis matters here: only `{100 * houston_evi_screen['daily_positive_share']:.1f}%` of wet-season days are positive, so dry-day spacing is part of the clustering problem rather than noise. Because dependence is modest, EI adjustment only slightly lowers the return levels, from `{_fmt_value(houston_rl10['return_level'])}` to `{_fmt_value(houston_rl10['return_level_ei_adjusted'])}` at 10 years and from `{_fmt_value(houston_rl50['return_level'])}` to `{_fmt_value(houston_rl50['return_level_ei_adjusted'])}` at 50 years.
                    '''
                )
            )
            """,
        ),
        *group_application_section(
            heading="""
            ## 4. Hydrologic Response Applications: Texas and Florida Streamflow

            The streamflow applications move one step downstream from weather into hydrologic response.
            Here the same daily discharge series is used for display, EVI, and EI, so the estimated `xi`, `theta`, and EI-adjusted return levels all describe the same calendar-day river process.
            This makes the streamflow cases the cleanest bridge from the meteorological Houston case to impact-facing insurance data.
            """,
            application_keys=["tx_streamflow", "fl_streamflow"],
            variable_prefix="streamflow",
            include_target=True,
            interpretation_code="""
            tx_stream_rl10 = _return_row("tx_streamflow", 10.0)
            fl_stream_rl10 = _return_row("fl_streamflow", 10.0)
            tx_stream_rl50 = _return_row("tx_streamflow", 50.0)
            fl_stream_rl50 = _return_row("fl_streamflow", 50.0)

            display(
                Markdown(
                    f'''
            **Interpretation.** The two river gauges are the most dependence-dominated applications in the package. BB-sliding-FGLS gives `theta = {_fmt_interval(streamflow_summary.loc[streamflow_summary['application'] == 'tx_streamflow', 'theta_hat_bb_sliding_fgls'].iloc[0], streamflow_summary.loc[streamflow_summary['application'] == 'tx_streamflow', 'theta_lo_bb_sliding_fgls'].iloc[0], streamflow_summary.loc[streamflow_summary['application'] == 'tx_streamflow', 'theta_hi_bb_sliding_fgls'].iloc[0], digits=3)}` in Texas and `{_fmt_interval(streamflow_summary.loc[streamflow_summary['application'] == 'fl_streamflow', 'theta_hat_bb_sliding_fgls'].iloc[0], streamflow_summary.loc[streamflow_summary['application'] == 'fl_streamflow', 'theta_lo_bb_sliding_fgls'].iloc[0], streamflow_summary.loc[streamflow_summary['application'] == 'fl_streamflow', 'theta_hi_bb_sliding_fgls'].iloc[0], digits=3)}` in Florida, corresponding to average flood-wave cluster sizes of roughly `{_fmt_value(streamflow_summary.loc[streamflow_summary['application'] == 'tx_streamflow', 'mean_cluster_size'].iloc[0])}` and `{_fmt_value(streamflow_summary.loc[streamflow_summary['application'] == 'fl_streamflow', 'mean_cluster_size'].iloc[0])}` days.

            That dependence materially changes the extrapolation story. In Texas, the 10-year return level drops from `{_fmt_value(tx_stream_rl10['return_level'])}` to `{_fmt_value(tx_stream_rl10['return_level_ei_adjusted'])}` once EI is accounted for; in Florida, the same adjustment moves the 10-year level from `{_fmt_value(fl_stream_rl10['return_level'])}` to `{_fmt_value(fl_stream_rl10['return_level_ei_adjusted'])}`. The 50-year levels show the same pattern, so these streamflow cases are the clearest demonstration that clustering can dominate return-level interpretation.
                    '''
                )
            )
            """,
        ),
        *group_application_section(
            heading="""
            ## 5. Hazard-to-Impact Applications: Texas and Florida NFIP Building Payout Waves

            The NFIP cases are intentionally constructed differently from the raw physical-hazard series.
            We keep the **calendar-day zero-filled daily payout totals** for display and EI so the clustering analysis preserves claim-wave timing,
            but we fit EVI and return levels on the **positive-payout-day** series so the tail extrapolation is not diluted by long runs of structural zeros.
            The return levels reported here should therefore be read as **claim-active-day return levels**, not calendar-day return levels.
            """,
            application_keys=["tx_nfip_claims", "fl_nfip_claims"],
            variable_prefix="nfip",
            include_registry=True,
            interpretation_code="""
            tx_nfip_evi_screen = _screening_row("tx_nfip_claims", "evi")
            fl_nfip_evi_screen = _screening_row("fl_nfip_claims", "evi")
            tx_nfip_ei_screen = _screening_row("tx_nfip_claims", "ei")
            fl_nfip_ei_screen = _screening_row("fl_nfip_claims", "ei")
            tx_nfip_rl10 = _return_row("tx_nfip_claims", 10.0)
            fl_nfip_rl10 = _return_row("fl_nfip_claims", 10.0)

            display(
                Markdown(
                    f'''
            **Interpretation.** The NFIP applications are the clearest heavy-tail examples in the package. Texas has `xi = {_fmt_interval(nfip_summary.loc[nfip_summary['application'] == 'tx_nfip_claims', 'xi_hat'].iloc[0], nfip_summary.loc[nfip_summary['application'] == 'tx_nfip_claims', 'xi_lo'].iloc[0], nfip_summary.loc[nfip_summary['application'] == 'tx_nfip_claims', 'xi_hi'].iloc[0])}` and Florida has `{_fmt_interval(nfip_summary.loc[nfip_summary['application'] == 'fl_nfip_claims', 'xi_hat'].iloc[0], nfip_summary.loc[nfip_summary['application'] == 'fl_nfip_claims', 'xi_lo'].iloc[0], nfip_summary.loc[nfip_summary['application'] == 'fl_nfip_claims', 'xi_hi'].iloc[0])}`, so the claim-active-day tail is much heavier than in the physical-hazard series.

            The split-series design is also justified by the raw timing structure: only `{100 * tx_nfip_ei_screen['daily_positive_share']:.1f}%` of Texas calendar days and `{100 * fl_nfip_ei_screen['daily_positive_share']:.1f}%` of Florida calendar days carry positive payouts, so EI needs the zero-filled daily axis to preserve claim-wave timing. On that calendar scale, BB-sliding-FGLS gives `theta ~ {_fmt_value(nfip_summary.loc[nfip_summary['application'] == 'tx_nfip_claims', 'theta_hat_bb_sliding_fgls'].iloc[0])}` in Texas and `{_fmt_value(nfip_summary.loc[nfip_summary['application'] == 'fl_nfip_claims', 'theta_hat_bb_sliding_fgls'].iloc[0])}` in Florida, implying mean claim-wave sizes of about `{_fmt_value(nfip_summary.loc[nfip_summary['application'] == 'tx_nfip_claims', 'mean_cluster_size'].iloc[0])}` and `{_fmt_value(nfip_summary.loc[nfip_summary['application'] == 'fl_nfip_claims', 'mean_cluster_size'].iloc[0])}` active days. The 10-year claim-active-day return levels, `{_fmt_value(tx_nfip_rl10['return_level'])}` in Texas and `{_fmt_value(fl_nfip_rl10['return_level'])}` in Florida, should therefore be read together with the EI evidence for multi-day claim waves.
                    '''
                )
            )
            """,
        ),
        *single_application_section(
            heading="""
            ## 6. Secondary Application: Phoenix Hot-Dry Severity

            The secondary case keeps the methodology univariate but modernizes the climate-risk framing.
            The severity index is built from warm-season positive temperature anomalies and rolling precipitation deficits, so annual maxima retain a direct environmental interpretation.
            """,
            application_key="phoenix_hot_dry_severity",
            variable_prefix="phoenix",
            include_target=True,
            interpretation_code="""
            phoenix_rl10 = _return_row("phoenix_hot_dry_severity", 10.0)
            phoenix_rl50 = _return_row("phoenix_hot_dry_severity", 50.0)

            display(
                Markdown(
                    f'''
            **Interpretation.** Phoenix is a useful contrast case because the tail is milder than Houston but the clustering is stronger. The EVI fit `xi = {_fmt_interval(phoenix_summary.iloc[0]['xi_hat'], phoenix_summary.iloc[0]['xi_lo'], phoenix_summary.iloc[0]['xi_hi'])}` is comparatively small, yet the EI estimate `theta = {_fmt_interval(phoenix_summary.iloc[0]['theta_hat_bb_sliding_fgls'], phoenix_summary.iloc[0]['theta_lo_bb_sliding_fgls'], phoenix_summary.iloc[0]['theta_hi_bb_sliding_fgls'])}` implies an average cluster size of about `{_fmt_value(phoenix_summary.iloc[0]['mean_cluster_size'])}` hot-dry days.

            The comparison with Houston is therefore conceptually useful: Houston is more tail-heavy but less persistent, whereas Phoenix is less tail-heavy but more persistent. In Phoenix the EI-adjusted return levels fall from `{_fmt_value(phoenix_rl10['return_level'])}` to `{_fmt_value(phoenix_rl10['return_level_ei_adjusted'])}` at 10 years and from `{_fmt_value(phoenix_rl50['return_level'])}` to `{_fmt_value(phoenix_rl50['return_level_ei_adjusted'])}` at 50 years, so persistence still matters even though the upper tail is relatively mild.
                    '''
                )
            )
            """,
        ),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    root = resolve_repo_dirs()["DIR_WORK"]
    notebook = build_notebook()
    (root / "scripts" / "vignette.ipynb").write_text(json.dumps(notebook, indent=1))


if __name__ == "__main__":
    main()
