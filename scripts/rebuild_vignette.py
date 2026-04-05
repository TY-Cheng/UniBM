"""Programmatically rebuild the research vignette notebook.

The notebook is treated as a generated artifact so the public story stays in
sync with the current module structure, benchmark figures, and manuscript
exports. Keeping the notebook source here also makes large content changes
reviewable in plain Python rather than raw notebook JSON.
"""

from __future__ import annotations

import json
import textwrap

from config import resolve_repo_dirs


def _cell_lines(text: str) -> list[str]:
    content = textwrap.dedent(text).strip()
    return [line + "\n" for line in content.splitlines()]


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _cell_lines(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _cell_lines(text),
    }


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
            3. Houston wet-season precipitation as the flagship application;
            4. Phoenix hot-dry severity as a modern climate-risk secondary application;
            5. one appendix-style legacy diagnostic retained outside the main paper narrative.
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
            from pathlib import Path
            import json
            import sys

            _CWD = Path.cwd().resolve()
            for _candidate in (_CWD, _CWD.parent):
                _scripts_dir = _candidate / "scripts"
                if (_scripts_dir / "config.py").exists():
                    sys.path.insert(0, str(_scripts_dir))
                    break
            else:
                raise FileNotFoundError("Could not locate scripts/config.py from the current notebook session.")

            from unibm._runtime import prepare_matplotlib_env

            prepare_matplotlib_env("unibm-notebook")
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            from config import resolve_repo_dirs
            from IPython.display import IFrame, display
            from workflows.benchmark_design import CORE_METHODS
            from workflows.evi_benchmark_external import (
                interval_sharpness_story_latex,
                interval_sharpness_story_table,
                plot_interval_sharpness_scatter,
                plot_target_plus_external_panels,
                target_plus_external_story_latex,
                target_plus_external_story_table,
            )
            from workflows.ei_report import (
                ei_core_story_table,
                ei_interval_story_table,
                ei_story_latex,
                ei_targets_story_table,
                plot_ei_core_panels,
                plot_ei_interval_sharpness_scatter,
                plot_ei_targets_panels,
            )
            from workflows.evi_report import (
                benchmark_story_latex,
                benchmark_story_table,
                benchmark_table,
                plot_benchmark_panels,
            )

            DIRS = resolve_repo_dirs()
            ROOT = DIRS["DIR_WORK"]
            OUT = DIRS["DIR_OUT_APPLICATIONS"]
            BENCH = DIRS["DIR_OUT_BENCHMARK"]
            FIG_DIR = DIRS["DIR_MANUSCRIPT_FIGURE"]
            """
        ),
        code_cell(
            """
            REBUILD_OUTPUTS = False
            REQUIRED_OUTPUTS = [
                OUT / "application_screening.csv",
                OUT / "application_summary.csv",
                OUT / "application_methods.csv",
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
                from workflows.application import build_application_outputs
                from workflows.evi_report import build_evi_benchmark_manuscript_outputs
                from workflows.ei_report import build_ei_benchmark_manuscript_outputs

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
            from data_prep.ghcn import prepare_hot_dry_series, prepare_precipitation_series
            from workflows.application_screening import screen_extreme_series
            from unibm import (
                estimate_extremal_index_reciprocal,
                estimate_evi_quantile,
                estimate_return_level,
                plot_scaling_fit,
            )
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
            The EVI suite fixes `theta in {1.00, 0.70, 0.50, 0.35}` and sweeps
            `xi in {0.10, 0.20, 0.50, 1.0, 2.0, 3.0, 5.0, 10.0}`.
            The raw synthetic families are:
            `frechet_max_ar`,
            `moving_maxima_q2`,
            and `pareto_additive_ar1`.
            The shared lower endpoint `theta = 0.35` is chosen because the moving-maxima family with `q = 2` cannot go below `1/3`.
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

            core_table = benchmark_story_table(benchmark, methods=CORE_METHODS, benchmark_set="main")
            target_table = target_plus_external_story_table(
                benchmark,
                external_benchmark,
                benchmark_set="main",
            )

            display(core_table)
            print(
                benchmark_story_latex(
                    benchmark,
                    methods=CORE_METHODS,
                    benchmark_set="main",
                    caption="Core EVI benchmark comparison across the xi grid 0.10 to 10.00 at fixed theta in {1.00, 0.70, 0.50, 0.35}. Cells report median APE (IQR) / median Winkler interval score (IQR). All interval metrics use 95% CI (alpha = 0.05).",
                    label="tab:vignette-benchmark-core",
                )
            )

            display(target_table)
            print(
                target_plus_external_story_latex(
                    benchmark,
                    external_benchmark,
                    benchmark_set="main",
                    caption="Target-comparison EVI benchmark across the xi grid 0.10 to 10.00 at fixed theta in {1.00, 0.70, 0.50, 0.35}. Cells report median APE (IQR) / median Winkler interval score (IQR). All interval metrics use 95% CI (alpha = 0.05).",
                    label="tab:vignette-benchmark-targets",
                )
            )

            plot_benchmark_panels(
                benchmark,
                benchmark_set="main",
                methods=CORE_METHODS,
                title="Necessary components: from disjoint OLS baselines to sliding-median FGLS",
                legend_mode="explicit",
                interval_style="errorbar",
            )
            plt.show()

            plot_target_plus_external_panels(
                benchmark,
                external_benchmark,
                benchmark_set="main",
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
            display(IFrame(str(FIG_DIR / "benchmark_overview.pdf"), width="100%", height=1180))
            display(benchmark_table(benchmark, benchmark_set="main"))
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
                benchmark_set="main",
            )
            display(sharpness_table)
            print(
                interval_sharpness_story_latex(
                    benchmark,
                    external_benchmark,
                    benchmark_set="main",
                    caption="Appendix EVI interval sharpness-versus-calibration summary across the xi grid 0.10 to 10.00 at fixed theta in {1.00, 0.70, 0.50, 0.35}. All interval metrics use 95% CI (alpha = 0.05).",
                    label="tab:vignette-benchmark-interval",
                )
            )

            plot_interval_sharpness_scatter(
                benchmark,
                external_benchmark,
                benchmark_set="main",
                title="Appendix: 95% interval sharpness versus calibration",
            )
            plt.show()
            display(IFrame(str(FIG_DIR / "benchmark_interval_sharpness.pdf"), width="100%", height=520))
            """
        ),
        md_cell(
            """
            ## 1B. Native EI Benchmark

            The new EI suite mirrors the EVI benchmark structurally, but it swaps the target of comparison:
            `xi` is fixed at representative values `0.50, 1.0, 5.0, 10.0`, while
            `theta` is swept over `1.00, 0.85, 0.70, 0.55, 0.45, 0.35`.
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
            ei_core_table = ei_core_story_table(ei_benchmark, benchmark_set="main")
            ei_target_table = ei_targets_story_table(
                ei_benchmark,
                ei_external_benchmark,
                benchmark_set="main",
            )
            ei_interval_table = ei_interval_story_table(
                ei_benchmark,
                ei_external_benchmark,
                benchmark_set="main",
            )

            display(ei_core_table)
            print(
                ei_story_latex(
                    ei_core_table,
                    caption="EI core benchmark across the theta grid 0.35 to 1.00 at fixed xi in {0.50, 1.0, 5.0, 10.0}. Cells report median APE (IQR) / median Winkler interval score (IQR). All interval metrics use 95% CI (alpha = 0.05).",
                    label="tab:vignette-benchmark-ei-core",
                )
            )

            display(ei_target_table)
            print(
                ei_story_latex(
                    ei_target_table,
                    caption="EI target benchmark across the theta grid 0.35 to 1.00 at fixed xi in {0.50, 1.0, 5.0, 10.0}. Cells report median APE (IQR) / median Winkler interval score (IQR). All interval metrics use 95% CI (alpha = 0.05).",
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

            display(IFrame(str(FIG_DIR / "benchmark_ei_summary.pdf"), width="100%", height=580))
            display(IFrame(str(FIG_DIR / "benchmark_ei_targets.pdf"), width="100%", height=580))
            display(IFrame(str(FIG_DIR / "benchmark_ei_interval_sharpness.pdf"), width="100%", height=580))
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
            application_summary = pd.read_csv(OUT / "application_summary.csv")
            display(screening)
            display(application_summary)

            with open(ROOT / "data" / "metadata" / "sources.json") as fh:
                sources = json.load(fh)
            sources
            """
        ),
        md_cell(
            """
            ## 3. Flagship Application: Houston Wet-Season Daily Precipitation

            This is the main manuscript-facing application because block maxima are native, return levels are interpretable for flood-related risk, and the record is long enough to study finite-block behavior under serial dependence.
            """
        ),
        code_cell(
            """
            houston = prepare_precipitation_series(ROOT / "data" / "raw" / "ghcn" / "USW00012918.csv.gz")
            houston_fit = estimate_evi_quantile(
                houston.series.values,
                quantile=0.5,
                sliding=True,
                bootstrap_reps=120,
                random_state=7,
            )
            houston_levels = estimate_return_level(
                houston_fit,
                years=np.array([1, 10, 25, 50]),
                observations_per_year=183.0,
            )
            houston_ei = estimate_extremal_index_reciprocal(houston.series)
            houston_screen = screen_extreme_series(houston.series, name="houston_hobby_precipitation")

            display(pd.Series(houston_screen.to_record()))
            display(
                pd.DataFrame(
                    {
                        "return_horizon_years": [1, 10, 25, 50],
                        "median_block_maximum_mm": houston_levels,
                    }
                )
            )

            fig, axes = plt.subplots(2, 1, figsize=(8, 5))
            axes[0].plot(houston.series.index, houston.series.values, color="tab:blue", lw=0.6)
            axes[0].set_ylabel("precipitation (mm)")
            axes[0].set_title("Houston wet-season daily precipitation")
            axes[0].grid(alpha=0.25)
            axes[1].plot(houston.annual_maxima.index, houston.annual_maxima.values, color="tab:red", lw=0.9)
            axes[1].set_xlabel("year")
            axes[1].set_ylabel("annual max (mm)")
            axes[1].grid(alpha=0.25)
            fig.tight_layout()
            plt.show()
            """
        ),
        code_cell(
            """
            plot_scaling_fit(
                houston_fit,
                title="Houston sliding-block quantile scaling",
                ylabel="log median block maximum",
            )
            plt.show()

            houston_methods = pd.read_csv(OUT / "application_methods.csv")
            display(houston_methods[houston_methods["application"] == "houston_hobby_precipitation"])
            pd.Series(
                {
                    "northrop_reciprocal_ei": houston_ei.northrop_estimate,
                    "bb_reciprocal_ei": houston_ei.bb_estimate,
                }
            )
            """
        ),
        md_cell(
            """
            ## 4. Secondary Application: Phoenix Hot-Dry Severity

            The secondary case keeps the methodology univariate but modernizes the climate-risk framing.
            The severity index is built from warm-season positive temperature anomalies and rolling precipitation deficits, so annual maxima retain a direct environmental interpretation.
            """
        ),
        code_cell(
            """
            phoenix = prepare_hot_dry_series(ROOT / "data" / "raw" / "ghcn" / "USW00023183.csv.gz")
            phoenix_fit = estimate_evi_quantile(
                phoenix.series.values,
                quantile=0.5,
                sliding=True,
                bootstrap_reps=120,
                random_state=7,
            )
            phoenix_levels = estimate_return_level(
                phoenix_fit,
                years=np.array([1, 10, 25, 50]),
                observations_per_year=214.0,
            )
            phoenix_ei = estimate_extremal_index_reciprocal(phoenix.series)
            phoenix_screen = screen_extreme_series(phoenix.series, name="phoenix_hot_dry_severity")

            display(pd.Series(phoenix_screen.to_record()))
            display(
                pd.DataFrame(
                    {
                        "return_horizon_years": [1, 10, 25, 50],
                        "median_block_maximum_severity": phoenix_levels,
                    }
                )
            )

            fig, axes = plt.subplots(2, 1, figsize=(8, 5))
            axes[0].plot(phoenix.series.index, phoenix.series.values, color="tab:orange", lw=0.6)
            axes[0].set_ylabel("severity")
            axes[0].set_title("Phoenix warm-season hot-dry severity")
            axes[0].grid(alpha=0.25)
            axes[1].plot(phoenix.annual_maxima.index, phoenix.annual_maxima.values, color="tab:red", lw=0.9)
            axes[1].set_xlabel("year")
            axes[1].set_ylabel("annual max severity")
            axes[1].grid(alpha=0.25)
            fig.tight_layout()
            plt.show()
            """
        ),
        code_cell(
            """
            plot_scaling_fit(
                phoenix_fit,
                title="Phoenix sliding-block quantile scaling",
                ylabel="log median block maximum",
            )
            plt.show()

            phoenix_methods = pd.read_csv(OUT / "application_methods.csv")
            display(phoenix_methods[phoenix_methods["application"] == "phoenix_hot_dry_severity"])
            pd.Series(
                {
                    "northrop_reciprocal_ei": phoenix_ei.northrop_estimate,
                    "bb_reciprocal_ei": phoenix_ei.bb_estimate,
                }
            )
            """
        ),
        md_cell(
            """
            ## Appendix-Style Legacy Diagnostic

            The earlier cryosphere examples are no longer part of the manuscript’s main pitch, but they remain useful as supplementary demonstrations of the screening and clustering diagnostics.
            """
        ),
        code_cell(
            """
            df_snowcover = pd.read_pickle(ROOT / "data" / "df_snowcover.pkl.gz")
            greenland = np.log(df_snowcover["Greenland"].max()) - np.log(df_snowcover["Greenland"])
            greenland = greenland.dropna()
            greenland_screen = screen_extreme_series(
                greenland,
                name="greenland_land_log_loss",
                min_xi_lower=-1.0,
            )
            greenland_ei = estimate_extremal_index_reciprocal(greenland)

            display(pd.Series(greenland_screen.to_record()))
            pd.Series(
                {
                    "northrop_reciprocal_ei": greenland_ei.northrop_estimate,
                    "bb_reciprocal_ei": greenland_ei.bb_estimate,
                }
            )
            """
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
