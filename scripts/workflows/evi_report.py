"""EVI benchmark aggregation, scoring, visualization, and manuscript artifact emission.

This module consolidates internal EVI benchmark reporting (summary tables, panel
plots) with manuscript-facing artifact generation (LaTeX tables, PDF figures).

Reporting functions
-------------------
benchmark_summary, benchmark_table, benchmark_story_table, benchmark_story_latex,
plot_benchmark_panels

Manuscript functions
--------------------
write_evi_benchmark_manuscript_artifacts, build_evi_benchmark_manuscript_outputs
"""
# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    _STANDALONE_SCRIPT = True
else:
    _STANDALONE_SCRIPT = False

from unibm._runtime import prepare_matplotlib_env

prepare_matplotlib_env("unibm-benchmark")
import matplotlib
from matplotlib.lines import Line2D

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if _STANDALONE_SCRIPT:
    from workflows.benchmark_common import (
        IQR_LOWER,
        IQR_UPPER,
        format_median_iqr,
        quantile_agg,
        render_latex_table,
        wilson_interval,
    )
    from workflows.benchmark_design import (
        BENCHMARK_SET_LABELS,
        BLOCK_LINESTYLES,
        CORE_METHODS,
        FAMILY_LABELS,
        UNIVERSAL_BENCHMARK_SET,
        family_label,
        ordered_families,
        METHOD_LABELS,
        METHOD_LOOKUP,
        METHOD_ORDER,
        METRIC_LABELS,
        REGRESSION_MARKERS,
        TARGET_COLORS,
        method_style,
        sort_by_family_order,
        sort_by_method_order,
    )
else:
    from .benchmark_common import (
        IQR_LOWER,
        IQR_UPPER,
        format_median_iqr,
        quantile_agg,
        render_latex_table,
        wilson_interval,
    )
    from .benchmark_design import (
        BENCHMARK_SET_LABELS,
        BLOCK_LINESTYLES,
        CORE_METHODS,
        FAMILY_LABELS,
        UNIVERSAL_BENCHMARK_SET,
        family_label,
        ordered_families,
        METHOD_LABELS,
        METHOD_LOOKUP,
        METHOD_ORDER,
        METRIC_LABELS,
        REGRESSION_MARKERS,
        TARGET_COLORS,
        method_style,
        sort_by_family_order,
        sort_by_method_order,
    )

# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def benchmark_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate replicate-level benchmark results to the plotting/reporting level."""
    grouped = (
        df.groupby(
            ["benchmark_set", "family", "xi_true", "theta_true", "phi", "method"],
            dropna=False,
            as_index=False,
        )
        .agg(
            n_obs=("n_obs", "median"),
            n_rep=("rep", "nunique"),
            n_cover=("covered", "sum"),
            xi_hat_mean=("xi_hat", "mean"),
            xi_hat_sd=("xi_hat", "std"),
            bias=("signed_error", "mean"),
            mae=("abs_error", "mean"),
            mape=("relative_error", "mean"),
            mape_sd=("relative_error", "std"),
            ape_median=("relative_error", "median"),
            ape_q25=("relative_error", quantile_agg(IQR_LOWER)),
            ape_q75=("relative_error", quantile_agg(IQR_UPPER)),
            interval_width_mean=("interval_width", "mean"),
            interval_width_median=("interval_width", "median"),
            interval_width_q25=("interval_width", quantile_agg(IQR_LOWER)),
            interval_width_q75=("interval_width", quantile_agg(IQR_UPPER)),
            interval_score_mean=("interval_score", "mean"),
            interval_score_median=("interval_score", "median"),
            interval_score_q25=("interval_score", quantile_agg(IQR_LOWER)),
            interval_score_q75=("interval_score", quantile_agg(IQR_UPPER)),
            coverage=("covered", "mean"),
            plateau_lo=("plateau_lo", "median"),
            plateau_hi=("plateau_hi", "median"),
        )
        .reset_index(drop=True)
    )
    grouped["xi_hat_sd"] = grouped["xi_hat_sd"].fillna(0.0)
    grouped["mape_sd"] = grouped["mape_sd"].fillna(0.0)
    grouped["xi_hat_se"] = grouped["xi_hat_sd"] / np.sqrt(grouped["n_rep"])
    grouped["mape_se"] = grouped["mape_sd"] / np.sqrt(grouped["n_rep"])
    grouped["coverage_se"] = np.sqrt(
        grouped["coverage"] * (1 - grouped["coverage"]) / grouped["n_rep"].clip(lower=1)
    )
    wilson_bounds = grouped.apply(
        lambda row: wilson_interval(row["n_cover"], int(row["n_rep"])),
        axis=1,
        result_type="expand",
    )
    grouped["coverage_lo"] = wilson_bounds[0]
    grouped["coverage_hi"] = wilson_bounds[1]
    grouped["method_label"] = grouped["method"].map(METHOD_LABELS)
    grouped["block_scheme"] = grouped["method"].map(
        lambda method: METHOD_LOOKUP[method].block_scheme
    )
    grouped["summary_target"] = grouped["method"].map(
        lambda method: METHOD_LOOKUP[method].summary_target
    )
    grouped["regression"] = grouped["method"].map(lambda method: METHOD_LOOKUP[method].regression)
    grouped["scenario"] = grouped.apply(
        lambda row: (
            f"{row['benchmark_set']}_{row['family']}_xi{row['xi_true']:.2f}"
            f"_theta{row['theta_true']:.2f}"
        ),
        axis=1,
    )
    return sort_by_method_order(grouped)


_EVI_METRIC_Y_UPPER_STEPS = {
    "ape": (1.05, 1.25, 1.5, 2.0, 3.0, 5.0),
    "interval_score": (
        5.0,
        10.0,
        20.0,
        25.0,
        30.0,
        50.0,
        75.0,
        100.0,
        150.0,
        200.0,
        250.0,
        300.0,
        400.0,
        500.0,
    ),
}


def _round_up_metric_upper(metric: str, value: float) -> float:
    """Round one metric upper bound to a stable manuscript-friendly display scale."""
    steps = _EVI_METRIC_Y_UPPER_STEPS.get(metric)
    if steps is None or not np.isfinite(value):
        return float(value)
    padded = max(float(value) * 1.02, steps[0])
    for step in steps:
        if padded <= step:
            return float(step)
    return float(steps[-1])


def _panel_metric_ylim(
    frame: pd.DataFrame,
    *,
    metric: str,
    methods: Iterable[str],
) -> tuple[float, float] | None:
    """Choose a row-wise y-limit that keeps the plotted UniBM methods fully visible."""
    method_list = [method for method in methods if method in frame["method"].unique()]
    if not method_list:
        return None
    _, _, upper_col = _metric_columns(metric)
    value_col = upper_col if upper_col is not None else _metric_columns(metric)[0]
    values = frame.loc[frame["method"].isin(method_list), value_col].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return (0.0, _round_up_metric_upper(metric, float(np.max(finite))))


def benchmark_table(summary: pd.DataFrame, *, benchmark_set: str | None = None) -> pd.DataFrame:
    """Return the long-form benchmark table used in the notebook appendix."""
    columns = [
        "benchmark_set",
        "family",
        "theta_true",
        "phi",
        "xi_true",
        "summary_target",
        "block_scheme",
        "regression",
        "method",
        "method_label",
        "xi_hat_mean",
        "mae",
        "mape",
        "ape_median",
        "ape_q25",
        "ape_q75",
        "interval_width_mean",
        "interval_width_median",
        "interval_width_q25",
        "interval_width_q75",
        "interval_score_mean",
        "interval_score_median",
        "interval_score_q25",
        "interval_score_q75",
        "coverage",
        "coverage_lo",
        "coverage_hi",
        "plateau_lo",
        "plateau_hi",
    ]
    subset = summary.loc[:, columns].copy()
    if benchmark_set is not None:
        subset = subset.loc[subset["benchmark_set"] == benchmark_set]
    return sort_by_method_order(subset)


def benchmark_story_table(
    summary: pd.DataFrame,
    *,
    methods: Iterable[str],
    benchmark_set: str = UNIVERSAL_BENCHMARK_SET,
) -> pd.DataFrame:
    """Summarize the benchmark story into one compact LaTeX-friendly table."""
    subset = summary.loc[summary["benchmark_set"] == benchmark_set].copy()
    methods = [method for method in methods if method in subset["method"].unique()]
    label_order = [METHOD_LABELS[method] for method in methods]
    subset = subset[subset["method"].isin(methods)]
    aggregated = subset.groupby(
        ["family", "theta_true", "method", "method_label"],
        as_index=False,
        dropna=False,
        observed=True,
    ).agg(
        median_ape=("ape_median", "median"),
        ape_q25=("ape_median", quantile_agg(IQR_LOWER)),
        ape_q75=("ape_median", quantile_agg(IQR_UPPER)),
        median_interval_score=("interval_score_median", "median"),
        interval_score_q25=("interval_score_median", quantile_agg(IQR_LOWER)),
        interval_score_q75=("interval_score_median", quantile_agg(IQR_UPPER)),
    )
    aggregated["summary_cell"] = aggregated.apply(
        lambda row: (
            f"{format_median_iqr(row['median_ape'], row['ape_q25'], row['ape_q75'])} / "
            f"{format_median_iqr(row['median_interval_score'], row['interval_score_q25'], row['interval_score_q75'])}"
        ),
        axis=1,
    )
    table = (
        aggregated.pivot(
            index=["family", "theta_true"],
            columns="method_label",
            values="summary_cell",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    ordered_columns = ["family", "theta_true"] + [
        label for label in label_order if label in table.columns
    ]
    return sort_by_family_order(table.loc[:, ordered_columns], sort_columns=["theta_true"])


def benchmark_story_latex(
    summary: pd.DataFrame,
    *,
    methods: Iterable[str],
    benchmark_set: str = UNIVERSAL_BENCHMARK_SET,
    caption: str,
    label: str,
) -> str:
    """Render the compact benchmark story table without depending on notebook tooling."""
    table = benchmark_story_table(summary, methods=methods, benchmark_set=benchmark_set)
    return render_latex_table(table, caption=caption, label=label)


# ---------------------------------------------------------------------------
# Panel plot helpers
# ---------------------------------------------------------------------------


def _metric_columns(metric: str) -> tuple[str, str | None, str | None]:
    metric_map = {
        "ape": ("ape_median", "ape_q25", "ape_q75"),
        "coverage": ("coverage", "coverage_lo", "coverage_hi"),
        "interval_score": ("interval_score_median", "interval_score_q25", "interval_score_q75"),
        "mape": ("mape", None, None),
    }
    return metric_map[metric]


def _stress_cutoff(subset: pd.DataFrame) -> float | None:
    benchmark_sets = subset["benchmark_set"].drop_duplicates().tolist()
    if len(benchmark_sets) <= 1 or not {"main", "stress"}.issubset(set(benchmark_sets)):
        return None
    main_max = subset.loc[subset["benchmark_set"] == "main", "xi_true"].max()
    stress_min = subset.loc[subset["benchmark_set"] == "stress", "xi_true"].min()
    if np.isfinite(main_max) and np.isfinite(stress_min) and main_max < stress_min:
        return float(np.sqrt(main_max * stress_min))
    return None


def _x_multipliers(methods: list[str], interval_style: str) -> dict[str, float]:
    if interval_style != "errorbar" or len(methods) <= 1:
        return {method: 1.0 for method in methods}
    offset_powers = np.linspace(-0.035, 0.035, len(methods))
    return {method: float(10**power) for method, power in zip(methods, offset_powers, strict=True)}


def _add_explicit_legend(fig: plt.Figure, methods: list[str], available_methods: set[str]) -> None:
    handles = []
    for method in methods:
        if method not in available_methods:
            continue
        style = method_style(method)
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=6,
                lw=1.5,
                markerfacecolor=style["markerfacecolor"],
                markeredgecolor=style["markeredgecolor"],
                label=METHOD_LABELS[method],
            )
        )
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=min(3, max(1, len(handles))),
        frameon=False,
        fontsize=9,
        columnspacing=1.2,
        handletextpad=0.5,
    )


def _add_grouped_legends(fig: plt.Figure, subset: pd.DataFrame) -> None:
    target_handles = [
        Line2D([0], [0], color=TARGET_COLORS[target], lw=2.0, label=target)
        for target in ["median", "mean", "mode"]
        if target in subset["summary_target"].unique()
    ]
    block_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            lw=2.0,
            linestyle=BLOCK_LINESTYLES[scheme],
            label=scheme,
        )
        for scheme in ["sliding", "disjoint"]
        if scheme in subset["block_scheme"].unique()
    ]
    regression_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            marker=REGRESSION_MARKERS[regression],
            markersize=6,
            linestyle="None",
            markerfacecolor="0.25" if regression == "FGLS" else "white",
            markeredgecolor="0.25",
            label=regression,
        )
        for regression in ["FGLS", "OLS"]
        if regression in subset["regression"].unique()
    ]
    target_legend = fig.legend(
        target_handles,
        [handle.get_label() for handle in target_handles],
        title="target",
        loc="lower center",
        bbox_to_anchor=(0.2, 0.012),
        ncol=max(1, len(target_handles)),
        frameon=False,
        fontsize=9,
        title_fontsize=9,
    )
    fig.add_artist(target_legend)
    block_legend = fig.legend(
        block_handles,
        [handle.get_label() for handle in block_handles],
        title="blocks",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=max(1, len(block_handles)),
        frameon=False,
        fontsize=9,
        title_fontsize=9,
    )
    fig.add_artist(block_legend)
    fig.legend(
        regression_handles,
        [handle.get_label() for handle in regression_handles],
        title="fit",
        loc="lower center",
        bbox_to_anchor=(0.8, 0.012),
        ncol=max(1, len(regression_handles)),
        frameon=False,
        fontsize=9,
        title_fontsize=9,
    )


def plot_benchmark_panels(
    summary: pd.DataFrame,
    *,
    benchmark_set: str | None = None,
    families: Iterable[str] | None = None,
    methods: Iterable[str] = METHOD_ORDER,
    metrics: Iterable[str] = ("ape", "interval_score"),
    file_path: Path | None = None,
    dpi: int = 600,
    title: str | None = None,
    band_alpha: float = 0.08,
    legend_mode: str = "grouped",
    interval_style: str = "band",
    save: bool = False,
) -> None:
    """Plot benchmark metrics against xi using shared bootstrap uncertainty bars.

    The errorbar mode offsets methods slightly on the x-axis so empirical IQR
    whiskers remain visible even when several methods share nearly identical curves.
    """
    subset = (
        summary.copy()
        if benchmark_set is None
        else summary[summary["benchmark_set"] == benchmark_set].copy()
    )
    if subset.empty:
        raise ValueError(f"No benchmark rows found for benchmark_set={benchmark_set!r}.")
    if families is None:
        families = ordered_families(subset["family"].drop_duplicates().tolist())
    else:
        families = ordered_families(families)
    metrics = list(metrics)
    methods = [method for method in methods if method in subset["method"].unique()]
    theta_values = sorted(subset["theta_true"].drop_duplicates().tolist())
    nrows = len(families) * len(metrics)
    ncols = len(theta_values)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(3.9 * ncols, 2.8 * nrows),
        dpi=dpi,
        sharex=True,
        sharey="row",
    )
    axes = np.asarray(axes, dtype=object).reshape(nrows, ncols)
    stress_cutoff = _stress_cutoff(subset)
    x_multipliers = _x_multipliers(methods, interval_style)
    for family_idx, family in enumerate(families):
        family_frame = subset[subset["family"] == family]
        for metric_idx, metric in enumerate(metrics):
            row_idx = family_idx * len(metrics) + metric_idx
            center_col, lower_col, upper_col = _metric_columns(metric)
            ylim = _panel_metric_ylim(family_frame, metric=metric, methods=methods)
            for col_idx, theta in enumerate(theta_values):
                ax = axes[row_idx, col_idx]
                theta_frame = family_frame[family_frame["theta_true"] == theta]
                for method_idx, method in enumerate(methods):
                    method_frame = theta_frame[theta_frame["method"] == method].sort_values(
                        "xi_true"
                    )
                    if method_frame.empty:
                        continue
                    style = method_style(method)
                    x = method_frame["xi_true"].to_numpy(dtype=float)
                    x_plot = x * x_multipliers[method]
                    y = method_frame[center_col].to_numpy(dtype=float)
                    if lower_col is not None and upper_col is not None:
                        lo = method_frame[lower_col].to_numpy(dtype=float)
                        hi = method_frame[upper_col].to_numpy(dtype=float)
                        if interval_style == "band":
                            ax.fill_between(
                                x_plot,
                                lo,
                                hi,
                                color=style["color"],
                                alpha=band_alpha,
                                linewidth=0,
                            )
                        elif interval_style == "errorbar":
                            yerr = np.vstack(
                                [
                                    np.maximum(y - lo, 0.0),
                                    np.maximum(hi - y, 0.0),
                                ]
                            )
                            ax.errorbar(
                                x_plot,
                                y,
                                yerr=yerr,
                                fmt="none",
                                ecolor=style["color"],
                                elinewidth=0.6,
                                capsize=3,
                                capthick=0.6,
                                alpha=0.9,
                                barsabove=True,
                                zorder=1.0 + 0.2 * method_idx,
                            )
                    ax.plot(
                        x_plot,
                        y,
                        marker=style["marker"],
                        ms=4.2,
                        lw=1.2,
                        linestyle=style["linestyle"],
                        color=style["color"],
                        markerfacecolor=style["markerfacecolor"],
                        markeredgecolor=style["markeredgecolor"],
                        markeredgewidth=0.7,
                        alpha=0.95,
                        zorder=2.0 + 0.2 * method_idx,
                    )
                ax.set_xscale("log")
                if ylim is not None:
                    ax.set_ylim(*ylim)
                if stress_cutoff is not None:
                    ax.axvline(stress_cutoff, color="0.5", linestyle="--", lw=0.8, alpha=0.9)
                ax.grid(alpha=0.25)
                if row_idx == 0:
                    ax.set_title(f"$\\theta$ = {theta:.2f}")
                if col_idx == 0:
                    ax.set_ylabel(f"{family_label(family)}\n{METRIC_LABELS[metric]}")
                if row_idx == nrows - 1:
                    ax.set_xlabel("true $\\xi$")
    if legend_mode == "explicit":
        _add_explicit_legend(fig, methods, set(subset["method"].unique()))
    else:
        _add_grouped_legends(fig, subset)
    if title is None:
        title = (
            "Combined benchmark"
            if benchmark_set is None
            else f"{BENCHMARK_SET_LABELS.get(benchmark_set, benchmark_set)} benchmark"
        )
    fig.suptitle(title, y=0.985)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    if save and file_path is not None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(file_path)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Manuscript artifact emission (absorbed from main_evi_benchmark_manuscript.py)
# ---------------------------------------------------------------------------


def _main_evi_benchmark_n_obs(summary: pd.DataFrame) -> int:
    """Infer the EVI benchmark sample size from the universal summary rows."""
    rows = summary.loc[summary["benchmark_set"] == UNIVERSAL_BENCHMARK_SET, "n_obs"].dropna()
    if rows.empty:
        raise ValueError("Benchmark summary does not contain any universal-benchmark rows.")
    return int(round(float(rows.median())))


def write_evi_benchmark_manuscript_artifacts(
    benchmark_summary_df: pd.DataFrame,
    external_benchmark_summary: pd.DataFrame,
    *,
    fig_dir: Path,
    table_dir: Path,
) -> None:
    """Write EVI benchmark manuscript tables and figures from cached CSV summaries."""
    # Deferred import avoids a circular dependency and keeps this module usable
    # both as `python -m workflows.evi_report` and as a direct script.
    if _STANDALONE_SCRIPT:
        from workflows.evi_benchmark_external import (
            interval_sharpness_story_latex,
            plot_interval_sharpness_scatter,
            plot_target_plus_external_panels,
            target_plus_external_story_latex,
        )
    else:
        from .evi_benchmark_external import (
            interval_sharpness_story_latex,
            plot_interval_sharpness_scatter,
            plot_target_plus_external_panels,
            target_plus_external_story_latex,
        )

    n_obs = _main_evi_benchmark_n_obs(benchmark_summary_df)
    (table_dir / "benchmark_core_main.tex").write_text(
        benchmark_story_latex(
            benchmark_summary_df,
            methods=CORE_METHODS,
            benchmark_set=UNIVERSAL_BENCHMARK_SET,
            caption=(
                f"Necessary-components EVI benchmark on the Universal grid across the xi grid "
                f"0.01 to 10.00 at fixed theta in {{0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 1.00}} "
                f"with n_obs={n_obs}. "
                "Cells report median APE (IQR) / median Winkler interval score (IQR) "
                "summarized over the xi grid. All interval metrics use 95\\% CI "
                "(alpha = 0.05)."
            ),
            label="tab:benchmark-core-main",
        )
    )
    (table_dir / "benchmark_targets_main.tex").write_text(
        target_plus_external_story_latex(
            benchmark_summary_df,
            external_benchmark_summary,
            benchmark_set=UNIVERSAL_BENCHMARK_SET,
            caption=(
                f"Target-comparison EVI benchmark on the Universal grid across the xi grid "
                f"0.01 to 10.00 at fixed theta in {{0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 1.00}} "
                f"with n_obs={n_obs}. "
                "Cells report median APE (IQR) / median interval score (IQR) "
                "summarized over the xi grid. All interval metrics use 95\\% CI "
                "(alpha = 0.05)."
            ),
            label="tab:benchmark-targets-main",
        )
    )
    (table_dir / "benchmark_interval_main.tex").write_text(
        interval_sharpness_story_latex(
            benchmark_summary_df,
            external_benchmark_summary,
            benchmark_set=UNIVERSAL_BENCHMARK_SET,
            caption=(
                f"Appendix interval sharpness-versus-calibration summary on the Universal xi "
                f"grid 0.01 to 10.00 at fixed theta in {{0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 1.00}} "
                f"with n_obs={n_obs}. Cells report median 95\\% interval width / "
                "median coverage / median interval score."
            ),
            label="tab:benchmark-interval-main",
        )
    )
    (table_dir / "benchmark_overview_main.tex").write_text(
        render_latex_table(
            benchmark_table(benchmark_summary_df, benchmark_set=UNIVERSAL_BENCHMARK_SET),
            caption=(
                f"Appendix full EVI benchmark overview on the Universal grid across the xi "
                f"grid 0.01 to 10.00 at fixed theta in {{0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 1.00}} "
                f"with n_obs={n_obs}."
            ),
            label="tab:benchmark-overview-main",
        )
    )
    plot_benchmark_panels(
        benchmark_summary_df,
        benchmark_set=UNIVERSAL_BENCHMARK_SET,
        methods=CORE_METHODS,
        title="Necessary components: from disjoint OLS baselines to sliding-median FGLS",
        legend_mode="explicit",
        interval_style="errorbar",
        file_path=fig_dir / "benchmark_summary.pdf",
        save=True,
    )
    plot_benchmark_panels(
        benchmark_summary_df,
        methods=METHOD_ORDER,
        title="Appendix: full benchmark overview",
        band_alpha=0.04,
        legend_mode="grouped",
        interval_style="errorbar",
        file_path=fig_dir / "benchmark_overview.pdf",
        save=True,
    )
    plot_target_plus_external_panels(
        benchmark_summary_df,
        external_benchmark_summary,
        benchmark_set=UNIVERSAL_BENCHMARK_SET,
        title="Target comparison under sliding-block FGLS",
        file_path=fig_dir / "benchmark_targets.pdf",
        save=True,
    )
    plot_interval_sharpness_scatter(
        benchmark_summary_df,
        external_benchmark_summary,
        benchmark_set=UNIVERSAL_BENCHMARK_SET,
        title="Appendix: 95% interval sharpness versus calibration",
        file_path=fig_dir / "benchmark_interval_sharpness.pdf",
        save=True,
    )


def build_evi_benchmark_manuscript_outputs(root: Path | str = ".") -> dict[str, Path]:
    """Materialize EVI benchmark manuscript figures and LaTeX tables."""
    from config import resolve_repo_dirs

    if _STANDALONE_SCRIPT:
        from workflows.evi_benchmark import load_or_materialize_evi_benchmark_outputs
    else:
        from .evi_benchmark import load_or_materialize_evi_benchmark_outputs

    dirs = resolve_repo_dirs(root)
    fig_dir = dirs["DIR_MANUSCRIPT_FIGURE"]
    table_dir = dirs["DIR_MANUSCRIPT_TABLE"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    benchmark_outputs = load_or_materialize_evi_benchmark_outputs(root, force=False)
    write_evi_benchmark_manuscript_artifacts(
        benchmark_outputs.summary,
        benchmark_outputs.external_summary,
        fig_dir=fig_dir,
        table_dir=table_dir,
    )
    return {
        "benchmark_summary": benchmark_outputs.summary_path,
        "external_benchmark_summary": benchmark_outputs.external_summary_path,
        "benchmark_core_main": table_dir / "benchmark_core_main.tex",
        "benchmark_targets_main": table_dir / "benchmark_targets_main.tex",
        "benchmark_interval_main": table_dir / "benchmark_interval_main.tex",
        "benchmark_overview_main": table_dir / "benchmark_overview_main.tex",
        "benchmark_summary_figure": fig_dir / "benchmark_summary.pdf",
        "benchmark_overview_figure": fig_dir / "benchmark_overview.pdf",
        "benchmark_targets_figure": fig_dir / "benchmark_targets.pdf",
        "benchmark_interval_sharpness_figure": fig_dir / "benchmark_interval_sharpness.pdf",
    }


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main() -> None:
    outputs = build_evi_benchmark_manuscript_outputs()
    for name, path in outputs.items():
        print(f"{name}: {path}")


__all__ = [
    "benchmark_story_latex",
    "benchmark_story_table",
    "benchmark_summary",
    "benchmark_table",
    "build_evi_benchmark_manuscript_outputs",
    "plot_benchmark_panels",
    "write_evi_benchmark_manuscript_artifacts",
]


if __name__ == "__main__":
    main()
