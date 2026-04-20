"""Appendix-focused published EVI xi-estimator comparisons.

These benchmarks reuse the same synthetic series as the main UniBM workflow but
compare against published xi estimators outside the UniBM regression pipeline.
Most are classical raw-sample threshold estimators; max-spectrum is the
closest block-maxima-style published comparator. The benchmark uses each
external estimator's native asymptotic Gaussian/Wald interval.
"""
# ruff: noqa: E402

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Iterable, Literal

from unibm._runtime import prepare_matplotlib_env

prepare_matplotlib_env("unibm-benchmark")
import matplotlib
from matplotlib.lines import Line2D
import sys

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from unibm.evi.spectrum import estimate_max_spectrum_evi
from unibm.evi.tail import (
    ExternalXiEstimate,
    estimate_dedh_moment_evi,
    estimate_hill_evi,
    estimate_pickands_evi,
)

from benchmark.design import (
    ordered_families,
    METHOD_LABELS,
    SimulationConfig,
    TARGET_METHODS,
    UNIVERSAL_BENCHMARK_SET,
    family_label,
    default_simulation_configs,
    load_or_simulate_series_bank,
    method_style,
    resolve_benchmark_workers,
    scenario_random_state,
    sort_by_family_order,
)
from benchmark.common import (
    add_wilson_bounds,
    format_median_iqr,
    IQR_LOWER,
    IQR_UPPER,
    interval_score,
    interval_contains,
    interval_width,
    panel_metric_ylim,
    quantile_agg,
    render_latex_table,
)


PROPOSED_EVI_METHODS = [
    "sliding_median_fgls",
]
EXTERNAL_BASELINE_METHODS = [
    "hill_raw",
    "max_spectrum_raw",
    "pickands_raw",
    "dedh_moment_raw",
]
EXTERNAL_METHOD_ORDER = [*PROPOSED_EVI_METHODS, *EXTERNAL_BASELINE_METHODS]
TARGET_PLUS_EXTERNAL_METHODS = [
    *TARGET_METHODS,
    "hill_raw",
    "max_spectrum_raw",
    "pickands_raw",
    "dedh_moment_raw",
]
INTERVAL_DIAGNOSTIC_METHODS = TARGET_PLUS_EXTERNAL_METHODS
EXTERNAL_METHOD_LABELS = {
    "sliding_median_fgls": "median-sliding-FGLS",
    "hill_raw": "Hill",
    "max_spectrum_raw": "Max-spectrum",
    "pickands_raw": "Pickands",
    "dedh_moment_raw": "DEdH-moment",
}
EXTERNAL_METHOD_COLORS = {
    "sliding_median_fgls": "tab:blue",
    "hill_raw": "tab:orange",
    "max_spectrum_raw": "tab:brown",
    "pickands_raw": "tab:green",
    "dedh_moment_raw": "tab:red",
}
EXTERNAL_METHOD_MARKERS = {
    "sliding_median_fgls": "s",
    "hill_raw": "o",
    "max_spectrum_raw": "P",
    "pickands_raw": "^",
    "dedh_moment_raw": "D",
}
EXTERNAL_METHOD_LINESTYLES = {
    "sliding_median_fgls": "-",
    "hill_raw": "--",
    "max_spectrum_raw": "--",
    "pickands_raw": "--",
    "dedh_moment_raw": "--",
}
BENCHMARK_ALPHA = 0.05
EXTERNAL_ESTIMATORS = {
    "hill_raw": estimate_hill_evi,
    "max_spectrum_raw": estimate_max_spectrum_evi,
    "pickands_raw": estimate_pickands_evi,
    "dedh_moment_raw": estimate_dedh_moment_evi,
}

_EXTERNAL_TUNING_AXES = {
    "hill_raw": "k",
    "pickands_raw": "k",
    "dedh_moment_raw": "k",
    "max_spectrum_raw": "scale_start",
}
_EVI_METRIC_Y_UPPER_STEPS = {
    "ape": (1.05, 1.25, 1.5, 2.0, 3.0, 5.0),
    "interval_score": (
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


def _estimator_from_method(method: str):
    """Resolve one external xi estimator from its benchmark method id."""
    return EXTERNAL_ESTIMATORS[method]


def _failed_external_estimate(method: str) -> ExternalXiEstimate:
    """Return a sentinel estimate when one external baseline is undefined."""
    return ExternalXiEstimate(
        method=method,
        xi_hat=float("nan"),
        selected_level=None,
        stable_window=None,
        path_level=(),
        path_xi=(),
        standard_error=float("nan"),
        confidence_interval=(float("nan"), float("nan")),
        ci_method="asymptotic",
        tuning_axis=_EXTERNAL_TUNING_AXES.get(method, "k"),
        fixed_upper_level=None,
    )


def _validate_external_ci_method(ci_method: str) -> None:
    """Validate the external EVI benchmark interval mode."""
    if ci_method == "bootstrap":
        raise NotImplementedError(
            "The external EVI benchmark no longer supports ci_method='bootstrap'; "
            "use the estimators' native asymptotic intervals."
        )
    if ci_method != "asymptotic":
        raise ValueError("ci_method must be 'asymptotic'.")


def _external_result_row(
    cfg: SimulationConfig,
    rep: int,
    estimate: ExternalXiEstimate,
    confidence_interval: tuple[float, float],
) -> dict[str, Any]:
    signed_error = estimate.xi_hat - cfg.xi_true
    abs_error = abs(signed_error)
    ci_lo, ci_hi = confidence_interval
    return {
        "benchmark_set": cfg.benchmark_set,
        "family": cfg.family,
        "scenario": cfg.scenario,
        "rep": rep,
        "n_obs": cfg.n_obs,
        "method": estimate.method,
        "method_label": EXTERNAL_METHOD_LABELS[estimate.method],
        "xi_true": cfg.xi_true,
        "theta_true": cfg.theta_true,
        "phi": cfg.phi,
        "xi_hat": estimate.xi_hat,
        "standard_error": estimate.standard_error,
        "ci_method": "asymptotic",
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "signed_error": signed_error,
        "abs_error": abs_error,
        "relative_error": abs_error / cfg.xi_true,
        "interval_width": interval_width(ci_lo, ci_hi),
        "interval_score": interval_score(cfg.xi_true, ci_lo, ci_hi, alpha=BENCHMARK_ALPHA),
        "covered": interval_contains(confidence_interval, cfg.xi_true)
        if np.all(np.isfinite(confidence_interval))
        else False,
        "selected_k": np.nan if estimate.selected_k is None else float(estimate.selected_k),
        "stable_k_lo": (
            np.nan if estimate.stable_window is None else float(estimate.stable_window.lo)
        ),
        "stable_k_hi": (
            np.nan if estimate.stable_window is None else float(estimate.stable_window.hi)
        ),
        "tuning_axis": estimate.tuning_axis,
        "fixed_upper_level": (
            np.nan if estimate.fixed_upper_level is None else float(estimate.fixed_upper_level)
        ),
    }


def evaluate_external_config(
    cfg: SimulationConfig,
    *,
    random_state: int = 0,
    ci_method: Literal["asymptotic", "bootstrap"] = "asymptotic",
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Evaluate the appendix external xi estimators on one benchmark scenario.

    The external benchmark uses each estimator's native asymptotic Gaussian/Wald
    interval and does not maintain a separate shared-bootstrap sensitivity path.
    """
    _validate_external_ci_method(ci_method)
    rows: list[dict[str, Any]] = []
    series_bank = load_or_simulate_series_bank(
        cfg,
        random_state=random_state,
        cache_dir=cache_dir,
    )
    for rep, vec in enumerate(series_bank):
        estimates: dict[str, ExternalXiEstimate] = {}
        for method, estimator in EXTERNAL_ESTIMATORS.items():
            try:
                estimates[method] = estimator(vec)
            except ValueError:
                estimates[method] = _failed_external_estimate(method)
        for method, estimate in estimates.items():
            rows.append(
                _external_result_row(
                    cfg,
                    rep,
                    estimate,
                    estimate.confidence_interval,
                )
            )
    return pd.DataFrame(rows)


def _evaluate_external_config_worker(
    args: tuple[SimulationConfig, int, str, Path | None],
) -> pd.DataFrame:
    """Process-pool wrapper for one external benchmark scenario."""
    cfg, random_state, ci_method, cache_dir = args
    return evaluate_external_config(
        cfg,
        random_state=random_state,
        ci_method=ci_method,
        cache_dir=cache_dir,
    )


def external_benchmark_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate replicate-level external benchmark results."""
    grouped = (
        df.groupby(
            ["benchmark_set", "family", "xi_true", "theta_true", "phi", "method", "ci_method"],
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
            tuning_axis=("tuning_axis", "first"),
            selected_k=("selected_k", "median"),
            stable_k_lo=("stable_k_lo", "median"),
            stable_k_hi=("stable_k_hi", "median"),
            fixed_upper_level=("fixed_upper_level", "median"),
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
    grouped = add_wilson_bounds(grouped, success_col="n_cover", total_col="n_rep")
    grouped["method_label"] = grouped["method"].map(EXTERNAL_METHOD_LABELS)
    grouped["scenario"] = grouped.apply(
        lambda row: (
            f"{row['benchmark_set']}_{row['family']}_xi{row['xi_true']:.2f}"
            f"_theta{row['theta_true']:.2f}"
        ),
        axis=1,
    )
    grouped["method"] = pd.Categorical(
        grouped["method"],
        categories=EXTERNAL_BASELINE_METHODS,
        ordered=True,
    )
    grouped["family"] = pd.Categorical(
        grouped["family"], categories=ordered_families(grouped["family"]), ordered=True
    )
    return grouped.sort_values(
        ["benchmark_set", "family", "theta_true", "xi_true", "method", "ci_method"]
    ).reset_index(drop=True)


def _summary_columns(*, include_intervals: bool) -> list[str]:
    """Return the minimal summary columns needed for tables or plots."""
    base = [
        "benchmark_set",
        "family",
        "theta_true",
        "phi",
        "xi_true",
        "method",
        "ape_median",
        "interval_score_median",
    ]
    if not include_intervals:
        return base
    return [
        "benchmark_set",
        "family",
        "theta_true",
        "phi",
        "xi_true",
        "method",
        "ape_median",
        "ape_q25",
        "ape_q75",
        "interval_score_mean",
        "interval_score_median",
        "interval_score_q25",
        "interval_score_q75",
        "coverage",
        "coverage_lo",
        "coverage_hi",
        "interval_width_mean",
        "interval_width_median",
        "interval_width_q25",
        "interval_width_q75",
    ]


def _stack_benchmark_summaries(
    internal_summary: pd.DataFrame,
    external_summary: pd.DataFrame,
    *,
    benchmark_set: str,
    methods: Iterable[str],
    include_intervals: bool,
) -> pd.DataFrame:
    """Combine internal and external benchmark summaries on a shared schema."""
    method_list = list(methods)
    columns = _summary_columns(include_intervals=include_intervals)
    internal = internal_summary.loc[
        (internal_summary["benchmark_set"] == benchmark_set)
        & (internal_summary["method"].isin(method_list)),
        columns,
    ].copy()
    external = external_summary.loc[
        (external_summary["benchmark_set"] == benchmark_set)
        & (external_summary["method"].isin(method_list)),
        columns,
    ].copy()
    return pd.concat([internal, external], ignore_index=True)


def external_story_table(
    internal_summary: pd.DataFrame,
    external_summary: pd.DataFrame,
    *,
    benchmark_set: str = UNIVERSAL_BENCHMARK_SET,
    methods: Iterable[str] = EXTERNAL_METHOD_ORDER,
) -> pd.DataFrame:
    """Create a compact appendix table for proposed-vs-external xi comparison."""
    methods = [method for method in methods if method in EXTERNAL_METHOD_ORDER]
    combined = _stack_benchmark_summaries(
        internal_summary,
        external_summary,
        benchmark_set=benchmark_set,
        methods=methods,
        include_intervals=False,
    )
    combined["method_label"] = combined["method"].map(EXTERNAL_METHOD_LABELS)
    aggregated = combined.groupby(
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
        EXTERNAL_METHOD_LABELS[method]
        for method in methods
        if EXTERNAL_METHOD_LABELS[method] in table.columns
    ]
    return sort_by_family_order(table.loc[:, ordered_columns], sort_columns=["theta_true"])


def _render_story_latex(
    table: pd.DataFrame,
    *,
    caption: str,
    label: str,
) -> str:
    """Render one benchmark story table with the shared LaTeX helper."""
    return render_latex_table(table, caption=caption, label=label)


def external_story_latex(
    internal_summary: pd.DataFrame,
    external_summary: pd.DataFrame,
    *,
    benchmark_set: str = UNIVERSAL_BENCHMARK_SET,
    caption: str,
    label: str,
) -> str:
    """Render the appendix external-comparison table as standalone LaTeX."""
    table = external_story_table(
        internal_summary,
        external_summary,
        benchmark_set=benchmark_set,
    )
    return _render_story_latex(table, caption=caption, label=label)


def target_plus_external_story_table(
    internal_summary: pd.DataFrame,
    external_summary: pd.DataFrame,
    *,
    benchmark_set: str = UNIVERSAL_BENCHMARK_SET,
    methods: Iterable[str] = TARGET_PLUS_EXTERNAL_METHODS,
) -> pd.DataFrame:
    """Combine the sliding-FGLS target comparison with external xi baselines."""
    methods = [method for method in methods if method in TARGET_PLUS_EXTERNAL_METHODS]
    combined = _stack_benchmark_summaries(
        internal_summary,
        external_summary,
        benchmark_set=benchmark_set,
        methods=methods,
        include_intervals=False,
    )
    combined["method_label"] = combined["method"].map(_mixed_method_label)
    aggregated = combined.groupby(
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
    ordered_labels = [
        METHOD_LABELS[method] if method in METHOD_LABELS else EXTERNAL_METHOD_LABELS[method]
        for method in methods
    ]
    ordered_columns = ["family", "theta_true"] + [
        label for label in ordered_labels if label in table.columns
    ]
    return sort_by_family_order(table.loc[:, ordered_columns], sort_columns=["theta_true"])


def target_plus_external_story_latex(
    internal_summary: pd.DataFrame,
    external_summary: pd.DataFrame,
    *,
    benchmark_set: str = UNIVERSAL_BENCHMARK_SET,
    caption: str,
    label: str,
) -> str:
    """Render the mixed target-plus-external comparison as standalone LaTeX."""
    table = target_plus_external_story_table(
        internal_summary,
        external_summary,
        benchmark_set=benchmark_set,
    )
    return _render_story_latex(table, caption=caption, label=label)


def interval_sharpness_story_table(
    internal_summary: pd.DataFrame,
    external_summary: pd.DataFrame,
    *,
    benchmark_set: str = UNIVERSAL_BENCHMARK_SET,
    methods: Iterable[str] = INTERVAL_DIAGNOSTIC_METHODS,
) -> pd.DataFrame:
    """Summarize 95% interval sharpness and calibration across the xi grid."""
    methods = [method for method in methods if method in TARGET_PLUS_EXTERNAL_METHODS]
    combined = _stack_benchmark_summaries(
        internal_summary,
        external_summary,
        benchmark_set=benchmark_set,
        methods=methods,
        include_intervals=True,
    )
    combined["method_label"] = combined["method"].map(_mixed_method_label)
    summary = combined.groupby(
        ["family", "theta_true", "method", "method_label"],
        as_index=False,
        dropna=False,
        observed=True,
    ).agg(
        median_interval_width=("interval_width_median", "median"),
        coverage_median=("coverage", "median"),
        median_interval_score=("interval_score_median", "median"),
    )
    summary["method"] = pd.Categorical(summary["method"], categories=methods, ordered=True)
    summary = sort_by_family_order(summary, sort_columns=["theta_true", "method"])
    return summary.loc[
        :,
        [
            "family",
            "theta_true",
            "method_label",
            "median_interval_width",
            "coverage_median",
            "median_interval_score",
        ],
    ]


def interval_sharpness_story_latex(
    internal_summary: pd.DataFrame,
    external_summary: pd.DataFrame,
    *,
    benchmark_set: str = UNIVERSAL_BENCHMARK_SET,
    caption: str,
    label: str,
) -> str:
    """Render the interval sharpness-calibration appendix table as LaTeX."""
    table = interval_sharpness_story_table(
        internal_summary,
        external_summary,
        benchmark_set=benchmark_set,
    ).copy()
    table["median_interval_width"] = table["median_interval_width"].map(lambda x: f"{x:.3f}")
    table["coverage_median"] = table["coverage_median"].map(lambda x: f"{x:.3f}")
    table["median_interval_score"] = table["median_interval_score"].map(lambda x: f"{x:.3f}")
    return _render_story_latex(table, caption=caption, label=label)


_METRIC_COLUMNS = {
    "ape": ("ape_median", "ape_q25", "ape_q75"),
    "interval_score": ("interval_score_median", "interval_score_q25", "interval_score_q75"),
}


def _metric_columns(metric: str) -> tuple[str, str, str]:
    return _METRIC_COLUMNS[metric]


def plot_external_comparison_panels(
    internal_summary: pd.DataFrame,
    external_summary: pd.DataFrame,
    *,
    benchmark_set: str = UNIVERSAL_BENCHMARK_SET,
    methods: Iterable[str] = EXTERNAL_METHOD_ORDER,
    file_path: Path | None = None,
    dpi: int = 600,
    title: str | None = None,
    save: bool = False,
) -> None:
    """Plot appendix xi-comparison curves with interval score above APE."""
    methods = [method for method in methods if method in EXTERNAL_METHOD_ORDER]
    combined = _stack_benchmark_summaries(
        internal_summary,
        external_summary,
        benchmark_set=benchmark_set,
        methods=methods,
        include_intervals=True,
    )
    if combined.empty:
        raise ValueError(f"No external benchmark rows found for benchmark_set={benchmark_set!r}.")

    families = ordered_families(combined["family"].drop_duplicates().tolist())
    theta_values = sorted(combined["theta_true"].drop_duplicates().tolist())
    metrics = ["interval_score", "ape"]
    fig, axes = plt.subplots(
        nrows=len(families) * len(metrics),
        ncols=len(theta_values),
        figsize=(3.9 * len(theta_values), 2.8 * len(families) * len(metrics)),
        dpi=dpi,
        sharex=True,
        sharey="row",
    )
    axes = np.asarray(axes, dtype=object).reshape(len(families) * len(metrics), len(theta_values))
    x_offsets = np.linspace(-0.025, 0.025, len(methods))

    for family_idx, family in enumerate(families):
        family_frame = combined[combined["family"] == family]
        internal_methods = [method for method in methods if method in METHOD_LABELS]
        ylim_methods = internal_methods if internal_methods else list(methods)
        for metric_idx, metric in enumerate(metrics):
            row_idx = family_idx * len(metrics) + metric_idx
            center_col, lower_col, upper_col = _metric_columns(metric)
            ylim = panel_metric_ylim(
                family_frame,
                metric=metric,
                methods=ylim_methods,
                metric_columns=_METRIC_COLUMNS,
                upper_steps=_EVI_METRIC_Y_UPPER_STEPS,
            )
            for col_idx, theta in enumerate(theta_values):
                ax = axes[row_idx, col_idx]
                theta_frame = family_frame[family_frame["theta_true"] == theta]
                for method_idx, method in enumerate(methods):
                    method_frame = theta_frame[theta_frame["method"] == method].sort_values(
                        "xi_true"
                    )
                    if method_frame.empty:
                        continue
                    x = method_frame["xi_true"].to_numpy(dtype=float)
                    x_plot = x * (10 ** x_offsets[method_idx])
                    y = method_frame[center_col].to_numpy(dtype=float)
                    lo = method_frame[lower_col].to_numpy(dtype=float)
                    hi = method_frame[upper_col].to_numpy(dtype=float)
                    yerr = np.vstack([np.maximum(y - lo, 0.0), np.maximum(hi - y, 0.0)])
                    color = EXTERNAL_METHOD_COLORS[method]
                    ax.errorbar(
                        x_plot,
                        y,
                        yerr=yerr,
                        fmt="none",
                        ecolor=color,
                        elinewidth=0.75,
                        capsize=3,
                        capthick=0.75,
                        alpha=0.95,
                        barsabove=True,
                        zorder=1.0 + 0.2 * method_idx,
                    )
                    ax.plot(
                        x_plot,
                        y,
                        color=color,
                        linestyle=EXTERNAL_METHOD_LINESTYLES[method],
                        marker=EXTERNAL_METHOD_MARKERS[method],
                        ms=4.2,
                        lw=1.2,
                        alpha=0.95,
                        zorder=2.0 + 0.2 * method_idx,
                    )
                ax.set_xscale("log")
                if ylim is not None:
                    ax.set_ylim(*ylim)
                ax.grid(alpha=0.25)
                if row_idx == 0:
                    ax.set_title(f"$\\theta$ = {theta:.2f}")
                if col_idx == 0:
                    ylabel = (
                        "absolute percentage error"
                        if metric == "ape"
                        else "Winkler interval score"
                    )
                    ax.set_ylabel(f"{family_label(family)}\n{ylabel}")
                if row_idx == len(families) * len(metrics) - 1:
                    ax.set_xlabel("true $\\xi$")

    handles = [
        Line2D(
            [0],
            [0],
            color=EXTERNAL_METHOD_COLORS[method],
            linestyle=EXTERNAL_METHOD_LINESTYLES[method],
            marker=EXTERNAL_METHOD_MARKERS[method],
            markersize=5.5,
            lw=1.4,
            label=EXTERNAL_METHOD_LABELS[method],
        )
        for method in methods
        if method in combined["method"].unique()
    ]
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=min(4, max(1, len(handles))),
        frameon=False,
        fontsize=9,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    if title is None:
        title = "Appendix: proposed method versus published xi baselines"
    fig.suptitle(title, y=0.985)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    if save and file_path is not None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(file_path)
        plt.close(fig)


def _mixed_method_label(method: str) -> str:
    if method in METHOD_LABELS:
        return METHOD_LABELS[method]
    return EXTERNAL_METHOD_LABELS[method]


def _mixed_method_style(method: str) -> dict[str, Any]:
    if method in METHOD_LABELS:
        return method_style(method)
    return {
        "color": EXTERNAL_METHOD_COLORS[method],
        "linestyle": EXTERNAL_METHOD_LINESTYLES[method],
        "marker": EXTERNAL_METHOD_MARKERS[method],
        "markerfacecolor": "white",
        "markeredgecolor": EXTERNAL_METHOD_COLORS[method],
    }


def plot_target_plus_external_panels(
    internal_summary: pd.DataFrame,
    external_summary: pd.DataFrame,
    *,
    benchmark_set: str = UNIVERSAL_BENCHMARK_SET,
    methods: Iterable[str] = TARGET_PLUS_EXTERNAL_METHODS,
    file_path: Path | None = None,
    dpi: int = 600,
    title: str | None = None,
    save: bool = False,
) -> None:
    """Plot mixed target/external curves with interval score above APE."""
    methods = [method for method in methods if method in TARGET_PLUS_EXTERNAL_METHODS]
    combined = _stack_benchmark_summaries(
        internal_summary,
        external_summary,
        benchmark_set=benchmark_set,
        methods=methods,
        include_intervals=True,
    )
    if combined.empty:
        raise ValueError(
            f"No target/external benchmark rows found for benchmark_set={benchmark_set!r}."
        )

    families = ordered_families(combined["family"].drop_duplicates().tolist())
    theta_values = sorted(combined["theta_true"].drop_duplicates().tolist())
    metrics = ["interval_score", "ape"]
    fig, axes = plt.subplots(
        nrows=len(families) * len(metrics),
        ncols=len(theta_values),
        figsize=(3.9 * len(theta_values), 2.8 * len(families) * len(metrics)),
        dpi=dpi,
        sharex=True,
        sharey="row",
    )
    axes = np.asarray(axes, dtype=object).reshape(len(families) * len(metrics), len(theta_values))
    x_offsets = np.linspace(-0.03, 0.03, len(methods))

    for family_idx, family in enumerate(families):
        family_frame = combined[combined["family"] == family]
        internal_methods = [method for method in methods if method in METHOD_LABELS]
        ylim_methods = internal_methods if internal_methods else list(methods)
        for metric_idx, metric in enumerate(metrics):
            row_idx = family_idx * len(metrics) + metric_idx
            center_col, lower_col, upper_col = _metric_columns(metric)
            ylim = panel_metric_ylim(
                family_frame,
                metric=metric,
                methods=ylim_methods,
                metric_columns=_METRIC_COLUMNS,
                upper_steps=_EVI_METRIC_Y_UPPER_STEPS,
            )
            for col_idx, theta in enumerate(theta_values):
                ax = axes[row_idx, col_idx]
                theta_frame = family_frame[family_frame["theta_true"] == theta]
                for method_idx, method in enumerate(methods):
                    method_frame = theta_frame[theta_frame["method"] == method].sort_values(
                        "xi_true"
                    )
                    if method_frame.empty:
                        continue
                    x = method_frame["xi_true"].to_numpy(dtype=float)
                    x_plot = x * (10 ** x_offsets[method_idx])
                    y = method_frame[center_col].to_numpy(dtype=float)
                    lo = method_frame[lower_col].to_numpy(dtype=float)
                    hi = method_frame[upper_col].to_numpy(dtype=float)
                    yerr = np.vstack([np.maximum(y - lo, 0.0), np.maximum(hi - y, 0.0)])
                    style = _mixed_method_style(method)
                    ax.errorbar(
                        x_plot,
                        y,
                        yerr=yerr,
                        fmt="none",
                        ecolor=style["color"],
                        elinewidth=0.75,
                        capsize=3,
                        capthick=0.75,
                        alpha=0.95,
                        barsabove=True,
                        zorder=1.0 + 0.2 * method_idx,
                    )
                    ax.plot(
                        x_plot,
                        y,
                        color=style["color"],
                        linestyle=style["linestyle"],
                        marker=style["marker"],
                        markerfacecolor=style["markerfacecolor"],
                        markeredgecolor=style["markeredgecolor"],
                        markeredgewidth=0.7,
                        ms=4.2,
                        lw=1.2,
                        alpha=0.95,
                        zorder=2.0 + 0.2 * method_idx,
                    )
                ax.set_xscale("log")
                if ylim is not None:
                    ax.set_ylim(*ylim)
                ax.grid(alpha=0.25)
                if row_idx == 0:
                    ax.set_title(f"$\\theta$ = {theta:.2f}")
                if col_idx == 0:
                    ylabel = (
                        "absolute percentage error"
                        if metric == "ape"
                        else "Winkler interval score"
                    )
                    ax.set_ylabel(f"{family_label(family)}\n{ylabel}")
                if row_idx == len(families) * len(metrics) - 1:
                    ax.set_xlabel("true $\\xi$")

    handles = [
        Line2D(
            [0],
            [0],
            color=_mixed_method_style(method)["color"],
            linestyle=_mixed_method_style(method)["linestyle"],
            marker=_mixed_method_style(method)["marker"],
            markersize=5.5,
            lw=1.4,
            markerfacecolor=_mixed_method_style(method)["markerfacecolor"],
            markeredgecolor=_mixed_method_style(method)["markeredgecolor"],
            label=_mixed_method_label(method),
        )
        for method in methods
        if method in combined["method"].unique()
    ]
    n_legend_cols = min(3, max(1, len(handles)))
    legend_rows = int(np.ceil(len(handles) / n_legend_cols)) if handles else 1
    bottom_margin = 0.035 + 0.022 * legend_rows
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        ncol=n_legend_cols,
        frameon=False,
        fontsize=9,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    if title is None:
        title = "Target comparison under sliding-block FGLS with published xi baselines"
    show_title = bool(title)
    if show_title:
        fig.suptitle(title, y=0.982)
    fig.tight_layout(rect=(0, bottom_margin, 1, 0.95 if show_title else 0.992))
    if save and file_path is not None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(file_path)
        plt.close(fig)


def plot_interval_sharpness_scatter(
    internal_summary: pd.DataFrame,
    external_summary: pd.DataFrame,
    *,
    benchmark_set: str = UNIVERSAL_BENCHMARK_SET,
    methods: Iterable[str] = INTERVAL_DIAGNOSTIC_METHODS,
    file_path: Path | None = None,
    dpi: int = 600,
    title: str | None = None,
    save: bool = False,
) -> None:
    """Plot median 95% interval width against median coverage for mixed methods.

    This is a compact appendix diagnostic: the ideal region is narrow intervals
    with coverage close to the 0.95 reference line.
    """
    table = interval_sharpness_story_table(
        internal_summary,
        external_summary,
        benchmark_set=benchmark_set,
        methods=methods,
    )
    if table.empty:
        raise ValueError(f"No interval-diagnostic rows found for benchmark_set={benchmark_set!r}.")
    families = ordered_families(table["family"].drop_duplicates().tolist())
    theta_values = sorted(table["theta_true"].drop_duplicates().tolist())
    fig, axes = plt.subplots(
        nrows=len(families),
        ncols=len(theta_values),
        figsize=(3.8 * len(theta_values), 3.1 * len(families)),
        dpi=dpi,
        sharex="row",
        sharey=True,
    )
    axes = np.asarray(axes, dtype=object).reshape(len(families), len(theta_values))
    # Keep the legend and panel ordering aligned with the requested mixed method list.
    method_list = [method for method in methods if method in TARGET_PLUS_EXTERNAL_METHODS]

    for row_idx, family in enumerate(families):
        for col_idx, theta in enumerate(theta_values):
            ax = axes[row_idx, col_idx]
            panel = table[(table["family"] == family) & (table["theta_true"] == theta)]
            for method in method_list:
                label = _mixed_method_label(method)
                row = panel[panel["method_label"] == label]
                if row.empty:
                    continue
                style = _mixed_method_style(method)
                width = float(row["median_interval_width"].iloc[0])
                coverage = float(row["coverage_median"].iloc[0])
                ax.scatter(
                    width,
                    coverage,
                    s=36,
                    color=style["color"],
                    marker=style["marker"],
                    facecolors=style["markerfacecolor"],
                    edgecolors=style["markeredgecolor"],
                    linewidths=0.8,
                    zorder=3,
                )
            ax.axhline(0.95, color="black", linestyle=":", lw=1)
            ax.grid(alpha=0.25)
            ax.set_ylim(-0.02, 1.02)
            if row_idx == 0:
                ax.set_title(f"$\\theta$ = {theta:.2f}")
            if col_idx == 0:
                ax.set_ylabel(f"{family_label(family)}\nmedian coverage")
            if row_idx == len(families) - 1:
                ax.set_xlabel("median 95% interval width")

    handles = [
        Line2D(
            [0],
            [0],
            color=_mixed_method_style(method)["color"],
            linestyle="None",
            marker=_mixed_method_style(method)["marker"],
            markersize=6,
            markerfacecolor=_mixed_method_style(method)["markerfacecolor"],
            markeredgecolor=_mixed_method_style(method)["markeredgecolor"],
            label=_mixed_method_label(method),
        )
        for method in method_list
    ]
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(3, max(1, len(handles))),
        frameon=False,
        fontsize=9,
    )
    if title is None:
        title = "Appendix: 95% interval sharpness versus calibration"
    fig.suptitle(title, y=0.985)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    if save and file_path is not None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(file_path)
        plt.close(fig)


def run_external_benchmark(
    *,
    random_state: int = 0,
    configs: list[SimulationConfig] | None = None,
    ci_method: Literal["asymptotic", "bootstrap"] = "asymptotic",
    cache_dir: Path | None = None,
    max_workers: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the appendix external-xi benchmark on the default simulation grid."""
    _validate_external_ci_method(ci_method)
    if configs is None:
        configs = default_simulation_configs()
    workers = resolve_benchmark_workers(len(configs), max_workers=max_workers)
    tasks = [
        (
            cfg,
            scenario_random_state(cfg, master_seed=random_state),
            ci_method,
            cache_dir,
        )
        for cfg in configs
    ]
    if workers == 1:
        frames = [_evaluate_external_config_worker(task) for task in tasks]
    else:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        try:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
                frames = list(executor.map(_evaluate_external_config_worker, tasks, chunksize=1))
        except (OSError, PermissionError):
            frames = [_evaluate_external_config_worker(task) for task in tasks]
    frames = [frame for frame in frames if not frame.empty]
    detail = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary = external_benchmark_summary(detail)
    return detail, summary


__all__ = [
    "EXTERNAL_METHOD_LABELS",
    "EXTERNAL_METHOD_ORDER",
    "INTERVAL_DIAGNOSTIC_METHODS",
    "TARGET_PLUS_EXTERNAL_METHODS",
    "evaluate_external_config",
    "external_benchmark_summary",
    "external_story_latex",
    "external_story_table",
    "interval_sharpness_story_latex",
    "interval_sharpness_story_table",
    "plot_external_comparison_panels",
    "plot_interval_sharpness_scatter",
    "plot_target_plus_external_panels",
    "run_external_benchmark",
    "target_plus_external_story_latex",
    "target_plus_external_story_table",
]
