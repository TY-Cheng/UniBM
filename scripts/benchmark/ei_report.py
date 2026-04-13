"""EI benchmark reporting, tables, plotting, and manuscript artifact emission.

This module consolidates internal EI benchmark reporting (summary tables, panel
plots, interval-sharpness scatter) with manuscript-facing artifact generation
(LaTeX tables, PDF figures).

Reporting functions
-------------------
ei_core_story_table, ei_targets_story_table, ei_interval_story_table,
ei_story_latex, build_ei_shrinkage_sensitivity_summary,
plot_ei_core_panels, plot_ei_targets_panels, plot_ei_overview_panels,
plot_ei_interval_sharpness_scatter, plot_ei_shrinkage_sensitivity

Manuscript functions
--------------------
write_ei_benchmark_manuscript_artifacts, build_ei_benchmark_manuscript_outputs
"""
# ruff: noqa: E402

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable

if __package__ in {None, ""}:
    import importlib.util

    _helper_path = Path(__file__).resolve().parents[1] / "shared" / "import_bootstrap.py"
    _spec = importlib.util.spec_from_file_location("_shared_import_bootstrap", _helper_path)
    if _spec is None or _spec.loader is None:  # pragma: no cover - import bootstrap failure
        raise ImportError(f"Could not load import bootstrap helper from {_helper_path}.")
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    _module.ensure_scripts_on_path_from_entry(__file__)

from unibm._runtime import prepare_matplotlib_env

prepare_matplotlib_env("unibm-benchmark-ei-report")
import matplotlib
from matplotlib.lines import Line2D

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from unibm.ei import (
    EI_ALPHA,
    EI_CI_LEVEL,
    estimate_pooled_bm_ei,
    prepare_ei_bundle,
)

from benchmark.design import (
    UNIVERSAL_BENCHMARK_SET,
    family_label,
    load_or_simulate_series_bank,
    ordered_families,
    scenario_random_state,
    sort_by_family_order,
)
from benchmark.ei_eval import (
    EI_BOOTSTRAP_REPS,
    EI_EXTERNAL_METHODS,
    EI_FGLS_METHODS,
    EI_INTERNAL_METHODS,
    EI_METHOD_COLORS,
    EI_METHOD_LABELS,
    EI_METHOD_LINESTYLES,
    EI_METHOD_MARKERS,
    EI_TARGET_INTERNAL_METHODS,
    _load_or_compute_ei_bootstrap_bundle,
)
from benchmark.common import (
    IQR_LOWER,
    IQR_UPPER,
    format_median_iqr,
    interval_score,
    quantile_agg,
    render_latex_table,
)
from shared.runtime import status

EI_SHRINKAGE_GRID = (0.00, 0.15, 0.35, 0.55, 0.75, 1.00)
EI_SHRINKAGE_METHODS = ("northrop_sliding_fgls", "bb_sliding_fgls")
_EI_SHRINKAGE_REQUIRED_COLUMNS = {
    "benchmark_set",
    "family",
    "n_obs",
    "method",
    "delta",
    "median_ape",
    "median_coverage",
    "median_interval_score",
}

# ---------------------------------------------------------------------------
# Story-table helpers
# ---------------------------------------------------------------------------


def _ei_marker_facecolor(method: str) -> str:
    return EI_METHOD_COLORS[method]


def _ei_marker_edgecolor(method: str) -> str:
    return EI_METHOD_COLORS[method]


def _story_table(
    summary: pd.DataFrame,
    *,
    methods: Iterable[str],
    benchmark_set: str = UNIVERSAL_BENCHMARK_SET,
) -> pd.DataFrame:
    """Collapse a method list into the manuscript-friendly EI story-table layout."""
    methods = [method for method in methods if method in summary["method"].unique()]
    subset = summary.loc[
        (summary["benchmark_set"] == benchmark_set) & (summary["method"].isin(methods))
    ].copy()
    subset["method_label"] = subset["method"].map(EI_METHOD_LABELS)
    aggregated = subset.groupby(
        ["family", "xi_true", "method", "method_label"],
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
            index=["family", "xi_true"], columns="method_label", values="summary_cell"
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    ordered_columns = ["family", "xi_true"] + [
        EI_METHOD_LABELS[method] for method in methods if EI_METHOD_LABELS[method] in table.columns
    ]
    return sort_by_family_order(table.loc[:, ordered_columns], sort_columns=["xi_true"])


def ei_core_story_table(
    summary: pd.DataFrame,
    *,
    benchmark_set: str = UNIVERSAL_BENCHMARK_SET,
) -> pd.DataFrame:
    """Return the core internal pooled-BM EI comparison table."""
    return _story_table(summary, methods=EI_INTERNAL_METHODS, benchmark_set=benchmark_set)


def ei_targets_story_table(
    internal_summary: pd.DataFrame,
    external_summary: pd.DataFrame,
    *,
    benchmark_set: str = UNIVERSAL_BENCHMARK_SET,
) -> pd.DataFrame:
    """Return the mixed pooled-vs-native EI comparison table."""
    combined = pd.concat([internal_summary, external_summary], ignore_index=True)
    methods = [*EI_TARGET_INTERNAL_METHODS, *EI_EXTERNAL_METHODS]
    return _story_table(combined, methods=methods, benchmark_set=benchmark_set)


def ei_interval_story_table(
    internal_summary: pd.DataFrame,
    external_summary: pd.DataFrame,
    *,
    benchmark_set: str = UNIVERSAL_BENCHMARK_SET,
) -> pd.DataFrame:
    """Summarize interval sharpness and coverage across the EI grid."""
    combined = pd.concat([internal_summary, external_summary], ignore_index=True)
    methods = [*EI_FGLS_METHODS, *EI_EXTERNAL_METHODS]
    subset = combined.loc[
        (combined["benchmark_set"] == benchmark_set) & (combined["method"].isin(methods))
    ].copy()
    subset["method_label"] = subset["method"].map(EI_METHOD_LABELS)
    summary = subset.groupby(
        ["family", "xi_true", "method", "method_label"],
        as_index=False,
        dropna=False,
        observed=True,
    ).agg(
        median_interval_width=("interval_width_median", "median"),
        coverage_median=("coverage", "median"),
        median_interval_score=("interval_score_median", "median"),
    )
    summary["method"] = pd.Categorical(summary["method"], categories=methods, ordered=True)
    summary = sort_by_family_order(summary, sort_columns=["xi_true", "method"])
    return summary.loc[
        :,
        [
            "family",
            "xi_true",
            "method_label",
            "median_interval_width",
            "coverage_median",
            "median_interval_score",
        ],
    ]


def ei_story_latex(
    table: pd.DataFrame,
    *,
    caption: str,
    label: str,
) -> str:
    """Render one EI story table as standalone LaTeX."""
    return render_latex_table(table, caption=caption, label=label)


# ---------------------------------------------------------------------------
# Shrinkage-sensitivity helpers
# ---------------------------------------------------------------------------


def _ei_shrinkage_sensitivity_output_path(out_dir: Path) -> Path:
    """Return the canonical appendix CSV for EI shrinkage sensitivity."""
    return out_dir / "benchmark_ei_shrinkage_sensitivity.csv"


def _contains(interval: tuple[float, float], value: float) -> bool:
    """Check whether a nominal interval covers the truth."""
    return bool(interval[0] <= value <= interval[1])


def _ei_method_components(method: str) -> tuple[str, bool]:
    """Decode one pooled-BM EI method id into its path family and block scheme."""
    if method.startswith("northrop_"):
        base_path = "northrop"
    elif method.startswith("bb_"):
        base_path = "bb"
    else:
        raise ValueError(f"Unsupported EI shrinkage method: {method}")
    return base_path, "_sliding_" in method


def _ei_shrinkage_sensitivity_contract_ok(
    summary: pd.DataFrame,
    *,
    configs: Iterable[object],
    deltas: tuple[float, ...],
    methods: tuple[str, ...],
) -> bool:
    """Validate a cached EI shrinkage summary against the requested design."""
    if not _EI_SHRINKAGE_REQUIRED_COLUMNS.issubset(summary.columns):
        return False
    expected_families = sorted({str(cfg.family) for cfg in configs})
    expected_sets = sorted({str(cfg.benchmark_set) for cfg in configs})
    expected_n_obs = sorted({int(cfg.n_obs) for cfg in configs})
    observed_deltas = sorted(float(value) for value in summary["delta"].dropna().unique())
    observed_methods = sorted(str(value) for value in summary["method"].dropna().unique())
    observed_families = sorted(str(value) for value in summary["family"].dropna().unique())
    observed_sets = sorted(str(value) for value in summary["benchmark_set"].dropna().unique())
    observed_n_obs = sorted(int(value) for value in summary["n_obs"].dropna().unique())
    return (
        observed_deltas == list(deltas)
        and observed_methods == sorted(methods)
        and observed_families == expected_families
        and observed_sets == expected_sets
        and observed_n_obs == expected_n_obs
    )


def build_ei_shrinkage_sensitivity_summary(
    root: Path | str = ".",
    *,
    configs: list[object] | None = None,
    deltas: Iterable[float] = EI_SHRINKAGE_GRID,
    methods: Iterable[str] = EI_SHRINKAGE_METHODS,
    force: bool = False,
) -> tuple[pd.DataFrame, Path]:
    """Materialize the appendix EI shrinkage-sensitivity CSV.

    The sensitivity run reuses the cached synthetic series and pooled-BM EI
    bootstrap bundles. Only the covariance-shrinkage value is varied, and only
    for the headline sliding-window pooled-FGLS EI workflows.
    """
    from benchmark.design import default_ei_simulation_configs
    from benchmark.ei_benchmark import EI_BENCHMARK_RANDOM_STATE
    from config import resolve_repo_dirs

    dirs = resolve_repo_dirs(root)
    out_dir = dirs["DIR_OUT_BENCHMARK"]
    cache_dir = dirs["DIR_OUT_BENCHMARK_CACHE"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = _ei_shrinkage_sensitivity_output_path(out_dir)

    resolved_configs = default_ei_simulation_configs() if configs is None else list(configs)
    shrinkage_values = tuple(float(delta) for delta in deltas)
    selected_methods = tuple(str(method) for method in methods)
    if not selected_methods:
        raise ValueError("At least one EI shrinkage method must be requested.")
    unsupported = [method for method in selected_methods if method not in EI_FGLS_METHODS]
    if unsupported:
        joined = ", ".join(unsupported)
        raise ValueError(
            f"EI shrinkage sensitivity only supports pooled-FGLS methods, got: {joined}"
        )

    if not force and output_path.exists():
        cached = pd.read_csv(output_path)
        if _ei_shrinkage_sensitivity_contract_ok(
            cached,
            configs=resolved_configs,
            deltas=shrinkage_values,
            methods=selected_methods,
        ):
            status("ei_report", "reusing cached EI shrinkage-sensitivity CSV")
            return cached, output_path

    status(
        "ei_report",
        "building appendix EI shrinkage sensitivity from cached scenario series",
    )
    detail_rows: list[dict[str, float | int | str]] = []
    for cfg in resolved_configs:
        scenario_seed = scenario_random_state(cfg, master_seed=EI_BENCHMARK_RANDOM_STATE)
        series_bank = load_or_simulate_series_bank(
            cfg,
            random_state=scenario_seed,
            cache_dir=cache_dir,
        )
        for rep, vec in enumerate(series_bank):
            bundle = prepare_ei_bundle(vec)
            cache_key = f"{cfg.scenario}__seed{scenario_seed}__rep{rep:04d}"
            bootstrap_results = _load_or_compute_ei_bootstrap_bundle(
                vec,
                bundle=bundle,
                cache_dir=cache_dir,
                cache_key=cache_key,
                reps=EI_BOOTSTRAP_REPS,
                random_state=scenario_seed + 10_000 * rep,
            )
            for method in selected_methods:
                base_path, sliding = _ei_method_components(method)
                for delta in shrinkage_values:
                    estimate = estimate_pooled_bm_ei(
                        bundle,
                        base_path=base_path,
                        sliding=sliding,
                        regression="FGLS",
                        bootstrap_result=bootstrap_results[(base_path, sliding)],
                        covariance_shrinkage=delta,
                    )
                    ci_lo, ci_hi = estimate.confidence_interval
                    detail_rows.append(
                        {
                            "benchmark_set": cfg.benchmark_set,
                            "family": cfg.family,
                            "n_obs": int(cfg.n_obs),
                            "xi_true": float(cfg.xi_true),
                            "theta_true": float(cfg.theta_true),
                            "phi": float(cfg.phi),
                            "rep": int(rep),
                            "method": method,
                            "delta": float(delta),
                            "ape": float(
                                abs(estimate.theta_hat - cfg.theta_true) / abs(cfg.theta_true)
                            ),
                            "interval_score": float(
                                interval_score(
                                    cfg.theta_true,
                                    ci_lo,
                                    ci_hi,
                                    alpha=EI_ALPHA,
                                )
                            ),
                            "covered": float(_contains((ci_lo, ci_hi), cfg.theta_true)),
                        }
                    )
    detail = pd.DataFrame(detail_rows)
    scenario_summary = (
        detail.groupby(
            [
                "benchmark_set",
                "family",
                "n_obs",
                "xi_true",
                "theta_true",
                "phi",
                "method",
                "delta",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            ape_median=("ape", "median"),
            interval_score_median=("interval_score", "median"),
            coverage=("covered", "mean"),
        )
        .reset_index(drop=True)
    )
    summary = (
        scenario_summary.groupby(
            ["benchmark_set", "family", "n_obs", "method", "delta"],
            as_index=False,
            dropna=False,
        )
        .agg(
            median_ape=("ape_median", "median"),
            median_coverage=("coverage", "median"),
            median_interval_score=("interval_score_median", "median"),
        )
        .reset_index(drop=True)
    )
    summary["method"] = pd.Categorical(
        summary["method"], categories=selected_methods, ordered=True
    )
    summary = sort_by_family_order(summary, sort_columns=["method", "delta"])
    summary.to_csv(output_path, index=False)
    return summary, output_path


def plot_ei_shrinkage_sensitivity(
    summary: pd.DataFrame,
    *,
    benchmark_set: str = UNIVERSAL_BENCHMARK_SET,
    file_path: Path | None = None,
    dpi: int = 600,
    title: str | None = None,
    save: bool = False,
) -> None:
    """Plot appendix EI shrinkage sensitivity against the fixed delta grid."""
    subset = summary.loc[summary["benchmark_set"] == benchmark_set].copy()
    if subset.empty:
        raise ValueError(
            f"No EI shrinkage-sensitivity rows found for benchmark_set={benchmark_set!r}."
        )
    methods = [method for method in EI_SHRINKAGE_METHODS if method in subset["method"].unique()]
    if not methods:
        methods = sorted(str(method) for method in subset["method"].drop_duplicates())
    families = ordered_families(subset["family"].drop_duplicates().tolist())
    metrics = [
        ("median_ape", "median APE"),
        ("median_coverage", "median coverage"),
        ("median_interval_score", "median Winkler interval score"),
    ]
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue"])
    fig, axes = plt.subplots(
        nrows=len(methods),
        ncols=len(metrics),
        figsize=(5.0 * len(metrics), 3.4 * len(methods)),
        dpi=dpi,
        sharex=True,
        squeeze=False,
    )
    for row_idx, method in enumerate(methods):
        method_frame = subset.loc[subset["method"] == method].copy()
        for col_idx, (metric, ylabel) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for family_idx, family in enumerate(families):
                family_frame = method_frame.loc[method_frame["family"] == family].sort_values(
                    "delta"
                )
                if family_frame.empty:
                    continue
                ax.plot(
                    family_frame["delta"].to_numpy(dtype=float),
                    family_frame[metric].to_numpy(dtype=float),
                    marker="o",
                    lw=1.8,
                    color=color_cycle[family_idx % len(color_cycle)],
                    label=family_label(family),
                )
            if metric == "median_coverage":
                ax.axhline(EI_CI_LEVEL, color="0.4", linestyle="--", linewidth=1.0)
                ax.set_ylim(0.0, 1.02)
            else:
                values = method_frame[metric].to_numpy(dtype=float)
                finite = values[np.isfinite(values)]
                if finite.size:
                    ax.set_ylim(0.0, float(np.max(finite) * 1.08))
            if row_idx == 0:
                ax.set_title(ylabel, fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(f"{EI_METHOD_LABELS.get(method, method)}\n{ylabel}")
            else:
                ax.set_ylabel(ylabel)
            ax.set_xlabel(r"FGLS shrinkage $\delta$")
            ax.set_xticks(EI_SHRINKAGE_GRID)
            ax.grid(alpha=0.2, linewidth=0.6)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="best")
    if title:
        fig.suptitle(title, fontsize=13, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.97 if title else 1))
    if save:
        if file_path is None:
            raise ValueError("file_path must be supplied when save=True.")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Panel plot helpers
# ---------------------------------------------------------------------------

_EI_METRIC_Y_UPPER_STEPS = {
    "ape": (1.05, 1.25, 1.5, 2.0, 3.0, 5.0),
    "interval_score": (2.0, 3.0, 5.0, 10.0, 20.0, 25.0, 30.0, 40.0, 50.0),
}


def _round_up_metric_upper(metric: str, value: float) -> float:
    """Round one metric upper bound to a stable display scale."""
    steps = _EI_METRIC_Y_UPPER_STEPS.get(metric)
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
    metric_cols = {
        "ape": ("ape_median", "ape_q25", "ape_q75"),
        "interval_score": ("interval_score_median", "interval_score_q25", "interval_score_q75"),
    }
    _, _, upper_col = metric_cols[metric]
    values = frame.loc[frame["method"].isin(method_list), upper_col].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return (0.0, _round_up_metric_upper(metric, float(np.max(finite))))


def _plot_panels(
    summary: pd.DataFrame,
    *,
    methods: Iterable[str],
    title: str,
    file_path: Path | None = None,
    save: bool = False,
) -> None:
    """Plot panel grids on the theta scale for a chosen EI method subset."""
    methods = [method for method in methods if method in summary["method"].unique()]
    subset = summary.loc[summary["method"].isin(methods)].copy()
    families = ordered_families(subset["family"].drop_duplicates())
    xi_values = sorted(subset["xi_true"].drop_duplicates().tolist())
    theta_values = sorted(subset["theta_true"].drop_duplicates().tolist())
    theta_ticks = np.asarray(theta_values, dtype=float)
    metrics = ["ape", "interval_score"]
    nrows = len(families) * len(metrics)
    ncols = len(xi_values)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.6 * ncols, 3.0 * nrows),
        dpi=600,
        sharex=True,
        sharey="row",
    )
    axes = np.asarray(axes, dtype=object).reshape(nrows, ncols)
    # On a log-theta axis, multiplicative jitter preserves the local geometry of
    # the true theta grid while keeping overlapping methods distinguishable.
    x_offsets = np.linspace(-0.025, 0.025, len(methods))
    x_multipliers = 10**x_offsets
    theta_lo = float(theta_ticks.min() / x_multipliers.max() * 0.98)
    theta_hi = float(theta_ticks.max() * x_multipliers.max() * 1.02)
    metric_cols = {
        "ape": ("ape_median", "ape_q25", "ape_q75"),
        "interval_score": ("interval_score_median", "interval_score_q25", "interval_score_q75"),
    }
    internal_methods = [method for method in methods if method in EI_INTERNAL_METHODS]
    ylim_methods = internal_methods if internal_methods else list(methods)
    for family_idx, family in enumerate(families):
        family_frame = subset[subset["family"] == family]
        for metric_idx, metric in enumerate(metrics):
            row_idx = family_idx * len(metrics) + metric_idx
            center_col, lower_col, upper_col = metric_cols[metric]
            ylim = _panel_metric_ylim(family_frame, metric=metric, methods=ylim_methods)
            for col_idx, xi in enumerate(xi_values):
                ax = axes[row_idx, col_idx]
                xi_frame = family_frame[family_frame["xi_true"] == xi]
                for method_idx, method in enumerate(methods):
                    method_frame = xi_frame[xi_frame["method"] == method].sort_values("theta_true")
                    if method_frame.empty:
                        continue
                    x = method_frame["theta_true"].to_numpy(dtype=float)
                    x_plot = x * x_multipliers[method_idx]
                    y = method_frame[center_col].to_numpy(dtype=float)
                    lo = method_frame[lower_col].to_numpy(dtype=float)
                    hi = method_frame[upper_col].to_numpy(dtype=float)
                    yerr = np.vstack([np.maximum(y - lo, 0.0), np.maximum(hi - y, 0.0)])
                    color = EI_METHOD_COLORS[method]
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
                        zorder=1.0 + 0.2 * method_idx,
                    )
                    ax.plot(
                        x_plot,
                        y,
                        color=color,
                        linestyle=EI_METHOD_LINESTYLES[method],
                        marker=EI_METHOD_MARKERS[method],
                        markerfacecolor=_ei_marker_facecolor(method),
                        markeredgecolor=_ei_marker_edgecolor(method),
                        markeredgewidth=0.7,
                        ms=4.2,
                        lw=1.2,
                        alpha=0.95,
                        zorder=2.0 + 0.2 * method_idx,
                    )
                ax.set_xscale("log")
                ax.xaxis.set_minor_formatter(plt.NullFormatter())
                ax.set_xlim(theta_lo, theta_hi)
                if ylim is not None:
                    ax.set_ylim(*ylim)
                ax.set_xticks(theta_ticks)
                ax.set_xticklabels(
                    [f"{theta:.2f}" for theta in theta_ticks],
                    fontsize=7,
                    rotation=35,
                    ha="right",
                )
                ax.tick_params(axis="x", pad=2)
                ax.grid(alpha=0.25)
                if row_idx == 0:
                    ax.set_title(f"$\\xi$ = {xi:.2f}")
                if col_idx == 0:
                    ylabel = (
                        "absolute percentage error"
                        if metric == "ape"
                        else "Winkler interval score"
                    )
                    ax.set_ylabel(f"{family_label(family)}\n{ylabel}", fontsize=8)
                if row_idx == nrows - 1:
                    ax.set_xlabel("true $\\theta$")
    n_legend_cols = min(4, max(1, len(methods)))
    legend_rows = int(np.ceil(len(methods) / n_legend_cols))
    bottom_margin = 0.04 + 0.025 * legend_rows
    handles = [
        Line2D(
            [0],
            [0],
            color=EI_METHOD_COLORS[method],
            linestyle=EI_METHOD_LINESTYLES[method],
            marker=EI_METHOD_MARKERS[method],
            markersize=5.5,
            lw=1.4,
            markerfacecolor=_ei_marker_facecolor(method),
            markeredgecolor=_ei_marker_edgecolor(method),
            label=EI_METHOD_LABELS[method],
        )
        for method in methods
    ]
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=n_legend_cols,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(title, y=0.99, fontsize=11)
    fig.tight_layout(rect=(0, bottom_margin, 1, 0.96))
    if save and file_path is not None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(file_path)
        plt.close(fig)


def plot_ei_core_panels(
    summary: pd.DataFrame, *, file_path: Path | None = None, save: bool = False
) -> None:
    """Plot the eight internal pooled BM EI methods."""
    _plot_panels(
        summary,
        methods=EI_INTERNAL_METHODS,
        title="EI benchmark: pooled BM methods",
        file_path=file_path,
        save=save,
    )


def plot_ei_targets_panels(
    internal_summary: pd.DataFrame,
    external_summary: pd.DataFrame,
    *,
    file_path: Path | None = None,
    save: bool = False,
) -> None:
    """Plot the mixed EI target comparison."""
    combined = pd.concat([internal_summary, external_summary], ignore_index=True)
    _plot_panels(
        combined,
        methods=[*EI_TARGET_INTERNAL_METHODS, *EI_EXTERNAL_METHODS],
        title="EI benchmark: selected pooled BM methods versus threshold and native sliding comparators",
        file_path=file_path,
        save=save,
    )


def plot_ei_overview_panels(
    internal_summary: pd.DataFrame,
    external_summary: pd.DataFrame,
    *,
    file_path: Path | None = None,
    save: bool = False,
) -> None:
    """Plot the full EI method roster on a common theta-scale panel grid."""
    combined = pd.concat([internal_summary, external_summary], ignore_index=True)
    _plot_panels(
        combined,
        methods=[*EI_INTERNAL_METHODS, *EI_EXTERNAL_METHODS],
        title="EI benchmark: full method overview",
        file_path=file_path,
        save=save,
    )


def plot_ei_interval_sharpness_scatter(
    internal_summary: pd.DataFrame,
    external_summary: pd.DataFrame,
    *,
    file_path: Path | None = None,
    save: bool = False,
) -> None:
    """Plot median interval width against median coverage across the EI grid."""
    table = ei_interval_story_table(internal_summary, external_summary)
    families = ordered_families(table["family"].drop_duplicates())
    xi_values = sorted(table["xi_true"].drop_duplicates().tolist())
    methods = [*EI_FGLS_METHODS, *EI_EXTERNAL_METHODS]
    fig, axes = plt.subplots(
        nrows=len(families),
        ncols=len(xi_values),
        figsize=(3.8 * len(xi_values), 3.1 * len(families)),
        dpi=600,
        sharex="row",
        sharey=True,
    )
    axes = np.asarray(axes, dtype=object).reshape(len(families), len(xi_values))
    for row_idx, family in enumerate(families):
        for col_idx, xi in enumerate(xi_values):
            ax = axes[row_idx, col_idx]
            panel = table[(table["family"] == family) & (table["xi_true"] == xi)]
            for method in methods:
                label = EI_METHOD_LABELS[method]
                row = panel[panel["method_label"] == label]
                if row.empty:
                    continue
                ax.scatter(
                    float(row["median_interval_width"].iloc[0]),
                    float(row["coverage_median"].iloc[0]),
                    s=36,
                    color=EI_METHOD_COLORS[method],
                    marker=EI_METHOD_MARKERS[method],
                    facecolors=_ei_marker_facecolor(method),
                    edgecolors=_ei_marker_edgecolor(method),
                    linewidths=0.8,
                    zorder=3,
                )
            ax.axhline(EI_CI_LEVEL, color="black", linestyle=":", lw=1)
            ax.grid(alpha=0.25)
            ax.set_ylim(-0.02, 1.02)
            if row_idx == 0:
                ax.set_title(f"$\\xi$ = {xi:.2f}")
            if col_idx == 0:
                ax.set_ylabel(f"{family_label(family)}\nmedian coverage")
            if row_idx == len(families) - 1:
                ax.set_xlabel("median 95% interval width")
    handles = [
        Line2D(
            [0],
            [0],
            color=EI_METHOD_COLORS[method],
            linestyle="None",
            marker=EI_METHOD_MARKERS[method],
            markersize=6,
            markerfacecolor=_ei_marker_facecolor(method),
            markeredgecolor=_ei_marker_edgecolor(method),
            label=EI_METHOD_LABELS[method],
        )
        for method in methods
    ]
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(4, max(1, len(handles))),
        frameon=False,
        fontsize=9,
    )
    fig.suptitle("EI benchmark: 95% interval sharpness versus calibration", y=0.985)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    if save and file_path is not None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(file_path)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Manuscript artifact emission (absorbed from main_ei_benchmark_manuscript.py)
# ---------------------------------------------------------------------------


def _main_ei_benchmark_n_obs(summary: pd.DataFrame) -> int:
    rows = summary.loc[summary["benchmark_set"] == UNIVERSAL_BENCHMARK_SET, "n_obs"].dropna()
    if rows.empty:
        raise ValueError("EI benchmark summary does not contain any projected-suite rows.")
    return int(round(float(rows.median())))


def write_ei_benchmark_manuscript_artifacts(
    benchmark_summary: pd.DataFrame,
    external_benchmark_summary: pd.DataFrame,
    *,
    shrinkage_sensitivity_summary: pd.DataFrame | None = None,
    fig_dir: Path,
    table_dir: Path,
) -> None:
    """Write EI benchmark manuscript tables and figures from cached CSV summaries."""
    n_obs = _main_ei_benchmark_n_obs(benchmark_summary)
    (table_dir / "benchmark_ei_core_main.tex").write_text(
        ei_story_latex(
            ei_core_story_table(benchmark_summary, benchmark_set=UNIVERSAL_BENCHMARK_SET),
            caption=(
                f"EI core benchmark on the projected short-record persistence suite with "
                f"theta in {{0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 1.0}}, "
                f"xi in {{0.01, 0.50, 1.0, 5.0}}, and the Frechet max-AR, moving-maxima q=99, "
                f"and Pareto additive AR(1) families, with n_obs={n_obs}. "
                "Cells report median APE (IQR) / median Winkler interval score (IQR) over "
                "the theta grid. All interval metrics use 95\\% CI (alpha = 0.05)."
            ),
            label="tab:benchmark-ei-core-main",
        )
    )
    (table_dir / "benchmark_ei_targets_main.tex").write_text(
        ei_story_latex(
            ei_targets_story_table(
                benchmark_summary,
                external_benchmark_summary,
                benchmark_set=UNIVERSAL_BENCHMARK_SET,
            ),
            caption=(
                f"EI target benchmark on the projected short-record persistence suite with "
                f"theta in {{0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 1.0}}, "
                f"xi in {{0.01, 0.50, 1.0, 5.0}}, and the Frechet max-AR, moving-maxima q=99, "
                f"and Pareto additive AR(1) families, with n_obs={n_obs}. "
                "Cells report median APE (IQR) / median Winkler interval score (IQR) over "
                "the theta grid. All interval metrics use 95\\% CI (alpha = 0.05), but native "
                "interval constructions differ across methods, so the table is descriptive and "
                "is not used to rank cross-class interval calibration."
            ),
            label="tab:benchmark-ei-targets-main",
        )
    )
    interval_table = ei_interval_story_table(
        benchmark_summary,
        external_benchmark_summary,
        benchmark_set=UNIVERSAL_BENCHMARK_SET,
    ).copy()
    interval_table["median_interval_width"] = interval_table["median_interval_width"].map(
        lambda x: f"{x:.3f}"
    )
    interval_table["coverage_median"] = interval_table["coverage_median"].map(lambda x: f"{x:.3f}")
    interval_table["median_interval_score"] = interval_table["median_interval_score"].map(
        lambda x: f"{x:.3f}"
    )
    (table_dir / "benchmark_ei_interval_main.tex").write_text(
        render_latex_table(
            interval_table,
            caption=(
                f"EI interval sharpness-versus-calibration summary on the projected EI suite with "
                f"theta in {{0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 1.0}}, "
                f"xi in {{0.01, 0.50, 1.0, 5.0}}, and the Frechet max-AR, moving-maxima q=99, "
                f"and Pareto additive AR(1) families, with n_obs={n_obs}. "
                "Cells report median 95\\% interval width / "
                "median coverage / median interval score."
            ),
            label="tab:benchmark-ei-interval-main",
        )
    )
    (table_dir / "benchmark_ei_overview_main.tex").write_text(
        render_latex_table(
            benchmark_summary.loc[
                benchmark_summary["benchmark_set"] == UNIVERSAL_BENCHMARK_SET
            ].copy(),
            caption=(
                f"Appendix full EI benchmark overview on the projected EI suite with theta in "
                f"{{0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 1.0}}, xi in {{0.01, 0.50, 1.0, 5.0}}, "
                f"and the Frechet max-AR, moving-maxima q=99, and Pareto additive AR(1) families, "
                f"with n_obs={n_obs}."
            ),
            label="tab:benchmark-ei-overview-main",
        )
    )
    plot_ei_core_panels(
        benchmark_summary,
        file_path=fig_dir / "benchmark_ei_summary.pdf",
        save=True,
    )
    plot_ei_targets_panels(
        benchmark_summary,
        external_benchmark_summary,
        file_path=fig_dir / "benchmark_ei_targets.pdf",
        save=True,
    )
    plot_ei_interval_sharpness_scatter(
        benchmark_summary,
        external_benchmark_summary,
        file_path=fig_dir / "benchmark_ei_interval_sharpness.pdf",
        save=True,
    )
    plot_ei_overview_panels(
        benchmark_summary,
        external_benchmark_summary,
        file_path=fig_dir / "benchmark_ei_overview.pdf",
        save=True,
    )
    if shrinkage_sensitivity_summary is not None:
        plot_ei_shrinkage_sensitivity(
            shrinkage_sensitivity_summary,
            benchmark_set=UNIVERSAL_BENCHMARK_SET,
            title="Appendix: EI shrinkage sensitivity for pooled sliding-FGLS workflows",
            file_path=fig_dir / "benchmark_ei_shrinkage_sensitivity.pdf",
            save=True,
        )


def build_ei_benchmark_manuscript_outputs(root: Path | str = ".") -> dict[str, Path]:
    """Materialize EI benchmark manuscript figures and LaTeX tables."""
    from config import resolve_repo_dirs

    from benchmark.ei_benchmark import load_or_materialize_ei_benchmark_outputs

    dirs = resolve_repo_dirs(root)
    fig_dir = dirs["DIR_MANUSCRIPT_FIGURE"]
    table_dir = dirs["DIR_MANUSCRIPT_TABLE"]
    out_dir = dirs["DIR_OUT_BENCHMARK"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    status("ei_report", "loading benchmark summaries")
    benchmark_outputs = load_or_materialize_ei_benchmark_outputs(root, force=False)
    status("ei_report", "building shrinkage sensitivity summary")
    shrinkage_sensitivity_summary, shrinkage_sensitivity_path = (
        build_ei_shrinkage_sensitivity_summary(root, force=False)
    )
    status("ei_report", "writing manuscript figures and LaTeX tables")
    write_ei_benchmark_manuscript_artifacts(
        benchmark_outputs.summary,
        benchmark_outputs.external_summary,
        shrinkage_sensitivity_summary=shrinkage_sensitivity_summary,
        fig_dir=fig_dir,
        table_dir=table_dir,
    )
    return {
        "benchmark_ei_summary": benchmark_outputs.summary_path,
        "benchmark_ei_external_summary": benchmark_outputs.external_summary_path,
        "benchmark_ei_shrinkage_sensitivity_data": shrinkage_sensitivity_path,
        "benchmark_ei_core_main": table_dir / "benchmark_ei_core_main.tex",
        "benchmark_ei_targets_main": table_dir / "benchmark_ei_targets_main.tex",
        "benchmark_ei_interval_main": table_dir / "benchmark_ei_interval_main.tex",
        "benchmark_ei_overview_main": table_dir / "benchmark_ei_overview_main.tex",
        "benchmark_ei_summary_figure": fig_dir / "benchmark_ei_summary.pdf",
        "benchmark_ei_targets_figure": fig_dir / "benchmark_ei_targets.pdf",
        "benchmark_ei_interval_sharpness_figure": fig_dir / "benchmark_ei_interval_sharpness.pdf",
        "benchmark_ei_overview_figure": fig_dir / "benchmark_ei_overview.pdf",
        "benchmark_ei_shrinkage_sensitivity_figure": (
            fig_dir / "benchmark_ei_shrinkage_sensitivity.pdf"
        ),
    }


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main() -> None:
    outputs = build_ei_benchmark_manuscript_outputs()
    for name, path in outputs.items():
        status("ei_report", f"{name}: {path}")


__all__ = [
    "build_ei_benchmark_manuscript_outputs",
    "build_ei_shrinkage_sensitivity_summary",
    "ei_core_story_table",
    "ei_interval_story_table",
    "ei_story_latex",
    "ei_targets_story_table",
    "plot_ei_core_panels",
    "plot_ei_interval_sharpness_scatter",
    "plot_ei_overview_panels",
    "plot_ei_shrinkage_sensitivity",
    "plot_ei_targets_panels",
    "write_ei_benchmark_manuscript_artifacts",
]


if __name__ == "__main__":
    main()
