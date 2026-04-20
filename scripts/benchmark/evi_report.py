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
    import importlib.util

    _helper_path = Path(__file__).resolve().parents[1] / "shared" / "import_bootstrap.py"
    _spec = importlib.util.spec_from_file_location("_shared_import_bootstrap", _helper_path)
    if _spec is None or _spec.loader is None:  # pragma: no cover - import bootstrap failure
        raise ImportError(f"Could not load import bootstrap helper from {_helper_path}.")
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    _module.ensure_scripts_on_path_from_entry(__file__)

from unibm._runtime import prepare_matplotlib_env

prepare_matplotlib_env("unibm-benchmark")
import matplotlib
from matplotlib.lines import Line2D

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark.common import (
    IQR_LOWER,
    IQR_UPPER,
    add_wilson_bounds,
    format_median_iqr,
    interval_score,
    interval_contains,
    panel_metric_ylim,
    quantile_agg,
    render_latex_table,
)
from benchmark.design import (
    BENCHMARK_SET_LABELS,
    BLOCK_LINESTYLES,
    CORE_METHODS,
    STRESS_BENCHMARK_SET,
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
from shared.runtime import status

# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

EVI_SHRINKAGE_GRID = (0.00, 0.15, 0.35, 0.55, 0.75, 1.00)
RECORD_LENGTH_METHODS = (
    "disjoint_median_ols",
    "sliding_median_ols",
    "sliding_median_fgls",
)
RECORD_LENGTH_FAMILIES = ("frechet_max_ar", "pareto_additive_ar1")
RECORD_LENGTH_N_OBS = (200, 365, 730)
RECORD_LENGTH_THETA = 0.10
_EVI_SHRINKAGE_REQUIRED_COLUMNS = {
    "benchmark_set",
    "family",
    "n_obs",
    "delta",
    "median_ape",
    "median_coverage",
    "median_interval_score",
}


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
    grouped = add_wilson_bounds(grouped, success_col="n_cover", total_col="n_rep")
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


def _shrinkage_sensitivity_output_path(out_dir: Path) -> Path:
    """Return the canonical appendix CSV for EVI shrinkage sensitivity."""
    return out_dir / "benchmark_shrinkage_sensitivity.csv"


def _stress_summary_output_path(out_dir: Path) -> Path:
    """Return the canonical appendix CSV for the slow-convergence stress suite."""
    return out_dir / "benchmark_stress_summary.csv"


def _record_length_summary_output_path(out_dir: Path) -> Path:
    """Return the canonical appendix CSV for the record-length sensitivity."""
    return out_dir / "benchmark_record_length_sensitivity.csv"


def _shrinkage_sensitivity_contract_ok(
    summary: pd.DataFrame,
    *,
    configs: Iterable[object],
    deltas: tuple[float, ...],
) -> bool:
    """Validate a cached shrinkage-sensitivity summary against the requested design."""
    if not _EVI_SHRINKAGE_REQUIRED_COLUMNS.issubset(summary.columns):
        return False
    expected_families = sorted({str(cfg.family) for cfg in configs})
    expected_sets = sorted({str(cfg.benchmark_set) for cfg in configs})
    expected_n_obs = sorted({int(cfg.n_obs) for cfg in configs})
    observed_deltas = sorted(float(value) for value in summary["delta"].dropna().unique())
    observed_families = sorted(str(value) for value in summary["family"].dropna().unique())
    observed_sets = sorted(str(value) for value in summary["benchmark_set"].dropna().unique())
    observed_n_obs = sorted(int(value) for value in summary["n_obs"].dropna().unique())
    return (
        observed_deltas == list(deltas)
        and observed_families == expected_families
        and observed_sets == expected_sets
        and observed_n_obs == expected_n_obs
    )


def build_evi_shrinkage_sensitivity_summary(
    root: Path | str = ".",
    *,
    configs: list[object] | None = None,
    deltas: Iterable[float] = EVI_SHRINKAGE_GRID,
    max_workers: int | None = None,
    force: bool = False,
) -> tuple[pd.DataFrame, Path]:
    """Materialize the appendix EVI shrinkage-sensitivity CSV.

    The sensitivity run reuses the existing benchmark scenario cache and the
    original sample's bootstrap backbone. Only the covariance-shrinkage value is
    varied, and only for the headline sliding-median-FGLS severity workflow.
    """
    from benchmark.design import (
        default_evi_simulation_configs,
        fit_methods_for_series,
        load_or_simulate_series_bank,
        resolve_benchmark_workers,
        scenario_random_state,
    )
    from benchmark.evi_benchmark import BENCHMARK_ALPHA, BENCHMARK_RANDOM_STATE
    from config import resolve_repo_dirs
    from unibm.evi import estimate_target_scaling

    dirs = resolve_repo_dirs(root)
    out_dir = dirs["DIR_OUT_BENCHMARK"]
    cache_dir = dirs["DIR_OUT_BENCHMARK_CACHE"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = _shrinkage_sensitivity_output_path(out_dir)

    resolved_configs = default_evi_simulation_configs() if configs is None else list(configs)
    shrinkage_values = tuple(float(delta) for delta in deltas)
    if not force and output_path.exists():
        cached = pd.read_csv(output_path)
        if _shrinkage_sensitivity_contract_ok(
            cached,
            configs=resolved_configs,
            deltas=shrinkage_values,
        ):
            status("evi_report", "reusing cached EVI shrinkage-sensitivity CSV")
            return cached, output_path

    status(
        "evi_report",
        "building appendix EVI shrinkage sensitivity from cached scenario series",
    )
    detail_rows: list[dict[str, float | int | str]] = []
    workers = resolve_benchmark_workers(len(resolved_configs), max_workers=max_workers)
    _ = workers  # Sensitivity reuses scenario caches sequentially to maximize reuse.
    for cfg in resolved_configs:
        scenario_seed = scenario_random_state(cfg, master_seed=BENCHMARK_RANDOM_STATE)
        series_bank = load_or_simulate_series_bank(
            cfg,
            random_state=scenario_seed,
            cache_dir=cache_dir,
        )
        for rep, vec in enumerate(series_bank):
            headline_fit = fit_methods_for_series(
                vec,
                quantile=cfg.quantile,
                random_state=rep,
                method_ids=["sliding_median_fgls"],
                cache_dir=cache_dir,
                cache_key=f"{cfg.scenario}__seed{scenario_seed}__rep{rep:04d}",
            )["sliding_median_fgls"]
            for delta in shrinkage_values:
                if np.isclose(delta, 0.35):
                    fit = headline_fit
                else:
                    fit = estimate_target_scaling(
                        vec,
                        target="quantile",
                        quantile=cfg.quantile,
                        sliding=True,
                        bootstrap_reps=0,
                        random_state=rep,
                        curve=headline_fit.curve,
                        plateau=headline_fit.plateau,
                        bootstrap_result=headline_fit.bootstrap,
                        covariance_shrinkage=delta,
                    )
                ci_lo, ci_hi = fit.confidence_interval
                detail_rows.append(
                    {
                        "benchmark_set": cfg.benchmark_set,
                        "family": cfg.family,
                        "n_obs": int(cfg.n_obs),
                        "xi_true": float(cfg.xi_true),
                        "theta_true": float(cfg.theta_true),
                        "phi": float(cfg.phi),
                        "rep": int(rep),
                        "delta": float(delta),
                        "ape": float(abs(fit.slope - cfg.xi_true) / abs(cfg.xi_true)),
                        "interval_score": float(
                            interval_score(
                                cfg.xi_true,
                                ci_lo,
                                ci_hi,
                                alpha=BENCHMARK_ALPHA,
                            )
                        ),
                        "covered": float(interval_contains((ci_lo, ci_hi), cfg.xi_true)),
                    }
                )
    detail = pd.DataFrame(detail_rows)
    scenario_summary = (
        detail.groupby(
            ["benchmark_set", "family", "n_obs", "xi_true", "theta_true", "phi", "delta"],
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
            ["benchmark_set", "family", "n_obs", "delta"],
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
    summary = sort_by_family_order(summary, sort_columns=["delta"])
    summary.to_csv(output_path, index=False)
    return summary, output_path


def build_evi_stress_suite_summary(
    root: Path | str = ".",
    *,
    configs: list[object] | None = None,
    max_workers: int | None = None,
    force: bool = False,
) -> tuple[pd.DataFrame, Path]:
    """Materialize the appendix slow-convergence EVI stress-suite summary CSV."""
    from benchmark.design import stress_evi_simulation_configs
    from benchmark.evi_benchmark import BENCHMARK_RANDOM_STATE, run_evi_benchmark
    from config import resolve_repo_dirs

    dirs = resolve_repo_dirs(root)
    out_dir = dirs["DIR_OUT_BENCHMARK"]
    cache_dir = dirs["DIR_OUT_BENCHMARK_CACHE"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = _stress_summary_output_path(out_dir)

    resolved_configs = stress_evi_simulation_configs() if configs is None else list(configs)
    if not force and output_path.exists():
        cached = pd.read_csv(output_path)
        expected_families = sorted({str(cfg.family) for cfg in resolved_configs})
        expected_sets = sorted({str(cfg.benchmark_set) for cfg in resolved_configs})
        expected_xi = sorted({float(cfg.xi_true) for cfg in resolved_configs})
        expected_theta = sorted({float(cfg.theta_true) for cfg in resolved_configs})
        if (
            {"benchmark_set", "family", "xi_true", "theta_true", "method"}.issubset(cached.columns)
            and sorted(str(value) for value in cached["benchmark_set"].dropna().unique())
            == expected_sets
            and sorted(str(value) for value in cached["family"].dropna().unique())
            == expected_families
            and sorted(float(value) for value in cached["xi_true"].dropna().unique())
            == expected_xi
            and sorted(float(value) for value in cached["theta_true"].dropna().unique())
            == expected_theta
        ):
            status("evi_report", "reusing cached slow-convergence EVI stress summary")
            return cached, output_path

    status("evi_report", "building slow-convergence EVI stress-suite summary")
    _, summary = run_evi_benchmark(
        random_state=BENCHMARK_RANDOM_STATE,
        configs=resolved_configs,
        cache_dir=cache_dir,
        max_workers=max_workers,
    )
    summary.to_csv(output_path, index=False)
    return summary, output_path


def _record_length_sensitivity_contract_ok(
    summary: pd.DataFrame,
    *,
    families: Iterable[str],
    n_obs_values: Iterable[int],
    methods: Iterable[str],
) -> bool:
    required = {
        "family",
        "n_obs",
        "method",
        "median_ape",
        "ape_q25",
        "ape_q75",
        "median_interval_score",
        "interval_score_q25",
        "interval_score_q75",
        "median_coverage",
    }
    if not required.issubset(summary.columns):
        return False
    observed_families = sorted(str(value) for value in summary["family"].dropna().unique())
    observed_n_obs = sorted(int(value) for value in summary["n_obs"].dropna().unique())
    observed_methods = sorted(str(value) for value in summary["method"].dropna().unique())
    return (
        observed_families == sorted(str(value) for value in families)
        and observed_n_obs == sorted(int(value) for value in n_obs_values)
        and observed_methods == sorted(str(value) for value in methods)
    )


def build_evi_record_length_sensitivity_summary(
    root: Path | str = ".",
    *,
    configs: list[object] | None = None,
    max_workers: int | None = None,
    force: bool = False,
) -> tuple[pd.DataFrame, Path]:
    """Materialize the appendix EVI record-length sensitivity summary CSV."""
    from benchmark.design import (
        BENCHMARK_MONTE_CARLO_REPS,
        default_evi_simulation_configs,
        fit_methods_for_series,
    )
    from benchmark.evi_benchmark import BENCHMARK_ALPHA, BENCHMARK_RANDOM_STATE
    from config import resolve_repo_dirs

    dirs = resolve_repo_dirs(root)
    out_dir = dirs["DIR_OUT_BENCHMARK"]
    cache_dir = dirs["DIR_OUT_BENCHMARK_CACHE"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = _record_length_summary_output_path(out_dir)

    if configs is None:
        resolved_configs: list[object] = []
        for n_obs in RECORD_LENGTH_N_OBS:
            resolved_configs.extend(
                default_evi_simulation_configs(
                    families=RECORD_LENGTH_FAMILIES,
                    theta_values=(RECORD_LENGTH_THETA,),
                    n_obs=int(n_obs),
                    reps=BENCHMARK_MONTE_CARLO_REPS,
                )
            )
    else:
        resolved_configs = list(configs)
    expected_families = sorted({str(cfg.family) for cfg in resolved_configs})
    expected_n_obs = sorted({int(cfg.n_obs) for cfg in resolved_configs})
    if not force and output_path.exists():
        cached = pd.read_csv(output_path)
        if _record_length_sensitivity_contract_ok(
            cached,
            families=expected_families,
            n_obs_values=expected_n_obs,
            methods=RECORD_LENGTH_METHODS,
        ):
            status("evi_report", "reusing cached EVI record-length sensitivity summary")
            return cached, output_path

    from benchmark.design import load_or_simulate_series_bank, scenario_random_state

    status("evi_report", "building appendix EVI record-length sensitivity summary")
    detail_rows: list[dict[str, float | int | str]] = []
    _ = max_workers  # Reserved for future parallelization; current run favors cache reuse.
    for cfg in resolved_configs:
        scenario_seed = scenario_random_state(cfg, master_seed=BENCHMARK_RANDOM_STATE)
        series_bank = load_or_simulate_series_bank(
            cfg,
            random_state=scenario_seed,
            cache_dir=cache_dir,
        )
        for rep, vec in enumerate(series_bank):
            fits = fit_methods_for_series(
                vec,
                quantile=cfg.quantile,
                random_state=rep,
                method_ids=RECORD_LENGTH_METHODS,
                cache_dir=cache_dir,
                cache_key=f"{cfg.scenario}__seed{scenario_seed}__rep{rep:04d}",
            )
            for method, fit in fits.items():
                ci_lo, ci_hi = fit.confidence_interval
                detail_rows.append(
                    {
                        "family": str(cfg.family),
                        "n_obs": int(cfg.n_obs),
                        "xi_true": float(cfg.xi_true),
                        "theta_true": float(cfg.theta_true),
                        "method": str(method),
                        "ape": float(abs(fit.slope - cfg.xi_true) / abs(cfg.xi_true)),
                        "interval_score": float(
                            interval_score(
                                cfg.xi_true,
                                ci_lo,
                                ci_hi,
                                alpha=BENCHMARK_ALPHA,
                            )
                        ),
                        "covered": float(interval_contains((ci_lo, ci_hi), cfg.xi_true)),
                    }
                )
    detail = pd.DataFrame(detail_rows)
    scenario_summary = (
        detail.groupby(
            ["family", "n_obs", "xi_true", "theta_true", "method"],
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
            ["family", "n_obs", "method"],
            as_index=False,
            dropna=False,
        )
        .agg(
            median_ape=("ape_median", "median"),
            ape_q25=("ape_median", quantile_agg(IQR_LOWER)),
            ape_q75=("ape_median", quantile_agg(IQR_UPPER)),
            median_interval_score=("interval_score_median", "median"),
            interval_score_q25=("interval_score_median", quantile_agg(IQR_LOWER)),
            interval_score_q75=("interval_score_median", quantile_agg(IQR_UPPER)),
            median_coverage=("coverage", "median"),
        )
        .reset_index(drop=True)
    )
    summary["method_label"] = summary["method"].map(METHOD_LABELS)
    summary["family"] = pd.Categorical(
        summary["family"],
        categories=ordered_families(expected_families),
        ordered=True,
    )
    summary = summary.sort_values(["family", "n_obs", "method"]).reset_index(drop=True)
    summary["family"] = summary["family"].astype(str)
    summary.to_csv(output_path, index=False)
    return summary, output_path


def evi_record_length_sensitivity_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Render the appendix record-length sensitivity table for headline EVI methods."""
    subset = summary.loc[summary["method"].isin(RECORD_LENGTH_METHODS)].copy()
    subset["Family"] = subset["family"].map(family_label)
    subset["Summary"] = subset.apply(
        lambda row: (
            f"{format_median_iqr(row['median_ape'], row['ape_q25'], row['ape_q75'])} / "
            f"{format_median_iqr(row['median_interval_score'], row['interval_score_q25'], row['interval_score_q75'])}"
        ),
        axis=1,
    )
    table = (
        subset.pivot(
            index=["Family", "n_obs"],
            columns="method_label",
            values="Summary",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    ordered_columns = ["Family", "n_obs"] + [
        METHOD_LABELS[method]
        for method in RECORD_LENGTH_METHODS
        if METHOD_LABELS[method] in table.columns
    ]
    return table.loc[:, ordered_columns]


def plot_evi_shrinkage_sensitivity(
    summary: pd.DataFrame,
    *,
    benchmark_set: str = UNIVERSAL_BENCHMARK_SET,
    file_path: Path | None = None,
    dpi: int = 600,
    title: str | None = None,
    save: bool = False,
) -> None:
    """Plot appendix EVI shrinkage sensitivity against the fixed delta grid."""
    subset = summary.loc[summary["benchmark_set"] == benchmark_set].copy()
    if subset.empty:
        raise ValueError(
            f"No shrinkage-sensitivity rows found for benchmark_set={benchmark_set!r}."
        )
    families = ordered_families(subset["family"].drop_duplicates().tolist())
    metrics = [
        ("median_interval_score", "median Winkler interval score"),
        ("median_coverage", "median coverage"),
        ("median_ape", "median APE"),
    ]
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue"])
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(metrics),
        figsize=(5.2 * len(metrics), 3.7),
        dpi=dpi,
        sharex=True,
    )
    axes = np.asarray(axes, dtype=object).reshape(-1)
    for idx, (metric, ylabel) in enumerate(metrics):
        ax = axes[idx]
        for family_idx, family in enumerate(families):
            family_frame = subset.loc[subset["family"] == family].sort_values("delta")
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
            ax.axhline(0.95, color="0.4", linestyle="--", linewidth=1.0)
            ax.set_ylim(0.0, 1.02)
        else:
            values = subset[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size:
                ax.set_ylim(0.0, float(np.max(finite) * 1.08))
        ax.set_xlabel(r"FGLS shrinkage $\delta$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(EVI_SHRINKAGE_GRID)
        ax.grid(alpha=0.2, linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=9, loc="best")
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


_METRIC_COLUMNS = {
    "ape": ("ape_median", "ape_q25", "ape_q75"),
    "coverage": ("coverage", "coverage_lo", "coverage_hi"),
    "interval_score": ("interval_score_median", "interval_score_q25", "interval_score_q75"),
    "mape": ("mape", None, None),
}


def _metric_columns(metric: str) -> tuple[str, str | None, str | None]:
    return _METRIC_COLUMNS[metric]


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


def _add_explicit_legend(
    fig: plt.Figure,
    methods: list[str],
    available_methods: set[str],
    *,
    anchor_y: float = 0.006,
) -> None:
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
        bbox_to_anchor=(0.5, anchor_y),
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
    metrics: Iterable[str] = ("interval_score", "ape"),
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
            ylim = panel_metric_ylim(
                family_frame,
                metric=metric,
                methods=methods,
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
        _add_explicit_legend(
            fig,
            methods,
            set(subset["method"].unique()),
            anchor_y=0.006,
        )
        bottom_margin = 0.058
    else:
        _add_grouped_legends(fig, subset)
        bottom_margin = 0.09
    if title is None:
        title = (
            "Combined benchmark"
            if benchmark_set is None
            else f"{BENCHMARK_SET_LABELS.get(benchmark_set, benchmark_set)} benchmark"
        )
    show_title = bool(title)
    if show_title:
        fig.suptitle(title, y=0.982)
    fig.tight_layout(rect=(0, bottom_margin, 1, 0.95 if show_title else 0.992))
    if save and file_path is not None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(file_path)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Manuscript artifact emission (absorbed from main_evi_benchmark_manuscript.py)
# ---------------------------------------------------------------------------


def _main_evi_benchmark_n_obs(summary: pd.DataFrame) -> int:
    """Infer the EVI benchmark sample size from the projected-suite summary rows."""
    rows = summary.loc[summary["benchmark_set"] == UNIVERSAL_BENCHMARK_SET, "n_obs"].dropna()
    if rows.empty:
        raise ValueError("Benchmark summary does not contain any projected-suite rows.")
    return int(round(float(rows.median())))


def write_evi_benchmark_manuscript_artifacts(
    benchmark_summary_df: pd.DataFrame,
    external_benchmark_summary: pd.DataFrame,
    *,
    shrinkage_sensitivity_summary: pd.DataFrame | None = None,
    stress_summary: pd.DataFrame | None = None,
    record_length_summary: pd.DataFrame | None = None,
    fig_dir: Path,
    table_dir: Path,
) -> None:
    """Write EVI benchmark manuscript tables and figures from cached CSV summaries."""
    # Deferred import avoids a circular dependency between report and mixed-baseline helpers.
    from benchmark.evi_external import (
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
                f"Necessary-components EVI benchmark on the projected short-record severity suite "
                f"with xi in {{0.01, 0.03, 0.10, 0.30, 1.0, 3.0, 10.0}}, "
                f"theta in {{0.01, 0.10, 0.50, 1.0}}, and the Fréchet max-AR, moving-maxima q=99, "
                f"and Pareto additive AR(1) families, with n_obs={n_obs}. "
                "Cells report median Winkler interval score (IQR) / median APE (IQR) "
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
                f"Target-comparison EVI benchmark on the projected short-record severity suite "
                f"with xi in {{0.01, 0.03, 0.10, 0.30, 1.0, 3.0, 10.0}}, "
                f"theta in {{0.01, 0.10, 0.50, 1.0}}, and the Fréchet max-AR, moving-maxima q=99, "
                f"and Pareto additive AR(1) families, with n_obs={n_obs}. "
                "Cells report median interval score (IQR) / median APE (IQR) "
                "summarized over the xi grid. All interval metrics use 95\\% CI "
                "(alpha = 0.05), but native interval constructions differ across methods, "
                "so the table is descriptive and is not used to rank cross-class interval calibration."
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
                f"Appendix interval sharpness-versus-calibration summary on the projected EVI suite "
                f"with xi in {{0.01, 0.03, 0.10, 0.30, 1.0, 3.0, 10.0}}, "
                f"theta in {{0.01, 0.10, 0.50, 1.0}}, and the Fréchet max-AR, moving-maxima q=99, "
                f"and Pareto additive AR(1) families, with n_obs={n_obs}. Cells report median 95\\% interval width / "
                "median coverage / median interval score."
            ),
            label="tab:benchmark-interval-main",
        )
    )
    (table_dir / "benchmark_overview_main.tex").write_text(
        render_latex_table(
            benchmark_table(benchmark_summary_df, benchmark_set=UNIVERSAL_BENCHMARK_SET),
            caption=(
                f"Appendix full EVI benchmark overview on the projected EVI suite with xi in "
                f"{{0.01, 0.03, 0.10, 0.30, 1.0, 3.0, 10.0}}, theta in {{0.01, 0.10, 0.50, 1.0}}, "
                f"and the Fréchet max-AR, moving-maxima q=99, and Pareto additive AR(1) families, "
                f"with n_obs={n_obs}."
            ),
            label="tab:benchmark-overview-main",
        )
    )
    plot_benchmark_panels(
        benchmark_summary_df,
        benchmark_set=UNIVERSAL_BENCHMARK_SET,
        methods=CORE_METHODS,
        title="",
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
        title="",
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
    if shrinkage_sensitivity_summary is not None:
        plot_evi_shrinkage_sensitivity(
            shrinkage_sensitivity_summary,
            benchmark_set=UNIVERSAL_BENCHMARK_SET,
            title="Appendix: EVI shrinkage sensitivity for sliding-median-FGLS",
            file_path=fig_dir / "benchmark_shrinkage_sensitivity.pdf",
            save=True,
        )
    if stress_summary is not None:
        (table_dir / "benchmark_stress_main.tex").write_text(
            benchmark_story_latex(
                stress_summary,
                methods=CORE_METHODS,
                benchmark_set=STRESS_BENCHMARK_SET,
                caption=(
                    "Appendix slow-convergence stress suite for the headline EVI workflow. "
                    "The design uses absolute-Student-t moving-maxima q=99 series with "
                    "xi in {0.10, 0.30, 1.0}, theta in {0.10, 0.50, 1.0}, and n_obs=365. "
                    "Cells report median Winkler interval score (IQR) / median APE (IQR) "
                    "summarized over the xi grid. The purpose is robustness checking under "
                    "heavy-tailed but slower-converging block-maxima behavior, not a second "
                    "headline benchmark."
                ),
                label="tab:benchmark-stress-main",
            )
        )
        plot_benchmark_panels(
            stress_summary,
            benchmark_set=STRESS_BENCHMARK_SET,
            methods=CORE_METHODS,
            title="Appendix: slow-convergence EVI stress suite",
            legend_mode="explicit",
            interval_style="errorbar",
            file_path=fig_dir / "benchmark_stress_summary.pdf",
            save=True,
        )
    if record_length_summary is not None:
        (table_dir / "benchmark_record_length_main.tex").write_text(
            render_latex_table(
                evi_record_length_sensitivity_table(record_length_summary),
                caption=(
                    "Appendix EVI record-length sensitivity for the headline within-BM "
                    "severity comparison. The table holds theta fixed at 0.10, compares "
                    "n_obs in {200, 365, 730} for the Fréchet max-AR and Pareto additive AR(1) "
                    "families, and reports median Winkler interval score (IQR) / median APE "
                    "(IQR) across the xi grid for disjoint-median-OLS, sliding-median-OLS, "
                    "and sliding-median-FGLS. The purpose is to delimit how the short-record "
                    "benchmark narrative transports across nearby record lengths, not to create "
                    "a second headline benchmark."
                ),
                label="tab:benchmark-record-length-main",
            )
        )


def build_evi_benchmark_manuscript_outputs(root: Path | str = ".") -> dict[str, Path]:
    """Materialize EVI benchmark manuscript figures and LaTeX tables."""
    from config import resolve_repo_dirs

    from benchmark.evi_benchmark import load_or_materialize_evi_benchmark_outputs

    dirs = resolve_repo_dirs(root)
    fig_dir = dirs["DIR_MANUSCRIPT_FIGURE"]
    table_dir = dirs["DIR_MANUSCRIPT_TABLE"]
    out_dir = dirs["DIR_OUT_BENCHMARK"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    status("evi_report", "loading benchmark summaries")
    benchmark_outputs = load_or_materialize_evi_benchmark_outputs(root, force=False)
    status("evi_report", "building shrinkage sensitivity summary")
    shrinkage_sensitivity_summary, shrinkage_sensitivity_path = (
        build_evi_shrinkage_sensitivity_summary(root, force=False)
    )
    status("evi_report", "building slow-convergence stress summary")
    stress_summary, stress_summary_path = build_evi_stress_suite_summary(root, force=False)
    status("evi_report", "building record-length sensitivity summary")
    record_length_summary, record_length_summary_path = (
        build_evi_record_length_sensitivity_summary(root, force=False)
    )
    status("evi_report", "writing manuscript figures and LaTeX tables")
    write_evi_benchmark_manuscript_artifacts(
        benchmark_outputs.summary,
        benchmark_outputs.external_summary,
        shrinkage_sensitivity_summary=shrinkage_sensitivity_summary,
        stress_summary=stress_summary,
        record_length_summary=record_length_summary,
        fig_dir=fig_dir,
        table_dir=table_dir,
    )
    return {
        "benchmark_summary": benchmark_outputs.summary_path,
        "external_benchmark_summary": benchmark_outputs.external_summary_path,
        "benchmark_shrinkage_sensitivity_data": shrinkage_sensitivity_path,
        "benchmark_stress_summary_data": stress_summary_path,
        "benchmark_record_length_sensitivity_data": record_length_summary_path,
        "benchmark_core_main": table_dir / "benchmark_core_main.tex",
        "benchmark_targets_main": table_dir / "benchmark_targets_main.tex",
        "benchmark_interval_main": table_dir / "benchmark_interval_main.tex",
        "benchmark_overview_main": table_dir / "benchmark_overview_main.tex",
        "benchmark_stress_main": table_dir / "benchmark_stress_main.tex",
        "benchmark_record_length_main": table_dir / "benchmark_record_length_main.tex",
        "benchmark_summary_figure": fig_dir / "benchmark_summary.pdf",
        "benchmark_overview_figure": fig_dir / "benchmark_overview.pdf",
        "benchmark_targets_figure": fig_dir / "benchmark_targets.pdf",
        "benchmark_interval_sharpness_figure": fig_dir / "benchmark_interval_sharpness.pdf",
        "benchmark_shrinkage_sensitivity_figure": fig_dir / "benchmark_shrinkage_sensitivity.pdf",
        "benchmark_stress_summary_figure": fig_dir / "benchmark_stress_summary.pdf",
    }


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main() -> None:
    outputs = build_evi_benchmark_manuscript_outputs()
    for name, path in outputs.items():
        status("evi_report", f"{name}: {path}")


__all__ = [
    "benchmark_story_latex",
    "benchmark_story_table",
    "benchmark_summary",
    "benchmark_table",
    "build_evi_record_length_sensitivity_summary",
    "build_evi_stress_suite_summary",
    "build_evi_shrinkage_sensitivity_summary",
    "build_evi_benchmark_manuscript_outputs",
    "evi_record_length_sensitivity_table",
    "plot_evi_shrinkage_sensitivity",
    "plot_benchmark_panels",
    "write_evi_benchmark_manuscript_artifacts",
]


if __name__ == "__main__":
    main()
