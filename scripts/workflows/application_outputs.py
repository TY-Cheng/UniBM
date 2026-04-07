"""Application tables, figures, and export orchestration."""
# ruff: noqa: E402

from __future__ import annotations

import json
from pathlib import Path
import sys

from unibm._runtime import prepare_matplotlib_env

prepare_matplotlib_env("unibm-application")
import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

from config import resolve_repo_dirs
import numpy as np
import pandas as pd

from .application_fit import build_application_bundles_from_inputs
from .application_inputs import (
    build_application_inputs,
    ensure_ghcn_raw_data,
    ensure_nfip_raw_data,
    ensure_usgs_raw_data,
    load_usgs_frozen_sites,
)
from .application_metadata import ensure_application_metadata
from .application_screening import screen_extreme_series, screen_extremal_index_series
from .application_specs import (
    APPLICATION_RANDOM_STATE,
    RETURN_LEVEL_HORIZONS,
    ApplicationBundle,
)
from .workflow_runtime import status
from .benchmark_common import render_latex_table
from .benchmark_design import METHOD_LABELS, METHOD_LOOKUP, fit_methods_for_series
from unibm.core import estimate_return_level
from unibm.plotting import plot_scaling_fit


def _application_observations_per_year(bundle: ApplicationBundle) -> float:
    """Return the effective observation rate used in return-level mapping."""
    if bundle.spec.observations_per_year is not None:
        return float(bundle.spec.observations_per_year)
    series = bundle.prepared.evi.series
    n_years = max((series.index.max() - series.index.min()).days / 365.25, 1.0)
    return float(series.size / n_years)


def _format_interval(center: float, lo: float, hi: float) -> str:
    """Format one estimate and interval for compact application tables."""
    if not (np.isfinite(center) and np.isfinite(lo) and np.isfinite(hi)):
        return "NA"
    return f"{center:.2f} [{lo:.2f}, {hi:.2f}]"


def _save_figure_pair(fig, file_path: Path) -> None:
    """Save the publication figure to the requested path."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path)


def _should_close_figure(close: bool | None) -> bool:
    """Close figures automatically in batch runs but keep them alive in notebooks."""
    if close is not None:
        return bool(close)
    return "ipykernel" not in sys.modules


def _role_series_rows(
    bundle: ApplicationBundle,
    *,
    derived_dir: Path,
) -> list[dict[str, object]]:
    """Write and register the role-specific series behind one application."""
    rows: list[dict[str, object]] = []
    series_map = {
        "display": bundle.prepared.display,
        "evi": bundle.prepared.evi,
        "ei": bundle.prepared.ei,
    }
    for role, prepared in series_map.items():
        file_path = derived_dir / "applications" / f"{bundle.spec.key}__{role}.csv.gz"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        prepared.to_frame().to_csv(file_path, compression="gzip")
        rows.append(
            {
                "application": bundle.spec.key,
                "label": bundle.spec.label,
                "provider": bundle.spec.provider,
                "secondary_case": bundle.spec.secondary_case,
                "role": role,
                "series_name": prepared.name,
                "value_name": prepared.value_name,
                "n_obs": int(prepared.series.size),
                "start": str(prepared.series.index.min().date()),
                "end": str(prepared.series.index.max().date()),
                "derived_file": str(file_path),
                **prepared.metadata,
            }
        )
    return rows


def _application_return_level_rows(bundle: ApplicationBundle) -> list[dict[str, object]]:
    """Return long-form return-level summaries for one application."""
    observations_per_year = _application_observations_per_year(bundle)
    unconditional = estimate_return_level(
        bundle.evi_fit,
        RETURN_LEVEL_HORIZONS,
        observations_per_year=observations_per_year,
    )
    adjusted = None
    if bundle.spec.provider != "fema":
        adjusted = estimate_return_level(
            bundle.evi_fit,
            RETURN_LEVEL_HORIZONS,
            observations_per_year=observations_per_year,
            extremal_index=bundle.ei_primary.theta_hat,
        )
    rows: list[dict[str, object]] = []
    for idx, horizon in enumerate(RETURN_LEVEL_HORIZONS):
        rows.append(
            {
                "application": bundle.spec.key,
                "label": bundle.spec.label,
                "provider": bundle.spec.provider,
                "return_level_basis": bundle.spec.return_level_basis,
                "horizon_years": float(horizon),
                "return_level": float(unconditional[idx]),
                "return_level_ei_adjusted": (
                    float("nan") if adjusted is None else float(adjusted[idx])
                ),
                "theta_hat": float(bundle.ei_primary.theta_hat),
            }
        )
    return rows


def application_summary_record(bundle: ApplicationBundle) -> dict[str, object]:
    """Summarize the primary EVI/EI application outputs for CSV/JSON export."""
    return {
        "application": bundle.spec.key,
        "label": bundle.spec.label,
        "provider": bundle.spec.provider,
        "secondary_case": bundle.spec.secondary_case,
        "n_display_obs": int(bundle.prepared.display.series.size),
        "n_evi_obs": int(bundle.prepared.evi.series.size),
        "n_ei_obs": int(bundle.prepared.ei.series.size),
        "start": str(bundle.prepared.display.series.index.min().date()),
        "end": str(bundle.prepared.display.series.index.max().date()),
        "xi_hat": float(bundle.evi_fit.slope),
        "xi_lo": float(bundle.evi_fit.confidence_interval[0]),
        "xi_hi": float(bundle.evi_fit.confidence_interval[1]),
        "plateau_lo": int(bundle.evi_fit.plateau_bounds[0]),
        "plateau_hi": int(bundle.evi_fit.plateau_bounds[1]),
        "theta_hat_bb_sliding_fgls": float(bundle.ei_primary.theta_hat),
        "theta_lo_bb_sliding_fgls": float(bundle.ei_primary.confidence_interval[0]),
        "theta_hi_bb_sliding_fgls": float(bundle.ei_primary.confidence_interval[1]),
        "theta_hat_k_gaps": float(bundle.ei_comparator.theta_hat),
        "theta_lo_k_gaps": float(bundle.ei_comparator.confidence_interval[0]),
        "theta_hi_k_gaps": float(bundle.ei_comparator.confidence_interval[1]),
        "mean_cluster_size": float(1.0 / bundle.ei_primary.theta_hat),
        "ei_stable_level_lo": (
            np.nan
            if bundle.ei_primary.stable_window is None
            else float(bundle.ei_primary.stable_window.lo)
        ),
        "ei_stable_level_hi": (
            np.nan
            if bundle.ei_primary.stable_window is None
            else float(bundle.ei_primary.stable_window.hi)
        ),
        "return_level_basis": bundle.spec.return_level_basis,
        "observations_per_year": _application_observations_per_year(bundle),
    }


def application_summary_table(bundles: list[ApplicationBundle]) -> pd.DataFrame:
    """Build the manuscript-facing cross-application summary table."""
    rows: list[dict[str, object]] = []
    for bundle in bundles:
        rows.append(
            {
                "Application": bundle.spec.label,
                "Provider": bundle.spec.provider.upper(),
                "$\\xi$": _format_interval(
                    float(bundle.evi_fit.slope),
                    float(bundle.evi_fit.confidence_interval[0]),
                    float(bundle.evi_fit.confidence_interval[1]),
                ),
                "$\\theta$ (BB-FGLS)": _format_interval(
                    float(bundle.ei_primary.theta_hat),
                    float(bundle.ei_primary.confidence_interval[0]),
                    float(bundle.ei_primary.confidence_interval[1]),
                ),
                "$\\theta$ (K-gaps)": _format_interval(
                    float(bundle.ei_comparator.theta_hat),
                    float(bundle.ei_comparator.confidence_interval[0]),
                    float(bundle.ei_comparator.confidence_interval[1]),
                ),
                "Mean cluster size": (
                    f"{(1.0 / bundle.ei_primary.theta_hat):.2f}"
                    if np.isfinite(bundle.ei_primary.theta_hat) and bundle.ei_primary.theta_hat > 0
                    else "NA"
                ),
            }
        )
    return pd.DataFrame(rows)


def application_return_level_table(bundles: list[ApplicationBundle]) -> pd.DataFrame:
    """Build a compact manuscript-facing return-level comparison table."""
    rows: list[dict[str, object]] = []
    for bundle in bundles:
        table_rows = pd.DataFrame(_application_return_level_rows(bundle)).set_index(
            "horizon_years"
        )

        def _fmt(level: float, adjusted: bool = False) -> str:
            column = "return_level_ei_adjusted" if adjusted else "return_level"
            if level not in table_rows.index:
                return "NA"
            value = float(table_rows.loc[level, column])
            return f"{value:.2f}" if np.isfinite(value) else "NA"

        rows.append(
            {
                "Application": bundle.spec.label,
                "Basis": bundle.spec.return_level_basis,
                "1y RL": _fmt(1.0),
                "10y RL": _fmt(10.0),
                "10y RL (EI-adj)": _fmt(10.0, adjusted=True),
                "50y RL": _fmt(50.0),
                "50y RL (EI-adj)": _fmt(50.0, adjusted=True),
            }
        )
    return pd.DataFrame(rows)


def application_ei_table(bundles: list[ApplicationBundle]) -> pd.DataFrame:
    """Build a manuscript-facing EI comparison table."""
    rows: list[dict[str, object]] = []
    for bundle in bundles:
        stable_window = (
            "NA"
            if bundle.ei_primary.stable_window is None
            else f"{int(bundle.ei_primary.stable_window.lo)}-{int(bundle.ei_primary.stable_window.hi)}"
        )
        rows.append(
            {
                "Application": bundle.spec.label,
                "$\\theta$ (BB-FGLS)": _format_interval(
                    float(bundle.ei_primary.theta_hat),
                    float(bundle.ei_primary.confidence_interval[0]),
                    float(bundle.ei_primary.confidence_interval[1]),
                ),
                "Stable window": stable_window,
                "$\\theta$ (K-gaps)": _format_interval(
                    float(bundle.ei_comparator.theta_hat),
                    float(bundle.ei_comparator.confidence_interval[0]),
                    float(bundle.ei_comparator.confidence_interval[1]),
                ),
                "Mean cluster size": (
                    f"{(1.0 / bundle.ei_primary.theta_hat):.2f}"
                    if np.isfinite(bundle.ei_primary.theta_hat) and bundle.ei_primary.theta_hat > 0
                    else "NA"
                ),
            }
        )
    return pd.DataFrame(rows)


def application_method_rows(bundle: ApplicationBundle) -> list[dict[str, object]]:
    """Create the EVI method-comparison table used in the notebook/appendix."""
    rows: list[dict[str, object]] = []
    observations_per_year = _application_observations_per_year(bundle)
    fits = fit_methods_for_series(
        bundle.prepared.evi.series.values,
        quantile=bundle.spec.quantile,
        random_state=APPLICATION_RANDOM_STATE,
        reuse_fits={"sliding_median_fgls": bundle.evi_fit},
    )
    for method, fit in fits.items():
        spec = METHOD_LOOKUP[method]
        if fit.target == "quantile":
            one_year, ten_year = estimate_return_level(
                fit,
                years=np.asarray([1.0, 10.0]),
                observations_per_year=observations_per_year,
            )
        else:
            one_year, ten_year = float("nan"), float("nan")
        rows.append(
            {
                "application": bundle.spec.key,
                "provider": bundle.spec.provider,
                "return_level_basis": bundle.spec.return_level_basis,
                "method": method,
                "method_label": METHOD_LABELS[method],
                "summary_target": spec.summary_target,
                "block_scheme": spec.block_scheme,
                "regression": spec.regression,
                "xi_hat": float(fit.slope),
                "xi_lo": float(fit.confidence_interval[0]),
                "xi_hi": float(fit.confidence_interval[1]),
                "plateau_lo": int(fit.plateau_bounds[0]),
                "plateau_hi": int(fit.plateau_bounds[1]),
                "one_year_level": float(one_year),
                "ten_year_level": float(ten_year),
            }
        )
    return rows


def application_ei_method_rows(bundle: ApplicationBundle) -> list[dict[str, object]]:
    """Create the primary formal-EI comparison table for one application."""
    primary = bundle.ei_primary
    comparator = bundle.ei_comparator
    return [
        {
            "application": bundle.spec.key,
            "provider": bundle.spec.provider,
            "method": "bb_sliding_fgls",
            "theta_hat": float(primary.theta_hat),
            "theta_lo": float(primary.confidence_interval[0]),
            "theta_hi": float(primary.confidence_interval[1]),
            "standard_error": float(primary.standard_error),
            "stable_level_lo": (
                np.nan if primary.stable_window is None else float(primary.stable_window.lo)
            ),
            "stable_level_hi": (
                np.nan if primary.stable_window is None else float(primary.stable_window.hi)
            ),
            "mean_cluster_size": float(1.0 / primary.theta_hat),
            "ci_method": primary.ci_method,
            "ci_variant": primary.ci_variant,
        },
        {
            "application": bundle.spec.key,
            "provider": bundle.spec.provider,
            "method": "k_gaps",
            "theta_hat": float(comparator.theta_hat),
            "theta_lo": float(comparator.confidence_interval[0]),
            "theta_hi": float(comparator.confidence_interval[1]),
            "standard_error": float(comparator.standard_error),
            "stable_level_lo": np.nan,
            "stable_level_hi": np.nan,
            "mean_cluster_size": float(1.0 / comparator.theta_hat),
            "ci_method": comparator.ci_method,
            "ci_variant": comparator.ci_variant,
        },
    ]


def _plot_daily_and_annual(
    prepared,
    *,
    ylabel: str,
    title: str,
    file_path: Path | None = None,
    save: bool = False,
    close: bool | None = None,
) -> None:
    """Write a two-panel time-series/annual-maxima figure for one application."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7.2, 5.2), sharex=False, dpi=600)
    axes[0].plot(
        prepared.series.index,
        prepared.series.values,
        color="tab:blue",
        lw=0.6,
        alpha=0.85,
    )
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(title)
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        prepared.annual_maxima.index,
        prepared.annual_maxima.values,
        marker="o",
        ms=2.4,
        lw=0.8,
        color="tab:red",
    )
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel(f"annual max {ylabel}")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    if save and file_path is not None:
        _save_figure_pair(fig, file_path)
    if _should_close_figure(close):
        plt.close(fig)


def _plot_target_stability(
    bundle: ApplicationBundle,
    *,
    title: str,
    file_path: Path | None = None,
    save: bool = False,
    close: bool | None = None,
) -> None:
    """Compare median/mean/mode block summaries on the fitted block-size grid."""
    import matplotlib.pyplot as plt

    from unibm.diagnostics import target_stability_summary

    summary = target_stability_summary(
        bundle.prepared.evi.series.values,
        block_sizes=bundle.evi_fit.block_sizes,
        sliding=True,
        quantile=bundle.spec.quantile,
    )
    quantile_column = (
        "median"
        if np.isclose(bundle.spec.quantile, 0.5)
        else f"quantile_tau_{bundle.spec.quantile:.2f}"
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=600)
    ax.plot(
        summary["block_size"],
        summary[quantile_column],
        label="median block quantile"
        if quantile_column == "median"
        else f"block quantile (tau={bundle.spec.quantile:.2f})",
        color="tab:blue",
        lw=1.2,
    )
    ax.plot(summary["block_size"], summary["mean"], label="block mean", color="tab:orange", lw=1.0)
    ax.plot(summary["block_size"], summary["mode"], label="block mode", color="tab:green", lw=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("block size")
    ax.set_ylabel("block-maxima summary")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save and file_path is not None:
        _save_figure_pair(fig, file_path)
    if _should_close_figure(close):
        plt.close(fig)


def _plot_ei_fit(
    bundle: ApplicationBundle,
    *,
    file_path: Path | None = None,
    save: bool = False,
    close: bool | None = None,
) -> None:
    """Plot the BB sliding EI path with pooled and K-gaps estimates overlaid."""
    import matplotlib.pyplot as plt

    path = bundle.ei_bundle.paths[("bb", True)]
    finite_mask = np.isfinite(path.theta_path)
    levels = path.block_sizes[finite_mask].astype(float)
    theta_path = path.theta_path[finite_mask].astype(float)
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=600)
    ax.plot(
        np.log(levels),
        theta_path,
        color="tab:red",
        marker="D",
        ms=3.5,
        lw=1.2,
        label="BB sliding path",
    )
    if bundle.ei_primary.stable_window is not None:
        lo = float(bundle.ei_primary.stable_window.lo)
        hi = float(bundle.ei_primary.stable_window.hi)
        ax.axvspan(np.log(lo), np.log(hi), color="tab:red", alpha=0.12, label="stable window")
    ax.axhline(
        bundle.ei_primary.theta_hat,
        color="tab:red",
        lw=1.2,
        linestyle="-",
        label="BB-sliding-FGLS",
    )
    ax.axhline(
        bundle.ei_comparator.theta_hat,
        color="tab:green",
        lw=1.2,
        linestyle="--",
        label="K-gaps",
    )
    ax.set_xlabel("log(block size)")
    ax.set_ylabel("extremal index")
    ax.set_title(f"{bundle.spec.label} extremal-index path")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save and file_path is not None:
        _save_figure_pair(fig, file_path)
    if _should_close_figure(close):
        plt.close(fig)


def _plot_return_levels(
    bundle: ApplicationBundle,
    *,
    file_path: Path | None = None,
    save: bool = False,
    close: bool | None = None,
) -> None:
    """Plot return levels for one application."""
    import matplotlib.pyplot as plt

    rows = pd.DataFrame(_application_return_level_rows(bundle))
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=600)
    ax.plot(
        rows["horizon_years"],
        rows["return_level"],
        marker="o",
        color="tab:blue",
        lw=1.2,
        label="UniBM return level",
    )
    adjusted = rows["return_level_ei_adjusted"].to_numpy(dtype=float)
    if np.any(np.isfinite(adjusted)):
        ax.plot(
            rows["horizon_years"],
            adjusted,
            marker="s",
            color="tab:red",
            lw=1.2,
            label="EI-adjusted return level",
        )
    ax.set_xscale("log")
    ax.set_yscale(bundle.spec.return_level_yscale)
    ax.set_xlabel(bundle.spec.return_level_label)
    ax.set_ylabel(bundle.spec.ylabel)
    ax.set_title(f"{bundle.spec.label} return levels")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save and file_path is not None:
        _save_figure_pair(fig, file_path)
    if _should_close_figure(close):
        plt.close(fig)


def _plot_application_overview(
    bundles: list[ApplicationBundle],
    *,
    file_path: Path | None = None,
    save: bool = False,
    close: bool | None = None,
) -> None:
    """Plot cross-application comparisons of xi, theta, and mean cluster size."""
    import matplotlib.pyplot as plt

    labels = [bundle.spec.label for bundle in bundles]
    y = np.arange(len(bundles))
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10.5, 4.8), sharey=True, dpi=600)
    xi = np.asarray([bundle.evi_fit.slope for bundle in bundles], dtype=float)
    xi_lo = np.asarray([bundle.evi_fit.confidence_interval[0] for bundle in bundles], dtype=float)
    xi_hi = np.asarray([bundle.evi_fit.confidence_interval[1] for bundle in bundles], dtype=float)
    theta = np.asarray([bundle.ei_primary.theta_hat for bundle in bundles], dtype=float)
    theta_lo = np.asarray(
        [bundle.ei_primary.confidence_interval[0] for bundle in bundles],
        dtype=float,
    )
    theta_hi = np.asarray(
        [bundle.ei_primary.confidence_interval[1] for bundle in bundles],
        dtype=float,
    )
    cluster = 1.0 / theta
    axes[0].errorbar(
        xi,
        y,
        xerr=np.vstack([xi - xi_lo, xi_hi - xi]),
        fmt="o",
        color="tab:blue",
        capsize=2,
    )
    axes[1].errorbar(
        theta,
        y,
        xerr=np.vstack([theta - theta_lo, theta_hi - theta]),
        fmt="o",
        color="tab:red",
        capsize=2,
    )
    axes[2].scatter(cluster, y, color="tab:purple", s=18)
    axes[0].set_xlabel("xi")
    axes[1].set_xlabel("theta")
    axes[2].set_xlabel("1 / theta")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Application overview: tail severity and clustering")
    fig.tight_layout()
    if save and file_path is not None:
        _save_figure_pair(fig, file_path)
    if _should_close_figure(close):
        plt.close(fig)


def plot_application_time_series(
    bundle: ApplicationBundle,
    *,
    close: bool | None = None,
) -> None:
    """Plot the application display series and annual maxima inline."""
    _plot_daily_and_annual(
        bundle.prepared.display,
        ylabel=bundle.spec.ylabel,
        title=bundle.spec.time_series_title,
        close=close,
    )


def plot_application_scaling(
    bundle: ApplicationBundle,
    *,
    close: bool | None = None,
) -> None:
    """Plot the application EVI scaling fit inline."""
    plot_scaling_fit(
        bundle.evi_fit,
        title=bundle.spec.scaling_title,
        ylabel=bundle.spec.scaling_ylabel,
        close=close,
    )


def plot_application_target_stability(
    bundle: ApplicationBundle,
    *,
    close: bool | None = None,
) -> None:
    """Plot the application target-stability comparison inline when available."""
    if bundle.spec.target_stability_title is None or bundle.spec.provider == "fema":
        return
    _plot_target_stability(
        bundle,
        title=bundle.spec.target_stability_title,
        close=close,
    )


def plot_application_ei(
    bundle: ApplicationBundle,
    *,
    close: bool | None = None,
) -> None:
    """Plot the application EI path inline."""
    _plot_ei_fit(bundle, close=close)


def plot_application_return_levels(
    bundle: ApplicationBundle,
    *,
    close: bool | None = None,
) -> None:
    """Plot the application return-level curves inline."""
    _plot_return_levels(bundle, close=close)


def plot_application_overview(
    bundles: list[ApplicationBundle],
    *,
    close: bool | None = None,
) -> None:
    """Plot the cross-application overview inline."""
    _plot_application_overview(bundles, close=close)


def write_application_figures(bundle: ApplicationBundle, fig_dir: Path) -> None:
    """Write manuscript-ready application figures for one bundle."""
    _plot_daily_and_annual(
        bundle.prepared.display,
        ylabel=bundle.spec.ylabel,
        title=bundle.spec.time_series_title,
        file_path=fig_dir / f"application_ts_{bundle.spec.figure_stem}.pdf",
        save=True,
    )
    plot_scaling_fit(
        bundle.evi_fit,
        file_path=fig_dir / f"application_evi_{bundle.spec.figure_stem}.pdf",
        save=True,
        title=bundle.spec.scaling_title,
        ylabel=bundle.spec.scaling_ylabel,
    )
    if bundle.spec.target_stability_title is not None and bundle.spec.provider != "fema":
        _plot_target_stability(
            bundle,
            title=bundle.spec.target_stability_title,
            file_path=fig_dir / f"application_target_{bundle.spec.figure_stem}.pdf",
            save=True,
        )
    _plot_ei_fit(
        bundle,
        file_path=fig_dir / f"application_ei_{bundle.spec.figure_stem}.pdf",
        save=True,
    )
    _plot_return_levels(
        bundle,
        file_path=fig_dir / f"application_rl_{bundle.spec.figure_stem}.pdf",
        save=True,
    )


def _provider_metadata_rows(
    bundle: ApplicationBundle,
    *,
    raw_path: Path | None = None,
) -> list[dict[str, object]]:
    """Return provider metadata rows for JSON sidecars."""
    rows: list[dict[str, object]] = []
    for role, prepared in {
        "display": bundle.prepared.display,
        "evi": bundle.prepared.evi,
        "ei": bundle.prepared.ei,
    }.items():
        rows.append(
            {
                "application": bundle.spec.key,
                "provider": bundle.spec.provider,
                "role": role,
                "raw_file": None if raw_path is None else str(raw_path),
                **prepared.metadata,
            }
        )
    return rows


def _usgs_site_audit_frame(metadata_dir: Path) -> pd.DataFrame:
    """Build a lightweight shortlist audit from the candidate and frozen registries."""
    candidate_path = metadata_dir / "usgs_candidate_sites.json"
    with candidate_path.open() as fh:
        candidate_map = json.load(fh)
    frozen_map = load_usgs_frozen_sites(metadata_dir)
    rows: list[dict[str, object]] = []
    for state_code, candidates in candidate_map.items():
        frozen_site = frozen_map.get(str(state_code).upper(), {}).get("site_no")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            rows.append(
                {
                    "state_code": str(state_code).upper(),
                    "site_no": str(candidate.get("site_no", "")),
                    "station_name": str(candidate.get("station_name", "")),
                    "selected": str(candidate.get("site_no", "")) == str(frozen_site),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["state_code", "selected", "site_no"], ascending=[True, False, True]
    )


def build_application_outputs(root: Path | str = ".") -> dict[str, Path]:
    """Materialize all application-side CSVs, metadata, and figures."""
    dirs = resolve_repo_dirs(root)
    metadata_app_dir = dirs["DIR_DATA_METADATA_APPLICATION"]
    derived_dir = dirs["DIR_DATA_DERIVED"]
    out_dir = dirs["DIR_OUT_APPLICATIONS"]
    fig_dir = dirs["DIR_MANUSCRIPT_FIGURE"]
    table_dir = dirs["DIR_MANUSCRIPT_TABLE"]
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    metadata_app_dir.mkdir(parents=True, exist_ok=True)
    ensure_application_metadata(metadata_app_dir)

    status("application", "ensuring raw inputs")
    raw_paths = {
        **ensure_ghcn_raw_data(dirs["DIR_DATA_RAW_GHCN"]),
        **ensure_usgs_raw_data(
            dirs["DIR_DATA_RAW_USGS"],
            metadata_dir=metadata_app_dir,
        ),
        **ensure_nfip_raw_data(dirs["DIR_DATA_RAW_FEMA"]),
    }
    status("application", "building application inputs")
    inputs = build_application_inputs(dirs, raw_paths=raw_paths)
    status("application", "building application bundles")
    bundles = build_application_bundles_from_inputs(inputs)
    series_registry_rows: list[dict[str, object]] = []
    screening_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    return_level_rows: list[dict[str, object]] = []
    method_rows: list[dict[str, object]] = []
    ei_method_rows: list[dict[str, object]] = []
    provider_metadata: dict[str, list[dict[str, object]]] = {"ghcn": [], "usgs": [], "fema": []}

    for bundle in bundles:
        status("application", f"collecting outputs for {bundle.spec.label}")
        series_registry_rows.extend(_role_series_rows(bundle, derived_dir=derived_dir))
        provider_metadata[bundle.spec.provider].extend(
            _provider_metadata_rows(bundle, raw_path=raw_paths.get(bundle.spec.key))
        )
        evi_review = screen_extreme_series(
            bundle.prepared.evi.series, name=bundle.spec.key
        ).to_record()
        evi_review["analysis_type"] = "evi"
        screening_rows.append(evi_review)
        ei_review = screen_extremal_index_series(
            bundle.prepared.ei.series,
            name=bundle.spec.key,
            allow_zeros=bundle.spec.ei_allow_zeros,
        ).to_record()
        screening_rows.append(ei_review)
        summary_rows.append(application_summary_record(bundle))
        return_level_rows.extend(_application_return_level_rows(bundle))
        method_rows.extend(application_method_rows(bundle))
        ei_method_rows.extend(application_ei_method_rows(bundle))
        status("application", f"writing figures for {bundle.spec.label}")
        write_application_figures(bundle, fig_dir)

    status("application", "writing application tables and metadata")
    series_registry = pd.DataFrame(series_registry_rows).sort_values(["application", "role"])
    series_registry.to_csv(out_dir / "application_series_registry.csv", index=False)

    screening = pd.DataFrame(screening_rows).sort_values(["name", "analysis_type"])
    screening.to_csv(out_dir / "application_screening.csv", index=False)

    summary = pd.DataFrame(summary_rows).sort_values(["provider", "application"])
    summary.to_csv(out_dir / "application_summary.csv", index=False)
    with (out_dir / "application_summary.json").open("w") as fh:
        json.dump(summary_rows, fh, indent=2)

    pd.DataFrame(return_level_rows).sort_values(["application", "horizon_years"]).to_csv(
        out_dir / "application_return_levels.csv",
        index=False,
    )
    _usgs_site_audit_frame(metadata_app_dir).to_csv(
        out_dir / "application_usgs_site_screening.csv",
        index=False,
    )
    pd.DataFrame(method_rows).sort_values(["application", "method"]).to_csv(
        out_dir / "application_methods.csv",
        index=False,
    )
    pd.DataFrame(ei_method_rows).sort_values(["application", "method"]).to_csv(
        out_dir / "application_ei_methods.csv",
        index=False,
    )
    for provider, rows in provider_metadata.items():
        with (metadata_app_dir / f"{provider}_sources.json").open("w") as fh:
            json.dump(rows, fh, indent=2)

    status("application", "writing cross-application overview figure")
    _plot_application_overview(
        bundles,
        file_path=fig_dir / "application_overview.pdf",
        save=True,
    )
    status("application", "writing application LaTeX summary table")
    (table_dir / "application_summary_main.tex").write_text(
        render_latex_table(
            application_summary_table(bundles),
            caption=(
                "Application-side UniBM summary across the six manuscript case studies. "
                "Cells report the primary sliding-median-FGLS estimate of $\\xi$, the primary "
                "BB-sliding-FGLS estimate of $\\theta$, the K-gaps comparator, and the implied "
                "mean cluster size $1/\\theta$."
            ),
            label="tab:application-summary-main",
        )
    )
    status("application", "writing application LaTeX return-level table")
    (table_dir / "application_return_levels_main.tex").write_text(
        render_latex_table(
            application_return_level_table(bundles),
            caption=(
                "Application-side UniBM return-level summary across the manuscript case studies. "
                "Non-FEMA rows report baseline and EI-adjusted return levels at 1, 10, and 50 years. "
                "NFIP rows are reported on the claim-active-day basis, so EI-adjusted calendar-day levels "
                "are intentionally omitted in this first application package."
            ),
            label="tab:application-return-levels-main",
        )
    )
    status("application", "writing application LaTeX EI comparison table")
    (table_dir / "application_ei_main.tex").write_text(
        render_latex_table(
            application_ei_table(bundles),
            caption=(
                "Application-side extremal-index summary across the manuscript case studies. "
                "Cells report the primary BB-sliding-FGLS estimate of $\\theta$, its selected stable "
                "window, the K-gaps comparator, and the implied mean cluster size $1/\\theta$."
            ),
            label="tab:application-ei-main",
        )
    )
    return {
        "application_series_registry": out_dir / "application_series_registry.csv",
        "application_screening": out_dir / "application_screening.csv",
        "application_summary": out_dir / "application_summary.csv",
        "application_return_levels": out_dir / "application_return_levels.csv",
        "application_methods": out_dir / "application_methods.csv",
        "application_ei_methods": out_dir / "application_ei_methods.csv",
        "application_usgs_site_screening": out_dir / "application_usgs_site_screening.csv",
        "application_summary_main": table_dir / "application_summary_main.tex",
        "application_return_levels_main": table_dir / "application_return_levels_main.tex",
        "application_ei_main": table_dir / "application_ei_main.tex",
    }


__all__ = [
    "application_ei_table",
    "application_ei_method_rows",
    "application_method_rows",
    "application_return_level_table",
    "application_summary_record",
    "application_summary_table",
    "build_application_outputs",
    "plot_application_ei",
    "plot_application_overview",
    "plot_application_return_levels",
    "plot_application_scaling",
    "plot_application_target_stability",
    "plot_application_time_series",
    "write_application_figures",
]
