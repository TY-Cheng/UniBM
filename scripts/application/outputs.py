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

from application.fit import build_application_bundles_from_inputs, fit_application_ei_estimates
from application.inputs import (
    build_application_inputs,
    ensure_ghcn_raw_data,
    ensure_nfip_raw_data,
    ensure_usgs_raw_data,
    load_usgs_frozen_sites,
)
from application.metadata import ensure_application_metadata
from application.screening import screen_extreme_series, screen_extremal_index_series
from application.specs import (
    APPLICATION_EI_METHOD_IDS,
    APPLICATION_EVI_METHOD_IDS,
    APPLICATION_RANDOM_STATE,
    RETURN_LEVEL_HORIZONS,
    ApplicationBundle,
)
from shared.runtime import status
from benchmark.common import render_latex_table
from benchmark.design import METHOD_LABELS, METHOD_LOOKUP, fit_methods_for_series
from unibm.core import estimate_return_level
from unibm.plotting import plot_scaling_fit


def _application_observations_per_year(bundle: ApplicationBundle) -> float:
    """Return the effective observation rate used in return-level mapping."""
    if bundle.spec.observations_per_year is not None:
        return float(bundle.spec.observations_per_year)
    series = bundle.prepared.evi.series
    n_years = max((series.index.max() - series.index.min()).days / 365.25, 1.0)
    return float(series.size / n_years)


def _bundle_has_formal_ei(bundle: ApplicationBundle) -> bool:
    """Return whether the application participates in the formal EI workflow."""
    return bool(bundle.spec.formal_ei and bundle.ei_bb_sliding_fgls is not None)


def _bundle_ei_estimates(bundle: ApplicationBundle) -> dict[str, object]:
    """Return the main application EI estimates keyed by method id."""
    return bundle.ei_method_map


def _stable_window_text(estimate) -> str:
    """Format one EI stable window for compact tables."""
    if estimate.stable_window is None:
        return "NA"
    return f"{int(estimate.stable_window.lo)}-{int(estimate.stable_window.hi)}"


def seasonal_monthly_pit_unit_frechet(series: pd.Series) -> pd.Series:
    """Map a daily series to a seasonal-adjusted unit-Frechet scale by month.

    The transform is deterministic and preserves the original daily index.
    Within each calendar month, values are mapped to scaled empirical ranks
    `u = rank / (n + 1)` and then transformed by the unit-Frechet inverse CDF
    `x = -1 / log(u)`.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Seasonal EI sensitivity requires a DatetimeIndex.")
    values = pd.to_numeric(series, errors="coerce")
    transformed = pd.Series(np.nan, index=series.index, dtype=float, name=series.name)
    for month in range(1, 13):
        month_mask = values.index.month == month
        month_values = values.loc[month_mask]
        finite_mask = np.isfinite(month_values.to_numpy(dtype=float))
        if not np.any(finite_mask):
            continue
        finite_values = month_values.iloc[np.flatnonzero(finite_mask)]
        ranks = finite_values.rank(method="average").to_numpy(dtype=float)
        u = np.clip(ranks / (finite_values.size + 1.0), 1e-12, 1.0 - 1e-12)
        transformed.loc[finite_values.index] = 1.0 / (-np.log(u))
    return transformed


def _seasonal_adjusted_ei_method_rows(bundle: ApplicationBundle) -> list[dict[str, object]]:
    """Build seasonal-adjusted EI sensitivity rows for one application."""
    if not _bundle_has_formal_ei(bundle):
        return []
    transformed = seasonal_monthly_pit_unit_frechet(bundle.prepared.ei.series)
    _, estimates = fit_application_ei_estimates(
        transformed,
        allow_zeros=False,
        label=f"{bundle.spec.label} seasonal-adjusted EI sensitivity",
        status_prefix="application",
    )
    rows: list[dict[str, object]] = []
    for method in APPLICATION_EI_METHOD_IDS:
        estimate = estimates[method]
        rows.append(
            {
                "application": bundle.spec.key,
                "provider": bundle.spec.provider,
                "transform": "monthly_pit_unit_frechet",
                "method": method,
                "theta_hat": float(estimate.theta_hat),
                "theta_lo": float(estimate.confidence_interval[0]),
                "theta_hi": float(estimate.confidence_interval[1]),
                "standard_error": float(estimate.standard_error),
                "stable_level_lo": (
                    np.nan if estimate.stable_window is None else float(estimate.stable_window.lo)
                ),
                "stable_level_hi": (
                    np.nan if estimate.stable_window is None else float(estimate.stable_window.hi)
                ),
                "mean_cluster_size": float(1.0 / estimate.theta_hat),
                "ci_method": estimate.ci_method,
                "ci_variant": estimate.ci_variant,
            }
        )
    return rows


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
    series_map: dict[str, object] = {
        "display": bundle.prepared.display,
        "evi": bundle.prepared.evi,
    }
    if bundle.spec.formal_ei:
        series_map["ei"] = bundle.prepared.ei
    else:
        stale_ei_path = derived_dir / "applications" / f"{bundle.spec.key}__ei.csv.gz"
        if stale_ei_path.exists():
            stale_ei_path.unlink()
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
    if bundle.spec.provider != "fema" and _bundle_has_formal_ei(bundle):
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
                "theta_hat": (
                    float("nan")
                    if not _bundle_has_formal_ei(bundle)
                    else float(bundle.ei_primary.theta_hat)
                ),
            }
        )
    return rows


def application_summary_record(bundle: ApplicationBundle) -> dict[str, object]:
    """Summarize the primary EVI/EI application outputs for CSV/JSON export."""
    bb = bundle.ei_bb_sliding_fgls
    northrop = bundle.ei_northrop_sliding_fgls
    k_gaps = bundle.ei_k_gaps
    ferro = bundle.ei_ferro_segers
    return {
        "application": bundle.spec.key,
        "label": bundle.spec.label,
        "provider": bundle.spec.provider,
        "secondary_case": bundle.spec.secondary_case,
        "n_display_obs": int(bundle.prepared.display.series.size),
        "n_evi_obs": int(bundle.prepared.evi.series.size),
        "n_ei_obs": (int(bundle.prepared.ei.series.size) if bundle.spec.formal_ei else np.nan),
        "start": str(bundle.prepared.display.series.index.min().date()),
        "end": str(bundle.prepared.display.series.index.max().date()),
        "xi_hat": float(bundle.evi_fit.slope),
        "xi_lo": float(bundle.evi_fit.confidence_interval[0]),
        "xi_hi": float(bundle.evi_fit.confidence_interval[1]),
        "plateau_lo": int(bundle.evi_fit.plateau_bounds[0]),
        "plateau_hi": int(bundle.evi_fit.plateau_bounds[1]),
        "theta_hat_bb_sliding_fgls": np.nan if bb is None else float(bb.theta_hat),
        "theta_lo_bb_sliding_fgls": (np.nan if bb is None else float(bb.confidence_interval[0])),
        "theta_hi_bb_sliding_fgls": (np.nan if bb is None else float(bb.confidence_interval[1])),
        "theta_hat_northrop_sliding_fgls": (
            np.nan if northrop is None else float(northrop.theta_hat)
        ),
        "theta_lo_northrop_sliding_fgls": (
            np.nan if northrop is None else float(northrop.confidence_interval[0])
        ),
        "theta_hi_northrop_sliding_fgls": (
            np.nan if northrop is None else float(northrop.confidence_interval[1])
        ),
        "theta_hat_k_gaps": np.nan if k_gaps is None else float(k_gaps.theta_hat),
        "theta_lo_k_gaps": (np.nan if k_gaps is None else float(k_gaps.confidence_interval[0])),
        "theta_hi_k_gaps": (np.nan if k_gaps is None else float(k_gaps.confidence_interval[1])),
        "theta_hat_ferro_segers": (np.nan if ferro is None else float(ferro.theta_hat)),
        "theta_lo_ferro_segers": (
            np.nan if ferro is None else float(ferro.confidence_interval[0])
        ),
        "theta_hi_ferro_segers": (
            np.nan if ferro is None else float(ferro.confidence_interval[1])
        ),
        "mean_cluster_size": (np.nan if bb is None else float(1.0 / bb.theta_hat)),
        "ei_stable_level_lo": (
            np.nan if bb is None or bb.stable_window is None else float(bb.stable_window.lo)
        ),
        "ei_stable_level_hi": (
            np.nan if bb is None or bb.stable_window is None else float(bb.stable_window.hi)
        ),
        "ei_northrop_stable_level_lo": (
            np.nan
            if northrop is None or northrop.stable_window is None
            else float(northrop.stable_window.lo)
        ),
        "ei_northrop_stable_level_hi": (
            np.nan
            if northrop is None or northrop.stable_window is None
            else float(northrop.stable_window.hi)
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
                    float("nan")
                    if bundle.ei_bb_sliding_fgls is None
                    else float(bundle.ei_bb_sliding_fgls.theta_hat),
                    float("nan")
                    if bundle.ei_bb_sliding_fgls is None
                    else float(bundle.ei_bb_sliding_fgls.confidence_interval[0]),
                    float("nan")
                    if bundle.ei_bb_sliding_fgls is None
                    else float(bundle.ei_bb_sliding_fgls.confidence_interval[1]),
                ),
                "$\\theta$ (Northrop-FGLS)": _format_interval(
                    float("nan")
                    if bundle.ei_northrop_sliding_fgls is None
                    else float(bundle.ei_northrop_sliding_fgls.theta_hat),
                    float("nan")
                    if bundle.ei_northrop_sliding_fgls is None
                    else float(bundle.ei_northrop_sliding_fgls.confidence_interval[0]),
                    float("nan")
                    if bundle.ei_northrop_sliding_fgls is None
                    else float(bundle.ei_northrop_sliding_fgls.confidence_interval[1]),
                ),
                "Mean cluster size": (
                    f"{(1.0 / bundle.ei_bb_sliding_fgls.theta_hat):.2f}"
                    if bundle.ei_bb_sliding_fgls is not None
                    and np.isfinite(bundle.ei_bb_sliding_fgls.theta_hat)
                    and bundle.ei_bb_sliding_fgls.theta_hat > 0
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
        if not _bundle_has_formal_ei(bundle):
            continue
        rows.append(
            {
                "Application": bundle.spec.label,
                "$\\theta$ (BB-FGLS)": _format_interval(
                    float(bundle.ei_bb_sliding_fgls.theta_hat),
                    float(bundle.ei_bb_sliding_fgls.confidence_interval[0]),
                    float(bundle.ei_bb_sliding_fgls.confidence_interval[1]),
                ),
                "BB stable window": _stable_window_text(bundle.ei_bb_sliding_fgls),
                "$\\theta$ (Northrop-FGLS)": _format_interval(
                    float(bundle.ei_northrop_sliding_fgls.theta_hat),
                    float(bundle.ei_northrop_sliding_fgls.confidence_interval[0]),
                    float(bundle.ei_northrop_sliding_fgls.confidence_interval[1]),
                ),
                "Northrop stable window": _stable_window_text(bundle.ei_northrop_sliding_fgls),
                "$\\theta$ (K-gaps)": _format_interval(
                    float(bundle.ei_k_gaps.theta_hat),
                    float(bundle.ei_k_gaps.confidence_interval[0]),
                    float(bundle.ei_k_gaps.confidence_interval[1]),
                ),
                "$\\theta$ (Ferro-Segers)": _format_interval(
                    float(bundle.ei_ferro_segers.theta_hat),
                    float(bundle.ei_ferro_segers.confidence_interval[0]),
                    float(bundle.ei_ferro_segers.confidence_interval[1]),
                ),
            }
        )
    return pd.DataFrame(rows)


def application_method_rows(bundle: ApplicationBundle) -> list[dict[str, object]]:
    """Create the default application-side EVI summary rows."""
    rows: list[dict[str, object]] = []
    observations_per_year = _application_observations_per_year(bundle)
    fits = fit_methods_for_series(
        bundle.prepared.evi.series.values,
        quantile=bundle.spec.quantile,
        random_state=APPLICATION_RANDOM_STATE,
        method_ids=APPLICATION_EVI_METHOD_IDS,
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
    if not _bundle_has_formal_ei(bundle):
        return []
    rows: list[dict[str, object]] = []
    for method in APPLICATION_EI_METHOD_IDS:
        estimate = bundle.ei_method_map[method]
        rows.append(
            {
                "application": bundle.spec.key,
                "provider": bundle.spec.provider,
                "method": method,
                "theta_hat": float(estimate.theta_hat),
                "theta_lo": float(estimate.confidence_interval[0]),
                "theta_hi": float(estimate.confidence_interval[1]),
                "standard_error": float(estimate.standard_error),
                "stable_level_lo": (
                    np.nan if estimate.stable_window is None else float(estimate.stable_window.lo)
                ),
                "stable_level_hi": (
                    np.nan if estimate.stable_window is None else float(estimate.stable_window.hi)
                ),
                "mean_cluster_size": float(1.0 / estimate.theta_hat),
                "ci_method": estimate.ci_method,
                "ci_variant": estimate.ci_variant,
            }
        )
    return rows


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


def _target_stability_frame(bundle: ApplicationBundle) -> pd.DataFrame:
    """Return target-stability summaries on the fitted block-size grid."""
    from unibm.diagnostics import target_stability_summary

    return target_stability_summary(
        bundle.prepared.evi.series.values,
        block_sizes=bundle.evi_fit.block_sizes,
        sliding=True,
        quantile=bundle.spec.quantile,
    )


def _draw_target_stability_ax(ax, bundle: ApplicationBundle, *, title: str) -> None:
    """Draw the target-stability panel on a provided axis."""
    summary = _target_stability_frame(bundle)
    quantile_column = (
        "median"
        if np.isclose(bundle.spec.quantile, 0.5)
        else f"quantile_tau_{bundle.spec.quantile:.2f}"
    )
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
    ax.legend(fontsize=8)


def _draw_display_series_ax(ax, bundle: ApplicationBundle) -> None:
    """Draw the raw display series as a secondary context panel."""
    prepared = bundle.prepared.display
    ax.plot(
        prepared.series.index,
        prepared.series.values,
        color="tab:blue",
        lw=0.55,
        alpha=0.85,
    )
    ax.set_xlabel("date")
    ax.set_ylabel(bundle.spec.ylabel)
    ax.set_title(f"{bundle.spec.label} daily series")
    ax.grid(alpha=0.3)


def _draw_scaling_ax(ax, bundle: ApplicationBundle) -> None:
    """Draw the headline median-sliding-FGLS scaling fit on a provided axis."""
    fit = bundle.evi_fit
    x = np.asarray(fit.log_block_sizes, dtype=float)
    y = np.asarray(fit.log_values, dtype=float)
    plateau_mask = np.asarray(fit.plateau_mask, dtype=bool)
    ax.scatter(x=x, y=y, s=12, alpha=0.7, color="tab:blue", label="log block summary")
    ax.scatter(
        x=x[plateau_mask],
        y=y[plateau_mask],
        s=18,
        alpha=0.9,
        color="tab:red",
        label="selected plateau",
    )
    fitted = fit.intercept + fit.slope * x[plateau_mask]
    ax.plot(
        x[plateau_mask],
        fitted,
        color="black",
        linestyle="--",
        lw=1.1,
        label=f"slope = {fit.slope:.3f}",
    )
    ax.axvline(np.log(fit.plateau_bounds[0]), color="grey", linestyle=":", lw=1)
    ax.axvline(np.log(fit.plateau_bounds[1]), color="grey", linestyle=":", lw=1)
    ax.set_xlabel("log(block size)")
    ax.set_ylabel(bundle.spec.scaling_ylabel)
    ax.set_title(bundle.spec.scaling_title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def _draw_ei_ax(ax, bundle: ApplicationBundle) -> None:
    """Draw the four-method EI comparison panel on a provided axis."""
    if not _bundle_has_formal_ei(bundle):
        raise ValueError(f"{bundle.spec.label} does not participate in formal EI analysis.")
    assert bundle.ei_bundle is not None
    assert bundle.ei_bb_sliding_fgls is not None
    assert bundle.ei_northrop_sliding_fgls is not None
    assert bundle.ei_k_gaps is not None
    assert bundle.ei_ferro_segers is not None
    bb_path = bundle.ei_bundle.paths[("bb", True)]
    northrop_path = bundle.ei_bundle.paths[("northrop", True)]
    bb_color = "#b22222"
    northrop_color = "#1565c0"

    def _finite_path(path):
        mask = np.isfinite(path.theta_path)
        return np.log(path.block_sizes[mask].astype(float)), path.theta_path[mask].astype(float)

    bb_x, bb_theta = _finite_path(bb_path)
    northrop_x, northrop_theta = _finite_path(northrop_path)
    ax.plot(bb_x, bb_theta, color=bb_color, marker="D", ms=3.0, lw=1.1, label="BB sliding path")
    ax.plot(
        northrop_x,
        northrop_theta,
        color=northrop_color,
        marker="o",
        ms=2.8,
        lw=1.0,
        alpha=0.85,
        label="Northrop sliding path",
    )
    bb_window = bundle.ei_bb_sliding_fgls.stable_window
    northrop_window = bundle.ei_northrop_sliding_fgls.stable_window
    if (
        bb_window is not None
        and northrop_window is not None
        and (
            float(bb_window.lo) == float(northrop_window.lo)
            and float(bb_window.hi) == float(northrop_window.hi)
        )
    ):
        ax.axvspan(
            np.log(float(bb_window.lo)),
            np.log(float(bb_window.hi)),
            facecolor=bb_color,
            edgecolor=bb_color,
            alpha=0.05,
            hatch="///",
            label="BB stable window",
        )
        ax.axvspan(
            np.log(float(northrop_window.lo)),
            np.log(float(northrop_window.hi)),
            facecolor=northrop_color,
            edgecolor=northrop_color,
            alpha=0.03,
            hatch="\\\\",
            label="Northrop stable window",
        )
    else:
        if bb_window is not None:
            ax.axvspan(
                np.log(float(bb_window.lo)),
                np.log(float(bb_window.hi)),
                color=bb_color,
                alpha=0.08,
                label="BB stable window",
            )
        if northrop_window is not None:
            ax.axvspan(
                np.log(float(northrop_window.lo)),
                np.log(float(northrop_window.hi)),
                color=northrop_color,
                alpha=0.05,
                label="Northrop stable window",
            )
    ax.axhline(
        bundle.ei_bb_sliding_fgls.theta_hat,
        color=bb_color,
        lw=1.2,
        linestyle="-",
        label="BB-sliding-FGLS",
    )
    ax.axhline(
        bundle.ei_northrop_sliding_fgls.theta_hat,
        color=northrop_color,
        lw=1.2,
        linestyle="-.",
        label="Northrop-sliding-FGLS",
    )
    ax.axhline(
        bundle.ei_k_gaps.theta_hat,
        color="tab:green",
        lw=1.2,
        linestyle="--",
        label="K-gaps",
    )
    ax.axhline(
        bundle.ei_ferro_segers.theta_hat,
        color="tab:purple",
        lw=1.2,
        linestyle=":",
        label="Ferro-Segers",
    )
    ax.set_xlabel("log(block size)")
    ax.set_ylabel("extremal index")
    ax.set_title(f"{bundle.spec.label} extremal-index comparison")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncols=2)


def _draw_return_levels_ax(ax, bundle: ApplicationBundle) -> None:
    """Draw the application return-level comparison on a provided axis."""
    rows = pd.DataFrame(_application_return_level_rows(bundle))
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
    ax.legend(fontsize=8)


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

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=600)
    _draw_target_stability_ax(ax, bundle, title=title)
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
    """Plot the application EI comparison path with four methods overlaid."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=600)
    _draw_ei_ax(ax, bundle)
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

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=600)
    _draw_return_levels_ax(ax, bundle)
    fig.tight_layout()
    if save and file_path is not None:
        _save_figure_pair(fig, file_path)
    if _should_close_figure(close):
        plt.close(fig)


def _plot_composite_application_diagnostics(
    bundle: ApplicationBundle,
    *,
    file_path: Path | None = None,
    save: bool = False,
    close: bool | None = None,
) -> None:
    """Plot the 2x2 composite application diagnostic figure."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11.2, 8.2), dpi=600)
    _draw_target_stability_ax(
        axes[0, 0],
        bundle,
        title=bundle.spec.target_stability_title or f"{bundle.spec.label} target stability",
    )
    _draw_scaling_ax(axes[0, 1], bundle)
    if _bundle_has_formal_ei(bundle):
        _draw_ei_ax(axes[1, 0], bundle)
    else:
        _draw_display_series_ax(axes[1, 0], bundle)
    _draw_return_levels_ax(axes[1, 1], bundle)
    fig.suptitle(f"{bundle.spec.label}: application diagnostics", y=0.995)
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
    theta = np.asarray(
        [
            np.nan if not _bundle_has_formal_ei(bundle) else bundle.ei_primary.theta_hat
            for bundle in bundles
        ],
        dtype=float,
    )
    theta_lo = np.asarray(
        [
            np.nan
            if not _bundle_has_formal_ei(bundle)
            else bundle.ei_primary.confidence_interval[0]
            for bundle in bundles
        ],
        dtype=float,
    )
    theta_hi = np.asarray(
        [
            np.nan
            if not _bundle_has_formal_ei(bundle)
            else bundle.ei_primary.confidence_interval[1]
            for bundle in bundles
        ],
        dtype=float,
    )
    cluster = np.divide(
        1.0,
        theta,
        out=np.full_like(theta, np.nan),
        where=np.isfinite(theta) & (theta > 0.0),
    )
    axes[0].errorbar(
        xi,
        y,
        xerr=np.vstack([xi - xi_lo, xi_hi - xi]),
        fmt="o",
        color="tab:blue",
        capsize=2,
    )
    theta_mask = np.isfinite(theta) & np.isfinite(theta_lo) & np.isfinite(theta_hi)
    if np.any(theta_mask):
        theta_y = y[theta_mask]
        axes[1].errorbar(
            theta[theta_mask],
            theta_y,
            xerr=np.vstack(
                [
                    theta[theta_mask] - theta_lo[theta_mask],
                    theta_hi[theta_mask] - theta[theta_mask],
                ]
            ),
            fmt="o",
            color="tab:red",
            capsize=2,
        )
    cluster_mask = np.isfinite(cluster)
    if np.any(cluster_mask):
        axes[2].scatter(cluster[cluster_mask], y[cluster_mask], color="tab:purple", s=18)
    axes[0].set_xlabel("xi")
    axes[1].set_xlabel("theta")
    axes[2].set_xlabel("1 / theta")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Application overview: tail severity and formal clustering")
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
    if bundle.spec.target_stability_title is None:
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
    if not _bundle_has_formal_ei(bundle):
        raise ValueError(f"{bundle.spec.label} does not participate in formal EI analysis.")
    _plot_ei_fit(bundle, close=close)


def plot_application_return_levels(
    bundle: ApplicationBundle,
    *,
    close: bool | None = None,
) -> None:
    """Plot the application return-level curves inline."""
    _plot_return_levels(bundle, close=close)


def plot_application_composite(
    bundle: ApplicationBundle,
    *,
    close: bool | None = None,
) -> None:
    """Plot the composite application diagnostic figure inline."""
    _plot_composite_application_diagnostics(bundle, close=close)


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
    if bundle.spec.target_stability_title is not None:
        _plot_target_stability(
            bundle,
            title=bundle.spec.target_stability_title,
            file_path=fig_dir / f"application_target_{bundle.spec.figure_stem}.pdf",
            save=True,
        )
    ei_path = fig_dir / f"application_ei_{bundle.spec.figure_stem}.pdf"
    if _bundle_has_formal_ei(bundle):
        _plot_ei_fit(
            bundle,
            file_path=ei_path,
            save=True,
        )
    elif ei_path.exists():
        ei_path.unlink()
    _plot_return_levels(
        bundle,
        file_path=fig_dir / f"application_rl_{bundle.spec.figure_stem}.pdf",
        save=True,
    )
    _plot_composite_application_diagnostics(
        bundle,
        file_path=fig_dir / f"application_composite_{bundle.spec.figure_stem}.pdf",
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
    if bundle.spec.formal_ei:
        rows.append(
            {
                "application": bundle.spec.key,
                "provider": bundle.spec.provider,
                "role": "ei",
                "raw_file": None if raw_path is None else str(raw_path),
                **bundle.prepared.ei.metadata,
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
    seasonal_ei_method_rows: list[dict[str, object]] = []
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
        if bundle.spec.formal_ei:
            ei_review = screen_extremal_index_series(
                bundle.prepared.ei.series,
                name=bundle.spec.key,
                allow_zeros=bundle.spec.ei_allow_zeros,
            ).to_record()
            screening_rows.append(ei_review)
        summary_rows.append(application_summary_record(bundle))
        return_level_rows.extend(_application_return_level_rows(bundle))
        method_rows.extend(application_method_rows(bundle))
        if bundle.spec.formal_ei:
            ei_method_rows.extend(application_ei_method_rows(bundle))
            seasonal_ei_method_rows.extend(_seasonal_adjusted_ei_method_rows(bundle))
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
    pd.DataFrame(seasonal_ei_method_rows).sort_values(["application", "method"]).to_csv(
        out_dir / "application_ei_seasonal_methods.csv",
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
                "Cells report the headline sliding-median-FGLS estimate of $\\xi$. "
                "Formal EI summaries are only reported for the streamflow and NFIP claim-wave "
                "applications, where the headline BB-sliding-FGLS estimate of $\\theta$, the "
                "Northrop-sliding-FGLS pooled-BM comparator, and the implied mean cluster size "
                "$1/\\theta$ are substantively interpreted."
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
                "Streamflow rows report baseline and formal-EI-adjusted return levels at 1, 10, and "
                "50 years. Houston and Phoenix are retained as EVI-only environmental applications, so "
                "their EI-adjusted columns are intentionally blank. NFIP rows are reported on the "
                "claim-active-day basis, so EI-adjusted calendar-day levels are intentionally omitted."
            ),
            label="tab:application-return-levels-main",
        )
    )
    status("application", "writing application LaTeX EI comparison table")
    (table_dir / "application_ei_main.tex").write_text(
        render_latex_table(
            application_ei_table(bundles),
            caption=(
                "Application-side extremal-index summary for the formal EI applications only. "
                "Cells report the BB-sliding-FGLS and Northrop-sliding-FGLS pooled-BM estimates "
                "together with the K-gaps and Ferro-Segers threshold comparators for the "
                "streamflow and NFIP claim-wave case studies. Stable windows are shown for the "
                "two pooled-BM paths."
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
        "application_ei_seasonal_methods": out_dir / "application_ei_seasonal_methods.csv",
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
    "plot_application_composite",
    "plot_application_ei",
    "plot_application_overview",
    "plot_application_return_levels",
    "plot_application_scaling",
    "plot_application_target_stability",
    "plot_application_time_series",
    "seasonal_monthly_pit_unit_frechet",
    "write_application_figures",
]
