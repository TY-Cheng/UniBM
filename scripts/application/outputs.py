"""Application tables, figures, and export orchestration."""
# ruff: noqa: E402

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

from unibm._runtime import prepare_matplotlib_env

prepare_matplotlib_env("unibm-application")
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
matplotlib.rcParams["hatch.linewidth"] = 1.1

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
    APPLICATION_DESIGN_LIFE_TAUS,
    APPLICATION_RANDOM_STATE,
    DESIGN_LIFE_LEVEL_HORIZONS,
    ApplicationBundle,
)
from shared.runtime import status
from benchmark.common import render_latex_table
from benchmark.design import METHOD_LABELS, METHOD_LOOKUP, fit_methods_for_series
from unibm.core import block_summary_curve
from unibm.models import ScalingFit


@dataclass(frozen=True)
class _WindowBandLegendSpec:
    """Legend-only proxy for a stable-window band with diagonal stripes."""

    facecolor: tuple[float, float, float, float]
    linecolor: tuple[float, float, float, float]
    direction: str


@dataclass(frozen=True)
class _ApplicationTauScalingView:
    """Application-layer shared-xi quantile view for one tau value."""

    tau: float
    intercept: float
    slope: float
    block_sizes: np.ndarray
    values: np.ndarray
    plateau_mask: np.ndarray
    headline: bool

    @property
    def log_block_sizes(self) -> np.ndarray:
        return np.log(self.block_sizes.astype(float))

    @property
    def log_values(self) -> np.ndarray:
        return np.log(self.values.astype(float))

    @property
    def plateau_bounds(self) -> tuple[int, int]:
        plateau_block_sizes = self.block_sizes[self.plateau_mask]
        return int(plateau_block_sizes[0]), int(plateau_block_sizes[-1])

    def fitted_log_values(self) -> np.ndarray:
        return self.intercept + self.slope * self.log_block_sizes

    def design_life_levels(self, years: np.ndarray, *, observations_per_year: float) -> np.ndarray:
        block_sizes = observations_per_year * np.asarray(years, dtype=float)
        return np.exp(self.intercept + self.slope * np.log(block_sizes))


_TAU_STYLES: dict[float, dict[str, object]] = {
    0.5: {"color": "#1f77b4", "linestyle": "-", "linewidth": 1.8, "alpha": 0.98},
    0.9: {"color": "#ff7f0e", "linestyle": "--", "linewidth": 1.25, "alpha": 0.92},
    0.95: {"color": "#2ca02c", "linestyle": "-.", "linewidth": 1.25, "alpha": 0.92},
    0.99: {"color": "#9467bd", "linestyle": ":", "linewidth": 1.35, "alpha": 0.95},
}
_DESIGN_LIFE_AXIS_TICKS = np.asarray([1.0, 2.0, 5.0, 10.0, 25.0, 50.0], dtype=float)


class _WindowBandLegendHandler(HandlerBase):
    """Draw diagonal stable-window stripes inside legend swatches."""

    def create_artists(
        self,
        legend,
        orig_handle: _WindowBandLegendSpec,
        xdescent: float,
        ydescent: float,
        width: float,
        height: float,
        fontsize: float,
        trans,
    ) -> list[object]:
        rect = Rectangle(
            (xdescent, ydescent),
            width,
            height,
            facecolor=orig_handle.facecolor,
            edgecolor="none",
            linewidth=0.0,
            transform=trans,
        )
        artists: list[object] = [rect]
        spacing = max(width / 16.0, 1.2)
        diagonal_dx = height
        cursor = xdescent - diagonal_dx
        while cursor <= xdescent + width:
            if orig_handle.direction == "forward":
                y0 = max(0.0, (xdescent - cursor) / diagonal_dx)
                y1 = min(1.0, (xdescent + width - cursor) / diagonal_dx)
                if y0 < y1:
                    artists.append(
                        Line2D(
                            [
                                cursor + diagonal_dx * y0,
                                cursor + diagonal_dx * y1,
                            ],
                            [
                                ydescent + y0 * height,
                                ydescent + y1 * height,
                            ],
                            color=orig_handle.linecolor,
                            linewidth=0.65,
                            transform=trans,
                        )
                    )
            else:
                y0 = max(0.0, (cursor + diagonal_dx - (xdescent + width)) / diagonal_dx)
                y1 = min(1.0, (cursor + diagonal_dx - xdescent) / diagonal_dx)
                if y0 < y1:
                    artists.append(
                        Line2D(
                            [
                                cursor + diagonal_dx * (1.0 - y0),
                                cursor + diagonal_dx * (1.0 - y1),
                            ],
                            [
                                ydescent + y0 * height,
                                ydescent + y1 * height,
                            ],
                            color=orig_handle.linecolor,
                            linewidth=0.65,
                            transform=trans,
                        )
                    )
            cursor += spacing
        return artists


def _application_observations_per_year(bundle: ApplicationBundle) -> float:
    """Return the effective observation rate used in design-life-level mapping."""
    if bundle.spec.observations_per_year is not None:
        return float(bundle.spec.observations_per_year)
    series = bundle.prepared.evi.series
    n_years = max((series.index.max() - series.index.min()).days / 365.25, 1.0)
    return float(series.size / n_years)


def _tau_label(tau: float) -> str:
    """Return a compact display label for one design-life level."""
    return f"tau={float(tau):.2f}"


def _format_design_life_tick(value: float, _pos: float) -> str:
    """Format one log-scale design-life tick without decimal clutter."""
    if value >= 1.0 and np.isclose(value, round(value)):
        return f"{int(round(value))}"
    return f"{value:g}"


def _apply_design_life_xaxis(ax) -> None:
    """Apply a stable, denser x-axis layout for design-life plots."""
    tick_values = _DESIGN_LIFE_AXIS_TICKS[
        (_DESIGN_LIFE_AXIS_TICKS >= DESIGN_LIFE_LEVEL_HORIZONS.min())
        & (_DESIGN_LIFE_AXIS_TICKS <= DESIGN_LIFE_LEVEL_HORIZONS.max())
    ]
    ax.set_xscale("log")
    ax.set_xlim(float(DESIGN_LIFE_LEVEL_HORIZONS.min()), float(DESIGN_LIFE_LEVEL_HORIZONS.max()))
    ax.xaxis.set_major_locator(FixedLocator(tick_values))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_design_life_tick))
    ax.xaxis.set_minor_locator(NullLocator())


def _tau_style(tau: float) -> dict[str, object]:
    """Return plotting style for one application-layer tau curve."""
    for key, style in _TAU_STYLES.items():
        if np.isclose(float(tau), float(key)):
            return style
    return {"color": "0.35", "linestyle": "-", "linewidth": 1.1, "alpha": 0.9}


def _align_curve_values_to_block_sizes(curve, block_sizes: np.ndarray) -> np.ndarray:
    """Align a block-summary curve to an explicit positive block-size grid."""
    lookup = {
        int(block_size): float(value)
        for block_size, value in zip(
            curve.positive_block_sizes, curve.positive_values, strict=True
        )
    }
    aligned: list[float] = []
    for block_size in np.asarray(block_sizes, dtype=int):
        value = lookup.get(int(block_size))
        if value is None:
            raise ValueError(
                "Derived tau block-summary curve does not cover the headline positive "
                f"block-size grid; missing block size {int(block_size)}."
            )
        aligned.append(value)
    return np.asarray(aligned, dtype=float)


def _tau_scaling_views_for_fit(
    series: pd.Series, headline_fit: ScalingFit
) -> list[_ApplicationTauScalingView]:
    """Build the shared-xi tau grid for any given quantile scaling fit."""
    block_sizes = np.asarray(headline_fit.block_sizes, dtype=int)
    plateau_mask = np.asarray(headline_fit.plateau_mask, dtype=bool)
    log_block_sizes = np.asarray(headline_fit.log_block_sizes, dtype=float)
    views: list[_ApplicationTauScalingView] = []

    for tau in APPLICATION_DESIGN_LIFE_TAUS:
        tau_value = float(tau)
        if np.isclose(tau_value, float(headline_fit.quantile)):
            values = np.asarray(headline_fit.values, dtype=float)
            intercept = float(headline_fit.intercept)
            slope = float(headline_fit.slope)
            headline = True
        else:
            curve = block_summary_curve(
                series.values,
                block_sizes=block_sizes,
                sliding=headline_fit.sliding,
                quantile=tau_value,
                target="quantile",
            )
            values = _align_curve_values_to_block_sizes(curve, block_sizes)
            slope = float(headline_fit.slope)
            intercept = float(
                np.mean(np.log(values[plateau_mask]) - slope * log_block_sizes[plateau_mask])
            )
            headline = False
        views.append(
            _ApplicationTauScalingView(
                tau=tau_value,
                intercept=intercept,
                slope=slope,
                block_sizes=block_sizes.astype(float),
                values=values,
                plateau_mask=plateau_mask.copy(),
                headline=headline,
            )
        )
    return views


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


def _wrapped_axis_label(text: str, *, prefix: str | None = None) -> str:
    """Wrap long parenthetical units onto a second line for axis readability."""
    if " (" in text and len(text) >= 24:
        head, tail = text.split(" (", 1)
        wrapped = f"{head}\n({tail}"
    else:
        wrapped = text
    if prefix is None:
        return wrapped
    if "\n" in wrapped:
        head, tail = wrapped.split("\n", 1)
        return f"{prefix} {head}\n{tail}"
    return f"{prefix} {wrapped}"


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


def _application_design_life_level_rows(bundle: ApplicationBundle) -> list[dict[str, object]]:
    """Return long-form design-life-level summaries for one application."""
    observations_per_year = _application_observations_per_year(bundle)
    rows: list[dict[str, object]] = []
    for view in _tau_scaling_views_for_fit(bundle.prepared.evi.series, bundle.evi_fit):
        design_life_levels = view.design_life_levels(
            DESIGN_LIFE_LEVEL_HORIZONS,
            observations_per_year=observations_per_year,
        )
        for idx, years in enumerate(DESIGN_LIFE_LEVEL_HORIZONS):
            rows.append(
                {
                    "application": bundle.spec.key,
                    "label": bundle.spec.label,
                    "provider": bundle.spec.provider,
                    "design_life_level_basis": bundle.spec.design_life_level_basis,
                    "tau": float(view.tau),
                    "is_headline_tau": bool(view.headline),
                    "shared_xi": True,
                    "design_life_years": float(years),
                    "design_life_level": float(design_life_levels[idx]),
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
        "design_life_level_basis": bundle.spec.design_life_level_basis,
        "headline_tau": float(bundle.evi_fit.quantile),
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


def application_design_life_level_table(bundles: list[ApplicationBundle]) -> pd.DataFrame:
    """Build a compact manuscript-facing design-life-level comparison table."""
    rows: list[dict[str, object]] = []
    for bundle in bundles:
        table_rows = pd.DataFrame(_application_design_life_level_rows(bundle))

        def _fmt(frame: pd.DataFrame, level: float) -> str:
            if frame.empty:
                return "NA"
            match = frame.loc[frame["design_life_years"] == float(level), "design_life_level"]
            if match.empty:
                return "NA"
            value = float(match.iloc[0])
            return f"{value:.2f}" if np.isfinite(value) else "NA"

        for tau in APPLICATION_DESIGN_LIFE_TAUS:
            tau_rows = table_rows.loc[np.isclose(table_rows["tau"], float(tau))]
            rows.append(
                {
                    "Application": bundle.spec.label,
                    "Basis": bundle.spec.design_life_level_basis,
                    "$\\tau$": f"{float(tau):.2f}",
                    "1y level": _fmt(tau_rows, 1.0),
                    "10y level": _fmt(tau_rows, 10.0),
                    "50y level": _fmt(tau_rows, 50.0),
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
            tau_views = _tau_scaling_views_for_fit(bundle.prepared.evi.series, fit)
            for view in tau_views:
                tau_value = float(view.tau)
                one_year, ten_year = view.design_life_levels(
                    np.asarray([1.0, 10.0]),
                    observations_per_year=observations_per_year,
                )
                rows.append(
                    {
                        "application": bundle.spec.key,
                        "provider": bundle.spec.provider,
                        "design_life_level_basis": bundle.spec.design_life_level_basis,
                        "tau": tau_value,
                        "is_headline_tau": bool(np.isclose(tau_value, float(fit.quantile))),
                        "shared_xi": True,
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
                        "one_year_design_life_level": float(one_year),
                        "ten_year_design_life_level": float(ten_year),
                    }
                )
        else:
            rows.append(
                {
                    "application": bundle.spec.key,
                    "provider": bundle.spec.provider,
                    "design_life_level_basis": bundle.spec.design_life_level_basis,
                    "tau": float("nan"),
                    "is_headline_tau": False,
                    "shared_xi": False,
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
                    "one_year_design_life_level": float("nan"),
                    "ten_year_design_life_level": float("nan"),
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
    display_yscale: str = "linear",
    annual_max_yscale: str = "linear",
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
    axes[0].set_ylabel(_wrapped_axis_label(ylabel))
    axes[0].set_yscale(display_yscale)
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
    axes[1].set_ylabel(_wrapped_axis_label(ylabel, prefix="annual max"))
    axes[1].set_yscale(annual_max_yscale)
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
    ax.set_ylabel(_wrapped_axis_label(bundle.spec.ylabel))
    ax.set_title(f"{bundle.spec.label} daily series")
    ax.grid(alpha=0.3)


def _draw_scaling_ax(ax, bundle: ApplicationBundle) -> None:
    """Draw the multi-tau shared-xi scaling panel on a provided axis."""
    tau_views = _tau_scaling_views_for_fit(bundle.prepared.evi.series, bundle.evi_fit)
    headline = next(view for view in tau_views if view.headline)

    for view in tau_views:
        style = _tau_style(view.tau)
        x = np.asarray(view.log_block_sizes, dtype=float)
        y = np.asarray(view.log_values, dtype=float)
        ax.plot(
            x,
            y,
            color=style["color"],
            linewidth=0.85 if not view.headline else 1.0,
            linestyle="-",
            alpha=0.22 if not view.headline else 0.30,
            zorder=1.0,
        )
        ax.plot(
            x,
            np.asarray(view.fitted_log_values(), dtype=float),
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            label=_tau_label(view.tau),
            zorder=2.0 if view.headline else 1.8,
        )

    plateau_mask = np.asarray(headline.plateau_mask, dtype=bool)
    ax.scatter(
        headline.log_block_sizes,
        headline.log_values,
        s=11,
        alpha=0.55,
        color=_tau_style(headline.tau)["color"],
        label=f"{_tau_label(headline.tau)} path",
        zorder=2.1,
    )
    ax.scatter(
        headline.log_block_sizes[plateau_mask],
        headline.log_values[plateau_mask],
        s=20,
        alpha=0.95,
        color="tab:red",
        label=f"{_tau_label(headline.tau)} plateau",
        zorder=2.3,
    )
    ax.axvline(np.log(headline.plateau_bounds[0]), color="grey", linestyle=":", lw=1)
    ax.axvline(np.log(headline.plateau_bounds[1]), color="grey", linestyle=":", lw=1)
    ax.set_xlabel("log(block size)")
    ax.set_ylabel(bundle.spec.scaling_ylabel)
    ax.set_title(bundle.spec.scaling_title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncols=2, title=f"shared ξ = {headline.slope:.3f}")


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
    bb_window = bundle.ei_bb_sliding_fgls.stable_window
    northrop_window = bundle.ei_northrop_sliding_fgls.stable_window
    bb_face = matplotlib.colors.to_rgba(bb_color, 0.028)
    bb_line = matplotlib.colors.to_rgba(bb_color, 0.095)
    northrop_face = matplotlib.colors.to_rgba(northrop_color, 0.028)
    northrop_line = matplotlib.colors.to_rgba(northrop_color, 0.095)

    def _draw_window_band(
        lo: float,
        hi: float,
        *,
        facecolor: tuple[float, float, float, float],
        linecolor: tuple[float, float, float, float],
        direction: str,
        zorder: float,
    ) -> _WindowBandLegendSpec:
        fig = ax.figure
        fig.canvas.draw()
        x_min, x_max = ax.get_xlim()
        left = (lo - x_min) / (x_max - x_min)
        right = (hi - x_min) / (x_max - x_min)
        bbox = ax.get_window_extent(fig.canvas.get_renderer())
        diagonal_dx = bbox.height / max(bbox.width, 1.0)

        rect = Rectangle(
            (left, 0.0),
            right - left,
            1.0,
            transform=ax.transAxes,
            facecolor=facecolor,
            edgecolor="none",
            linewidth=0.0,
            zorder=zorder,
        )
        ax.add_patch(rect)

        spacing = max((right - left) / 16.0, 0.012)
        segments: list[list[tuple[float, float]]] = []
        if direction == "forward":
            cursor = left - diagonal_dx
            while cursor <= right:
                y0 = max(0.0, (left - cursor) / diagonal_dx)
                y1 = min(1.0, (right - cursor) / diagonal_dx)
                if y0 < y1:
                    segments.append(
                        [
                            (cursor + diagonal_dx * y0, y0),
                            (cursor + diagonal_dx * y1, y1),
                        ]
                    )
                cursor += spacing
        else:
            cursor = left - diagonal_dx
            while cursor <= right:
                y0 = max(0.0, (cursor + diagonal_dx - right) / diagonal_dx)
                y1 = min(1.0, (cursor + diagonal_dx - left) / diagonal_dx)
                if y0 < y1:
                    segments.append(
                        [
                            (cursor + diagonal_dx * (1.0 - y0), y0),
                            (cursor + diagonal_dx * (1.0 - y1), y1),
                        ]
                    )
                cursor += spacing
        stripes = LineCollection(
            segments,
            colors=[linecolor],
            linewidths=0.65,
            transform=ax.transAxes,
            zorder=zorder + 0.01,
        )
        ax.add_collection(stripes)
        return _WindowBandLegendSpec(
            facecolor=facecolor,
            linecolor=linecolor,
            direction=direction,
        )

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    band_handles: list[_WindowBandLegendSpec] = []
    band_labels: list[str] = []
    if bb_window is not None:
        band_handles.append(
            _draw_window_band(
                np.log(float(bb_window.lo)),
                np.log(float(bb_window.hi)),
                facecolor=bb_face,
                linecolor=bb_line,
                direction="forward",
                zorder=0.15,
            )
        )
        band_labels.append("BB stable window")
    if northrop_window is not None:
        band_handles.append(
            _draw_window_band(
                np.log(float(northrop_window.lo)),
                np.log(float(northrop_window.hi)),
                facecolor=northrop_face,
                linecolor=northrop_line,
                direction="backward",
                zorder=0.16,
            )
        )
        band_labels.append("Northrop stable window")
    ax.set_xlabel("log(block size)")
    ax.set_ylabel("extremal index")
    ax.set_title(f"{bundle.spec.label} extremal-index comparison")
    ax.grid(alpha=0.3)
    handles = legend_handles[:2] + band_handles + legend_handles[2:]
    labels = legend_labels[:2] + band_labels + legend_labels[2:]
    ax.legend(
        handles,
        labels,
        fontsize=8,
        ncols=2,
        handler_map={_WindowBandLegendSpec: _WindowBandLegendHandler()},
    )


def _draw_design_life_levels_ax(ax, bundle: ApplicationBundle) -> None:
    """Draw the application design-life-level comparison on a provided axis."""
    rows = pd.DataFrame(_application_design_life_level_rows(bundle))
    for tau in APPLICATION_DESIGN_LIFE_TAUS:
        tau_value = float(tau)
        tau_rows = rows.loc[np.isclose(rows["tau"], tau_value)].sort_values("design_life_years")
        style = _tau_style(tau_value)
        ax.plot(
            tau_rows["design_life_years"],
            tau_rows["design_life_level"],
            marker="o" if np.isclose(tau_value, 0.5) else None,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            alpha=style["alpha"],
            label=f"design-life level ({_tau_label(tau_value)})",
        )
    _apply_design_life_xaxis(ax)
    ax.set_yscale(bundle.spec.design_life_level_yscale)
    ax.set_xlabel(bundle.spec.design_life_level_label)
    ax.set_ylabel(bundle.spec.ylabel)
    ax.set_title(f"{bundle.spec.label} design-life levels")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncols=2)


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


def _plot_scaling_panel(
    bundle: ApplicationBundle,
    *,
    file_path: Path | None = None,
    save: bool = False,
    close: bool | None = None,
) -> None:
    """Plot the multi-tau application scaling panel."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=600)
    _draw_scaling_ax(ax, bundle)
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


def _plot_design_life_levels(
    bundle: ApplicationBundle,
    *,
    file_path: Path | None = None,
    save: bool = False,
    close: bool | None = None,
) -> None:
    """Plot design-life levels for one application."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=600)
    _draw_design_life_levels_ax(ax, bundle)
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
    _draw_design_life_levels_ax(axes[1, 1], bundle)
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
        display_yscale=bundle.spec.time_series_display_yscale,
        annual_max_yscale=bundle.spec.time_series_annual_max_yscale,
        close=close,
    )


def plot_application_scaling(
    bundle: ApplicationBundle,
    *,
    close: bool | None = None,
) -> None:
    """Plot the application EVI scaling fit inline."""
    _plot_scaling_panel(bundle, close=close)


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


def plot_application_design_life_levels(
    bundle: ApplicationBundle,
    *,
    close: bool | None = None,
) -> None:
    """Plot the application design-life-level curves inline."""
    _plot_design_life_levels(bundle, close=close)


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
        display_yscale=bundle.spec.time_series_display_yscale,
        annual_max_yscale=bundle.spec.time_series_annual_max_yscale,
        file_path=fig_dir / f"application_ts_{bundle.spec.figure_stem}.pdf",
        save=True,
    )
    _plot_scaling_panel(
        bundle,
        file_path=fig_dir / f"application_evi_{bundle.spec.figure_stem}.pdf",
        save=True,
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
    _plot_design_life_levels(
        bundle,
        file_path=fig_dir / f"application_design_life_{bundle.spec.figure_stem}.pdf",
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
    design_life_level_rows: list[dict[str, object]] = []
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
        design_life_level_rows.extend(_application_design_life_level_rows(bundle))
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

    pd.DataFrame(design_life_level_rows).sort_values(
        ["application", "tau", "design_life_years"]
    ).to_csv(
        out_dir / "application_design_life_levels.csv",
        index=False,
    )
    _usgs_site_audit_frame(metadata_app_dir).to_csv(
        out_dir / "application_usgs_site_screening.csv",
        index=False,
    )
    pd.DataFrame(method_rows).sort_values(["application", "tau", "method"]).to_csv(
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
    status("application", "writing application LaTeX design-life-level table")
    (table_dir / "application_design_life_levels_main.tex").write_text(
        render_latex_table(
            application_design_life_level_table(bundles),
            caption=(
                "Application-side UniBM design-life-level summary across the manuscript case studies. "
                "Rows show the shared-$\\xi$ application tau grid "
                "(`tau = 0.50, 0.90, 0.95, 0.99`) obtained by evaluating the fitted block-maximum "
                "quantile scaling law at 1-, 10-, and 50-year design-life spans. The `tau = 0.50` row is the "
                "headline median design-life level, while higher-tau rows are increasingly conservative "
                "upper design-life levels derived by reusing the same plateau and slope with tau-specific "
                "intercept shifts. Streamflow rows are on the calendar-day basis, while NFIP rows are "
                "on the claim-active-day basis."
            ),
            label="tab:application-design-life-levels-main",
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
        "application_design_life_levels": out_dir / "application_design_life_levels.csv",
        "application_methods": out_dir / "application_methods.csv",
        "application_ei_methods": out_dir / "application_ei_methods.csv",
        "application_ei_seasonal_methods": out_dir / "application_ei_seasonal_methods.csv",
        "application_usgs_site_screening": out_dir / "application_usgs_site_screening.csv",
        "application_summary_main": table_dir / "application_summary_main.tex",
        "application_design_life_levels_main": table_dir
        / "application_design_life_levels_main.tex",
        "application_ei_main": table_dir / "application_ei_main.tex",
    }


__all__ = [
    "application_ei_table",
    "application_ei_method_rows",
    "application_method_rows",
    "application_design_life_level_table",
    "application_summary_record",
    "application_summary_table",
    "build_application_outputs",
    "plot_application_composite",
    "plot_application_ei",
    "plot_application_design_life_levels",
    "plot_application_overview",
    "plot_application_scaling",
    "plot_application_target_stability",
    "plot_application_time_series",
    "seasonal_monthly_pit_unit_frechet",
    "write_application_figures",
]
