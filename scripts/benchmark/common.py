"""Shared benchmark scoring and table helpers used by EVI and EI workflows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

IQR_LOWER = 0.25
IQR_UPPER = 0.75


def wilson_interval(successes: float, n_obs: int, z_crit: float = 1.96) -> tuple[float, float]:
    """Return a Wilson interval for a bounded coverage probability."""
    if n_obs <= 0:
        return (np.nan, np.nan)
    p_hat = float(successes) / float(n_obs)
    denom = 1.0 + z_crit**2 / n_obs
    center = (p_hat + z_crit**2 / (2 * n_obs)) / denom
    half_width = z_crit * np.sqrt((p_hat * (1 - p_hat) + z_crit**2 / (4 * n_obs)) / n_obs) / denom
    return max(0.0, center - half_width), min(1.0, center + half_width)


def add_wilson_bounds(
    frame: pd.DataFrame,
    *,
    success_col: str,
    total_col: str,
    lower_col: str = "coverage_lo",
    upper_col: str = "coverage_hi",
    z_crit: float = 1.96,
) -> pd.DataFrame:
    """Attach Wilson-interval bounds to an aggregated benchmark frame."""
    lower = np.full(frame.shape[0], np.nan, dtype=float)
    upper = np.full(frame.shape[0], np.nan, dtype=float)
    if not frame.empty:
        successes = pd.to_numeric(frame[success_col], errors="coerce").to_numpy(dtype=float)
        totals = pd.to_numeric(frame[total_col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(successes) & np.isfinite(totals) & (totals > 0)
        if np.any(valid):
            p_hat = successes[valid] / totals[valid]
            denom = 1.0 + z_crit**2 / totals[valid]
            center = (p_hat + z_crit**2 / (2.0 * totals[valid])) / denom
            half_width = (
                z_crit
                * np.sqrt(
                    (p_hat * (1.0 - p_hat) + z_crit**2 / (4.0 * totals[valid])) / totals[valid]
                )
                / denom
            )
            lower[valid] = np.maximum(0.0, center - half_width)
            upper[valid] = np.minimum(1.0, center + half_width)
    frame[lower_col] = lower
    frame[upper_col] = upper
    return frame


def interval_contains(interval: tuple[float, float], value: float) -> bool:
    """Return whether a closed interval contains the requested value."""
    return bool(interval[0] <= value <= interval[1])


def latex_escape(text: object) -> str:
    """Escape a value for safe inclusion in a simple LaTeX table."""
    value = str(text)
    sentinel = "<<UNIBMBS>>"
    value = value.replace("\\", sentinel)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for raw, repl in replacements.items():
        value = value.replace(raw, repl)
    return value.replace(sentinel, r"\textbackslash{}")


def render_latex_table(table: pd.DataFrame, *, caption: str, label: str) -> str:
    """Render a small flat table without notebook-specific dependencies."""
    columns = [str(col) for col in table.columns]
    alignment = "ll" + "c" * max(len(columns) - 2, 0)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\hline",
        " & ".join(latex_escape(col) for col in columns) + r" \\",
        r"\hline",
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append(" & ".join(latex_escape(value) for value in row) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def quantile_agg(q: float):
    """Return a named aggregation callable for one empirical quantile."""

    def _agg(values: pd.Series) -> float:
        arr = np.asarray(values, dtype=float)
        if not np.isfinite(arr).any():
            return float("nan")
        return float(np.nanquantile(arr, q))

    return _agg


def format_median_iqr(center: float, q25: float, q75: float) -> str:
    """Format one scalar summary as `median (IQR)` for compact tables."""
    if not (np.isfinite(center) and np.isfinite(q25) and np.isfinite(q75)):
        return "NA"
    return f"{center:.2f} ({(q75 - q25):.2f})"


def interval_width(interval_lo: float, interval_hi: float) -> float:
    """Return the width of a finite confidence interval."""
    if not (np.isfinite(interval_lo) and np.isfinite(interval_hi)):
        return np.nan
    return float(interval_hi - interval_lo)


def interval_score(
    truth: float,
    interval_lo: float,
    interval_hi: float,
    *,
    alpha: float,
) -> float:
    """Return the Winkler interval score for a central `(1 - alpha)` interval."""
    if not (np.isfinite(truth) and np.isfinite(interval_lo) and np.isfinite(interval_hi)):
        return np.nan
    width = interval_width(interval_lo, interval_hi)
    if truth < interval_lo:
        return float(width + (2.0 / alpha) * (interval_lo - truth))
    if truth > interval_hi:
        return float(width + (2.0 / alpha) * (truth - interval_hi))
    return float(width)


def bootstrap_percentile_interval(
    draws: Iterable[float] | np.ndarray,
    *,
    ci_level: float = 0.95,
    min_draws: int = 5,
) -> tuple[float, float]:
    """Return a percentile interval from bootstrap draws."""
    samples = np.asarray(list(draws), dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size < min_draws:
        return (np.nan, np.nan)
    alpha = 1.0 - ci_level
    return (
        float(np.quantile(samples, alpha / 2)),
        float(np.quantile(samples, 1.0 - alpha / 2)),
    )


def round_up_metric_upper(
    metric: str,
    value: float,
    *,
    upper_steps: Mapping[str, tuple[float, ...]],
) -> float:
    """Round one metric upper bound to a stable display scale."""
    steps = upper_steps.get(metric)
    if steps is None or not np.isfinite(value):
        return float(value)
    padded = max(float(value) * 1.02, steps[0])
    for step in steps:
        if padded <= step:
            return float(step)
    return float(steps[-1])


def panel_metric_ylim(
    frame: pd.DataFrame,
    *,
    metric: str,
    methods: Iterable[str],
    metric_columns: Mapping[str, tuple[str, str | None, str | None]],
    upper_steps: Mapping[str, tuple[float, ...]],
    method_col: str = "method",
) -> tuple[float, float] | None:
    """Choose a stable y-limit for one benchmark panel."""
    available_methods = set(frame[method_col].dropna().tolist())
    method_list = [method for method in methods if method in available_methods]
    if not method_list:
        return None
    center_col, _, upper_col = metric_columns[metric]
    value_col = upper_col if upper_col is not None else center_col
    values = frame.loc[frame[method_col].isin(method_list), value_col].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return (
        0.0,
        round_up_metric_upper(metric, float(np.max(finite)), upper_steps=upper_steps),
    )


__all__ = [
    "IQR_LOWER",
    "IQR_UPPER",
    "add_wilson_bounds",
    "bootstrap_percentile_interval",
    "format_median_iqr",
    "interval_score",
    "interval_contains",
    "interval_width",
    "latex_escape",
    "panel_metric_ylim",
    "quantile_agg",
    "render_latex_table",
    "round_up_metric_upper",
    "wilson_interval",
]
