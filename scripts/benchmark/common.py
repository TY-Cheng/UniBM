"""Shared benchmark scoring and table helpers used by EVI and EI workflows."""

from __future__ import annotations

from collections.abc import Iterable

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


__all__ = [
    "IQR_LOWER",
    "IQR_UPPER",
    "bootstrap_percentile_interval",
    "format_median_iqr",
    "interval_score",
    "interval_width",
    "latex_escape",
    "quantile_agg",
    "render_latex_table",
    "wilson_interval",
]
