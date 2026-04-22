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


def render_latex_table(
    table: pd.DataFrame,
    *,
    caption: str,
    label: str,
    environment: str = "table",
    position: str = "htbp",
    font_size: str | None = None,
    resize_to_width: str | None = None,
    tabcolsep: str | None = None,
    header_latex: Mapping[str, str] | None = None,
) -> str:
    """Render a small flat table without notebook-specific dependencies."""
    columns = [str(col) for col in table.columns]
    alignment = "ll" + "c" * max(len(columns) - 2, 0)
    lines = [
        rf"\begin{{{environment}}}[{position}]",
        r"\centering",
    ]
    if font_size is not None:
        lines.append(font_size)
    if tabcolsep is not None:
        lines.append(rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}")
    lines.extend([rf"\caption{{{latex_escape(caption)}}}", rf"\label{{{label}}}"])
    if resize_to_width is not None:
        lines.append(rf"\resizebox{{{resize_to_width}}}{{!}}{{%")
    lines.extend(
        [
            rf"\begin{{tabular}}{{{alignment}}}",
            r"\hline",
            " & ".join(
                header_latex.get(col, latex_escape(col))
                if header_latex is not None
                else latex_escape(col)
                for col in columns
            )
            + r" \\",
            r"\hline",
        ]
    )
    for row in table.itertuples(index=False, name=None):
        lines.append(" & ".join(latex_escape(value) for value in row) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}"])
    if resize_to_width is not None:
        lines.append(r"}")
    lines.append(rf"\end{{{environment}}}")
    return "\n".join(lines)


def render_grouped_latex_table(
    table: pd.DataFrame,
    *,
    row_label: str,
    groups: list[tuple[str, list[tuple[str, str]]]],
    second_header_row_label: str | None = None,
    second_header_row_label_raw: bool = False,
    caption: str,
    label: str,
    environment: str = "table",
    position: str = "htbp",
    font_size: str | None = None,
    resize_to_width: str | None = None,
    fit_to_width: str | None = None,
    row_label_width: str = r"0.18\textwidth",
    tabcolsep: str | None = None,
    pair_medians_only: bool = False,
    group_break_after_rows: list[int] | None = None,
    group_break_command: str = r"\addlinespace[2pt]",
    group_break_commands_by_row: Mapping[int, str] | None = None,
    arraystretch: str | None = None,
) -> str:
    """Render a table with one row label column and a two-level grouped header."""
    flat_columns = [column_key for _, columns in groups for column_key, _ in columns]
    alignment = "l" + "c" * len(flat_columns)
    group_breaks = set(group_break_after_rows or [])
    break_command_map = dict(group_break_commands_by_row or {})

    def _render_row_label(value: object) -> str:
        text = str(value)
        if "-" not in text:
            return r"\shortstack[l]{" + latex_escape(text) + "}"
        if text.count("-") >= 2:
            head, tail = text.split("-", maxsplit=1)
            parts = [f"{head}-", tail]
            return r"\shortstack[l]{" + r" \\ ".join(latex_escape(part) for part in parts) + "}"
        if len(text) >= 10:
            head, tail = text.split("-", maxsplit=1)
            parts = [f"{head}-", tail]
            return r"\shortstack[l]{" + r" \\ ".join(latex_escape(part) for part in parts) + "}"
        return r"\shortstack[l]{" + latex_escape(text) + "}"

    def _render_body_cell(value: object, *, compact_pair: bool = False) -> str:
        if compact_pair and isinstance(value, str) and " / " in value:
            parts = [part.strip() for part in value.split(" / ", maxsplit=1)]
            if pair_medians_only:
                parts = [part.split(" (", maxsplit=1)[0].strip() for part in parts]
            return r"\shortstack[c]{" + r" \\ ".join(latex_escape(part) for part in parts) + "}"
        return latex_escape(value)

    lines = [
        rf"\begin{{{environment}}}[{position}]",
        r"\centering",
    ]
    if font_size is not None:
        lines.append(font_size)
    if tabcolsep is not None:
        lines.append(rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}")
    if arraystretch is not None:
        lines.append(rf"\renewcommand{{\arraystretch}}{{{arraystretch}}}")
    lines.extend([rf"\caption{{{latex_escape(caption)}}}", rf"\label{{{label}}}"])
    if resize_to_width is not None:
        lines.append(rf"\resizebox{{{resize_to_width}}}{{!}}{{%")
    if fit_to_width is None:
        tabular_begin = rf"\begin{{tabular}}{{{alignment}}}"
        tabular_end = r"\end{tabular}"
    else:
        x_columns = "".join([r">{\centering\arraybackslash}X" for _ in flat_columns])
        tabular_begin = (
            rf"\begin{{tabularx}}{{{fit_to_width}}}"
            rf"{{>{{\raggedright\arraybackslash}}m{{{row_label_width}}}{x_columns}}}"
        )
        tabular_end = r"\end{tabularx}"
    lines.extend([tabular_begin, r"\hline"])
    top_header = [latex_escape(row_label)]
    top_header.extend(
        rf"\multicolumn{{{len(columns)}}}{{c}}{{{latex_escape(group)}}}"
        for group, columns in groups
    )
    lines.append(" & ".join(top_header) + r" \\")
    if second_header_row_label is None:
        rendered_second_header_row_label = ""
    elif second_header_row_label_raw:
        rendered_second_header_row_label = second_header_row_label
    else:
        rendered_second_header_row_label = latex_escape(second_header_row_label)
    second_header = [rendered_second_header_row_label] + [
        latex_escape(column_label) for _, columns in groups for _, column_label in columns
    ]
    lines.append(" & ".join(second_header) + r" \\")
    lines.append(r"\hline")
    for row_idx, row in enumerate(table.itertuples(index=False, name=None), start=1):
        rendered_row = [_render_row_label(row[0])]
        rendered_row.extend(_render_body_cell(value, compact_pair=True) for value in row[1:])
        lines.append(" & ".join(rendered_row) + r" \\")
        if row_idx in group_breaks:
            lines.append(break_command_map.get(row_idx, group_break_command))
    lines.extend([r"\hline", tabular_end])
    if resize_to_width is not None:
        lines.append(r"}")
    lines.append(rf"\end{{{environment}}}")
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
