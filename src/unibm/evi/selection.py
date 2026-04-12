"""Plateau-window selection for EVI block-summary regressions."""

from __future__ import annotations

import numpy as np

from .._numeric import prefix_sum
from .models import PlateauWindow


def select_penultimate_window(
    log_block_sizes: np.ndarray,
    log_values: np.ndarray,
    *,
    min_points: int = 5,
    trim_fraction: float = 0.15,
    curvature_penalty: float = 2.0,
) -> PlateauWindow:
    """Choose an intermediate block-size window by balancing linearity and curvature."""
    x = np.asarray(log_block_sizes, dtype=float)
    y = np.asarray(log_values, dtype=float)
    n = x.size
    if n < min_points:
        raise ValueError("Not enough positive block summaries to select a plateau.")
    lo = int(np.floor(n * trim_fraction))
    hi = n - lo
    lo = min(lo, max(n - min_points, 0))
    if hi - lo < min_points:
        lo = 0
        hi = n
    prefix_x = prefix_sum(x)
    prefix_y = prefix_sum(y)
    prefix_x2 = prefix_sum(x * x)
    prefix_xy = prefix_sum(x * y)
    prefix_y2 = prefix_sum(y * y)
    local_slopes = np.diff(y) / np.diff(x)
    slope_curvature_prefix = prefix_sum(np.abs(np.diff(local_slopes)))
    best: tuple[float, int, int] | None = None
    for start in range(lo, hi - min_points + 1):
        for stop in range(start + min_points, hi + 1):
            window_len = stop - start
            sum_x = prefix_x[stop] - prefix_x[start]
            sum_y = prefix_y[stop] - prefix_y[start]
            sum_x2 = prefix_x2[stop] - prefix_x2[start]
            sum_xy = prefix_xy[stop] - prefix_xy[start]
            sum_y2 = prefix_y2[stop] - prefix_y2[start]
            denominator = window_len * sum_x2 - sum_x * sum_x
            if denominator <= 0:
                X = np.column_stack([np.ones(window_len, dtype=float), x[start:stop]])
                beta, *_ = np.linalg.lstsq(X, y[start:stop], rcond=None)
                resid = y[start:stop] - (X @ beta)
                mse = float(np.mean(resid**2))
            else:
                slope = (window_len * sum_xy - sum_x * sum_y) / denominator
                intercept = (sum_y - slope * sum_x) / window_len
                sse = (
                    sum_y2
                    - 2.0 * intercept * sum_y
                    - 2.0 * slope * sum_xy
                    + window_len * intercept * intercept
                    + 2.0 * intercept * slope * sum_x
                    + slope * slope * sum_x2
                )
                mse = max(float(sse) / window_len, 0.0)
            if window_len > 2:
                curvature_total = slope_curvature_prefix[stop - 2] - slope_curvature_prefix[start]
                curvature = float(curvature_total / (window_len - 2))
            else:
                curvature = 0.0
            score = (mse + float(curvature_penalty) * curvature) / np.sqrt(window_len)
            if best is None or score < best[0]:
                best = (score, start, stop)
    assert best is not None
    _, start, stop = best
    mask = np.zeros(n, dtype=bool)
    mask[start:stop] = True
    return PlateauWindow(
        start=start,
        stop=stop,
        score=float(best[0]),
        mask=mask,
        x=x[start:stop],
        y=y[start:stop],
    )
