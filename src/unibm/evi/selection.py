"""Plateau-window selection for EVI block-summary regressions."""

from __future__ import annotations

import numpy as np

from .._numeric import candidate_window_batches, prefix_sum
from .models import PlateauWindow


def select_penultimate_window(
    log_block_sizes: np.ndarray,
    log_values: np.ndarray,
    *,
    min_points: int = 5,
    curvature_penalty: float = 2.0,
) -> PlateauWindow:
    """Select the lowest-scoring contiguous window of paired log summaries.

    Inputs must be finite one-dimensional arrays of equal length, with
    strictly increasing ``log_block_sizes``. Search the full supplied range,
    scoring every contiguous window of at least ``min_points`` by
    ``(OLS MSE + curvature_penalty * curvature) / sqrt(length)``.
    Curvature is the mean absolute change between adjacent local slopes.
    Return a ``PlateauWindow`` whose start/stop and mask index the input
    arrays; stop is exclusive. This is a heuristic selection rule, not a
    statistical test for a true scaling plateau.
    """
    x = np.asarray(log_block_sizes, dtype=float)
    y = np.asarray(log_values, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or y.size != x.size:
        raise ValueError("log_block_sizes and log_values must be matching one-dimensional arrays.")
    if (
        isinstance(min_points, bool)
        or not isinstance(min_points, (int, np.integer))
        or min_points < 2
    ):
        raise ValueError("min_points must be an integer at least 2.")
    curvature_penalty = float(curvature_penalty)
    if not np.isfinite(curvature_penalty) or curvature_penalty < 0.0:
        raise ValueError("curvature_penalty must be finite and non-negative.")
    n = x.size
    if n < min_points:
        raise ValueError("Not enough positive block summaries to select a plateau.")
    if not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError("log_block_sizes must be finite and strictly increasing.")
    if not np.all(np.isfinite(y)):
        raise ValueError("log_values must be finite.")
    # Prefix moments make each candidate OLS score independent of window length.
    prefix_x = prefix_sum(x)
    prefix_y = prefix_sum(y)
    prefix_x2 = prefix_sum(x * x)
    prefix_xy = prefix_sum(x * y)
    prefix_y2 = prefix_sum(y * y)
    local_slopes = np.diff(y) / np.diff(x)
    slope_curvature_prefix = prefix_sum(np.abs(np.diff(local_slopes)))
    best: tuple[float, int, int] | None = None
    for start, stop in candidate_window_batches(n, min_points):
        window_len = stop - start
        sum_x = prefix_x[stop] - prefix_x[start]
        sum_y = prefix_y[stop] - prefix_y[start]
        sum_x2 = prefix_x2[stop] - prefix_x2[start]
        sum_xy = prefix_xy[stop] - prefix_xy[start]
        sum_y2 = prefix_y2[stop] - prefix_y2[start]
        denominator = window_len * sum_x2 - sum_x * sum_x
        degenerate = denominator <= 0
        slope = np.divide(
            window_len * sum_xy - sum_x * sum_y,
            denominator,
            out=np.zeros(len(start)),
            where=~degenerate,
        )
        intercept = (sum_y - slope * sum_x) / window_len
        sse = (
            sum_y2
            - 2.0 * intercept * sum_y
            - 2.0 * slope * sum_xy
            + window_len * intercept * intercept
            + 2.0 * intercept * slope * sum_x
            + slope * slope * sum_x2
        )
        mse = np.maximum(sse / window_len, 0.0)
        # Prefix subtraction can cancel for nearly identical large x values.
        # Keep the original least-squares fallback for those windows only.
        for i in np.flatnonzero(degenerate):
            a, b = start[i], stop[i]
            X = np.column_stack([np.ones(b - a, dtype=float), x[a:b]])
            beta, *_ = np.linalg.lstsq(X, y[a:b], rcond=None)
            mse[i] = np.mean((y[a:b] - X @ beta) ** 2)
        curvature = np.zeros(len(start))
        curved = window_len > 2
        curvature[curved] = (
            slope_curvature_prefix[stop[curved] - 2] - slope_curvature_prefix[start[curved]]
        ) / (window_len[curved] - 2)
        score = (mse + curvature_penalty * curvature) / np.sqrt(window_len)
        i = int(np.argmin(np.where(np.isnan(score), np.inf, score)))
        # Preserve strict-< selection even if finite inputs overflow a score.
        if best is None and np.isnan(score[0]):
            i = 0
        if best is None or score[i] < best[0]:
            best = (float(score[i]), int(start[i]), int(stop[i]))
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
