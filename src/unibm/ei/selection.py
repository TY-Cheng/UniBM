"""Stable-window selection helpers for canonical extremal-index paths."""

from __future__ import annotations

import numpy as np

from .._block_grid import validate_block_sizes
from .._numeric import prefix_sum
from .models import EiPathBundle, EiStableWindow


def select_stable_path_window(
    block_sizes: np.ndarray,
    z_path: np.ndarray,
    *,
    min_points: int = 4,
    roughness_penalty: float = 0.75,
    curvature_penalty: float = 0.5,
) -> tuple[EiStableWindow, np.ndarray]:
    """Select a flat window of ``z = log(1 / theta)`` over increasing block sizes.

    ``z_path`` must be 1D and aligned with ``block_sizes``. Drop non-finite z
    entries and examine every contiguous window of at least ``min_points``
    retained levels across the full supplied range.

    Minimize variance plus weighted mean absolute first and second differences,
    divided by the square root of the window length. Differences are across
    adjacent retained levels, without adjusting for block-size spacing. Exact
    ties keep the first window encountered. Return inclusive block-size bounds
    and a boolean mask aligned with the *finite* path, not the original grid.
    This is a tuning heuristic, not a test of stationarity or constant theta.
    """
    levels = validate_block_sizes(block_sizes)
    z = np.asarray(z_path, dtype=float)
    if z.ndim != 1 or z.size != levels.size:
        raise ValueError("z_path must be one-dimensional and match block_sizes.")
    if (
        isinstance(min_points, bool)
        or not isinstance(min_points, (int, np.integer))
        or min_points < 2
    ):
        raise ValueError("min_points must be an integer at least 2.")
    roughness_penalty = float(roughness_penalty)
    if not np.isfinite(roughness_penalty) or roughness_penalty < 0.0:
        raise ValueError("roughness_penalty must be finite and non-negative.")
    curvature_penalty = float(curvature_penalty)
    if not np.isfinite(curvature_penalty) or curvature_penalty < 0.0:
        raise ValueError("curvature_penalty must be finite and non-negative.")
    mask = np.isfinite(z)
    levels = levels[mask]
    z = z[mask]
    if levels.size < min_points:
        raise ValueError("Not enough finite EI path values to select a stable window.")
    prefix_z = prefix_sum(z)
    prefix_z2 = prefix_sum(z * z)
    abs_diff1_prefix = prefix_sum(np.abs(np.diff(z)))
    abs_diff2_prefix = prefix_sum(np.abs(np.diff(np.diff(z))))
    best: tuple[float, int, int] | None = None
    for start in range(levels.size - min_points + 1):
        for stop in range(start + min_points, levels.size + 1):
            window_len = stop - start
            sum_z = prefix_z[stop] - prefix_z[start]
            sum_z2 = prefix_z2[stop] - prefix_z2[start]
            mean_z = float(sum_z / window_len)
            variance = max(float(sum_z2 / window_len - mean_z * mean_z), 0.0)
            if window_len > 1:
                roughness_total = abs_diff1_prefix[stop - 1] - abs_diff1_prefix[start]
                roughness = float(roughness_total / (window_len - 1))
            else:
                roughness = 0.0
            if window_len > 2:
                curvature_total = abs_diff2_prefix[stop - 2] - abs_diff2_prefix[start]
                curvature = float(curvature_total / (window_len - 2))
            else:
                curvature = 0.0
            score = (
                variance
                + float(roughness_penalty) * roughness
                + float(curvature_penalty) * curvature
            ) / np.sqrt(stop - start)
            if best is None or score < best[0]:
                best = (score, start, stop)
    assert best is not None
    _, start, stop = best
    selected_mask = np.zeros(mask.sum(), dtype=bool)
    selected_mask[start:stop] = True
    window = EiStableWindow(int(levels[start]), int(levels[stop - 1]))
    return window, selected_mask


def extract_stable_path_window(path: EiPathBundle) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned finite block levels and z values inside the stored window.

    Include both window endpoints; raise ``ValueError`` if no levels remain
    or if the path fixes a single block size without selecting a window.
    The returned arrays preserve the original path order.
    """
    if path.stable_window is None:
        raise ValueError("A fixed-b path has no stable window; use native BM inference.")
    finite_mask = np.isfinite(path.z_path)
    finite_levels = path.block_sizes[finite_mask]
    finite_z = path.z_path[finite_mask]
    window_mask = (finite_levels >= path.stable_window.lo) & (
        finite_levels <= path.stable_window.hi
    )
    selected_levels = finite_levels[window_mask]
    selected_z = finite_z[window_mask]
    if selected_levels.size == 0:
        raise ValueError("Stable EI window did not retain any finite transformed path values.")
    return selected_levels, selected_z
