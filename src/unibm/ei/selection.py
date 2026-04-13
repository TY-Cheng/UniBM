"""Stable-window selection helpers for canonical extremal-index paths."""

from __future__ import annotations

import numpy as np

from .._numeric import prefix_sum
from .models import EiPathBundle, EiStableWindow


def select_stable_path_window(
    block_sizes: np.ndarray,
    z_path: np.ndarray,
    *,
    min_points: int = 4,
    trim_fraction: float = 0.15,
    roughness_penalty: float = 0.75,
    curvature_penalty: float = 0.5,
) -> tuple[EiStableWindow, np.ndarray]:
    """Choose the most stable contiguous block-size window on the transformed EI path."""
    levels = np.asarray(block_sizes, dtype=int)
    z = np.asarray(z_path, dtype=float)
    mask = np.isfinite(z)
    levels = levels[mask]
    z = z[mask]
    if levels.size < min_points:
        raise ValueError("Not enough finite EI path values to select a stable window.")
    lo = int(np.floor(levels.size * trim_fraction))
    hi = levels.size - lo
    if hi - lo < min_points:
        lo = 0
        hi = levels.size
    prefix_z = prefix_sum(z)
    prefix_z2 = prefix_sum(z * z)
    abs_diff1_prefix = prefix_sum(np.abs(np.diff(z)))
    abs_diff2_prefix = prefix_sum(np.abs(np.diff(np.diff(z))))
    best: tuple[float, int, int] | None = None
    for start in range(lo, hi - min_points + 1):
        for stop in range(start + min_points, hi + 1):
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
    """Return the selected stable block levels and transformed values for one path."""
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
