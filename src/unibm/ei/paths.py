"""BM-path construction for canonical extremal-index workflows."""

from __future__ import annotations

import numpy as np

from ..cdf import empirical_cdf
from .._window_ops import sliding_window_extreme_valid
from ._stats import EI_TINY
from .models import EiPathBundle
from .selection import select_stable_path_window


BM_PATH_KEYS = (
    ("northrop", True),
    ("northrop", False),
    ("bb", True),
    ("bb", False),
)


def _rolling_window_minima(
    scores: np.ndarray,
    block_size: int,
    *,
    sliding: bool,
) -> np.ndarray:
    """Return minima of finite windows of ``block_size`` scores.

    Sliding windows advance one observation; disjoint windows discard an
    incomplete trailing block. Windows containing non-finite scores are omitted.
    Return an empty array for block sizes below two or above the series length.
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if scores.size < block_size or block_size < 2:
        return np.asarray([], dtype=float)
    if sliding:
        return sliding_window_extreme_valid(scores, block_size, reducer="min")
    n_block = scores.size // block_size
    if n_block < 1:
        return np.asarray([], dtype=float)
    windows = scores[: n_block * block_size].reshape(n_block, block_size)
    valid = np.all(np.isfinite(windows), axis=1)
    return windows.min(axis=1)[valid]


def _path_point_from_statistics(
    base_path: str,
    statistics: np.ndarray,
    *,
    block_size: int,
) -> tuple[float, float, float]:
    """Map a non-empty block-statistic sample to ``(theta, 1 / theta, log(1 / theta))``.

    Northrop uses the reciprocal sample mean. BB subtracts ``1 / block_size``
    from that reciprocal; both paths cap theta at 1, with BB also using
    ``EI_TINY`` as a positive numerical floor.
    """
    mean_stat = float(np.mean(statistics))
    if base_path == "northrop":
        eir = max(mean_stat, 1.0)
        theta = float(1.0 / eir)
    elif base_path == "bb":
        theta = float(max((1.0 / max(mean_stat, EI_TINY)) - 1.0 / float(block_size), EI_TINY))
        theta = min(theta, 1.0)
        eir = float(1.0 / theta)
    else:
        raise ValueError(f"Unknown BM EI base path: {base_path}")
    return theta, eir, float(np.log(eir))


def _compute_path_arrays_from_scores(
    base_path: str,
    scores: np.ndarray,
    block_sizes: np.ndarray,
    *,
    sliding: bool,
    collect_statistics: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray] | None]:
    """Compute aligned theta, reciprocal-theta, z, and window-count arrays.

    For each block size, multiply score-window minima by that size. Levels with
    no valid windows retain NaNs and a zero count. The fifth return value is a
    block-size-to-statistics dictionary when requested, otherwise ``None``.
    """
    theta_path = np.full(block_sizes.size, np.nan, dtype=float)
    eir_path = np.full(block_sizes.size, np.nan, dtype=float)
    z_path = np.full(block_sizes.size, np.nan, dtype=float)
    sample_counts = np.zeros(block_sizes.size, dtype=int)
    sample_statistics = {} if collect_statistics else None
    for idx, block_size in enumerate(np.asarray(block_sizes, dtype=int)):
        minima = _rolling_window_minima(scores, int(block_size), sliding=sliding)
        if minima.size == 0:
            continue
        statistics = float(block_size) * minima
        if sample_statistics is not None:
            sample_statistics[int(block_size)] = statistics
        sample_counts[idx] = statistics.size
        theta, eir, z = _path_point_from_statistics(
            base_path,
            statistics,
            block_size=int(block_size),
        )
        theta_path[idx] = theta
        eir_path[idx] = eir
        z_path[idx] = z
    return theta_path, eir_path, z_path, sample_counts, sample_statistics


def _build_path_from_scores(
    base_path: str,
    scores: np.ndarray,
    block_sizes: np.ndarray,
    *,
    sliding: bool,
) -> EiPathBundle:
    """Construct one BM path, selecting a window only for a multilevel grid.

    Retain per-level statistics for native inference. ``selected_level`` is the
    sole supplied block size or the smallest size inside the selected window.
    A fixed single level has no selected stable window.
    """
    theta_path, eir_path, z_path, sample_counts, sample_statistics = (
        _compute_path_arrays_from_scores(
            base_path,
            np.asarray(scores, dtype=float),
            np.asarray(block_sizes, dtype=int),
            sliding=sliding,
            collect_statistics=True,
        )
    )
    if block_sizes.size == 1:
        stable_window = None
        selected_level = int(block_sizes[0])
    else:
        stable_window, stable_mask = select_stable_path_window(block_sizes, z_path)
        selected_level = int(block_sizes[np.isfinite(z_path)][stable_mask][0])
    return EiPathBundle(
        base_path=base_path,
        sliding=bool(sliding),
        block_sizes=np.asarray(block_sizes, dtype=int),
        theta_path=theta_path,
        eir_path=eir_path,
        z_path=z_path,
        sample_counts=sample_counts,
        sample_statistics=sample_statistics or {},
        stable_window=stable_window,
        selected_level=selected_level,
    )


def _build_bm_z_paths_from_values(
    values: np.ndarray,
    block_sizes: np.ndarray,
    *,
    path_keys: tuple[tuple[str, bool], ...] = BM_PATH_KEYS,
) -> dict[tuple[str, bool], np.ndarray]:
    """Return requested z-path arrays keyed by ``(base_path, sliding)``.

    Recompute scaled empirical ranks from this series, then use ``-log(F)``
    scores for Northrop or ``1 - F`` for BB. Unlike full path preparation, this
    does not select a stable window or retain window-level statistics.
    """
    cdf_values = np.asarray(empirical_cdf(values)(values), dtype=float)
    cdf_values = np.clip(cdf_values, EI_TINY, 1.0 - EI_TINY)
    score_lookup = {
        "northrop": -np.log(cdf_values),
        "bb": 1.0 - cdf_values,
    }
    draws: dict[tuple[str, bool], np.ndarray] = {}
    for base_path, sliding in path_keys:
        _, _, z_path, _, _ = _compute_path_arrays_from_scores(
            base_path,
            score_lookup[base_path],
            np.asarray(block_sizes, dtype=int),
            sliding=sliding,
            collect_statistics=False,
        )
        draws[(base_path, sliding)] = z_path
    return draws


def _build_bm_paths_from_values(
    values: np.ndarray,
    block_sizes: np.ndarray,
    *,
    path_keys: tuple[tuple[str, bool], ...] = BM_PATH_KEYS,
) -> dict[tuple[str, bool], EiPathBundle]:
    """Build only the requested Northrop/BB and sliding/disjoint paths.

    ``values`` and ``block_sizes`` have already been validated by preparation.
    Scaled empirical ranks yield ``-log(F)`` and ``1 - F`` score series; minima
    of these decreasing transforms correspond to maxima of the original data.
    Return a dictionary keyed by ``(base_path, sliding)``. A single supplied
    block size bypasses window selection for fixed-b native inference.
    """
    cdf_values = np.asarray(empirical_cdf(values)(values), dtype=float)
    cdf_values = np.clip(cdf_values, EI_TINY, 1.0 - EI_TINY)
    score_lookup = {
        "northrop": -np.log(cdf_values),
        "bb": 1.0 - cdf_values,
    }
    return {
        (base_path, sliding): _build_path_from_scores(
            base_path,
            score_lookup[base_path],
            block_sizes,
            sliding=sliding,
        )
        for base_path, sliding in path_keys
    }
