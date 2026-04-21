"""Application-side stationarity, scaling-fit, and DLL uncertainty diagnostics."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from application.specs import ApplicationBundle
from unibm.ei import EiStableWindow, estimate_pooled_bm_ei
from unibm.ei.bootstrap import bootstrap_bm_ei_path_draws
from unibm.evi import (
    DEFAULT_CURVATURE_PENALTY,
    PlateauWindow,
    ScalingFit,
    estimate_design_life_level,
    estimate_design_life_level_interval,
    estimate_target_scaling,
)
from unibm._bootstrap_sampling import draw_circular_block_bootstrap_samples


APPLICATION_DIAGNOSTIC_TOP_K = 3


def _format_compact_number(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    magnitude = abs(float(value))
    if magnitude == 0:
        return "0"
    if magnitude >= 1e4 or magnitude < 1e-2:
        formatted = f"{float(value):.3g}"
        return formatted.replace("e+0", "e").replace("e+", "e").replace("e0", "e0")
    if magnitude >= 100:
        return f"{float(value):.0f}"
    if magnitude >= 1:
        return f"{float(value):.2f}"
    return f"{float(value):.3f}"


def _format_interval(center: float, lo: float, hi: float) -> str:
    if not (np.isfinite(center) and np.isfinite(lo) and np.isfinite(hi)):
        return "NA"
    return (
        f"{_format_compact_number(center)} "
        f"[{_format_compact_number(lo)}, {_format_compact_number(hi)}]"
    )


def _format_range(lo: float, hi: float, *, precision: int = 2) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "NA"
    if precision == 2:
        return f"[{_format_compact_number(lo)}, {_format_compact_number(hi)}]"
    return f"[{lo:.{precision}f}, {hi:.{precision}f}]"


def _top_penultimate_windows(
    fit: ScalingFit,
    *,
    top_k: int = APPLICATION_DIAGNOSTIC_TOP_K,
    min_points: int = 5,
    trim_fraction: float = 0.15,
    curvature_penalty: float = DEFAULT_CURVATURE_PENALTY,
) -> list[PlateauWindow]:
    x = np.asarray(fit.log_block_sizes, dtype=float)
    y = np.asarray(fit.log_values, dtype=float)
    n_obs = x.size
    lo = int(np.floor(n_obs * trim_fraction))
    hi = n_obs - lo
    lo = min(lo, max(n_obs - min_points, 0))
    if hi - lo < min_points:
        lo = 0
        hi = n_obs
    candidates: list[tuple[float, int, int]] = []
    for start in range(lo, hi - min_points + 1):
        for stop in range(start + min_points, hi + 1):
            window_x = x[start:stop]
            window_y = y[start:stop]
            slope, intercept = np.polyfit(window_x, window_y, 1)
            fitted = intercept + slope * window_x
            resid = window_y - fitted
            mse = float(np.mean(resid**2))
            local_slopes = np.diff(window_y) / np.diff(window_x)
            curvature = (
                float(np.mean(np.abs(np.diff(local_slopes)))) if local_slopes.size > 1 else 0.0
            )
            score = (mse + float(curvature_penalty) * curvature) / np.sqrt(stop - start)
            candidates.append((float(score), int(start), int(stop)))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    windows: list[PlateauWindow] = []
    seen: set[tuple[int, int]] = set()
    for score, start, stop in candidates:
        key = (start, stop)
        if key in seen:
            continue
        seen.add(key)
        mask = np.zeros(n_obs, dtype=bool)
        mask[start:stop] = True
        windows.append(
            PlateauWindow(
                start=start,
                stop=stop,
                score=score,
                mask=mask,
                x=x[start:stop],
                y=y[start:stop],
            )
        )
        if len(windows) >= top_k:
            break
    return windows


def fit_evi_window_variants(
    bundle: ApplicationBundle,
    *,
    top_k: int = APPLICATION_DIAGNOSTIC_TOP_K,
) -> list[ScalingFit]:
    fits: list[ScalingFit] = []
    for plateau in _top_penultimate_windows(bundle.evi_fit, top_k=top_k):
        fits.append(
            estimate_target_scaling(
                bundle.prepared.evi.series.values,
                target="quantile",
                quantile=bundle.spec.quantile,
                sliding=True,
                bootstrap_reps=0,
                curve=bundle.evi_fit.curve,
                plateau=plateau,
                bootstrap_result=bundle.evi_fit.bootstrap,
            )
        )
    return fits


def application_design_life_interval_record(bundle: ApplicationBundle) -> dict[str, object]:
    observations_per_year = application_observations_per_year(bundle)
    variants = fit_evi_window_variants(bundle, top_k=APPLICATION_DIAGNOSTIC_TOP_K)
    dll10_env = np.asarray(
        [
            estimate_design_life_level(
                variant,
                10.0,
                observations_per_year=observations_per_year,
            )
            for variant in variants
        ],
        dtype=float,
    )
    dll50_env = np.asarray(
        [
            estimate_design_life_level(
                variant,
                50.0,
                observations_per_year=observations_per_year,
            )
            for variant in variants
        ],
        dtype=float,
    )
    dll10 = float(
        estimate_design_life_level(
            bundle.evi_fit,
            10.0,
            observations_per_year=observations_per_year,
        )
    )
    dll10_lo, dll10_hi = estimate_design_life_level_interval(
        bundle.evi_fit,
        years=10.0,
        observations_per_year=observations_per_year,
    )
    dll50 = float(
        estimate_design_life_level(
            bundle.evi_fit,
            50.0,
            observations_per_year=observations_per_year,
        )
    )
    dll50_lo, dll50_hi = estimate_design_life_level_interval(
        bundle.evi_fit,
        years=50.0,
        observations_per_year=observations_per_year,
    )
    return {
        "application": bundle.spec.key,
        "label": bundle.spec.label,
        "provider": bundle.spec.provider,
        "design_life_level_basis": bundle.spec.design_life_level_basis,
        "tau": float(bundle.evi_fit.quantile),
        "dll10": dll10,
        "dll10_lo": float(dll10_lo),
        "dll10_hi": float(dll10_hi),
        "dll10_env_lo": float(np.min(dll10_env)) if dll10_env.size else float("nan"),
        "dll10_env_hi": float(np.max(dll10_env)) if dll10_env.size else float("nan"),
        "dll50": dll50,
        "dll50_lo": float(dll50_lo),
        "dll50_hi": float(dll50_hi),
        "dll50_env_lo": float(np.min(dll50_env)) if dll50_env.size else float("nan"),
        "dll50_env_hi": float(np.max(dll50_env)) if dll50_env.size else float("nan"),
    }


def application_design_life_interval_table(bundles: list[ApplicationBundle]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for bundle in manuscript_bundles(bundles):
        record = application_design_life_interval_record(bundle)
        basis = str(record["design_life_level_basis"]).replace("_", "-")
        rows.append(
            {
                "Application": record["label"],
                "Basis": basis,
                "10y median DLL": _format_compact_number(float(record["dll10"])),
                "10y conditional 95% CI": _format_range(
                    float(record["dll10_lo"]),
                    float(record["dll10_hi"]),
                ),
                "10y plateau envelope": _format_range(
                    float(record["dll10_env_lo"]),
                    float(record["dll10_env_hi"]),
                ),
                "50y median DLL": _format_compact_number(float(record["dll50"])),
                "50y conditional 95% CI": _format_range(
                    float(record["dll50_lo"]),
                    float(record["dll50_hi"]),
                ),
                "50y plateau envelope": _format_range(
                    float(record["dll50_env_lo"]),
                    float(record["dll50_env_hi"]),
                ),
            }
        )
    return pd.DataFrame(rows)


def manuscript_bundles(bundles: list[ApplicationBundle]) -> list[ApplicationBundle]:
    keys = (
        "tx_streamflow",
        "fl_streamflow",
        "tx_nfip_claims",
        "fl_nfip_claims",
    )
    order = {key: idx for idx, key in enumerate(keys)}
    return sorted(
        [bundle for bundle in bundles if bundle.spec.key in order],
        key=lambda bundle: order[bundle.spec.key],
    )


def application_observations_per_year(bundle: ApplicationBundle) -> float:
    if bundle.spec.observations_per_year is not None:
        return float(bundle.spec.observations_per_year)
    series = bundle.prepared.evi.series
    n_years = max((series.index.max() - series.index.min()).days / 365.25, 1.0)
    return float(series.size / n_years)


def _top_ei_windows(
    path,
    *,
    top_k: int = APPLICATION_DIAGNOSTIC_TOP_K,
    min_points: int = 4,
    trim_fraction: float = 0.15,
    roughness_penalty: float = 0.75,
    curvature_penalty: float = 0.5,
) -> list[tuple[EiStableWindow, np.ndarray, float]]:
    levels = np.asarray(path.block_sizes, dtype=int)
    z_path = np.asarray(path.z_path, dtype=float)
    mask = np.isfinite(z_path)
    levels = levels[mask]
    z_path = z_path[mask]
    lo = int(np.floor(levels.size * trim_fraction))
    hi = levels.size - lo
    if hi - lo < min_points:
        lo = 0
        hi = levels.size
    candidates: list[tuple[float, int, int]] = []
    for start in range(lo, hi - min_points + 1):
        for stop in range(start + min_points, hi + 1):
            window = z_path[start:stop]
            variance = float(np.mean((window - window.mean()) ** 2))
            first_diff = np.diff(window)
            roughness = float(np.mean(np.abs(first_diff))) if first_diff.size else 0.0
            curvature = float(np.mean(np.abs(np.diff(first_diff)))) if first_diff.size > 1 else 0.0
            score = (
                variance
                + float(roughness_penalty) * roughness
                + float(curvature_penalty) * curvature
            ) / np.sqrt(stop - start)
            candidates.append((float(score), int(start), int(stop)))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    windows: list[tuple[EiStableWindow, np.ndarray, float]] = []
    seen: set[tuple[int, int]] = set()
    for score, start, stop in candidates:
        key = (start, stop)
        if key in seen:
            continue
        seen.add(key)
        selected_levels = levels[start:stop]
        windows.append(
            (
                EiStableWindow(int(selected_levels[0]), int(selected_levels[-1])),
                selected_levels,
                float(score),
            )
        )
        if len(windows) >= top_k:
            break
    return windows


def _bootstrap_ei_path_draws_for_bundle(
    bundle: ApplicationBundle,
) -> dict[tuple[str, bool], np.ndarray]:
    assert bundle.ei_bundle is not None
    sample_bank = draw_circular_block_bootstrap_samples(
        bundle.prepared.ei.series.to_numpy(dtype=float),
        reps=120,
        random_state=7,
    )
    return bootstrap_bm_ei_path_draws(
        sample_bank.samples,
        block_sizes=bundle.ei_bundle.block_sizes,
        allow_zeros=bundle.spec.ei_allow_zeros,
    )


def _materialize_application_ei_bootstrap_result(
    selected_levels: np.ndarray,
    *,
    bundle: ApplicationBundle,
    path_draws: dict[tuple[str, bool], np.ndarray],
    base_path: str,
    sliding: bool,
) -> dict[str, np.ndarray | None]:
    assert bundle.ei_bundle is not None
    full_levels = np.asarray(bundle.ei_bundle.block_sizes, dtype=int)
    selected_idx = [int(np.flatnonzero(full_levels == level)[0]) for level in selected_levels]
    selected_draws = path_draws[(base_path, sliding)][:, selected_idx]
    valid_draws = selected_draws[np.all(np.isfinite(selected_draws), axis=1)]
    covariance = None
    if valid_draws.shape[0] >= 2:
        covariance = np.atleast_2d(np.cov(valid_draws, rowvar=False))
    return {
        "block_sizes": np.asarray(selected_levels, dtype=int),
        "samples": valid_draws,
        "covariance": covariance,
    }


def fit_ei_window_variants(
    bundle: ApplicationBundle,
    *,
    top_k: int = APPLICATION_DIAGNOSTIC_TOP_K,
) -> list[tuple[object, float]]:
    if bundle.ei_bundle is None or bundle.ei_bb_sliding_fgls is None:
        return []
    path = bundle.ei_bundle.paths[("bb", True)]
    path_draws = _bootstrap_ei_path_draws_for_bundle(bundle)
    variants: list[tuple[object, float]] = []
    for window, selected_levels, score in _top_ei_windows(path, top_k=top_k):
        bootstrap_result = _materialize_application_ei_bootstrap_result(
            selected_levels,
            bundle=bundle,
            path_draws=path_draws,
            base_path="bb",
            sliding=True,
        )
        updated_path = replace(
            path,
            stable_window=window,
            selected_level=int(selected_levels[0]),
        )
        updated_paths = dict(bundle.ei_bundle.paths)
        updated_paths[("bb", True)] = updated_path
        updated_bundle = replace(bundle.ei_bundle, paths=updated_paths)
        estimate = estimate_pooled_bm_ei(
            updated_bundle,
            base_path="bb",
            sliding=True,
            regression="FGLS",
            bootstrap_result=bootstrap_result,
        )
        variants.append((estimate, float(score)))
    return variants


__all__ = [
    "APPLICATION_DIAGNOSTIC_TOP_K",
    "application_design_life_interval_record",
    "application_design_life_interval_table",
    "application_observations_per_year",
    "fit_ei_window_variants",
    "fit_evi_window_variants",
    "manuscript_bundles",
]
