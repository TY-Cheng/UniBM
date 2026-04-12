"""Extremal-index benchmark workflow and summary aggregation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from unibm.ei import (
    EI_ALPHA,
    ExtremalIndexEstimate,
    bootstrap_bm_ei_path_draws,
    estimate_ferro_segers,
    estimate_k_gaps,
    estimate_native_bm_ei,
    estimate_pooled_bm_ei,
    extract_stable_path_window,
    prepare_ei_bundle,
)

from benchmark.design import (
    _atomic_savez,
    _try_load_npz,
    BENCHMARK_CACHE_VERSION,
    default_ei_simulation_configs,
    load_or_draw_raw_bootstrap_samples,
    load_or_simulate_series_bank,
    ordered_families,
    resolve_benchmark_workers,
    scenario_random_state,
)
from benchmark.common import (
    IQR_LOWER,
    IQR_UPPER,
    interval_score,
    interval_width,
    quantile_agg,
    wilson_interval,
)
from shared.runtime import status

EI_INTERNAL_METHODS = [
    "northrop_disjoint_ols",
    "northrop_disjoint_fgls",
    "northrop_sliding_ols",
    "northrop_sliding_fgls",
    "bb_disjoint_ols",
    "bb_disjoint_fgls",
    "bb_sliding_ols",
    "bb_sliding_fgls",
]
EI_FGLS_METHODS = [
    "northrop_disjoint_fgls",
    "northrop_sliding_fgls",
    "bb_disjoint_fgls",
    "bb_sliding_fgls",
]
EI_TARGET_INTERNAL_METHODS = [
    "northrop_disjoint_fgls",
    "northrop_sliding_fgls",
    "bb_disjoint_fgls",
    "bb_sliding_fgls",
]
EI_EXTERNAL_METHODS = [
    "ferro_segers",
    "k_gaps",
    "northrop_sliding_native",
    "bb_sliding_native",
]
EI_ALL_METHODS = [*EI_INTERNAL_METHODS, *EI_EXTERNAL_METHODS]

EI_METHOD_LABELS = {
    "northrop_disjoint_ols": "Northrop-disjoint-OLS",
    "northrop_disjoint_fgls": "Northrop-disjoint-FGLS",
    "northrop_sliding_ols": "Northrop-sliding-OLS",
    "northrop_sliding_fgls": "Northrop-sliding-FGLS",
    "bb_disjoint_ols": "BB-disjoint-OLS",
    "bb_disjoint_fgls": "BB-disjoint-FGLS",
    "bb_sliding_ols": "BB-sliding-OLS",
    "bb_sliding_fgls": "BB-sliding-FGLS",
    "ferro_segers": "Ferro-Segers",
    "k_gaps": "K-gaps",
    "northrop_sliding_native": "Northrop",
    "bb_sliding_native": "BB",
}
EI_METHOD_COLORS = {
    "northrop_disjoint_ols": "tab:blue",
    "northrop_disjoint_fgls": "tab:blue",
    "northrop_sliding_ols": "tab:cyan",
    "northrop_sliding_fgls": "tab:cyan",
    "bb_disjoint_ols": "tab:red",
    "bb_disjoint_fgls": "tab:red",
    "bb_sliding_ols": "tab:pink",
    "bb_sliding_fgls": "tab:pink",
    "ferro_segers": "tab:orange",
    "k_gaps": "tab:green",
    "northrop_sliding_native": "teal",
    "bb_sliding_native": "purple",
}
EI_METHOD_LINESTYLES = {
    method: ("-" if "sliding" in method else "--") for method in EI_ALL_METHODS
}
EI_METHOD_LINESTYLES.update(
    {
        "ferro_segers": "--",
        "k_gaps": "--",
        "bb_sliding_native": "-",
    }
)
EI_METHOD_MARKERS = {
    "northrop_disjoint_ols": "o",
    "northrop_disjoint_fgls": "s",
    "northrop_sliding_ols": "o",
    "northrop_sliding_fgls": "s",
    "bb_disjoint_ols": "^",
    "bb_disjoint_fgls": "D",
    "bb_sliding_ols": "^",
    "bb_sliding_fgls": "D",
    "ferro_segers": "P",
    "k_gaps": "X",
    "northrop_sliding_native": "v",
    "bb_sliding_native": "<",
}
EI_BOOTSTRAP_REPS = 120
EI_BM_PATH_KEYS = (
    ("northrop", False),
    ("northrop", True),
    ("bb", False),
    ("bb", True),
)


def _ei_bootstrap_cache_file(
    cache_dir: Path,
    *,
    cache_key: str,
    reps: int,
) -> Path:
    """Return the cache path for one series-wide pooled BM EI covariance bundle."""
    return (
        cache_dir
        / "ei_internal_bootstrap"
        / f"{BENCHMARK_CACHE_VERSION}__{cache_key}__reps{reps}.npz"
    )


def _ei_bundle_prefix(base_path: str, sliding: bool) -> str:
    """Encode one EI path/scheme pair into a stable cache-key prefix."""
    return f"{base_path}__{'sliding' if sliding else 'disjoint'}"


def _load_cached_ei_bootstrap_bundle(
    *,
    cache_dir: Path | None,
    cache_key: str,
    selected_levels_by_key: dict[tuple[str, bool], np.ndarray],
    reps: int,
) -> dict[tuple[str, bool], dict[str, np.ndarray | None]] | None:
    """Load one series-wide EI covariance bundle if the selected windows still match."""
    if cache_dir is None:
        return None
    cache_file = _ei_bootstrap_cache_file(
        cache_dir,
        cache_key=cache_key,
        reps=reps,
    )
    if not cache_file.exists():
        return None
    loaded = _try_load_npz(cache_file)
    if loaded is None:
        return None
    with loaded as data:
        results: dict[tuple[str, bool], dict[str, np.ndarray | None]] = {}
        for key, selected_levels in selected_levels_by_key.items():
            selected_levels = np.asarray(selected_levels, dtype=int)
            prefix = _ei_bundle_prefix(*key)
            block_sizes_key = f"{prefix}__block_sizes"
            if block_sizes_key not in data.files:
                return None
            boot_levels = np.asarray(data[block_sizes_key], dtype=int)
            if not np.array_equal(boot_levels, selected_levels):
                return None
            covariance = np.asarray(data[f"{prefix}__covariance"], dtype=float)
            if covariance.size == 0:
                covariance = None
            results[key] = {
                "block_sizes": boot_levels,
                "samples": np.asarray(data[f"{prefix}__samples"], dtype=float),
                "covariance": covariance,
            }
        return results


def _save_cached_ei_bootstrap_bundle(
    *,
    cache_dir: Path | None,
    cache_key: str,
    reps: int,
    bundles: dict[tuple[str, bool], dict[str, np.ndarray | None]],
) -> None:
    """Persist all pooled BM EI covariance bundles for one series in one file."""
    if cache_dir is None:
        return
    arrays: dict[str, Any] = {}
    for key, result in bundles.items():
        prefix = _ei_bundle_prefix(*key)
        arrays[f"{prefix}__block_sizes"] = np.asarray(result["block_sizes"], dtype=int)
        arrays[f"{prefix}__samples"] = np.asarray(result["samples"], dtype=float)
        covariance = result.get("covariance")
        arrays[f"{prefix}__covariance"] = (
            np.asarray(covariance, dtype=float)
            if covariance is not None
            else np.empty((0, 0), dtype=float)
        )
    cache_file = _ei_bootstrap_cache_file(
        cache_dir,
        cache_key=cache_key,
        reps=reps,
    )
    _atomic_savez(cache_file, compressed=False, **arrays)


def _materialize_ei_bootstrap(
    transformed_draws: np.ndarray,
    *,
    selected_levels: np.ndarray,
    reps: int,
) -> dict[str, np.ndarray | None]:
    """Build and optionally cache one pooled BM EI covariance bundle."""
    selected_levels = np.asarray(selected_levels, dtype=int)
    z_draws = np.asarray(transformed_draws, dtype=float)
    if z_draws.shape != (reps, selected_levels.size):
        raise ValueError("Transformed EI bootstrap draws do not match the selected level grid.")
    z_valid = z_draws[np.all(np.isfinite(z_draws), axis=1)]
    covariance = None
    if z_valid.shape[0] >= 2:
        covariance = np.atleast_2d(np.cov(z_valid, rowvar=False))
    return {
        "block_sizes": selected_levels,
        "samples": z_valid,
        "covariance": covariance,
    }


def _load_or_compute_ei_bootstrap_bundle(
    vec: np.ndarray,
    *,
    bundle: Any,
    cache_dir: Path | None,
    cache_key: str,
    reps: int,
    random_state: int,
) -> dict[tuple[str, bool], dict[str, np.ndarray | None]]:
    """Materialize all pooled-BM EI covariance bundles from one shared raw bootstrap bank.

    The first cache layer stores raw circular bootstrap series shared across
    Northrop/BB and sliding/disjoint variants. The second cache layer stores
    the path-specific transformed `z = log(1/theta)` draws restricted to the
    original replicate's stable window.
    """
    selected_levels_by_key = {
        key: extract_stable_path_window(bundle.paths[key])[0] for key in EI_BM_PATH_KEYS
    }
    cached = _load_cached_ei_bootstrap_bundle(
        cache_dir=cache_dir,
        cache_key=cache_key,
        selected_levels_by_key=selected_levels_by_key,
        reps=reps,
    )
    if cached is not None:
        return cached
    raw_bootstrap_samples = load_or_draw_raw_bootstrap_samples(
        vec,
        cache_dir=cache_dir,
        cache_key=cache_key,
        bootstrap_reps=reps,
        random_state=random_state,
    )
    full_draws = bootstrap_bm_ei_path_draws(raw_bootstrap_samples, block_sizes=bundle.block_sizes)
    results: dict[tuple[str, bool], dict[str, np.ndarray | None]] = {}
    for key in EI_BM_PATH_KEYS:
        base_path, sliding = key
        selected_levels = selected_levels_by_key[key]
        full_levels = np.asarray(bundle.block_sizes, dtype=int)
        idx = [int(np.flatnonzero(full_levels == level)[0]) for level in selected_levels]
        results[key] = _materialize_ei_bootstrap(
            full_draws[key][:, idx],
            selected_levels=selected_levels,
            reps=reps,
        )
    _save_cached_ei_bootstrap_bundle(
        cache_dir=cache_dir,
        cache_key=cache_key,
        reps=reps,
        bundles=results,
    )
    return results


def _ei_result_row(
    cfg: Any,
    rep: int,
    estimate: ExtremalIndexEstimate,
) -> dict[str, Any]:
    """Convert one EI estimate into the shared detail-row schema."""
    ci_lo, ci_hi = estimate.confidence_interval
    abs_error = abs(estimate.theta_hat - cfg.theta_true)
    return {
        "benchmark_set": cfg.benchmark_set,
        "family": cfg.family,
        "scenario": cfg.scenario,
        "rep": rep,
        "n_obs": cfg.n_obs,
        "method": estimate.method,
        "method_label": EI_METHOD_LABELS[estimate.method],
        "xi_true": cfg.xi_true,
        "theta_true": cfg.theta_true,
        "phi": cfg.phi,
        "theta_hat": estimate.theta_hat,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "ci_method": estimate.ci_method,
        "ci_variant": estimate.ci_variant,
        "standard_error": estimate.standard_error,
        "signed_error": estimate.theta_hat - cfg.theta_true,
        "abs_error": abs_error,
        "relative_error": abs_error / cfg.theta_true,
        "interval_width": interval_width(ci_lo, ci_hi),
        "interval_score": interval_score(cfg.theta_true, ci_lo, ci_hi, alpha=EI_ALPHA),
        "covered": bool(
            np.isfinite(ci_lo) and np.isfinite(ci_hi) and ci_lo <= cfg.theta_true <= ci_hi
        ),
        "selected_level": (
            np.nan if estimate.selected_level is None else float(estimate.selected_level)
        ),
        "stable_level_lo": (
            np.nan if estimate.stable_window is None else float(estimate.stable_window.lo)
        ),
        "stable_level_hi": (
            np.nan if estimate.stable_window is None else float(estimate.stable_window.hi)
        ),
        "selected_threshold_quantile": (
            np.nan
            if estimate.selected_threshold_quantile is None
            else float(estimate.selected_threshold_quantile)
        ),
        "selected_threshold_value": (
            np.nan
            if estimate.selected_threshold_value is None
            else float(estimate.selected_threshold_value)
        ),
        "selected_run_k": (
            np.nan if estimate.selected_run_k is None else float(estimate.selected_run_k)
        ),
        "block_scheme": estimate.block_scheme,
        "base_path": estimate.base_path,
        "regression": estimate.regression,
    }


def _collapse_group_flag(values: pd.Series) -> str:
    """Keep a group-level flag readable when replicate-level variants differ."""
    labels = [str(value) for value in values.dropna().unique()]
    if not labels:
        return "default"
    return labels[0] if len(labels) == 1 else "mixed"


def summarize_ei_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate replicate-level EI results to the reporting layer."""
    grouped = (
        df.groupby(
            ["benchmark_set", "family", "xi_true", "theta_true", "phi", "method", "ci_method"],
            dropna=False,
            as_index=False,
        )
        .agg(
            n_obs=("n_obs", "median"),
            n_rep=("rep", "nunique"),
            n_cover=("covered", "sum"),
            theta_hat_mean=("theta_hat", "mean"),
            theta_hat_sd=("theta_hat", "std"),
            bias=("signed_error", "mean"),
            mae=("abs_error", "mean"),
            mape=("relative_error", "mean"),
            mape_sd=("relative_error", "std"),
            ape_median=("relative_error", "median"),
            ape_q25=("relative_error", quantile_agg(IQR_LOWER)),
            ape_q75=("relative_error", quantile_agg(IQR_UPPER)),
            interval_width_mean=("interval_width", "mean"),
            interval_width_median=("interval_width", "median"),
            interval_width_q25=("interval_width", quantile_agg(IQR_LOWER)),
            interval_width_q75=("interval_width", quantile_agg(IQR_UPPER)),
            interval_score_mean=("interval_score", "mean"),
            interval_score_median=("interval_score", "median"),
            interval_score_q25=("interval_score", quantile_agg(IQR_LOWER)),
            interval_score_q75=("interval_score", quantile_agg(IQR_UPPER)),
            coverage=("covered", "mean"),
            selected_level=("selected_level", "median"),
            stable_level_lo=("stable_level_lo", "median"),
            stable_level_hi=("stable_level_hi", "median"),
            selected_threshold_quantile=("selected_threshold_quantile", "median"),
            selected_run_k=("selected_run_k", "median"),
            block_scheme=("block_scheme", "first"),
            base_path=("base_path", "first"),
            regression=("regression", "first"),
            ci_variant=("ci_variant", _collapse_group_flag),
        )
        .reset_index(drop=True)
    )
    grouped["theta_hat_sd"] = grouped["theta_hat_sd"].fillna(0.0)
    grouped["mape_sd"] = grouped["mape_sd"].fillna(0.0)
    grouped["theta_hat_se"] = grouped["theta_hat_sd"] / np.sqrt(grouped["n_rep"])
    wilson_bounds = grouped.apply(
        lambda row: wilson_interval(row["n_cover"], int(row["n_rep"])),
        axis=1,
        result_type="expand",
    )
    grouped["coverage_lo"] = wilson_bounds[0]
    grouped["coverage_hi"] = wilson_bounds[1]
    grouped["scenario"] = grouped.apply(
        lambda row: (
            f"{row['benchmark_set']}_{row['family']}_xi{row['xi_true']:.2f}"
            f"_theta{row['theta_true']:.2f}"
        ),
        axis=1,
    )
    grouped["method_label"] = grouped["method"].map(EI_METHOD_LABELS)
    grouped["family"] = pd.Categorical(
        grouped["family"],
        categories=ordered_families(grouped["family"]),
        ordered=True,
    )
    return grouped.sort_values(
        ["benchmark_set", "family", "xi_true", "theta_true", "method"]
    ).reset_index(drop=True)


def evaluate_ei_config(
    cfg: Any,
    *,
    random_state: int = 0,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all internal and external EI methods on one benchmark scenario."""
    internal_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    series_bank = load_or_simulate_series_bank(cfg, random_state=random_state, cache_dir=cache_dir)
    for rep, vec in enumerate(series_bank):
        bundle = prepare_ei_bundle(vec)
        cache_key = f"{cfg.scenario}__seed{random_state}__rep{rep:04d}"
        bootstrap_results = _load_or_compute_ei_bootstrap_bundle(
            vec,
            bundle=bundle,
            cache_dir=cache_dir,
            cache_key=cache_key,
            reps=EI_BOOTSTRAP_REPS,
            random_state=random_state + 10_000 * rep,
        )
        for base_path in ("northrop", "bb"):
            for sliding in (False, True):
                internal_rows.append(
                    _ei_result_row(
                        cfg,
                        rep,
                        estimate_pooled_bm_ei(
                            bundle,
                            base_path=base_path,
                            sliding=sliding,
                            regression="OLS",
                        ),
                    )
                )
                internal_rows.append(
                    _ei_result_row(
                        cfg,
                        rep,
                        estimate_pooled_bm_ei(
                            bundle,
                            base_path=base_path,
                            sliding=sliding,
                            regression="FGLS",
                            bootstrap_result=bootstrap_results[(base_path, sliding)],
                        ),
                    )
                )
            # Native EI comparators follow the original semiparametric BM papers:
            # use the sliding-blocks version together with each method's own CI.
            external_rows.append(
                _ei_result_row(
                    cfg,
                    rep,
                    estimate_native_bm_ei(
                        bundle,
                        base_path=base_path,
                        sliding=True,
                        use_adjusted_chandwich=(base_path == "northrop"),
                    ),
                )
            )
        external_rows.append(_ei_result_row(cfg, rep, estimate_ferro_segers(bundle)))
        external_rows.append(_ei_result_row(cfg, rep, estimate_k_gaps(bundle)))
    return pd.DataFrame(internal_rows), pd.DataFrame(external_rows)


def _evaluate_ei_config_worker(
    args: tuple[Any, int, Path | None],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process-pool wrapper for one EI benchmark scenario."""
    cfg, random_state, cache_dir = args
    return evaluate_ei_config(cfg, random_state=random_state, cache_dir=cache_dir)


def run_ei_benchmark(
    *,
    random_state: int = 0,
    configs: list[Any] | None = None,
    cache_dir: Path | None = None,
    max_workers: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full EI benchmark grid and return detail/summary tables."""
    if configs is None:
        configs = default_ei_simulation_configs()
    workers = resolve_benchmark_workers(len(configs), max_workers=max_workers)
    status(
        "ei_benchmark",
        f"evaluating {len(configs)} EI scenarios with {workers} worker process"
        f"{'' if workers == 1 else 'es'}",
    )
    tasks = [
        (cfg, scenario_random_state(cfg, master_seed=random_state), cache_dir) for cfg in configs
    ]
    if workers == 1:
        frames = [_evaluate_ei_config_worker(task) for task in tasks]
    else:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        try:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
                frames = list(executor.map(_evaluate_ei_config_worker, tasks, chunksize=1))
        except (OSError, PermissionError):
            frames = [_evaluate_ei_config_worker(task) for task in tasks]
    internal_frames = [internal for internal, _ in frames if not internal.empty]
    external_frames = [external for _, external in frames if not external.empty]
    internal_detail = (
        pd.concat(internal_frames, ignore_index=True) if internal_frames else pd.DataFrame()
    )
    external_detail = (
        pd.concat(external_frames, ignore_index=True) if external_frames else pd.DataFrame()
    )
    status("ei_benchmark", "aggregating EI benchmark summaries")
    internal_summary = summarize_ei_benchmark(internal_detail)
    external_summary = summarize_ei_benchmark(external_detail)
    return internal_detail, internal_summary, external_detail, external_summary


__all__ = [
    "EI_ALL_METHODS",
    "EI_EXTERNAL_METHODS",
    "EI_FGLS_METHODS",
    "EI_INTERNAL_METHODS",
    "EI_METHOD_LABELS",
    "evaluate_ei_config",
    "run_ei_benchmark",
    "summarize_ei_benchmark",
]
