"""Benchmark design helpers.

This module keeps the simulation design and the method grid together so the
public workflow entrypoint can stay focused on orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from unibm.bootstrap import (
    circular_block_summary_bootstrap_multi_target,
    draw_circular_block_bootstrap_samples,
)
from unibm.core import (
    block_summary_curve,
    estimate_target_scaling,
    generate_block_sizes,
    select_penultimate_window,
)
from unibm.models import BlockSummaryCurve, PlateauWindow, ScalingFit


# The full 12-method EVI grid is expensive, but the paper-scale benchmark should
# still use enough resampling to stabilize the internal FGLS covariance
# estimates and any optional bootstrap-based sensitivity runs.
FGLS_BOOTSTRAP_REPS = 32
COMMON_BOOTSTRAP_REPS = 32
BENCHMARK_CACHE_VERSION = "2026-04-03-benchmark-cache-v11"
LEGACY_BENCHMARK_CACHE_VERSION = "2026-04-03-benchmark-cache-v11"
DEFAULT_BENCHMARK_WORKERS = 4
BENCHMARK_MASTER_SEED = 20260401


@dataclass(frozen=True)
class MethodSpec:
    """A single benchmark method in the factorial comparison grid."""

    method_id: str
    block_scheme: str
    summary_target: str
    regression: str
    sliding: bool
    bootstrap_reps: int


METHOD_SPECS = (
    MethodSpec("sliding_median_fgls", "sliding", "median", "FGLS", True, FGLS_BOOTSTRAP_REPS),
    MethodSpec("sliding_median_ols", "sliding", "median", "OLS", True, 0),
    MethodSpec("disjoint_median_fgls", "disjoint", "median", "FGLS", False, FGLS_BOOTSTRAP_REPS),
    MethodSpec("disjoint_median_ols", "disjoint", "median", "OLS", False, 0),
    MethodSpec("sliding_mean_fgls", "sliding", "mean", "FGLS", True, FGLS_BOOTSTRAP_REPS),
    MethodSpec("sliding_mean_ols", "sliding", "mean", "OLS", True, 0),
    MethodSpec("disjoint_mean_fgls", "disjoint", "mean", "FGLS", False, FGLS_BOOTSTRAP_REPS),
    MethodSpec("disjoint_mean_ols", "disjoint", "mean", "OLS", False, 0),
    MethodSpec("sliding_mode_fgls", "sliding", "mode", "FGLS", True, FGLS_BOOTSTRAP_REPS),
    MethodSpec("sliding_mode_ols", "sliding", "mode", "OLS", True, 0),
    MethodSpec("disjoint_mode_fgls", "disjoint", "mode", "FGLS", False, FGLS_BOOTSTRAP_REPS),
    MethodSpec("disjoint_mode_ols", "disjoint", "mode", "OLS", False, 0),
)
METHOD_ORDER = [spec.method_id for spec in METHOD_SPECS]
METHOD_LOOKUP = {spec.method_id: spec for spec in METHOD_SPECS}
METHOD_LABELS = {
    spec.method_id: f"{spec.summary_target}-{spec.block_scheme}-{spec.regression}"
    for spec in METHOD_SPECS
}

TARGET_COLORS = {
    "median": "tab:blue",
    "mean": "tab:green",
    "mode": "tab:red",
}
BLOCK_LINESTYLES = {
    "sliding": "-",
    "disjoint": "--",
}
REGRESSION_MARKERS = {
    "FGLS": "s",
    "OLS": "o",
}
FAMILY_LABELS = {
    "frechet_max_ar": "Frechet max-AR",
    "moving_maxima_q2": "Moving Maxima (q=2)",
    "pareto_additive_ar1": "Pareto additive AR1",
}
FAMILY_ORDER = (
    "frechet_max_ar",
    "moving_maxima_q2",
    "pareto_additive_ar1",
)
BENCHMARK_SET_LABELS = {
    "main": "Main range",
    "stress": "Stress test",
}
CORE_METHODS = (
    "disjoint_mean_ols",
    "disjoint_mode_ols",
    "disjoint_median_ols",
    "disjoint_median_fgls",
    "sliding_median_ols",
    "sliding_median_fgls",
)
TARGET_METHODS = (
    "sliding_median_fgls",
    "sliding_mean_fgls",
    "sliding_mode_fgls",
)
METRIC_LABELS = {
    "ape": "absolute percentage error",
    "mape": "mean absolute percentage error",
    "coverage": "interval coverage",
    "interval_score": "Winkler interval score",
}
SIMULATION_BURN_IN = 2000


def sort_by_method_order(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the canonical family and method ordering to a benchmark table."""
    ordered = frame.copy()
    ordered["family"] = pd.Categorical(ordered["family"], categories=FAMILY_ORDER, ordered=True)
    ordered["method"] = pd.Categorical(ordered["method"], categories=METHOD_ORDER, ordered=True)
    sort_columns = ["benchmark_set", "family"]
    if "theta_true" in ordered.columns:
        sort_columns.append("theta_true")
    elif "phi" in ordered.columns:
        sort_columns.append("phi")
    if "xi_true" in ordered.columns:
        sort_columns.append("xi_true")
    sort_columns.append("method")
    return ordered.sort_values(sort_columns).reset_index(drop=True)


def ordered_families(values: Iterable[str]) -> list[str]:
    """Return families in the manuscript order, preserving unknown extras last."""
    seen = {str(value) for value in values}
    ordered = [family for family in FAMILY_ORDER if family in seen]
    extras = sorted(seen.difference(FAMILY_ORDER))
    return ordered + extras


def sort_by_family_order(
    frame: pd.DataFrame,
    *,
    family_col: str = "family",
    sort_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Sort a table by the canonical family order and optional extra keys."""
    ordered = frame.copy()
    ordered[family_col] = pd.Categorical(
        ordered[family_col],
        categories=FAMILY_ORDER,
        ordered=True,
    )
    keys = [family_col]
    if sort_columns is not None:
        keys.extend(sort_columns)
    return ordered.sort_values(keys).reset_index(drop=True)


def method_style(method: str) -> dict[str, Any]:
    """Encode target, block scheme, and regression into a plotting style."""
    spec = METHOD_LOOKUP[method]
    color = TARGET_COLORS[spec.summary_target]
    marker_face = color if spec.regression == "FGLS" else "white"
    return {
        "color": color,
        "linestyle": BLOCK_LINESTYLES[spec.block_scheme],
        "marker": REGRESSION_MARKERS[spec.regression],
        "markerfacecolor": marker_face,
        "markeredgecolor": color,
    }


@dataclass(frozen=True)
class SimulationConfig:
    """A single benchmark scenario on the xi/theta/family grid."""

    benchmark_set: str
    family: str
    xi_true: float
    theta_true: float
    phi: float
    n_obs: int
    reps: int
    quantile: float = 0.5

    @property
    def scenario(self) -> str:
        return (
            f"{self.benchmark_set}_{self.family}_xi{self.xi_true:.2f}"
            f"_theta{self.theta_true:.2f}_n{self.n_obs}_r{self.reps}"
        )

    @property
    def legacy_scenario(self) -> str:
        return f"{self.benchmark_set}_{self.family}_xi{self.xi_true:.2f}_phi{self.phi:.2f}"

    @property
    def xi(self) -> float:
        """Backward-compatible alias for the xi truth parameter."""
        return self.xi_true

    @property
    def theta(self) -> float:
        """Backward-compatible alias for the theta truth parameter."""
        return self.theta_true


def scenario_random_state(
    cfg: SimulationConfig,
    *,
    master_seed: int = BENCHMARK_MASTER_SEED,
) -> int:
    """Return a stable scenario seed shared across EVI and EI workflows.

    The seed is tied to the scenario identity rather than to a workflow-specific
    iteration order. That keeps the cached raw series bank reusable across the
    internal EVI, external EVI, and EI benchmark stacks whenever they target
    the same truth pair `(xi_true, theta_true)`.
    """
    key = f"{int(master_seed)}::{cfg.scenario}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**32)


def map_theta_to_phi_ar(theta: float, xi: float) -> float:
    """Map theta to phi for max-AR and positive additive AR(1) processes."""
    return float((1.0 - theta) ** xi)


def map_theta_to_phi_mm2(theta: float, xi: float) -> float:
    """Map theta to phi for moving maxima process with q=2."""
    if not (1.0 / 3.0 <= theta <= 1.0):
        # We cap it at bounds for floating point limits instead of erroring in sweeping contexts
        theta = max(1.0 / 3.0, min(theta, 1.0))
    if np.isclose(theta, 1.0):
        return 0.0
    u = (-1.0 + np.sqrt(4.0 / theta - 3.0)) / 2.0
    return float(u**xi)


def map_theta_to_phi(family: str, theta: float, xi: float) -> float:
    """Map true extremal index theta to construction parameter phi."""
    if family in ("frechet_max_ar", "pareto_additive_ar1"):
        return map_theta_to_phi_ar(theta, xi)
    if family == "moving_maxima_q2":
        return map_theta_to_phi_mm2(theta, xi)
    raise ValueError(f"Unknown family for phi inversion: {family}")


def theta_from_phi(family: str, phi: float, xi: float) -> float:
    """Map construction parameter phi back to the closed-form theta truth."""
    if family in ("frechet_max_ar", "pareto_additive_ar1"):
        return float(1.0 - phi ** (1.0 / xi))
    if family == "moving_maxima_q2":
        x = phi ** (1.0 / xi)
        return float(1.0 / (1.0 + x + x**2))
    raise ValueError(f"Unknown family for theta mapping: {family}")


def _sample_pareto_innovations(
    xi: float,
    n_obs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw positive Pareto-type innovations with tail index `xi`."""
    tiny = np.finfo(float).tiny
    uniforms = np.clip(rng.random(n_obs), tiny, 1 - tiny)
    return np.power(1 - uniforms, -xi)


def _sample_frechet_innovations(
    xi: float,
    n_obs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw positive Fréchet innovations with extreme-value index `xi`."""
    tiny = np.finfo(float).tiny
    uniforms = np.clip(rng.random(n_obs), tiny, 1 - tiny)
    return np.power(-np.log(uniforms), -xi)


def _simulate_positive_additive_ar1(
    innovations: np.ndarray,
    *,
    phi: float,
    burn_in: int = SIMULATION_BURN_IN,
) -> np.ndarray:
    """Simulate a positive additive AR(1) process from heavy-tailed innovations."""
    if not 0 <= phi < 1:
        raise ValueError("phi must satisfy 0 <= phi < 1 for the positive additive AR(1) design.")
    innovations = np.asarray(innovations, dtype=float)
    total = innovations.size
    series = np.empty(total, dtype=float)
    series[0] = innovations[0]
    for idx in range(1, total):
        series[idx] = phi * series[idx - 1] + innovations[idx]
    return series[burn_in:]


def _simulate_frechet_max_ar(
    innovations: np.ndarray,
    *,
    phi: float,
    burn_in: int = SIMULATION_BURN_IN,
) -> np.ndarray:
    """Simulate a max-autoregressive extreme-value process."""
    if not 0 <= phi < 1:
        raise ValueError("phi must satisfy 0 <= phi < 1 for the max-AR design.")
    innovations = np.asarray(innovations, dtype=float)
    total = innovations.size
    series = np.empty(total, dtype=float)
    series[0] = innovations[0]
    for idx in range(1, total):
        series[idx] = max(phi * series[idx - 1], innovations[idx])
    return series[burn_in:]


def simulate_moving_maxima_series(
    xi: float,
    phi: float,
    n_obs: int,
    rng: np.random.Generator,
    *,
    q: int = 2,
    burn_in: int = SIMULATION_BURN_IN,
) -> np.ndarray:
    """Simulate a moving maxima process of order q."""
    weights = np.array([phi**j for j in range(q + 1)], dtype=float)
    innovations = _sample_frechet_innovations(xi=xi, n_obs=n_obs + burn_in + q, rng=rng)
    windows = np.lib.stride_tricks.sliding_window_view(innovations, q + 1)
    reversed_weights = weights[::-1]
    series = np.max(windows * reversed_weights, axis=-1)
    return series[burn_in : burn_in + n_obs]


def simulate_moving_maxima_q2_series(
    xi: float,
    phi: float,
    n_obs: int,
    rng: np.random.Generator,
    *,
    burn_in: int = SIMULATION_BURN_IN,
) -> np.ndarray:
    """Simulate a moving maxima process of order 2."""
    return simulate_moving_maxima_series(
        xi=xi, phi=phi, n_obs=n_obs, rng=rng, q=2, burn_in=burn_in
    )


def simulate_pareto_additive_ar1_series(
    xi: float,
    phi: float,
    n_obs: int,
    rng: np.random.Generator,
    *,
    burn_in: int = SIMULATION_BURN_IN,
) -> np.ndarray:
    """Simulate a heavy-tailed additive AR(1) series with Pareto innovations."""
    innovations = _sample_pareto_innovations(xi=xi, n_obs=n_obs + burn_in, rng=rng)
    return _simulate_positive_additive_ar1(
        innovations,
        phi=phi,
        burn_in=burn_in,
    )


def simulate_frechet_max_ar_series(
    xi: float,
    phi: float,
    n_obs: int,
    rng: np.random.Generator,
    *,
    burn_in: int = SIMULATION_BURN_IN,
) -> np.ndarray:
    """Simulate a Fréchet max-AR process with persistence parameter `phi`."""
    innovations = _sample_frechet_innovations(xi=xi, n_obs=n_obs + burn_in, rng=rng)
    return _simulate_frechet_max_ar(
        innovations,
        phi=phi,
        burn_in=burn_in,
    )


def simulate_series(cfg: SimulationConfig, rng: np.random.Generator) -> np.ndarray:
    """Generate one synthetic series for a configured family and dependence level."""
    if cfg.family == "pareto_additive_ar1":
        return simulate_pareto_additive_ar1_series(cfg.xi_true, cfg.phi, cfg.n_obs, rng)
    if cfg.family == "frechet_max_ar":
        return simulate_frechet_max_ar_series(cfg.xi_true, cfg.phi, cfg.n_obs, rng)
    if cfg.family == "moving_maxima_q2":
        return simulate_moving_maxima_q2_series(cfg.xi_true, cfg.phi, cfg.n_obs, rng)
    raise ValueError(f"Unknown family: {cfg.family}")


def default_evi_simulation_configs(
    *,
    theta_values: Iterable[float] = (1.00, 0.70, 0.50, 0.35),
    xi_values_main: Iterable[float] = (0.10, 0.20, 0.50, 1.0, 2.0, 3.0, 5.0, 10.0),
    xi_values_stress: Iterable[float] = (),
    n_obs: int = 365,
    reps: int | None = None,
    reps_main: int = 32,
    reps_stress: int = 8,
    quantile: float = 0.5,
) -> list[SimulationConfig]:
    """Build the default EVI grid (fixed theta, sweep xi)."""
    if reps is not None:
        reps_main = int(reps)
        reps_stress = int(reps)
    configs: list[SimulationConfig] = []
    settings = [
        ("main", xi_values_main, reps_main),
        ("stress", xi_values_stress, reps_stress),
    ]
    for benchmark_set, xi_values, reps in settings:
        for family in FAMILY_ORDER:
            for theta in theta_values:
                for xi in xi_values:
                    phi = map_theta_to_phi(family, float(theta), float(xi))
                    configs.append(
                        SimulationConfig(
                            benchmark_set=benchmark_set,
                            family=family,
                            xi_true=float(xi),
                            theta_true=float(theta),
                            phi=float(phi),
                            n_obs=n_obs,
                            reps=reps,
                            quantile=quantile,
                        )
                    )
    return configs


def default_ei_simulation_configs(
    *,
    xi_values: Iterable[float] = (0.50, 1.0, 5.0, 10.0),
    theta_values_main: Iterable[float] = (1.00, 0.85, 0.70, 0.55, 0.45, 0.35),
    theta_values_stress: Iterable[float] = (),
    n_obs: int = 365,
    reps: int | None = None,
    reps_main: int = 32,
    reps_stress: int = 8,
    quantile: float = 0.5,
) -> list[SimulationConfig]:
    """Build the default EI grid (fixed xi, sweep theta)."""
    if reps is not None:
        reps_main = int(reps)
        reps_stress = int(reps)
    configs: list[SimulationConfig] = []
    settings = [
        ("main", theta_values_main, reps_main),
        ("stress", theta_values_stress, reps_stress),
    ]
    for benchmark_set, theta_values, reps in settings:
        for family in FAMILY_ORDER:
            for xi in xi_values:
                for theta in theta_values:
                    phi = map_theta_to_phi(family, float(theta), float(xi))
                    configs.append(
                        SimulationConfig(
                            benchmark_set=benchmark_set,
                            family=family,
                            xi_true=float(xi),
                            theta_true=float(theta),
                            phi=float(phi),
                            n_obs=n_obs,
                            reps=reps,
                            quantile=quantile,
                        )
                    )
    return configs


def default_simulation_configs(**kwargs: Any) -> list[SimulationConfig]:
    """Backward-compatible alias for the default EVI benchmark grid."""
    if "xi_values" in kwargs and "xi_values_main" not in kwargs:
        kwargs["xi_values_main"] = kwargs.pop("xi_values")
    unsupported_legacy = sorted(
        set(kwargs).intersection(
            {
                "families",
                "phi_values",
                "phi_values_main",
                "phi_values_stress",
                "tau_values",
                "tau_values_main",
                "tau_values_stress",
            }
        )
    )
    if unsupported_legacy:
        joined = ", ".join(unsupported_legacy)
        raise TypeError(
            "default_simulation_configs no longer accepts "
            f"{joined}. The benchmark now uses (xi_true, theta_true) as the "
            "truth grid with a fixed family set. Use "
            "default_evi_simulation_configs(theta_values=..., xi_values_main=...) "
            "or default_ei_simulation_configs(xi_values=..., theta_values_main=...)."
        )
    return default_evi_simulation_configs(**kwargs)


def _series_cache_file(
    cache_dir: Path,
    cfg: SimulationConfig,
    *,
    random_state: int,
    legacy: bool = False,
) -> Path:
    """Return the on-disk cache path for one scenario's simulated series bank."""
    scenario_key = cfg.legacy_scenario if legacy else cfg.scenario
    version = LEGACY_BENCHMARK_CACHE_VERSION if legacy else BENCHMARK_CACHE_VERSION
    return cache_dir / "series" / f"{version}__{scenario_key}__seed{random_state}.npz"


def _raw_bootstrap_cache_file(
    cache_dir: Path,
    *,
    cache_key: str,
    bootstrap_reps: int,
) -> Path:
    """Return the raw-series bootstrap bank cache path for external estimators.

    The current benchmark uses raw-series bootstrap samples only for optional
    external-estimator sensitivity runs. Internal UniBM methods report their
    native Wald/FGLS intervals and therefore do not consume this cache.
    """
    return (
        cache_dir
        / "raw_bootstrap"
        / f"{BENCHMARK_CACHE_VERSION}__{cache_key}__reps{bootstrap_reps}.npz"
    )


def load_or_simulate_series_bank(
    cfg: SimulationConfig,
    *,
    random_state: int,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Load or generate the synthetic series bank for one scenario.

    Internal and external benchmarks both consume the same cached bank so the
    comparison is based on identical simulated time series across reruns.
    """
    if cache_dir is not None:
        cache_file = _series_cache_file(cache_dir, cfg, random_state=random_state)
        legacy_cache_file = _series_cache_file(
            cache_dir,
            cfg,
            random_state=random_state,
            legacy=True,
        )

        for p in (cache_file, legacy_cache_file):
            if p.exists():
                with np.load(p) as data:
                    series_bank = np.asarray(data["series"], dtype=float)
                if series_bank.shape == (cfg.reps, cfg.n_obs):
                    if p == legacy_cache_file and not cache_file.exists():
                        cache_file.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(cache_file, series=series_bank)
                    return series_bank

    rng = np.random.default_rng(random_state)
    series_bank = np.empty((cfg.reps, cfg.n_obs), dtype=float)
    for rep in range(cfg.reps):
        series_bank[rep] = simulate_series(cfg, rng)
    if cache_dir is not None:
        cache_file = _series_cache_file(cache_dir, cfg, random_state=random_state)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, series=series_bank)
    return series_bank


def load_or_draw_raw_bootstrap_samples(
    vec: np.ndarray,
    *,
    cache_dir: Path | None,
    cache_key: str,
    bootstrap_reps: int,
    random_state: int,
) -> np.ndarray:
    """Load or draw one raw-series circular bootstrap sample bank.

    This cache is shared anywhere the benchmark needs dependence-preserving raw
    time-series resamples. In practice that means the optional external-xi
    percentile sensitivity path and the EI pooled-BM FGLS covariance builder.
    """
    values = np.asarray(vec, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if cache_dir is not None:
        cache_file = _raw_bootstrap_cache_file(
            cache_dir,
            cache_key=cache_key,
            bootstrap_reps=bootstrap_reps,
        )
        if cache_file.exists():
            with np.load(cache_file) as data:
                samples = np.asarray(data["samples"], dtype=float)
            if samples.shape == (bootstrap_reps, values.size):
                return samples
    bootstrap_bank = draw_circular_block_bootstrap_samples(
        values,
        reps=bootstrap_reps,
        random_state=random_state,
    )
    samples = bootstrap_bank.samples
    if cache_dir is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, samples=samples, block_size=bootstrap_bank.block_size)
    return samples


def resolve_benchmark_workers(
    n_jobs: int,
    *,
    max_workers: int | None = None,
) -> int:
    """Choose a conservative process count for benchmark scenario parallelism.

    The default caps worker count at a small number because each worker may
    trigger BLAS-backed NumPy operations. Overcommitting cores can easily make
    the benchmark slower rather than faster on laptops.
    """
    if n_jobs <= 1:
        return 1
    if max_workers is None:
        env_value = os.environ.get("UNIBM_BENCHMARK_WORKERS")
        if env_value is not None:
            try:
                max_workers = int(env_value)
            except ValueError:
                max_workers = None
    cpu_count = os.cpu_count() or 1
    if max_workers is None:
        max_workers = DEFAULT_BENCHMARK_WORKERS
    return max(1, min(int(max_workers), cpu_count, n_jobs))


def _internal_bootstrap_cache_file(
    cache_dir: Path,
    *,
    cache_key: str,
    sliding: bool,
    quantile: float,
    reps: int,
) -> Path:
    """Return the on-disk cache path for one scheme's shared FGLS bootstrap."""
    scheme = "sliding" if sliding else "disjoint"
    return (
        cache_dir
        / "internal_bootstrap"
        / f"{BENCHMARK_CACHE_VERSION}__{cache_key}__{scheme}__q{quantile:.4f}__reps{reps}.npz"
    )


def _save_bootstrap_results_bundle(
    cache_file: Path,
    bootstrap_results: dict[str, dict[str, Any]],
) -> None:
    """Persist one scheme's multi-target bootstrap results to disk."""
    arrays: dict[str, Any] = {"targets": np.asarray(list(bootstrap_results), dtype="U32")}
    for target, result in bootstrap_results.items():
        arrays[f"{target}__block_sizes"] = np.asarray(result["block_sizes"], dtype=int)
        arrays[f"{target}__samples"] = np.asarray(result["samples"], dtype=float)
        covariance = result.get("covariance")
        arrays[f"{target}__has_covariance"] = np.asarray(covariance is not None, dtype=bool)
        arrays[f"{target}__covariance"] = (
            np.asarray(covariance, dtype=float)
            if covariance is not None
            else np.empty((0, 0), dtype=float)
        )
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, **arrays)


def _load_bootstrap_results_bundle(cache_file: Path) -> dict[str, dict[str, Any]]:
    """Reload one scheme's cached multi-target bootstrap results from disk."""
    with np.load(cache_file) as data:
        targets = [str(target) for target in np.asarray(data["targets"])]
        results: dict[str, dict[str, Any]] = {}
        for target in targets:
            has_covariance = bool(np.asarray(data.get(f"{target}__has_covariance", True)).item())
            covariance = np.asarray(data[f"{target}__covariance"], dtype=float)
            results[target] = {
                "block_sizes": np.asarray(data[f"{target}__block_sizes"], dtype=int),
                "samples": np.asarray(data[f"{target}__samples"], dtype=float),
                "covariance": covariance if has_covariance and covariance.size else None,
                "target": target,
            }
    return results


def _internal_target_name(summary_target: str) -> str:
    """Map benchmark labels onto the core estimator target names."""
    return "quantile" if summary_target == "median" else summary_target


def _scheme_method_ids(*, sliding: bool, summary_target: str) -> tuple[str, str]:
    """Return the OLS/FGLS method ids for one scheme-target pair."""
    scheme_name = "sliding" if sliding else "disjoint"
    return f"{scheme_name}_{summary_target}_ols", f"{scheme_name}_{summary_target}_fgls"


def _shared_curve_and_plateau(
    vec: np.ndarray,
    *,
    block_sizes: np.ndarray,
    quantile: float,
    sliding: bool,
    summary_target: str,
    existing_fit: ScalingFit | None = None,
) -> tuple[BlockSummaryCurve, PlateauWindow]:
    """Build or reuse the block-summary curve and plateau for one target.

    OLS and FGLS only differ in the regression weighting. The block-summary
    curve itself and the selected plateau should therefore be shared whenever
    possible so the benchmark isolates the covariance effect cleanly.
    """
    if existing_fit is not None:
        return existing_fit.curve, existing_fit.plateau
    internal_target = _internal_target_name(summary_target)
    curve = block_summary_curve(
        vec,
        block_sizes,
        sliding=sliding,
        quantile=quantile,
        target=internal_target,
    )
    plateau = select_penultimate_window(
        curve.log_block_sizes,
        curve.log_values,
        min_points=5,
        trim_fraction=0.15,
    )
    return curve, plateau


def _scheme_bootstrap_results(
    vec: np.ndarray,
    *,
    block_sizes: np.ndarray,
    quantile: float,
    sliding: bool,
    specs: list[MethodSpec],
    random_state: int,
    cache_dir: Path | None = None,
    cache_key: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Build one shared bootstrap backbone and evaluate every FGLS target on it."""
    fgls_targets = tuple(
        dict.fromkeys(
            _internal_target_name(spec.summary_target)
            for spec in specs
            if spec.regression == "FGLS"
        )
    )
    if not fgls_targets:
        return {}
    if cache_dir is not None and cache_key is not None:
        cache_file = _internal_bootstrap_cache_file(
            cache_dir,
            cache_key=cache_key,
            sliding=sliding,
            quantile=quantile,
            reps=FGLS_BOOTSTRAP_REPS,
        )
        if cache_file.exists():
            cached = _load_bootstrap_results_bundle(cache_file)
            if set(cached) == set(fgls_targets):
                return cached
    bootstrap_results = circular_block_summary_bootstrap_multi_target(
        vec,
        block_sizes,
        targets=fgls_targets,
        quantile=quantile,
        sliding=sliding,
        reps=FGLS_BOOTSTRAP_REPS,
        random_state=random_state,
    )
    if cache_dir is not None and cache_key is not None:
        _save_bootstrap_results_bundle(cache_file, bootstrap_results)
    return bootstrap_results


def fit_methods_for_series(
    vec: np.ndarray,
    *,
    quantile: float,
    random_state: int,
    reuse_fits: dict[str, ScalingFit] | None = None,
    cache_dir: Path | None = None,
    cache_key: str | None = None,
    fgls_bootstrap_overrides: dict[str, dict[str, Any] | None] | None = None,
    allow_scheme_bootstrap: bool = True,
) -> dict[str, ScalingFit]:
    """Fit every benchmark method to one synthetic series.

    The benchmark compares target choice, block extraction scheme, and whether
    covariance-aware regression is used. FGLS methods share one bootstrap
    backbone per block scheme (`sliding` and `disjoint`) so median/mean/mode
    comparisons reuse the same dependence-preserving resamples instead of
    paying the bootstrap cost three times.

    When `fgls_bootstrap_overrides` is supplied, those cached covariance
    summaries are reused instead of drawing fresh scheme-level bootstraps. This
    is the key trick that keeps optional raw-bootstrap sensitivity runs
    feasible: each raw bootstrap resample re-fits the FGLS point estimator
    using the original sample's covariance backbone rather than nesting
    another bootstrap inside every resample.
    """
    reuse_fits = {} if reuse_fits is None else dict(reuse_fits)
    fgls_bootstrap_overrides = (
        {} if fgls_bootstrap_overrides is None else dict(fgls_bootstrap_overrides)
    )
    values = np.asarray(vec, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    block_sizes = generate_block_sizes(n_obs=values.size)
    fits: dict[str, ScalingFit] = dict(reuse_fits)
    grouped_specs = {
        True: [spec for spec in METHOD_SPECS if spec.sliding],
        False: [spec for spec in METHOD_SPECS if not spec.sliding],
    }
    for sliding, specs in grouped_specs.items():
        scheme_bootstrap_results: dict[str, dict[str, Any]] = {}
        missing_targets = [
            _internal_target_name(spec.summary_target)
            for spec in specs
            if spec.regression == "FGLS"
            and f"{'sliding' if sliding else 'disjoint'}_{spec.summary_target}_fgls"
            not in fgls_bootstrap_overrides
        ]
        if allow_scheme_bootstrap and missing_targets:
            scheme_bootstrap_results = _scheme_bootstrap_results(
                values,
                block_sizes=block_sizes,
                quantile=quantile,
                sliding=sliding,
                specs=specs,
                random_state=random_state,
                cache_dir=cache_dir,
                cache_key=cache_key,
            )
        summary_targets = tuple(dict.fromkeys(spec.summary_target for spec in specs))
        for summary_target in summary_targets:
            ols_id, fgls_id = _scheme_method_ids(
                sliding=sliding,
                summary_target=summary_target,
            )
            existing_fit = fits.get(fgls_id) or fits.get(ols_id)
            shared_curve, shared_plateau = _shared_curve_and_plateau(
                values,
                block_sizes=block_sizes,
                quantile=quantile,
                sliding=sliding,
                summary_target=summary_target,
                existing_fit=existing_fit,
            )
            internal_target = _internal_target_name(summary_target)
            if ols_id not in fits:
                fits[ols_id] = estimate_target_scaling(
                    values,
                    target=internal_target,
                    quantile=quantile,
                    sliding=sliding,
                    bootstrap_reps=0,
                    random_state=random_state,
                    curve=shared_curve,
                    plateau=shared_plateau,
                )
            if fgls_id not in fits:
                bootstrap_result = fgls_bootstrap_overrides.get(fgls_id)
                if bootstrap_result is None:
                    bootstrap_result = scheme_bootstrap_results.get(internal_target)
                effective_bootstrap_reps = (
                    FGLS_BOOTSTRAP_REPS if bootstrap_result is not None else 0
                )
                fits[fgls_id] = estimate_target_scaling(
                    values,
                    target=internal_target,
                    quantile=quantile,
                    sliding=sliding,
                    bootstrap_reps=effective_bootstrap_reps,
                    random_state=random_state,
                    curve=shared_curve,
                    plateau=shared_plateau,
                    bootstrap_result=bootstrap_result,
                )
    return fits
