"""Benchmark design helpers.

This module keeps the simulation design and the method grid together so the
public workflow entrypoint can stay focused on orchestration.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable
from zipfile import BadZipFile

import numpy as np
import pandas as pd
from scipy.optimize import brentq

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
FGLS_BOOTSTRAP_REPS = 120
COMMON_BOOTSTRAP_REPS = 32
BENCHMARK_CACHE_VERSION = "2026-04-06-benchmark-cache-v14-projected-grid-bundled-cache"
LEGACY_BENCHMARK_CACHE_VERSIONS = (
    "2026-04-06-benchmark-cache-v13-universal-q",
    "2026-04-03-benchmark-cache-v11",
)
LEGACY_SCENARIO_CACHE_VERSIONS = ("2026-04-03-benchmark-cache-v11",)
DEFAULT_BENCHMARK_WORKERS = 4
BENCHMARK_MASTER_SEED = 20260401
UNIVERSAL_BENCHMARK_SET = "universal"
UNIVERSAL_XI_VALUES = (0.01, 0.03, 0.10, 0.30, 1.0, 3.0, 10.0)
UNIVERSAL_THETA_VALUES = (0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 1.0)
EVI_DEFAULT_THETA_VALUES = (0.01, 0.10, 0.50, 1.0)
EI_DEFAULT_XI_VALUES = (0.01, 0.50, 1.0, 5.0)
UNIVERSAL_FAMILIES = (
    "frechet_max_ar",
    "moving_maxima_q9",
    "pareto_additive_ar1",
)
EVI_DEFAULT_FAMILIES = (
    "frechet_max_ar",
    "moving_maxima_q99",
    "pareto_additive_ar1",
)
EI_DEFAULT_FAMILIES = UNIVERSAL_FAMILIES
MOVING_MAXIMA_FAMILY_PATTERN = re.compile(r"^moving_maxima_q(?P<q>[1-9]\d*)$")


def _try_load_npz(cache_file: Path) -> Any | None:
    """Open an ``.npz`` cache file, dropping it if it is corrupted."""
    try:
        return np.load(cache_file)
    except (BadZipFile, EOFError, OSError, ValueError):
        try:
            cache_file.unlink()
        except FileNotFoundError:
            pass
        return None


def _atomic_savez(cache_file: Path, *, compressed: bool = True, **arrays: Any) -> None:
    """Atomically persist an ``.npz`` cache bundle."""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=cache_file.parent,
        prefix=f"{cache_file.stem}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with tmp_path.open("wb") as handle:
            if compressed:
                np.savez_compressed(handle, **arrays)
            else:
                np.savez(handle, **arrays)
        os.replace(tmp_path, cache_file)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass


def _atomic_savez_compressed(cache_file: Path, **arrays: Any) -> None:
    """Atomically persist a compressed ``.npz`` cache bundle."""
    _atomic_savez(cache_file, compressed=True, **arrays)


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
    "moving_maxima_q9": "Moving Maxima (q=9)",
    "moving_maxima_q99": "Moving Maxima (q=99)",
    "pareto_additive_ar1": "Pareto additive AR1",
}
FAMILY_ORDER = (
    "frechet_max_ar",
    "moving_maxima_q9",
    "moving_maxima_q99",
    "pareto_additive_ar1",
)
BENCHMARK_SET_LABELS = {
    UNIVERSAL_BENCHMARK_SET: "Universal grid",
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


def parse_moving_maxima_q(family: str) -> int | None:
    """Return the moving-maxima order encoded in a family id, if present."""
    match = MOVING_MAXIMA_FAMILY_PATTERN.fullmatch(str(family))
    if match is None:
        return None
    return int(match.group("q"))


def family_label(family: str) -> str:
    """Render a family id into a stable manuscript-friendly label."""
    q = parse_moving_maxima_q(family)
    if q is not None:
        return f"Moving Maxima (q={q})"
    return FAMILY_LABELS.get(family, family)


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


def _moving_maxima_denominator(x: float, q: int) -> float:
    """Return 1 + x + ... + x^q on the unit interval."""
    if np.isclose(x, 1.0):
        return float(q + 1)
    exponents = np.arange(q + 1, dtype=float)
    return float(np.sum(np.power(float(x), exponents)))


def map_theta_to_phi_moving_maxima(theta: float, xi: float, q: int) -> float:
    """Map theta to phi for a moving-maxima process of order q."""
    theta_min = 1.0 / float(q + 1)
    theta = float(np.clip(theta, theta_min, 1.0))
    if np.isclose(theta, 1.0):
        return 0.0
    if np.isclose(theta, theta_min):
        return 1.0

    def root_func(x: float) -> float:
        return (1.0 / _moving_maxima_denominator(x, q)) - theta

    x_hat = float(brentq(root_func, 0.0, 1.0, xtol=1e-12, rtol=1e-10))
    return float(x_hat**xi)


def map_theta_to_phi(family: str, theta: float, xi: float) -> float:
    """Map true extremal index theta to construction parameter phi."""
    if family in ("frechet_max_ar", "pareto_additive_ar1"):
        return map_theta_to_phi_ar(theta, xi)
    q = parse_moving_maxima_q(family)
    if q is not None:
        return map_theta_to_phi_moving_maxima(theta, xi, q)
    raise ValueError(f"Unknown family for phi inversion: {family}")


def theta_from_phi(family: str, phi: float, xi: float) -> float:
    """Map construction parameter phi back to the closed-form theta truth."""
    if family in ("frechet_max_ar", "pareto_additive_ar1"):
        return float(1.0 - phi ** (1.0 / xi))
    q = parse_moving_maxima_q(family)
    if q is not None:
        x = float(phi ** (1.0 / xi))
        return float(1.0 / _moving_maxima_denominator(x, q))
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
    q = parse_moving_maxima_q(cfg.family)
    if q is not None:
        return simulate_moving_maxima_series(cfg.xi_true, cfg.phi, cfg.n_obs, rng, q=q)
    raise ValueError(f"Unknown family: {cfg.family}")


def default_evi_simulation_configs(
    *,
    xi_values: Iterable[float] = UNIVERSAL_XI_VALUES,
    theta_values: Iterable[float] = EVI_DEFAULT_THETA_VALUES,
    families: Iterable[str] = EVI_DEFAULT_FAMILIES,
    n_obs: int = 365,
    reps: int = 32,
    quantile: float = 0.5,
    benchmark_set: str = UNIVERSAL_BENCHMARK_SET,
) -> list[SimulationConfig]:
    """Build the default EVI benchmark grid.

    The EVI workflow keeps the full xi range but only a representative slice of
    theta values so manuscript plots stay readable and routine reruns remain
    tractable.
    """
    configs: list[SimulationConfig] = []
    family_values = tuple(str(family) for family in families)
    for family in family_values:
        for theta in theta_values:
            for xi in xi_values:
                phi = map_theta_to_phi(family, float(theta), float(xi))
                configs.append(
                    SimulationConfig(
                        benchmark_set=str(benchmark_set),
                        family=family,
                        xi_true=float(xi),
                        theta_true=float(theta),
                        phi=float(phi),
                        n_obs=n_obs,
                        reps=int(reps),
                        quantile=quantile,
                    )
                )
    return configs


def default_ei_simulation_configs(
    *,
    xi_values: Iterable[float] = EI_DEFAULT_XI_VALUES,
    theta_values: Iterable[float] = UNIVERSAL_THETA_VALUES,
    families: Iterable[str] = EI_DEFAULT_FAMILIES,
    n_obs: int = 365,
    reps: int = 32,
    quantile: float = 0.5,
    benchmark_set: str = UNIVERSAL_BENCHMARK_SET,
) -> list[SimulationConfig]:
    """Build the default EI benchmark grid.

    The EI workflow keeps the full theta range but only a representative slice
    of xi values so the benchmark still covers weak to very heavy tails without
    exploding the scenario count.
    """
    configs: list[SimulationConfig] = []
    family_values = tuple(str(family) for family in families)
    for family in family_values:
        for xi in xi_values:
            for theta in theta_values:
                phi = map_theta_to_phi(family, float(theta), float(xi))
                configs.append(
                    SimulationConfig(
                        benchmark_set=str(benchmark_set),
                        family=family,
                        xi_true=float(xi),
                        theta_true=float(theta),
                        phi=float(phi),
                        n_obs=n_obs,
                        reps=int(reps),
                        quantile=quantile,
                    )
                )
    return configs


def default_simulation_configs(**kwargs: Any) -> list[SimulationConfig]:
    """Backward-compatible alias for the default EVI benchmark grid."""
    legacy_map = {
        "xi_values_main": "xi_values",
        "theta_values_main": "theta_values",
    }
    for old_key, new_key in legacy_map.items():
        if old_key in kwargs and new_key not in kwargs:
            kwargs[new_key] = kwargs.pop(old_key)
    unsupported_legacy = sorted(
        set(kwargs).intersection(
            {
                "xi_values_stress",
                "theta_values_stress",
                "phi_values",
                "phi_values_main",
                "phi_values_stress",
                "tau_values",
                "tau_values_main",
                "tau_values_stress",
                "reps_main",
                "reps_stress",
            }
        )
    )
    if unsupported_legacy:
        joined = ", ".join(unsupported_legacy)
        raise TypeError(
            "default_simulation_configs no longer accepts "
            f"{joined}. The benchmark now uses one universal (xi_true, theta_true) grid. "
            "Use default_evi_simulation_configs(xi_values=..., theta_values=...) or "
            "default_ei_simulation_configs(xi_values=..., theta_values=...)."
        )
    return default_evi_simulation_configs(**kwargs)


def _series_cache_file(
    cache_dir: Path,
    cfg: SimulationConfig,
    *,
    random_state: int,
    version: str | None = None,
) -> Path:
    """Return the on-disk cache path for one scenario's simulated series bank."""
    scenario_key = cfg.scenario
    resolved_version = BENCHMARK_CACHE_VERSION if version is None else str(version)
    if resolved_version in LEGACY_SCENARIO_CACHE_VERSIONS:
        scenario_key = cfg.legacy_scenario
    return cache_dir / "series" / f"{resolved_version}__{scenario_key}__seed{random_state}.npz"


def _raw_bootstrap_cache_file(
    cache_dir: Path,
    *,
    cache_key: str,
    bootstrap_reps: int,
    version: str | None = None,
) -> Path:
    """Return the raw-series bootstrap bank cache path for external estimators.

    The current benchmark uses raw-series bootstrap samples only for optional
    external-estimator sensitivity runs. Internal UniBM methods report their
    native Wald/FGLS intervals and therefore do not consume this cache.
    """
    resolved_version = BENCHMARK_CACHE_VERSION if version is None else str(version)
    return (
        cache_dir / "raw_bootstrap" / f"{resolved_version}__{cache_key}__reps{bootstrap_reps}.npz"
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
        legacy_cache_files = [
            _series_cache_file(cache_dir, cfg, random_state=random_state, version=version)
            for version in LEGACY_BENCHMARK_CACHE_VERSIONS
        ]
        for p in (cache_file, *legacy_cache_files):
            if p.exists():
                loaded = _try_load_npz(p)
                if loaded is None:
                    continue
                with loaded as data:
                    series_bank = np.asarray(data["series"], dtype=float)
                if series_bank.shape == (cfg.reps, cfg.n_obs):
                    if p != cache_file and not cache_file.exists():
                        _atomic_savez_compressed(cache_file, series=series_bank)
                    return series_bank

    rng = np.random.default_rng(random_state)
    series_bank = np.empty((cfg.reps, cfg.n_obs), dtype=float)
    for rep in range(cfg.reps):
        series_bank[rep] = simulate_series(cfg, rng)
    if cache_dir is not None:
        cache_file = _series_cache_file(cache_dir, cfg, random_state=random_state)
        _atomic_savez_compressed(cache_file, series=series_bank)
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
        legacy_cache_files = [
            _raw_bootstrap_cache_file(
                cache_dir,
                cache_key=cache_key,
                bootstrap_reps=bootstrap_reps,
                version=version,
            )
            for version in LEGACY_BENCHMARK_CACHE_VERSIONS
        ]
        for p in (cache_file, *legacy_cache_files):
            if p.exists():
                loaded = _try_load_npz(p)
                if loaded is None:
                    continue
                with loaded as data:
                    samples = np.asarray(data["samples"], dtype=float)
                    block_size = np.asarray(data.get("block_size", np.nan))
                if samples.shape == (bootstrap_reps, values.size):
                    if p != cache_file and not cache_file.exists():
                        _atomic_savez_compressed(
                            cache_file,
                            samples=samples,
                            block_size=block_size,
                        )
                    return samples
    bootstrap_bank = draw_circular_block_bootstrap_samples(
        values,
        reps=bootstrap_reps,
        random_state=random_state,
    )
    samples = bootstrap_bank.samples
    if cache_dir is not None:
        _atomic_savez_compressed(
            cache_file,
            samples=samples,
            block_size=bootstrap_bank.block_size,
        )
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
    quantile: float,
    reps: int,
) -> Path:
    """Return the on-disk cache path for one series-wide EVI bootstrap bundle."""
    return (
        cache_dir
        / "internal_bootstrap"
        / f"{BENCHMARK_CACHE_VERSION}__{cache_key}__q{quantile:.4f}__reps{reps}.npz"
    )


def _save_bootstrap_results_bundle(
    cache_file: Path,
    bundle: dict[str, dict[str, dict[str, Any]]],
) -> None:
    """Persist one series-wide EVI bootstrap bundle to disk."""
    arrays: dict[str, Any] = {"schemes": np.asarray(list(bundle), dtype="U16")}
    for scheme, bootstrap_results in bundle.items():
        arrays[f"{scheme}__targets"] = np.asarray(list(bootstrap_results), dtype="U32")
        for target, result in bootstrap_results.items():
            prefix = f"{scheme}__{target}"
            arrays[f"{prefix}__block_sizes"] = np.asarray(result["block_sizes"], dtype=int)
            arrays[f"{prefix}__samples"] = np.asarray(result["samples"], dtype=float)
            covariance = result.get("covariance")
            arrays[f"{prefix}__has_covariance"] = np.asarray(covariance is not None, dtype=bool)
            arrays[f"{prefix}__covariance"] = (
                np.asarray(covariance, dtype=float)
                if covariance is not None
                else np.empty((0, 0), dtype=float)
            )
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez(cache_file, compressed=False, **arrays)


def _load_bootstrap_results_bundle(cache_file: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Reload one series-wide cached EVI bootstrap bundle from disk."""
    loaded = _try_load_npz(cache_file)
    if loaded is None:
        raise FileNotFoundError(cache_file)
    with loaded as data:
        schemes = [str(scheme) for scheme in np.asarray(data["schemes"])]
        bundle: dict[str, dict[str, dict[str, Any]]] = {}
        for scheme in schemes:
            targets = [str(target) for target in np.asarray(data[f"{scheme}__targets"])]
            scheme_results: dict[str, dict[str, Any]] = {}
            for target in targets:
                prefix = f"{scheme}__{target}"
                has_covariance = bool(
                    np.asarray(data.get(f"{prefix}__has_covariance", True)).item()
                )
                covariance = np.asarray(data[f"{prefix}__covariance"], dtype=float)
                scheme_results[target] = {
                    "block_sizes": np.asarray(data[f"{prefix}__block_sizes"], dtype=int),
                    "samples": np.asarray(data[f"{prefix}__samples"], dtype=float),
                    "covariance": covariance if has_covariance and covariance.size else None,
                    "target": target,
                }
            bundle[scheme] = scheme_results
    return bundle


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
            quantile=quantile,
            reps=FGLS_BOOTSTRAP_REPS,
        )
        scheme_name = "sliding" if sliding else "disjoint"
        if cache_file.exists():
            try:
                cached_bundle = _load_bootstrap_results_bundle(cache_file)
            except FileNotFoundError:
                cached_bundle = {}
            cached = cached_bundle.get(scheme_name, {})
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
        existing_bundle = {}
        if cache_file.exists():
            try:
                existing_bundle = _load_bootstrap_results_bundle(cache_file)
            except FileNotFoundError:
                existing_bundle = {}
        existing_bundle["sliding" if sliding else "disjoint"] = bootstrap_results
        _save_bootstrap_results_bundle(cache_file, existing_bundle)
    return bootstrap_results


def fit_methods_for_series(
    vec: np.ndarray,
    *,
    quantile: float,
    random_state: int,
    method_ids: Iterable[str] | None = None,
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

    Parameters
    ----------
    method_ids
        Optional subset of method identifiers to fit. When provided, the
        shared scheme-level bootstrap only materializes the targets required by
        that subset.
    """
    reuse_fits = {} if reuse_fits is None else dict(reuse_fits)
    fgls_bootstrap_overrides = (
        {} if fgls_bootstrap_overrides is None else dict(fgls_bootstrap_overrides)
    )
    selected_specs = list(METHOD_SPECS)
    if method_ids is not None:
        selected_specs = []
        for method_id in method_ids:
            try:
                selected_specs.append(METHOD_LOOKUP[str(method_id)])
            except KeyError as exc:
                raise ValueError(f"Unknown benchmark method id: {method_id!r}") from exc
        if not selected_specs:
            return {}
    values = np.asarray(vec, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    block_sizes = generate_block_sizes(n_obs=values.size)
    fits: dict[str, ScalingFit] = dict(reuse_fits)
    grouped_specs = {
        True: [spec for spec in selected_specs if spec.sliding],
        False: [spec for spec in selected_specs if not spec.sliding],
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
    return {
        spec.method_id: fits[spec.method_id] for spec in selected_specs if spec.method_id in fits
    }
