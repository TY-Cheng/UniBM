"""Synthetic EVI benchmark main entrypoint.

This module owns the raw benchmark computation and CSV materialization. The
manuscript workflow reads those cached CSVs and is solely responsible for
manuscript-facing figures and LaTeX tables.
"""
# ruff: noqa: E402

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    import importlib.util

    _helper_path = Path(__file__).resolve().parents[1] / "shared" / "import_bootstrap.py"
    _spec = importlib.util.spec_from_file_location("_shared_import_bootstrap", _helper_path)
    if _spec is None or _spec.loader is None:  # pragma: no cover - import bootstrap failure
        raise ImportError(f"Could not load import bootstrap helper from {_helper_path}.")
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    _module.ensure_scripts_on_path_from_entry(__file__)

from config import resolve_repo_dirs
from unibm.models import ScalingFit

from benchmark.design import (
    BENCHMARK_MASTER_SEED,
    BENCHMARK_SET_LABELS,
    BLOCK_LINESTYLES,
    default_evi_simulation_configs,
    FAMILY_LABELS,
    FGLS_BOOTSTRAP_REPS,
    METHOD_LABELS,
    METHOD_LOOKUP,
    METHOD_ORDER,
    METHOD_SPECS,
    METRIC_LABELS,
    REGRESSION_MARKERS,
    resolve_benchmark_workers,
    TARGET_COLORS,
    MethodSpec,
    SimulationConfig,
    default_simulation_configs,
    fit_methods_for_series,
    load_or_simulate_series_bank,
    scenario_random_state,
    sort_by_method_order,
)
from benchmark.common import (
    interval_score,
    interval_width,
)
from benchmark.evi_report import (
    benchmark_summary,
)
from benchmark.evi_external import (
    EXTERNAL_ESTIMATORS,
    run_external_benchmark,
)
from shared.runtime import status


BENCHMARK_ALPHA = 0.05
BENCHMARK_RANDOM_STATE = BENCHMARK_MASTER_SEED

_INTERNAL_SUMMARY_REQUIRED_COLUMNS = {
    "benchmark_set",
    "family",
    "n_obs",
    "xi_true",
    "theta_true",
    "phi",
    "method",
    "ape_median",
    "ape_q25",
    "ape_q75",
    "summary_target",
    "block_scheme",
    "regression",
    "mape",
    "coverage",
    "coverage_lo",
    "coverage_hi",
    "interval_width_mean",
    "interval_score_mean",
    "interval_score_q25",
    "interval_score_q75",
}
_EXTERNAL_SUMMARY_REQUIRED_COLUMNS = {
    "benchmark_set",
    "family",
    "n_obs",
    "xi_true",
    "theta_true",
    "phi",
    "method",
    "ci_method",
    "ape_median",
    "ape_q25",
    "ape_q75",
    "mape",
    "coverage",
    "coverage_lo",
    "coverage_hi",
    "interval_width_mean",
    "interval_score_mean",
    "interval_score_q25",
    "interval_score_q75",
}


@dataclass(frozen=True)
class EviBenchmarkOutputs:
    """Paths plus loaded EVI summary tables for one sample-size regime."""

    detail_path: Path
    summary_path: Path
    external_detail_path: Path
    external_summary_path: Path
    summary: pd.DataFrame
    external_summary: pd.DataFrame


def _contains(interval: tuple[float, float], value: float) -> bool:
    return bool(interval[0] <= value <= interval[1])


def _result_row(
    cfg: SimulationConfig,
    rep: int,
    method: str,
    fit: ScalingFit,
) -> dict[str, Any]:
    spec = METHOD_LOOKUP[method]
    signed_error = fit.slope - cfg.xi_true
    abs_error = abs(signed_error)
    ci_lo, ci_hi = fit.confidence_interval
    return {
        "benchmark_set": cfg.benchmark_set,
        "family": cfg.family,
        "scenario": cfg.scenario,
        "rep": rep,
        "n_obs": cfg.n_obs,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "block_scheme": spec.block_scheme,
        "summary_target": spec.summary_target,
        "regression": spec.regression,
        "xi_true": cfg.xi_true,
        "theta_true": cfg.theta_true,
        "phi": cfg.phi,
        "xi_hat": fit.slope,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "signed_error": signed_error,
        "abs_error": abs_error,
        "relative_error": abs_error / cfg.xi_true,
        "interval_width": interval_width(ci_lo, ci_hi),
        "interval_score": interval_score(
            cfg.xi_true,
            ci_lo,
            ci_hi,
            alpha=BENCHMARK_ALPHA,
        ),
        "covered": _contains(fit.confidence_interval, cfg.xi_true),
        "plateau_lo": fit.plateau_bounds[0],
        "plateau_hi": fit.plateau_bounds[1],
    }


def evaluate_config(
    cfg: SimulationConfig,
    *,
    random_state: int = 0,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Evaluate every benchmark method on every replicate of one scenario."""
    rows: list[dict[str, Any]] = []
    series_bank = load_or_simulate_series_bank(
        cfg,
        random_state=random_state,
        cache_dir=cache_dir,
    )
    for rep, vec in enumerate(series_bank):
        fits = fit_methods_for_series(
            vec,
            quantile=cfg.quantile,
            random_state=rep,
            cache_dir=cache_dir,
            cache_key=f"{cfg.scenario}__seed{random_state}__rep{rep:04d}",
        )
        for method, fit in fits.items():
            rows.append(_result_row(cfg, rep, method, fit))
    return pd.DataFrame(rows)


def _evaluate_config_worker(args: tuple[SimulationConfig, int, Path | None]) -> pd.DataFrame:
    """Process-pool wrapper for one internal benchmark scenario."""
    cfg, random_state, cache_dir = args
    return evaluate_config(
        cfg,
        random_state=random_state,
        cache_dir=cache_dir,
    )


def run_evi_benchmark(
    random_state: int = 0,
    *,
    configs: list[SimulationConfig] | None = None,
    cache_dir: Path | None = None,
    max_workers: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full default EVI benchmark grid and return detail and summary tables.

    Scenario-level multiprocessing is the main runtime win here because each
    `SimulationConfig` is independent and can read/write its own cache files
    without interfering with the others.
    """
    if configs is None:
        configs = default_evi_simulation_configs()
    workers = resolve_benchmark_workers(len(configs), max_workers=max_workers)
    status(
        "evi_benchmark",
        f"evaluating {len(configs)} internal scenarios with {workers} worker process"
        f"{'' if workers == 1 else 'es'}",
    )
    tasks = [
        (cfg, scenario_random_state(cfg, master_seed=random_state), cache_dir) for cfg in configs
    ]
    if workers == 1:
        frames = [_evaluate_config_worker(task) for task in tasks]
    else:
        # Keep BLAS thread counts low per worker to avoid oversubscribing cores.
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        try:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
                frames = list(executor.map(_evaluate_config_worker, tasks, chunksize=1))
        except (OSError, PermissionError):
            # Some constrained environments disallow the semaphore/process-pool
            # setup required by `ProcessPoolExecutor`. Fall back to sequential
            # execution there; normal local runs still benefit from parallelism.
            frames = [_evaluate_config_worker(task) for task in tasks]
    frames = [frame for frame in frames if not frame.empty]
    detail = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    status("evi_benchmark", "aggregating internal benchmark summaries")
    summary = benchmark_summary(detail)
    return detail, summary


def _resolve_benchmark_n_obs() -> int:
    """Resolve an optional benchmark sample-size override from the environment."""
    raw_value = os.environ.get("UNIBM_BENCHMARK_N_OBS")
    if raw_value is None:
        return 365
    try:
        n_obs = int(raw_value)
    except ValueError as exc:
        raise ValueError("UNIBM_BENCHMARK_N_OBS must be an integer.") from exc
    if n_obs < 32:
        raise ValueError("UNIBM_BENCHMARK_N_OBS must be at least 32.")
    return n_obs


def _output_suffix_for_n_obs(n_obs: int) -> str:
    """Return the filename suffix for a benchmark sample-size regime."""
    return "" if n_obs == 365 else f"_n{n_obs}"


def _output_paths(out_dir: Path, *, n_obs: int) -> dict[str, Path]:
    """Return the canonical CSV paths for one benchmark sample-size regime."""
    suffix = _output_suffix_for_n_obs(n_obs)
    return {
        "detail": out_dir / f"detail{suffix}.csv",
        "summary": out_dir / f"summary{suffix}.csv",
        "external_detail": out_dir / f"external_detail{suffix}.csv",
        "external_summary": out_dir / f"external_summary{suffix}.csv",
    }


def _expected_universal_xi(configs: list[SimulationConfig]) -> list[float]:
    """Return the canonical sorted xi grid for the universal benchmark set."""
    return sorted({cfg.xi_true for cfg in configs})


def _expected_universal_n_obs(configs: list[SimulationConfig]) -> list[int]:
    """Return the canonical n_obs values for the universal benchmark set."""
    return sorted({cfg.n_obs for cfg in configs})


def _expected_universal_theta(configs: list[SimulationConfig]) -> list[float]:
    """Return the canonical theta grid for the universal benchmark set."""
    return sorted({cfg.theta_true for cfg in configs})


def _expected_families(configs: list[SimulationConfig]) -> list[str]:
    """Return the canonical family set for the current benchmark suite."""
    return sorted({str(cfg.family) for cfg in configs})


def _summary_matches_contract(
    summary: pd.DataFrame,
    *,
    required_columns: set[str],
    expected_methods: set[str],
    configs: list[SimulationConfig],
) -> bool:
    """Check whether a cached summary matches the current benchmark contract."""
    method_names = set(summary["method"].dropna().unique()) if "method" in summary else set()
    benchmark_sets = (
        set(summary["benchmark_set"].dropna().unique()) if "benchmark_set" in summary else set()
    )
    families = sorted(summary["family"].dropna().astype(str).unique())
    universal_xi = sorted(summary["xi_true"].dropna().unique())
    universal_theta = sorted(summary["theta_true"].dropna().unique())
    universal_n_obs = sorted(summary["n_obs"].dropna().unique())
    expected_sets = {cfg.benchmark_set for cfg in configs}
    return (
        required_columns.issubset(summary.columns)
        and method_names == expected_methods
        and benchmark_sets == expected_sets
        and families == _expected_families(configs)
        and universal_xi == _expected_universal_xi(configs)
        and universal_theta == _expected_universal_theta(configs)
        and universal_n_obs == _expected_universal_n_obs(configs)
    )


def load_or_materialize_evi_benchmark_outputs(
    root: Path | str = ".",
    *,
    n_obs: int | None = None,
    max_workers: int | None = None,
    force: bool = False,
) -> EviBenchmarkOutputs:
    """Load current EVI benchmark CSVs or rebuild them from the scenario cache.

    `force=True` always refreshes the CSV layer while still reusing the finer
    per-scenario cache under `out/benchmark/cache`.
    """
    dirs = resolve_repo_dirs(root)
    out_dir = dirs["DIR_OUT_BENCHMARK"]
    cache_dir = dirs["DIR_OUT_BENCHMARK_CACHE"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    resolved_n_obs = _resolve_benchmark_n_obs() if n_obs is None else int(n_obs)
    configs = default_evi_simulation_configs(n_obs=resolved_n_obs)
    paths = _output_paths(out_dir, n_obs=resolved_n_obs)
    status(
        "evi_benchmark",
        f"resolving benchmark outputs for n_obs={resolved_n_obs} (force={force})",
    )
    if (
        not force
        and paths["detail"].exists()
        and paths["summary"].exists()
        and paths["external_detail"].exists()
        and paths["external_summary"].exists()
    ):
        summary = pd.read_csv(paths["summary"])
        external_summary = pd.read_csv(paths["external_summary"])
        if _summary_matches_contract(
            summary,
            required_columns=_INTERNAL_SUMMARY_REQUIRED_COLUMNS,
            expected_methods=set(METHOD_ORDER),
            configs=configs,
        ) and _summary_matches_contract(
            external_summary,
            required_columns=_EXTERNAL_SUMMARY_REQUIRED_COLUMNS,
            expected_methods=set(EXTERNAL_ESTIMATORS),
            configs=configs,
        ):
            status("evi_benchmark", "reusing cached internal and external benchmark CSVs")
            return EviBenchmarkOutputs(
                detail_path=paths["detail"],
                summary_path=paths["summary"],
                external_detail_path=paths["external_detail"],
                external_summary_path=paths["external_summary"],
                summary=summary,
                external_summary=external_summary,
            )

    status("evi_benchmark", "running internal EVI benchmark grid")
    detail, summary = run_evi_benchmark(
        random_state=BENCHMARK_RANDOM_STATE,
        configs=configs,
        cache_dir=cache_dir,
        max_workers=max_workers,
    )
    status("evi_benchmark", "writing internal EVI benchmark CSVs")
    detail.to_csv(paths["detail"], index=False)
    summary.to_csv(paths["summary"], index=False)
    status("evi_benchmark", "running external EVI benchmark comparators")
    external_detail, external_summary = run_external_benchmark(
        random_state=BENCHMARK_RANDOM_STATE,
        configs=configs,
        ci_method="asymptotic",
        cache_dir=cache_dir,
        max_workers=max_workers,
    )
    status("evi_benchmark", "writing external EVI benchmark CSVs")
    external_detail.to_csv(paths["external_detail"], index=False)
    external_summary.to_csv(paths["external_summary"], index=False)
    return EviBenchmarkOutputs(
        detail_path=paths["detail"],
        summary_path=paths["summary"],
        external_detail_path=paths["external_detail"],
        external_summary_path=paths["external_summary"],
        summary=summary,
        external_summary=external_summary,
    )


def main() -> None:
    force = os.environ.get("UNIBM_FORCE_BENCHMARK", "").strip() in {"1", "true", "TRUE", "yes"}
    outputs = load_or_materialize_evi_benchmark_outputs(force=force)
    status("evi_benchmark", f"detail: {outputs.detail_path}")
    status("evi_benchmark", f"summary: {outputs.summary_path}")
    status("evi_benchmark", f"external_detail: {outputs.external_detail_path}")
    status("evi_benchmark", f"external_summary: {outputs.external_summary_path}")


__all__ = [
    "BENCHMARK_SET_LABELS",
    "BLOCK_LINESTYLES",
    "FAMILY_LABELS",
    "FGLS_BOOTSTRAP_REPS",
    "METHOD_LABELS",
    "METHOD_LOOKUP",
    "METHOD_ORDER",
    "METHOD_SPECS",
    "METRIC_LABELS",
    "REGRESSION_MARKERS",
    "TARGET_COLORS",
    "BENCHMARK_RANDOM_STATE",
    "EviBenchmarkOutputs",
    "MethodSpec",
    "SimulationConfig",
    "benchmark_summary",
    "default_evi_simulation_configs",
    "default_simulation_configs",
    "evaluate_config",
    "fit_methods_for_series",
    "load_or_materialize_evi_benchmark_outputs",
    "run_evi_benchmark",
    "sort_by_method_order",
]


if __name__ == "__main__":
    main()
