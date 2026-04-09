"""Synthetic EI benchmark raw-computation entrypoint."""
# ruff: noqa: E402

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

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
from benchmark.design import (
    BENCHMARK_MASTER_SEED,
    SimulationConfig,
    default_ei_simulation_configs,
)
from benchmark.ei_eval import EI_EXTERNAL_METHODS, EI_INTERNAL_METHODS, run_ei_benchmark
from shared.runtime import status

EI_BENCHMARK_RANDOM_STATE = BENCHMARK_MASTER_SEED

_EI_SUMMARY_REQUIRED_COLUMNS = {
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
    "coverage",
    "coverage_lo",
    "coverage_hi",
    "interval_width_mean",
    "interval_score_mean",
    "interval_score_q25",
    "interval_score_q75",
}


@dataclass(frozen=True)
class EiBenchmarkOutputs:
    """Paths plus loaded EI summary tables for one sample-size regime."""

    detail_path: Path
    summary_path: Path
    external_detail_path: Path
    external_summary_path: Path
    summary: pd.DataFrame
    external_summary: pd.DataFrame


def _resolve_benchmark_n_obs() -> int:
    return 365


def _output_suffix_for_n_obs(n_obs: int) -> str:
    return "" if n_obs == 365 else f"_n{n_obs}"


def _output_paths(out_dir: Path, *, n_obs: int) -> dict[str, Path]:
    suffix = _output_suffix_for_n_obs(n_obs)
    return {
        "detail": out_dir / f"ei_detail{suffix}.csv",
        "summary": out_dir / f"ei_summary{suffix}.csv",
        "external_detail": out_dir / f"ei_external_detail{suffix}.csv",
        "external_summary": out_dir / f"ei_external_summary{suffix}.csv",
    }


def _expected_universal_xi(configs: list[SimulationConfig]) -> list[float]:
    return sorted({cfg.xi_true for cfg in configs})


def _expected_universal_theta(configs: list[SimulationConfig]) -> list[float]:
    return sorted({cfg.theta_true for cfg in configs})


def _expected_families(configs: list[SimulationConfig]) -> list[str]:
    return sorted({str(cfg.family) for cfg in configs})


def _expected_universal_n_obs(configs: list[SimulationConfig]) -> list[int]:
    return sorted({int(cfg.n_obs) for cfg in configs})


def _summary_matches_contract(
    summary: pd.DataFrame,
    *,
    expected_methods: set[str],
    configs: list[SimulationConfig],
) -> bool:
    method_names = set(summary["method"].dropna().unique()) if "method" in summary else set()
    benchmark_sets = (
        set(summary["benchmark_set"].dropna().unique()) if "benchmark_set" in summary else set()
    )
    families = sorted(summary["family"].dropna().astype(str).unique())
    universal_xi = sorted(summary["xi_true"].dropna().unique())
    universal_theta = sorted(summary["theta_true"].dropna().unique())
    universal_n_obs = sorted(summary["n_obs"].dropna().astype(int).unique())
    expected_sets = {cfg.benchmark_set for cfg in configs}
    return (
        _EI_SUMMARY_REQUIRED_COLUMNS.issubset(summary.columns)
        and method_names == expected_methods
        and benchmark_sets == expected_sets
        and families == _expected_families(configs)
        and universal_xi == _expected_universal_xi(configs)
        and universal_theta == _expected_universal_theta(configs)
        and universal_n_obs == _expected_universal_n_obs(configs)
    )


def load_or_materialize_ei_benchmark_outputs(
    root: Path | str = ".",
    *,
    n_obs: int | None = None,
    max_workers: int | None = None,
    force: bool = False,
) -> EiBenchmarkOutputs:
    """Load current EI benchmark CSVs or rebuild them from cache."""
    dirs = resolve_repo_dirs(root)
    out_dir = dirs["DIR_OUT_BENCHMARK"]
    cache_dir = dirs["DIR_OUT_BENCHMARK_CACHE"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    resolved_n_obs = _resolve_benchmark_n_obs() if n_obs is None else int(n_obs)
    configs = default_ei_simulation_configs(n_obs=resolved_n_obs)
    paths = _output_paths(out_dir, n_obs=resolved_n_obs)
    status(
        "ei_benchmark",
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
            expected_methods=set(EI_INTERNAL_METHODS),
            configs=configs,
        ) and _summary_matches_contract(
            external_summary,
            expected_methods=set(EI_EXTERNAL_METHODS),
            configs=configs,
        ):
            status("ei_benchmark", "reusing cached internal and external benchmark CSVs")
            return EiBenchmarkOutputs(
                detail_path=paths["detail"],
                summary_path=paths["summary"],
                external_detail_path=paths["external_detail"],
                external_summary_path=paths["external_summary"],
                summary=summary,
                external_summary=external_summary,
            )

    status("ei_benchmark", "running pooled-BM and external EI benchmark grid")
    detail, summary, external_detail, external_summary = run_ei_benchmark(
        random_state=EI_BENCHMARK_RANDOM_STATE,
        configs=configs,
        cache_dir=cache_dir,
        max_workers=max_workers,
    )
    status("ei_benchmark", "writing EI benchmark CSVs")
    detail.to_csv(paths["detail"], index=False)
    summary.to_csv(paths["summary"], index=False)
    external_detail.to_csv(paths["external_detail"], index=False)
    external_summary.to_csv(paths["external_summary"], index=False)
    return EiBenchmarkOutputs(
        detail_path=paths["detail"],
        summary_path=paths["summary"],
        external_detail_path=paths["external_detail"],
        external_summary_path=paths["external_summary"],
        summary=summary,
        external_summary=external_summary,
    )


def main() -> None:
    force = os.environ.get("UNIBM_FORCE_BENCHMARK", "").strip() in {"1", "true", "TRUE", "yes"}
    outputs = load_or_materialize_ei_benchmark_outputs(force=force)
    status("ei_benchmark", f"detail: {outputs.detail_path}")
    status("ei_benchmark", f"summary: {outputs.summary_path}")
    status("ei_benchmark", f"external_detail: {outputs.external_detail_path}")
    status("ei_benchmark", f"external_summary: {outputs.external_summary_path}")


if __name__ == "__main__":
    main()
