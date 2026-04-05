"""Synthetic EI benchmark raw-computation entrypoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import resolve_repo_dirs
from workflows.benchmark_design import (
    BENCHMARK_MASTER_SEED,
    SimulationConfig,
    default_ei_simulation_configs,
)
from workflows.ei_benchmark_eval import EI_EXTERNAL_METHODS, EI_INTERNAL_METHODS, run_ei_benchmark

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


def _expected_main_xi(configs: list[SimulationConfig]) -> list[float]:
    return sorted({cfg.xi_true for cfg in configs if cfg.benchmark_set == "main"})


def _expected_main_theta(configs: list[SimulationConfig]) -> list[float]:
    return sorted({cfg.theta_true for cfg in configs if cfg.benchmark_set == "main"})


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
    main_xi = sorted(summary.loc[summary["benchmark_set"] == "main", "xi_true"].dropna().unique())
    main_theta = sorted(
        summary.loc[summary["benchmark_set"] == "main", "theta_true"].dropna().unique()
    )
    expected_sets = {cfg.benchmark_set for cfg in configs}
    return (
        _EI_SUMMARY_REQUIRED_COLUMNS.issubset(summary.columns)
        and method_names == expected_methods
        and benchmark_sets == expected_sets
        and main_xi == _expected_main_xi(configs)
        and main_theta == _expected_main_theta(configs)
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
            return EiBenchmarkOutputs(
                detail_path=paths["detail"],
                summary_path=paths["summary"],
                external_detail_path=paths["external_detail"],
                external_summary_path=paths["external_summary"],
                summary=summary,
                external_summary=external_summary,
            )

    detail, summary, external_detail, external_summary = run_ei_benchmark(
        random_state=EI_BENCHMARK_RANDOM_STATE,
        configs=configs,
        cache_dir=cache_dir,
        max_workers=max_workers,
    )
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
    outputs = load_or_materialize_ei_benchmark_outputs(force=True)
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(outputs.summary)


if __name__ == "__main__":
    main()
