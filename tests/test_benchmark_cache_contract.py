from __future__ import annotations
# ruff: noqa: E402

import itertools
import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
for path in (SCRIPTS_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark.design import (
    default_ei_simulation_configs,
    default_evi_simulation_configs,
    parse_moving_maxima_q,
)
from benchmark.ei_benchmark import _summary_matches_contract as ei_summary_matches_contract
from benchmark.evi_benchmark import (
    _INTERNAL_SUMMARY_REQUIRED_COLUMNS,
    _summary_matches_contract as evi_summary_matches_contract,
)


def _make_frame(
    *,
    methods: set[str],
    benchmark_set: str,
    families: tuple[str, ...],
    xi_values: tuple[float, ...],
    theta_values: tuple[float, ...],
    n_obs: int,
    required_columns: set[str],
) -> pd.DataFrame:
    rows = []
    for method, family, xi_true, theta_true in itertools.product(
        sorted(methods),
        families,
        xi_values,
        theta_values,
    ):
        row = {
            column: 0.0
            for column in required_columns
            if column not in {"benchmark_set", "family", "method"}
        }
        row.update(
            {
                "benchmark_set": benchmark_set,
                "family": family,
                "n_obs": n_obs,
                "xi_true": xi_true,
                "theta_true": theta_true,
                "phi": 0.0,
                "method": method,
                "ci_method": "wald",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


class BenchmarkCacheContractTests(unittest.TestCase):
    def test_benchmark_design_only_accepts_moving_maxima_q99(self) -> None:
        self.assertEqual(parse_moving_maxima_q("moving_maxima_q99"), 99)

    def test_ei_summary_contract_rejects_family_mismatch(self) -> None:
        configs = default_ei_simulation_configs(
            xi_values=(0.50,),
            theta_values=(0.25,),
            families=("frechet_max_ar", "moving_maxima_q99"),
            reps=1,
            n_obs=365,
        )
        summary = _make_frame(
            methods={"northrop_sliding_fgls"},
            benchmark_set="universal",
            families=("frechet_max_ar", "pareto_additive_ar1"),
            xi_values=(0.50,),
            theta_values=(0.25,),
            n_obs=365,
            required_columns={
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
            },
        )
        self.assertFalse(
            ei_summary_matches_contract(
                summary,
                expected_methods={"northrop_sliding_fgls"},
                configs=configs,
            )
        )

    def test_ei_summary_contract_rejects_n_obs_mismatch(self) -> None:
        configs = default_ei_simulation_configs(
            xi_values=(0.50,),
            theta_values=(0.25,),
            families=("frechet_max_ar",),
            reps=1,
            n_obs=365,
        )
        summary = _make_frame(
            methods={"northrop_sliding_fgls"},
            benchmark_set="universal",
            families=("frechet_max_ar",),
            xi_values=(0.50,),
            theta_values=(0.25,),
            n_obs=730,
            required_columns={
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
            },
        )
        self.assertFalse(
            ei_summary_matches_contract(
                summary,
                expected_methods={"northrop_sliding_fgls"},
                configs=configs,
            )
        )

    def test_evi_summary_contract_rejects_family_mismatch(self) -> None:
        configs = default_evi_simulation_configs(
            xi_values=(0.10,),
            theta_values=(0.50,),
            families=("frechet_max_ar", "moving_maxima_q99"),
            reps=1,
            n_obs=365,
        )
        summary = _make_frame(
            methods={"sliding_median_fgls"},
            benchmark_set="universal",
            families=("frechet_max_ar", "pareto_additive_ar1"),
            xi_values=(0.10,),
            theta_values=(0.50,),
            n_obs=365,
            required_columns=_INTERNAL_SUMMARY_REQUIRED_COLUMNS,
        )
        self.assertFalse(
            evi_summary_matches_contract(
                summary,
                required_columns=_INTERNAL_SUMMARY_REQUIRED_COLUMNS,
                expected_methods={"sliding_median_fgls"},
                configs=configs,
            )
        )


if __name__ == "__main__":
    unittest.main()
