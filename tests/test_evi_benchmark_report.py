from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from benchmark.design import (
    BENCHMARK_MONTE_CARLO_REPS,
    STRESS_BENCHMARK_SET,
    STRESS_MOVING_MAXIMA_FAMILY,
    default_evi_simulation_configs,
    stress_evi_simulation_configs,
)
from benchmark.evi_report import (
    EVI_SHRINKAGE_GRID,
    build_evi_record_length_sensitivity_summary,
    build_evi_stress_suite_summary,
    build_evi_shrinkage_sensitivity_summary,
    evi_record_length_sensitivity_table,
    plot_benchmark_panels,
)


class EviBenchmarkReportTests(unittest.TestCase):
    def test_benchmark_panels_can_save_publication_and_web_formats(self) -> None:
        summary = pd.DataFrame(
            {
                "benchmark_set": ["universal", "universal"],
                "family": ["frechet_max_ar", "frechet_max_ar"],
                "method": ["sliding_median_fgls", "sliding_median_fgls"],
                "summary_target": ["median", "median"],
                "block_scheme": ["sliding", "sliding"],
                "regression": ["FGLS", "FGLS"],
                "theta_true": [0.5, 0.5],
                "xi_true": [0.1, 0.3],
                "ape_median": [0.2, 0.3],
                "ape_q25": [0.1, 0.2],
                "ape_q75": [0.3, 0.4],
                "interval_score_median": [0.5, 0.7],
                "interval_score_q25": [0.4, 0.6],
                "interval_score_q75": [0.6, 0.8],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            publication_path = output_dir / "summary.pdf"
            web_path = output_dir / "summary.png"
            plot_benchmark_panels(
                summary,
                benchmark_set="universal",
                methods=("sliding_median_fgls",),
                file_path=publication_path,
                web_path=web_path,
                save=True,
            )

            self.assertGreater(publication_path.stat().st_size, 0)
            self.assertGreater(web_path.stat().st_size, 0)

    def test_build_evi_shrinkage_sensitivity_summary_emits_expected_grid_and_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            configs = default_evi_simulation_configs(
                xi_values=(0.10,),
                theta_values=(0.50,),
                families=("frechet_max_ar",),
                n_obs=64,
                reps=1,
            )
            summary, output_path = build_evi_shrinkage_sensitivity_summary(
                root=tmpdir,
                configs=configs,
                force=True,
            )
            self.assertEqual(
                sorted(summary["delta"].dropna().unique().tolist()),
                list(EVI_SHRINKAGE_GRID),
            )
            self.assertEqual(summary["family"].drop_duplicates().tolist(), ["frechet_max_ar"])
            self.assertIn("median_ape", summary.columns)
            self.assertIn("median_coverage", summary.columns)
            self.assertIn("median_interval_score", summary.columns)
            self.assertTrue(output_path.exists())
            persisted = pd.read_csv(output_path)
            self.assertIn("delta", persisted.columns)
            self.assertIn("family", persisted.columns)
            self.assertIn("median_ape", persisted.columns)
            self.assertIn("median_coverage", persisted.columns)
            self.assertIn("median_interval_score", persisted.columns)

    def test_build_evi_record_length_sensitivity_summary_emits_expected_n_obs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            configs: list[object] = []
            for n_obs in (64, 96):
                configs.extend(
                    default_evi_simulation_configs(
                        xi_values=(0.10,),
                        theta_values=(0.10,),
                        families=("frechet_max_ar",),
                        n_obs=n_obs,
                        reps=1,
                    )
                )
            summary, output_path = build_evi_record_length_sensitivity_summary(
                root=tmpdir,
                configs=configs,
                force=True,
            )
            self.assertEqual(sorted(summary["n_obs"].dropna().unique().tolist()), [64, 96])
            self.assertEqual(summary["family"].drop_duplicates().tolist(), ["frechet_max_ar"])
            self.assertIn("median_interval_score", summary.columns)
            self.assertTrue(output_path.exists())
            persisted = pd.read_csv(output_path)
            self.assertIn("median_coverage", persisted.columns)

    def test_record_length_sensitivity_table_formats_method_columns(self) -> None:
        summary = pd.DataFrame(
            {
                "family": ["frechet_max_ar"] * 3,
                "n_obs": [200, 200, 200],
                "method": [
                    "disjoint_median_ols",
                    "sliding_median_ols",
                    "sliding_median_fgls",
                ],
                "method_label": [
                    "median-disjoint-OLS",
                    "median-sliding-OLS",
                    "median-sliding-FGLS",
                ],
                "median_ape": [0.4, 0.3, 0.2],
                "ape_q25": [0.3, 0.2, 0.1],
                "ape_q75": [0.5, 0.4, 0.3],
                "median_interval_score": [4.0, 2.0, 0.8],
                "interval_score_q25": [3.0, 1.5, 0.6],
                "interval_score_q75": [5.0, 2.5, 1.0],
                "median_coverage": [0.8, 0.9, 0.95],
            }
        )

        table = evi_record_length_sensitivity_table(summary)

        self.assertEqual(table.shape[0], 1)
        self.assertEqual(table.iloc[0]["Family"], "Fréchet max-AR")
        self.assertEqual(int(table.iloc[0]["n_obs"]), 200)
        self.assertIn("median-sliding-FGLS", table.columns)

    def test_stress_configs_use_abs_student_t_family_and_stress_set(self) -> None:
        configs = stress_evi_simulation_configs()

        self.assertTrue(configs)
        self.assertEqual({cfg.benchmark_set for cfg in configs}, {STRESS_BENCHMARK_SET})
        self.assertEqual({cfg.family for cfg in configs}, {STRESS_MOVING_MAXIMA_FAMILY})
        self.assertEqual({cfg.n_obs for cfg in configs}, {365})
        self.assertEqual({cfg.reps for cfg in configs}, {BENCHMARK_MONTE_CARLO_REPS})

    def test_build_evi_stress_suite_summary_emits_expected_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            configs = stress_evi_simulation_configs(
                xi_values=(0.10,),
                theta_values=(0.50,),
                reps=1,
            )
            summary, output_path = build_evi_stress_suite_summary(
                root=tmpdir,
                configs=configs,
                force=True,
                max_workers=1,
            )
            self.assertEqual(summary["benchmark_set"].drop_duplicates().tolist(), ["stress"])
            self.assertEqual(
                summary["family"].drop_duplicates().tolist(),
                [STRESS_MOVING_MAXIMA_FAMILY],
            )
            self.assertTrue(output_path.exists())
            persisted = pd.read_csv(output_path)
            self.assertIn("benchmark_set", persisted.columns)
            self.assertIn("family", persisted.columns)
            self.assertIn("method", persisted.columns)
