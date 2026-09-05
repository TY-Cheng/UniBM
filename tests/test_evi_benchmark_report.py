from __future__ import annotations

import tempfile
import unittest
from unittest import mock
from pathlib import Path

import pandas as pd
import numpy as np

from benchmark.design import default_evi_simulation_configs
from benchmark.evi_report import (
    EVI_SHRINKAGE_GRID,
    benchmark_summary,
    benchmark_story_table,
    build_evi_shrinkage_sensitivity_summary,
    plot_benchmark_panels,
)
from benchmark.evi_benchmark import evaluate_config
from unibm.evi import estimate_target_scaling


class EviBenchmarkReportTests(unittest.TestCase):
    def test_shrinkage_sensitivity_does_not_relabel_default_as_point35(self) -> None:
        configs = default_evi_simulation_configs(
            xi_values=(0.10,),
            theta_values=(0.50,),
            families=("frechet_max_ar",),
            n_obs=64,
            reps=1,
        )
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            mock.patch("unibm.evi.estimate_target_scaling", wraps=estimate_target_scaling) as fit,
        ):
            build_evi_shrinkage_sensitivity_summary(
                root=tmpdir,
                configs=configs,
                deltas=(0.35, 0.37),
                force=True,
            )
        self.assertEqual(
            [call.kwargs["covariance_shrinkage"] for call in fit.call_args_list], [0.35]
        )

    def test_primary_score_is_arithmetic_mean_with_mcse(self) -> None:
        cfg = default_evi_simulation_configs(
            xi_values=(0.10,),
            theta_values=(0.50,),
            families=("frechet_max_ar",),
            n_obs=64,
            reps=1,
        )[0]
        template = evaluate_config(cfg, random_state=29).iloc[[0]]
        detail = pd.concat([template] * 3, ignore_index=True)
        detail["rep"] = [0, 1, 2]
        detail["interval_score"] = [1.0, 2.0, 27.0]
        summary = benchmark_summary(detail)
        row = summary.iloc[0]
        self.assertEqual(row.interval_score_mean, 10.0)
        self.assertEqual(row.interval_score_median, 2.0)
        self.assertAlmostEqual(
            row.interval_score_mcse, np.std([1.0, 2.0, 27.0], ddof=1) / np.sqrt(3)
        )
        story = benchmark_story_table(summary, methods=[str(row.method)])
        self.assertTrue(story.iloc[0, 2].startswith("10.000 /"))

    def test_benchmark_detail_records_actual_regression_provenance(self) -> None:
        cfg = default_evi_simulation_configs(
            xi_values=(0.10,),
            theta_values=(0.50,),
            families=("frechet_max_ar",),
            n_obs=64,
            reps=1,
        )[0]

        detail = evaluate_config(cfg, random_state=29)

        self.assertTrue({"regression_policy", "regression", "ci_variant"}.issubset(detail.columns))
        self.assertTrue((detail["regression_policy"] == detail["regression"]).all())
        self.assertEqual(set(detail["ci_variant"]), {"hc0", "bootstrap_cov"})

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
                "interval_score_mean": [0.55, 0.75],
                "interval_score_lo": [0.45, 0.65],
                "interval_score_hi": [0.65, 0.85],
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
            self.assertIn("mean_interval_score", summary.columns)
            self.assertTrue(output_path.exists())
            persisted = pd.read_csv(output_path)
            self.assertIn("delta", persisted.columns)
            self.assertIn("family", persisted.columns)
            self.assertIn("median_ape", persisted.columns)
            self.assertIn("median_coverage", persisted.columns)
            self.assertIn("mean_interval_score", persisted.columns)
