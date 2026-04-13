from __future__ import annotations
# ruff: noqa: E402

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
for path in (SCRIPTS_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import resolve_repo_dirs
from benchmark.design import (
    STRESS_BENCHMARK_SET,
    STRESS_MOVING_MAXIMA_FAMILY,
    default_evi_simulation_configs,
    stress_evi_simulation_configs,
)
from benchmark.evi_benchmark import EviBenchmarkOutputs
from benchmark.ei_report import write_ei_benchmark_manuscript_artifacts
from benchmark.evi_report import (
    EVI_SHRINKAGE_GRID,
    build_evi_benchmark_manuscript_outputs,
    build_evi_record_length_sensitivity_summary,
    build_evi_stress_suite_summary,
    build_evi_shrinkage_sensitivity_summary,
    evi_record_length_sensitivity_table,
    write_evi_benchmark_manuscript_artifacts,
)


class EviBenchmarkReportTests(unittest.TestCase):
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

    def test_build_evi_benchmark_manuscript_outputs_returns_shrinkage_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            out_dir = root / "out" / "benchmark"
            summary_path = out_dir / "summary.csv"
            external_summary_path = out_dir / "external_summary.csv"
            fake_outputs = EviBenchmarkOutputs(
                detail_path=out_dir / "detail.csv",
                summary_path=summary_path,
                external_detail_path=out_dir / "external_detail.csv",
                external_summary_path=external_summary_path,
                summary=pd.DataFrame({"method": ["sliding_median_fgls"]}),
                external_summary=pd.DataFrame({"method": ["hill"]}),
            )
            shrinkage_path = out_dir / "benchmark_shrinkage_sensitivity.csv"
            with (
                patch(
                    "benchmark.evi_benchmark.load_or_materialize_evi_benchmark_outputs",
                    return_value=fake_outputs,
                ),
                patch(
                    "benchmark.evi_report.build_evi_shrinkage_sensitivity_summary",
                    return_value=(pd.DataFrame(), shrinkage_path),
                ),
                patch(
                    "benchmark.evi_report.build_evi_stress_suite_summary",
                    return_value=(pd.DataFrame(), out_dir / "benchmark_stress_summary.csv"),
                ),
                patch(
                    "benchmark.evi_report.build_evi_record_length_sensitivity_summary",
                    return_value=(
                        pd.DataFrame(),
                        out_dir / "benchmark_record_length_sensitivity.csv",
                    ),
                ),
                patch("benchmark.evi_report.write_evi_benchmark_manuscript_artifacts"),
            ):
                outputs = build_evi_benchmark_manuscript_outputs(root)
            manuscript_figure_dir = resolve_repo_dirs(root)["DIR_MANUSCRIPT_FIGURE"]
            manuscript_table_dir = resolve_repo_dirs(root)["DIR_MANUSCRIPT_TABLE"]
            self.assertEqual(outputs["benchmark_shrinkage_sensitivity_data"], shrinkage_path)
            self.assertEqual(
                outputs["benchmark_shrinkage_sensitivity_figure"],
                manuscript_figure_dir / "benchmark_shrinkage_sensitivity.pdf",
            )
            self.assertEqual(
                outputs["benchmark_stress_summary_figure"],
                manuscript_figure_dir / "benchmark_stress_summary.pdf",
            )
            self.assertEqual(
                outputs["benchmark_record_length_main"],
                manuscript_table_dir / "benchmark_record_length_main.tex",
            )

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
        self.assertEqual(table.iloc[0]["Family"], "Frechet max-AR")
        self.assertEqual(int(table.iloc[0]["n_obs"]), 200)
        self.assertIn("median-sliding-FGLS", table.columns)

    def test_stress_configs_use_abs_student_t_family_and_stress_set(self) -> None:
        configs = stress_evi_simulation_configs()

        self.assertTrue(configs)
        self.assertEqual({cfg.benchmark_set for cfg in configs}, {STRESS_BENCHMARK_SET})
        self.assertEqual({cfg.family for cfg in configs}, {STRESS_MOVING_MAXIMA_FAMILY})
        self.assertEqual({cfg.n_obs for cfg in configs}, {365})
        self.assertEqual({cfg.reps for cfg in configs}, {32})

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

    def test_evi_manuscript_artifact_captions_match_projected_suite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fig_dir = root / "Figure"
            table_dir = root / "Table"
            fig_dir.mkdir(parents=True, exist_ok=True)
            table_dir.mkdir(parents=True, exist_ok=True)
            benchmark_summary = pd.DataFrame({"benchmark_set": ["universal"], "n_obs": [365]})
            external_summary = pd.DataFrame({"benchmark_set": ["universal"]})
            record_length_summary = pd.DataFrame(
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
            with (
                patch(
                    "benchmark.evi_report.benchmark_story_latex",
                    side_effect=lambda *args, caption, label, **kwargs: caption,
                ),
                patch(
                    "benchmark.evi_external.target_plus_external_story_latex",
                    side_effect=lambda *args, caption, label, **kwargs: caption,
                ),
                patch(
                    "benchmark.evi_external.interval_sharpness_story_latex",
                    side_effect=lambda *args, caption, label, **kwargs: caption,
                ),
                patch(
                    "benchmark.evi_report.render_latex_table",
                    side_effect=lambda *args, caption, label, **kwargs: caption,
                ),
                patch("benchmark.evi_report.benchmark_table", return_value=pd.DataFrame()),
                patch("benchmark.evi_report.plot_benchmark_panels"),
                patch("benchmark.evi_report.plot_evi_shrinkage_sensitivity"),
                patch("benchmark.evi_external.plot_interval_sharpness_scatter"),
                patch("benchmark.evi_external.plot_target_plus_external_panels"),
            ):
                write_evi_benchmark_manuscript_artifacts(
                    benchmark_summary,
                    external_summary,
                    shrinkage_sensitivity_summary=pd.DataFrame(),
                    record_length_summary=record_length_summary,
                    fig_dir=fig_dir,
                    table_dir=table_dir,
                )
            core_caption = (table_dir / "benchmark_core_main.tex").read_text()
            targets_caption = (table_dir / "benchmark_targets_main.tex").read_text()
            record_length_caption = (table_dir / "benchmark_record_length_main.tex").read_text()
            self.assertIn("projected short-record severity suite", core_caption)
            self.assertIn("{0.01, 0.10, 0.50, 1.0}", core_caption)
            self.assertIn("moving-maxima q=99", core_caption)
            self.assertNotIn("Universal grid", core_caption)
            self.assertIn("projected short-record severity suite", targets_caption)
            self.assertIn("not used to rank cross-class interval calibration", targets_caption)
            self.assertIn("record-length sensitivity", record_length_caption)

    def test_ei_manuscript_artifact_captions_match_projected_suite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fig_dir = root / "Figure"
            table_dir = root / "Table"
            fig_dir.mkdir(parents=True, exist_ok=True)
            table_dir.mkdir(parents=True, exist_ok=True)
            benchmark_summary = pd.DataFrame({"benchmark_set": ["universal"], "n_obs": [365]})
            external_summary = pd.DataFrame({"benchmark_set": ["universal"]})
            with (
                patch(
                    "benchmark.ei_report.ei_story_latex",
                    side_effect=lambda *args, caption, label, **kwargs: caption,
                ),
                patch(
                    "benchmark.ei_report.render_latex_table",
                    side_effect=lambda *args, caption, label, **kwargs: caption,
                ),
                patch("benchmark.ei_report.ei_core_story_table", return_value=pd.DataFrame()),
                patch("benchmark.ei_report.ei_targets_story_table", return_value=pd.DataFrame()),
                patch(
                    "benchmark.ei_report.ei_interval_story_table",
                    return_value=pd.DataFrame(
                        {
                            "median_interval_width": [0.1],
                            "coverage_median": [0.95],
                            "median_interval_score": [0.2],
                        }
                    ),
                ),
                patch("benchmark.ei_report.plot_ei_core_panels"),
                patch("benchmark.ei_report.plot_ei_targets_panels"),
                patch("benchmark.ei_report.plot_ei_interval_sharpness_scatter"),
                patch("benchmark.ei_report.plot_ei_overview_panels"),
            ):
                write_ei_benchmark_manuscript_artifacts(
                    benchmark_summary,
                    external_summary,
                    fig_dir=fig_dir,
                    table_dir=table_dir,
                )
            core_caption = (table_dir / "benchmark_ei_core_main.tex").read_text()
            interval_caption = (table_dir / "benchmark_ei_interval_main.tex").read_text()
            self.assertIn("projected short-record persistence suite", core_caption)
            self.assertIn("{0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 1.0}", core_caption)
            self.assertIn("moving-maxima q=99", core_caption)
            self.assertNotIn("Universal grid", core_caption)
            self.assertIn("projected EI suite", interval_caption)
            targets_caption = (table_dir / "benchmark_ei_targets_main.tex").read_text()
            self.assertIn("not used to rank cross-class interval calibration", targets_caption)


if __name__ == "__main__":
    unittest.main()
