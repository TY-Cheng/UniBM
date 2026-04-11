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
from benchmark.design import default_evi_simulation_configs
from benchmark.evi_benchmark import EviBenchmarkOutputs
from benchmark.ei_report import write_ei_benchmark_manuscript_artifacts
from benchmark.evi_report import (
    EVI_SHRINKAGE_GRID,
    build_evi_benchmark_manuscript_outputs,
    build_evi_shrinkage_sensitivity_summary,
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
                patch("benchmark.evi_report.write_evi_benchmark_manuscript_artifacts"),
            ):
                outputs = build_evi_benchmark_manuscript_outputs(root)
            manuscript_figure_dir = resolve_repo_dirs(root)["DIR_MANUSCRIPT_FIGURE"]
            self.assertEqual(outputs["benchmark_shrinkage_sensitivity_data"], shrinkage_path)
            self.assertEqual(
                outputs["benchmark_shrinkage_sensitivity_figure"],
                manuscript_figure_dir / "benchmark_shrinkage_sensitivity.pdf",
            )

    def test_evi_manuscript_artifact_captions_match_projected_suite(self) -> None:
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
                    fig_dir=fig_dir,
                    table_dir=table_dir,
                )
            core_caption = (table_dir / "benchmark_core_main.tex").read_text()
            targets_caption = (table_dir / "benchmark_targets_main.tex").read_text()
            self.assertIn("projected short-record severity suite", core_caption)
            self.assertIn("{0.01, 0.10, 0.50, 1.0}", core_caption)
            self.assertIn("moving-maxima q=99", core_caption)
            self.assertNotIn("Universal grid", core_caption)
            self.assertIn("projected short-record severity suite", targets_caption)

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


if __name__ == "__main__":
    unittest.main()
