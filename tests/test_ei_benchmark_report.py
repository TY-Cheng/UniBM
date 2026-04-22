from __future__ import annotations
# ruff: noqa: E402

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pandas as pd

try:
    from . import _path_setup as test_paths
except ImportError:  # pragma: no cover
    import _path_setup as test_paths

test_paths.ensure_repo_import_paths()

from benchmark.design import default_ei_simulation_configs
from benchmark.ei_eval import run_ei_benchmark
from benchmark.ei_report import (
    EI_SHRINKAGE_GRID,
    EI_SHRINKAGE_METHODS,
    build_ei_shrinkage_sensitivity_summary,
    write_ei_benchmark_manuscript_artifacts,
)


class EiBenchmarkReportTests(unittest.TestCase):
    def test_build_ei_shrinkage_sensitivity_summary_emits_expected_grid_and_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            configs = default_ei_simulation_configs(
                xi_values=(0.50,),
                theta_values=(0.25,),
                families=("frechet_max_ar",),
                n_obs=64,
                reps=1,
            )
            summary, output_path = build_ei_shrinkage_sensitivity_summary(
                root=tmpdir,
                configs=configs,
                force=True,
            )
            self.assertEqual(
                sorted(summary["delta"].dropna().unique().tolist()),
                list(EI_SHRINKAGE_GRID),
            )
            self.assertEqual(
                summary["method"].drop_duplicates().tolist(),
                list(EI_SHRINKAGE_METHODS),
            )
            self.assertEqual(summary["family"].drop_duplicates().tolist(), ["frechet_max_ar"])
            self.assertIn("median_ape", summary.columns)
            self.assertIn("median_coverage", summary.columns)
            self.assertIn("median_interval_score", summary.columns)
            self.assertTrue(output_path.exists())
            persisted = pd.read_csv(output_path)
            self.assertIn("delta", persisted.columns)
            self.assertIn("method", persisted.columns)
            self.assertIn("median_ape", persisted.columns)
            self.assertIn("median_coverage", persisted.columns)
            self.assertIn("median_interval_score", persisted.columns)

    def test_write_ei_benchmark_manuscript_artifacts_materializes_expected_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            configs = default_ei_simulation_configs(
                xi_values=(0.50,),
                theta_values=(0.25,),
                families=("frechet_max_ar",),
                n_obs=64,
                reps=1,
            )
            _, internal_summary, _, external_summary = run_ei_benchmark(
                configs=configs,
                cache_dir=Path(tmpdir) / "cache",
                max_workers=1,
            )
            fig_dir = Path(tmpdir) / "figures"
            table_dir = Path(tmpdir) / "tables"
            fig_dir.mkdir(parents=True, exist_ok=True)
            table_dir.mkdir(parents=True, exist_ok=True)
            with (
                mock.patch("benchmark.ei_report.plot_ei_core_panels") as core_panels,
                mock.patch("benchmark.ei_report.plot_ei_targets_panels") as target_panels,
                mock.patch(
                    "benchmark.ei_report.plot_ei_interval_sharpness_scatter"
                ) as sharpness_scatter,
                mock.patch("benchmark.ei_report.plot_ei_overview_panels") as overview_panels,
                mock.patch(
                    "benchmark.ei_report.plot_ei_shrinkage_sensitivity"
                ) as shrinkage_panels,
            ):
                write_ei_benchmark_manuscript_artifacts(
                    internal_summary,
                    external_summary,
                    shrinkage_sensitivity_summary=None,
                    fig_dir=fig_dir,
                    table_dir=table_dir,
                )

            summary_path = table_dir / "benchmark_ei_summary_main.tex"
            interval_path = table_dir / "benchmark_ei_interval_main.tex"
            overview_path = table_dir / "benchmark_ei_overview_main.tex"
            self.assertTrue(summary_path.exists())
            self.assertTrue(interval_path.exists())
            self.assertTrue(overview_path.exists())

            summary_tex = summary_path.read_text()
            self.assertIn(r"\label{tab:benchmark-ei-summary-main}", summary_tex)
            self.assertIn(r"\multicolumn{1}{c}{Fréchet max-AR}", summary_tex)
            self.assertIn(r"true $\xi$", summary_tex)
            self.assertIn(r"\shortstack[l]{Northrop-", summary_tex)

            interval_tex = interval_path.read_text()
            self.assertIn(r"\label{tab:benchmark-ei-interval-main}", interval_tex)
            self.assertIn("EI interval sharpness-versus-calibration summary", interval_tex)
            self.assertIn(r"median\_interval\_score", interval_tex)

            overview_tex = overview_path.read_text()
            self.assertIn(r"\label{tab:benchmark-ei-overview-main}", overview_tex)
            self.assertIn("Appendix full EI benchmark overview", overview_tex)

            core_panels.assert_called_once_with(
                internal_summary,
                title="",
                file_path=fig_dir / "benchmark_ei_summary.pdf",
                save=True,
            )
            target_panels.assert_called_once_with(
                internal_summary,
                external_summary,
                title="",
                file_path=fig_dir / "benchmark_ei_targets.pdf",
                save=True,
            )
            sharpness_scatter.assert_called_once_with(
                internal_summary,
                external_summary,
                file_path=fig_dir / "benchmark_ei_interval_sharpness.pdf",
                save=True,
            )
            overview_panels.assert_called_once_with(
                internal_summary,
                external_summary,
                file_path=fig_dir / "benchmark_ei_overview.pdf",
                save=True,
            )
            shrinkage_panels.assert_not_called()


if __name__ == "__main__":
    unittest.main()
