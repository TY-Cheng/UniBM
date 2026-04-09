from __future__ import annotations
# ruff: noqa: E402

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import resolve_repo_dirs
from benchmark.design import default_ei_simulation_configs
from benchmark.ei_benchmark import EiBenchmarkOutputs
from benchmark.ei_report import (
    EI_SHRINKAGE_GRID,
    EI_SHRINKAGE_METHODS,
    build_ei_benchmark_manuscript_outputs,
    build_ei_shrinkage_sensitivity_summary,
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

    def test_build_ei_benchmark_manuscript_outputs_returns_shrinkage_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            out_dir = root / "out" / "benchmark"
            summary_path = out_dir / "ei_summary.csv"
            external_summary_path = out_dir / "ei_external_summary.csv"
            fake_outputs = EiBenchmarkOutputs(
                detail_path=out_dir / "ei_detail.csv",
                summary_path=summary_path,
                external_detail_path=out_dir / "ei_external_detail.csv",
                external_summary_path=external_summary_path,
                summary=pd.DataFrame({"method": ["bb_sliding_fgls"]}),
                external_summary=pd.DataFrame({"method": ["k_gaps"]}),
            )
            shrinkage_path = out_dir / "benchmark_ei_shrinkage_sensitivity.csv"
            with (
                patch(
                    "benchmark.ei_benchmark.load_or_materialize_ei_benchmark_outputs",
                    return_value=fake_outputs,
                ),
                patch(
                    "benchmark.ei_report.build_ei_shrinkage_sensitivity_summary",
                    return_value=(pd.DataFrame(), shrinkage_path),
                ),
                patch("benchmark.ei_report.write_ei_benchmark_manuscript_artifacts"),
            ):
                outputs = build_ei_benchmark_manuscript_outputs(root)
            manuscript_figure_dir = resolve_repo_dirs(root)["DIR_MANUSCRIPT_FIGURE"]
            self.assertEqual(outputs["benchmark_ei_shrinkage_sensitivity_data"], shrinkage_path)
            self.assertEqual(
                outputs["benchmark_ei_shrinkage_sensitivity_figure"],
                manuscript_figure_dir / "benchmark_ei_shrinkage_sensitivity.pdf",
            )


if __name__ == "__main__":
    unittest.main()
