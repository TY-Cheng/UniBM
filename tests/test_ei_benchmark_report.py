from __future__ import annotations
# ruff: noqa: E402

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
for path in (SCRIPTS_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark.design import default_ei_simulation_configs
from benchmark.ei_report import (
    EI_SHRINKAGE_GRID,
    EI_SHRINKAGE_METHODS,
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


if __name__ == "__main__":
    unittest.main()
