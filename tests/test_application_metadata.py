from __future__ import annotations
# ruff: noqa: E402

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from application.metadata import ensure_application_metadata


class ApplicationMetadataTests(unittest.TestCase):
    def test_ensure_application_metadata_materializes_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metadata_dir = Path(tmp) / "metadata" / "application"
            outputs = ensure_application_metadata(metadata_dir)

            self.assertTrue(outputs["candidate_sites"].exists())
            self.assertTrue(outputs["frozen_sites"].exists())
            self.assertTrue(outputs["cpi_2025_base"].exists())

            cpi = pd.read_csv(outputs["cpi_2025_base"])
            self.assertIn("year", cpi.columns)
            self.assertIn("cpi_2025_base", cpi.columns)
            self.assertEqual(float(cpi.loc[cpi["year"] == 2025, "cpi_2025_base"].iloc[0]), 100.0)


if __name__ == "__main__":
    unittest.main()
