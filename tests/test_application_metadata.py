from __future__ import annotations
# ruff: noqa: E402

import tempfile
import unittest
from pathlib import Path

import pandas as pd

try:
    from . import _path_setup as test_paths
except ImportError:  # pragma: no cover
    import _path_setup as test_paths

test_paths.ensure_repo_import_paths()

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
