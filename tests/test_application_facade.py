from __future__ import annotations
# ruff: noqa: E402

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from workflows import application


class ApplicationFacadeTests(unittest.TestCase):
    def test_facade_exports_expected_symbols(self) -> None:
        expected = {
            "ApplicationSpec",
            "ApplicationPreparedInputs",
            "ApplicationBundle",
            "build_application_inputs",
            "build_application_bundle",
            "build_application_bundles",
            "build_application_outputs",
            "load_usgs_frozen_sites",
        }
        self.assertTrue(expected.issubset(set(application.__all__)))
        for name in expected:
            self.assertTrue(hasattr(application, name))


if __name__ == "__main__":
    unittest.main()
