from __future__ import annotations
# ruff: noqa: E402

import unittest

try:
    from . import _path_setup as test_paths
except ImportError:  # pragma: no cover
    import _path_setup as test_paths

test_paths.ensure_repo_import_paths()

import application.build as application


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
            "plot_application_composite",
            "seasonal_monthly_pit_unit_frechet",
        }
        self.assertTrue(expected.issubset(set(application.__all__)))
        for name in expected:
            self.assertTrue(hasattr(application, name))


if __name__ == "__main__":
    unittest.main()
