from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from application import inputs
from application.inputs import build_application_inputs, ensure_ghcn_raw_data
from application.specs import CLIMATE_APPLICATIONS


class ApplicationInputsTests(unittest.TestCase):
    def test_climate_only_build_does_not_require_usgs_metadata(self) -> None:
        raw_paths = {
            "houston_hobby_precipitation": Path("houston.csv.gz"),
            "phoenix_hot_dry_severity": Path("phoenix.csv.gz"),
        }
        dirs = {
            "DIR_DATA_METADATA_APPLICATION": Path("missing-metadata"),
            "DIR_DATA_RAW_GHCN": Path("ghcn"),
        }
        prepared = mock.sentinel.prepared
        with (
            mock.patch.object(inputs, "ensure_application_metadata") as metadata,
            mock.patch.object(inputs, "ensure_ghcn_raw_data", return_value=raw_paths),
            mock.patch.object(inputs, "prepare_precipitation_series", return_value=prepared),
            mock.patch.object(inputs, "prepare_hot_dry_series", return_value=prepared),
        ):
            result = build_application_inputs(dirs, specs=CLIMATE_APPLICATIONS)

        metadata.assert_not_called()
        self.assertEqual(set(result), set(raw_paths))

    def test_missing_canonical_raw_input_fails_offline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "refresh-data"):
                ensure_ghcn_raw_data(Path(tmp))
