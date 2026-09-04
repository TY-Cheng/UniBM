from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from application.metadata import ensure_application_metadata


class ApplicationMetadataTests(unittest.TestCase):
    def test_ensure_application_metadata_validates_tracked_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metadata_dir = Path(tmp) / "metadata" / "application"
            metadata_dir.mkdir(parents=True)
            candidates = {
                "TX": [{"site_no": "08066500", "station_name": "Texas station"}],
                "FL": [{"site_no": "02366500", "station_name": "Florida station"}],
            }
            frozen = {
                state: {**records[0], "state_code": state} for state, records in candidates.items()
            }
            (metadata_dir / "usgs_candidate_sites.json").write_text(json.dumps(candidates))
            (metadata_dir / "usgs_frozen_sites.json").write_text(json.dumps(frozen))

            outputs = ensure_application_metadata(metadata_dir)

            self.assertTrue(outputs["candidate_sites"].exists())
            self.assertTrue(outputs["frozen_sites"].exists())

    def test_swapped_frozen_site_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metadata_dir = Path(tmp)
            candidates = {
                "TX": [{"site_no": "08066500", "station_name": "Texas station"}],
                "FL": [{"site_no": "02366500", "station_name": "Florida station"}],
            }
            frozen = {
                "TX": {
                    "site_no": "02366500",
                    "station_name": "Florida station",
                    "state_code": "TX",
                },
                "FL": {**candidates["FL"][0], "state_code": "FL"},
            }
            (metadata_dir / "usgs_candidate_sites.json").write_text(json.dumps(candidates))
            (metadata_dir / "usgs_frozen_sites.json").write_text(json.dumps(frozen))

            with self.assertRaisesRegex(ValueError, "candidate"):
                ensure_application_metadata(metadata_dir)

    def test_missing_application_metadata_fails_offline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "Restore the tracked file"):
                ensure_application_metadata(Path(tmp))
