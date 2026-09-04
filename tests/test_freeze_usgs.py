from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from application import freeze_usgs


class FreezeUsgsTests(unittest.TestCase):
    def test_candidate_missing_analysis_cutoff_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)
            index = pd.date_range("2019-01-01", "2024-12-31", freq="D")
            pd.DataFrame(
                {
                    "date": index,
                    "discharge_cfs": 1.0,
                    "site_no": "02366500",
                    "station_name": "Station",
                }
            ).to_csv(raw / "usgs_02366500.csv.gz", index=False, compression="gzip")

            with self.assertRaisesRegex(ValueError, "analysis cutoff"):
                freeze_usgs._candidate_screening_rows(
                    state_code="FL",
                    candidates=[{"site_no": "02366500", "station_name": "Station"}],
                    raw_dir=raw,
                )

    def test_freeze_rejects_state_with_no_recommended_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            metadata = root / "metadata"
            output = root / "out"
            metadata.mkdir()
            (metadata / "usgs_candidate_sites.json").write_text(
                json.dumps({"FL": [{"site_no": "02366500", "station_name": "Station"}]}),
                encoding="utf-8",
            )
            row = {
                "state_code": "FL",
                "site_no": "02366500",
                "station_name": "Station",
                "recommended": False,
                "supports_frechet_working_model": True,
                "plateau_points": 5,
                "n_years": 40.0,
                "xi_lower": 0.1,
            }
            dirs = {
                "DIR_DATA_RAW_USGS": raw,
                "DIR_DATA_METADATA_APPLICATION": metadata,
                "DIR_OUT_APPLICATIONS": output,
            }
            with (
                mock.patch.object(freeze_usgs, "resolve_repo_dirs", return_value=dirs),
                mock.patch.object(freeze_usgs, "ensure_application_metadata"),
                mock.patch.object(freeze_usgs, "_candidate_screening_rows", return_value=[row]),
            ):
                with self.assertRaisesRegex(ValueError, "no recommended USGS candidate"):
                    freeze_usgs.freeze_usgs_station_selection(root)
