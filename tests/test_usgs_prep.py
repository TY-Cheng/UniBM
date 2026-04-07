from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.data_prep.usgs import (
    _extract_usgs_daily_series,
    _normalize_site_no,
    prepare_usgs_streamflow_series,
    usgs_daily_discharge_needs_refresh,
)


class UsgsPrepTests(unittest.TestCase):
    def test_extract_usgs_daily_series_prefers_mean_stat_code(self) -> None:
        payload = {
            "value": {
                "timeSeries": [
                    {
                        "name": "USGS:02236000:00060:00001",
                        "sourceInfo": {"siteName": "Station"},
                        "values": [{"value": [{"dateTime": "2024-01-01", "value": "99"}]}],
                    },
                    {
                        "name": "USGS:02236000:00060:00003",
                        "sourceInfo": {"siteName": "Station"},
                        "values": [{"value": [{"dateTime": "2024-01-01", "value": "10"}]}],
                    },
                ]
            }
        }

        series, station_name = _extract_usgs_daily_series(payload)

        self.assertEqual(station_name, "Station")
        self.assertEqual(float(series.iloc[0]), 10.0)

    def test_normalize_site_no_preserves_leading_zeroes(self) -> None:
        self.assertEqual(_normalize_site_no("02236000"), "02236000")
        self.assertEqual(_normalize_site_no("02236000.0"), "02236000")

    def test_prepare_usgs_streamflow_series_preserves_site_match_with_leading_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usgs_02236000.csv.gz"
            index = pd.date_range("2020-01-01", periods=365, freq="D")
            pd.DataFrame(
                {
                    "date": index.strftime("%Y-%m-%d"),
                    "discharge_cfs": [10.0 + (i % 5) for i in range(index.size)],
                    "site_no": ["02236000"] * index.size,
                    "station_name": ["Station"] * index.size,
                }
            ).to_csv(path, index=False, compression="gzip")

            prepared = prepare_usgs_streamflow_series(
                path,
                state_code="FL",
                site_no="02236000",
                station_name="Station",
            )

            self.assertEqual(prepared.metadata["site_no"], "02236000")
            self.assertEqual(len(prepared.series), 365)

    def test_usgs_daily_discharge_needs_refresh_flags_single_day_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usgs_short.csv.gz"
            pd.DataFrame(
                {
                    "date": ["2026-04-05"],
                    "discharge_cfs": [1870.0],
                    "site_no": ["02236000"],
                    "station_name": ["Station"],
                }
            ).to_csv(path, index=False, compression="gzip")

            self.assertTrue(usgs_daily_discharge_needs_refresh(path))


if __name__ == "__main__":
    unittest.main()
