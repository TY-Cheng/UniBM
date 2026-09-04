from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.data_prep.usgs import (
    _extract_usgs_daily_series,
    _normalize_site_no,
    download_usgs_daily_discharge,
    prepare_usgs_streamflow_series,
    usgs_daily_discharge_needs_refresh,
)


class UsgsPrepTests(unittest.TestCase):
    def test_preparation_uses_continuous_suffix_and_complete_water_years(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usgs_02236000.csv.gz"
            index = pd.date_range("2023-01-01", "2025-12-31", freq="D").difference(
                [pd.Timestamp("2023-08-31")]
            )
            pd.DataFrame(
                {
                    "date": index.strftime("%Y-%m-%d"),
                    "discharge_cfs": range(1, index.size + 1),
                    "site_no": ["02236000"] * index.size,
                    "station_name": ["Station"] * index.size,
                }
            ).to_csv(path, index=False, compression="gzip")

            prepared = prepare_usgs_streamflow_series(
                path,
                state_code="FL",
                site_no="02236000",
            )

        expected_index = pd.date_range("2023-09-01", "2025-12-31", freq="D")
        self.assertTrue(prepared.series.index.equals(expected_index))
        self.assertEqual(prepared.annual_maxima.index.tolist(), [2024, 2025])
        self.assertEqual(prepared.annual_maxima.index.name, "water_year")
        self.assertEqual(prepared.metadata["continuity_start"], "2023-09-01")
        self.assertEqual(prepared.metadata["maxima_period"], "water_year")

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
            index = pd.date_range("2025-01-01", "2025-12-31", freq="D")
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

    def test_analysis_cutoff_applies_to_download_and_preparation(self) -> None:
        self.assertEqual(
            inspect.signature(download_usgs_daily_discharge).parameters["end_date"].default,
            "2025-12-31",
        )
        with self.assertRaisesRegex(ValueError, "end_date must be explicit"):
            download_usgs_daily_discharge("02236000", "unused.csv.gz", end_date=None)  # type: ignore[arg-type]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "usgs_02236000.csv.gz"
            index = pd.date_range("2025-01-01", "2026-12-31", freq="D")
            pd.DataFrame(
                {
                    "date": index.strftime("%Y-%m-%d"),
                    "discharge_cfs": [10.0] * index.size,
                    "site_no": ["02236000"] * index.size,
                    "station_name": ["Station"] * index.size,
                }
            ).to_csv(path, index=False, compression="gzip")

            prepared = prepare_usgs_streamflow_series(
                path,
                state_code="FL",
                site_no="02236000",
            )

            self.assertEqual(prepared.series.index.max(), pd.Timestamp("2025-12-31"))
            self.assertEqual(prepared.metadata["analysis_end_date"], "2025-12-31")


if __name__ == "__main__":
    unittest.main()
