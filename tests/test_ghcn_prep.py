from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from scripts.data_prep.ghcn import (
    download_ghcn_station,
    ghcn_station_data_needs_refresh,
    prepare_hot_dry_series,
    prepare_precipitation_series,
    read_ghcn_station_csv,
)


class GhcnPrepTests(unittest.TestCase):
    @mock.patch("scripts.data_prep.ghcn.urlretrieve")
    def test_download_is_truncated_at_analysis_cutoff(self, mocked_retrieve) -> None:
        def write_download(_url: str, target: Path) -> None:
            pd.DataFrame(
                [
                    ["USW00000001", 20251231, "PRCP", 10, "", "", "", ""],
                    ["USW00000001", 20260101, "PRCP", 20, "", "", "", ""],
                ]
            ).to_csv(target, header=False, index=False, compression="gzip")

        mocked_retrieve.side_effect = write_download
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "station.csv.gz"

            download_ghcn_station("USW00000001.csv.gz", path)

            downloaded = read_ghcn_station_csv(path)
        self.assertEqual(downloaded["date"].max(), pd.Timestamp("2025-12-31"))

    def test_precipitation_uses_covered_season_suffix_and_keeps_dry_days(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "houston.csv"
            missing = pd.Timestamp("2023-08-15")
            lines = [
                f"USW00012918,{date:%Y%m%d},PRCP,{0 if date.day % 2 else 10},,,,\n"
                for date in pd.date_range("2022-01-01", "2025-12-31", freq="D")
                if date != missing
            ]
            path.write_text("".join(lines), encoding="utf-8")

            prepared = prepare_precipitation_series(path)

        expected = pd.DatetimeIndex(
            [
                date
                for date in pd.date_range("2022-06-01", "2025-11-30", freq="D")
                if date.month in {6, 7, 8, 9, 10, 11} and date != missing
            ]
        )
        self.assertTrue(prepared.series.index.equals(expected))
        self.assertEqual(len(prepared.series), 4 * 183 - 1)
        self.assertTrue((prepared.series == 0).any())
        self.assertEqual(prepared.annual_maxima.index.tolist(), [2022, 2023, 2024, 2025])
        self.assertEqual(prepared.metadata["season_start_year"], 2022)
        self.assertEqual(prepared.metadata["maxima_period"], "season")
        self.assertEqual(prepared.metadata["coverage_threshold"], 0.97)
        self.assertEqual(prepared.metadata["coverage_expected_days"], 4 * 183)
        self.assertEqual(prepared.metadata["coverage_valid_days"], 4 * 183 - 1)
        self.assertAlmostEqual(prepared.metadata["coverage_fraction"], (4 * 183 - 1) / (4 * 183))

    def test_hot_dry_gates_final_rolling_positions_and_keeps_zero_days(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phoenix.csv"
            missing_precipitation = pd.Timestamp("2023-10-30")
            lines: list[str] = []
            for date in pd.date_range("2022-01-01", "2025-12-31", freq="D"):
                tmax = 280 + (date.dayofyear % 30) + 2 * (date.year - 2022)
                precipitation = 10 if date.dayofyear % 9 == 0 else 0
                lines.append(f"USW00023183,{date:%Y%m%d},TMAX,{tmax},,,,\n")
                if date != missing_precipitation:
                    lines.append(f"USW00023183,{date:%Y%m%d},PRCP,{precipitation},,,,\n")
            path.write_text("".join(lines), encoding="utf-8")

            prepared = prepare_hot_dry_series(path)

        expected = pd.DatetimeIndex(
            [
                date
                for date in pd.date_range("2022-04-01", "2025-10-31", freq="D")
                if date.month in {4, 5, 6, 7, 8, 9, 10}
                and date not in {missing_precipitation, pd.Timestamp("2023-10-31")}
            ]
        )
        self.assertTrue(prepared.series.index.equals(expected))
        self.assertEqual(len(prepared.series), 4 * 214 - 2)
        self.assertTrue((prepared.series == 0).any())
        self.assertEqual(prepared.annual_maxima.index.tolist(), [2022, 2023, 2024, 2025])
        self.assertEqual(prepared.metadata["season_start_year"], 2022)
        self.assertEqual(prepared.metadata["rolling_min_periods"], 30)
        self.assertEqual(prepared.metadata["coverage_threshold"], 0.97)
        self.assertEqual(prepared.metadata["coverage_expected_days"], 4 * 214)
        self.assertEqual(prepared.metadata["coverage_valid_days"], 4 * 214 - 2)
        self.assertAlmostEqual(prepared.metadata["coverage_fraction"], (4 * 214 - 2) / (4 * 214))

    def test_precipitation_starts_after_a_season_below_the_coverage_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "houston.csv"
            missing = set(pd.date_range("2023-06-01", periods=6, freq="D"))
            lines = [
                f"USW00012918,{date:%Y%m%d},PRCP,10,,,,\n"
                for date in pd.date_range("2022-01-01", "2025-12-31", freq="D")
                if date not in missing
            ]
            path.write_text("".join(lines), encoding="utf-8")

            prepared = prepare_precipitation_series(path)

        self.assertEqual(prepared.metadata["season_start_year"], 2024)
        self.assertEqual(prepared.annual_maxima.index.tolist(), [2024, 2025])

    def test_read_ghcn_station_csv_only_materializes_needed_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "station.csv"
            path.write_text(
                "USW00000001,20200101,PRCP,12,,,X,\nUSW00000001,20200102,TMAX,301,,,X,\n",
                encoding="utf-8",
            )

            frame = read_ghcn_station_csv(path)

            self.assertEqual(
                list(frame.columns), ["station_id", "date", "element", "value", "qflag"]
            )
            self.assertEqual(str(frame["element"].dtype), "category")
            self.assertTrue(frame["qflag"].isna().all())

    def test_ghcn_station_data_needs_refresh_flags_invalid_and_accepts_valid_extract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            broken = root / "broken.csv"
            broken.write_text("", encoding="utf-8")
            self.assertTrue(ghcn_station_data_needs_refresh(broken, required_elements=("PRCP",)))

            valid = root / "valid.csv"
            rows: list[str] = []
            for year in range(2010, 2016):
                rows.append(f"USW00000001,{year}0101,PRCP,12,,,,\n")
                rows.append(f"USW00000001,{year}0701,TMAX,301,,,,\n")
            valid.write_text("".join(rows), encoding="utf-8")
            self.assertFalse(
                ghcn_station_data_needs_refresh(
                    valid,
                    required_elements=("PRCP", "TMAX"),
                    expected_station_id="USW00000001",
                    min_rows=4,
                    min_span_days=365,
                )
            )
            self.assertTrue(
                ghcn_station_data_needs_refresh(
                    valid,
                    required_elements=("PRCP", "TMAX"),
                    expected_station_id="USW00000002",
                    min_rows=4,
                    min_span_days=365,
                )
            )

    def test_prepare_hot_dry_series_retains_complete_terminal_year(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phoenix.csv"
            lines: list[str] = []
            for date in pd.date_range("2024-01-01", "2025-12-31", freq="D"):
                tmax = 300
                if date.strftime("%Y-%m-%d") == "2025-07-15":
                    tmax = 450
                lines.append(f"USW00023183,{date:%Y%m%d},TMAX,{tmax},,,,\n")
                lines.append(f"USW00023183,{date:%Y%m%d},PRCP,10,,,,\n")
            path.write_text("".join(lines), encoding="utf-8")

            prepared = prepare_hot_dry_series(path)

            self.assertFalse(prepared.series.empty)
            self.assertEqual(int(prepared.series.index.year.max()), 2025)

    def test_analysis_cutoff_makes_live_station_updates_inert(self) -> None:
        def write_station(path: Path, end_date: str) -> None:
            lines: list[str] = []
            for date in pd.date_range("2022-01-01", end_date, freq="D"):
                year_offset = date.year - 2022
                tmax = 250 + 5 * year_offset + date.dayofyear % 20
                precipitation = 50 - 5 * year_offset + date.dayofyear % 10
                lines.append(f"USW00000001,{date:%Y%m%d},TMAX,{tmax},,,,\n")
                lines.append(f"USW00000001,{date:%Y%m%d},PRCP,{precipitation},,,,\n")
            path.write_text("".join(lines), encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            through_2025 = root / "through_2025.csv"
            through_2026 = root / "through_2026.csv"
            write_station(through_2025, "2025-12-31")
            write_station(through_2026, "2026-12-31")

            for prepare in (prepare_precipitation_series, prepare_hot_dry_series):
                expected = prepare(through_2025)
                actual = prepare(through_2026)

                pd.testing.assert_series_equal(actual.series, expected.series)
                self.assertEqual(actual.metadata["analysis_end_date"], "2025-12-31")


if __name__ == "__main__":
    unittest.main()
