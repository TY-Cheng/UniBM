from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from application.normalization import prepare_normalized_climate

from scripts.data_prep.ghcn import (
    download_ghcn_station,
    ghcn_station_data_needs_refresh,
    read_ghcn_station_csv,
)


class GhcnPrepTests(unittest.TestCase):
    def test_normalized_climate_keeps_full_calendar_zeros_and_respects_quality_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "station.csv"
            dates = pd.date_range("2016-01-01", "2025-12-31")
            gap = pd.Timestamp("2017-03-01")
            rows = []
            for date in dates:
                # The flagged rainfall day breaks the daily series and every
                # Phoenix 30-day precipitation window containing that day.
                rain = 0 if date.day % 2 else 10 + 10 * (date.year - 2016)
                quality = "X" if date == gap else ""
                rows.extend(
                    [
                        f"USW00000001,{date:%Y%m%d},PRCP,{rain},,{quality},,\n",
                        f"USW00000001,{date:%Y%m%d},TMAX,{250 + 10 * (date.year - 2016)},,,,\n",
                    ]
                )
            path.write_text("".join(rows))
            for key, expected_start in (
                ("houston", gap + pd.Timedelta(days=31)),
                ("phoenix", gap + pd.Timedelta(days=60)),
            ):
                result = prepare_normalized_climate(path, key)
                self.assertEqual(result.series.index.min(), expected_start)
                self.assertTrue(
                    result.series.index.equals(pd.date_range(expected_start, "2025-12-31"))
                )
                if key == "houston":
                    self.assertTrue((result.series == 0).any())
                self.assertFalse(result.series.isna().any())
                self.assertEqual(result.metadata["months"], list(range(1, 13)))
                self.assertEqual(result.metadata["warmup_days"], 30)

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

    def test_analysis_cutoff_makes_live_station_updates_inert(self) -> None:
        def write_station(path: Path, end_date: str) -> None:
            lines: list[str] = []
            for date in pd.date_range("2016-01-01", end_date, freq="D"):
                year_offset = date.year - 2016
                tmax = 250 + 5 * year_offset + date.dayofyear % 20
                precipitation = 20 + 5 * year_offset + date.dayofyear % 10
                lines.append(f"USW00000001,{date:%Y%m%d},TMAX,{tmax},,,,\n")
                lines.append(f"USW00000001,{date:%Y%m%d},PRCP,{precipitation},,,,\n")
            path.write_text("".join(lines), encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            through_2025 = root / "through_2025.csv"
            through_2026 = root / "through_2026.csv"
            write_station(through_2025, "2025-12-31")
            write_station(through_2026, "2026-12-31")

            for key in ("houston", "phoenix"):
                expected = prepare_normalized_climate(through_2025, key)
                actual = prepare_normalized_climate(through_2026, key)

                pd.testing.assert_series_equal(actual.series, expected.series)
                self.assertEqual(actual.series.index.max(), pd.Timestamp("2025-12-31"))


if __name__ == "__main__":
    unittest.main()
