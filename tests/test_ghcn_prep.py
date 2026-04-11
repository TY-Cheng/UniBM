from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.data_prep.ghcn import (
    ghcn_station_data_needs_refresh,
    prepare_hot_dry_series,
    read_ghcn_station_csv,
)


class GhcnPrepTests(unittest.TestCase):
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
                    min_rows=4,
                    min_span_days=365,
                )
            )

    def test_prepare_hot_dry_series_retains_complete_terminal_year(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phoenix.csv"
            lines: list[str] = []
            for date in pd.date_range("2020-01-01", "2021-12-31", freq="D"):
                tmax = 300
                if date.strftime("%Y-%m-%d") == "2021-07-15":
                    tmax = 450
                lines.append(f"USW00023183,{date:%Y%m%d},TMAX,{tmax},,,,\n")
                lines.append(f"USW00023183,{date:%Y%m%d},PRCP,10,,,,\n")
            path.write_text("".join(lines), encoding="utf-8")

            prepared = prepare_hot_dry_series(path)

            self.assertFalse(prepared.series.empty)
            self.assertEqual(int(prepared.series.index.year.max()), 2021)


if __name__ == "__main__":
    unittest.main()
