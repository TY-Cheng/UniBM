from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.data_prep.ghcn import read_ghcn_station_csv


class GhcnPrepTests(unittest.TestCase):
    def test_read_ghcn_station_csv_only_materializes_needed_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "station.csv"
            path.write_text(
                "USW00000001,20200101,PRCP,12,,,X,\n"
                "USW00000001,20200102,TMAX,301,,,X,\n",
                encoding="utf-8",
            )

            frame = read_ghcn_station_csv(path)

            self.assertEqual(list(frame.columns), ["station_id", "date", "element", "value", "qflag"])
            self.assertEqual(str(frame["element"].dtype), "category")
            self.assertTrue(frame["qflag"].isna().all())


if __name__ == "__main__":
    unittest.main()
