from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from scripts.data_prep import cpi


class CpiPrepTests(unittest.TestCase):
    def test_download_imputes_only_missing_october_2025_and_marks_it(self) -> None:
        records = [
            {
                "year": "2025",
                "period": f"M{month:02d}",
                "value": "-" if month == 10 else str(320.0 + month),
                "footnotes": [],
            }
            for month in range(1, 13)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "cpi_u_monthly.csv"
            with mock.patch.object(cpi, "_fetch_bls_window", return_value=records):
                cpi.download_monthly_cpi(output, start_year=2025, end_year=2025)
            frame = pd.read_csv(output)

        october = frame.loc[frame["date"] == "2025-10-01"].iloc[0]
        self.assertAlmostEqual(float(october["cpi_u"]), (329.0 * 331.0) ** 0.5, places=6)
        self.assertTrue(bool(october["is_imputed"]))
        self.assertIn("geometric mean", str(october["note"]))
        self.assertEqual(len(frame), 12)


if __name__ == "__main__":
    unittest.main()
