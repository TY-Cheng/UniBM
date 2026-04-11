from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.data_prep.fema import nfip_claims_needs_refresh, prepare_nfip_claim_series


class FemaPrepTests(unittest.TestCase):
    def test_nfip_series_roles_are_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            claims_path = root / "claims.csv.gz"
            cpi_path = root / "cpi.csv"
            pd.DataFrame(
                {
                    "state": ["TX", "TX", "TX", "TX", "FL"],
                    "dateOfLoss": [
                        "2024-01-01",
                        "2024-01-01",
                        "2024-01-03",
                        "2024-01-05",
                        "2024-01-01",
                    ],
                    "amountPaidOnBuildingClaim": [100.0, 50.0, 0.0, 200.0, 999.0],
                }
            ).to_csv(claims_path, index=False, compression="gzip")
            pd.DataFrame({"year": [2024, 2025], "cpi_2025_base": [97.4, 100.0]}).to_csv(
                cpi_path, index=False
            )

            prepared = prepare_nfip_claim_series(
                claims_path,
                state_code="TX",
                cpi_table_path=cpi_path,
            )

            display = prepared["display"].series
            evi = prepared["evi"].series
            ei = prepared["ei"].series
            self.assertEqual(display.index.min().strftime("%Y-%m-%d"), "2024-01-01")
            self.assertEqual(display.index.max().strftime("%Y-%m-%d"), "2024-01-05")
            self.assertEqual(len(display), 5)
            self.assertEqual(len(ei), 5)
            self.assertEqual(len(evi), 2)
            self.assertAlmostEqual(
                float(display.loc["2024-01-01"]), 150.0 * 100.0 / 97.4, places=6
            )
            self.assertAlmostEqual(float(display.loc["2024-01-02"]), 0.0, places=6)
            self.assertTrue((evi > 0).all())

    def test_nfip_claims_needs_refresh_flags_invalid_and_accepts_valid_extract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            broken = root / "broken.csv.gz"
            broken.write_bytes(b"")
            self.assertTrue(nfip_claims_needs_refresh(broken, state_code="TX"))

            valid = root / "valid.csv.gz"
            pd.DataFrame(
                {
                    "state": ["TX", "TX"],
                    "dateOfLoss": ["2024-01-01", "2024-01-02"],
                    "amountPaidOnBuildingClaim": [100.0, 0.0],
                }
            ).to_csv(valid, index=False, compression="gzip")
            self.assertFalse(nfip_claims_needs_refresh(valid, state_code="TX"))


if __name__ == "__main__":
    unittest.main()
