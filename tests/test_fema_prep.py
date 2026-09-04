from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from unittest import mock

import pandas as pd

from scripts.data_prep import fema
from scripts.data_prep.fema import (
    _download_nfip_claims_window,
    nfip_claims_needs_refresh,
    prepare_nfip_claim_series,
)


class FemaPrepTests(unittest.TestCase):
    def test_nfip_pagination_uses_unique_tie_breaker(self) -> None:
        records = [
            {
                "id": value,
                "state": "TX",
                "dateOfLoss": "2024-01-01",
                "amountPaidOnBuildingClaim": 1.0,
            }
            for value in ("a", "b", "c")
        ]
        with (
            mock.patch.object(
                fema,
                "_openfema_json_with_retries",
                side_effect=[{"FimaNfipClaims": records[:2]}, {"FimaNfipClaims": records[2:]}],
            ) as fetch,
            mock.patch.object(fema.time, "sleep"),
        ):
            frame = _download_nfip_claims_window(
                "TX",
                start_date="2024-01-01",
                end_date="2024-12-31",
                page_size=2,
            )

        queries = [parse_qs(urlparse(call.args[0]).query) for call in fetch.call_args_list]
        self.assertEqual(frame["id"].tolist(), ["a", "b", "c"])
        self.assertEqual([query["$skip"] for query in queries], [["0"], ["2"]])
        self.assertTrue(all(query["$orderby"] == ["dateOfLoss,id"] for query in queries))
        self.assertTrue(all(query["$select"][0].startswith("id,") for query in queries))

    def test_nfip_deflation_uses_loss_month_cpi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            claims_path = root / "claims.csv.gz"
            cpi_path = root / "cpi.csv"
            pd.DataFrame(
                {
                    "state": ["TX", "TX"],
                    "dateOfLoss": ["2024-01-01", "2024-02-01"],
                    "amountPaidOnBuildingClaim": [100.0, 100.0],
                }
            ).to_csv(claims_path, index=False, compression="gzip")
            pd.DataFrame(
                {
                    "date": ["2024-01-01", "2024-02-01"],
                    "cpi_u": [300.0, 310.0],
                    "is_imputed": [False, False],
                }
            ).to_csv(cpi_path, index=False)

            prepared = prepare_nfip_claim_series(
                claims_path,
                state_code="TX",
                cpi_table_path=cpi_path,
            )

        display = prepared["display"].series
        self.assertAlmostEqual(float(display.loc["2024-01-01"]), 100.0 * 321.943 / 300.0)
        self.assertAlmostEqual(float(display.loc["2024-02-01"]), 100.0 * 321.943 / 310.0)
        self.assertEqual(prepared["display"].metadata["cpi_frequency"], "monthly")
        self.assertEqual(prepared["display"].metadata["cpi_base_2025_annual_average"], 321.943)

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
            pd.DataFrame({"date": ["2024-01-01"], "cpi_u": [313.5], "is_imputed": [False]}).to_csv(
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
            self.assertEqual(display.index.max().strftime("%Y-%m-%d"), "2025-12-31")
            self.assertEqual(len(display), 731)
            self.assertEqual(len(ei), 731)
            self.assertEqual(len(evi), 2)
            self.assertAlmostEqual(
                float(display.loc["2024-01-01"]), 150.0 * 321.943 / 313.5, places=6
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

    def test_analysis_cutoff_excludes_later_claims_and_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            claims_path = root / "claims.csv.gz"
            cpi_path = root / "cpi.csv"
            pd.DataFrame(
                {
                    "state": ["TX", "TX"],
                    "dateOfLoss": ["2025-12-31", "2026-01-01"],
                    "amountPaidOnBuildingClaim": [100.0, 200.0],
                }
            ).to_csv(claims_path, index=False, compression="gzip")
            pd.DataFrame(
                {"date": ["2025-12-01"], "cpi_u": [324.054], "is_imputed": [False]}
            ).to_csv(cpi_path, index=False)

            prepared = prepare_nfip_claim_series(
                claims_path,
                state_code="TX",
                cpi_table_path=cpi_path,
            )

            for series in prepared.values():
                self.assertEqual(series.series.index.max(), pd.Timestamp("2025-12-31"))
                self.assertEqual(series.metadata["analysis_end_date"], "2025-12-31")

    def test_nfip_preparation_rejects_missing_claim_month_cpi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            claims_path = root / "claims.csv.gz"
            cpi_path = root / "cpi.csv"
            pd.DataFrame(
                {
                    "state": ["TX"],
                    "dateOfLoss": ["2024-02-01"],
                    "amountPaidOnBuildingClaim": [100.0],
                }
            ).to_csv(claims_path, index=False, compression="gzip")
            pd.DataFrame({"date": ["2024-01-01"], "cpi_u": [300.0], "is_imputed": [False]}).to_csv(
                cpi_path, index=False
            )

            with self.assertRaisesRegex(ValueError, "missing claim months: 2024-02"):
                prepare_nfip_claim_series(
                    claims_path,
                    state_code="TX",
                    cpi_table_path=cpi_path,
                )


if __name__ == "__main__":
    unittest.main()
