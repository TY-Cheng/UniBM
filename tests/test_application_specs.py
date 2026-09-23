from __future__ import annotations

import unittest

from application.specs import APPLICATIONS, CLIMATE_APPLICATIONS, REPORT_APPLICATIONS


class ApplicationSpecsTests(unittest.TestCase):
    def test_registry_separates_report_and_climate_cases(self) -> None:
        self.assertEqual(
            {spec.key for spec in REPORT_APPLICATIONS},
            {"tx_streamflow", "fl_streamflow", "tx_nfip_claims", "fl_nfip_claims"},
        )
        self.assertEqual(
            {spec.key for spec in CLIMATE_APPLICATIONS},
            {"houston_hobby_precipitation", "phoenix_hot_dry_severity"},
        )
        self.assertEqual(set(APPLICATIONS), set(REPORT_APPLICATIONS + CLIMATE_APPLICATIONS))
