from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from application.refresh_data import _assert_data_tree_clean


class RefreshDataTests(unittest.TestCase):
    @mock.patch("application.refresh_data.subprocess.run")
    def test_refresh_refuses_dirty_data_tree(self, mocked_run) -> None:
        mocked_run.return_value.stdout = " M data/raw/example.csv.gz\n"
        with self.assertRaisesRegex(RuntimeError, "dirty"):
            _assert_data_tree_clean(Path("/repo"))
