from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.data_prep._io import write_csv_gz_atomic


class DataIoTests(unittest.TestCase):
    def test_gzip_output_is_byte_deterministic(self) -> None:
        frame = pd.DataFrame({"date": ["2025-01-01"], "value": [1.25]})
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.csv.gz"
            second = Path(tmp) / "second.csv.gz"

            write_csv_gz_atomic(frame, first)
            write_csv_gz_atomic(frame, second)

            self.assertEqual(first.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()
