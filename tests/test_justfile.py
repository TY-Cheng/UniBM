from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JUSTFILE = ROOT / "justfile"


class JustfileTests(unittest.TestCase):
    def test_justfile_declares_manuscript_target(self) -> None:
        self.assertIn("\nmanuscript workers=", JUSTFILE.read_text())

    def test_just_list_mentions_manuscript_target_when_available(self) -> None:
        just_exe = shutil.which("just")
        if just_exe is None:
            self.skipTest("`just` is not available in the test environment.")
        result = subprocess.run(
            [just_exe, "--list"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("manuscript", result.stdout)


if __name__ == "__main__":
    unittest.main()
