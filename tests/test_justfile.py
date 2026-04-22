from __future__ import annotations

import shutil
import subprocess
import unittest

try:
    from . import _path_setup as test_paths
except ImportError:  # pragma: no cover
    import _path_setup as test_paths

ROOT = test_paths.ROOT
JUSTFILE = ROOT / "justfile"


class JustfileTests(unittest.TestCase):
    def test_justfile_declares_main_repo_targets(self) -> None:
        text = JUSTFILE.read_text()
        self.assertIn("\nfull workers=", text)
        self.assertIn("\ndocs:", text)
        self.assertIn("\ndocs-serve:", text)
        self.assertIn("\nbenchmark workers=", text)
        self.assertIn("\napplication workers=", text)
        self.assertIn("\nvignette:", text)

    def test_just_list_mentions_main_repo_targets_when_available(self) -> None:
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
        self.assertIn("docs", result.stdout)
        self.assertIn("docs-serve", result.stdout)
        self.assertIn("benchmark", result.stdout)
        self.assertIn("application", result.stdout)
        self.assertIn("vignette", result.stdout)


if __name__ == "__main__":
    unittest.main()
