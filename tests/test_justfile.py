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
        self.assertNotIn("\ndocs-serve:", text)
        self.assertIn("just _docs-build", text)
        self.assertIn("\nbenchmark workers=", text)
        self.assertIn("\napplication workers=", text)
        self.assertIn("\ndata screening_bootstrap=", text)
        self.assertIn("\nvignette:", text)
        self.assertIn("\ntest:", text)
        self.assertIn("\nformat:", text)
        self.assertIn("uv run ruff format --check .", text)
        self.assertIn("uv run ruff check .", text)

    def test_justfile_rejects_repo_local_data_paths(self) -> None:
        text = JUSTFILE.read_text()
        self.assertIn("_require-external-data-dir", text)
        self.assertIn("DIR_DATA", text)
        self.assertNotIn("DATA" + "_DIR", text)
        self.assertIn("${repo_dir}/data", text)

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
        self.assertNotIn("docs-serve", result.stdout)
        self.assertIn("benchmark", result.stdout)
        self.assertIn("application", result.stdout)
        self.assertIn("data", result.stdout)
        self.assertIn("vignette", result.stdout)
        self.assertIn("test", result.stdout)
        self.assertIn("format", result.stdout)


if __name__ == "__main__":
    unittest.main()
