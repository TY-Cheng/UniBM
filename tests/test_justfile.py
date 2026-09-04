from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JUSTFILE = ROOT / "justfile"


class JustfileTests(unittest.TestCase):
    def test_justfile_declares_main_repo_targets(self) -> None:
        text = JUSTFILE.read_text()
        self.assertIn('export UV_LOCKED := "1"', text)
        self.assertIn("\nfull workers=", text)
        self.assertIn("\ndocs:", text)
        self.assertNotIn("\ndocs-serve:", text)
        self.assertIn("just _docs-build", text)
        self.assertIn("\nbenchmark workers=", text)
        self.assertIn("\napplication workers=", text)
        self.assertIn("\ndata screening_bootstrap=", text)
        self.assertIn("\nrefresh-data:", text)
        self.assertNotIn("\nvignette:", text)
        self.assertNotIn("\nweather-vignette:", text)
        self.assertIn("\ncheck:", text)
        self.assertIn("\ncheck-full:", text)
        self.assertNotIn("\ntest:", text)
        self.assertNotIn("\nverify:", text)
        self.assertIn("\nformat:", text)
        self.assertIn("uv run pytest --testmon -n auto", text)
        self.assertIn("uv run pytest -q tests/test_justfile.py", text)
        self.assertIn("uv run pytest -n auto --cov=src/unibm", text)
        self.assertIn("--cov-fail-under=89", text)
        self.assertIn("just --fmt --check", text)
        self.assertIn("uv run ruff format --check .", text)
        self.assertIn("uv run ruff check .", text)
        self.assertNotIn("notebooks/", text)
        self.assertNotIn("jupytext", text)
        self.assertNotIn("nbconvert", text)
        self.assertNotIn("\n    just format\n", text)
        self.assertNotIn("rm -rf site", text)

    def test_justfile_uses_repo_data_and_optional_environment_overrides(self) -> None:
        text = JUSTFILE.read_text()
        self.assertNotIn("_require-external-data-dir", text)
        self.assertNotIn("_require-external-uv-env", text)
        self.assertNotIn("DIR_DATA", text)
        self.assertNotIn("DIR_WORK", text)
        self.assertNotIn("DATA" + "_DIR", text)
        self.assertIn("python scripts/application/refresh_data.py", text)

    def test_justfile_guards_match_recipe_side_effects(self) -> None:
        text = JUSTFILE.read_text()
        self.assertIn("\n_require-workflow-env: _require-manuscript-dir", text)
        self.assertIn("\n_require-manuscript-dir:", text)
        self.assertIn('\ndata screening_bootstrap="20":', text)
        self.assertIn("\n_docs-build:", text)
        self.assertIn("\nformat:", text)
        self.assertIn("\nclean-generated: _require-manuscript-dir", text)

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
        self.assertIn("refresh-data", result.stdout)
        self.assertNotIn("vignette", result.stdout)
        self.assertIn("\n    check\n", result.stdout)
        self.assertIn("\n    check-full\n", result.stdout)
        self.assertNotIn("\n    test", result.stdout)
        self.assertNotIn("\n    verify", result.stdout)
        self.assertIn("format", result.stdout)


if __name__ == "__main__":
    unittest.main()
