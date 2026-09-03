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
        self.assertIn('export UV_LOCKED := "1"', text)
        self.assertIn("\nfull workers=", text)
        self.assertIn("\ndocs:", text)
        self.assertNotIn("\ndocs-serve:", text)
        self.assertIn("just _docs-build", text)
        self.assertIn("\nbenchmark workers=", text)
        self.assertIn("\napplication workers=", text)
        self.assertIn("\ndata screening_bootstrap=", text)
        self.assertIn("\nvignette:", text)
        self.assertIn("\ncheck: _require-external-uv-env", text)
        self.assertIn("\ncheck-full: _require-external-uv-env", text)
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
        self.assertIn("uv run ruff format notebooks/vignette.py", text)
        self.assertNotIn("\n    just format\n", text)
        self.assertNotIn("rm -rf site", text)

    def test_justfile_rejects_repo_local_data_paths(self) -> None:
        text = JUSTFILE.read_text()
        self.assertIn("_require-external-data-dir", text)
        self.assertIn("DIR_DATA", text)
        self.assertNotIn("DATA" + "_DIR", text)
        self.assertIn("${repo_dir}/data", text)

    def test_justfile_guards_match_recipe_side_effects(self) -> None:
        text = JUSTFILE.read_text()
        self.assertIn('env_abs="${env_path:A}"', text)
        self.assertIn(
            "\n_require-data-env: _require-external-uv-env _require-external-data-dir", text
        )
        self.assertIn(
            "\n_require-workflow-env: _require-data-env _require-manuscript-dir",
            text,
        )
        self.assertIn("\n_require-manuscript-dir:", text)
        self.assertIn('\ndata screening_bootstrap="20": _require-data-env', text)
        self.assertIn("\n_docs-build: _require-external-uv-env", text)
        self.assertIn("\nformat: _require-external-uv-env", text)
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
        self.assertIn("vignette", result.stdout)
        self.assertIn("\n    check\n", result.stdout)
        self.assertIn("\n    check-full\n", result.stdout)
        self.assertNotIn("\n    test", result.stdout)
        self.assertNotIn("\n    verify", result.stdout)
        self.assertIn("format", result.stdout)


if __name__ == "__main__":
    unittest.main()
