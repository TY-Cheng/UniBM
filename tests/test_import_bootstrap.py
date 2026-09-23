from __future__ import annotations

import sys
import unittest
from pathlib import Path

from shared.import_bootstrap import ensure_scripts_on_path_from_entry

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"


class ImportBootstrapTests(unittest.TestCase):
    def test_ensure_scripts_on_path_from_entry_returns_repo_scripts_dir(self) -> None:
        original_path = list(sys.path)
        try:
            repo_paths = {str(SCRIPTS_DIR), str(SRC_DIR)}
            sys.path[:] = [path for path in sys.path if path not in repo_paths]
            resolved = ensure_scripts_on_path_from_entry(SCRIPTS_DIR / "application" / "build.py")
            self.assertEqual(resolved, SCRIPTS_DIR.resolve())
            self.assertEqual(sys.path[0], str(SRC_DIR.resolve()))
            self.assertEqual(sys.path[1], str(SCRIPTS_DIR.resolve()))
        finally:
            sys.path[:] = original_path


def test_benchmark_entries_bootstrap_paths_in_fresh_interpreters():
    import os
    import subprocess

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    for name in ["evi_benchmark", "evi_report", "ei_report"]:
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import runpy,sys; runpy.run_path(sys.argv[1], run_name='import_probe')",
                str(SCRIPTS_DIR / "benchmark" / f"{name}.py"),
            ],
            cwd=SCRIPTS_DIR / "benchmark",
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
