from __future__ import annotations

import sys
import unittest
from pathlib import Path

from scripts.workflows.import_bootstrap import (
    bootstrap_notebook_scripts_dir,
    ensure_scripts_on_path_from_entry,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"


class ImportBootstrapTests(unittest.TestCase):
    def test_ensure_scripts_on_path_from_entry_returns_repo_scripts_dir(self) -> None:
        original_path = list(sys.path)
        try:
            if str(SCRIPTS_DIR) in sys.path:
                sys.path.remove(str(SCRIPTS_DIR))
            resolved = ensure_scripts_on_path_from_entry(
                SCRIPTS_DIR / "workflows" / "application.py"
            )
            self.assertEqual(resolved, SCRIPTS_DIR.resolve())
            self.assertEqual(sys.path[0], str(SCRIPTS_DIR.resolve()))
        finally:
            sys.path[:] = original_path

    def test_bootstrap_notebook_scripts_dir_finds_repo_from_root(self) -> None:
        original_path = list(sys.path)
        try:
            if str(SCRIPTS_DIR) in sys.path:
                sys.path.remove(str(SCRIPTS_DIR))
            resolved = bootstrap_notebook_scripts_dir(ROOT)
            self.assertEqual(resolved, SCRIPTS_DIR.resolve())
            self.assertEqual(sys.path[0], str(SCRIPTS_DIR.resolve()))
        finally:
            sys.path[:] = original_path


if __name__ == "__main__":
    unittest.main()
