from __future__ import annotations
# ruff: noqa: E402

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
for path in (SCRIPTS_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from shared.import_bootstrap import (
    bootstrap_notebook_scripts_dir,
    ensure_scripts_on_path_from_entry,
)


class ImportBootstrapTests(unittest.TestCase):
    def test_ensure_scripts_on_path_from_entry_returns_repo_scripts_dir(self) -> None:
        original_path = list(sys.path)
        try:
            for path in (SCRIPTS_DIR, SRC_DIR):
                if str(path) in sys.path:
                    sys.path.remove(str(path))
            resolved = ensure_scripts_on_path_from_entry(SCRIPTS_DIR / "application" / "build.py")
            self.assertEqual(resolved, SCRIPTS_DIR.resolve())
            self.assertEqual(sys.path[0], str(SRC_DIR.resolve()))
            self.assertEqual(sys.path[1], str(SCRIPTS_DIR.resolve()))
        finally:
            sys.path[:] = original_path

    def test_bootstrap_notebook_scripts_dir_finds_repo_from_root(self) -> None:
        original_path = list(sys.path)
        try:
            for path in (SCRIPTS_DIR, SRC_DIR):
                if str(path) in sys.path:
                    sys.path.remove(str(path))
            resolved = bootstrap_notebook_scripts_dir(ROOT)
            self.assertEqual(resolved, SCRIPTS_DIR.resolve())
            self.assertEqual(sys.path[0], str(SRC_DIR.resolve()))
            self.assertEqual(sys.path[1], str(SCRIPTS_DIR.resolve()))
        finally:
            sys.path[:] = original_path


if __name__ == "__main__":
    unittest.main()
