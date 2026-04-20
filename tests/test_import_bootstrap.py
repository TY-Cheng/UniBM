from __future__ import annotations
# ruff: noqa: E402

import sys
import unittest

try:
    from . import _path_setup as test_paths
except ImportError:  # pragma: no cover
    import _path_setup as test_paths

test_paths.ensure_repo_import_paths()
ROOT = test_paths.ROOT
SRC_DIR = test_paths.SRC_DIR
SCRIPTS_DIR = test_paths.SCRIPTS_DIR

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
