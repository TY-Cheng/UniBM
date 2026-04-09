from __future__ import annotations
# ruff: noqa: E402

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import resolve_repo_dirs


class ConfigPathTests(unittest.TestCase):
    def test_sibling_repo_layout_resolves_code_and_manuscript_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir).resolve()
            code_root = workspace / "UniBM"
            manuscript_root = workspace / "UniBM_manuscript"
            (code_root / "scripts").mkdir(parents=True, exist_ok=True)
            (code_root / "pyproject.toml").write_text("")
            manuscript_root.mkdir(parents=True, exist_ok=True)

            with patch.dict(os.environ, {}, clear=True):
                dirs = resolve_repo_dirs(workspace)

            self.assertEqual(dirs["DIR_WORK"], code_root)
            self.assertEqual(dirs["DIR_MANUSCRIPT"], manuscript_root)
            self.assertEqual(dirs["DIR_MANUSCRIPT_FIGURE"], manuscript_root / "Figure")
            self.assertEqual(dirs["DIR_WORKSPACE"], workspace)

    def test_explicit_temp_root_without_sibling_defaults_to_nested_manuscript(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            with patch.dict(os.environ, {}, clear=True):
                dirs = resolve_repo_dirs(root)

            self.assertEqual(dirs["DIR_WORK"], root)
            self.assertEqual(dirs["DIR_MANUSCRIPT"], root / "UniBM_manuscript")
            self.assertEqual(dirs["DIR_MANUSCRIPT_TABLE"], root / "UniBM_manuscript" / "Table")


if __name__ == "__main__":
    unittest.main()
