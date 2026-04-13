from __future__ import annotations
# ruff: noqa: E402

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
for path in (SCRIPTS_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import resolve_repo_dirs


class ConfigPathTests(unittest.TestCase):
    def test_explicit_environment_paths_override_workspace_guessing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir).resolve()
            code_root = workspace / "UniBM"
            artifact_root = workspace / "paper_artifacts"
            (code_root / "scripts").mkdir(parents=True, exist_ok=True)
            (code_root / "pyproject.toml").write_text("")
            artifact_root.mkdir(parents=True, exist_ok=True)

            with patch.dict(
                os.environ,
                {"DIR_MANUSCRIPT": str(artifact_root)},
                clear=True,
            ):
                dirs = resolve_repo_dirs(code_root)

            self.assertEqual(dirs["DIR_WORK"], code_root)
            self.assertEqual(dirs["DIR_MANUSCRIPT"], artifact_root)
            self.assertEqual(dirs["DIR_MANUSCRIPT_FIGURE"], artifact_root / "Figure")
            self.assertEqual(dirs["DIR_WORKSPACE"], workspace)

    def test_workspace_guessing_still_finds_code_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir).resolve()
            code_root = workspace / "UniBM"
            (code_root / "scripts").mkdir(parents=True, exist_ok=True)
            (code_root / "pyproject.toml").write_text("")
            with patch.dict(os.environ, {}, clear=True):
                dirs = resolve_repo_dirs(workspace)

            self.assertEqual(dirs["DIR_WORK"], code_root)
            self.assertEqual(dirs["DIR_WORKSPACE"], workspace)


if __name__ == "__main__":
    unittest.main()
