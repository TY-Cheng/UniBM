from __future__ import annotations
# ruff: noqa: E402

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

try:
    from . import _path_setup as test_paths
except ImportError:  # pragma: no cover
    import _path_setup as test_paths

test_paths.ensure_repo_import_paths()

from manuscript.artifact_manifest import build_paper_subset_manifest


class ManuscriptArtifactManifestTests(unittest.TestCase):
    def test_manifest_records_analysis_cutoff_and_code_revision(self) -> None:
        code_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manuscript_root = root / "UniBM_manuscript"
            data_root = root / "data"
            data_root.mkdir()
            with patch.dict(
                os.environ,
                {"DIR_DATA": str(data_root), "DIR_MANUSCRIPT": str(manuscript_root)},
                clear=False,
            ):
                manifest_path = build_paper_subset_manifest(code_root)

            payload = json.loads(manifest_path.read_text())
            expected_revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=code_root, text=True
            ).strip()
            self.assertEqual(payload["analysis_end_date"], "2025-12-31")
            self.assertEqual(payload["code_commit"], expected_revision)
            self.assertIsInstance(payload["code_worktree_dirty"], bool)


if __name__ == "__main__":
    unittest.main()
