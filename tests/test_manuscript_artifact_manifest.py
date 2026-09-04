from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from manuscript.artifact_manifest import build_paper_subset_manifest


class ManuscriptArtifactManifestTests(unittest.TestCase):
    def test_manifest_records_analysis_cutoff_and_code_revision(self) -> None:
        code_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manuscript_root = root / "UniBM_manuscript"
            with patch.dict(
                os.environ,
                {"DIR_MANUSCRIPT": str(manuscript_root)},
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
