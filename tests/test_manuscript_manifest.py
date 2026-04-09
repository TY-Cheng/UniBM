from __future__ import annotations
# ruff: noqa: E402

import json
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

from manuscript.artifact_manifest import build_paper_subset_manifest


class ManuscriptManifestTests(unittest.TestCase):
    def test_manifest_contains_target_figures_and_appendix_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            (root / "UniBM_manuscript").mkdir(parents=True, exist_ok=True)
            with patch.dict(os.environ, {}, clear=True):
                manifest_path = build_paper_subset_manifest(root)

            payload = json.loads(manifest_path.read_text())
            self.assertEqual(payload["paper_scope"], "curated four-case manuscript subset")
            self.assertEqual(payload["code_repo_root"], ".")
            self.assertEqual(payload["manuscript_repo_root"], "UniBM_manuscript")
            labels = {entry["label"] for entry in payload["entries"]}
            self.assertIn("fig:benchmark-evi-targets", labels)
            self.assertIn("fig:benchmark-ei-targets", labels)
            self.assertIn("tab:application-case-audit-main", labels)
            self.assertIn("tab:application-selection-sensitivity-main", labels)
            self.assertIn("tab:application-summary-main", labels)


if __name__ == "__main__":
    unittest.main()
