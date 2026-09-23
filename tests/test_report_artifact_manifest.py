from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from reports.artifact_manifest import build_report_subset_manifest


class ReportArtifactManifestTests(unittest.TestCase):
    def test_index_records_its_creation_state_without_generating_artifacts(self) -> None:
        code_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_root = root / "UniBM_report"
            with patch.dict(
                os.environ,
                {"UNIBM_REPORT_DIR": str(report_root)},
                clear=False,
            ):
                manifest_path = build_report_subset_manifest(code_root)

            payload = json.loads(manifest_path.read_text())
            expected_revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=code_root, text=True
            ).strip()
            self.assertEqual(payload["analysis_end_date"], "2025-12-31")
            self.assertEqual(payload["manifest_code_commit"], expected_revision)
            self.assertIsInstance(payload["manifest_code_worktree_dirty"], bool)
            self.assertEqual(manifest_path.parent, report_root.resolve())
            self.assertEqual(manifest_path.name, "report_subset_manifest.json")
            self.assertEqual(payload["report_scope"], "curated four-case report subset")
            self.assertEqual(
                {entry["placement"] for entry in payload["entries"]},
                {"primary", "primary-supporting", "supplementary"},
            )
            self.assertTrue(payload["report_root"].endswith(report_root.name))
            self.assertEqual(
                set(payload),
                {
                    "report_scope",
                    "analysis_end_date",
                    "manifest_code_commit",
                    "manifest_code_worktree_dirty",
                    "workspace_root",
                    "code_repo_root",
                    "report_root",
                    "entries",
                },
            )
            self.assertEqual([path.name for path in report_root.iterdir()], [manifest_path.name])
            tables = [entry for entry in payload["entries"] if entry["kind"] == "table"]
            self.assertEqual(len(tables), 7)
            self.assertTrue(all(not entry["label"].endswith("-main") for entry in tables))
            self.assertTrue(
                all(
                    not path.endswith("_main.tex")
                    for entry in tables
                    for path in entry["artifact_paths"]
                )
            )
