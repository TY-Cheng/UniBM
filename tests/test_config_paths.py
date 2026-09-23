from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config

ROOT = Path(__file__).resolve().parents[1]
resolve_repo_dirs = config.resolve_repo_dirs


class ConfigPathTests(unittest.TestCase):
    def test_reports_default_locally_without_discovering_external_projects(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            code = root / "UniBM"
            (code / "scripts").mkdir(parents=True)
            (code / "pyproject.toml").touch()
            report = root / "external_project"
            report.mkdir()
            for value in (None, "", "  "):
                with (
                    self.subTest(value=value),
                    patch.dict(
                        os.environ,
                        {} if value is None else {"UNIBM_REPORT_DIR": value},
                        clear=True,
                    ),
                ):
                    dirs = resolve_repo_dirs(code)
                    self.assertEqual(dirs["DIR_REPORT"], code / "out" / "reports")
                    self.assertEqual(list(report.iterdir()), [])
                    self.assertFalse((code / "out").exists())

    def test_explicit_report_path_is_relative_to_code_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            code = Path(tmpdir).resolve() / "code"
            code.mkdir()
            with patch.dict(os.environ, {"UNIBM_REPORT_DIR": "../deliverables"}, clear=True):
                dirs = resolve_repo_dirs(code)
            self.assertEqual(dirs["DIR_REPORT"], code.parent / "deliverables")

    def test_invalid_report_target_fails_instead_of_falling_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            target = root / "file"
            target.touch()
            for value in (target, target / "reports"):
                with (
                    self.subTest(value=value),
                    patch.dict(os.environ, {"UNIBM_REPORT_DIR": str(value)}, clear=True),
                ):
                    with self.assertRaisesRegex(ValueError, "UNIBM_REPORT_DIR"):
                        resolve_repo_dirs(root)

    def test_default_code_root_ignores_legacy_environment_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_root = Path(tmpdir).resolve()
            (legacy_root / "scripts").mkdir()
            (legacy_root / "pyproject.toml").write_text("")
            try:
                with patch.dict(os.environ, {"DIR_WORK": str(legacy_root)}, clear=False):
                    dirs = importlib.reload(config).resolve_repo_dirs()
            finally:
                importlib.reload(config)

        self.assertEqual(dirs["DIR_WORK"], ROOT)

    def test_report_subdirectory_cannot_redirect_exports_through_a_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir).resolve()
            report = root / "report"
            external = root / "external"
            report.mkdir()
            external.mkdir()
            (report / "Figure").symlink_to(external, target_is_directory=True)
            with patch.dict(os.environ, {"UNIBM_REPORT_DIR": str(report)}, clear=True):
                with self.assertRaisesRegex(ValueError, "symlink"):
                    resolve_repo_dirs(root)
            self.assertEqual(list(external.iterdir()), [])

    def test_repo_data_root_ignores_legacy_environment_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir).resolve()
            code_root = workspace / "UniBM"
            artifact_root = workspace / "report_artifacts"
            legacy_data_root = workspace / "external_data"
            (code_root / "scripts").mkdir(parents=True, exist_ok=True)
            (code_root / "pyproject.toml").write_text("")
            artifact_root.mkdir(parents=True, exist_ok=True)
            legacy_data_root.mkdir(parents=True, exist_ok=True)

            with patch.dict(
                os.environ,
                {
                    "UNIBM_REPORT_DIR": str(artifact_root),
                    "DIR_DATA": str(legacy_data_root),
                },
                clear=True,
            ):
                dirs = resolve_repo_dirs(code_root)

            self.assertEqual(dirs["DIR_WORK"], code_root)
            self.assertEqual(dirs["DIR_DATA"], code_root / "data")
            self.assertEqual(dirs["DIR_DATA_RAW"], code_root / "data" / "raw")
            self.assertEqual(dirs["DIR_DATA_DERIVED"], code_root / "data" / "derived")
            self.assertEqual(dirs["DIR_DATA_METADATA"], code_root / "data" / "metadata")
            self.assertEqual(dirs["DIR_REPORT"], artifact_root)
            self.assertEqual(dirs["DIR_REPORT_FIGURE"], artifact_root / "Figure")
            self.assertEqual(dirs["DIR_WORKSPACE"], workspace)

    def test_workspace_guessing_finds_code_and_repo_data_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir).resolve()
            code_root = workspace / "UniBM"
            (code_root / "scripts").mkdir(parents=True, exist_ok=True)
            (code_root / "pyproject.toml").write_text("")
            with patch.dict(os.environ, {}, clear=True):
                dirs = resolve_repo_dirs(workspace)

            self.assertEqual(dirs["DIR_WORK"], code_root)
            self.assertEqual(dirs["DIR_DATA"], code_root / "data")
            self.assertEqual(dirs["DIR_WORKSPACE"], code_root)
