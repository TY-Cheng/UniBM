from __future__ import annotations

import subprocess
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DistributionArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._tmpdir = tempfile.TemporaryDirectory()
        build_dir = Path(cls._tmpdir.name)
        subprocess.run(
            ["uv", "build", "--wheel", "--sdist", "--out-dir", str(build_dir)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        cls._wheel = next(build_dir.glob("unibm-0.1.0-*.whl"))
        cls._sdist = next(build_dir.glob("unibm-0.1.0.tar.gz"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()
        super().tearDownClass()

    def test_wheel_metadata_and_typed_marker_are_present(self) -> None:
        with zipfile.ZipFile(self._wheel) as wheel:
            names = wheel.namelist()
            self.assertIn("unibm/__init__.py", names)
            self.assertIn("unibm/py.typed", names)
            self.assertFalse(any(name.startswith("scripts/") for name in names))
            metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
            metadata = wheel.read(metadata_name).decode("utf-8")

        self.assertIn("Version: 0.1.0", metadata)
        self.assertIn(
            "Project-URL: Documentation, https://github.com/TY-Cheng/UniBM/tree/main/docs",
            metadata,
        )
        self.assertIn("Project-URL: Issues, https://github.com/TY-Cheng/UniBM/issues", metadata)
        self.assertIn(
            "Project-URL: Changelog, https://github.com/TY-Cheng/UniBM/releases",
            metadata,
        )

    def test_sdist_contains_library_sources_but_not_repo_workflow_directories(self) -> None:
        with tarfile.open(self._sdist, "r:gz") as sdist:
            names = sdist.getnames()

        prefix = "unibm-0.1.0/"
        self.assertIn(f"{prefix}src/unibm/__init__.py", names)
        self.assertIn(f"{prefix}src/unibm/py.typed", names)
        self.assertIn(f"{prefix}README.md", names)
        self.assertIn(f"{prefix}pyproject.toml", names)
        self.assertFalse(any(name.startswith(f"{prefix}scripts/application/") for name in names))
        self.assertFalse(any(name.startswith(f"{prefix}scripts/benchmark/") for name in names))
        self.assertFalse(any(name.startswith(f"{prefix}scripts/data_prep/") for name in names))
        self.assertFalse(any(name.startswith(f"{prefix}scripts/shared/") for name in names))
        self.assertFalse(any(name.startswith(f"{prefix}docs/") for name in names))
        self.assertFalse(any(name.startswith(f"{prefix}tests/") for name in names))
