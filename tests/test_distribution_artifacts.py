from __future__ import annotations

import importlib.util
import subprocess
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_UNIBM = ROOT / "src" / "unibm"
EXPECTED_PACKAGE_FILES = {
    path.relative_to(ROOT / "src").as_posix()
    for path in SRC_UNIBM.rglob("*")
    if path.is_file() and (path.suffix == ".py" or path.name == "py.typed")
}
_ABOUT_SPEC = importlib.util.spec_from_file_location(
    "_unibm_about",
    ROOT / "src" / "unibm" / "__about__.py",
)
assert _ABOUT_SPEC is not None
assert _ABOUT_SPEC.loader is not None
_ABOUT_MODULE = importlib.util.module_from_spec(_ABOUT_SPEC)
_ABOUT_SPEC.loader.exec_module(_ABOUT_MODULE)
PACKAGE_VERSION = _ABOUT_MODULE.__version__


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
        cls._wheel = next(build_dir.glob(f"unibm-{PACKAGE_VERSION}-*.whl"))
        cls._sdist = next(build_dir.glob(f"unibm-{PACKAGE_VERSION}.tar.gz"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()
        super().tearDownClass()

    def test_wheel_metadata_and_typed_marker_are_present(self) -> None:
        with zipfile.ZipFile(self._wheel) as wheel:
            names = wheel.namelist()
            package_files = {name for name in names if name.startswith("unibm/")}
            self.assertEqual(package_files, EXPECTED_PACKAGE_FILES)
            self.assertFalse(any(name.startswith("scripts/") for name in names))
            self.assertFalse(any(name.startswith("data/") for name in names))
            metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
            metadata = wheel.read(metadata_name).decode("utf-8")

        self.assertIn(f"Version: {PACKAGE_VERSION}", metadata)
        self.assertIn(
            "Project-URL: Documentation, https://ty-cheng.github.io/UniBM/",
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

        prefix = f"unibm-{PACKAGE_VERSION}/"
        package_prefix = f"{prefix}src/"
        package_files = {
            name.removeprefix(package_prefix)
            for name in names
            if name.startswith(f"{package_prefix}unibm/")
        }
        self.assertEqual(package_files, EXPECTED_PACKAGE_FILES)
        self.assertIn(f"{prefix}README.md", names)
        self.assertIn(f"{prefix}pyproject.toml", names)
        self.assertFalse(any(name.startswith(f"{prefix}scripts/application/") for name in names))
        self.assertFalse(any(name.startswith(f"{prefix}scripts/benchmark/") for name in names))
        self.assertFalse(any(name.startswith(f"{prefix}scripts/data_prep/") for name in names))
        self.assertFalse(any(name.startswith(f"{prefix}scripts/shared/") for name in names))
        self.assertFalse(any(name.startswith(f"{prefix}docs/") for name in names))
        self.assertFalse(any(name.startswith(f"{prefix}tests/") for name in names))
        self.assertFalse(any(name.startswith(f"{prefix}data/") for name in names))
