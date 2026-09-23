from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
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
            # The default build creates the wheel from the sdist, not the checkout.
            ["uv", "build", "--out-dir", str(build_dir)],
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
            self.assertTrue(
                all(
                    name.startswith(("unibm/", f"unibm-{PACKAGE_VERSION}.dist-info/"))
                    for name in names
                )
            )
            metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
            metadata = wheel.read(metadata_name).decode("utf-8")

        self.assertIn(f"Version: {PACKAGE_VERSION}", metadata)
        self.assertIn("Author-email: Tuoyuan Cheng <tuoyuan.cheng@nus.edu.sg>", metadata)
        self.assertIn("Maintainer-email: Tuoyuan Cheng <tuoyuan.cheng@nus.edu.sg>", metadata)
        self.assertIn(
            "Project-URL: Documentation, https://ty-cheng.github.io/UniBM/",
            metadata,
        )
        self.assertIn("Project-URL: Issues, https://github.com/TY-Cheng/UniBM/issues", metadata)
        self.assertIn(
            "Project-URL: Changelog, https://github.com/TY-Cheng/UniBM/releases",
            metadata,
        )

    def test_distributions_install_and_run_without_the_checkout(self) -> None:
        """Exercise both install routes using only declared runtime dependencies."""
        for artifact in (self._wheel, self._sdist):
            with self.subTest(artifact=artifact.name):
                result = subprocess.run(
                    [
                        "uv",
                        "run",
                        "--isolated",
                        "--no-project",
                        "--python",
                        sys.executable,
                        "--with",
                        str(artifact),
                        "python",
                        "-I",
                        str(ROOT / "tests" / "test_unibm_package_smoke.py"),
                    ],
                    cwd=self._tmpdir.name,
                    env={**os.environ, "MPLBACKEND": "Agg"},
                    capture_output=True,
                    text=True,
                    timeout=180,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

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
        # Hatchling preserves .gitignore as part of the source build configuration.
        build_files = {".gitignore", "LICENSE", "README.md", "pyproject.toml", "PKG-INFO"}
        self.assertEqual(
            {name.removeprefix(prefix) for name in names if not name.startswith(package_prefix)},
            build_files,
        )
