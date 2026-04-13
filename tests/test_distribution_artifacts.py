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
            self.assertIn("unibm/_block_grid.py", names)
            self.assertIn("unibm/_bootstrap_sampling.py", names)
            self.assertIn("unibm/cdf.py", names)
            self.assertIn("unibm/_numeric.py", names)
            self.assertIn("unibm/_runtime.py", names)
            self.assertIn("unibm/_validation.py", names)
            self.assertIn("unibm/_window_ops.py", names)
            self.assertIn("unibm/evi/__init__.py", names)
            self.assertIn("unibm/evi/_regression.py", names)
            self.assertIn("unibm/evi/blocks.py", names)
            self.assertIn("unibm/evi/bootstrap.py", names)
            self.assertIn("unibm/evi/design.py", names)
            self.assertIn("unibm/evi/estimation.py", names)
            self.assertIn("unibm/evi/plotting.py", names)
            self.assertIn("unibm/evi/selection.py", names)
            self.assertIn("unibm/evi/spectrum.py", names)
            self.assertIn("unibm/evi/summaries.py", names)
            self.assertIn("unibm/evi/tail.py", names)
            self.assertIn("unibm/evi/targets.py", names)
            self.assertIn("unibm/ei/__init__.py", names)
            self.assertIn("unibm/ei/bootstrap.py", names)
            self.assertIn("unibm/ei/models.py", names)
            self.assertIn("unibm/ei/paths.py", names)
            self.assertIn("unibm/ei/plotting.py", names)
            self.assertIn("unibm/ei/preparation.py", names)
            self.assertIn("unibm/ei/selection.py", names)
            self.assertIn("unibm/ei/threshold.py", names)
            self.assertIn("unibm/ei/_likelihood.py", names)
            self.assertIn("unibm/ei/_stats.py", names)
            self.assertIn("unibm/ei/_validation.py", names)
            self.assertIn("unibm/ei/bm.py", names)
            self.assertIn("unibm/py.typed", names)
            self.assertNotIn("unibm/_cdf.py", names)
            self.assertNotIn("unibm/_diagnostic_models.py", names)
            self.assertNotIn("unibm/diagnostics/__init__.py", names)
            self.assertNotIn("unibm/diagnostics/cdf.py", names)
            self.assertNotIn("unibm/diagnostics.py", names)
            self.assertNotIn("unibm/diagnostics/models.py", names)
            self.assertNotIn("unibm/diagnostics/reciprocal.py", names)
            self.assertNotIn("unibm/diagnostics/targets.py", names)
            self.assertNotIn("unibm/evi/_bootstrap_eval.py", names)
            self.assertNotIn("unibm/evi/_summaries.py", names)
            self.assertNotIn("unibm/evi/_tail.py", names)
            self.assertNotIn("unibm/evi/baselines.py", names)
            self.assertNotIn("unibm/ei/_internal.py", names)
            self.assertNotIn("unibm/evi/core.py", names)
            self.assertNotIn("unibm/evi/external.py", names)
            self.assertNotIn("unibm/ei/native.py", names)
            self.assertNotIn("unibm/core.py", names)
            self.assertNotIn("unibm/bootstrap.py", names)
            self.assertNotIn("unibm/external.py", names)
            self.assertNotIn("unibm/models.py", names)
            self.assertNotIn("unibm/extremal_index.py", names)
            self.assertNotIn("unibm/ei/_bootstrap.py", names)
            self.assertNotIn("unibm/ei/_paths.py", names)
            self.assertNotIn("unibm/ei/_native.py", names)
            self.assertNotIn("unibm/ei/_threshold.py", names)
            self.assertNotIn("unibm/ei/_shared.py", names)
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
        self.assertIn(f"{prefix}src/unibm/_block_grid.py", names)
        self.assertIn(f"{prefix}src/unibm/_bootstrap_sampling.py", names)
        self.assertIn(f"{prefix}src/unibm/cdf.py", names)
        self.assertIn(f"{prefix}src/unibm/_numeric.py", names)
        self.assertIn(f"{prefix}src/unibm/_runtime.py", names)
        self.assertIn(f"{prefix}src/unibm/_validation.py", names)
        self.assertIn(f"{prefix}src/unibm/_window_ops.py", names)
        self.assertIn(f"{prefix}src/unibm/evi/__init__.py", names)
        self.assertIn(f"{prefix}src/unibm/evi/_regression.py", names)
        self.assertIn(f"{prefix}src/unibm/evi/blocks.py", names)
        self.assertIn(f"{prefix}src/unibm/evi/bootstrap.py", names)
        self.assertIn(f"{prefix}src/unibm/evi/design.py", names)
        self.assertIn(f"{prefix}src/unibm/evi/estimation.py", names)
        self.assertIn(f"{prefix}src/unibm/evi/plotting.py", names)
        self.assertIn(f"{prefix}src/unibm/evi/selection.py", names)
        self.assertIn(f"{prefix}src/unibm/evi/spectrum.py", names)
        self.assertIn(f"{prefix}src/unibm/evi/summaries.py", names)
        self.assertIn(f"{prefix}src/unibm/evi/tail.py", names)
        self.assertIn(f"{prefix}src/unibm/evi/targets.py", names)
        self.assertIn(f"{prefix}src/unibm/ei/__init__.py", names)
        self.assertIn(f"{prefix}src/unibm/ei/bootstrap.py", names)
        self.assertIn(f"{prefix}src/unibm/ei/models.py", names)
        self.assertIn(f"{prefix}src/unibm/ei/paths.py", names)
        self.assertIn(f"{prefix}src/unibm/ei/plotting.py", names)
        self.assertIn(f"{prefix}src/unibm/ei/preparation.py", names)
        self.assertIn(f"{prefix}src/unibm/ei/selection.py", names)
        self.assertIn(f"{prefix}src/unibm/ei/threshold.py", names)
        self.assertIn(f"{prefix}src/unibm/ei/_likelihood.py", names)
        self.assertIn(f"{prefix}src/unibm/ei/_stats.py", names)
        self.assertIn(f"{prefix}src/unibm/ei/_validation.py", names)
        self.assertIn(f"{prefix}src/unibm/ei/bm.py", names)
        self.assertIn(f"{prefix}src/unibm/py.typed", names)
        self.assertNotIn(f"{prefix}src/unibm/_cdf.py", names)
        self.assertNotIn(f"{prefix}src/unibm/_diagnostic_models.py", names)
        self.assertNotIn(f"{prefix}src/unibm/diagnostics/__init__.py", names)
        self.assertNotIn(f"{prefix}src/unibm/diagnostics/cdf.py", names)
        self.assertNotIn(f"{prefix}src/unibm/diagnostics.py", names)
        self.assertNotIn(f"{prefix}src/unibm/diagnostics/models.py", names)
        self.assertNotIn(f"{prefix}src/unibm/diagnostics/reciprocal.py", names)
        self.assertNotIn(f"{prefix}src/unibm/diagnostics/targets.py", names)
        self.assertNotIn(f"{prefix}src/unibm/evi/_bootstrap_eval.py", names)
        self.assertNotIn(f"{prefix}src/unibm/evi/_summaries.py", names)
        self.assertNotIn(f"{prefix}src/unibm/evi/_tail.py", names)
        self.assertNotIn(f"{prefix}src/unibm/evi/baselines.py", names)
        self.assertNotIn(f"{prefix}src/unibm/ei/_internal.py", names)
        self.assertNotIn(f"{prefix}src/unibm/evi/core.py", names)
        self.assertNotIn(f"{prefix}src/unibm/evi/external.py", names)
        self.assertNotIn(f"{prefix}src/unibm/ei/native.py", names)
        self.assertIn(f"{prefix}README.md", names)
        self.assertIn(f"{prefix}pyproject.toml", names)
        self.assertNotIn(f"{prefix}src/unibm/core.py", names)
        self.assertNotIn(f"{prefix}src/unibm/bootstrap.py", names)
        self.assertNotIn(f"{prefix}src/unibm/external.py", names)
        self.assertNotIn(f"{prefix}src/unibm/models.py", names)
        self.assertNotIn(f"{prefix}src/unibm/extremal_index.py", names)
        self.assertNotIn(f"{prefix}src/unibm/ei/_bootstrap.py", names)
        self.assertNotIn(f"{prefix}src/unibm/ei/_paths.py", names)
        self.assertNotIn(f"{prefix}src/unibm/ei/_native.py", names)
        self.assertNotIn(f"{prefix}src/unibm/ei/_threshold.py", names)
        self.assertNotIn(f"{prefix}src/unibm/ei/_shared.py", names)
        self.assertFalse(any(name.startswith(f"{prefix}scripts/application/") for name in names))
        self.assertFalse(any(name.startswith(f"{prefix}scripts/benchmark/") for name in names))
        self.assertFalse(any(name.startswith(f"{prefix}scripts/data_prep/") for name in names))
        self.assertFalse(any(name.startswith(f"{prefix}scripts/shared/") for name in names))
        self.assertFalse(any(name.startswith(f"{prefix}docs/") for name in names))
        self.assertFalse(any(name.startswith(f"{prefix}tests/") for name in names))
