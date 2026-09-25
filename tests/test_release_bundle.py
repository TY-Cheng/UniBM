"""Keep the manual publishing workflow from accepting incomplete or altered files."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "verify_release", Path(__file__).resolve().parents[1] / ".github/scripts/verify_release.py"
)
assert SPEC is not None and SPEC.loader is not None
RELEASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RELEASE)


class ReleaseBundleTests(unittest.TestCase):
    def test_only_the_complete_unchanged_distribution_inventory_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            names = [
                "unibm-0.2.0.tar.gz",
                "unibm-0.2.0-py3-none-any.whl",
                "unibm-0.2.0-cp314-cp314-win_amd64.whl",
            ]
            manifest = {"version": "0.2.0", "artifacts": {}}
            for name in names:
                content = name.encode()
                (directory / name).write_bytes(content)
                manifest["artifacts"][name] = hashlib.sha256(content).hexdigest()
            RELEASE.verify_files(directory, manifest)

            extra = directory / "unreviewed.whl"
            extra.touch()
            with self.assertRaisesRegex(ValueError, "inventory"):
                RELEASE.verify_files(directory, manifest)
            extra.unlink()

            package = directory / names[0]
            package.write_bytes(b"changed after review")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                RELEASE.verify_files(directory, manifest)
            package.unlink()
            with self.assertRaisesRegex(ValueError, "inventory"):
                RELEASE.verify_files(directory, manifest)
