from __future__ import annotations
# ruff: noqa: E402

from contextlib import redirect_stdout
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
for path in (SCRIPTS_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from shared.runtime import resolve_bool_env, resolve_int_env, status


class WorkflowRuntimeTests(unittest.TestCase):
    def test_resolve_int_env_respects_default_and_minimum(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=False):
            self.assertEqual(resolve_int_env("UNSET_ENV", default=3, minimum=1), 3)
        with mock.patch.dict("os.environ", {"BAD_ENV": "oops"}, clear=False):
            self.assertEqual(resolve_int_env("BAD_ENV", default=5, minimum=2), 5)
        with mock.patch.dict("os.environ", {"LOW_ENV": "0"}, clear=False):
            self.assertEqual(resolve_int_env("LOW_ENV", default=5, minimum=2), 2)

    def test_status_emits_timestamped_prefix(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            status("application", "building bundles")
        output = buffer.getvalue().strip()
        self.assertRegex(
            output,
            r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\[application\] building bundles$",
        )

    def test_resolve_bool_env_understands_common_truthy_and_falsey_values(self) -> None:
        with mock.patch.dict("os.environ", {"BOOL_ENV": "yes"}, clear=False):
            self.assertTrue(resolve_bool_env("BOOL_ENV"))
        with mock.patch.dict("os.environ", {"BOOL_ENV": "0"}, clear=False):
            self.assertFalse(resolve_bool_env("BOOL_ENV", default=True))
        with mock.patch.dict("os.environ", {"BOOL_ENV": "unexpected"}, clear=False):
            self.assertTrue(resolve_bool_env("BOOL_ENV", default=True))


if __name__ == "__main__":
    unittest.main()
