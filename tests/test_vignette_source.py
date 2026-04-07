from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VIGNETTE_SOURCE = ROOT / "notebooks" / "vignette.py"


def _code_cells(text: str) -> list[str]:
    parts = re.split(r"^# %%.*$", text, flags=re.MULTILINE)
    cells: list[str] = []
    for part in parts:
        stripped = part.lstrip()
        if not stripped or stripped.startswith("[markdown]"):
            continue
        cells.append(part)
    return cells


class VignetteSourceTests(unittest.TestCase):
    def test_vignette_code_cells_compile(self) -> None:
        text = VIGNETTE_SOURCE.read_text()
        for source in _code_cells(text):
            compile(source, "<vignette-cell>", "exec")

    def test_vignette_references_composite_application_workflow(self) -> None:
        text = VIGNETTE_SOURCE.read_text()
        self.assertIn("plot_application_composite", text)
        self.assertIn("application_ei_seasonal_methods", text)
        self.assertIn("_load_vignette_api", text)
        self.assertNotIn("houston_ei =", text)
        self.assertNotIn("phoenix_ei =", text)


if __name__ == "__main__":
    unittest.main()
