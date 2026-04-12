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
        self.assertIn(
            "Definitions: Return Period, Design-Life Level, and `T`-Year Block-Maximum",
            text,
        )
        self.assertIn(
            '_display_workflow_svg(ROOT / "docs" / "_static" / "evi_workflow.dot")', text
        )
        self.assertIn('_display_workflow_svg(ROOT / "docs" / "_static" / "ei_workflow.dot")', text)
        self.assertIn("application_design_life_levels.csv", text)
        self.assertIn("_load_notebook_api", text)
        self.assertIn('"notebook_api.py"', text)
        self.assertNotIn("application_return_levels.csv", text)
        self.assertNotIn("EI-adjusted", text)
        self.assertNotIn("houston_ei =", text)
        self.assertNotIn("phoenix_ei =", text)


if __name__ == "__main__":
    unittest.main()
