from __future__ import annotations

import unittest
import tomllib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DocsSiteTests(unittest.TestCase):
    def test_guide_python_examples_execute(self) -> None:
        for page_name in ("getting-started.md", "worked-examples.md"):
            page = ROOT / "docs" / page_name
            blocks = re.findall(
                r"^```python\n(.*?)^```", page.read_text(), re.MULTILINE | re.DOTALL
            )
            self.assertTrue(blocks, page)
            namespace = {"__name__": "__docs_example__"}
            for index, code in enumerate(blocks):
                with self.subTest(page=page_name, block=index):
                    exec(compile(code, str(page), "exec"), namespace)

    def test_case_study_navigation_and_pages_are_complete(self) -> None:
        mkdocs = (ROOT / "mkdocs.yml").read_text()
        expected_pages = {
            "Cases": ROOT / "docs" / "cases" / "index.md",
            "Climate extremes": ROOT / "docs" / "cases" / "climate-extremes.md",
            "Streamflow": ROOT / "docs" / "cases" / "streamflow.md",
            "NFIP claims": ROOT / "docs" / "cases" / "nfip-claims.md",
            "Validation": ROOT / "docs" / "validation.md",
        }

        for label, path in expected_pages.items():
            self.assertIn(f"{label}:", mkdocs)
            self.assertTrue(path.is_file(), path)

    def test_notebook_workflow_is_removed(self) -> None:
        justfile = (ROOT / "justfile").read_text()
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
        dev_dependencies = set(pyproject["dependency-groups"]["dev"])

        self.assertFalse((ROOT / "notebooks").exists())
        self.assertFalse((ROOT / "scripts" / "notebook_api.py").exists())
        self.assertNotIn("notebooks/", justfile)
        self.assertNotIn("vignette:", justfile)
        self.assertNotIn("jupytext", dev_dependencies)
        self.assertNotIn("nbconvert", dev_dependencies)
        self.assertNotIn("ipykernel", dev_dependencies)

    def test_getting_started_marks_the_package_as_pre_release(self) -> None:
        getting_started = (ROOT / "docs" / "getting-started.md").read_text()

        self.assertIn("not yet released on PyPI", getting_started)
        self.assertNotIn("pip install unibm", getting_started)

    def test_case_and_validation_figures_are_tracked_static_assets(self) -> None:
        expected_assets = [
            "cases/houston_precipitation.png",
            "cases/phoenix_hotdry.png",
            "cases/tx_streamflow.png",
            "cases/fl_streamflow.png",
            "cases/tx_nfip_claims.png",
            "cases/fl_nfip_claims.png",
            "validation/evi_benchmark.png",
            "validation/ei_benchmark.png",
            "validation/evi_benchmark.csv",
            "validation/ei_benchmark.csv",
        ]

        for relative_path in expected_assets:
            path = ROOT / "docs" / "assets" / relative_path
            self.assertTrue(path.is_file(), path)
            self.assertGreater(path.stat().st_size, 0)

    def test_raw_html_figure_paths_match_built_page_depth(self) -> None:
        for page_name in ("climate-extremes.md", "streamflow.md", "nfip-claims.md"):
            page = (ROOT / "docs" / "cases" / page_name).read_text()
            self.assertIn('src="../../assets/cases/', page)
            self.assertNotIn('src="../assets/cases/', page)

        validation = (ROOT / "docs" / "validation.md").read_text()
        self.assertIn('src="../assets/validation/', validation)
        self.assertNotIn('src="assets/validation/', validation)


if __name__ == "__main__":
    unittest.main()
