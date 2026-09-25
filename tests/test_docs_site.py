from __future__ import annotations

from contextlib import chdir
import tempfile
import unittest
import tomllib
import re
import json
from pathlib import Path

from mkdocs.config import load_config

ROOT = Path(__file__).resolve().parents[1]


class DocsSiteTests(unittest.TestCase):
    def test_mermaid_rendering_is_owned_by_material(self) -> None:
        config = load_config(config_file=str(ROOT / "mkdocs.yml"))
        self.assertEqual(config.theme.name, "material")
        fences = config.mdx_configs["pymdownx.superfences"]["custom_fences"]
        self.assertTrue(any(fence["name"] == fence["class"] == "mermaid" for fence in fences))
        self.assertFalse(any("mermaid" in str(script) for script in config.extra_javascript))

    def test_guide_python_examples_execute(self) -> None:
        for page_name in ("getting-started.md", "worked-examples.md"):
            page = ROOT / "docs" / page_name
            blocks = re.findall(
                r"^```python\n(.*?)^```",
                page.read_text(encoding="utf-8"),
                re.MULTILINE | re.DOTALL,
            )
            self.assertTrue(blocks, page)
            namespace = {"__name__": "__docs_example__"}
            with tempfile.TemporaryDirectory() as tmp, chdir(tmp):
                for index, code in enumerate(blocks):
                    with self.subTest(page=page_name, block=index):
                        exec(compile(code, str(page), "exec"), namespace)

    def test_case_study_navigation_and_pages_are_complete(self) -> None:
        mkdocs = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
        expected_pages = {
            "Cases": ROOT / "docs" / "cases" / "index.md",
            "Houston precipitation": ROOT / "docs/cases/houston-precipitation.md",
            "Phoenix hot–dry severity": ROOT / "docs/cases/phoenix-hot-dry.md",
            "GOES soft X-rays": ROOT / "docs/cases/goes-xray.md",
            "SPY / QQQ losses": ROOT / "docs/cases/spy-qqq.md",
            "Streamflow": ROOT / "docs" / "cases" / "streamflow.md",
            "NFIP claims": ROOT / "docs" / "cases" / "nfip-claims.md",
            "Benchmark": ROOT / "docs" / "benchmark.md",
        }

        self.assertLess(mkdocs.index("  - Benchmark:"), mkdocs.index("  - Cases:"))
        for label, path in expected_pages.items():
            self.assertIn(f"{label}:", mkdocs)
            self.assertTrue(path.is_file(), path)

    def test_frozen_case_records_keep_inference_scopes_distinct(self) -> None:
        assets = ROOT / "docs/assets/cases"
        records = [json.loads(path.read_text(encoding="utf-8")) for path in assets.glob("*.json")]
        self.assertEqual(len(records), 9)
        for record in records:
            summary = record["summary"]
            if summary["application"] == "goes":
                self.assertFalse(record["ci_reported"])
                self.assertIsNone(summary["xi_lo"])
                self.assertIsNone(summary["xi_hi"])
                self.assertIsNone(summary["theta_hat_bb_sliding_fgls"])
                self.assertEqual(record["ei_methods"], [])
                self.assertLess(record["eligible_observations"], summary["n_evi_obs"])
            else:
                self.assertTrue(record["ci_reported"])
                self.assertEqual(summary["evi_regression"], "FGLS")
                self.assertEqual(len(record["ei_methods"]), 4)

    def test_notebook_workflow_is_removed(self) -> None:
        justfile = (ROOT / "justfile").read_text(encoding="utf-8")
        pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        dev_dependencies = set(pyproject["dependency-groups"]["dev"])

        self.assertFalse((ROOT / "notebooks").exists())
        self.assertFalse((ROOT / "scripts" / "notebook_api.py").exists())
        self.assertNotIn("notebooks/", justfile)
        self.assertNotIn("vignette:", justfile)
        self.assertNotIn("jupytext", dev_dependencies)
        self.assertNotIn("nbconvert", dev_dependencies)
        self.assertNotIn("ipykernel", dev_dependencies)

    def test_getting_started_documents_pypi_installation(self) -> None:
        getting_started = (ROOT / "docs" / "getting-started.md").read_text(encoding="utf-8")

        self.assertIn("https://pypi.org/project/unibm/", getting_started)
        self.assertIn("python -m pip install unibm", getting_started)
        self.assertNotIn("not yet released on PyPI", getting_started)

    def test_case_and_benchmark_figures_are_tracked_static_assets(self) -> None:
        expected_assets = [
            "cases/houston_precipitation.png",
            "cases/phoenix_hotdry.png",
            "cases/tx_streamflow.png",
            "cases/fl_streamflow.png",
            "cases/tx_nfip_claims.png",
            "cases/fl_nfip_claims.png",
            "cases/goes_normalized.png",
            "cases/spy_normalized.png",
            "cases/qqq_normalized.png",
            "benchmark/evi_benchmark.png",
            "benchmark/ei_benchmark.png",
            "benchmark/evi_benchmark.csv",
            "benchmark/ei_benchmark.csv",
        ]

        for relative_path in expected_assets:
            path = ROOT / "docs" / "assets" / relative_path
            self.assertTrue(path.is_file(), path)
            self.assertGreater(path.stat().st_size, 0)

    def test_raw_html_figure_paths_match_built_page_depth(self) -> None:
        for page_name in (
            "houston-precipitation.md",
            "phoenix-hot-dry.md",
            "goes-xray.md",
            "spy-qqq.md",
            "streamflow.md",
            "nfip-claims.md",
        ):
            page = (ROOT / "docs" / "cases" / page_name).read_text(encoding="utf-8")
            self.assertIn('src="../../assets/cases/', page)
            self.assertNotIn('src="../assets/cases/', page)

        validation = (ROOT / "docs" / "benchmark.md").read_text(encoding="utf-8")
        self.assertIn('src="../assets/benchmark/', validation)
        self.assertNotIn('src="assets/benchmark/', validation)


if __name__ == "__main__":
    unittest.main()
