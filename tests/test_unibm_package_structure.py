from __future__ import annotations

import ast
import importlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import unibm
from unibm import evi
from unibm import ei as ei_module

ROOT = Path(__file__).resolve().parents[1]
SRC_UNIBM = ROOT / "src" / "unibm"
FORBIDDEN_IMPORT_PREFIXES = (
    "scripts",
    "application",
    "benchmark",
    "shared",
    "notebook_api",
    "config",
)
REMOVED_FLAT_MODULES = (
    "unibm.core",
    "unibm.bootstrap",
    "unibm.external",
    "unibm.models",
    "unibm.extremal_index",
    "unibm.plotting",
)
REMOVED_LEGACY_MODULES = (
    "unibm.evi.core",
    "unibm.evi.external",
    "unibm.evi.baselines",
    "unibm.ei.native",
)
REMOVED_DIAGNOSTICS_MODULES = (
    "unibm.diagnostics",
    "unibm.diagnostics.cdf",
    "unibm.diagnostics.models",
    "unibm.diagnostics.reciprocal",
    "unibm.diagnostics.targets",
)


def _import_targets(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    targets: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            targets.append(node.module)
    return targets


class UniBmPackageStructureTests(unittest.TestCase):
    def test_canonical_grouped_subpackages_are_available(self) -> None:
        self.assertIs(importlib.import_module("unibm.evi"), evi)
        self.assertIs(importlib.import_module("unibm.ei"), ei_module)
        importlib.import_module("unibm.cdf")
        importlib.import_module("unibm.evi.blocks")
        importlib.import_module("unibm.evi.targets")
        importlib.import_module("unibm.evi.summaries")
        importlib.import_module("unibm.evi.selection")
        importlib.import_module("unibm.evi.estimation")
        importlib.import_module("unibm.evi.design")
        importlib.import_module("unibm.evi.plotting")
        importlib.import_module("unibm.evi.tail")
        importlib.import_module("unibm.evi.spectrum")
        importlib.import_module("unibm.evi.bootstrap")
        importlib.import_module("unibm.ei.preparation")
        importlib.import_module("unibm.ei.selection")
        importlib.import_module("unibm.ei.bm")
        importlib.import_module("unibm.ei.bootstrap")
        importlib.import_module("unibm.ei.paths")
        importlib.import_module("unibm.ei.plotting")
        importlib.import_module("unibm.ei.threshold")
        importlib.import_module("unibm.ei.models")
        self.assertIs(evi.estimate_evi_quantile, unibm.estimate_evi_quantile)
        self.assertIs(evi.estimate_design_life_level, unibm.estimate_design_life_level)
        self.assertTrue(hasattr(evi, "estimate_design_life_level_interval"))
        self.assertIs(
            evi.estimate_hill_evi,
            importlib.import_module("unibm.evi.tail").estimate_hill_evi,
        )
        self.assertIs(
            evi.estimate_max_spectrum_evi,
            importlib.import_module("unibm.evi.spectrum").estimate_max_spectrum_evi,
        )
        self.assertIs(
            evi.target_stability_summary,
            importlib.import_module("unibm.evi.targets").target_stability_summary,
        )
        self.assertIs(
            ei_module.select_stable_path_window,
            importlib.import_module("unibm.ei.selection").select_stable_path_window,
        )
        self.assertIs(
            ei_module.plot_ei_fit,
            importlib.import_module("unibm.ei.plotting").plot_ei_fit,
        )

    def test_top_level_package_lazy_loads_grouped_subpackages(self) -> None:
        self.assertIs(unibm.evi, evi)
        self.assertIs(unibm.ei, ei_module)
        self.assertEqual(
            set(unibm.__all__),
            {"__version__", "ei", "evi", "estimate_design_life_level", "estimate_evi_quantile"},
        )

    def test_numerical_imports_do_not_load_optional_plotting_stack(self) -> None:
        script = """
import json
import os
import sys

import unibm
from unibm import estimate_design_life_level, estimate_evi_quantile
import unibm.ei

print(json.dumps({
    "callable_headline_api": callable(estimate_design_life_level) and callable(estimate_evi_quantile),
    "matplotlib_loaded": any(name == "matplotlib" or name.startswith("matplotlib.") for name in sys.modules),
    "pandas_loaded": any(name == "pandas" or name.startswith("pandas.") for name in sys.modules),
    "mplconfigdir": os.environ.get("MPLCONFIGDIR"),
    "xdg_cache_home": os.environ.get("XDG_CACHE_HOME"),
}))
"""
        env = os.environ.copy()
        env.pop("MPLCONFIGDIR", None)
        env.pop("XDG_CACHE_HOME", None)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        result = json.loads(completed.stdout)

        self.assertTrue(result["callable_headline_api"])
        self.assertFalse(result["matplotlib_loaded"])
        self.assertFalse(result["pandas_loaded"])
        self.assertIsNone(result["mplconfigdir"])
        self.assertIsNone(result["xdg_cache_home"])

    def test_removed_flat_modules_are_not_importable(self) -> None:
        for module_name in REMOVED_FLAT_MODULES:
            with self.subTest(module_name=module_name):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(module_name)

    def test_removed_legacy_modules_are_not_importable(self) -> None:
        for module_name in REMOVED_LEGACY_MODULES:
            with self.subTest(module_name=module_name):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(module_name)

    def test_removed_diagnostics_modules_are_not_importable(self) -> None:
        for module_name in REMOVED_DIAGNOSTICS_MODULES:
            with self.subTest(module_name=module_name):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(module_name)

    def test_evi_grouped_namespace_does_not_leak_raw_bootstrap_sampling(self) -> None:
        for name in (
            "CircularBootstrapSampleBank",
            "default_circular_bootstrap_block_size",
            "draw_circular_block_bootstrap_sample",
            "draw_circular_block_bootstrap_samples",
        ):
            self.assertFalse(hasattr(evi, name), msg=f"unibm.evi leaked raw symbol {name}")

    def test_evi_grouped_namespace_does_not_export_removed_tail_selector_alias(self) -> None:
        self.assertFalse(hasattr(evi, "select_stable_tail_window"))

    def test_core_library_has_no_runtime_imports_from_repo_workflow_packages(self) -> None:
        for path in SRC_UNIBM.rglob("*.py"):
            imports = _import_targets(path)
            for target in imports:
                self.assertFalse(
                    target in FORBIDDEN_IMPORT_PREFIXES
                    or target.startswith(
                        tuple(f"{prefix}." for prefix in FORBIDDEN_IMPORT_PREFIXES)
                    ),
                    msg=f"{path} imports repo-workflow module {target!r}",
                )

    def test_source_tree_contains_no_removed_flat_import_statements(self) -> None:
        for source_root in (ROOT / "src", ROOT / "scripts", ROOT / "tests"):
            for path in source_root.rglob("*.py"):
                imports = _import_targets(path)
                self.assertNotIn("unibm.core", imports, msg=f"{path} still imports flat core")
                self.assertNotIn(
                    "unibm.bootstrap", imports, msg=f"{path} still imports flat bootstrap"
                )
