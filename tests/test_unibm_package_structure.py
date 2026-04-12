from __future__ import annotations

import ast
import importlib
import re
import unittest
from pathlib import Path

import unibm
from unibm import evi
from unibm import ei as ei_module


ROOT = Path(__file__).resolve().parents[1]
SRC_UNIBM = ROOT / "src" / "unibm"
TEXT_SCAN_SUFFIXES = {".py", ".rst", ".md"}
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
)
REMOVED_PRIVATE_EI_MODULES = (
    "unibm.ei._bootstrap",
    "unibm.ei._paths",
    "unibm.ei._native",
    "unibm.ei._threshold",
    "unibm.ei._shared",
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
        importlib.import_module("unibm.evi.bootstrap")
        importlib.import_module("unibm.ei.bootstrap")
        importlib.import_module("unibm.ei.paths")
        importlib.import_module("unibm.ei.native")
        importlib.import_module("unibm.ei.threshold")
        importlib.import_module("unibm.ei.models")
        self.assertIs(evi.estimate_evi_quantile, unibm.estimate_evi_quantile)
        self.assertIs(evi.estimate_design_life_level, unibm.estimate_design_life_level)

    def test_top_level_package_lazy_loads_grouped_subpackages(self) -> None:
        self.assertIs(unibm.evi, evi)
        self.assertIs(unibm.ei, ei_module)
        self.assertEqual(
            set(unibm.__all__),
            {"__version__", "ei", "evi", "estimate_design_life_level", "estimate_evi_quantile"},
        )

    def test_removed_flat_modules_are_not_importable(self) -> None:
        for module_name in REMOVED_FLAT_MODULES:
            with self.subTest(module_name=module_name):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(module_name)
        for module_name in REMOVED_PRIVATE_EI_MODULES:
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

    def test_repo_contains_no_old_flat_import_statements(self) -> None:
        pattern = re.compile(
            r"^\s*(?:from|import)\s+unibm\.(core|bootstrap|external|models|extremal_index|summaries|window_ops)\b"
        )
        private_ei_pattern = re.compile(
            r"^\s*(?:from|import)\s+unibm\.ei\._(bootstrap|paths|native|threshold|shared)\b"
        )
        skip_roots = {ROOT / "docs" / "_build", ROOT / "dist"}
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix not in TEXT_SCAN_SUFFIXES:
                continue
            if any(skip_root in path.parents for skip_root in skip_roots):
                continue
            text = path.read_text(encoding="utf-8")
            self.assertIsNone(
                pattern.search(text), msg=f"{path} still imports a removed flat module"
            )
            self.assertIsNone(
                private_ei_pattern.search(text),
                msg=f"{path} still imports a removed private EI module",
            )
