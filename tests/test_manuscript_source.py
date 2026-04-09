from __future__ import annotations
# ruff: noqa: E402

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from config import resolve_repo_dirs


MANUSCRIPT_SOURCE = resolve_repo_dirs(ROOT)["DIR_MANUSCRIPT"] / "0_manuscript.tex"


class ManuscriptSourceTests(unittest.TestCase):
    def test_manuscript_references_fixed_fgls_defaults_and_clock_split(self) -> None:
        text = MANUSCRIPT_SOURCE.read_text()
        self.assertIn(
            "UniBM: A Dependence-Aware Block-Maxima Workflow for Severity, Persistence, and Design-Life Levels",
            text,
        )
        self.assertIn(r"\delta=0.35", text)
        self.assertIn(r"\lambda_{\mathrm{curv}}=2.0", text)
        self.assertIn("Shrinkage sensitivity for the persistence-side FGLS fit", text)
        self.assertIn(r"Figure/benchmark_ei_shrinkage_sensitivity.pdf", text)
        self.assertIn(
            "two projected short-record suites sharing a common synthetic design philosophy",
            text,
        )
        self.assertIn(r"Figure/benchmark_targets.pdf", text)
        self.assertIn(r"Figure/benchmark_ei_targets.pdf", text)
        self.assertNotIn("universal truth grid", text)
        self.assertIn(
            r"external \(\xi\) baselines are reported with their native asymptotic intervals",
            text,
        )
        self.assertIn(
            "the applications are used more modestly as longer-record illustrations of the workflow's interpretability",
            text,
        )
        self.assertIn(
            "The applications are instead longer-record environmental illustrations",
            text,
        )
        self.assertIn(
            "A classical return level is usually framed through annual or block exceedance probabilities.",
            text,
        )
        self.assertIn(
            "The design-life level used here answers a different question",
            text,
        )
        self.assertIn("A note on clocks", text)
        self.assertIn(
            "does not identify a single calendar-time aggregate loss quantity",
            text,
        )
        self.assertIn(
            "The repository generates a broader exploratory artifact set than the submitted manuscript uses.",
            text,
        )
        self.assertIn(r"\texttt{paper\_subset\_manifest.json}", text)
        self.assertIn(r"\texttt{just manuscript}", text)
        self.assertIn(
            "curated four-application subset",
            text,
        )
        self.assertNotIn(r"UniBM\_manuscript/Figure/", text)
        self.assertNotIn(r"UniBM\_manuscript/Table/", text)
        self.assertNotIn("---", text)


if __name__ == "__main__":
    unittest.main()
