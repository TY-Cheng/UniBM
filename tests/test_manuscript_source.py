from __future__ import annotations
# ruff: noqa: E402

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
for path in (SCRIPTS_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import resolve_repo_dirs


MANUSCRIPT_SOURCE = resolve_repo_dirs(ROOT)["DIR_MANUSCRIPT"] / "0_manuscript.tex"


class ManuscriptSourceTests(unittest.TestCase):
    def test_manuscript_references_fixed_fgls_defaults_and_clock_split(self) -> None:
        text = MANUSCRIPT_SOURCE.read_text()
        self.assertIn(
            "Dependence-Aware Block-Maxima Inference for Severity, Persistence, and Design-Life Levels",
            text,
        )
        self.assertIn(r"\delta=0.35", text)
        self.assertIn(r"\lambda_{\mathrm{curv}}=2.0", text)
        self.assertIn(r"\subsection{Bootstrap/CI under overlap}", text)
        self.assertIn(r"\subsection{Design-life summaries}", text)
        self.assertIn(r"\subsection{Benchmark design}", text)
        self.assertIn(r"\subsection{EVI benchmark}", text)
        self.assertIn(r"\subsection{EI benchmark}", text)
        self.assertIn(r"\texttt{paper\_subset\_manifest.json}", text)
        self.assertIn(r"\texttt{just manuscript}", text)
        self.assertIn(r"\input{Snippet/application_case_context_main.tex}", text)
        self.assertIn(
            "The applications are instead longer-record environmental illustrations",
            text,
        )
        self.assertIn(
            "Design-life levels are derived from the severity branch, while the extremal index is reported as a complementary persistence descriptor.",
            text,
        )
        self.assertIn("Shrinkage sensitivity for the persistence-side FGLS fit", text)
        self.assertIn(r"Figure/benchmark_ei_shrinkage_sensitivity.pdf", text)
        self.assertIn(
            "two complementary short-record suites",
            text,
        )
        self.assertIn(r"Figure/benchmark_targets.pdf", text)
        self.assertIn(r"Figure/benchmark_ei_targets.pdf", text)
        self.assertNotIn("universal truth grid", text)
        self.assertIn(
            r"External EVI baselines then benchmark the headline severity workflow",
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
        self.assertIn(
            "not exposure-normalized portfolio risk and not a compound-loss model",
            text,
        )
        self.assertIn(
            "The repository generates a broader exploratory artifact set than the submitted manuscript uses.",
            text,
        )
        self.assertIn(
            "curated four-application subset",
            text,
        )
        self.assertNotIn("The manuscript should therefore be read as", text)
        self.assertNotIn("This is a central part of the contribution.", text)
        self.assertNotIn("The manuscript point is operational rather than theorem-heavy", text)
        self.assertNotIn("The manuscript claim should therefore remain calibrated", text)
        self.assertNotIn("The trade-off should be stated plainly.", text)
        self.assertNotIn("The manuscript-facing reading", text)
        self.assertNotIn(r"UniBM\_manuscript/Figure/", text)
        self.assertNotIn(r"UniBM\_manuscript/Table/", text)
        self.assertNotIn("---", text)


if __name__ == "__main__":
    unittest.main()
