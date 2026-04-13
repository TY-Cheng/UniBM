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
            "Dependence-Aware Block-Maxima Inference for Heavy-Tailed Environmental Severity, Persistence, and Design-Life Levels",
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
        self.assertIn(r"\input{Table/application_summary_main.tex}", text)
        self.assertIn(r"\input{Table/application_stationarity_main.tex}", text)
        self.assertIn(r"\input{Table/application_scaling_gof_main.tex}", text)
        self.assertIn(r"\input{Table/application_design_life_intervals_main.tex}", text)
        self.assertIn(r"\input{Table/application_ei_seasonal_sensitivity_main.tex}", text)
        self.assertIn(r"\input{Table/application_usgs_screening_main.tex}", text)
        self.assertIn(
            "The applications are instead longer-record environmental illustrations",
            text,
        )
        self.assertIn(
            "The severity branch relies on a log--log block-quantile scaling law that is appropriate as a working model only in the heavy-tailed Fr\\'echet domain",
            text,
        )
        self.assertIn(
            "POT and \\(r\\)-largest-order-statistics models remain important alternatives", text
        )
        self.assertIn("where \\((a)_+ = \\max(a,0)\\).", text)
        self.assertIn(
            "The bootstrap covariance estimator for cross-block-size sliding-block quantiles has not been formally shown to be consistent under this exact construction",
            text,
        )
        self.assertIn("heuristically calibrated rather than asymptotically exact", text)
        self.assertIn("conditional stationary extrapolations", text)
        self.assertIn("Shrinkage sensitivity for the persistence-side FGLS fit", text)
        self.assertIn(r"Figure/benchmark_ei_shrinkage_sensitivity.pdf", text)
        self.assertIn("projected heavy-tailed short-record suites", text)
        self.assertIn(r"Figure/benchmark_targets.pdf", text)
        self.assertIn(r"Figure/benchmark_ei_targets.pdf", text)
        self.assertIn(r"Figure/benchmark_stress_summary.pdf", text)
        self.assertIn(r"\input{Table/benchmark_stress_main.tex}", text)
        self.assertIn(r"\input{Table/benchmark_record_length_main.tex}", text)
        self.assertNotIn("universal truth grid", text)
        self.assertIn(
            r"External EVI baselines then benchmark the headline severity workflow",
            text,
        )
        self.assertIn(
            "not evidence used to rank cross-class interval calibration",
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
        self.assertIn(
            "illustrative replicated domains rather than representative statewide samples",
            text,
        )
        self.assertIn(
            "the final USGS sites were frozen from curated candidate pools after exploratory, method-informed screening",
            text,
        )
        self.assertIn(
            "At short manuscript-scale record lengths, the super-block rule can also collapse to very few effective resampled blocks",
            text,
        )
        self.assertIn(
            "The harder regimes nevertheless remain hard on the severity side as well.",
            text,
        )
        self.assertIn(
            "The four cases also do not carry equal evidentiary weight",
            text,
        )
        self.assertIn(
            "The Florida 50-year median design-life level is therefore retained only as a stationary counterfactual diagnostic, not as actionable design guidance.",
            text,
        )
        self.assertIn(
            "The submitted manuscript-facing code snapshot corresponds to UniBM commit \\texttt{721d60f}.",
            text,
        )
        self.assertIn(
            "pointwise intervals are conditional on the selected plateau, while top-three-window envelopes summarize local post-selection sensitivity",
            text,
        )
        self.assertNotIn("by about \\(126\\times\\) in Florida", text)
        self.assertNotIn("The manuscript should therefore be read as", text)
        self.assertNotIn("This is a central part of the contribution.", text)
        self.assertNotIn("The manuscript point is operational rather than theorem-heavy", text)
        self.assertNotIn("The manuscript claim should therefore remain calibrated", text)
        self.assertNotIn("The trade-off should be stated plainly.", text)
        self.assertNotIn("The manuscript-facing reading", text)
        self.assertNotIn(r"UniBM\_manuscript/Figure/", text)
        self.assertNotIn(r"UniBM\_manuscript/Table/", text)
        self.assertNotIn("---", text)
        self.assertNotIn("Detailed grant acknowledgements will be finalized", text)


if __name__ == "__main__":
    unittest.main()
