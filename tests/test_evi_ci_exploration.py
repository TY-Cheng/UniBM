"""Independent formula, sampler and degenerate-case checks for the CI study."""

import numpy as np
import pandas as pd
import pytest

from benchmark.evi_ci_exploration import (
    BASELINE,
    CANDIDATES,
    METHOD_INDEX,
    automatic_shrinkages,
    candidate_intervals,
    evaluate_series,
)
from unibm._bootstrap_precision import adaptive_covariance
from unibm.evi import estimate_evi_quantile
from unibm.evi._regression import Z_CRIT_95, _fit_linear_model
from benchmark.evi_ci_exploration_report import METRICS, balanced_summary


def test_automatic_intensities_match_direct_outer_product_formulas():
    rows = np.random.default_rng(20).normal(size=(128, 5)) @ np.tril(np.ones((5, 5)))
    n, p = rows.shape
    S = np.cov(rows, rowvar=False)
    centered = rows - rows.mean(axis=0)
    standardized = centered / np.sqrt(np.diag(S))
    products = np.array([np.outer(row, row) for row in standardized])
    correlation = products.sum(axis=0) / (n - 1)
    off = ~np.eye(p, dtype=bool)
    var_correlation = n / (n - 1) ** 2 * products.var(axis=0, ddof=1)
    ss = var_correlation[off].sum() / (correlation[off] ** 2).sum()
    outer = np.array([np.outer(row, row) for row in centered])
    ml = outer.mean(axis=0)
    target = np.eye(p) * np.trace(ml) / p
    lw = ((outer - ml) ** 2).sum() / (n * n * ((ml - target) ** 2).sum())
    tr_square = np.trace(ml @ ml)
    oas = ((1 - 2 / p) * tr_square + np.trace(ml) ** 2) / (
        (n + 1 - 2 / p) * (tr_square - np.trace(ml) ** 2 / p)
    )
    np.testing.assert_allclose(automatic_shrinkages(rows, S), np.clip([ss, lw, oas], 0, 1))


def test_projection_variance_and_bootstrap_centering():
    x = np.log(np.arange(8, 13))
    observed = 0.4 + 0.3 * x
    center = observed + 0.2 * x  # Deliberately different circular-bank slope.
    rows = center + np.random.default_rng(21).exponential(0.1, size=(512, 5))
    S = np.cov(rows, rowvar=False)
    result = candidate_intervals(rows, S, x, observed, center)
    np.testing.assert_allclose(
        result["bootstrap_se"] ** 2,
        result["draws"].var(axis=0, ddof=1)[METHOD_INDEX],
        rtol=1e-10,
        atol=1e-12,
    )
    index = CANDIDATES.index(("diagonal_0.37", "model_wald"))
    expected = _fit_linear_model(x, observed, S, 0.37)
    np.testing.assert_allclose(result["xi"][index], expected["slope"], atol=1e-10)
    np.testing.assert_allclose(result["model_se"][index], expected["standard_error"])
    for cov in ("diagonal_0.37", "ols"):
        basic = CANDIDATES.index((cov, "basic"))
        percentile = CANDIDATES.index((cov, "aligned_percentile"))
        bias_normal = CANDIDATES.index((cov, "bias_normal"))
        column = METHOD_INDEX[basic]
        errors = result["draws"][:, column] - result["center_xi"][basic]
        lo, hi = np.quantile(errors, [0.025, 0.975])
        xi = result["xi"][basic]
        np.testing.assert_allclose(result["endpoints"][basic], [xi - hi, xi - lo])
        np.testing.assert_allclose(result["endpoints"][percentile], [xi + lo, xi + hi])
        half = Z_CRIT_95 * errors.std(ddof=1)
        np.testing.assert_allclose(
            result["endpoints"][bias_normal],
            [xi - errors.mean() - half, xi - errors.mean() + half],
        )


@pytest.mark.parametrize("sliding", [False, True])
def test_baseline_reproduces_public_estimator_at_shared_final_budget(sliding):
    values = np.exp(np.random.default_rng(22).normal(size=365))
    result = evaluate_series(values, sliding=sliding, seed=7)
    assert len(result) == 49
    baseline = result.loc[result.method == BASELINE].iloc[0]
    assert result.bootstrap_reps.nunique() == 1
    fit = estimate_evi_quantile(
        values,
        regression="FGLS",
        sliding=sliding,
        random_state=7,
        bootstrap_reps=int(baseline.bootstrap_reps),
        covariance_shrinkage=0.37,  # Preserve the study's original reference method.
        n_threads=1,
    )
    np.testing.assert_allclose(baseline.xi_hat, fit.slope, atol=1e-9)
    np.testing.assert_allclose(
        [baseline.ci_lo, baseline.ci_hi], fit.confidence_interval, rtol=1e-7, atol=1e-9
    )


def test_flat_bootstrap_is_retained_and_not_certified_as_precise():
    result = evaluate_series(np.ones(365), sliding=True, seed=8)
    assert result.flat_window.all()
    assert not result.precision_met.any()
    assert (result.bootstrap_reps == 1024).all()
    assert result.loc[result.covariance != "ols", "error"].str.len().gt(0).all()
    ols = result.loc[result.covariance == "ols"]
    assert np.isfinite(ols[["ci_lo", "ci_hi"]]).all().all()


def test_precision_callback_receives_exact_paired_rows():
    calls = []

    def evaluate(covariance, rows):
        np.testing.assert_allclose(covariance, np.cov(rows, rowvar=False))
        calls.append(len(rows))
        return rows.mean(axis=0), np.ones(2)

    result = adaptive_covariance(lambda n, rng: rng.normal(size=(n, 2)), evaluate, random_state=1)
    assert 128 in calls and 112 in calls  # Full checkpoint and 7/8 delete-group rows.
    assert result["bootstrap_reps_used"] in (128, 256, 512, 768, 1024)


def test_summary_mcse_respects_shared_replication_indices():
    rows = []
    for scenario in range(3):
        for rep in range(4):
            row = {"scheme": "sliding", "method": "test", "scenario": str(scenario), "rep": rep}
            row.update({metric: 10 * scenario + rep for metric in METRICS})
            rows.append(row)
    summary = balanced_summary(pd.DataFrame(rows), replicates=4).iloc[0]
    assert summary.score == 11.5
    np.testing.assert_allclose(summary.score_mcse, np.std(np.arange(4), ddof=1) / np.sqrt(4))
    assert summary.n_scenarios == 3 and summary.n_valid == 12
