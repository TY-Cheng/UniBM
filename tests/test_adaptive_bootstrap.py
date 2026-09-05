"""Adaptive defaults retain explicit fixed-R and estimator identity contracts."""

import numpy as np
import pytest

from unibm._bootstrap_precision import adaptive_covariance
from unibm.evi import estimate_evi_quantile
from unibm.ei import bootstrap_bm_ei_path, estimate_pooled_bm_ei, prepare_ei_bundle


def test_evi_default_is_adaptive_but_integer_reps_remain_exact():
    values = np.exp(np.random.default_rng(11).normal(size=365))
    adaptive = estimate_evi_quantile(values, regression="FGLS", random_state=7)
    assert adaptive.bootstrap_reps_policy == "adaptive"
    assert adaptive.bootstrap_reps_used in (128, 256, 512, 768, 1024)
    assert adaptive.bootstrap_precision_met in (True, False)
    assert adaptive.bootstrap_mcse_targets == ("xi", "xi_ci_lo", "xi_ci_hi")
    fixed = estimate_evi_quantile(values, regression="FGLS", bootstrap_reps=120, random_state=7)
    assert fixed.bootstrap_reps_policy == "fixed"
    assert fixed.bootstrap_reps_used == fixed.bootstrap_reps_requested == 120
    assert fixed.bootstrap_precision_met is None
    np.testing.assert_array_equal(adaptive.bootstrap["samples"][:120], fixed.bootstrap["samples"])


@pytest.mark.parametrize("base_path, sliding", [("bb", True), ("bb", False), ("northrop", True)])
def test_ei_default_is_adaptive_and_covariance_reuse_preserves_precision_identity(
    base_path, sliding
):
    values = np.exp(np.random.default_rng(12).normal(size=365))
    bundle = prepare_ei_bundle(values, allow_zeros=False)
    result = bootstrap_bm_ei_path(
        values,
        allow_zeros=False,
        base_path=base_path,
        sliding=sliding,
        block_sizes=bundle.block_sizes,
        random_state=7,
    )
    fit = estimate_pooled_bm_ei(
        bundle,
        base_path=base_path,
        sliding=sliding,
        regression="FGLS",
        bootstrap_result=result,
    )
    assert fit.bootstrap_reps_policy == "adaptive"
    assert fit.bootstrap_reps_used in (128, 256, 512, 768, 1024)
    assert fit.bootstrap_precision_met in (True, False)
    assert len(fit.bootstrap_mcse) == 6  # Includes unconstrained z at the theta=1 boundary.
    other = estimate_pooled_bm_ei(
        bundle,
        base_path=base_path,
        sliding=sliding,
        regression="FGLS",
        bootstrap_result=result,
        covariance_shrinkage=0.17,
    )
    assert other.bootstrap_reps_policy == "adaptive"
    assert other.bootstrap_precision_met is None  # Its new weights were not checked.
    fixed = bootstrap_bm_ei_path(
        values,
        allow_zeros=False,
        base_path=base_path,
        sliding=sliding,
        block_sizes=bundle.block_sizes,
        random_state=7,
        reps=120,
    )
    np.testing.assert_array_equal(result["samples"][:120], fixed["samples"])
    assert fixed["bootstrap_reps_policy"] == "fixed"


def test_adaptive_cap_keeps_all_draws_and_reports_unmet_precision():
    increments = []

    def draw(count, rng):
        increments.append(count)
        return rng.normal(size=(count, 2))

    def evaluate(covariance):
        return np.diag(covariance), np.full(2, 1e-12)

    with pytest.warns(RuntimeWarning, match="R=1024"):
        result = adaptive_covariance(draw, evaluate, random_state=7)
    assert increments == [128, 128, 256, 256, 256]
    assert result["bootstrap_reps_used"] == 1024
    assert result["bootstrap_precision_met"] is False
    assert result["bootstrap_mcse_max_ratio"] > 0.10
    np.testing.assert_array_equal(
        result["samples"], np.random.default_rng(7).normal(size=(1024, 2))
    )


def test_adaptive_precision_can_stop_at_first_checkpoint():
    def draw(count, rng):
        return rng.normal(size=(count, 2))

    def evaluate(covariance):
        return np.zeros(2), np.ones(2)

    result = adaptive_covariance(draw, evaluate, random_state=7)
    assert result["bootstrap_reps_used"] == 128
    assert result["bootstrap_precision_met"] is True
    assert result["bootstrap_mcse"] == (0.0, 0.0)
