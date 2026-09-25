"""Check bounded EI optimization, production CI identity and held-out isolation."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import differential_evolution

from benchmark.wald_ci_scale_report import (
    cross_validate,
    endpoints,
    objective_grid,
    scenario_summary,
    theta_scale,
    z_scale,
)
from benchmark.evi_ci_scale_calibration import assign_folds
from unibm.ei._stats import _log_scale_theta_interval


def ei_frame(point, truth, half):
    point, truth, half = (np.asarray(x, dtype=float) for x in (point, truth, half))
    intervals = np.array(
        [_log_scale_theta_interval(-np.log(t), h / 1.96) for t, h in zip(point, half, strict=True)]
    )
    return pd.DataFrame(
        dict(
            estimate=point,
            truth=truth,
            z_standard_error=half / 1.96,
            ci_lo=intervals[:, 0],
            ci_hi=intervals[:, 1],
            valid=True,
        )
    )


def test_c_one_reproduces_intervals_including_boundary_and_zero_se():
    frame = ei_frame([0.2, 0.9, 1, 0.5], [0.1, 0.9, 1, 0.8], [0.1, 2, 0.3, 0])
    for scale in ("theta", "z"):
        lo, hi = endpoints(frame, scale, 1.0)
        np.testing.assert_allclose(np.c_[lo, hi], frame[["ci_lo", "ci_hi"]], rtol=1e-12)
        for c in (0.01, 2, 100):
            lo, hi = endpoints(frame, scale, c)
            assert ((0 <= lo) & (lo <= hi) & (hi <= 1)).all()


def test_theta_global_minimum_after_initial_positive_gradient():
    frame = pd.DataFrame(
        dict(estimate=[0.9, 0.1], truth=[0.9, 0.9], ci_lo=[0, 0.1], ci_hi=[1, 0.11])
    )
    weights = np.full(2, 0.5)
    c, status, _ = theta_scale(frame, weights)
    assert status == "piecewise_linear_minimum"
    np.testing.assert_allclose(c, 80, rtol=1e-12)
    np.testing.assert_allclose(objective_grid(frame, "theta", np.array([c]), weights), [0.9])


@pytest.mark.parametrize("seed", range(3))
def test_theta_sweep_matches_direct_all_knot_evaluation(seed):
    rng = np.random.default_rng(seed)
    point = rng.uniform(0.01, 1, 40)
    truth = rng.uniform(0.1, 1, 40)
    a = point * rng.uniform(0.01, 1, 40)
    b = (1 - point) * rng.uniform(0.01, 1, 40)
    frame = pd.DataFrame(dict(estimate=point, truth=truth, ci_lo=point - a, ci_hi=point + b))
    weights = rng.uniform(0.1, 1, 40)
    weights /= weights.sum()
    knots = np.r_[0, 1, (point - truth) / a, (truth - point) / b, point / a, (1 - point) / b]
    knots = knots[knots >= 0]
    ref = objective_grid(frame, "theta", knots, weights).min()
    c, status, _ = theta_scale(frame, weights)
    if status == "zero_boundary":
        np.testing.assert_allclose(ref, objective_grid(frame, "theta", np.array([0]), weights)[0])
    else:
        np.testing.assert_allclose(
            objective_grid(frame, "theta", np.array([c]), weights)[0], ref, rtol=1e-10
        )


def test_z_search_finds_nonlocal_minimum_and_matches_independent_optimizer():
    frame = ei_frame([0.9, 0.1], [0.9, 0.9], [10, 0.01])
    weights = np.full(2, 0.5)
    c, status, diag = z_scale(frame, weights)
    assert status == "numerical_minimum" and c > 100

    def fn(x):
        return objective_grid(frame, "z", np.array([x[0]]), weights)[0]

    reference = differential_evolution(
        fn, [(0, diag["upper_bound"])], rng=np.random.default_rng(5), tol=1e-10, polish=True
    )
    assert fn([c]) <= reference.fun + 1e-8
    dense = objective_grid(frame, "z", np.linspace(0, diag["upper_bound"], 10001), weights).min()
    assert fn([c]) <= dense + 1e-9


def test_theta_large_clipping_knots_match_direct_loss():
    rng = np.random.default_rng(1)
    point = rng.uniform(0.069, 1, 30)
    truth = rng.choice([0.1, 0.15, 0.25, 0.4, 0.6, 0.8, 1], 30)
    half = 10 ** rng.uniform(np.log10(1.2e-6), np.log10(0.43), 30)
    frame = ei_frame(point, truth, half)
    weights = np.full(30, 1 / 30)
    c, _, _ = theta_scale(frame, weights)
    a, b = point - frame.ci_lo, frame.ci_hi - point
    knots = np.r_[0, 1, (point - truth) / a, (truth - point) / b, point / a, (1 - point) / b]
    knots = knots[np.isfinite(knots) & (knots >= 0)]
    losses = objective_grid(frame, "theta", np.r_[c, knots], weights)
    np.testing.assert_allclose(losses[0], losses[1:].min(), atol=1e-10, rtol=1e-10)


def test_unavailable_calibration_uses_paired_denominator():
    detail = pd.DataFrame(
        dict(
            branch="ei",
            method="bb_sliding_fgls",
            scale="theta",
            scenario="a",
            family="pareto",
            xi_true=1,
            theta_true=0.5,
            rep=[0, 1],
            valid=True,
            raw_score=[1, 100],
            score=[0.5, np.nan],
            paired_delta=[-0.5, np.nan],
            raw_covered=[True, False],
            covered=[True, False],
            raw_width=[0.1, 0.2],
            width=[0.2, np.nan],
            precision_met=True,
            bootstrap_reps=128,
        )
    )
    result = scenario_summary(detail).iloc[0]
    assert result.raw_score == 50.5 and result.paired_raw_score == 1
    assert result.improvement_percent == 50
    assert result.coverage == 0.5 and result.calibration_unavailable == 1


def test_boundary_infimum_and_flat_objective():
    frame = ei_frame([1, 0.5], [1, 0.5], [0.3, 0.2])
    for solver in (theta_scale, z_scale):
        c, status, _ = solver(frame, np.full(2, 0.5))
        assert np.isnan(c) and status == "zero_boundary"
        flat = ei_frame([1, 0.5], [0.9, 0.6], [0, 0])
        c, status, _ = solver(flat, np.full(2, 0.5))
        assert c == 1 and status == "flat_minimum"


def test_cv_keeps_failed_rows_and_excludes_held_out_truth_from_both_ei_scales():
    frame = ei_frame(np.tile([0.3, 0.5], 10), np.tile([0.4, 0.6], 10), np.full(20, 0.2))
    frame = frame.assign(
        branch="ei",
        method="bb_sliding_fgls",
        scenario=np.repeat(["a", "b"], 10),
        rep=np.tile(np.arange(10), 2),
    )
    frame.loc[0, ["estimate", "ci_lo", "ci_hi", "z_standard_error"]] = np.nan
    frame.loc[0, "valid"] = False
    (
        original,
        _,
    ) = cross_validate(frame, 10)
    changed = frame.copy()
    fold = assign_folds(10)[changed.rep]
    changed.loc[fold == 0, "truth"] = 0.95
    other, _ = cross_validate(changed, 10)
    for scale in ("theta", "z"):
        a = original[(original.scale == scale) & (original.fold == 0)]
        b = other[(other.scale == scale) & (other.fold == 0)]
        np.testing.assert_array_equal(a.c_cv, b.c_cv)
    failed = original[~original.valid]
    assert len(failed) == 2 and failed.score.isna().all()
    assert not failed.covered.any() and not failed.raw_covered.any()
