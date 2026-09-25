"""Check scale optimization independently and rule out held-out truth leakage."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog

from benchmark.common import interval_score
from benchmark.evi_ci_scale_calibration import (
    assign_folds,
    cross_fit,
    interval_metrics,
    optimal_scale,
)


@pytest.mark.parametrize("seed", range(4))
def test_scale_optimizer_matches_independent_linear_program(seed):
    rng = np.random.default_rng(seed)
    endpoints = np.sort(rng.normal(size=(2, 20)), axis=0)
    lo, hi = endpoints
    target = rng.normal(size=20)
    weights = rng.uniform(0.1, 2, size=20)
    weights /= weights.sum()
    # Independent epigraph formulation: minimize width + two nonnegative losses.
    objective = np.r_[weights @ (hi - lo), 40 * weights, 40 * weights]
    constraints = np.zeros((40, 41))
    constraints[:20, 0] = lo
    constraints[20:, 0] = -hi
    constraints[:20, 1:21] = -np.eye(20)
    constraints[20:, 21:] = -np.eye(20)
    reference = linprog(
        objective, A_ub=constraints, b_ub=np.r_[target, -target], bounds=(0, None), method="highs"
    )
    assert reference.success
    c, status = optimal_scale(lo, hi, target, weights)
    if reference.x[0] == 0:
        assert np.isnan(c) and status == "zero_boundary"
    else:
        independent_scores = np.array(
            [
                interval_score(truth, c * lower, c * upper, alpha=0.05)
                for lower, upper, truth in zip(lo, hi, target, strict=True)
            ]
        )
        np.testing.assert_allclose(weights @ independent_scores, reference.fun, rtol=1e-10)


def test_open_boundary_and_directional_flat_loss_are_not_hidden():
    c, status = optimal_scale([-1], [1], [0], [1])
    assert np.isnan(c) and status == "zero_boundary"
    c, status = optimal_scale([0], [0], [2], [1])
    assert c == 1 and status == "flat_minimum"
    # Only a scaled interval entirely on one side of the estimate covers truth.
    c, _ = optimal_scale([1], [2], [3], [1])
    assert c == 1.5


def test_invalid_offsets_are_rejected():
    with pytest.raises(ValueError, match="ordered offsets"):
        optimal_scale([2], [1], [0], [1])


def test_cross_fit_does_not_use_held_out_truth():
    rows = []
    for scenario in range(2):
        for rep in range(10):
            rows.append(
                {
                    "scenario": str(scenario),
                    "rep": rep,
                    "scheme": "sliding",
                    "method": "test",
                    "xi_hat": 0.3,
                    "xi_true": 0.4 + 0.1 * scenario + 0.02 * rep,
                    "ci_lo": 0.1,
                    "ci_hi": 0.5,
                }
            )
    frame = pd.DataFrame(rows)
    detail, fits, raw = cross_fit(frame, 10)
    fold = assign_folds(10)
    assert np.bincount(fold).tolist() == [2] * 5
    assert len(detail) == len(frame)
    assert fits.loc[fits.held_out_fold >= 0].training_rows.eq(16).all()
    assert fits.loc[fits.held_out_fold == -1].training_rows.eq(20).all()
    changed = frame.copy()
    changed.loc[fold[changed.rep] == 0, "xi_true"] += 10
    other, other_fits, _ = cross_fit(changed, 10)
    np.testing.assert_array_equal(
        detail.loc[detail.fold == 0, "c_cv"],
        other.loc[other.fold == 0, "c_cv"],
    )
    assert (
        other.loc[other.fold == 0, "score"].mean() > detail.loc[detail.fold == 0, "score"].mean()
    )
    assert (
        fits.loc[fits.held_out_fold == -1, "c"].iloc[0]
        != other_fits.loc[other_fits.held_out_fold == -1, "c"].iloc[0]
    )
    for row in detail.itertuples():
        expected = interval_score(row.xi_true, row.scaled_lo, row.scaled_hi, alpha=0.05)
        np.testing.assert_allclose(row.score, expected)
    np.testing.assert_allclose(detail.paired_delta, detail.score - raw.score)


def test_scaling_preserves_asymmetric_anchor_and_original_point():
    frame = pd.DataFrame({"xi_hat": [1.0], "xi_true": [1.5], "ci_lo": [0.8], "ci_hi": [1.6]})
    result = interval_metrics(frame, 2.0)
    np.testing.assert_allclose(result[["scaled_lo", "scaled_hi"]], [[0.6, 2.2]])
    assert frame.xi_hat.iloc[0] == 1.0
