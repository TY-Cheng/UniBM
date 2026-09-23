"""Independent numerical oracles for the promoted algorithm and native kernels."""

from dataclasses import asdict

import numpy as np
import pytest

from unibm.evi import _accelerator
from unibm.evi._mode import mode_from_counts, prepare_mode_counts, weighted_density
from unibm.evi._quantile_bootstrap import (
    prepare_quantile_counts,
    quantile_from_counts,
    segment_multiplicities,
)
from unibm.evi.bootstrap import _evaluate_mode_bootstrap_column_batched
from unibm.evi.estimation import estimate_target_scaling
from unibm.evi.tail import _dedh_moment_path, _hill_path, _pickands_path


@pytest.mark.parametrize("native", [False, True])
def test_compressed_kde_matches_expanded_gaussians(monkeypatch, native):
    if native and _accelerator.kernels is None:
        pytest.skip("Optional native extension is not built")
    if not native:
        monkeypatch.setattr(_accelerator, "kernels", None)
    rng = np.random.default_rng(519)
    logs = np.sort(rng.uniform(0.1, 3, 19))
    counts = rng.integers(0, 6, size=(7, len(logs)))
    grid = np.broadcast_to(np.linspace(0.1, 3, 256), (7, 256)).copy()
    bandwidth = rng.uniform(0.02, 0.4, 7)
    expected = []
    for row, weights in enumerate(counts):
        expanded = np.repeat(logs, weights)
        expected.append(
            np.exp(-0.5 * ((grid[row, :, None] - expanded) / bandwidth[row]) ** 2).sum(axis=1)
        )
    np.testing.assert_allclose(
        weighted_density(logs, counts, grid, bandwidth), expected, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("shape", [(4, 17), (4, 1200)])
def test_native_and_numpy_quantile_ranks_use_identical_interpolation(monkeypatch, native, shape):
    if native and _accelerator.kernels is None:
        pytest.skip("Optional native extension is not built")
    if not native:
        monkeypatch.setattr(_accelerator, "kernels", None)
    rng = np.random.default_rng(712)
    bank = np.round(rng.lognormal(size=shape), 2) + 0.01
    draws = rng.integers(0, shape[0], size=(37, shape[0]))
    weights = segment_multiplicities(draws)
    table = prepare_quantile_counts(bank, max_bytes=64 * 1024**2)
    # Const memoryviews accept read-only input without copying or mutation.
    for arr in (*table, weights):
        arr.flags.writeable = False
    for q in [1e-9, 0.1, 1 / 3, 0.5, 0.95, 0.99, 1 - 1e-9]:
        expected = np.quantile(bank[draws].reshape(37, -1), q, axis=1, method="median_unbiased")
        actual = quantile_from_counts(table, weights, size=bank.size, quantile=q, max_bytes=1024)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("kind", ["continuous", "ties", "constant", "near_constant"])
def test_mode_counts_preserve_expanded_bootstrap_grid_selection(kind):
    rng = np.random.default_rng(719)
    bank = rng.lognormal(size=(4, 127))
    if kind == "ties":
        bank = np.round(bank, 1) + 0.1
    elif kind == "constant":
        bank[:] = 2.5
    elif kind == "near_constant":
        bank = 1 + bank * 1e-12
    draws = rng.integers(0, 4, size=(37, 4))
    values, counts = prepare_mode_counts(bank, max_bytes=64 * 1024**2)
    expected = _evaluate_mode_bootstrap_column_batched(bank[draws].reshape(37, -1))
    actual = mode_from_counts(values, segment_multiplicities(draws) @ counts)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert prepare_mode_counts(bank, max_bytes=0) is None
    for invalid in [0, -1, np.inf, np.nan]:
        bank[0, 0] = invalid
        assert prepare_mode_counts(bank, max_bytes=64 * 1024**2) is None


def _assert_result_equal(left, right):
    """Compare every returned field, including adaptive diagnostics and samples."""
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_result_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_result_equal(a, b)
    elif isinstance(left, (float, np.floating, np.ndarray)):
        np.testing.assert_allclose(left, right, rtol=1e-10, atol=1e-12, equal_nan=True)
    else:
        assert left == right


@pytest.mark.parametrize("target", ["mode", "quantile"])
@pytest.mark.parametrize("sliding", [False, True])
def test_native_and_fallback_preserve_full_adaptive_fit(monkeypatch, target, sliding):
    if _accelerator.kernels is None:
        pytest.skip("Optional native extension is not built")
    sample = np.random.default_rng(18).pareto(2, 256) + 1
    settings = dict(
        target=target,
        quantile=0.95,
        sliding=sliding,
        regression="FGLS",
        random_state=7,
        n_threads=3,
    )
    accelerated = estimate_target_scaling(sample, **settings)
    monkeypatch.setattr(_accelerator, "kernels", None)
    fallback = estimate_target_scaling(sample, **settings)
    _assert_result_equal(asdict(accelerated), asdict(fallback))


@pytest.mark.parametrize("scale", [1e-100, 1.0, 1e100])
@pytest.mark.parametrize("ties", [False, True])
def test_tail_paths_match_centered_definitions(scale, ties):
    ordered = np.sort(np.random.default_rng(321).pareto(2, 4096) + 1)[::-1]
    if ties:
        ordered[:200] = ordered[200]
    ordered *= scale
    levels = np.unique(np.rint(np.geomspace(8, 1024, 24)).astype(int))
    log_values = np.log(ordered)
    hill, moment, pickands = [], [], []
    for k in levels:
        excess = log_values[:k] - log_values[k]
        m1, m2 = np.mean(excess), np.mean(excess**2)
        hill.append(m1)
        denominator = 1 - m1 * m1 / m2 if m2 > 0 else 0
        moment.append(m1 + 1 - 0.5 / denominator if abs(denominator) >= 1e-10 else np.nan)
        a, b = ordered[k - 1] - ordered[2 * k - 1], ordered[2 * k - 1] - ordered[4 * k - 1]
        pickands.append(np.log(a / b) / np.log(2) if a > 0 and b > 0 else np.nan)
    for actual, expected in [
        (_hill_path(ordered, levels), hill),
        (_dedh_moment_path(ordered, levels), moment),
        (_pickands_path(ordered, levels), pickands),
    ]:
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12, equal_nan=True)


def test_native_kernel_rejects_unsafe_output_shapes_and_ranks():
    kernels = _accelerator.kernels
    if kernels is None:
        pytest.skip("Optional native extension is not built")
    prefix = np.array([[1, 0], [1, 1]], dtype=np.int64)
    weights = np.ones((1, 2), dtype=np.int64)
    for output, lo, hi in [
        (np.zeros((1, 1), dtype=np.int64), 0, 1),
        (np.zeros((1, 2), dtype=np.int64), 0, 2),
        (np.zeros((1, 2), dtype=np.int64), -1, 0),
    ]:
        with pytest.raises(ValueError):
            kernels.rank_indices(prefix, weights, lo, hi, output)
    with pytest.raises(ValueError, match="shape"):
        kernels.kde(np.ones(2), np.ones((1, 2)), np.ones((1, 4)), np.ones(1), np.ones((1, 3)))


def test_tied_upper_tail_keeps_exact_hill_selection():
    from unibm.evi.tail import estimate_hill_evi

    sample = np.r_[np.full(512, 2.5), np.ones(512)]
    fit = estimate_hill_evi(sample)
    assert fit.selected_level == 9
    assert (fit.stable_window.lo, fit.stable_window.hi) == (8, 13)
    np.testing.assert_array_equal(fit.path_xi, np.zeros(len(fit.path_xi)))


def test_dedh_narrow_upper_tail_preserves_centered_moments():
    from unibm.evi.tail import candidate_tail_counts

    rng = np.random.default_rng(17)
    sample = np.r_[np.exp(10 + rng.uniform(0, 0.01, 128)), np.ones(128)]
    ordered = np.sort(sample)[::-1]
    levels = candidate_tail_counts(len(sample))
    expected = []
    logs = np.log(ordered)
    for k in levels:
        excess = logs[:k] - logs[k]
        m1, m2 = np.mean(excess), np.mean(excess**2)
        expected.append(m1 + 1 - 0.5 / (1 - m1**2 / m2))
    np.testing.assert_allclose(
        _dedh_moment_path(ordered, levels), expected, rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize("width", [5e-5, 1e-3, 1e-2])
def test_dedh_near_singular_correction_keeps_centered_definition(width):
    rng = np.random.default_rng(17)
    sample = np.r_[np.exp(1 + rng.uniform(0, width, 128)), np.ones(128)]
    ordered = np.sort(sample)[::-1]
    excess = np.log(ordered[:128]) - np.log(ordered[128])
    m1, m2 = np.mean(excess), np.mean(excess**2)
    expected = m1 + 1 - 0.5 / (1 - m1 * m1 / m2)
    np.testing.assert_allclose(
        _dedh_moment_path(ordered, np.array([128])), [expected], rtol=1e-10, atol=1e-12
    )


def test_hill_preserves_centered_mean_across_extreme_dynamic_range():
    ordered = np.r_[1e300, np.full(16383, 2.5)]
    levels = np.array([4096, 16383])
    logs = np.log(ordered)
    expected = [np.mean(logs[:k] - logs[k]) for k in levels]
    np.testing.assert_allclose(_hill_path(ordered, levels), expected, rtol=1e-10, atol=1e-12)
