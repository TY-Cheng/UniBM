"""Independent numerical oracles for the promoted algorithm and native kernels."""

from dataclasses import asdict, replace

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
    bank[:, ::3] = 0  # Zero is an observation, including in the compressed path.
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


def test_quantile_table_budget_uses_distinct_values_and_checks_workspace():
    bank = np.resize(np.arange(7.0), (16, 64))
    budget = bank.size * 16
    table = prepare_quantile_counts(bank, max_bytes=budget)
    assert table is not None
    assert sum(array.nbytes for array in table) <= budget
    assert prepare_quantile_counts(bank, max_bytes=budget - 1) is None
    # The sorting allowance fits, but this dense table would exceed the budget.
    assert (
        prepare_quantile_counts(np.arange(bank.size).reshape(bank.shape), max_bytes=budget) is None
    )
    for invalid in (-1.0, np.nan, np.inf):
        bank[0, 0] = invalid
        assert prepare_quantile_counts(bank, max_bytes=budget) is None


@pytest.mark.parametrize("base", ["bb", "northrop"])
def test_long_ei_native_bootstrap_matches_fallback_and_threads(monkeypatch, base):
    from unibm.ei import bootstrap_bm_ei_path

    if _accelerator.kernels is None:
        pytest.skip("Optional native extension is not built")
    sample = np.round(np.random.default_rng(17).lognormal(size=4096))
    settings = dict(
        allow_zeros=True,
        base_path=base,
        sliding=True,
        block_sizes=np.array([2, 7, 16, 63, 256]),
        reps=37,
        bootstrap_block_length=7,
        random_state=71,
    )
    native = bootstrap_bm_ei_path(sample, n_threads=1, **settings)
    threaded = bootstrap_bm_ei_path(sample, n_threads=3, **settings)
    monkeypatch.setattr(_accelerator, "kernels", None)
    fallback = bootstrap_bm_ei_path(sample, n_threads=1, **settings)
    for result in (threaded, fallback):
        np.testing.assert_array_equal(result["samples"], native["samples"])
        np.testing.assert_array_equal(result["covariance"], native["covariance"])


def test_native_rolling_minimum_matches_numpy_and_checks_buffers():
    kernels = _accelerator.kernels
    if kernels is None:
        pytest.skip("Optional native extension is not built")
    data = np.round(np.random.default_rng(45).uniform(size=(3, 129)), 1)
    data.flags.writeable = False
    queue = np.empty(data.shape, dtype=np.int64)
    output = np.full(data.shape, np.nan)
    for b in (1, 2, 7, 64, 129):
        kernels.rolling_scaled_minimum(data, b, queue, output)
        expected = b * np.lib.stride_tricks.sliding_window_view(data, b, axis=1).min(axis=-1)
        np.testing.assert_array_equal(output[:, : expected.shape[1]], expected)
    for b, q, out in [
        (0, queue, output),
        (130, queue, output),
        (2, queue[:, :-1].copy(), output),
        (2, queue, output[:1]),
        (2, queue, output[:, :5].copy()),
    ]:
        with pytest.raises(ValueError):
            kernels.rolling_scaled_minimum(data, b, q, out)


@pytest.mark.parametrize("sliding", [False, True])
def test_batched_backbone_preserves_segment_boundaries_and_missing_values(sliding):
    from unibm.evi.bootstrap import build_block_summary_bootstrap_backbone

    segments = np.random.default_rng(6).lognormal(size=(4, 32))
    segments[0, 0], segments[1, -1], segments[2, 17] = np.nan, np.inf, 0
    backbone = build_block_summary_bootstrap_backbone(
        segments.ravel(), np.array([2, 7, 16]), super_block_size=32, reps=3, sliding=sliding
    )
    for b, actual in backbone.maxima_by_block.items():
        if sliding:
            wrapped = np.concatenate([segments, segments[:, : b - 1]], axis=1)
            expected = np.lib.stride_tricks.sliding_window_view(wrapped, b, axis=1).max(axis=-1)
        else:
            expected = segments[:, : 32 // b * b].reshape(4, -1, b).max(axis=-1)
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
        values, counts = prepare_mode_counts(bank, max_bytes=64 * 1024**2)
        assert np.all(np.isfinite(values) & (values > 0))
        np.testing.assert_array_equal(
            counts.sum(axis=1), np.sum(np.isfinite(bank) & (bank > 0), axis=1)
        )


def test_mode_count_budget_uses_actual_support_and_keeps_segment_membership():
    bank = np.resize([0, 2.5, 2.5, 7, np.nan, np.inf, -1], (16, 64))
    budget = bank.size * 16
    values, counts = prepare_mode_counts(bank, max_bytes=budget)
    np.testing.assert_array_equal(values, [2.5, 7])
    expected = np.column_stack([np.sum(bank == value, axis=1) for value in values])
    np.testing.assert_array_equal(counts, expected)
    assert values.nbytes + counts.nbytes <= budget
    assert prepare_mode_counts(bank, max_bytes=budget - 1) is None
    assert (
        prepare_mode_counts(np.arange(bank.size).reshape(bank.shape) + 1, max_bytes=budget) is None
    )
    empty_values, empty_counts = prepare_mode_counts(np.zeros_like(bank), max_bytes=budget)
    assert empty_values.shape == (0,)
    assert empty_counts.shape == (len(bank), 0)


@pytest.mark.parametrize("native", [False, True])
def test_mode_counts_signal_near_tied_peaks_for_expanded_retry(monkeypatch, native):
    if native and _accelerator.kernels is None:
        pytest.skip("Optional native extension is not built")
    if not native:
        monkeypatch.setattr(_accelerator, "kernels", None)
    # Two separated KDE peaks with equal height to floating-point precision.
    # Reordering the Gaussian sum can otherwise change the chosen peak by ~0.6.
    delta = 0.41021558366084954
    row = np.r_[np.expm1(np.r_[np.full(20, 0.2), np.full(30, 0.2 + delta)]), np.zeros(17)]
    selected = np.stack([row, np.zeros_like(row), np.zeros_like(row)])
    selected[2, 0] = 2.7
    values, counts = prepare_mode_counts(selected, max_bytes=64 * 1024**2)
    result = mode_from_counts(values, counts, selected=selected)
    assert np.isnan(result[0])  # Ask the owner to retain the original sum order.
    assert np.isnan(result[1])  # An empty positive sample stays invalid.
    assert result[2] == 2.7  # Singletons retain the observation, not a log round trip.


@pytest.mark.parametrize("budget", [0, 8192, 64 * 1024**2])
def test_extended_mode_tables_preserve_mixed_rows_and_chunked_retry(monkeypatch, budget):
    import unibm.evi.bootstrap as bootstrap

    delta = 0.41021558366084954
    row = np.r_[np.expm1(np.r_[np.full(20, 0.2), np.full(30, 0.2 + delta)]), np.zeros(17)]
    banks = [row[None, :], np.zeros((1, len(row))), np.zeros((1, len(row)))]
    banks[2][0, 0] = 2.7
    draws = np.zeros((37, 1), dtype=int)
    backbone = bootstrap.BlockSummaryBootstrapBackbone(
        block_sizes=np.array([2, 3, 4]),
        sliding=True,
        super_block_size=len(row),
        segment_draws=draws,
        maxima_by_block=dict(zip([2, 3, 4], banks)),
    )
    original = bootstrap._evaluate_mode_bootstrap_column_batched

    def small_kernel(selected, **kwargs):
        """Exercise the original row/column chunk grouping during each retry."""
        return original(selected, max_kernel_bytes=256 * 8 * 3, **kwargs)

    expected = np.column_stack([small_kernel(bank[draws].reshape(37, -1)) for bank in banks])
    monkeypatch.setattr(bootstrap, "BOOTSTRAP_WORKING_BYTES", budget)
    monkeypatch.setattr(bootstrap, "_evaluate_mode_bootstrap_column_batched", small_kernel)
    for threads in [1, 3]:
        with bootstrap._summary_evaluator(
            backbone, target="mode", quantile=0.5, n_threads=threads
        ) as evaluate:
            np.testing.assert_array_equal(evaluate(draws), expected)


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


@pytest.mark.parametrize("dtype", [object, np.longdouble])
def test_adaptive_cached_design_keeps_reused_plateau_numeric_conversion(dtype):
    sample = np.random.default_rng(18).pareto(2, 256) + 1
    settings = dict(regression="FGLS", random_state=7, n_threads=1)
    expected = estimate_target_scaling(sample, **settings)
    plateau = replace(expected.plateau, x=expected.plateau.x.astype(dtype))
    actual = estimate_target_scaling(sample, curve=expected.curve, plateau=plateau, **settings)
    np.testing.assert_array_equal(actual.cov_beta, expected.cov_beta)
    np.testing.assert_array_equal(actual.bootstrap["samples"], expected.bootstrap["samples"])
    assert actual.confidence_interval == expected.confidence_interval
    assert actual.bootstrap_reps_used == expected.bootstrap_reps_used
    assert actual.bootstrap_precision_met == expected.bootstrap_precision_met


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
