"""Independent sampling oracles and thread/batch invariance for the fast paths."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

import numpy as np
import pytest

from shared.runtime import bootstrap_thread_cap, initialize_numerical_worker
from unibm._bootstrap_sampling import draw_circular_block_bootstrap_samples
from unibm._parallel import resolve_n_threads
from unibm.ei import bootstrap_bm_ei_path
from unibm.ei.paths import _build_bm_z_paths_from_values
from unibm.evi import estimate_target_scaling
from unibm.evi.bootstrap import (
    build_block_summary_bootstrap_backbone,
    circular_block_summary_bootstrap_multi_target,
    evaluate_block_summary_bootstrap_backbone,
)
from unibm.evi._quantile_bootstrap import (
    prepare_quantile_counts,
    quantile_from_counts,
    segment_multiplicities,
)


@pytest.mark.parametrize("value", [0, -1, True, np.bool_(False), 1.2, "auto"])
def test_thread_caps_reject_ambiguous_values(value):
    with pytest.raises(ValueError, match="n_threads"):
        resolve_n_threads(value, n_tasks=8, n_obs=4096)


def test_worker_budget_does_not_modify_parent():
    assert bootstrap_thread_cap() is None
    with ProcessPoolExecutor(
        max_workers=1, mp_context=mp.get_context("spawn"), initializer=initialize_numerical_worker
    ) as pool:
        assert pool.submit(bootstrap_thread_cap).result() == 1
    assert bootstrap_thread_cap() is None
    assert resolve_n_threads(None, n_tasks=20, n_obs=365) == 1
    assert 1 <= resolve_n_threads(3, n_tasks=20, n_obs=4096) <= 3


@pytest.mark.parametrize("shape", [(4, 17), (4, 1200)])
def test_count_quantiles_match_expanded_numpy_even_at_ties_and_boundaries(shape):
    rng = np.random.default_rng(712)
    bank = np.round(rng.lognormal(size=shape), 2) + 0.01
    draws = rng.integers(0, shape[0], size=(37, shape[0]))
    weights = segment_multiplicities(draws)
    table = prepare_quantile_counts(bank, max_bytes=64 * 1024**2)
    assert table is not None
    assert prepare_quantile_counts(bank, max_bytes=0) is None
    for q in [1e-9, 0.1, 1 / 3, 0.5, 0.9, 0.95, 0.99, 1 - 1e-9]:
        expected = np.quantile(
            bank[draws].reshape(len(draws), -1), q, axis=1, method="median_unbiased"
        )
        actual = quantile_from_counts(
            table, weights, size=bank.size, quantile=q, max_bytes=64 * 1024**2
        )
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("sliding", [True, False])
@pytest.mark.parametrize("target", ["quantile", "mean", "mode"])
def test_evi_shared_draws_and_adaptive_stopping_are_thread_invariant(target, sliding):
    x = np.random.default_rng(18).pareto(2, 256) + 1
    fits = [
        estimate_target_scaling(
            x, target=target, regression="FGLS", sliding=sliding, random_state=7, n_threads=t
        )
        for t in [1, 3]
    ]
    a, b = fits
    np.testing.assert_array_equal(a.bootstrap["samples"], b.bootstrap["samples"])
    np.testing.assert_array_equal(a.cov_beta, b.cov_beta)
    assert a.confidence_interval == b.confidence_interval
    assert a.plateau_bounds == b.plateau_bounds
    assert a.bootstrap_reps_used == b.bootstrap_reps_used
    assert a.bootstrap_precision_met == b.bootstrap_precision_met
    blocks = np.array([2, 4, 8, 16])
    multi = circular_block_summary_bootstrap_multi_target(
        x, blocks, reps=17, sliding=sliding, random_state=11, n_threads=3
    )
    backbone = build_block_summary_bootstrap_backbone(
        x, blocks, reps=17, sliding=sliding, random_state=11
    )
    single = evaluate_block_summary_bootstrap_backbone(backbone, target=target, n_threads=1)
    np.testing.assert_array_equal(multi[target]["samples"], single["samples"])


@pytest.mark.parametrize(
    "base,sliding", [(b, s) for b in ["northrop", "bb"] for s in [True, False]]
)
def test_ei_batched_resampling_matches_raw_sampling_oracle(monkeypatch, base, sliding):
    import unibm.ei.bootstrap as bootstrap

    x = np.round(np.random.default_rng(17).lognormal(size=129))  # zeros and ties
    levels = np.array([2, 7, 16, 64])
    bank = draw_circular_block_bootstrap_samples(x, reps=37, block_size=7, random_state=71)
    expected = np.array(
        [
            _build_bm_z_paths_from_values(row, levels, path_keys=((base, sliding),))[base, sliding]
            for row in bank.samples
        ]
    )
    for budget in [1024, 64 * 1024**2]:
        monkeypatch.setattr(bootstrap, "BOOTSTRAP_WORKING_BYTES", budget)
        for threads in [1, 3, None]:
            observed = bootstrap_bm_ei_path(
                x,
                allow_zeros=True,
                base_path=base,
                sliding=sliding,
                block_sizes=levels,
                reps=37,
                bootstrap_block_length=7,
                random_state=71,
                n_threads=threads,
            )
            np.testing.assert_array_equal(observed["samples"], expected)


def test_concurrent_public_calls_do_not_share_rng_or_mutate_settings():
    x = np.random.default_rng(7).pareto(2, 256) + 1
    before = x.copy()

    def fit(seed):
        return estimate_target_scaling(
            x, regression="FGLS", bootstrap_reps=37, random_state=seed, n_threads=2
        ).bootstrap["samples"]

    expected = [fit(seed) for seed in [1, 2]]
    with ThreadPoolExecutor(max_workers=2) as pool:
        observed = list(pool.map(fit, [1, 2]))
    for a, b in zip(expected, observed):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(x, before)


@pytest.mark.parametrize("target", ["quantile", "mean", "mode"])
def test_public_backbone_keeps_label_order_dtype_and_batched_mode(monkeypatch, target):
    from dataclasses import replace
    import unibm.evi.bootstrap as bootstrap

    x = np.random.default_rng(21).lognormal(size=256)
    backbone = build_block_summary_bootstrap_backbone(x, np.array([2, 8, 16]), reps=37)
    # Public backbones can arrive with another dict order or storage dtype.
    banks = {b: bank.astype(np.float32) for b, bank in backbone.maxima_by_block.items()}
    changed = replace(backbone, maxima_by_block=dict(reversed(list(banks.items()))))
    expected = []
    for b in backbone.block_sizes:
        selected = banks[b].astype(float)[backbone.segment_draws].reshape(37, -1)
        if target == "quantile":
            values = np.quantile(selected, 0.95, axis=1, method="median_unbiased")
        elif target == "mean":
            values = selected.mean(axis=1)
        else:
            values = bootstrap._evaluate_mode_bootstrap_column_batched(selected)
        expected.append(np.log(values))
    monkeypatch.setattr(bootstrap, "BOOTSTRAP_WORKING_BYTES", 1024)
    observed = evaluate_block_summary_bootstrap_backbone(changed, target=target, quantile=0.95)
    np.testing.assert_array_equal(observed["samples"], np.column_stack(expected))


@pytest.mark.parametrize("q", [np.float16(0.95), np.float32(0.95), np.float64(0.95)])
def test_public_quantile_preserves_scalar_dtype(q):
    backbone = build_block_summary_bootstrap_backbone(
        np.random.default_rng(17).lognormal(size=128), np.array([2, 8, 16]), reps=5
    )
    expected = np.column_stack(
        [
            np.log(
                np.quantile(
                    bank[backbone.segment_draws].reshape(5, -1),
                    q,
                    axis=1,
                    method="median_unbiased",
                )
            )
            for bank in backbone.maxima_by_block.values()
        ]
    )
    observed = evaluate_block_summary_bootstrap_backbone(backbone, quantile=q)
    np.testing.assert_array_equal(observed["samples"], expected)


def test_mode_batches_keep_original_kernel_reduction_groups():
    from unibm.evi.bootstrap import _evaluate_mode_bootstrap_column_batched as mode

    rows = np.random.default_rng(172).lognormal(size=(17, 29))
    rows[2, 1:] = 0  # Singleton does not enter KDE.
    rows[5] = 0  # Empty row does not enter KDE.
    active = np.sum(rows > 0, axis=1) > 1
    # Tiny kernel budget exercises both row and column chunk boundaries.
    budget = 256 * 8 * 4
    expected = mode(rows, max_kernel_bytes=budget)
    observed = np.concatenate(
        [
            mode(
                rows[start : start + 3],
                max_kernel_bytes=budget,
                reduction_rows=int(active.sum()),
                reduction_offset=int(active[:start].sum()),
            )
            for start in range(0, len(rows), 3)
        ]
    )
    np.testing.assert_array_equal(observed, expected)


def test_thread_budget_obeys_cpu_affinity_and_task_count(monkeypatch):
    import unibm._parallel as parallel

    monkeypatch.setattr(parallel.os, "process_cpu_count", lambda: 8, raising=False)
    monkeypatch.setattr(parallel.os, "sched_getaffinity", lambda _: {1, 2}, raising=False)
    assert resolve_n_threads(None, n_tasks=20, n_obs=4096) == 2
    assert resolve_n_threads(8, n_tasks=1, n_obs=4096) == 1

    def unavailable(_):
        raise OSError("Affinity query is unavailable")

    monkeypatch.setattr(parallel.os, "sched_getaffinity", unavailable)
    assert resolve_n_threads(3, n_tasks=20, n_obs=4096) == 3
