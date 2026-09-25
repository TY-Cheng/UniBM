from __future__ import annotations

import unittest
import warnings

import numpy as np

from unibm._bootstrap_sampling import (
    default_circular_bootstrap_block_size,
    draw_circular_block_bootstrap_sample,
    draw_circular_block_bootstrap_samples,
)
from unibm._window_ops import circular_sliding_window_maximum
from unibm.evi.bootstrap import (
    BlockSummaryBootstrapBackbone,
    _disjoint_block_maxima,
    _evaluate_mode_bootstrap_column_batched,
    _segment_block_maxima,
    build_block_summary_bootstrap_backbone,
    circular_block_summary_bootstrap,
    circular_block_summary_bootstrap_multi_target,
    evaluate_block_summary_bootstrap_backbone,
)
from unibm.evi.summaries import summarize_block_maxima


class EviBootstrapTests(unittest.TestCase):
    @staticmethod
    def _positive_sample(size: int = 256, seed: int = 303) -> np.ndarray:
        rs = np.random.default_rng(seed)
        return rs.pareto(2.0, size) + 1.0

    @staticmethod
    def _loop_reference(
        backbone: BlockSummaryBootstrapBackbone,
        *,
        target: str,
        quantile: float = 0.5,
    ) -> tuple[np.ndarray, int]:
        block_sizes = np.asarray(backbone.block_sizes, dtype=int)
        samples = np.full((backbone.segment_draws.shape[0], block_sizes.size), np.nan, dtype=float)
        invalid_summary_count = 0
        for rep, draw in enumerate(backbone.segment_draws):
            for idx, block_size in enumerate(block_sizes):
                maxima = backbone.maxima_by_block[int(block_size)][draw].reshape(-1)
                summary = summarize_block_maxima(maxima, target=target, quantile=quantile)
                if np.isfinite(summary) and summary > 0:
                    samples[rep, idx] = np.log(summary)
                else:
                    invalid_summary_count += 1
        return samples, invalid_summary_count

    def test_bootstrap_primitives_validate_inputs(self) -> None:
        self.assertEqual(default_circular_bootstrap_block_size(100, minimum=10), 10)
        with self.assertRaisesRegex(ValueError, "n_obs must be positive"):
            default_circular_bootstrap_block_size(0)
        with self.assertRaisesRegex(
            ValueError, "Cannot bootstrap a series without any finite observations"
        ):
            draw_circular_block_bootstrap_sample(
                [np.nan, np.nan], block_size=2, rng=np.random.default_rng(1)
            )
        with self.assertRaisesRegex(ValueError, "reps must be at least 1"):
            draw_circular_block_bootstrap_samples(self._positive_sample(), reps=0)
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            circular_block_summary_bootstrap(
                np.ones((8, 8), dtype=float),
                np.array([4, 8], dtype=int),
                reps=1,
            )

    def test_sliding_disjoint_and_segment_block_maxima(self) -> None:
        segment = np.array([1.0, 3.0, 2.0, 5.0], dtype=float)
        np.testing.assert_allclose(
            circular_sliding_window_maximum(segment, 2), np.array([3.0, 3.0, 5.0, 5.0])
        )
        np.testing.assert_allclose(_disjoint_block_maxima(segment, 2), np.array([3.0, 5.0]))
        np.testing.assert_allclose(
            _segment_block_maxima(segment, 2, sliding=True),
            circular_sliding_window_maximum(segment, 2),
        )

    def test_draw_bootstrap_sample_bank_is_deterministic(self) -> None:
        sample = self._positive_sample()
        bank_a = draw_circular_block_bootstrap_samples(sample, reps=3, random_state=11)
        bank_b = draw_circular_block_bootstrap_samples(sample, reps=3, random_state=11)
        np.testing.assert_allclose(bank_a.samples, bank_b.samples)

    def test_raw_bootstrap_lengths_are_exact_or_rejected(self) -> None:
        values = np.arange(8.0)
        for invalid in (0, -3, 9, 1.9, True, np.bool_(False), "3"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "block_size must be an integer"):
                    draw_circular_block_bootstrap_sample(
                        values, block_size=invalid, rng=np.random.default_rng(11)
                    )
                with self.assertRaisesRegex(ValueError, "block_size must be an integer"):
                    draw_circular_block_bootstrap_samples(values, reps=2, block_size=invalid)
        for length in (1, np.int64(3), 8):
            with self.subTest(length=length):
                bank = draw_circular_block_bootstrap_samples(
                    values, reps=2, block_size=length, random_state=11
                )
                self.assertEqual(bank.block_size, length)
                rng = np.random.default_rng(11)
                for row in bank.samples:
                    starts = rng.integers(
                        0, len(values), size=(len(values) + length - 1) // length
                    )
                    expected = np.concatenate(
                        [
                            np.take(values, np.arange(start, start + length), mode="wrap")
                            for start in starts
                        ]
                    )[: len(values)]
                    np.testing.assert_array_equal(row, expected)

    def test_build_and_evaluate_backbone(self) -> None:
        sample = self._positive_sample(size=512)
        block_sizes = np.array([4, 8, 16], dtype=int)
        with self.assertRaisesRegex(ValueError, "block_sizes"):
            build_block_summary_bootstrap_backbone(
                sample,
                np.array([4.0, 8.5, 16.0]),
                reps=4,
            )
        self.assertIsNone(build_block_summary_bootstrap_backbone(sample[:32], block_sizes, reps=1))
        backbone = build_block_summary_bootstrap_backbone(
            sample,
            block_sizes,
            sliding=True,
            reps=4,
            super_block_size=64,
            random_state=5,
        )
        assert backbone is not None
        evaluated = evaluate_block_summary_bootstrap_backbone(backbone, target="quantile")
        self.assertEqual(tuple(evaluated["block_sizes"]), (4, 8, 16))
        self.assertIsNotNone(evaluated["covariance"])
        self.assertEqual(evaluated["target"], "quantile")
        self.assertEqual(evaluated["quantile"], 0.5)
        self.assertIs(evaluated["sliding"], True)

    def test_default_superblock_length_is_used_without_adjustment(self) -> None:
        for size, block_sizes, expected_length in (
            (4096, [16, 64], 128),  # 2B dominates.
            (4096, [4, 16], 64),  # sqrt(N) dominates.
            (128, [4, 16, 32], 64),  # Two segments keep the 2B length.
            (192, [4, 16, 32], 64),  # Three segments keep the 2B length.
        ):
            for sliding in (True, False):
                with self.subTest(size=size, block_sizes=block_sizes, sliding=sliding):
                    backbone = build_block_summary_bootstrap_backbone(
                        self._positive_sample(size=size),
                        np.asarray(block_sizes, dtype=int),
                        sliding=sliding,
                        reps=3,
                    )
                    assert backbone is not None
                    self.assertEqual(backbone.super_block_size, expected_length)
                    self.assertEqual(backbone.segment_draws.shape, (3, size // expected_length))

    def test_default_superblock_length_requires_two_complete_segments(self) -> None:
        values = self._positive_sample(size=128)
        for largest_block in (33, 65):  # L0 leaves one segment or exceeds N.
            for sliding in (True, False):
                with self.subTest(largest_block=largest_block, sliding=sliding):
                    levels = np.array([4, 16, largest_block])
                    self.assertIsNone(
                        build_block_summary_bootstrap_backbone(
                            values, levels, sliding=sliding, reps=3
                        )
                    )
                    result = circular_block_summary_bootstrap(
                        values, levels, sliding=sliding, reps=3
                    )
                    self.assertIsNone(result["covariance"])
                    self.assertEqual(result["samples"].shape, (0, len(levels)))

    def test_explicit_superblock_length_is_honored_with_two_or_three_segments(self) -> None:
        values = self._positive_sample(size=1000)
        levels = np.array([5, 10, 20, 40, 80])
        for length in (np.int64(400), 333):
            for sliding in (True, False):
                with self.subTest(length=length, sliding=sliding):
                    backbone = build_block_summary_bootstrap_backbone(
                        values, levels, reps=3, super_block_size=length, sliding=sliding
                    )
                    assert backbone is not None
                    self.assertEqual(backbone.super_block_size, length)
                    self.assertEqual(backbone.segment_draws.shape, (3, 1000 // length))

    def test_explicit_superblock_length_rejects_invalid_or_unsupported_values(self) -> None:
        values = self._positive_sample(size=1000)
        levels = np.array([5, 10, 20, 40, 80])
        for invalid in (0, -3, 80, 1001, 250.5, True, np.bool_(False), "400", 501, 1000):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "super_block_size must"):
                    build_block_summary_bootstrap_backbone(
                        values, levels, reps=3, super_block_size=invalid
                    )
        with self.assertRaisesRegex(ValueError, "super_block_size must"):
            circular_block_summary_bootstrap(values, levels, reps=1, super_block_size=True)

    def test_vectorized_quantile_mean_and_mode_match_loop_reference(self) -> None:
        sample = self._positive_sample(size=512, seed=77)
        block_sizes = np.array([4, 8, 16], dtype=int)
        backbone = build_block_summary_bootstrap_backbone(
            sample,
            block_sizes,
            sliding=True,
            reps=5,
            super_block_size=64,
            random_state=17,
        )
        assert backbone is not None

        for target in ("quantile", "mean", "mode"):
            expected_samples, expected_invalid = self._loop_reference(
                backbone,
                target=target,
                quantile=0.5,
            )
            evaluated = evaluate_block_summary_bootstrap_backbone(backbone, target=target)
            valid_rows = np.all(np.isfinite(expected_samples), axis=1)
            np.testing.assert_allclose(evaluated["samples"], expected_samples[valid_rows])
            self.assertEqual(evaluated["invalid_replicates"], int(np.sum(~valid_rows)))
            self.assertEqual(
                expected_invalid,
                int(np.size(expected_samples) - np.sum(np.isfinite(expected_samples))),
            )

    def test_batched_mode_column_matches_loop_reference(self) -> None:
        selected = np.array(
            [
                [1.0, 2.0, 2.5, 3.0],
                [5.0, np.nan, 7.0, 11.0],
                [4.0, 4.0, 4.0, 4.0],
                [np.nan, np.nan, np.nan, np.nan],
            ],
            dtype=float,
        )
        expected = np.array(
            [summarize_block_maxima(row, target="mode", quantile=0.5) for row in selected],
            dtype=float,
        )
        observed = _evaluate_mode_bootstrap_column_batched(selected)
        np.testing.assert_allclose(observed, expected, equal_nan=True)

    def test_batched_mode_column_matches_reference_with_tiny_memory_budget(self) -> None:
        selected = self._positive_sample(size=120, seed=91).reshape(3, 40)
        selected[1, ::7] = np.nan
        expected = np.array(
            [summarize_block_maxima(row, target="mode", quantile=0.5) for row in selected],
            dtype=float,
        )

        observed = _evaluate_mode_bootstrap_column_batched(
            selected,
            max_kernel_bytes=256 * np.dtype(float).itemsize,
        )

        np.testing.assert_allclose(observed, expected, equal_nan=True)

    def test_warning_and_multi_target_paths(self) -> None:
        backbone = BlockSummaryBootstrapBackbone(
            block_sizes=np.array([2], dtype=int),
            sliding=True,
            super_block_size=4,
            segment_draws=np.array([[0, 1], [1, 0]], dtype=int),
            maxima_by_block={2: np.zeros((2, 3), dtype=float)},
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            evaluated = evaluate_block_summary_bootstrap_backbone(backbone, target="mean")
        self.assertEqual(evaluated["samples"].shape[0], 0)
        self.assertTrue(
            any("non-positive bootstrap block summaries" in str(w.message) for w in caught)
        )

        sample = self._positive_sample(size=256, seed=9)
        block_sizes = np.array([4, 8, 16], dtype=int)
        multi = circular_block_summary_bootstrap_multi_target(
            sample,
            block_sizes,
            targets=("quantile", "mean"),
            reps=4,
            random_state=3,
        )
        self.assertEqual(set(multi), {"quantile", "mean"})
        single = circular_block_summary_bootstrap(
            sample, block_sizes, target="mode", reps=4, random_state=3
        )
        self.assertEqual(tuple(single["block_sizes"]), (4, 8, 16))


if __name__ == "__main__":
    unittest.main()
