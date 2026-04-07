from __future__ import annotations

import unittest
import warnings

import numpy as np

from scripts.unibm.bootstrap import (
    BlockSummaryBootstrapBackbone,
    _disjoint_block_maxima,
    _segment_block_maxima,
    _sliding_block_maxima,
    build_block_summary_bootstrap_backbone,
    circular_block_summary_bootstrap,
    circular_block_summary_bootstrap_multi_target,
    default_circular_bootstrap_block_size,
    draw_circular_block_bootstrap_sample,
    draw_circular_block_bootstrap_samples,
    evaluate_block_summary_bootstrap_backbone,
)


class UniBmBootstrapTests(unittest.TestCase):
    @staticmethod
    def _positive_sample(size: int = 256, seed: int = 303) -> np.ndarray:
        rs = np.random.default_rng(seed)
        return rs.pareto(2.0, size) + 1.0

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

    def test_sliding_disjoint_and_segment_block_maxima(self) -> None:
        segment = np.array([1.0, 3.0, 2.0, 5.0], dtype=float)
        np.testing.assert_allclose(
            _sliding_block_maxima(segment, 2), np.array([3.0, 3.0, 5.0, 5.0])
        )
        np.testing.assert_allclose(_disjoint_block_maxima(segment, 2), np.array([3.0, 5.0]))
        np.testing.assert_allclose(
            _segment_block_maxima(segment, 2, sliding=True),
            _sliding_block_maxima(segment, 2),
        )
        np.testing.assert_allclose(
            _segment_block_maxima(segment, 2, sliding=False),
            _disjoint_block_maxima(segment, 2),
        )

    def test_draw_bootstrap_sample_bank_is_deterministic(self) -> None:
        sample = self._positive_sample()
        bank_a = draw_circular_block_bootstrap_samples(sample, reps=3, random_state=11)
        bank_b = draw_circular_block_bootstrap_samples(sample, reps=3, random_state=11)
        self.assertEqual(bank_a.block_size, bank_b.block_size)
        np.testing.assert_allclose(bank_a.samples, bank_b.samples)

    def test_build_and_evaluate_backbone(self) -> None:
        sample = self._positive_sample(size=512)
        block_sizes = np.array([4, 8, 16], dtype=int)
        self.assertIsNone(build_block_summary_bootstrap_backbone(sample[:32], block_sizes, reps=1))
        backbone = build_block_summary_bootstrap_backbone(
            sample,
            block_sizes,
            sliding=True,
            reps=4,
            super_block_size=64,
            random_state=5,
        )
        self.assertIsNotNone(backbone)
        assert backbone is not None
        evaluated = evaluate_block_summary_bootstrap_backbone(backbone, target="quantile")
        self.assertEqual(tuple(evaluated["block_sizes"]), (4, 8, 16))
        self.assertIsNotNone(evaluated["covariance"])
        self.assertEqual(evaluated["samples"].shape[1], 3)

    def test_evaluate_backbone_warns_on_nonpositive_summaries(self) -> None:
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
        self.assertIsNone(evaluated["covariance"])
        self.assertGreater(evaluated["invalid_replicates"], 0)
        self.assertTrue(
            any("non-positive bootstrap block summaries" in str(w.message) for w in caught)
        )

    def test_multi_target_and_single_target_bootstrap_helpers(self) -> None:
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
        self.assertEqual(single["samples"].shape[1], 3)
        empty = circular_block_summary_bootstrap(sample, np.array([], dtype=int), reps=1)
        self.assertEqual(empty["samples"].shape[0], 0)
        self.assertIsNone(empty["covariance"])


if __name__ == "__main__":
    unittest.main()
