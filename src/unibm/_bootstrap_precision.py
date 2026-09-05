"""Shared finite-R diagnostics; not a CI-coverage or anytime-valid guarantee."""

from __future__ import annotations

from collections.abc import Callable
import warnings

import numpy as np


ADAPTIVE_REPS = (128, 256, 512, 768, 1024)
MCSE_TOLERANCE = 0.10
JACK_GROUPS = 8
JACK_PARTITIONS = 2


def adaptive_covariance(
    draw: Callable[[int, np.random.Generator], np.ndarray],
    evaluate: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    *,
    random_state: int | None,
) -> dict:
    """Append paired draws until all reported targets satisfy the MCSE rule.

    ``evaluate`` re-fits the SAME observed path/window with the supplied full-grid
    covariance and returns targets plus their statistical-SE denominators. Deletion
    acts on whole bootstrap rows. Failed covariance fits remain errors, not OLS.
    """
    rng = np.random.default_rng(random_state)
    # Diagnostic randomness must not perturb the raw-bootstrap RNG prefix.
    jack_rng = np.random.default_rng(np.random.SeedSequence(random_state).spawn(1)[0])
    pieces = []
    generated = 0
    for reps in ADAPTIVE_REPS:
        rows = draw(reps - generated, rng)
        if rows.ndim != 2 or len(rows) != reps - generated or not np.all(np.isfinite(rows)):
            raise ValueError("Adaptive bootstrap requires complete finite path rows.")
        pieces.append(rows)
        generated = reps
        samples = np.concatenate(pieces)
        covariance = np.atleast_2d(np.cov(samples, rowvar=False))
        _, scales = evaluate(covariance)
        deleted = []
        for _ in range(JACK_PARTITIONS):
            for group in jack_rng.permutation(reps).reshape(JACK_GROUPS, -1):
                keep = np.ones(reps, dtype=bool)
                keep[group] = False
                targets, _ = evaluate(np.atleast_2d(np.cov(samples[keep], rowvar=False)))
                deleted.append(targets)
        grouped = np.asarray(deleted).reshape(JACK_PARTITIONS, JACK_GROUPS, -1)
        centered = grouped - grouped.mean(axis=1, keepdims=True)
        mcse = np.sqrt((JACK_GROUPS - 1) / JACK_GROUPS * np.sum(centered**2, axis=1)).mean(axis=0)
        ratio = (
            float(np.max(mcse / scales))
            if np.all(np.isfinite(mcse)) and np.all(np.isfinite(scales)) and np.all(scales > 0)
            else float("inf")
        )
        met = ratio <= MCSE_TOLERANCE
        if met or reps == ADAPTIVE_REPS[-1]:
            if not met:
                warnings.warn(
                    "Adaptive bootstrap reached R=1024 without meeting the MCSE tolerance; "
                    "the fit is retained with bootstrap_precision_met=False.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            return {
                "samples": samples,
                "covariance": covariance,
                "bootstrap_reps_policy": "adaptive",
                "bootstrap_reps_requested": ADAPTIVE_REPS[-1],
                "bootstrap_reps_used": reps,
                "bootstrap_precision_met": bool(met),
                "bootstrap_mcse": tuple(float(value) for value in mcse),
                "bootstrap_mcse_max_ratio": ratio,
            }
    raise RuntimeError("Adaptive bootstrap failed to reach its cap.")


def matching_precision_metadata(
    bootstrap: dict | None, levels: np.ndarray, shrinkage: float
) -> dict:
    """Do not claim precision for a reused covariance fitted with a different window."""
    if bootstrap is None:
        return {}
    policy = {"bootstrap_reps_policy": bootstrap.get("bootstrap_reps_policy", "fixed")}
    if "bootstrap_precision_met" not in bootstrap:
        return policy
    if bootstrap.get("bootstrap_precision_shrinkage") != shrinkage or not np.array_equal(
        bootstrap.get("bootstrap_precision_levels"), levels
    ):
        return policy
    return {
        **policy,
        **{
            key: bootstrap[key]
            for key in (
                "bootstrap_precision_met",
                "bootstrap_mcse",
                "bootstrap_mcse_max_ratio",
                "bootstrap_mcse_targets",
            )
        },
    }
