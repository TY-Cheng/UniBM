"""Small validation helpers shared across UniBM modules."""

from __future__ import annotations

import warnings

import numpy as np


def as_1d_float_array(vec: np.ndarray | list[float]) -> np.ndarray:
    """Convert to a float array, rejecting scalars and multidimensional input.

    Empty arrays and non-finite values are allowed here; callers apply the
    length and support requirements of their own estimator.
    """
    arr = np.asarray(vec, dtype=float)
    if arr.ndim != 1:
        raise ValueError("Input series must be one-dimensional.")
    return arr


def warn_on_negative_values(
    vec: np.ndarray | list[float],
    *,
    context: str,
    stacklevel: int = 2,
) -> None:
    """Emit one RuntimeWarning per call if any finite observations are negative.

    ``context`` identifies the calling workflow and ``stacklevel`` controls
    the reported source location. This helper does not filter or mutate input;
    zeros and non-finite values do not trigger the warning.
    """
    arr = as_1d_float_array(vec)
    negative_count = int(np.sum(np.isfinite(arr) & (arr < 0)))
    if negative_count:
        warnings.warn(
            (
                f"{context} received {negative_count} negative observations. "
                "UniBM is designed for positive-support data; downstream positive-only "
                "steps may exclude or invalidate those values."
            ),
            RuntimeWarning,
            stacklevel=stacklevel,
        )


def warn_on_nonpositive_values(
    vec: np.ndarray | list[float],
    *,
    context: str,
    noun: str = "values",
    stacklevel: int = 2,
) -> None:
    """Report how many finite values a positive-only step will exclude.

    The warning counts both zeros and negative values. Filtering remains the
    caller's responsibility; ``context`` and ``noun`` describe that operation.
    """
    arr = as_1d_float_array(vec)
    nonpositive_count = int(np.sum(np.isfinite(arr) & (arr <= 0)))
    if nonpositive_count:
        warnings.warn(
            (
                f"{context} excluded {nonpositive_count} non-positive {noun}. "
                "This step requires strictly positive inputs."
            ),
            RuntimeWarning,
            stacklevel=stacklevel,
        )


def positive_finite_values(
    vec: np.ndarray | list[float],
    *,
    context: str,
    minimum_size: int = 0,
    stacklevel: int = 2,
) -> np.ndarray:
    """Filter to strictly positive finite observations, preserving their order.

    Warn about finite non-positive values; drop NaN and infinities silently.
    Raise ``ValueError`` if fewer than ``minimum_size`` observations survive.
    Removing entries compresses the time axis, so this helper is unsuitable
    when the original spacing must be retained for dependence estimation.
    """
    arr = as_1d_float_array(vec)
    warn_on_nonpositive_values(
        arr, context=context, noun="observations", stacklevel=stacklevel + 1
    )
    positive = arr[np.isfinite(arr) & (arr > 0)]
    if positive.size < minimum_size:
        raise ValueError(
            f"{context} requires at least {minimum_size} positive finite observations."
        )
    return positive


def validate_covariance_shrinkage(value: float) -> float:
    """Convert a scalar shrinkage weight to float and require it to lie in [0, 1].

    Reject booleans, non-finite values and unconvertible inputs with
    ``ValueError``. Zero keeps off-diagonal covariances; one removes them.
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("covariance_shrinkage must be finite and lie in [0, 1].")
    try:
        shrinkage = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("covariance_shrinkage must be finite and lie in [0, 1].") from exc
    if not np.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
        raise ValueError("covariance_shrinkage must be finite and lie in [0, 1].")
    return shrinkage


def _validated_covariance_matrix(
    covariance: np.ndarray,
    *,
    context: str,
) -> tuple[np.ndarray, float]:
    """Validate a covariance copy and return it with its mean diagonal variance.

    Require a nonempty finite square matrix, symmetry and positive
    semidefiniteness up to scale-dependent floating-point tolerances, and a
    strictly positive mean variance. Symmetrize tiny asymmetries and shift
    roundoff-sized negative eigenvalues to zero. Larger violations raise
    ``ValueError`` labelled with ``context``; the input is never modified.
    """
    try:
        cov = np.asarray(covariance, dtype=float).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a numeric square matrix.") from exc
    if cov.ndim != 2 or cov.shape[0] == 0 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"{context} must be a non-empty square matrix.")
    if not np.all(np.isfinite(cov)):
        raise ValueError(f"{context} must contain only finite values.")

    magnitude = float(np.max(np.abs(cov)))
    tolerance = 100.0 * np.finfo(float).eps * max(cov.shape[0], 1) * magnitude
    asymmetry = float(np.max(np.abs(cov - cov.T)))
    if asymmetry > tolerance:
        raise ValueError(f"{context} must be symmetric up to numerical tolerance.")
    cov = 0.5 * (cov + cov.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigen_tolerance = (
        100.0 * np.finfo(float).eps * max(cov.shape[0], 1) * float(np.max(np.abs(eigenvalues)))
    )
    minimum_eigenvalue = float(eigenvalues[0])
    if minimum_eigenvalue < -eigen_tolerance:
        raise ValueError(f"{context} must be positive semidefinite up to numerical tolerance.")
    if minimum_eigenvalue < 0.0:
        cov += np.eye(cov.shape[0]) * -minimum_eigenvalue

    scale = float(np.trace(cov) / cov.shape[0])
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{context} must have positive scale.")
    return cov, scale


def _block_label_indices(
    block_sizes: np.ndarray,
    selected_block_sizes: np.ndarray,
    *,
    context: str,
) -> np.ndarray:
    """Map selected block sizes to positions in the full covariance grid.

    Validate full-grid labels as unique finite integers of at least two.
    Selected labels are supplied by an already validated caller and their
    requested order is preserved. Missing labels raise ``ValueError``.
    """
    try:
        labels = np.asarray(block_sizes, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} block_sizes must be a numeric array.") from exc
    if labels.ndim != 1:
        raise ValueError(f"{context} block_sizes must be one-dimensional.")
    if not np.all(np.isfinite(labels)) or np.any(labels != np.floor(labels)) or np.any(labels < 2):
        raise ValueError(f"{context} block_sizes must be finite integers at least 2.")
    integer_labels = labels.astype(int)
    if np.unique(integer_labels).size != integer_labels.size:
        raise ValueError(f"{context} block_sizes must be unique.")
    selected = np.asarray(selected_block_sizes, dtype=int).reshape(-1)
    lookup = {int(label): idx for idx, label in enumerate(integer_labels)}
    missing = [int(label) for label in selected if int(label) not in lookup]
    if missing:
        raise ValueError(f"{context} block_sizes do not cover selected block sizes {missing}.")
    return np.asarray([lookup[int(label)] for label in selected], dtype=int)


def subset_covariance_by_labels(
    covariance: np.ndarray,
    block_sizes: np.ndarray,
    selected_block_sizes: np.ndarray,
    *,
    context: str,
) -> np.ndarray:
    """Validate the full covariance, then extract rows and columns by block size.

    ``block_sizes`` labels both axes of the square input matrix. Return a
    square matrix ordered by ``selected_block_sizes``, even if that order
    differs from the input. Invalid covariance entries outside the selected
    subset still cause validation to fail; they are not silently discarded.
    """
    try:
        cov = np.asarray(covariance, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be a numeric matrix.") from exc
    labels = np.asarray(block_sizes)
    if cov.shape != (labels.size, labels.size):
        raise ValueError(f"{context} shape must match its one-dimensional block_sizes labels.")
    indices = _block_label_indices(
        block_sizes,
        selected_block_sizes,
        context=context,
    )
    cov, _ = _validated_covariance_matrix(cov, context=context)
    return cov[np.ix_(indices, indices)]


def matrix_condition_number(matrix: np.ndarray) -> float:
    """Return the 2-norm condition number, or infinity on a linear-algebra error.

    NumPy linear-algebra errors, including invalid dimensionality, return
    infinity. Input conversion errors propagate to the caller.
    """
    try:
        return float(np.linalg.cond(np.asarray(matrix, dtype=float)))
    except np.linalg.LinAlgError:
        return float("inf")


def regularize_covariance(
    covariance: np.ndarray,
    *,
    covariance_shrinkage: float,
    context: str,
) -> np.ndarray:
    """Return a validated covariance shrunk toward its diagonal, plus a ridge.

    For weight a, use ``(1 - a) * covariance + a * diag(covariance)``. Then
    add ``max(mean_variance * 1e-8, 1e-12)`` to every diagonal entry to make
    the result positive definite for subsequent solves. Validation rejects
    materially indefinite or zero-scale input before regularization, so the
    ridge does not conceal an unusable covariance. The input is not modified.
    """
    shrinkage = validate_covariance_shrinkage(covariance_shrinkage)
    cov, scale = _validated_covariance_matrix(covariance, context=context)
    if shrinkage > 0.0:
        diagonal = np.diag(np.diag(cov))
        cov = (1.0 - shrinkage) * cov + shrinkage * diagonal
    ridge = max(scale * 1e-8, 1e-12)
    return cov + np.eye(cov.shape[0]) * ridge


__all__ = [
    "as_1d_float_array",
    "matrix_condition_number",
    "positive_finite_values",
    "regularize_covariance",
    "subset_covariance_by_labels",
    "validate_covariance_shrinkage",
    "warn_on_negative_values",
    "warn_on_nonpositive_values",
]
