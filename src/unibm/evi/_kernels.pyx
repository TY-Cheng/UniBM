# cython: language_level=3, boundscheck=False, wraparound=False
"""Private GIL-free KDE and integer rank loops; no NumPy C API or OpenMP."""
from libc.stdint cimport int64_t
from libc.math cimport exp

def kde(
    const double[::1] logs, const double[:, ::1] weights,
    const double[:, ::1] grid, const double[::1] bandwidth, double[:, ::1] out,
):
    """Fuse weighted Gaussian sums; callers supply positive finite bandwidths."""
    cdef Py_ssize_t row, point, value
    cdef double z, total
    if (
        weights.shape[0] != grid.shape[0]
        or weights.shape[1] != logs.shape[0]
        or bandwidth.shape[0] != grid.shape[0]
    ):
        raise ValueError("KDE input shape mismatch")
    if out.shape[0] != grid.shape[0] or out.shape[1] != grid.shape[1]:
        raise ValueError("KDE output shape mismatch")
    with nogil:
        for row in range(grid.shape[0]):
            for point in range(grid.shape[1]):
                total = 0
                for value in range(logs.shape[0]):
                    if weights[row, value] != 0:
                        z = (grid[row, point] - logs[value]) / bandwidth[row]
                        total += exp(-0.5 * z * z) * weights[row, value]
                out[row, point] = total


cdef inline Py_ssize_t find_rank(
    const int64_t[:, ::1] prefix, const int64_t[:, ::1] weights,
    Py_ssize_t row, int64_t rank, Py_ssize_t lo,
) noexcept nogil:
    """Binary search one row of pooled cumulative segment counts."""
    cdef Py_ssize_t hi = prefix.shape[0], mid, segment
    cdef int64_t count
    while lo < hi:
        mid = (lo + hi) // 2
        count = 0
        for segment in range(prefix.shape[1]):
            count += prefix[mid, segment] * weights[row, segment]
        if count > rank:
            hi = mid
        else:
            lo = mid + 1
    return lo

def rank_indices(const int64_t[:, ::1] prefix, const int64_t[:, ::1] weights,
                 int64_t lower, int64_t upper, int64_t[:, ::1] out):
    """Find two order statistics for internally validated integer count tables."""
    cdef Py_ssize_t row, segment, first
    cdef int64_t count
    if prefix.shape[0] == 0 or prefix.shape[1] != weights.shape[1]:
        raise ValueError("Invalid count table dimensions")
    if out.shape[0] != weights.shape[0] or out.shape[1] != 2 or lower < 0 or upper < lower:
        raise ValueError("Invalid output dimensions or ranks")
    # Tables come from nonnegative bincount/cumsum; protect the unchecked search bound.
    for row in range(weights.shape[0]):
        count = 0
        for segment in range(prefix.shape[1]):
            if weights[row, segment] < 0:
                raise ValueError("Negative segment multiplicity")
            count += prefix[prefix.shape[0] - 1, segment] * weights[row, segment]
        if upper >= count:
            raise ValueError("Requested rank exceeds pooled sample size")
    with nogil:
        for row in range(weights.shape[0]):
            first = find_rank(prefix, weights, row, lower, 0)
            out[row, 0] = first
            count = 0
            for segment in range(prefix.shape[1]):
                count += prefix[first, segment] * weights[row, segment]
            if count > upper:
                out[row, 1] = first
            else:
                out[row, 1] = find_rank(prefix, weights, row, upper, first + 1)
