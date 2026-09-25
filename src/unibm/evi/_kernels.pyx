# cython: language_level=3, boundscheck=False, wraparound=False
"""Private GIL-free KDE, rank and rolling-minimum loops; no NumPy C API or OpenMP."""
from libc.stdint cimport int64_t
from libc.math cimport exp

def kde(
    const double[::1] logs, const double[:, ::1] weights,
    const double[:, ::1] grid, const double[::1] bandwidth, double[:, ::1] out,
):
    """Fuse weighted Gaussian sums; callers supply positive finite bandwidths."""
    cdef Py_ssize_t row, point, value
    cdef double z, weight
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
                out[row, point] = 0
            # Traverse contiguous grid points while preserving each point's
            # original value-summation order and skipping zero multiplicities.
            for value in range(logs.shape[0]):
                weight = weights[row, value]
                if weight != 0:
                    for point in range(grid.shape[1]):
                        z = (grid[row, point] - logs[value]) / bandwidth[row]
                        out[row, point] += exp(-0.5 * z * z) * weight


def rolling_scaled_minimum(const double[:, ::1] data, Py_ssize_t b,
                           int64_t[:, ::1] queue, double[:, ::1] out):
    """Write b times each complete rolling minimum of finite rows into out.

    The caller owns separate reusable queue/output buffers and reduces the
    output with NumPy, preserving its row-summation order. A monotone queue
    visits each observation at most twice, independently within each row.
    """
    cdef Py_ssize_t row, i, head, tail
    if b < 1 or b > data.shape[1]:
        raise ValueError("block size must lie between one and the row length")
    if (queue.shape[0] != data.shape[0] or queue.shape[1] < data.shape[1]
            or out.shape[0] != data.shape[0] or out.shape[1] < data.shape[1] - b + 1):
        raise ValueError("rolling-minimum buffer shapes do not match the input")
    with nogil:
        for row in range(data.shape[0]):
            head = 0
            tail = 0
            for i in range(data.shape[1]):
                while head < tail and queue[row, head] <= i - b:
                    head += 1
                while tail > head and data[row, queue[row, tail - 1]] >= data[row, i]:
                    tail -= 1
                queue[row, tail] = i
                tail += 1
                if i >= b - 1:
                    out[row, i - b + 1] = b * data[row, queue[row, head]]


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
