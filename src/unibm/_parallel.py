"""Small, per-call thread budgets for NumPy/SciPy bootstrap work."""

from __future__ import annotations

from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np


# This bounds individual working batches, not retained input/output or total RSS.
BOOTSTRAP_WORKING_BYTES = 64 * 1024 * 1024


def validate_n_threads(n_threads: int | None) -> None:
    """Accept automatic selection or a positive integer cap, rejecting bools."""
    if n_threads is not None and (
        isinstance(n_threads, (bool, np.bool_))
        or not isinstance(n_threads, (int, np.integer))
        or n_threads < 1
    ):
        raise ValueError("n_threads must be None or a positive integer.")


def resolve_n_threads(n_threads: int | None, *, n_tasks: int, n_obs: int) -> int:
    """Bound threads by available CPUs and work; small automatic calls stay serial.

    ``n_threads`` controls UniBM's own pool, not BLAS or an external process
    pool. Callers running independent fits concurrently should pass a cap.
    No environment variables or process-global numerical settings are changed.
    """
    validate_n_threads(n_threads)
    process_cpu_count = getattr(os, "process_cpu_count", os.cpu_count)
    available = process_cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        try:
            available = min(available, len(os.sched_getaffinity(0)))
        except OSError:
            pass  # Some restricted hosts expose affinity but cannot query it.
    # ponytail: size heuristic, not autotuning; use n_threads for host-specific tuning.
    requested = (1 if n_obs < 2048 else 8) if n_threads is None else int(n_threads)
    return max(1, min(requested, available, n_tasks))


def bootstrap_executor(n_threads: int):
    """Own a pool for one full bootstrap call, without creating a serial pool."""
    return ThreadPoolExecutor(max_workers=n_threads) if n_threads > 1 else nullcontext(None)
