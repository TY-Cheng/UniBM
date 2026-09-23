"""Small shared runtime helpers for workflow modules."""

from __future__ import annotations

from datetime import datetime
import os


def status(prefix: str, message: str) -> None:
    """Print one stable timestamped workflow status message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}][{prefix}] {message}", flush=True)


def resolve_int_env(name: str, default: int, minimum: int = 0) -> int:
    """Read one integer environment variable with a lower bound."""
    raw = os.environ.get(name)
    if raw is None:
        return int(max(default, minimum))
    try:
        value = int(raw)
    except ValueError:
        value = default
    return int(max(value, minimum))


def resolve_bool_env(name: str, default: bool = False) -> bool:
    """Read one boolean-like environment variable."""
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


_WORKER_BOOTSTRAP_THREADS: int | None = None


def initialize_numerical_worker() -> None:
    """Give each repository-owned process one UniBM compute thread.

    This initializer is only installed on our process pools. Standalone calls
    retain automatic inner threading; external pool owners pass their own cap.
    BLAS/OpenMP keep the existing workflow environment configuration.
    """
    global _WORKER_BOOTSTRAP_THREADS
    _WORKER_BOOTSTRAP_THREADS = 1


def bootstrap_thread_cap() -> int | None:
    """Return the explicit inner budget assigned by this workflow's parent."""
    return _WORKER_BOOTSTRAP_THREADS


__all__ = [
    "resolve_bool_env",
    "resolve_int_env",
    "status",
    "bootstrap_thread_cap",
    "initialize_numerical_worker",
]
