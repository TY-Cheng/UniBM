"""Small runtime helpers that keep plotting imports fast and deterministic."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile


def _runtime_cache_suffix() -> str:
    """Use the OS user ID, or a username fallback, to separate cache directories."""
    if hasattr(os, "getuid"):
        return str(os.getuid())
    return os.environ.get("USERNAME") or os.environ.get("USER") or "default"


def _env_path_is_writable(path_str: str | None) -> bool:
    """Probe a cache directory by creating and removing a temporary file.

    Create missing parent directories as needed. An unset path or a filesystem
    error returns ``False``; this check can therefore create the directory it
    tests, even though the probe file is removed.
    """
    if not path_str:
        return False
    path = Path(path_str)
    probe_path: Path | None = None
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe_fd, probe_path_str = tempfile.mkstemp(prefix=".unibm-write-test-", dir=path)
        os.close(probe_fd)
        probe_path = Path(probe_path_str)
        probe_path.unlink(missing_ok=True)
        return True
    except OSError:
        if probe_path is not None:
            probe_path.unlink(missing_ok=True)
        return False


def prepare_matplotlib_env(cache_tag: str = "unibm") -> None:
    """Point Matplotlib/fontconfig caches at a writable temporary location.

    This helper may overwrite `MPLCONFIGDIR` and `XDG_CACHE_HOME` when the
    current values are missing or not writable, so plotting imports can succeed
    in restricted environments such as shared clusters or read-only home dirs.
    Call it before importing pyplot. It creates user-scoped directories under
    the system temporary directory but does not select a plotting backend.
    """
    root = Path(tempfile.gettempdir()) / f"{cache_tag}-runtime-cache-{_runtime_cache_suffix()}"
    mpl_dir = root / "matplotlib"
    xdg_dir = root / "xdg-cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    xdg_dir.mkdir(parents=True, exist_ok=True)
    if not _env_path_is_writable(os.environ.get("MPLCONFIGDIR")):
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    if not _env_path_is_writable(os.environ.get("XDG_CACHE_HOME")):
        os.environ["XDG_CACHE_HOME"] = str(xdg_dir)


__all__ = ["prepare_matplotlib_env"]
