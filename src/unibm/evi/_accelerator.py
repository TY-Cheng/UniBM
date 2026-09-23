"""Load optional kernels once; the NumPy implementation is always available."""

import os

try:
    if os.environ.get("UNIBM_NO_EXTENSIONS") == "1":
        raise ImportError("Native acceleration explicitly disabled")
    from . import _kernels as kernels
except ImportError:
    kernels = None
