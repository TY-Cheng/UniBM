from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"


def ensure_repo_import_paths() -> None:
    """Put repo-local scripts/src on sys.path for unittest discovery modes."""
    for path in (SCRIPTS_DIR, SRC_DIR):
        resolved = str(path)
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
