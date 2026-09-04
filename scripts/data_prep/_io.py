"""Small deterministic file writers shared by provider downloads."""

from __future__ import annotations

import gzip
import io
import os
from pathlib import Path
import tempfile

import pandas as pd


def write_csv_gz_atomic(
    frame: pd.DataFrame,
    output_path: Path | str,
    **to_csv_kwargs: object,
) -> None:
    """Atomically write a byte-stable gzip-compressed CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with tmp_path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                    frame.to_csv(text, index=False, **to_csv_kwargs)
        os.replace(tmp_path, output_path)
    finally:
        tmp_path.unlink(missing_ok=True)


__all__ = ["write_csv_gz_atomic"]
