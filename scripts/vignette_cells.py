"""Shared notebook cell builders for the generated research vignette."""

from __future__ import annotations

import textwrap


def cell_lines(text: str) -> list[str]:
    """Normalize one multiline string into notebook cell lines."""
    content = textwrap.dedent(text).strip()
    return [line + "\n" for line in content.splitlines()]


def md_cell(text: str) -> dict:
    """Build one markdown notebook cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": cell_lines(text),
    }


def code_cell(text: str) -> dict:
    """Build one code notebook cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": cell_lines(text),
    }


__all__ = ["cell_lines", "code_cell", "md_cell"]
