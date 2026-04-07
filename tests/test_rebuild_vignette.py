from __future__ import annotations

import unittest

from scripts.rebuild_vignette import build_notebook


class RebuildVignetteTests(unittest.TestCase):
    def test_build_notebook_code_cells_compile(self) -> None:
        notebook = build_notebook()
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            compile(source, "<vignette-cell>", "exec")


if __name__ == "__main__":
    unittest.main()
