"""Build the optional kernels; ordinary source installs need no C compiler."""

import os

from Cython.Build import cythonize
from setuptools import Extension, setup


extensions = []
if os.environ.get("UNIBM_NO_EXTENSIONS") != "1":
    # Native separators let setuptools prune generated sources on Windows too.
    extensions = cythonize(
        [Extension("unibm.evi._kernels", [os.path.join("src", "unibm", "evi", "_kernels.pyx")])],
        build_dir=os.path.join("build", "cython"),
        compiler_directives={"language_level": 3},
    )
    # cythonize replaces the Extension object and does not preserve optional.
    extensions[0].optional = True

setup(ext_modules=extensions)
