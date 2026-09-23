"""Build the optional kernels; ordinary source installs need no C compiler."""

import os

from Cython.Build import cythonize
from setuptools import Extension, setup


extensions = []
if os.environ.get("UNIBM_NO_EXTENSIONS") != "1":
    extensions = cythonize(
        [Extension("unibm.evi._kernels", ["src/unibm/evi/_kernels.pyx"])],
        build_dir="build/cython",
        compiler_directives={"language_level": 3},
    )
    # cythonize replaces the Extension object and does not preserve optional.
    extensions[0].optional = True

setup(ext_modules=extensions)
