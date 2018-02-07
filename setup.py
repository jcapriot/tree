from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize(Extension(
        "tree_ext",
        sources=["tree_ext.pyx", "tree.cpp"],
        language="c++",
        include_dirs=[np.get_include()],
    )))
