# Build with support for EXPLICIT_INVERSE or (if not defined) LU
# factorization each iteration.
#
# DEBUG flag prints out intermediate variables for inspection.

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy as np

extensions = [
    Extension(
        'test_revised_simplex',
        ['test_revised_simplex.pyx'],
        include_dirs=[np.get_include()],
        extra_link_args=['-llapack', '-lblas'],
        extra_compile_args=['-std=c++14'],
        define_macros=[
            ('EXPLICIT_INVERSE', None),
            ('DEBUG', None),
        ],
        undef_macros=[
            #'EXPLICIT_INVERSE',
            #'DEBUG'
        ],),
]

setup(
    ext_modules=cythonize(extensions),
)
