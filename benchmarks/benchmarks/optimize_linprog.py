"""
Benchmarks for Linear Programming
"""
from __future__ import division, print_function, absolute_import

# Import testing parameters
try:
    from scipy.optimize import linprog, OptimizeWarning
    from scipy.linalg import toeplitz
    from scipy.optimize.tests.test_linprog import lpgen_2d, magic_square
    from numpy.testing import suppress_warnings
    import numpy as np
    import os
except ImportError:
    pass

from .common import Benchmark

methods = ["revised simplex", "interior-point"]
problems = ["AFIRO", "BLEND"]


def klee_minty(D):
    A_1 = np.array([2**(i + 1) if i > 0 else 1 for i in range(D)])
    A1_ = np.zeros(D)
    A1_[0] = 1
    A_ub = toeplitz(A_1, A1_)
    b_ub = np.array([5**(i + 1) for i in range(D)])
    c = -np.array([2**(D - i - 1) for i in range(D)])
    xf = np.zeros(D)
    xf[-1] = 5**D
    obj = c @ xf
    return c, A_ub, b_ub, xf, obj


class MagicSquare(Benchmark):

    params = [
        methods,
        [(3, 1.7305505947214375), (4, 1.5485271031586025)]
    ]
    param_names = ['method', '(dimensions, objective)']

    def setup(self, meth, prob):
        dims, obj = prob
        self.A_eq, self.b_eq, self.c, numbers = magic_square(dims)

    def time_magic_square(self, meth, prob):
        dims, obj = prob
        with suppress_warnings() as sup:
            sup.filter(OptimizeWarning, "A_eq does not appear")
            res = linprog(c=self.c, A_eq=self.A_eq, b_eq=self.b_eq,
                          bounds=(0, 1), method=meth)
            np.testing.assert_allclose(obj, res.fun, rtol=1e-6, atol=1e-3)


class KleeMinty(Benchmark):

    params = [
        methods,
        [3, 6, 9]
    ]
    param_names = ['method', 'dimensions']

    def setup(self, meth, dims):
        self.c, self.A_ub, self.b_ub, self.xf, self.obj = klee_minty(dims)
        self.meth = meth

    def time_klee_minty(self, meth, dims):
        res = linprog(c=self.c, A_ub=self.A_ub, b_ub=self.b_ub, method=self.meth)
        np.testing.assert_allclose(self.obj, res.fun, rtol=1e-6, atol=1e-3)
        np.testing.assert_allclose(self.xf, res.x, rtol=1e-6, atol=1e-3)


class LpGen(Benchmark):
    params = [
        methods,
        range(20, 100, 20),
        range(20, 100, 20)
    ]
    param_names = ['method', 'm', 'n']

    def setup(self, meth, m, n):
        self.A, self.b, self.c = lpgen_2d(m, n)
        self.meth = meth

    def time_lpgen(self, meth, m, n):
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "scipy.linalg.solve\nIll-conditioned")
            linprog(c=self.c, A_ub=self.A, b_ub=self.b, method=self.meth)


class Netlib(Benchmark):
    params = [
        methods,
        problems
    ]
    param_names = ['method', 'problems']

    def setup(self, meth, prob):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        data = np.load(dir_path + "/linprog_benchmark_files/" + prob + ".npz")
        self.c = data["c"]
        self.A_eq = data["A_eq"]
        self.A_ub = data["A_ub"]
        self.b_ub = data["b_ub"]
        self.b_eq = data["b_eq"]
        self.bounds = (0, None)
        self.obj = float(data["obj"].flatten()[0])

    def time_netlib(self, meth, prob):
        res = linprog(c=self.c,
                      A_ub=self.A_ub,
                      b_ub=self.b_ub,
                      A_eq=self.A_eq,
                      b_eq=self.b_eq,
                      bounds=self.bounds,
                      method=meth)
        np.testing.assert_allclose(self.obj, res.fun)
