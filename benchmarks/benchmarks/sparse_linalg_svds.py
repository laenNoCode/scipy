import os
from .common import Benchmark, safe_import

with safe_import():
    from scipy.io import mmread
    from scipy.sparse.linalg import svds


class BenchSVDS(Benchmark):
    params = [
        [20, 50, 100],  # consider instead [0.01, 0.05, 0.1] of size,
        ["abb313", "illc1033", "illc1850", "qh1484", "rbs480a", "tols4000",
         "well1033", "well1850", "west0479", "west2021"],
        ['arpack', 'lobpcg', 'propack']
    ]
    param_names = ['k', 'problem', 'solver']

    def setup(self, k, problem, solver):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        datafile = os.path.join(dir_path, "svds_benchmark_files",
                                problem + ".mtx")
        self.A = mmread(datafile)

    def time_svds(self, k, problem, solver):
        # consider k = int(np.min(self.A.shape) * k)
        svds(self.A, k=k, solver=solver)