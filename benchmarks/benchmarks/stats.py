import warnings

import numpy as np

try:
    import scipy.stats as stats
except ImportError:
    pass

from .common import Benchmark

class Anderson_KSamp(Benchmark):
    def setup(self, *args):
        self.rand = [np.random.normal(loc=i, size=1000) for i in range(3)]

    def time_anderson_ksamp(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            stats.anderson_ksamp(self.rand)


class CorrelationFunctions(Benchmark):
    param_names = ['alternative']
    params = [
        ['two-sided', 'less', 'greater']
    ]

    def setup(self, mode):
        a = np.random.rand(2,2) * 10
        self.a = a

    def time_fisher_exact(self, alternative):
        oddsratio, pvalue = stats.fisher_exact(self.a, alternative=alternative)


class InferentialStats(Benchmark):
    def setup(self):
        np.random.seed(12345678)
        self.a = stats.norm.rvs(loc=5, scale=10, size=500)
        self.b = stats.norm.rvs(loc=8, scale=10, size=20)
        self.c = stats.norm.rvs(loc=8, scale=20, size=20)

    def time_ttest_ind_same_var(self):
        # test different sized sample with variances
        stats.ttest_ind(self.a, self.b)
        stats.ttest_ind(self.a, self.b, equal_var=False)

    def time_ttest_ind_diff_var(self):
        # test different sized sample with different variances
        stats.ttest_ind(self.a, self.c)
        stats.ttest_ind(self.a, self.c, equal_var=False)


class Distribution(Benchmark):
    param_names = ['distribution', 'properties']
    params = [
        ['cauchy', 'gamma', 'beta'],
        ['pdf', 'cdf', 'rvs', 'fit']
    ]

    def setup(self, distribution, properties):
        np.random.seed(12345678)
        self.x = np.random.rand(100)

    def time_distribution(self, distribution, properties):
        if distribution == 'gamma':
            if properties == 'pdf':
                stats.gamma.pdf(self.x, a=5, loc=4, scale=10)
            elif properties == 'cdf':
                stats.gamma.cdf(self.x, a=5, loc=4, scale=10)
            elif properties == 'rvs':
                stats.gamma.rvs(size=1000, a=5, loc=4, scale=10)
            elif properties == 'fit':
                stats.gamma.fit(self.x, loc=4, scale=10)
        elif distribution == 'cauchy':
            if properties == 'pdf':
                stats.cauchy.pdf(self.x, loc=4, scale=10)
            elif properties == 'cdf':
                stats.cauchy.cdf(self.x, loc=4, scale=10)
            elif properties == 'rvs':
                stats.cauchy.rvs(size=1000, loc=4, scale=10)
            elif properties == 'fit':
                stats.cauchy.fit(self.x, loc=4, scale=10)
        elif distribution == 'beta':
            if properties == 'pdf':
                stats.beta.pdf(self.x, a=5, b=3, loc=4, scale=10)
            elif properties == 'cdf':
                stats.beta.cdf(self.x, a=5, b=3, loc=4, scale=10)
            elif properties == 'rvs':
                stats.beta.rvs(size=1000, a=5, b=3, loc=4, scale=10)
            elif properties == 'fit':
                stats.beta.fit(self.x, loc=4, scale=10)

    # Retain old benchmark results (remove this if changing the benchmark)
    time_distribution.version = "fb22ae5386501008d945783921fe44aef3f82c1dafc40cddfaccaeec38b792b0"


class DescriptiveStats(Benchmark):
    param_names = ['n_levels']
    params = [
        [10, 1000]
    ]

    def setup(self, n_levels):
        np.random.seed(12345678)
        self.levels = np.random.randint(n_levels, size=(1000, 10))

    def time_mode(self, n_levels):
        stats.mode(self.levels, axis=0)


class GaussianKDE(Benchmark):
    def setup(self):
        np.random.seed(12345678)
        n = 2000
        m1 = np.random.normal(size=n)
        m2 = np.random.normal(scale=0.5, size=n)

        xmin = m1.min()
        xmax = m1.max()
        ymin = m2.min()
        ymax = m2.max()

        X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
        self.positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([m1, m2])
        self.kernel = stats.gaussian_kde(values)

    def time_gaussian_kde_evaluate_few_points(self):
        # test gaussian_kde evaluate on a small number of points
        self.kernel(self.positions[:, :10])

    def time_gaussian_kde_evaluate_many_points(self):
        # test gaussian_kde evaluate on many points
        self.kernel(self.positions)


class GroupSampling(Benchmark):
    param_names = ['dim']
    params = [[3, 10, 50, 200]]

    def setup(self, dim):
        np.random.seed(12345678)

    def time_unitary_group(self, dim):
        stats.unitary_group.rvs(dim)

    def time_ortho_group(self, dim):
        stats.ortho_group.rvs(dim)

    def time_special_ortho_group(self, dim):
        stats.special_ortho_group.rvs(dim)


class BinnedStatisticDD(Benchmark):

    params = ["count", "sum", "mean", "min", "max", "median", "std", np.std]

    def setup(self, statistic):
        np.random.seed(12345678)
        self.inp = np.random.rand(9999).reshape(3, 3333) * 200
        self.subbin_x_edges = np.arange(0, 200, dtype=np.float32)
        self.subbin_y_edges = np.arange(0, 200, dtype=np.float64)
        self.ret = stats.binned_statistic_dd(
            [self.inp[0], self.inp[1]], self.inp[2], statistic=statistic,
            bins=[self.subbin_x_edges, self.subbin_y_edges])

    def time_binned_statistic_dd(self, statistic):
        stats.binned_statistic_dd(
            [self.inp[0], self.inp[1]], self.inp[2], statistic=statistic,
            bins=[self.subbin_x_edges, self.subbin_y_edges])

    def time_binned_statistic_dd_reuse_bin(self, statistic):
        stats.binned_statistic_dd(
            [self.inp[0], self.inp[1]], self.inp[2], statistic=statistic,
            binned_statistic_result=self.ret)


class ContinuousFitAnalyticalMLEOverride(Benchmark):
    pretty_name = "Fit Methods Overridden with Analytical MLEs"
    param_names =  ["distribution", "loc_fixed", "scale_fixed", "shape1_fixed" ]
    dists = ["pareto", "laplace"]
    shape1, shape2, loc, scale = [[True, False]] * 4

    params = [dists,  loc, scale, shape1]

    distributions = {"pareto": {"self": stats.pareto, "floc": 0, "fscale": 2,
                                "shape1_fixed": 2, "fixed_shapes":{"b": 2}},
                     "laplace": {"self": stats.laplace, "floc": 0, "fscale": 2}
                     }

    def setup(self, dist_name, loc_fixed, scale_fixed, shape1_fixed):
        self.distn = self.distributions[dist_name]["self"]
        self.shapes = {}
        self.fixed = {}
        # add fixed `loc` and `scale`
        if loc_fixed:
            self.fixed['floc'] = self.distributions[dist_name]["floc"]
        if scale_fixed:
            self.fixed['fscale'] = self.distributions[dist_name]['fscale']

        if not self.distn.shapes:
            if shape1_fixed is not False:
                # only run this bench in the case that all shapes are false
                raise NotImplementedError("has no shapes")
        else:
            self.shapes = self.distributions[dist_name]['fixed_shapes']
            if shape1_fixed:
                self.fixed['f0'] = self.distributions[dist_name]['shape1_fixed']

        self.data = self.distn.rvs(size=10000, **self.shapes, loc=10, scale=3)
        if loc_fixed and scale_fixed and (shape1_fixed if self.distn.shapes else True):
            # all parameters were fixed.
            raise NotImplementedError("all param fix")

    def time_fit(self, dist_name, shape1_fixed, loc_fixed, scale_fixed):
        self.distn.fit(self.data, **self.fixed)
