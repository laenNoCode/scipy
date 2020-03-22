'''Make sure revised simplex implementation is working.
References
----------
.. [1] http://www.math.ubc.ca/~israel/m340/revsimp.pdf
.. [2] https://personal.utdallas.edu/~scniu/OPRE-6201/documents/LP06-Simplex-Tableau.pdf
.. [3] http://people.brunel.ac.uk/~mastjjb/jeb/or/morelp.html
.. [4] https://en.wikipedia.org/wiki/Simplex_algorithm
.. [5] http://www.math.wsu.edu/faculty/dzhang/201/Guideline%20to%20Simplex%20Method.pdf
.. [7] http://web.mit.edu/15.053/www/AMP-Chapter-02.pdf
'''

import unittest

import numpy as np
from scipy.optimize import OptimizeResult

from test_revised_simplex import test_linprog # pylint: disable=E0611

def call_linprog(c, A, b, bfs, B_tilde0=None, sense=-1):
    '''Turn into numpy arrays with correct type and order.
    Parameters
    ----------
    B_tilde0 : 1-D array or None, optional
        The indices of bfs that are active (needed if one of active is 0).
        If B_tilde0=None, it will be populated like np.argwhere(B_tilde0).
    sense : int, optional
        If sense==-1, it is a MAX problem, if sense==1 it is MIN.
    '''
    if sense not in [-1, 1]:
        raise ValueError('sense must be one of [-1, 1]')
    c = sense*np.array(c).astype('double')
    A = np.ascontiguousarray(np.array(A).astype('double'))
    b = np.array(b).astype('double')
    bfs = np.array(bfs).astype('double')
    if B_tilde0 is not None:
        B_tilde0 = np.array(B_tilde0).astype('uint64')
    res = OptimizeResult(test_linprog(c, A, b, bfs, B_tilde0))
    if sense == -1:
        res['fun'] *= -1
    return res

class TestRevisedSimplex(unittest.TestCase):
    '''Regression tests.'''

    def test_prob1(self):
        '''Example prob (2.1) from [1]_.'''
        c = [5, 4, 3, 0, 0, 0]
        A = [
            [2, 3, 1, 1, 0, 0],
            [4, 1, 2, 0, 1, 0],
            [3, 4, 2, 0, 0, 1],
        ]
        b = [5, 11, 8]
        bfs = [0, 0, 0, 5, 11, 8]
        res = call_linprog(c, A, b, bfs)

        # Compare to known solution
        x0 = [2, 0, 1, 0, 1, 0]
        z0 = 13
        self.assertEqual(x0, res['x'])
        self.assertEqual(z0, res['fun'])

    def test_prob2(self):
        '''Example prob from bottom of pg 25 in [1]_.'''
        c = [3, 2, -4, 0, 0, 0]
        A = [
            [1, 4, 0, 1, 0, 0],
            [2, 4, -2, 0, 1, 0],
            [1, 1, -2, 0, 0, 1],
        ]
        b = [5, 6, 2]
        bfs = [0, 0, 0, 5, 6, 2]
        res = call_linprog(c, A, b, bfs)
        x = [4, 0, 1, 1, 0, 0]
        fopt = 8
        self.assertEqual(x, res['x'])
        self.assertEqual(fopt, res['fun'])

    def test_prob_revised_simplex1(self):
        '''Example problem from [1]_.'''
        c = [0, 1, 1, 1, -1, 0, 0, 0]
        A = [
            [3, 1, 0, 0, -1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0, 1, 0],
            [-3, 0, 2, 1, 5, 0, 0, 1],
        ]
        b = [1, 2, 6]
        bfs = [0, 0, 0, 0, 0, 1, 2, 6]
        res = call_linprog(c, A, b, bfs)
        x0 = [0, 1, 1, 0, 0, 0, 0, 4]
        z0 = 2
        self.assertEqual(x0, res['x'])
        self.assertEqual(z0, res['fun'])

    def test_prob_revised_simplex2(self):
        '''Example from [4]_.'''
        c = [2, 3, 4, 0, 0]
        A = [
          [3, 2, 1, 1, 0],
          [2, 5, 3, 0, 1],
        ]
        b = [10, 15]
        bfs = [0, 0, 0, 10, 15]
        res = call_linprog(c, A, b, bfs)
        x0 = [0, 0, 5, 5, 0]
        self.assertEqual(x0, res['x'])

    def test_prob_revised_simplex3(self):
        '''https://staff.aub.edu.lb/~bm05/ENMG500/Set_3_revised_simplex.pdf'''
        c = [-2, -3, 0, 0]
        A = [
            [1, 1, 1, 0],
            [2, 1, 0, 1],
        ]
        b = [50, 30]
        bfs = [0, 0, 50, 30]
        res = call_linprog(c, A, b, bfs, sense=1)
        x0 = [0, 30, 20, 0]
        z0 = -90
        self.assertEqual(x0, res['x'])
        self.assertEqual(z0, res['fun'])

    def test_prob_revised_simplex4(self):
        '''Chapter 3 Example 3.4 (Nondegenerate)'''
        c = [3, 2, 0, 0]
        A = [
            [1, 1, 1, 0],
            [2, 1, 0, 1],
        ]
        b = [40, 60]
        bfs = [0, 0, 40, 60]
        res = call_linprog(c, A, b, bfs)
        x0 = [20, 20, 0, 0]
        z0 = 100
        self.assertEqual(x0, res['x'])
        self.assertEqual(z0, res['fun'])

    def test_prob_revised_simplex5(self):
        '''Chapter 3 Example 3.5 (Degenerate)'''
        c = [3, 2, 0, 0, 0]
        A = [
            [1, 1, 1, 0, 0],
            [2, 1, 0, 1, 0],
            [1, 0, 0, 0, 1],
        ]
        b = [40, 60, 30]
        bfs = [0, 0, 40, 60, 30]
        res = call_linprog(c, A, b, bfs)

    def test_prob3(self):
        '''Example prob from [2]_.'''
        c = [4, 3, 0, 0, 0, 0]
        A = [
            [2, 3, 1, 0, 0, 0],
            [-3, 2, 0, 1, 0, 0],
            [0, 2, 0, 0, 1, 0],
            [2, 1, 0, 0, 0, 1],
        ]
        b = [6, 3, 5, 4]
        bfs = [0, 0, 6, 3, 5, 4]
        res = call_linprog(c, A, b, bfs)
        x0 = [3/2, 1, 0, 11/2, 3, 0]
        z0 = 9
        self.assertEqual(x0, res['x'])
        self.assertEqual(z0, res['fun'])

    def test_1995_UG_exam(self):
        '''Example problem from [3]_.'''
        c = [13, 5, 0, 0, 0, 0]
        A = [
            [15, 7, 1, 0, 0, 0],
            [25, 45, 0, 1, 0, 0],
            [1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1],
        ]
        b = [20*60, 15*60, 37, 14]
        bfs = [0, 0] + b
        res = call_linprog(c, A, b, bfs)
        x0 = [36, 0]
        z0 = 343
        self.assertEqual(x0, res['x'][:2])
        self.assertEqual(z0, res['fun'] - 125)

    def test_1992_UG_exam(self):
        '''Example problem from [3]_.'''
        c = [3, 5, 0, 0]
        A = [
            [12, 25, 1, 0],
            [2/5, -1, 0, 1],
        ]
        b = [30*60, 0]
        bfs = [0, 0] + b
        B_tilde0 = [2, 3]
        res = call_linprog(c, A, b, bfs, B_tilde0)

        # Need to get rounded to match given exam solutions
        xres = [round(x0, 1) for x0 in res['x']]
        fres = round(sum([c0*x0 for c0, x0 in zip(c, xres)]), 1)

        x = [81.8, 32.7, 0, 0]
        fopt = 408.9
        self.assertEqual(x, xres)
        self.assertEqual(fopt, fres)

    def test_1986_UG_exam(self):
        '''Example problem from [3]_.'''
        c = [30, 10, 0, 0, 0]
        A = [
            [6, 3, 1, 0, 0],
            [3, -1, 0, 1, 0],
            [1, 1/4, 0, 0, 1],
        ]
        b = [40, 0, 4]
        bfs = [0, 0] + b
        B_tilde0 = [2, 3, 4]
        res = call_linprog(c, A, b, bfs, B_tilde0)
        dig = 6
        xres = [round(x0, dig) for x0 in res['x'][:2]]
        fres = round(res['fun'], dig)

        x = [round(4/3, dig), round(64/6, dig)]
        fopt = round(146.666666666666666666666666, dig)
        self.assertEqual(x, xres)
        self.assertEqual(fopt, fres)

    def test_wsu_example(self):
        '''Example from WSU guide [5]_.'''
        c = [3, 1, 0, 0]
        A = [
            [2, 1, 1, 0],
            [2, 3, 0, 1],
        ]
        b = [8, 12]
        bfs = [0, 0] + b
        res = call_linprog(c, A, b, bfs)

        x0 = [4, 0, 0, 4]
        z0 = 12
        self.assertEqual(x0, res['x'])
        self.assertEqual(z0, res['fun'])

    def test_simple_example(self):
        '''Simple example on pg 50 from [7]_.'''
        c = [6, 14, 13, 0, 0]
        A = [
            [1/2, 2, 1, 1, 0],
            [1, 2, 4, 0, 1],
        ]
        b = [24, 60]
        bfs = [0, 0, 0] + b
        res = call_linprog(c, A, b, bfs)

        # Round to match solution
        xres = [round(x0, 6) for x0 in res['x']]

        x = [36, 0, 6, 0, 0]
        fopt = 294
        self.assertEqual(x, xres)
        self.assertEqual(fopt, res['fun'])


if __name__ == '__main__':
    unittest.main()
