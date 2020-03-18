'''Make sure revised simplex implementation is working.'''

import unittest

import numpy as np
from scipy.optimize import OptimizeResult

from test_revised_simplex import test_linprog # pylint: disable=E0611

def call_linprog(c, A, b, bfs, eps1=10e-5):
    '''Turn into numpy arrays with correct type and order.'''
    c = np.array(c).astype('double')
    A = np.array(A, order='F').astype('double')
    b = np.array(b).astype('double')
    bfs = np.array(bfs).astype('double')
    return OptimizeResult(test_linprog(c, A, b, bfs, eps1))

class TestRevisedSimplex(unittest.TestCase):
    '''Regression tests.'''

    def test_prob_revised_simplex(self):
        '''Example problem from ...'''
        c = [0, 1, 1, 1, -2, 0, 0, 0]
        A = [
            [3, 1, 0, 0, -1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0, 1, 0],
            [-3, 0, 2, 1, 5, 0, 0, 1],
        ]
        b = [1, 2, 6]
        bfs = [0, 0, 0, 0, 0, 1, 2, 6]
        res = call_linprog(c, A, b, bfs)
        print(res)


if __name__ == '__main__':
    unittest.main()
