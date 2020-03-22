# distutils: language = c++
# cython: language_level=3
'''Testing adapter for C++ implementation of Simplex algorithm.'''

cimport cython
from libcpp.memory cimport unique_ptr
from cython.operator cimport dereference as dref

cimport numpy as np
import numpy as np

cdef extern from "../rs.cpp" nogil:
    pass

cdef extern from "../utils.hpp" nogil:
    cdef enum Status:
        StatusSuccess
        StatusUnbounded
        StatusLUFailure
        StatusDGETRF_illegal_value
        StatusInverseFailure
        StatusDGETRI_illegal_value
        StatusNotImplemented

    cdef cppclass LPResult:
        size_t m
        size_t n
        unique_ptr[double[]] x
        double fun
        size_t nit
        Status status

        LPResult() except +
        const double * get_x()

cdef extern from "../rs.hpp" nogil:
    LPResult revised_simplex(
        const size_t & m,
        const size_t & n,
        const double * c,
        const double * A,
        const double * b,
        const double * bfs,
        const size_t * B_tilde0)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def test_linprog(
        const double[::1] & c,
        const double[:, ::1] & A,
        const double[::1] & b,
        const double[::1] & bfs,
        size_t[::1] B_tilde0=None):
    '''Forward arguments to C++ implementation.'''

    cdef size_t m
    cdef size_t n
    if B_tilde0 is None:
        B_tilde0 = np.argwhere(bfs).flatten().astype('uint64')
    cdef size_t ii
    m = A.shape[0]
    n = A.shape[1]

    cdef LPResult res = revised_simplex(
        m, n, &c[0], &A[0][0], &b[0], &bfs[0], &B_tilde0[0])

    cdef const double * x_ptr = res.get_x()
    return {
        'x': [x_ptr[ii] for ii in range(n)],
        'nit': res.nit,
        'fun': res.fun,
    }
