# distutils: language = c++
# cython: language_level=3
'''Testing adapter for C++ implementation of Simplex algorithm.'''

cimport cython
from libcpp.memory cimport unique_ptr

cdef extern from "_revised_simplex.cpp" namespace "linprog" nogil:
    pass

cdef extern from "_revised_simplex.hpp" namespace "linprog" nogil:
    cdef cppclass RevisedSimplexResult[T]:
        unique_ptr[T[]] x
        size_t nit
        T fun
        T * get_x() # helper function for Cython to get at contents of unique_ptr

    RevisedSimplexResult[T] revised_simplex[T](
        const size_t m,
        const size_t n,
        const T * c,
        const T * A,
        const T * b,
        const T & eps1,
        const T * bfs)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def test_linprog(
        const double[::1] & c,
        const double[::1, :] & A, # Fortran order for now
        const double[::1] & b,
        const double[::1] & bfs,
        const double & eps1=10e-5):
    '''Forward arguments to C++ implementation.'''

    cdef size_t m
    cdef size_t n
    cdef size_t ii
    m = A.shape[0]
    n = A.shape[1]

    cdef RevisedSimplexResult[double] res = revised_simplex[double](
        m, n, &c[0], &A[0][0], &b[0], eps1, &bfs[0])

    return {
        'x': [res.get_x()[ii] for ii in range(n)],
        'nit': res.nit,
        'fun': res.fun,
    }
