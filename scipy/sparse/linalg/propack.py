"""
Python wrapper for PROPACK
--------------------------

PROPACK is a collection of Fortran routines for iterative computation
of partial SVDs of large matrices or linear operators.

Based on BSD licensed pypropack project:
  http://github.com/jakevdp/pypropack
  Author: Jake Vanderplas <vanderplas@astro.washington.edu>

PROPACK source is BSD licensed, and available at
  http://soi.stanford.edu/~rmunk/PROPACK/
"""

__all__ = ['svdp']

import numpy as np
from scipy.sparse.linalg import aslinearoperator
from scipy.linalg import LinAlgError

from ._propack import _spropack
from ._propack import _dpropack
from ._propack import _cpropack
from ._propack import _zpropack


_lansvd_dict = {
    'f': _spropack.slansvd,
    'd': _dpropack.dlansvd,
    'F': _cpropack.clansvd,
    'D': _zpropack.zlansvd,
}


_lansvd_irl_dict = {
    'f': _spropack.slansvd_irl,
    'd': _dpropack.dlansvd_irl,
    'F': _cpropack.clansvd_irl,
    'D': _zpropack.zlansvd_irl,
}

_which_converter = {
    'LM': 'L',
    'SM': 'S',
}

class _AProd:
    """
    Wrapper class for linear operator

    The call signature of the __call__ method matches the callback of
    the PROPACK routines.
    """
    def __init__(self, A):
        try:
            self.A = aslinearoperator(A)
        except TypeError:
            self.A = aslinearoperator(np.asarray(A))

    def __call__(self, transa, m, n, x, y, sparm, iparm):
        if transa == 'n':
            y[:] = self.A.matvec(x)
        else:
            y[:] = self.A.rmatvec(x)

    @property
    def shape(self):
        return self.A.shape

    @property
    def dtype(self):
        try:
            return self.A.dtype
        except AttributeError:
            return self.A.matvec(np.zeros(self.A.shape[1])).dtype


def svdp(A, k, which='LM', irl_mode=False, kmax=None,
         compute_u=True, compute_v=True, v0=None, full_output=False, tol=0,
         delta=None, eta=None, anorm=0, cgs=False, elr=True, blocksize=16,
         min_relgap=0.002, shifts=100, maxiter=None):
    """
    Compute the singular value decomposition of A using PROPACK

    Parameters
    ----------
    A : array_like, sparse matrix, or LinearOperator
        Operator for which svd will be computed.  If A is a LinearOperator
        object, it must define both ``matvec`` and ``rmatvec`` methods.
    k : int
        Number of singular values/vectors to compute
    which : string (optional)
        Which singluar triplets to compute:
        - 'LM': compute triplets corresponding to the k largest singular values
        - 'SM: compute triplets corresponding to the k smallest singular values
        which='SM' requires irl=True.
        Computes largest singular values by default.
    irl_mode : boolean (optional)
        If True, then compute SVD using iterative restarts.  Default is False.
    kmax : int (optional)
        Maximal number of iterations / maximal dimension of Krylov subspace.
        default is 5 * k
    compute_u : bool (optional)
        If True (default) then compute left singular vectors u
    compute_v : bool (optional)
        If True (default) then compute right singular vectors v
    tol : float (optional)
        The desired relative accuracy for computed singular values.
        If not specified, it will be set based on machine precision.
    v0 : starting vector (optional)
        Starting vector for iterations: should be of length A.shape[0].
        If not specified, PROPACK will generate a starting vector.
    full_output : boolean (optional)
        If True, then return info and sigma_bound.  Default is False
    delta : float (optional)
        Level of orthogonality to maintain between Lanczos vectors.
        Default is set based on machine precision.
    eta : float (optional)
        Orthogonality cutoff.  During reorthogonalization, vectors with
        component larger than eta along the Lanczos vector will be purged.
        Default is set based on machine precision.
    anorm : float (optional)
        Estimate of ||A||.  Default is zero.
    cgs : boolean (optional)
        If True, reorthogonalization is done using classical Gram-Schmidt.
        If False (default), it is done using modified Gram-Schmidt.
    elr : boolean (optional)
        If True (default), then extended local orthogonality is enforced
        when obtaining singular vectors.
    blocksize : int (optional)
        If computing u or v, blocksize controls how large a fraction of the
        work is done via fast BLAS level 3 operations.  A larger blocksize
        may lead to faster computation, at the expense of greater memory
        consumption.  blocksize must be >= 1; default is 16.
    min_relgap : float (optional)
        The smallest relative gap allowed between any shift in irl mode.
        Default = 0.001.  Accessed only if irl_mode == True.
    shifts : int (optional)
       Number of shifts per restart in irl mode.  Default is 100
        Accessed only if irl_mode == True.
    maxiter : int (optional)
        Maximum number of restarts in irl mode.  Default is 1000
        Accessed only if irl_mode == True.

    Returns
    -------
    u : ndarray
        The top k left singular vectors, shape = (A.shape[0], 3),
        returned only if compute_u is True.
    sigma : ndarray
        The top k singular values, shape = (k,)
    vt : ndarray
        The top k right singular vectors, shape = (3, A.shape[1]),
        returned only if compute_v is True.
    info : integer
        convergence info, returned only if full_output is True
        - INFO = 0 : The K largest (or smallest) singular triplets were
                     computed succesfully
        - INFO = J>0, J<K : An invariant subspace of dimension J was found.
        - INFO = -1 : K singular triplets did not converge within KMAX
                      iterations.
    sigma_bound : ndarray
        the error bounds on the singular values sigma, returned only if
        full_output is True

    Examples
    --------
    >>> from scipy.sparse.linalg import svdp
    >>> A = np.random.random((10, 20))
    >>> u, sigma, vt = svdp(A, 3)
    >>> np.set_printoptions(precision=3, suppress=True)
    >>> print(abs(np.dot(u.T, u)))
    [[ 1.  0.  0.]
     [ 0.  1.  0.]
     [ 0.  0.  1.]]
    >>> print(abs(np.dot(vt, vt.T)))
    [[ 1.  0.  0.]
     [ 0.  1.  0.]
     [ 0.  0.  1.]]
    """

    which = which.upper()
    if which not in {'LM', 'SM'}:
        raise ValueError("`which` must be either 'LM' or 'SM'")
    if not irl_mode and which == 'SM':
        raise ValueError("`which`='SM' requires irl_mode=True")

    aprod = _AProd(A)
    typ = aprod.dtype.char

    try:
        lansvd_irl = _lansvd_irl_dict[typ]
        lansvd = _lansvd_dict[typ]
    except KeyError:
        # work with non-supported types using native system precision
        if np.iscomplexobj(np.empty(0, dtype=typ)):
            typ = np.dtype(complex).char
        else:
            typ = np.dtype(float).char
        lansvd_irl = _lansvd_irl_dict[typ]
        lansvd = _lansvd_dict[typ]

    m, n = aprod.shape
    if (k < 1) or (k > min(m, n)):
        raise ValueError("k must be positive and not greater than m or n")

    if kmax is None:
        kmax = 5*k
    if maxiter is None:
        maxiter = 1000

    # guard against unnecessarily large kmax
    kmax = min(m + 1, n + 1, kmax)
    if kmax < k:
        raise ValueError(
            f"kmax must be greater than or equal to k: {kmax} < {k}")

    # convert python args to fortran args
    jobu = 'y' if compute_u else 'n'
    jobv = 'y' if compute_v else 'n'

    # these will be the output arrays
    u = np.zeros((m, kmax + 1), order='F', dtype=typ)
    v = np.zeros((n, kmax), order='F', dtype=typ)

    # Specify the starting vector.  if v0 is all zero, PROPACK will generate
    # a random starting vector: the random seed cannot be controlled in that
    # case, so we'll instead use numpy to generate a random vector
    if v0 is None:
        u[:, 0] = np.random.random(m)
        if np.iscomplexobj(np.empty(0, dtype=typ)):  # complex type
            u[:, 0] += 1j * np.random.random(m)
    else:
        try:
            u[:, 0] = v0
        except:
            raise ValueError(f"v0 must be of length {m}")

    # process options for the fit
    if delta is None:
        delta = np.sqrt(np.finfo(typ).eps)
    if eta is None:
        eta = np.finfo(typ).eps ** 0.75

    if irl_mode:
        doption = np.array((delta, eta, anorm, min_relgap), dtype=typ.lower())
    else:
        doption = np.array((delta, eta, anorm), dtype=typ.lower())

    ioption = np.array((int(bool(cgs)), int(bool(elr))), dtype='i')

    # Determine lwork & liwork:
    # the required lengths are specified in the PROPACK documentation
    # BLAS-3 blocking size is 16, but docs don't specify; it's almost surely a
    # power of 2.
    blocksize = max(2, int(blocksize))
    if compute_u or compute_v:
        lwork = m + n + 9*kmax + 5*kmax*kmax + 4 + max(3*kmax*kmax + 4*kmax + 4,
                                                       blocksize*max(m, n))
        liwork = 8*kmax
    else:
        lwork = m + n + 9*kmax + 2*kmax*kmax + 4 + max(m + n, 4*kmax + 4)
        liwork = 2*kmax + 1
    work = np.empty(lwork, dtype=typ.lower())
    iwork = np.empty(liwork, dtype=np.int32)

    # dummy arguments: these are passed to aprod, and not used in this wrapper
    dparm = np.empty(1, dtype=typ.lower())
    iparm = np.empty(1, dtype=np.int32)

    if typ.isupper():
        # PROPACK documentation is unclear on the required length of zwork.
        # Use the same length Julia's wrapper uses
        # see https://github.com/JuliaSmoothOptimizers/PROPACK.jl/
        zwork = np.empty(m + n + 32*m, dtype=typ)

        if irl_mode:
            u, sigma, bnd, v, info = lansvd_irl(_which_converter[which],
                                                jobu, jobv, m, n,
                                                shifts, k, maxiter, aprod,
                                                u, v, tol, work, zwork,
                                                iwork, doption, ioption,
                                                dparm, iparm)
        else:
            u, sigma, bnd, v, info = lansvd(jobu, jobv, m, n, k, aprod, u, v,
                                            tol, work, zwork, iwork, doption,
                                            ioption, dparm, iparm)
    else:
        if irl_mode:
            u, sigma, bnd, v, info = lansvd_irl(_which_converter[which],
                                                jobu, jobv, m, n,
                                                shifts, k, maxiter, aprod,
                                                u, v, tol, work, iwork,
                                                doption, ioption, dparm,
                                                iparm)
        else:
            u, sigma, bnd, v, info = lansvd(jobu, jobv, m, n, k, aprod, u, v,
                                            tol, work, iwork, doption,
                                            ioption, dparm, iparm)

    if info > 0:
        raise LinAlgError(
            f"An invariant subspace of dimension {info} was found.")
    if info < 0:
        raise LinAlgError(
            f"k={k} singular triplets did not converge within "
            f"kmax={kmax} iterations")

    # catch corner case where A is all zeros and v is not orthogonal
    if not np.any(v):
        v = np.eye(*v.shape)

    # construct return tuple based on caller's desired output
    return {
        (0, 0, 0): sigma,
        (1, 0, 0): (u[:, :k], sigma),
        (0, 1, 0): (sigma, v[:, :k].conj().T),
        (1, 1, 0): (u[:, :k], sigma, v[:, :k].conj().T),
        (0, 0, 1): (sigma, info, bnd),
        (1, 1, 1): (u[:, :k], sigma, v[:, :k].conj().T, info, bnd),
        (0, 1, 1): (sigma, v[:, :k].conj().T, info, bnd),
        (1, 0, 1): (u[:, :k], sigma, info, bnd),
    }[(compute_u, compute_v, full_output)]
