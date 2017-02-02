from __future__ import division, print_function, absolute_import

from warnings import warn

import numpy as np
from numpy import asarray, empty, ravel, nonzero
from scipy.sparse import (isspmatrix_csc, isspmatrix_csr, isspmatrix,
                          SparseEfficiencyWarning, csc_matrix, csr_matrix)
from scipy.linalg import LinAlgError

from . import _superlu

noScikit = False
try:
    import scikits.umfpack as umfpack
except ImportError:
    noScikit = True

useUmfpack = not noScikit

__all__ = ['use_solver', 'spsolve', 'splu', 'spilu', 'factorized',
           'MatrixRankWarning', 'spsolve_triangular']


class MatrixRankWarning(UserWarning):
    pass


def use_solver(**kwargs):
    """
    Select default sparse direct solver to be used.

    Parameters
    ----------
    useUmfpack : bool, optional
        Use UMFPACK over SuperLU. Has effect only if scikits.umfpack is
        installed. Default: True

    Notes
    -----
    The default sparse solver is umfpack when available
    (scikits.umfpack is installed). This can be changed by passing
    useUmfpack = False, which then causes the always present SuperLU
    based solver to be used.

    Umfpack requires a CSR/CSC matrix to have sorted column/row indices. If
    sure that the matrix fulfills this, pass ``assumeSortedIndices=True``
    to gain some speed.

    """
    if 'useUmfpack' in kwargs:
        globals()['useUmfpack'] = kwargs['useUmfpack']

    #TODO: pass other options to scikit

def _get_umf_family(A):
    """Get umfpack family string given the sparse matrix dtype."""
    family = {'di': 'di', 'Di': 'zi', 'dl': 'dl', 'Dl': 'zl'}
    dt = A.dtype.char + A.indices.dtype.char
    return family[dt]

def spsolve(A, b, permc_spec=None, use_umfpack=True):
    """Solve the sparse linear system Ax=b, where b may be a vector or a matrix.

    Parameters
    ----------
    A : ndarray or sparse matrix
        The square matrix A will be converted into CSC or CSR form
    b : ndarray or sparse matrix
        The matrix or vector representing the right hand side of the equation.
        If a vector, b.shape must be (n,) or (n, 1).
    permc_spec : str, optional
        How to permute the columns of the matrix for sparsity preservation.
        (default: 'COLAMD')

        - ``NATURAL``: natural ordering.
        - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
        - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
        - ``COLAMD``: approximate minimum degree column ordering
    use_umfpack : bool, optional
        if True (default) then use umfpack for the solution.  This is
        only referenced if b is a vector and ``scikit-umfpack`` is installed.

    Returns
    -------
    x : ndarray or sparse matrix
        the solution of the sparse linear equation.
        If b is a vector, then x is a vector of size A.shape[1]
        If b is a matrix, then x is a matrix of size (A.shape[1], b.shape[1])

    Notes
    -----
    For solving the matrix expression AX = B, this solver assumes the resulting
    matrix X is sparse, as is often the case for very sparse inputs.  If the
    resulting X is dense, the construction of this sparse result will be
    relatively expensive.  In that case, consider converting A to a dense
    matrix and using scipy.linalg.solve or its variants.
    """
    if not (isspmatrix_csc(A) or isspmatrix_csr(A)):
        A = csc_matrix(A)
        warn('spsolve requires A be CSC or CSR matrix format',
                SparseEfficiencyWarning)

    # b is a vector only if b have shape (n,) or (n, 1)
    b_is_sparse = isspmatrix(b)
    if not b_is_sparse:
        b = asarray(b)
    b_is_vector = ((b.ndim == 1) or (b.ndim == 2 and b.shape[1] == 1))

    A.sort_indices()
    A = A.asfptype()  # upcast to a floating point format
    result_dtype = np.promote_types(A.dtype, b.dtype)
    if A.dtype != result_dtype:
        A = A.astype(result_dtype)
    if b.dtype != result_dtype:
        b = b.astype(result_dtype)

    # validate input shapes
    M, N = A.shape
    if (M != N):
        raise ValueError("matrix must be square (has shape %s)" % ((M, N),))

    if M != b.shape[0]:
        raise ValueError("matrix - rhs dimension mismatch (%s - %s)"
                         % (A.shape, b.shape[0]))

    use_umfpack = use_umfpack and useUmfpack

    if b_is_vector and use_umfpack:
        if b_is_sparse:
            b_vec = b.toarray()
        else:
            b_vec = b
        b_vec = asarray(b_vec, dtype=A.dtype).ravel()

        if noScikit:
            raise RuntimeError('Scikits.umfpack not installed.')

        if A.dtype.char not in 'dD':
            raise ValueError("convert matrix data to double, please, using"
                  " .astype(), or set linsolve.useUmfpack = False")

        umf = umfpack.UmfpackContext(_get_umf_family(A))
        x = umf.linsolve(umfpack.UMFPACK_A, A, b_vec,
                         autoTranspose=True)
    else:
        if b_is_vector and b_is_sparse:
            b = b.toarray()
            b_is_sparse = False

        if not b_is_sparse:
            if isspmatrix_csc(A):
                flag = 1  # CSC format
            else:
                flag = 0  # CSR format

            options = dict(ColPerm=permc_spec)
            x, info = _superlu.gssv(N, A.nnz, A.data, A.indices, A.indptr,
                                    b, flag, options=options)
            if info != 0:
                warn("Matrix is exactly singular", MatrixRankWarning)
                x.fill(np.nan)
            if b_is_vector:
                x = x.ravel()
        else:
            # b is sparse
            Afactsolve = factorized(A)

            if not isspmatrix_csc(b):
                warn('spsolve is more efficient when sparse b '
                     'is in the CSC matrix format', SparseEfficiencyWarning)
                b = csc_matrix(b)

            # Create a sparse output matrix by repeatedly applying
            # the sparse factorization to solve columns of b.
            data_segs = []
            row_segs = []
            col_segs = []
            for j in range(b.shape[1]):
                bj = b[:, j].A.ravel()
                xj = Afactsolve(bj)
                w = np.flatnonzero(xj)
                segment_length = w.shape[0]
                row_segs.append(w)
                col_segs.append(np.ones(segment_length, dtype=int)*j)
                data_segs.append(np.asarray(xj[w], dtype=A.dtype))
            sparse_data = np.concatenate(data_segs)
            sparse_row = np.concatenate(row_segs)
            sparse_col = np.concatenate(col_segs)
            x = A.__class__((sparse_data, (sparse_row, sparse_col)),
                           shape=b.shape, dtype=A.dtype)

    return x


def splu(A, permc_spec=None, diag_pivot_thresh=None,
         drop_tol=None, relax=None, panel_size=None, options=dict()):
    """
    Compute the LU decomposition of a sparse, square matrix.

    Parameters
    ----------
    A : sparse matrix
        Sparse matrix to factorize. Should be in CSR or CSC format.
    permc_spec : str, optional
        How to permute the columns of the matrix for sparsity preservation.
        (default: 'COLAMD')

        - ``NATURAL``: natural ordering.
        - ``MMD_ATA``: minimum degree ordering on the structure of A^T A.
        - ``MMD_AT_PLUS_A``: minimum degree ordering on the structure of A^T+A.
        - ``COLAMD``: approximate minimum degree column ordering

    diag_pivot_thresh : float, optional
        Threshold used for a diagonal entry to be an acceptable pivot.
        See SuperLU user's guide for details [1]_
    drop_tol : float, optional
        (deprecated) No effect.
    relax : int, optional
        Expert option for customizing the degree of relaxing supernodes.
        See SuperLU user's guide for details [1]_
    panel_size : int, optional
        Expert option for customizing the panel size.
        See SuperLU user's guide for details [1]_
    options : dict, optional
        Dictionary containing additional expert options to SuperLU.
        See SuperLU user guide [1]_ (section 2.4 on the 'Options' argument)
        for more details. For example, you can specify
        ``options=dict(Equil=False, IterRefine='SINGLE'))``
        to turn equilibration off and perform a single iterative refinement.

    Returns
    -------
    invA : scipy.sparse.linalg.SuperLU
        Object, which has a ``solve`` method.

    See also
    --------
    spilu : incomplete LU decomposition

    Notes
    -----
    This function uses the SuperLU library.

    References
    ----------
    .. [1] SuperLU http://crd.lbl.gov/~xiaoye/SuperLU/

    """

    if not isspmatrix_csc(A):
        A = csc_matrix(A)
        warn('splu requires CSC matrix format', SparseEfficiencyWarning)

    A.sort_indices()
    A = A.asfptype()  # upcast to a floating point format

    M, N = A.shape
    if (M != N):
        raise ValueError("can only factor square matrices")  # is this true?

    _options = dict(DiagPivotThresh=diag_pivot_thresh, ColPerm=permc_spec,
                    PanelSize=panel_size, Relax=relax)
    if options is not None:
        _options.update(options)
    return _superlu.gstrf(N, A.nnz, A.data, A.indices, A.indptr,
                          ilu=False, options=_options)


def spilu(A, drop_tol=None, fill_factor=None, drop_rule=None, permc_spec=None,
          diag_pivot_thresh=None, relax=None, panel_size=None, options=None):
    """
    Compute an incomplete LU decomposition for a sparse, square matrix.

    The resulting object is an approximation to the inverse of `A`.

    Parameters
    ----------
    A : (N, N) array_like
        Sparse matrix to factorize
    drop_tol : float, optional
        Drop tolerance (0 <= tol <= 1) for an incomplete LU decomposition.
        (default: 1e-4)
    fill_factor : float, optional
        Specifies the fill ratio upper bound (>= 1.0) for ILU. (default: 10)
    drop_rule : str, optional
        Comma-separated string of drop rules to use.
        Available rules: ``basic``, ``prows``, ``column``, ``area``,
        ``secondary``, ``dynamic``, ``interp``. (Default: ``basic,area``)

        See SuperLU documentation for details.

    Remaining other options
        Same as for `splu`

    Returns
    -------
    invA_approx : scipy.sparse.linalg.SuperLU
        Object, which has a ``solve`` method.

    See also
    --------
    splu : complete LU decomposition

    Notes
    -----
    To improve the better approximation to the inverse, you may need to
    increase `fill_factor` AND decrease `drop_tol`.

    This function uses the SuperLU library.

    """
    if not isspmatrix_csc(A):
        A = csc_matrix(A)
        warn('splu requires CSC matrix format', SparseEfficiencyWarning)

    A.sort_indices()
    A = A.asfptype()  # upcast to a floating point format

    M, N = A.shape
    if (M != N):
        raise ValueError("can only factor square matrices")  # is this true?

    _options = dict(ILU_DropRule=drop_rule, ILU_DropTol=drop_tol,
                    ILU_FillFactor=fill_factor,
                    DiagPivotThresh=diag_pivot_thresh, ColPerm=permc_spec,
                    PanelSize=panel_size, Relax=relax)
    if options is not None:
        _options.update(options)
    return _superlu.gstrf(N, A.nnz, A.data, A.indices, A.indptr,
                          ilu=True, options=_options)


def factorized(A):
    """
    Return a function for solving a sparse linear system, with A pre-factorized.

    Parameters
    ----------
    A : (N, N) array_like
        Input.

    Returns
    -------
    solve : callable
        To solve the linear system of equations given in `A`, the `solve`
        callable should be passed an ndarray of shape (N,).

    Examples
    --------
    >>> from scipy.sparse.linalg import factorized
    >>> A = np.array([[ 3. ,  2. , -1. ],
    ...               [ 2. , -2. ,  4. ],
    ...               [-1. ,  0.5, -1. ]])
    >>> solve = factorized(A) # Makes LU decomposition.
    >>> rhs1 = np.array([1, -2, 0])
    >>> solve(rhs1) # Uses the LU factors.
    array([ 1., -2., -2.])

    """
    if useUmfpack:
        if noScikit:
            raise RuntimeError('Scikits.umfpack not installed.')

        if not isspmatrix_csc(A):
            A = csc_matrix(A)
            warn('splu requires CSC matrix format', SparseEfficiencyWarning)

        A.sort_indices()
        A = A.asfptype()  # upcast to a floating point format

        if A.dtype.char not in 'dD':
            raise ValueError("convert matrix data to double, please, using"
                  " .astype(), or set linsolve.useUmfpack = False")

        umf = umfpack.UmfpackContext(_get_umf_family(A))

        # Make LU decomposition.
        umf.numeric(A)

        def solve(b):
            return umf.solve(umfpack.UMFPACK_A, A, b, autoTranspose=True)

        return solve
    else:
        return splu(A).solve


def spsolve_triangular(A, b, lower=True, copy_A=True, overwrite_b=False):
    """
    Solve the equation `A x = b` for `x`, assuming A is a triangular matrix.

    Parameters
    ----------
    A : (M, M) sparse matrix
        A sparse square triangular matrix. Should be in CSR format.
    b : (M,) or (M, N) array_like
        Right-hand side matrix in `A x = b`
    lower : bool, optional
        Whether `A` is a lower or upper triangular matrix.
        Default is lower triangular matrix.
    copy_A : bool, optional
        Whether `A` should be copied so that it remains unmodified. The indices
        of `A` are going to be sorted and zero entries going to be removed.
        Disabling gives a performance gain. Default is True.
    overwrite_b : bool, optional
        Allow overwriting data in `b`.
        Enabling gives a performance gain. Default is False.

    Returns
    -------
    x : (M,) or (M, N) ndarray
        Solution to the system `A x = b`.  Shape of return matches shape of `b`.

    Raises
    ------
    LinAlgError
        If `A` is singular or not triangular.
    ValueError
        If shape of `A` or shape of `b` do not match the requirements.

    Notes
    -----
    .. versionadded:: 0.20.0
    """

    # Check the input for correct type and format.
    if not isspmatrix_csr(A):
        warn('CSR matrix format is required. Converting to CSR matrix.',
             SparseEfficiencyWarning)
        A = csr_matrix(A)
    elif copy_A:
        A = A.copy()

    if A.shape[0] != A.shape[1]:
        raise ValueError('A must be a square matrix but its shape is {}.'.format(A.shape))

    A.eliminate_zeros()
    A.sort_indices()

    b = np.asanyarray(b)

    if b.ndim not in [1, 2]:
        raise ValueError('b must have 1 or 2 dims but its shape is {}.'.format(b.shape))
    if A.shape[0] != b.shape[0]:
        raise ValueError('The size of the dimensions of A must be equal to '
            'the size of the first dimension of b but the shape of A is '
            '{} and the shape of b is {}.'.format(A.shape, b.shape))

    # Init x as copy of b.
    if overwrite_b:
        x = b
    else:
        x = b.copy()

    # Chose forward or backward order.
    if lower:
        row_indices = range(len(b))
    else:
        row_indices = range(len(b)-1, -1, -1)

    # Fill x iteratively.
    for i in row_indices:

        # Get indices for i-th row.
        indptr_start = A.indptr[i]
        indptr_stop = A.indptr[i+1]
        if lower:
            A_diagonal_index_row_i = indptr_stop-1
            A_off_diagonal_indices_row_i = slice(indptr_start,indptr_stop-1)
        else:
            A_diagonal_index_row_i = indptr_start
            A_off_diagonal_indices_row_i = slice(indptr_start+1,indptr_stop)

        # Check regularity and triangularity of A.
        if indptr_stop <= indptr_start or A.indices[A_diagonal_index_row_i] < i:
            raise LinAlgError('A is singular: '
                '{}th diagonal is zero!'.format(i))
        if A.indices[A_diagonal_index_row_i] > i:
            raise LinAlgError('A is no triangular matrix: entry '
                '[{},{}] is not zero!'.format(i, A.indices[A_diagonal_index_row_i]))

        # Incorporate off-diagonal entries.
        A_column_indices_in_row_i = A.indices[A_off_diagonal_indices_row_i]
        A_values_in_row_i = A.data[A_off_diagonal_indices_row_i]
        x[i] -= np.dot(x[A_column_indices_in_row_i].T, A_values_in_row_i)

        # Compute i-th entry of x.
        x[i] /= A.data[A_diagonal_index_row_i]

    return x
