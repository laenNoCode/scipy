"""Constraints definition for minimize."""
from __future__ import division, print_function, absolute_import
import numpy as np
from ._hessian_update_strategy import BFGS
from ._differentiable_functions import (
    VectorFunction, LinearVectorFunction, IdentityVectorFunction)
from .optimize import OptimizeWarning
from warnings import warn
from scipy._lib._numpy_compat import suppress_warnings
from scipy.sparse import issparse

class NonlinearConstraint(object):
    """Nonlinear constraint on the variables.

    The constraint has the general inequality form::

        lb <= fun(x) <= ub

    Here the vector of independent variables x is passed as ndarray of shape
    (n,) and ``fun`` returns a vector with m components.

    It is possible to use equal bounds to represent an equality constraint or
    infinite bounds to represent a one-sided constraint.

    Parameters
    ----------
    fun : callable
        The function defining the constraint.
        The signature is ``fun(x) -> array_like, shape (m,)``.
    lb, ub : array_like
        Lower and upper bounds on the constraint. Each array must have the
        shape (m,) or be a scalar, in the latter case a bound will be the same
        for all components of the constraint. Use ``np.inf`` with an
        appropriate sign to specify a one-sided constraint.
        Set components of `lb` and `ub` equal to represent an equality
        constraint. Note that you can mix constraints of different types:
        interval, one-sided or equality, by setting different components of
        `lb` and `ub` as  necessary.
    jac : {callable,  '2-point', '3-point', 'cs'}, optional
        Method of computing the Jacobian matrix (an m-by-n matrix,
        where element (i, j) is the partial derivative of f[i] with
        respect to x[j]).  The keywords {'2-point', '3-point',
        'cs'} select a finite difference scheme for the numerical estimation.
        A callable must have the following signature:
        ``jac(x) -> {ndarray, sparse matrix}, shape (m, n)``.
        Default is '2-point'.
    hess : {callable, '2-point', '3-point', 'cs', HessianUpdateStrategy, None}, optional
        Method for computing the Hessian matrix. The keywords
        {'2-point', '3-point', 'cs'} select a finite difference scheme for
        numerical  estimation.  Alternatively, objects implementing
        `HessianUpdateStrategy` interface can be used to approximate the
        Hessian. Currently available implementations are:

            - `BFGS` (default option)
            - `SR1`

        A callable must return the Hessian matrix of ``dot(fun, v)`` and
        must have the following signature:
        ``hess(x, v) -> {LinearOperator, sparse matrix, array_like}, shape (n, n)``.
        Here ``v`` is ndarray with shape (m,) containing Lagrange multipliers.
    keep_feasible : array_like of bool, optional
        Whether to keep the constraint components feasible throughout
        iterations. A single value set this property for all components.
        Default is False. Has no effect for equality constraints.
    finite_diff_rel_step: None or array_like, optional
        Relative step size for the finite difference approximation. Default is
        None, which will select a reasonable value automatically depending
        on a finite difference scheme.
    finite_diff_jac_sparsity: {None, array_like, sparse matrix}, optional
        Defines the sparsity structure of the Jacobian matrix for finite
        difference estimation, its shape must be (m, n). If the Jacobian has
        only few non-zero elements in *each* row, providing the sparsity
        structure will greatly speed up the computations. A zero entry means
        that a corresponding element in the Jacobian is identically zero.
        If provided, forces the use of 'lsmr' trust-region solver.
        If None (default) then dense differencing will be used.

    Notes
    -----
    Finite difference schemes {'2-point', '3-point', 'cs'} may be used for
    approximating either the Jacobian or the Hessian. We, however, do not allow
    its use for approximating both simultaneously. Hence whenever the Jacobian
    is estimated via finite-differences, we require the Hessian to be estimated
    using one of the quasi-Newton strategies.

    The scheme 'cs' is potentially the most accurate, but requires the function
    to correctly handles complex inputs and be analytically continuable to the
    complex plane. The scheme '3-point' is more accurate than '2-point' but
    requires twice as many operations.
    """
    def __init__(self, fun, lb, ub, jac='2-point', hess=BFGS(),
                 keep_feasible=False, finite_diff_rel_step=None,
                 finite_diff_jac_sparsity=None):
        self.fun = fun
        self.lb = lb
        self.ub = ub
        self.finite_diff_rel_step = finite_diff_rel_step
        self.finite_diff_jac_sparsity = finite_diff_jac_sparsity
        self.jac = jac
        self.hess = hess
        self.keep_feasible = keep_feasible


class LinearConstraint(object):
    """Linear constraint on the variables.

    The constraint has the general inequality form::

        lb <= A.dot(x) <= ub

    Here the vector of independent variables x is passed as ndarray of shape
    (n,) and the matrix A has shape (m, n).

    It is possible to use equal bounds to represent an equality constraint or
    infinite bounds to represent a one-sided constraint.

    Parameters
    ----------
    A : {array_like, sparse matrix}, shape (m, n)
        Matrix defining the constraint.
    lb, ub : array_like
        Lower and upper bounds on the constraint. Each array must have the
        shape (m,) or be a scalar, in the latter case a bound will be the same
        for all components of the constraint. Use ``np.inf`` with an
        appropriate sign to specify a one-sided constraint.
        Set components of `lb` and `ub` equal to represent an equality
        constraint. Note that you can mix constraints of different types:
        interval, one-sided or equality, by setting different components of
        `lb` and `ub` as  necessary.
    keep_feasible : array_like of bool, optional
        Whether to keep the constraint components feasible throughout
        iterations. A single value set this property for all components.
        Default is False. Has no effect for equality constraints.
    """
    def __init__(self, A, lb, ub, keep_feasible=False):
        self.A = A
        self.lb = lb
        self.ub = ub
        self.keep_feasible = keep_feasible


class Bounds(object):
    """Bounds constraint on the variables.

    The constraint has the general inequality form::

        lb <= x <= ub

    It is possible to use equal bounds to represent an equality constraint or
    infinite bounds to represent a one-sided constraint.

    Parameters
    ----------
    lb, ub : array_like, optional
        Lower and upper bounds on independent variables. Each array must
        have the same size as x or be a scalar, in which case a bound will be
        the same for all the variables. Set components of `lb` and `ub` equal
        to fix a variable. Use ``np.inf`` with an appropriate sign to disable
        bounds on all or some variables. Note that you can mix constraints of
        different types: interval, one-sided or equality, by setting different
        components of `lb` and `ub` as necessary.
    keep_feasible : array_like of bool, optional
        Whether to keep the constraint components feasible throughout
        iterations. A single value set this property for all components.
        Default is False. Has no effect for equality constraints.
    """
    def __init__(self, lb, ub, keep_feasible=False):
        self.lb = lb
        self.ub = ub
        self.keep_feasible = keep_feasible

    def __repr__(self):
        if np.any(self.keep_feasible):
            return "{}({!r}, {!r}, keep_feasible={!r})".format(type(self).__name__, self.lb, self.ub, self.keep_feasible)
        else:
            return "{}({!r}, {!r})".format(type(self).__name__, self.lb, self.ub)


class PreparedConstraint(object):
    """Constraint prepared from a user defined constraint.

    On creation it will check whether a constraint definition is valid and
    the initial point is feasible. If created successfully, it will contain
    the attributes listed below.

    Parameters
    ----------
    constraint : {NonlinearConstraint, LinearConstraint`, Bounds}
        Constraint to check and prepare.
    x0 : array_like
        Initial vector of independent variables.
    sparse_jacobian : bool or None, optional
        If bool, then the Jacobian of the constraint will be converted
        to the corresponded format if necessary. If None (default), such
        conversion is not made.
    finite_diff_bounds : 2-tuple, optional
        Lower and upper bounds on the independent variables for the finite
        difference approximation, if applicable. Defaults to no bounds.

    Attributes
    ----------
    fun : {VectorFunction, LinearVectorFunction, IdentityVectorFunction}
        Function defining the constraint wrapped by one of the convenience
        classes.
    bounds : 2-tuple
        Contains lower and upper bounds for the constraints --- lb and ub.
        These are converted to ndarray and have a size equal to the number of
        the constraints.
    keep_feasible : ndarray
         Array indicating which components must be kept feasible with a size
         equal to the number of the constraints.
    """
    def __init__(self, constraint, x0, sparse_jacobian=None,
                 finite_diff_bounds=(-np.inf, np.inf)):
        if isinstance(constraint, NonlinearConstraint):
            fun = VectorFunction(constraint.fun, x0,
                                 constraint.jac, constraint.hess,
                                 constraint.finite_diff_rel_step,
                                 constraint.finite_diff_jac_sparsity,
                                 finite_diff_bounds, sparse_jacobian)
        elif isinstance(constraint, LinearConstraint):
            fun = LinearVectorFunction(constraint.A, x0, sparse_jacobian)
        elif isinstance(constraint, Bounds):
            fun = IdentityVectorFunction(x0, sparse_jacobian)
        else:
            raise ValueError("`constraint` of an unknown type is passed.")

        m = fun.m
        lb = np.asarray(constraint.lb, dtype=float)
        ub = np.asarray(constraint.ub, dtype=float)
        if lb.ndim == 0:
            lb = np.resize(lb, m)
        if ub.ndim == 0:
            ub = np.resize(ub, m)

        keep_feasible = np.asarray(constraint.keep_feasible, dtype=bool)
        if keep_feasible.ndim == 0:
            keep_feasible = np.resize(keep_feasible, m)
        if keep_feasible.shape != (m,):
            raise ValueError("`keep_feasible` has a wrong shape.")

        mask = keep_feasible & (lb != ub)
        f0 = fun.f
        if np.any(f0[mask] < lb[mask]) or np.any(f0[mask] > ub[mask]):
            raise ValueError("`x0` is infeasible with respect to some "
                             "inequality constraint with `keep_feasible` "
                             "set to True.")

        self.fun = fun
        self.bounds = (lb, ub)
        self.keep_feasible = keep_feasible

    def is_satisfied(self, x):
        """Whether a solution satisfies the constraint.

        Parameters
        ----------
        x : array-like
            Vector of independent variables.

        Returns
        -------
        satisfied : bool
            Whether the vector of independent variables satisfies the
            constraint.
        """
        with suppress_warnings() as sup:
            sup.filter(UserWarning)
            ev = self.fun.fun(np.asarray(x))
        feasible_lb = np.all(ev >= self.bounds[0])
        feasible_ub = np.all(ev <= self.bounds[1])

        return feasible_lb and feasible_ub

    def excess_violation(self, x):
        """How much the constraint is exceeded by.

        Parameters
        ----------
        x : array-like
            Vector of independent variables

        Returns
        -------
        excess : array-like
            How much the constraint is exceeded by, for each of the
            constraints specified by `PreparedConstraint.fun`.
        """
        with suppress_warnings() as sup:
            sup.filter(UserWarning)
            ev = self.fun.fun(np.asarray(x))

        excess_lb = np.maximum(self.bounds[0] - ev, 0)
        excess_ub = np.maximum(ev - self.bounds[1], 0)

        return excess_lb + excess_ub


def new_bounds_to_old(lb, ub, n):
    """Convert the new bounds representation to the old one.

    The new representation is a tuple (lb, ub) and the old one is a list
    containing n tuples, i-th containing lower and upper bound on a i-th
    variable.
    """
    lb = np.asarray(lb)
    ub = np.asarray(ub)
    if lb.ndim == 0:
        lb = np.resize(lb, n)
    if ub.ndim == 0:
        ub = np.resize(ub, n)

    lb = [x if x > -np.inf else None for x in lb]
    ub = [x if x < np.inf else None for x in ub]

    return list(zip(lb, ub))


def old_bound_to_new(bounds):
    """Convert the old bounds representation to the new one.

    The new representation is a tuple (lb, ub) and the old one is a list
    containing n tuples, i-th containing lower and upper bound on a i-th
    variable.
    """
    lb, ub = zip(*bounds)
    lb = np.array([x if x is not None else -np.inf for x in lb])
    ub = np.array([x if x is not None else np.inf for x in ub])
    return lb, ub


def strict_bounds(lb, ub, keep_feasible, n_vars):
    """Remove bounds which are not asked to be kept feasible."""
    strict_lb = np.resize(lb, n_vars).astype(float)
    strict_ub = np.resize(ub, n_vars).astype(float)
    keep_feasible = np.resize(keep_feasible, n_vars)
    strict_lb[~keep_feasible] = -np.inf
    strict_ub[~keep_feasible] = np.inf
    return strict_lb, strict_ub


def new_constraint_to_old(con, x0):
    """
    Converts new-style constraint objects to old-style constraint dictionaries.
    """
    if isinstance(con, NonlinearConstraint):
        if (con.finite_diff_jac_sparsity is not None or
                con.finite_diff_rel_step is not None or
                not isinstance(con.hess, BFGS) or  # misses user specified BFGS
                con.keep_feasible):
            warn("Constraint options `finite_diff_jac_sparsity`, "
                 "`finite_diff_rel_step`, `keep_feasible`, and `hess`"
                 "are ignored by this method.", OptimizeWarning)

        fun = con.fun
        if callable(con.jac):
            jac = con.jac
        else:
            jac = None

    else:  # LinearConstraint
        if con.keep_feasible:
            warn("Constraint option `keep_feasible` is ignored by this "
                 "method.", OptimizeWarning)

        A = con.A
        if issparse(A):
            A = A.todense()
        fun = lambda x: np.dot(A, x)
        jac = lambda x: A

    # FIXME: when bugs in VectorFunction/LinearVectorFunction are worked out,
    # use pcon.fun.fun and pcon.fun.jac. Until then, get fun/jac above.
    pcon = PreparedConstraint(con, x0)
    lb, ub = pcon.bounds

    i_eq = lb == ub
    i_bound_below = np.logical_xor(lb != -np.inf, i_eq)
    i_bound_above = np.logical_xor(ub != np.inf, i_eq)
    i_unbounded = np.logical_and(lb == -np.inf, ub == np.inf)

    if np.any(i_unbounded):
        warn("At least one constraint is unbounded above and below. Such "
             "constraints are ignored.", OptimizeWarning)

    ceq = []
    if np.any(i_eq):
        def f_eq(x):
            y = np.array(fun(x)).flatten()
            return y[i_eq] - lb[i_eq]
        ceq = [{"type": "eq", "fun": f_eq}]

        if jac is not None:
            def j_eq(x):
                dy = jac(x)
                if issparse(dy):
                    dy = dy.todense()
                dy = np.atleast_2d(dy)
                return dy[i_eq, :]
            ceq[0]["jac"] = j_eq

    cineq = []
    n_bound_below = np.sum(i_bound_below)
    n_bound_above = np.sum(i_bound_above)
    if n_bound_below + n_bound_above:
        def f_ineq(x):
            y = np.zeros(n_bound_below + n_bound_above)
            y_all = np.array(fun(x)).flatten()
            y[:n_bound_below] = y_all[i_bound_below] - lb[i_bound_below]
            y[n_bound_below:] = -(y_all[i_bound_above] - ub[i_bound_above])
            return y
        cineq = [{"type": "ineq", "fun": f_ineq}]

        if jac is not None:
            def j_ineq(x):
                dy = np.zeros((n_bound_below + n_bound_above, len(x0)))
                dy_all = jac(x)
                if issparse(dy_all):
                    dy_all = dy_all.todense()
                dy_all = np.atleast_2d(dy_all)
                dy[:n_bound_below, :] = dy_all[i_bound_below]
                dy[n_bound_below:, :] = -dy_all[i_bound_above]
                return dy
            cineq[0]["jac"] = j_ineq

    old_constraints = ceq + cineq

    if len(old_constraints) > 1:
        warn("Equality and inequality constraints are specified in the same "
             "element of the constraint list. For efficient use with this "
             "method, equality and inequality constraints should be specified "
             "in separate elements of the constraint list. ", OptimizeWarning)
    return old_constraints


def old_constraint_to_new(ic, con):
    """
    Converts old-style constraint dictionaries to new-style constraint objects.
    """
    # check type
    try:
        ctype = con['type'].lower()
    except KeyError:
        raise KeyError('Constraint %d has no type defined.' % ic)
    except TypeError:
        raise TypeError('Constraints must be a sequence of dictionaries.')
    except AttributeError:
        raise TypeError("Constraint's type must be a string.")
    else:
        if ctype not in ['eq', 'ineq']:
            raise ValueError("Unknown constraint type '%s'." % con['type'])
    if 'fun' not in con:
        raise ValueError('Constraint %d has no function defined.' % ic)

    lb = 0
    if ctype == 'eq':
        ub = 0
    else:
        ub = np.inf

    jac = '2-point'
    if 'args' in con:
        args = con['args']
        fun = lambda x: con['fun'](x, *args)
        if 'jac' in con:
            jac = lambda x: con['jac'](x, *args)
    else:
        fun = con['fun']
        if 'jac' in con:
            jac = con['jac']

    return NonlinearConstraint(fun, lb, ub, jac)
