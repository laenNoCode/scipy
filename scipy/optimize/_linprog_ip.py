"""
An interior-point method for linear programming.
"""
# Author: Matt Haberland

from __future__ import print_function, division, absolute_import
import numpy as np
import scipy as sp
import scipy.sparse as sps
from warnings import warn
from scipy.linalg import LinAlgError
from .optimize import OptimizeResult, OptimizeWarning, _check_unknown_options
#from scipy.optimize import _check_unknown_options
from scipy.optimize._remove_redundancy import _remove_redundancy

def _clean_inputs(
        c,
        A_ub=None,
        b_ub=None,
        A_eq=None,
        b_eq=None,
        bounds=None):
    """
    Given user inputs for a linear programming problem, return the
    objective vector, upper bound constraints, equality constraints,
    and simple bounds in a preferred format.

    Parameters
    ----------
    c : array_like
        Coefficients of the linear objective function to be minimized.
    A_ub : array_like, optional
        2-D array which, when matrix-multiplied by x, gives the values of the
        upper-bound inequality constraints at x.
    b_ub : array_like, optional
        1-D array of values representing the upper-bound of each inequality
        constraint (row) in A_ub.
    A_eq : array_like, optional
        2-D array which, when matrix-multiplied by x, gives the values of the
        equality constraints at x.
    b_eq : array_like, optional
        1-D array of values representing the RHS of each equality constraint
        (row) in A_eq.
    bounds : sequence, optional
        ``(min, max)`` pairs for each element in ``x``, defining
        the bounds on that parameter. Use None for one of ``min`` or
        ``max`` when there is no bound in that direction. By default
        bounds are ``(0, None)`` (non-negative)
        If a sequence containing a single tuple is provided, then ``min`` and
        ``max`` will be applied to all variables in the problem.

    Returns
    -------
    c : 1-D array
        Coefficients of the linear objective function to be minimized.
    A_ub : 2-D array
        2-D array which, when matrix-multiplied by x, gives the values of the
        upper-bound inequality constraints at x.
    b_ub : 1-D array
        1-D array of values representing the upper-bound of each inequality
        constraint (row) in A_ub.
    A_eq : 2-D array
        2-D array which, when matrix-multiplied by x, gives the values of the
        equality constraints at x.
    b_eq : 1-D array
        1-D array of values representing the RHS of each equality constraint
        (row) in A_eq.
    bounds : sequence of tuples
        ``(min, max)`` pairs for each element in ``x``, defining
        the bounds on that parameter. Use None for each of ``min`` or
        ``max`` when there is no bound in that direction. By default
        bounds are ``(0, None)`` (non-negative)

    """

    try:
        if c is None:
            raise TypeError
        try:
            c = np.asarray(c, dtype=float).copy().squeeze()
        except:
            raise TypeError
        if c.size == 1:
            c = c.reshape((-1))
        n_x = len(c)
        if n_x == 0 or len(c.shape) != 1:
            raise ValueError(
                "Invalid input for linprog: c should be a 1D array; it must "
                "not have more than one non-singleton dimension")
        if not(np.isfinite(c).all()):
            raise ValueError(
                "Invalid input for linprog: c must not contain values "
                "inf, nan, or None")
    except TypeError:
        raise TypeError(
            "Invalid input for linprog: c must be a 1D array of numerical "
            "coefficients")

    try:
        try:
            A_ub = np.zeros(
                (0, n_x), dtype=float) if A_ub is None else np.asarray(
                A_ub, dtype=float).copy()
        except:
            raise TypeError
        n_ub = len(A_ub)
        if len(A_ub.shape) != 2 or A_ub.shape[1] != len(c):
            raise ValueError(
                "Invalid input for linprog: A_ub must have exactly two "
                "dimensions, and the number of columns in A_ub must be "
                "equal to the size of c ")
        if not(np.isfinite(A_ub).all()):
            raise ValueError(
                "Invalid input for linprog: A_ub must not contain values "
                "inf, nan, or None")
    except TypeError:
        raise TypeError(
            "Invalid input for linprog: A_ub must be a numerical 2D array "
            "with each row representing an upper bound inequality constraint")

    try:
        try:
            b_ub = np.array(
                [], dtype=float) if b_ub is None else np.asarray(
                b_ub, dtype=float).copy().squeeze()
        except:
            raise TypeError
        if b_ub.size == 1:
            b_ub = b_ub.reshape((-1))
        if len(b_ub.shape) != 1:
            raise ValueError(
                "Invalid input for linprog: b_ub should be a 1D array; it "
                "must not have more than one non-singleton dimension")
        if len(b_ub) != n_ub:
            raise ValueError(
                "Invalid input for linprog: The number of rows in A_ub must "
                "be equal to the number of values in b_ub")
        if not(np.isfinite(b_ub).all()):
            raise ValueError(
                "Invalid input for linprog: b_ub must not contain values "
                "inf, nan, or None")
    except TypeError:
        raise TypeError(
            "Invalid input for linprog: b_ub must be a 1D array of "
            "numerical values, each representing the upper bound of an "
            "inequality constraint (row) in A_ub")

    try:
        try:
            A_eq = np.zeros(
                (0, n_x), dtype=float) if A_eq is None else np.asarray(
                A_eq, dtype=float).copy()
        except:
            raise TypeError
        n_eq = len(A_eq)
        if len(A_eq.shape) != 2 or A_eq.shape[1] != len(c):
            raise ValueError(
                "Invalid input for linprog: A_eq must have exactly two "
                "dimensions, and the number of columns in A_eq must be "
                "equal to the size of c ")
        if not(np.isfinite(A_eq).all()):
            raise ValueError(
                "Invalid input for linprog: A_eq must not contain values "
                "inf, nan, or None")
    except TypeError:
        raise TypeError(
            "Invalid input for linprog: A_eq must be a 2D array with each "
            "row representing an equality constraint")

    try:
        try:
            b_eq = np.array(
                [], dtype=float) if b_eq is None else np.asarray(
                b_eq, dtype=float).copy().squeeze()
        except:
            raise TypeError
        if b_eq.size == 1:
            b_eq = b_eq.reshape((-1))
        if len(b_eq.shape) != 1:
            raise ValueError(
                "Invalid input for linprog: b_eq should be a 1D array; it "
                "must not have more than one non-singleton dimension")
        if len(b_eq) != n_eq:
            raise ValueError(
                "Invalid input for linprog: the number of rows in A_eq "
                "must be equal to the number of values in b_eq")
        if not(np.isfinite(b_eq).all()):
            raise ValueError(
                "Invalid input for linprog: b_eq must not contain values "
                "inf, nan, or None")
    except TypeError:
        raise TypeError(
            "Invalid input for linprog: b_eq must be a 1D array of "
            "numerical values, each representing the right hand side of an "
            "equality constraints (row) in A_eq")

    # "If a sequence containing a single tuple is provided, then min and max
    # will be applied to all variables in the problem."
    # linprog doesn't treat this right: it didn't accept a list with one tuple
    # in it
    try:
        if isinstance(bounds, str):
            raise TypeError
        if bounds is None or len(bounds) == 0:
            bounds = [(0, None)] * n_x
        elif len(bounds) == 1:
            b = bounds[0]
            if len(b) != 2:
                raise ValueError(
                    "Invalid input for linprog: exactly one lower bound and "
                    "one upper bound must be specified for each element of x")
            bounds = [b] * n_x
        elif len(bounds) == n_x:
            try:
                len(bounds[0])
            except:
                bounds = [(bounds[0], bounds[1])] * n_x
            for i, b in enumerate(bounds):
                if len(b) != 2:
                    raise ValueError(
                        "Invalid input for linprog, bound " +
                        str(i) +
                        " " +
                        str(b) +
                        ": exactly one lower bound and one upper bound must "
                        "be specified for each element of x")
        elif len(bounds) == 2 and np.isreal(bounds[0]) \
                and np.isreal(bounds[1]):
            bounds = [(bounds[0], bounds[1])] * n_x
        else:
            raise ValueError(
                "Invalid input for linprog: exactly one lower bound and one "
                "upper bound must be specified for each element of x")

        clean_bounds = []  # also creates a copy so user's object isn't changed
        for i, b in enumerate(bounds):
            if b[0] is not None and b[1] is not None and b[0] > b[1]:
                raise ValueError(
                    "Invalid input for linprog, bound " +
                    str(i) +
                    " " +
                    str(b) +
                    ": a lower bound must be less than or equal to the "
                    "corresponding upper bound")
            if b[0] == np.inf:
                raise ValueError(
                    "Invalid input for linprog, bound " +
                    str(i) +
                    " " +
                    str(b) +
                    ": infinity is not a valid lower bound")
            if b[1] == -np.inf:
                raise ValueError(
                    "Invalid input for linprog, bound " +
                    str(i) +
                    " " +
                    str(b) +
                    ": negative infinity is not a valid upper bound")
            lb = float(b[0]) if b[0] is not None and b[0] != -np.inf else None
            ub = float(b[1]) if b[1] is not None and b[1] != np.inf else None
            clean_bounds.append((lb, ub))
        bounds = clean_bounds
    except ValueError as e:
        if "could not convert string to float" in e.args[0]:
            raise TypeError
        else:
            raise e
    except TypeError as e:
        print(e)
        raise TypeError(
            "Invalid input for linprog: bounds must be a sequence of "
            "(min,max) pairs, each defining bounds on an element of x ")

    return c, A_ub, b_ub, A_eq, b_eq, bounds


def _presolve(c, A_ub, b_ub, A_eq, b_eq, bounds):
    """
    Given inputs for a linear programming problem in preferred format, presolve
    the problem: identify trivially infeasibilities, redundancies, and
    and unboundedness, tighten bounds where possible, and eliminate fixed
    variables.

    Parameters
    ----------
    c : 1-D array
        Coefficients of the linear objective function to be minimized.
    A_ub : 2-D array
        2-D array which, when matrix-multiplied by x, gives the values of the
        upper-bound inequality constraints at x.
    b_ub : 1-D array
        1-D array of values representing the upper-bound of each inequality
        constraint (row) in A_ub.
    A_eq : 2-D array
        2-D array which, when matrix-multiplied by x, gives the values of the
        equality constraints at x.
    b_eq : 1-D array
        1-D array of values representing the RHS of each equality constraint
        (row) in A_eq.
    bounds : sequence of tuples
        ``(min, max)`` pairs for each element in ``x``, defining
        the bounds on that parameter. Use None for each of ``min`` or
        ``max`` when there is no bound in that direction.

    Returns
    -------
    c : 1-D array
        Coefficients of the linear objective function to be minimized.
    c0 : 1-D array
        Constant term in objective function due to fixed (and eliminated)
        variables.
    A_ub : 2-D array
        2-D array which, when matrix-multiplied by x, gives the values of the
        upper-bound inequality constraints at x. Unnecessary rows/columns
        have been removed.
    b_ub : 1-D array
        1-D array of values representing the upper-bound of each inequality
        constraint (row) in A_ub. Unnecessary elements have been removed.
    A_eq : 2-D array
        2-D array which, when matrix-multiplied by x, gives the values of the
        equality constraints at x. Unnecessary rows/columns have been removed.
    b_eq : 1-D array
        1-D array of values representing the RHS of each equality constraint
        (row) in A_eq. Unnecessary elements have been removed.
    bounds : sequence of tuples
        ``(min, max)`` pairs for each element in ``x``, defining
        the bounds on that parameter. Use None for each of ``min`` or
        ``max`` when there is no bound in that direction. Bounds have been
        tightened where possible.
    x : 1-D array
        Solution vector (when the solution is trivial and can be determined
        in presolve)
    undo: list of tuples
        (index, value) pairs that record the original index and fixed value
        for each variable removed from the problem
    complete: boolean
        Whether the solution is complete (solved or determined to be infeasible
        or unbounded in presolve)
    status : int
        An integer representing the exit status of the optimization:
         0 : Optimization terminated successfully
         1 : Iteration limit reached
         2 : Problem appears to be infeasible
         3 : Problem appears to be unbounded
    message : str
        A string descriptor of the exit status of the optimization.

    References
    ----------
    .. [5] Andersen, Erling D., and Knud D. Andersen. "Presolving in linear
       programming." Mathematical Programming 71.2 (1995): 221-245.

    """
    # ideas from Reference [5] by Andersen and Andersen
    # however, unlike the reference, this is performed before converting
    # problem to standard form
    # There are a few advantages:
    #  * artificial variables have not been added, so matrices are smaller
    #  * bounds have not been converted to constraints yet. (It is better to
    #    do that after presolve because presolve may adjust the simple bounds.)

    tol = 1e-9    # tolerance for equality. should this be exposed to user?

    undo = []               # record of variables eliminated from problem
    # constant term in cost function may be added if variables are eliminated
    c0 = 0
    complete = False        # complete is True if detected infeasible/unbounded
    x = np.zeros(c.shape)   # this is solution vector if completed in presolve

    status = 0              # all OK unless determined otherwise
    message = ""

    # Standard form for bounds (from _clean_inputs) is list of tuples
    # but numpy array is more convenient here
    # In retrospect, numpy array should have been the standard
    bounds = np.array(bounds)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    lb[np.equal(lb, None)] = -np.inf
    ub[np.equal(ub, None)] = np.inf
    bounds = bounds.astype(float)
    lb = lb.astype(float)
    ub = ub.astype(float)

    m_eq, n = A_eq.shape
    m_ub, n = A_ub.shape

    # zero row in equality constraints
    zero_row = np.sum(A_eq != 0, axis=1) == 0
    if np.any(zero_row):
        if np.any(
            np.logical_and(
                zero_row,
                np.abs(b_eq) > tol)):  # test_zero_row_1
            # infeasible if RHS is not zero
            status = 2
            message = "The problem is (trivially) infeasible due to a row " \
                "of zeros in the equality constraint matrix with a nonzero " \
                "corresponding constraint value."
            complete = True
            return c, c0, A_ub, b_ub, A_eq, b_eq, bounds, \
                x, undo, complete, status, message
        else:  # test_zero_row_2
            # if RHS is zero, we can eliminate this equation entirely
            A_eq = A_eq[np.logical_not(zero_row)]
            b_eq = b_eq[np.logical_not(zero_row)]

    # zero row in inequality constraints
    zero_row = np.sum(A_ub != 0, axis=1) == 0
    if np.any(zero_row):
        if np.any(np.logical_and(zero_row, b_ub < -tol)):  # test_zero_row_1
            # infeasible if RHS is less than zero (because LHS is zero)
            status = 2
            message = "The problem is (trivially) infeasible due to a row " \
                "of zeros in the equality constraint matrix with a nonzero " \
                "corresponding  constraint value."
            complete = True
            return c, c0, A_ub, b_ub, A_eq, b_eq, bounds, \
                x, undo, complete, status, message
        else:  # test_zero_row_2
            # if LHS is >= 0, we can eliminate this constraint entirely
            A_ub = A_ub[np.logical_not(zero_row)]
            b_ub = b_ub[np.logical_not(zero_row)]

    # zero column in (both) constraints
    # this indicates that a variable isn't constrained and can be removed
    A = np.vstack((A_eq, A_ub))
    if len(A > 0):
        zero_col = np.sum(A != 0, axis=0) == 0
        # variable will be at upper or lower bound, depending on objective
        x[np.logical_and(zero_col, c < 0)] = ub[
            np.logical_and(zero_col, c < 0)]
        x[np.logical_and(zero_col, c > 0)] = lb[
            np.logical_and(zero_col, c > 0)]
        if np.any(np.isinf(x)):  # if an unconstrained variable has no bound
            status = 3
            message = "If feasible, the problem is (trivially) unbounded " \
                "due  to a zero column in the constraint matrices. If you " \
                "wish to check whether the problem is infeasible, turn " \
                "presolve off."
            complete = True
            return c, c0, A_ub, b_ub, A_eq, b_eq, bounds, \
                x, undo, complete, status, message
        # variables will equal upper/lower bounds will be removed later
        lb[np.logical_and(zero_col, c < 0)] = ub[
            np.logical_and(zero_col, c < 0)]
        ub[np.logical_and(zero_col, c > 0)] = lb[
            np.logical_and(zero_col, c > 0)]

    # row singleton in equality constraints
    # this fixes a variable and removes the constraint
    singleton_row = np.sum(A_eq != 0, axis=1) == 1
    cols = np.where(A_eq[singleton_row, :])[1]
    rows = np.where(singleton_row)[0]
    if len(rows) > 0:
        for row, col in zip(rows, cols):
            val = b_eq[row] / A_eq[row, col]
            if not lb[col] - tol <= val <= ub[col] + tol:
                # infeasible if fixed value is not within bounds
                status = 2
                message = "The problem is (trivially) infeasible because a " \
                    "singleton row in the equality constraints is " \
                    "inconsistent with the bounds."
                complete = True
                return c, c0, A_ub, b_ub, A_eq, b_eq, bounds, \
                    x, undo, complete, status, message
            else:
                # sets upper and lower bounds at that fixed value - variable
                # will be removed later
                lb[col] = val
                ub[col] = val
        A_eq = A_eq[np.logical_not(singleton_row), :]
        b_eq = b_eq[np.logical_not(singleton_row)]

    # row singleton in inequality constraints
    # this indicates a simple bound and the constraint can be removed
    # simple bounds may be adjusted here
    # After all of the simple bound information is combined here, get_Abc will
    # turn the simple bounds into constraints
    singleton_row = np.sum(A_ub != 0, axis=1) == 1
    cols = np.where(A_ub[singleton_row, :])[1]
    rows = np.where(singleton_row)[0]
    if len(rows) > 0:
        for row, col in zip(rows, cols):
            val = b_ub[row] / A_ub[row, col]
            if A_ub[row, col] > 0:  # upper bound
                if val < lb[col] - tol:  # infeasible
                    complete = True
                elif val < ub[col]:  # new upper bound
                    ub[col] = val
            else:  # lower bound
                if val > ub[col] + tol:  # infeasible
                    complete = True
                elif val > lb[col]:  # new lower bound
                    lb[col] = val
            if complete:
                status = 2
                message = "The problem is (trivially) infeasible because a " \
                    "singleton row in the upper bound constraints is " \
                    "inconsistent with the bounds."
                return c, c0, A_ub, b_ub, A_eq, b_eq, bounds, \
                    x, undo, complete, status, message
        A_ub = A_ub[np.logical_not(singleton_row), :]
        b_ub = b_ub[np.logical_not(singleton_row)]

    # identical bounds indicate that variable can be removed
    i_f = np.abs(lb - ub) < tol   # indices of "fixed" variables
    i_nf = np.logical_not(i_f)  # indices of "not fixed" variables
    if np.any(i_f):
        c0 += c[i_f].dot(lb[i_f])
        b_eq = b_eq - A_eq[:, i_f].dot(lb[i_f])
        b_ub = b_ub - A_ub[:, i_f].dot(lb[i_f])
        c = c[i_nf]
        x = x[i_nf]
        A_eq = A_eq[:, i_nf]
        A_ub = A_ub[:, i_nf]
        # record of variables to be added back in
        undo = [np.where(i_f)[0], lb[i_f]]
        # don't remove these entries from bounds; they'll be used later.

    # no constraints indicates that problem is trivial
    if A_eq.size == 0 and A_ub.size == 0:
        b_eq = np.array([])
        b_ub = np.array([])
        # test_empty_constraint_1
        if c.size == 0:
            status = 0
            message = "The solution was determined in presolve as there are \
            no non-trivial constraints."
        elif np.any(np.logical_and(c < 0, ub == np.inf)) or \
                np.any(np.logical_and(c > 0, lb == -np.inf)):
                # test_no_constraints()
            status = 3
            message = "If feasible, the problem is (trivially) unbounded " \
                "because there are no constraints and at least one element " \
                "of c is negative. If you wish to check whether the " \
                "problem is infeasible, turn presolve off."
        else:  # test_empty_constraint_2
            status = 0
            message = "The solution was determined in presolve as there are " \
                "no non-trivial constraints."
        complete = True
        x[c < 0] = ub[c < 0]
        x[c > 0] = lb[c > 0]
        # if this is not the last step of presolve, should convert bounds back
        # to array and return here

    # *sigh* - convert bounds back to their standard form (list of tuples)
    # again, in retrospect, numpy array would be standard form
    lb[np.equal(lb, -np.inf)] = None
    ub[np.equal(ub, np.inf)] = None
    bounds = np.hstack((lb[:, np.newaxis], ub[:, np.newaxis]))
    bounds = bounds.tolist()
    for i, row in enumerate(bounds):
        for j, col in enumerate(row):
            if str(
                    col) == "nan":  # comparing col to float("nan") and
                                    # np.nan doesn't work. should use np.isnan
                bounds[i][j] = None

    # remove redundant (linearly dependent) rows from equality constraints
    if A_eq.size > 0 and np.linalg.matrix_rank(A_eq) < A_eq.shape[0]:
        warn("A_eq does not appear to be of full row rank. To improve "
             "performance, check the problem formulation for redundant "
             "equality constraints.", OptimizeWarning)
        A_eq, b_eq, status, message = _remove_redundancy(A_eq, b_eq)
        if status != 0:
            complete = True

    return c, c0, A_ub, b_ub, A_eq, b_eq, bounds, \
        x, undo, complete, status, message


def _get_Abc(
        c,
        c0=0,
        A_ub=None,
        b_ub=None,
        A_eq=None,
        b_eq=None,
        bounds=None,
        undo=[]):
    """
    Given a linear programming problem of the form:

    minimize:     c^T * x

    subject to:   A_ub * x <= b_ub
                  A_eq * x == b_eq
                  bounds[i][0] < x_i < bounds[i][1]

    return the problem in standard form:
    minimize:     c'^T * x'

    subject to:   A * x' == b
                  0 < x' < oo

    by adding slack variables and making variable substitutions as necessary.

    Parameters
    ----------
    c : 1-D array
        Coefficients of the linear objective function to be minimized.
        Components corresponding with fixed variables have been eliminated.
    c0 : float
        Constant term in objective function due to fixed (and eliminated)
        variables.
    A_ub : 2-D array
        2-D array which, when matrix-multiplied by x, gives the values of the
        upper-bound inequality constraints at x. Unnecessary rows/columns
        have been removed.
    b_ub : 1-D array
        1-D array of values representing the upper-bound of each inequality
        constraint (row) in A_ub. Unnecessary elements have been removed.
    A_eq : 2-D array
        2-D array which, when matrix-multiplied by x, gives the values of the
        equality constraints at x. Unnecessary rows/columns have been removed.
    b_eq : 1-D array
        1-D array of values representing the RHS of each equality constraint
        (row) in A_eq. Unnecessary elements have been removed.
    bounds : sequence of tuples
        ``(min, max)`` pairs for each element in ``x``, defining
        the bounds on that parameter. Use None for each of ``min`` or
        ``max`` when there is no bound in that direction. Bounds have been
        tightened where possible.
    undo: list of tuples
        (index, value) pairs that record the original index and fixed value
        for each variable removed from the problem

    Returns
    -------
    A : 2-D array
        2-D array which, when matrix-multiplied by x, gives the values of the
        equality constraints at x (for standard form problem).
    b : 1-D array
        1-D array of values representing the RHS of each equality constraint
        (row) in A (for standard form problem).
    c : 1-D array
        Coefficients of the linear objective function to be minimized (for
        standard form problem).
    c0 : float
        Constant term in objective function due to fixed (and eliminated)
        variables.

    References
    ----------
    .. [6] Bertsimas, Dimitris, and J. Tsitsiklis. "Introduction to linear
           programming." Athena Scientific 1 (1997): 997.

    """

    # check/enforce consistency
    n_x = len(c)

    fixed_x = set()
    if len(undo) > 0:
        # these are indices of variables removed from the problem
        # however, their bounds are still part of the bounds list
        fixed_x = set(undo[0])
    # they are needed elsewhere, but not here
    bounds = [bounds[i] for i in range(len(bounds)) if i not in fixed_x]
    # in retrospect, the standard form of bounds should have been an n x 2
    # array maybe change it someday

    # modify problem such that all variables have only non-negativity bounds
    if bounds is not None:
        for i, b in enumerate(bounds):
            lb, ub = b
            if lb is None and ub is None:
                # unbounded: substitute xi = xi+ + xi-
                A_ub = np.hstack((A_ub, -A_ub[:, i:i + 1]))
                A_eq = np.hstack((A_eq, -A_eq[:, i:i + 1]))
                c = np.concatenate((c, [-c[i]]))
                n_x = len(c)
            elif lb == ub:  # this shouldn't happen if preprocessing is on
                # bounds equal: convert to equality constraint
                Arow = np.zeros((1, n_x))
                Arow[0, i] = 1
                A_eq = np.vstack((A_eq, Arow))
                b_eq = np.concatenate((b_eq, np.array([1])))
            else:
                if lb is None:
                    # unbounded below: substitute xi = -xi' (unbounded above)
                    lb, ub = -ub, lb
                    c[i] *= -1
                    A_ub[:, i] *= -1
                    A_eq[:, i] *= -1
                if ub is not None:
                    # upper bound: add inequality constraint
                    Arow = np.zeros((1, n_x))
                    Arow[0, i] = 1
                    A_ub = np.vstack((A_ub, Arow))
                    b_ub = np.concatenate((b_ub, np.array([ub])))
                if lb is not None:  # this MUST be if, not elif
                    # lower bound: substitute xi = xi' + lb
                    # now there is a constant term in objective
                    c0 += lb * c[i]
                    b_ub -= lb * A_ub[:, i]
                    b_eq -= lb * A_eq[:, i]

    A1 = np.hstack([A_ub, np.eye(len(A_ub))])  # add slack variables
    A2 = np.hstack([A_eq, np.zeros((len(A_eq), len(A_ub)))])
    A = np.vstack((A1, A2))
    b = np.concatenate((b_ub, b_eq))
    c = np.concatenate((c, np.zeros((len(A_ub),))))

    return A, b, c, c0


def _postprocess(
        x,
        c,
        A_ub=None,
        b_ub=None,
        A_eq=None,
        b_eq=None,
        bounds=None,
        complete=False,
        undo=[],
        status=0,
        message=""):
    """
    Given solution x to presolved, standard form linear program x, add
    fixed variables back into the problem and undo the variable substitutions
    to get solution to original linear program. Also, calculate the objective
    function value, slack in original upper bound constraints, and residuals
    in original equality constraints.

    Parameters
    ----------
    x : 1-D array
        Solution vector to the standard-form problem.
    c : 1-D array
        Original coefficients of the linear objective function to be minimized.
    A_ub : 2-D array
        Original upper bound constraint matrix.
    b_ub : 1-D array
        Original upper bound constraint vector.
    A_eq : 2-D array
        Original equality constraint matrix.
    b_eq : 1-D array
        Original equality constraint vector.
    bounds : sequence of tuples
        Bounds, as modified in presolve
    complete : boolean
        Whether the solution is was determined in presolve (True if so)
    undo: list of tuples
        (index, value) pairs that record the original index and fixed value
        for each variable removed from the problem
    status : int
        An integer representing the exit status of the optimization::
             0 : Optimization terminated successfully
             1 : Iteration limit reached
             2 : Problem appears to be infeasible
             3 : Problem appears to be unbounded
             4 : Serious numerical difficulties encountered
    message : str
        A string descriptor of the exit status of the optimization.

    Returns
    -------
    x : 1-D array
        Solution vector to original linear programming problem
    fun: float
        optimal objective value for original problem
    slack: 1-D array
        The (non-negative) slack in the upper bound constraints, that is,
        b_ub - A_ub * x
    con : 1-D array
        The (nominally zero) residuals of the equality constraints, that is,
        b - A_eq * x
    status : int
        An integer representing the exit status of the optimization::
             0 : Optimization terminated successfully
             1 : Iteration limit reached
             2 : Problem appears to be infeasible
             3 : Problem appears to be unbounded
             4 : Serious numerical difficulties encountered
    message : str
        A string descriptor of the exit status of the optimization.

    """
    # note that all the inputs are the ORIGINAL, unmodified versions
    # no rows, columns have been removed
    # the only exception is bounds; it has been modified
    # we need these modified values to undo the variable substitutions
    # in retrospect, perhaps this could have been simplified if the "undo"
    # variable also contained information for undoing variable substitutions

    n_x = len(c)

    # we don't have to undo variable substitutions for fixed variables that
    # were removed from the problem
    no_adjust = set()

    # if there were variables removed from the problem, add them back into the
    # solution vector
    if len(undo) > 0:
        no_adjust = set(undo[0])
        x = x.tolist()
        for i, val in zip(undo[0], undo[1]):
            x.insert(i, val)
        x = np.array(x)

    # now undo variable substitutions
    # if "complete", problem was solved in presolve; don't do anything here
    if not complete and bounds is not None:  # bounds are never none, probably
        n_unbounded = 0
        for i, b in enumerate(bounds):
            if i in no_adjust:
                continue
            lb, ub = b
            if lb is None and ub is None:
                n_unbounded += 1
                x[i] = x[i] - x[n_x + n_unbounded - 1]
            else:
                if lb is None:
                    x[i] = ub - x[i]
                else:
                    x[i] += lb

    n_x = len(c)
    x = x[:n_x]  # all the rest of the variables were artificial
    fun = x.dot(c)
    slack = b_ub - np.dot(A_ub, x)  # report slack for ORIGINAL UB constraints
    # report residuals of ORIGINAL EQ constraints
    con = b_eq - np.dot(A_eq, x)

    if status == 0 and (np.isnan(x).any() or np.isnan(fun) or
                        np.isnan(slack).any() or np.isnan(con).any()):
        status = 4
        message = "Numerical difficulties were encountered but no errors " \
                  "were raised. This is known to occur if the 'presolve' " \
                  "option is False, 'sparse' is True, and A_eq includes " \
                  "redundant rows. If you encounter this under different " \
                  "circumstances, please submit a bug report. Otherwise, " \
                  "remove linearly dependent equations from your equality " \
                  "constraints or enable presolve."

    return x, fun, slack, con, status, message


def _get_solver(sparse=False, lstsq=False, sym_pos=False, cholesky=False):
    """
    Given solver options, return a handle to the appropriate linear system
    solver.

    Parameters
    ----------
    sparse : boolean
        True if the system to be solved is sparse. This is typically set
        True when the original A_ub and A_eq matrices are sparse.
    lstsq : boolean
        True if the system is ill-conditioned and/or (nearly) singular and
        thus a more robust least-squares solver is desired. This is sometimes
        needed as the solution is approached.
    sym_pos : boolean
        True if the system matrix is symmetric positive definite
        Sometimes this needs to be set false as the solution is approached,
        even when the system be symmetric positive definite, due to numerical
        difficulties.
    cholesky : boolean
        True if the system is to be solved by explicit Cholesky decomposition
        followed by explicit forward/backward substitution. This is
        occasionally faster for very large, symmetric positive definite
        systems when a system with the same matrix is to be solved with
        several right hand sides.

    Returns
    -------
    solve : function
        Handle to the appropriate solver function

    """
    if sparse:
        if lstsq:  # this doesn't seem to work
            # solve = lambda M, r: sps.linalg.lsqr(M, r)[0]
            def solve(M, r): 
                return sps.linsalg.lsqr(M, r)[0]
        else:
            # solve = lambda M, r: sps.linalg.spsolve(
            # M, r, permc_spec="MMD_AT_PLUS_A")
            def solve(M, r): 
                return sps.linalg.spsolve(M, r, permc_spec="MMD_AT_PLUS_A")
            # in tests MMD_AT_PLUS_A was often fastest.
            # should this be exposed as an option?

    else:
        if lstsq:  # sometimes necessary as solution is approached
            # solve = lambda M, r: sp.linalg.lstsq(M, r)[0]
            def solve(M, r): 
                return sp.linalg.lstsq(M, r)[0]
        elif cholesky:
            solve = _fb_subs
        else:
            # this seems to cache the matrix factorization, so solving
            # with multiple right hand sides is much faster
            # solve = lambda M, r, sym_pos = sym_pos: sp.linalg.solve(
            # M, r, sym_pos=sym_pos)
            def solve(M, r, sym_pos=sym_pos): 
                return sp.linalg.solve(M, r, sym_pos=sym_pos)

    return solve


def _get_delta(
            A,
            b,
            c,
            x,
            y,
            z,
            tau,
            kappa,
            gamma,
            eta,
            sparse=False,
            lstsq=False,
            sym_pos=False,
            cholesky=False,
            pc=True,
            ip=False):
    """
    Given standard form problem defined by A, b, and c; current variable
    estimates x, y, z, tau, and kappa; algorithmic parameters gamma and eta;
    and options sparse, lstsq, sym_pos, cholesky, pc (predictor-corrector),
    and ip (initial point improvement), get the search direction for
    increments to the variable estimates.

    Parameters
    ----------
    As defined in [1], except:
    sparse : boolean
        True if the system to be solved is sparse. This is typically set
        True when the original A_ub and A_eq matrices are sparse.
    lstsq : boolean
        True if the system is ill-conditioned and/or (nearly) singular and
        thus a more robust least-squares solver is desired. This is sometimes
        needed as the solution is approached.
    sym_pos : boolean
        True if the system matrix is symmetric positive definite
        Sometimes this needs to be set false as the solution is approached,
        even when the system be symmetric positive definite, due to numerical
        difficulties.
    cholesky : boolean
        True if the system is to be solved by explicit Cholesky decomposition
        followed by explicit forward/backward substitution. This is
        occasionally faster for very large, symmetric positive definite
        systems when a system with the same matrix is to be solved with
        several right hand sides.
    pc : boolean
        True if the predictor-corrector method of Mehrota is to be used. This
        is almost always (if not always) beneficial. Even though it requires
        the solution of an additional linear system, the factorization
        is typically (implicitly) reused so solution is efficient, and the
        number of algorithm iterations is typically reduced.
    ip : boolean
        True if the improved initial point suggestion due to [1] section 4.3
        is desired. It's unclear whether this is beneficial.

    Returns
    -------
    Search directions as defined in [1]

    References
    ----------
    .. [1] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.

    """

    solve = _get_solver(sparse, lstsq, sym_pos, cholesky)
    n_x = len(x)

    # [1] Equation 8.8
    r_P = b * tau - A.dot(x)
    r_D = c * tau - A.T.dot(y) - z
    r_G = c.dot(x) - b.transpose().dot(y) + kappa
    mu = (x.dot(z) + tau * kappa) / (n_x + 1)

    #  Assemble M from [1] Equation 8.31
    Dinv = x / z
    if sparse:
        # sparse requires Dinv to be diag matrix
        M = A.dot(sps.diags(Dinv, 0, format="csc").dot(A.T))
    else:
        # dense does not; use broadcasting
        M = A.dot(Dinv.reshape(-1, 1) * A.T)

    # For very large problems it's more efficient to take Cholesky
    # decomposition first, then use custom fwd/back solve function _fb_subs
    # multiple times.
    # For most problems, calling sp.linalg.solve w/ sym_pos = True
    # is faster. I am pretty certain it caches the factorization for multiple
    # uses and checks the incoming matrix to see if it's the same as the one
    # it already factorized. (I can't explain the speed otherwise.)
    if cholesky:
        L = sp.linalg.cholesky(M, lower=True)

    # pc: "predictor-corrector" [1] Section 4.1
    # In development this option could be turned off
    # but it always seems to improve performance substantially
    n_corrections = 1 if pc else 0

    i = 0
    alpha, d_x, d_z, d_tau, d_kappa = 0, 0, 0, 0, 0
    while i <= n_corrections:
        # Reference [1] Eq. 8.6
        rhatp = eta(gamma) * r_P
        rhatd = eta(gamma) * r_D
        rhatg = np.array(eta(gamma) * r_G).reshape((1,))

        # Reference [1] Eq. 8.7
        rhatxs = gamma * mu - x * z
        rhattk = np.array(gamma * mu - tau * kappa).reshape((1,))

        if i == 1:
            if ip:  # if the correction is to get "initial point"
                # Reference [1] Eq. 8.23
                rhatxs = (1 - alpha) * gamma * mu - \
                    x * z - alpha**2 * d_x * d_z
                rhattk = np.array(
                    (1 -
                     alpha) *
                    gamma *
                    mu -
                    tau *
                    kappa -
                    alpha**2 *
                    d_tau *
                    d_kappa).reshape(
                    (1,
                     ))
            else:  # if the correction is for "predictor-corrector"
                # Reference [1] Eq. 8.13
                rhatxs -= d_x * d_z
                rhattk -= d_tau * d_kappa

        # sometimes numerical difficulties arise as the solution is approached
        # this loop tries to solve the equations using a sequence of functions
        # for solve.
        # 1. _fb_subs: cholesky factorization then forward/back subs,
        # 2. scipy.linalg.solve w/ sym_pos = True,
        # 3. scipy.linalg.solve w/ sym_pos = False, and if all else fails
        # 4. scipy.linalg.lstsq
        solved = False
        while(not solved):
            try:
                solve_this = L if cholesky else M
                # [1] Equation 8.28
                p, q = _sym_solve(Dinv, solve_this, A, c, b, solve)
                # [1] Equation 8.29
                u, v = _sym_solve(Dinv, solve_this, A, rhatd -
                                  (1 / x) * rhatxs, rhatp, solve)
                solved = True
            except (LinAlgError, ValueError) as e:
                # Usually this doesn't happen. If it does, it happens when
                # there are redundant constraints or when approaching the
                # solution. If so, change solver.
                cholesky = False
                if not lstsq:
                    if sym_pos:
                        warn(
                            "Solving system with option 'sym_pos':True "
                            "failed. It is normal for this to happen "
                            "occasionally, especially as the solution is "
                            "approached. However, if you see this frequently, "
                            "consider setting option 'sym_pos' to False.",
                            OptimizeWarning)
                        sym_pos = False
                    else:
                        warn(
                            "Solving system with option 'sym_pos':False "
                            "failed. This may happen occasionally, "
                            "especially as the solution is "
                            "approached. However, if you see this frequently, "
                            "your problem may be numerically challenging. "
                            "If you cannot improve the formulation, consider "
                            "setting 'lstsq' to True.", OptimizeWarning)
                        lstsq = True
                else:
                    raise e
                solve = _get_solver(sparse, lstsq, sym_pos)

        # [1] Results after 8.29
        d_tau = (rhatg + 1 / tau * rhattk - (-c.dot(u) + b.dot(v))) / \
            (1 / tau * kappa + (-c.dot(p) + b.dot(q)))
        d_x = u + p * d_tau
        d_y = v + q * d_tau

        # [1] Relations between  after 8.25 and 8.26
        d_z = (1 / x) * (rhatxs - z * d_x)
        d_kappa = 1 / tau * (rhattk - kappa * d_tau)

        # [1] 8.12 and "Let alpha be the maximal possible step..." before 8.23
        alpha = _get_step(x, d_x, z, d_z, tau, d_tau, kappa, d_kappa, 1)
        if ip:  # initial point - see [1] 4.4
            gamma = 10
        else:  # predictor-corrector, [1] definition after 8.12
            beta1 = 0.1  # [1] pg. 220 (Table 8.1)
            gamma = (1 - alpha)**2 * min(beta1, (1 - alpha))
        i += 1

    return d_x, d_y, d_z, d_tau, d_kappa


def _fb_subs(L, r):
    """
    Given lower triangular Cholesky factor L and right-hand side r, solve the
    system L L^T = r by substitution.

    Parameters
    ----------
    L : 2-D array (or sparse matrix)
        Lower triangular matrix, typically a Cholesky factor
    r : 1-D array
        Right hand side vector

    Returns
    -------
    r : 1-D array
        Solution of linear system

    """
    y = sp.linalg.solve_triangular(L, r, lower=True)
    x = sp.linalg.solve_triangular(L.T, y, lower=False)
    return x


def _sym_solve(Dinv, M, A, r1, r2, solve):
    """
    An implementation of [1] equation 8.31 and 8.32

    References
    ----------
    .. [1] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.

    """
    # [1] 8.31
    r = r2 + A.dot(Dinv * r1)
    v = solve(M, r)
    # [1] 8.32
    u = Dinv * (A.T.dot(v) - r1)
    return u, v


def _get_step(x, d_x, z, d_z, tau, d_tau, kappa, d_kappa, alpha0):
    """
    An implementation of [1] equation 8.21

    References
    ----------
    .. [1] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.

    """
    # [1] 4.3 Equation 8.21, ignoring 8.20 requirement
    # same step is taken in primal and dual spaces
    # alpha0 is basically beta3 from [1] Table 8.1, but instead of beta3
    # the value 1 is used in Mehrota corrector and initial point correction
    i_x = d_x < 0
    i_z = d_z < 0
    alpha_x = alpha0 * np.min(x[i_x] / -d_x[i_x]) if np.any(i_x) else 1
    alpha_tau = alpha0 * tau / -d_tau if d_tau < 0 else 1
    alpha_z = alpha0 * np.min(z[i_z] / -d_z[i_z]) if np.any(i_z) else 1
    alpha_kappa = alpha0 * kappa / -d_kappa if d_kappa < 0 else 1
    alpha = np.min([1, alpha_x, alpha_tau, alpha_z, alpha_kappa])
    return alpha


def _get_message(status):
    """
    Given problem status code, return a more detailed message.

    Parameters
    ----------
    status : int
        An integer representing the exit status of the optimization:
         0 : Optimization terminated successfully
         1 : Iteration limit reached
         2 : Problem appears to be infeasible
         3 : Problem appears to be unbounded
         4 : Serious numerical difficulties encountered.

    Returns
    -------
    message : str
        A string descriptor of the exit status of the optimization.

    """
    messages = \
        ["Optimization terminated successfully.",
         "The iteration limit was reached before the algorithm converged.",
         "The algorithm terminated successfully and determined that the " 
         "problem is infeasible.",
         "The algorithm terminated successfully and determined that the " 
         "problem is unbounded.",
         "Numerical difficulties were encountered before the problem " 
         "converged. Please check your problem formulation for errors, " 
         "independence of linear equality constraints, and reasonable " 
         "scaling and matrix condition numbers. If you continue to " 
         "encounter this error, please submit a bug report."
         ]
    return messages[status]


def _do_step(x, y, z, tau, kappa, d_x, d_y, d_z, d_tau, d_kappa, alpha):
    """
    An implementation of [1] Equation 8.9

    References
    ----------
    .. [1] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.

    """
    x = x + alpha * d_x
    tau = tau + alpha * d_tau
    z = z + alpha * d_z
    kappa = kappa + alpha * d_kappa
    y = y + alpha * d_y
    return x, y, z, tau, kappa


def _get_blind_start(shape):
    """
    Return the starting point from [1] 4.4

    References
    ----------
    .. [1] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.

    """
    m, n = shape
    x0 = np.ones(n)
    y0 = np.zeros(m)
    z0 = np.ones(n)
    tau0 = 1
    kappa0 = 1
    return x0, y0, z0, tau0, kappa0


def _indicators(A, b, c, c0, x, y, z, tau, kappa):
    """
    Implementation of several equations from [1] used as indicators of
    the status of optimization.

    References
    ----------
    .. [1] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.

    """

    # residuals for termination are relative to initial values
    x0, y0, z0, tau0, kappa0 = _get_blind_start(A.shape)

    # See [1], Section 4 - The Homogeneous Algorithm, Equation 8.8
    # r_p = lambda x, tau: b * tau - A.dot(x)
    def r_p(x, tau): 
        return b * tau - A.dot(x)

    # r_d = lambda y, z, tau: c * tau - A.T.dot(y) - z
    def r_d(y, z, tau): 
        return c * tau - A.T.dot(y) - z

    # r_g = lambda x, y, kappa: kappa + c.dot(x) - b.dot(y)
    def r_g(x, y, kappa): 
        return kappa + c.dot(x) - b.dot(y)

    # np.dot unpacks if they are arrays of size one
    # mu = lambda x, tau, z, kappa: (
    # x.dot(z) + np.dot(tau, kappa)) / (len(x) + 1)
    def mu(x, tau, z, kappa): 
        return (x.dot(z) + np.dot(tau, kappa)) / (len(x) + 1)

    obj = c.dot(x / tau) + c0
    # norm = lambda a: np.linalg.norm(a)

    def norm(a): 
        return np.linalg.norm(a)

    # See [1], Section 4.5 - The Stopping Criteria
    r_p0 = r_p(x0, tau0)
    r_d0 = r_d(y0, z0, tau0)
    r_g0 = r_g(x0, y0, kappa0)
    mu_0 = mu(x0, tau0, z0, kappa0)
    rho_A = norm(c.T.dot(x) - b.T.dot(y)) / (tau + norm(b.T.dot(y)))
    rho_p = norm(r_p(x, tau)) / max(1, norm(r_p0))
    rho_d = norm(r_d(y, z, tau)) / max(1, norm(r_d0))
    rho_g = norm(r_g(x, y, kappa)) / max(1, norm(r_g0))
    rho_mu = mu(x, tau, z, kappa) / mu_0
    return rho_p, rho_d, rho_A, rho_g, rho_mu, obj


def _display_iter(rho_p, rho_d, rho_g, alpha, rho_mu, obj, header=False):
    """
    Print indicators of optimization status to the console.

    Parameters
    ----------
    rho_p : float
        The (normalized) primal feasibility, see [1] 4.5
    rho_d : float
        The (normalized) dual feasibility, see [1] 4.5
    rho_g : float
        The (normalized) duality gap, see [1] 4.5
    alpha : float
        The step size, see [1] 4.3
    rho_mu : float
        The (normalized) path parameter, see [1] 4.5
    obj : float
        The objective function value of the current iterate
    header : boolean
        True if a header is to be printed

    References
    ----------
    .. [1] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.

    """
    if header:
        print("Primal Feasibility ",
              "Dual Feasibility   ",
              "Duality Gap        ",
              "Step               ",
              "Path Parameter     ",
              "Objective          ")

    print('{0:<20}{1:<20}{2:<20}{3:<20}{4:<20}{5:<20}'.format(
        rho_p,
        rho_d,
        rho_g,
        alpha,
        rho_mu,
        obj))


def _ip_hsd(A, b, c, c0, alpha0, beta, maxiter, disp, tol, 
            sparse, lstsq, sym_pos, cholesky, pc, ip):
    """
    Solve a linear programming problem in standard form:

    minimize:     c'^T * x'

    subject to:   A * x' == b
                  0 < x' < oo

    using the interior point method of [1].

    Parameters
    ----------
    A : 2-D array
        2-D array which, when matrix-multiplied by x, gives the values of the
        equality constraints at x (for standard form problem).
    b : 1-D array
        1-D array of values representing the RHS of each equality constraint
        (row) in A (for standard form problem).
    c : 1-D array
        Coefficients of the linear objective function to be minimized (for
        standard form problem).
    c0 : float
        Constant term in objective function due to fixed (and eliminated)
        variables. (Purely for display.)
    alpha0 : float
        The maximal step size for Mehrota's predictor-corrector search
        direction; see \beta3 of [1] Table 8.1
    beta : float
        The desired reduction of the path parameter mu (see  [3])
    maxiter : int
        The maximum number of iterations of the algorithm
    disp : boolean
        True if indicators of optimization status are to be printed to the
        console each iteration
    tol : float
        Termination tolerance; see [1] 4.5
    sparse : boolean
        True if the problem is to be treated as sparse.
    lstsq : boolean
        True if the problem is expected to be very poorly conditioned. This
        should always be False unless severe numerical difficulties are
        encountered.
    sym_pos : boolean
        True if the problem is expected to yield a well conditioned
        symmetric positive definite normal equation matrix (almost always).
    cholesky : boolean
        True if the normal equations are to be solved by explicit Cholesky
        decomposition followed by explicit forward/backward substitution.
        This is occasionally faster for very large problems, but should
        typically be set False.
    pc : boolean
        True if the predictor-corrector method of Mehrota is to be used. This
        is almost always (if not always) beneficial.
    ip : boolean
        True if the improved initial point suggestion due to [1] section 4.3
        is desired. It's unclear whether this is beneficial.

    Returns
    -------
    x_hat : float
        Solution vector (for standard form problem).
    status : int
        An integer representing the exit status of the optimization:
         0 : Optimization terminated successfully
         1 : Iteration limit reached
         2 : Problem appears to be infeasible
         3 : Problem appears to be unbounded
         4 : Serious numerical difficulties encountered.
    message : str
        A string descriptor of the exit status of the optimization.
    iteration : int
        The number of iterations taken to solve the problem

    References
    ----------
    .. [1] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.
    .. [3] Freund, Robert M. "Primal-Dual Interior-Point Methods for Linear
           Programming based on Newton's Method." Unpublished Course Notes,
           March 2004. Available 2/25/2017 at:
           https://ocw.mit.edu/courses/sloan-school-of-management/15-084j-nonlinear-programming-spring-2004/lecture-notes/lec14_int_pt_mthd.pdf

    """

    iteration = 0

    # default initial point
    x, y, z, tau, kappa = _get_blind_start(A.shape)

    # first iteration is special improvement of initial point
    ip = ip if pc else False

    # [1] 4.5
    rho_p, rho_d, rho_A, rho_g, rho_mu, obj = _indicators(
        A, b, c, c0, x, y, z, tau, kappa)
    go = rho_p > tol or rho_d > tol or rho_A > tol  # we might get lucky : )

    if disp:
        _display_iter(rho_p, rho_d, rho_g, "-", rho_mu, obj, header=True)

    status = 0
    message = "Optimization terminated successfully."

    if sparse:
        A = sps.csr_matrix(A)
        A.T = A.transpose()  # A.T is defined for sparse matrices but is slow
        # Redefine it to avoid calculating again
        # This is fine as long as A doesn't change

    while go:

        iteration += 1

        if ip:  # initial point
            # [1] Section 4.4
            gamma = 1

            # eta = lambda g: 1
            def eta(g): 
                return 1
        else:
            # gamma = 0 in predictor step according to [1] 4.1
            # if predictor/corrector is off, use mean of complementarity [3]
            # 5.1 / [4] Below Figure 10-4
            gamma = 0 if pc else beta * np.mean(z * x)
            # [1] Section 4.1

            # eta = lambda g=gamma: 1 - g
            def eta(g=gamma): 
                return 1 - g

        try:
            # Solve [1] 8.6 and 8.7/8.13/8.23
            d_x, d_y, d_z, d_tau, d_kappa = _get_delta(
                A, b, c, x, y, z, tau, kappa, gamma, eta,
                sparse, lstsq, sym_pos, cholesky, pc, ip)

            if ip:  # initial point
                # [1] 4.4
                # Formula after 8.23 takes a full step regardless if this will take
                # it negative
                x, y, z, tau, kappa = _do_step(
                    x, y, z, tau, kappa, d_x, d_y, d_z, d_tau, d_kappa, alpha=1)
                x[x < 1] = 1
                z[z < 1] = 1
                tau = max(1, tau)
                kappa = max(1, kappa)
                ip = False  # done with initial point
            else:
                # [1] Section 4.3
                alpha = _get_step(
                    x,
                    d_x,
                    z,
                    d_z,
                    tau,
                    d_tau,
                    kappa,
                    d_kappa,
                    alpha0)
                # [1] Equation 8.9
                x, y, z, tau, kappa = _do_step(
                    x, y, z, tau, kappa, d_x, d_y, d_z, d_tau, d_kappa, alpha)

        except (LinAlgError, FloatingPointError):
            # this can happen when sparse solver is used and presolve 
            # is turned off. I've never seen it otherwise.
            status = 4
            message = _get_message(status)
            break

        # [1] 4.5
        rho_p, rho_d, rho_A, rho_g, rho_mu, obj = _indicators(
            A, b, c, c0, x, y, z, tau, kappa)
        go = rho_p > tol or rho_d > tol or rho_A > tol

        if disp:
            _display_iter(rho_p, rho_d, rho_g, alpha, rho_mu, obj)

        # [1] 4.5
        inf1 = rho_p < tol and rho_d < tol and rho_g < tol and tau < tol * \
            max(1, kappa)
        inf2 = rho_mu < tol and tau < tol * min(1, kappa)
        if inf1 or inf2:
            # [1] Lemma 8.4 / Theorem 8.3
            if b.transpose().dot(y) > tol:
                status = 2
            else:  # elif c.T.dot(x) < tol: ? Probably not necessary.
                status = 3
            message = _get_message(status)
            break
        elif iteration >= maxiter:
            status = 1
            message = _get_message(status)
            break

    if disp:
        print(message)

    x_hat = x / tau
    # [1] Statement after Theorem 8.2
    return x_hat, status, message, iteration


def _linprog_ip(
        c,
        A_ub=None,
        b_ub=None,
        A_eq=None,
        b_eq=None,
        bounds=None,
        callback=None,
        alpha0=.99995,
        beta=0.1,
        maxiter=1000,
        disp=False,
        tol=1e-8,
        sparse=False,
        lstsq=False,
        sym_pos=True,
        cholesky=False,
        pc=True,
        ip=False,
        presolve=True,
        **unknown_options):
    """
    Minimize a linear objective function subject to linear
    equality and inequality constraints using the interior point method of [1]

    Linear Programming is intended to solve the following problem form:

    Minimize:     c^T * x

    Subject to:   A_ub * x <= b_ub
                  A_eq * x == b_eq
                  bounds[i][0] < x_i < bounds[i][1]

    Parameters
    ----------
    c : array_like
        Coefficients of the linear objective function to be minimized.
    A_ub : array_like, optional
        2-D array which, when matrix-multiplied by x, gives the values of the
        upper-bound inequality constraints at x.
    b_ub : array_like, optional
        1-D array of values representing the upper-bound of each inequality
        constraint (row) in A_ub.
    A_eq : array_like, optional
        2-D array which, when matrix-multiplied by x, gives the values of the
        equality constraints at x.
    b_eq : array_like, optional
        1-D array of values representing the RHS of each equality constraint
        (row) in A_eq.
    bounds : sequence, optional
        ``(min, max)`` pairs for each element in ``x``, defining
        the bounds on that parameter. Use None for one of ``min`` or
        ``max`` when there is no bound in that direction. By default
        bounds are ``(0, None)`` (non-negative)
        If a sequence containing a single tuple is provided, then ``min`` and
        ``max`` will be applied to all variables in the problem.
        
    Options
    -------
    maxiter : int (default = 1000)
        The maximum number of iterations of the algorithm
    disp : boolean (default = False)
        True if indicators of optimization status are to be printed to
        the console each iteration
    tol : float (default = 1e-8)
        Termination tolerance to be used for all termination criteria;
        see [1] 4.5
    alpha0 : float (default = 0.99995)
        The maximal step size for Mehrota's predictor-corrector search
        direction; see \beta3 of [1] Table 8.1
    beta : float (default = 0.1)
        The desired reduction of the path parameter \mu (see [3]) when
        Mehrota's predictor-corrector is not in use (uncommon).
    sparse : boolean (default = False)
        True if the problem is to be treated as sparse. Try setting
        this to True if your constraint matrices are sparse and the
        problem is relatively large.
    lstsq : boolean (default = False)
        True if the problem is expected to be very poorly conditioned.
        This should always be False unless severe numerical
        difficulties are encountered. Leave this at the default unless
        you receive a warning message suggesting otherwise.
    sym_pos : boolean (default = True)
        True if the problem is expected to yield a well conditioned
        symmetric positive definite normal equation matrix
        (almost always). Leave this at the default unless you receive
        a warning message suggesting otherwise.
    cholesky : boolean (default = False)
        True if the normal equations are to be solved by explicit
        Cholesky decomposition followed by explicit forward/backward
        substitution. This is occasionally faster for very large
        problems, but should typically be set False.
    pc : boolean (default = True)
        True if the predictor-corrector method of Mehrota is to be
        used. This is almost always (if not always) beneficial.
    ip : boolean (default = False)
        True if the improved initial point suggestion due to [1]
        section 4.3 is desired. Whether this is beneficial or not
        depends on the problem.
    presolve : boolean (default = True)
        True if presolve routine should be run. The presolve routine is almost
        always useful because it can detect trivial infeasibilities
        and unboundedness, eliminate fixed variables, remove redundancies, 
        etc... One circumstance in which it might be turned off is when
        it detects that the problem is trivially unbounded but it is possible
        that there is an infeasibility it cannot detect.

    Returns
    -------
    A `scipy.optimize.OptimizeResult` consisting of the following fields:

        x : ndarray
            The independent variable vector which optimizes the linear
            programming problem.
        fun : float
            The optimal value of the objective function
        con : float
            The residuals of the equality constraints (nominally zero).
        slack : ndarray
            The values of the slack variables.  Each slack variable corresponds
            to an inequality constraint.  If the slack is zero, then the
            corresponding constraint is active.
        success : bool
            Returns True if the algorithm succeeded in finding an optimal
            solution.
        status : int
            An integer representing the exit status of the optimization::
                
                 0 : Optimization terminated successfully
                 1 : Iteration limit reached
                 2 : Problem appears to be infeasible
                 3 : Problem appears to be unbounded
                 4 : Serious numerical difficulties encountered
                 
        nit : int
            The number of iterations performed.
        message : str
            A string descriptor of the exit status of the optimization.
    
    Notes
    -----
    
    This method implements the algorithm outlined in [1]_ with ideas from [5]_ 
    and a structure inspired by the simpler methods of [3]_ and [4]_.
    
    First, a presolve procedure based on [5]_ attempts to identify trivial 
    infeasibilities, trivial unboundedness, and potential problem 
    simplifications. Specifically, it checks for:
        
        rows of zeros in `A_eq` or `A_ub`, representing trivial constraints;
        columns of zeros in both `A_eq` and `A_ub`, representing unconstrained 
            variables;
        column singletons in `A_eq`, representing fixed variables; and
        column singletons in `A_ub`, representing simple bounds. 
    
    If presolve reveals that the problem is unbounded (e.g. an unconstrained 
    and unbounded variable has negative cost) or infeasible (e.g. a row of 
    zeros in `A_eq` corresponds with a nonzero in `b_eq`), the solver 
    terminates with the appropriate status code. Note that presolve terminates 
    as soon as any sign of unboundedness is detected; consequently, a problem 
    may be reported as unbounded when in reality the problem is infeasible 
    (but infeasibility has not been detected yet). Therefore, if the output 
    message states that unboundedness is detected in presolve and it is 
    necessary to know whether the problem is actually infeasible, the option 
    `presolve=False` should be set.
    
    If neither infeasibility nor unboundedness are detected in a single pass 
    of the presolve check, bounds are tightened where possible and fixed 
    variables are removed from the problem. Then, singular value decomposition 
    is used to identify linearly dependent rows of the `A_eq` matrix, which 
    are removed (unless they represent an infeasibility) to avoid numerical 
    difficulties in the primary solve routine. 
    
    Several potential improvements can be made here: additional presolve 
    checks outlined in [5]_ should be implemented, the presolve routine should 
    be run multiple times (until no further simplifications can be made), a 
    more efficient method for detecting redundancy (e.g. that of [2]_) should 
    be used, and presolve should be implemented for sparse matrices.
    
    After presolve, the problem is transformed to standard form by converting 
    the (tightened) simple bounds to upper bound constraints, introducing 
    non-negative slack variables for inequality constraints, and expressing 
    unbounded variables as the difference between two non-negative variables.
    
    The primal-dual path following method begins with initial ‘guesses’ of 
    the primal and dual variables of the standard form problem and iteratively 
    attempts to solve the (nonlinear) Karush-Kuhn-Tucker conditions for the 
    problem with a gradually reduced logarithmic barrier term added to the 
    objective. This particular implementation uses a homogeneous self-dual 
    formulation, which provides certificates of infeasibility or unboundedness 
    where applicable.
    
    The default initial point for the primal and dual variables is that 
    defined in [1]_ Section 4.4 Equation 8.22. Optionally (by setting initial 
    point option `ip=True`), a (potentially) improved starting point can be 
    calculated according to the additional recommendations of [1]_ Section 4.4.
    
    A search direction is calculated using the predictor-corrector method 
    (single correction) proposed by Mehrota and detailed in [1]_ Section 4.1.  
    (A potential improvement would be to implement the method of multiple 
    corrections described in [1]_ Section 4.2.) In practice, this is 
    accomplished by solving the normal equations, [1]_ Section 5.1 Equations 
    8.31 and 8.32, derived from the Newton equations [1]_ Section 5 Equations 
    8.25 (compare to [1]_ Section 4 Equations 8.6-8.8). The advantage of 
    solving the normal equations rather than 8.25 directly is that the 
    matrices involved are symmetric positive definite, so Cholesky 
    decomposition can be used rather than the more expensive LU factorization. 
    
    In code, this is accomplished using `scipy.linalg.solve` with 
    `sym_pos=True`. Based on speed tests, this appears to cache the 
    Cholesky decomposition of the matrix for later use, which is beneficial 
    as the same system is solved four times with different right hand sides 
    in each iteration of the algorithm. However, for very large, dense 
    systems, it may be beneficial for the Cholesky decomposition to be 
    calculated explicitly using `scipy.linalg.cholesky` and the decomposed 
    systems to be solved by explicit forward/backward substitutions via 
    `scipy.linalg.solve_triangular`; this is possible by setting the option 
    `cholesky=True`.
    
    In problems with redundancy (e.g. if presolve is turned off with option 
    `presolve=False`) or if the matrices become ill-conditioned (e.g. as the 
    solution is approached and some decision variables approach zero), 
    Cholesky decomposition can fail. Should this occur, successively more 
    robust solvers (`scipy.linalg.solve` with `sym_pos=False` then 
    `scipy.linalg.lstsq`) are tried, at the cost of computational efficiency. 
    These solvers can be used from the outset by setting the options 
    `sym_pos=False` and `lstsq=True`, respectively.
    
    Note that with the option `sparse=True`, the normal equations are solved 
    using `scipy.sparse.linalg.spsolve`. Unfortunately, this uses the more 
    expensive LU decomposition from the outset, but for large, sparse problems, 
    the use of sparse linear algebra techniques improves the solve speed 
    despite the use of LU rather than Cholesky decomposition. A simple 
    improvement would be to use the sparse Cholesky decomposition of `CHOLMOD` 
    via `scikit-sparse` when available. 
    
    Other potential improvements for combatting issues associated with dense 
    columns in otherwise sparse problems are outlined in [1]_ Section 5.3 and 
    [7]_ Section 4.1-4.2; the latter also discusses the alleviation of 
    accuracy issues associated with the substitution approach to free 
    variables.
    
    After calculating the search direction, the maximum possible step size 
    that does not violate the non-negativity constraints is calculated, and 
    the smaller of this step size and unity is applied (as in [1]_ Section 
    4.1. [1]_ Section 4.3 suggests improvements for choosing the step size.
    
    The new point is tested according to the termination conditions of [1]_ 
    Section 4.5. The same tolerance, which can be set using the `tol` option, 
    is used for all checks. If optimality, unboundedness, or infeasibility is 
    detected, the solve procedure terminates; otherwise it repeats.
    
    If optimality is achieved, a postsolve procedure undoes transformations 
    associated with presolve and converting to standard form. It then 
    calculates the residuals (equality constraint violations, which should 
    be very small) and slacks (difference between the left and right hand 
    sides of the upper bound constraints) of the original problem, which are 
    returned with the solution in an `OptimizeResult` object.

    References
    ----------
    .. [1] Andersen, Erling D., and Knud D. Andersen. "The MOSEK interior point
           optimizer for linear programming: an implementation of the
           homogeneous algorithm." High performance optimization. Springer US,
           2000. 197-232.
    .. [2] Andersen, Erling D. "Finding all linearly dependent rows in
           large-scale linear programming." Optimization Methods and Software
           6.3 (1995): 219-227.
    .. [3] Freund, Robert M. "Primal-Dual Interior-Point Methods for Linear
           Programming based on Newton's Method." Unpublished Course Notes,
           March 2004. Available 2/25/2017 at
           https://ocw.mit.edu/courses/sloan-school-of-management/15-084j-nonlinear-programming-spring-2004/lecture-notes/lec14_int_pt_mthd.pdf
    .. [4] Fourer, Robert. "Solving Linear Programs by Interior-Point Methods."
           Unpublished Course Notes, August 26, 2005. Available 2/25/2017 at
           http://www.4er.org/CourseNotes/Book%20B/B-III.pdf
    .. [5] Andersen, Erling D., and Knud D. Andersen. "Presolving in linear
           programming." Mathematical Programming 71.2 (1995): 221-245.
    .. [6] Bertsimas, Dimitris, and J. Tsitsiklis. "Introduction to linear
           programming." Athena Scientific 1 (1997): 997.
    .. [7] Andersen, Erling D., et al. Implementation of interior point methods 
           for large scale linear programming. HEC/Université de Genève, 1996.

    """
    
    _check_unknown_options(unknown_options)
    
    if callback is not None:
        raise NotImplementedError("method 'interior-point' does not support "
                                  "callback functions.")
             
        # These should be warnings, not Errors
    if sparse and lstsq:
        warn("Invalid option combination 'sparse':True "
             "and 'lstsq':True; Sparse least squares is not recommended.",
             OptimizeWarning)

    if sparse and not sym_pos:
        warn("Invalid option combination 'sparse':True "
             "and 'sym_pos':False; option 'sym_pos' has no effect when "
             "'sparse' is set True.",
             OptimizeWarning)

    if sparse and cholesky:
        warn("Invalid option combination 'sparse':True "
             "and 'cholesky':True; option 'cholesky' has no effect when "
             "'sparse' is set True.",
             OptimizeWarning)

    if lstsq and cholesky:
        warn("Invalid option combination 'lstsq':True "
             "and 'cholesky':True; option 'cholesky' has no effect when "
             "'lstsq' is set True.",
             OptimizeWarning)

    # This can be an error
    if not sym_pos and cholesky:
        raise ValueError(
            "Invalid option combination 'sym_pos':False "
            "and 'cholesky';True: Cholesky decomposition is only possible "
            "for symmetric positive definite matrices.")

    iteration = 0
    complete = False    # will become True if solved in presolve
    undo = []

    # Convert lists to numpy arrays, etc...
    c, A_ub, b_ub, A_eq, b_eq, bounds = _clean_inputs(
        c, A_ub, b_ub, A_eq, b_eq, bounds)

    # Keep the original arrays to calculate slack/residuals for original
    # problem.
    c_o, A_ub_o, b_ub_o, A_eq_o, b_eq_o = c.copy(
    ), A_ub.copy(), b_ub.copy(), A_eq.copy(), b_eq.copy()

    # Solve trivial problem, eliminate variables, tighten bounds, etc...
    c0 = 0  # we might get a constant term in the objective
    if presolve is True:
        c, c0, A_ub, b_ub, A_eq, b_eq, bounds, x, undo, complete, status, \
            message = _presolve(c, A_ub, b_ub, A_eq, b_eq, bounds)

    # If not solved in presolve, solve it
    if not complete:
        # Convert problem to standard form
        A, b, c, c0 = _get_Abc(c, c0, A_ub, b_ub, A_eq, b_eq, bounds, undo)
        # Solve the problem
        x, status, message, iteration = _ip_hsd(A, b, c, c0, alpha0, beta, 
                                                maxiter, disp, tol, sparse, 
                                                lstsq, sym_pos, cholesky, 
                                                pc, ip)

    # Eliminate artificial variables, re-introduce presolved variables, etc...
    # need modified bounds here to translate variables appropriately
    x, fun, slack, con, status, message = _postprocess(
        x, c_o, A_ub_o, b_ub_o, A_eq_o, b_eq_o,
        bounds, complete, undo, status, message)

    sol = {
        'x': x,
        'fun': fun,
        'slack': slack,
        'con': con,
        'status': status,
        'message': message,
        'nit': iteration,
        "success": status == 0}

    return OptimizeResult(sol)
