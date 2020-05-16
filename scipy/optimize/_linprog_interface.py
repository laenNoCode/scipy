'''Make linprog interfaces for external solvers.'''

from distutils.spawn import find_executable
import subprocess
import pathlib
from warnings import warn

from scipy.optimize import OptimizeWarning

def _glpsol_interface(
        infile,
        fmt='freemps',
        display=False,
        seed=None,
        solver='simplex',
        flow=None,
        sense='min',
        scale=True,
        ranges=None,
        tmlim=None,
        memlim=None,
        basis_fac=None,
        simplex_options=None,
        ip_options=None,
        mip_options=None,
        exe='glpsol'):
    '''Interface to stand-alone LP/MIP GLPSOL solver.

    Parameters
    ----------
    infile : str
        File that holds the LP/MIP model.
    fmt : { 'mps', 'freemps', 'lp', 'glp', 'math' }
        Format of the LP/MIP model, ``infile``.
        Default is ``freemps``.
    display : bool
        Send display output to filename
        (for ``format='math'`` only); by default the output is
        sent to terminal
    seed : int or '?'
        Initialize pseudo-random number generator used in
        MathProg model with specified seed (any integer);
    solver : { 'simplex', 'interior' }
        Use simplex (LP/MIP) or interior point method (LP only).
        Default is ``simplex``.
    flow : { 'mincost', 'maxflow', 'cnf' }
        Read min-cost, maximum flow, or CNF-SAT problem in
        DIMACS format.
    sense : { 'min', 'max' }
        Minimization or maximization problem.
        Default is ``min``.
    scale : bool
        Scale the problem. Default is ``True``.
    ranges : str
        Write sensitivity analysis report to file ``ranges`` in
        printable format (simplex only).
    tmlim : int
        Limit solution time to ``tmlim`` seconds.
        Default is no time limit.
    memlim : int
        Limit available memory to ``memlim`` megabytes.
        Default is no memory limit.
    basis_fac : { 'luf+ft', 'luf+cbg', 'luf+cgr', 'btf+cbg', 'btf+cgr' }
        LP basis factorization strategy. Default is ``luf+ft``.
        These are combinations of the following strategies:

            - ``luf`` : plain LU-factorization
            - ``btf`` : block triangular LU-factorization
            - ``ft`` : Forrest-Tomlin update
            - ``cbg`` : Schur complement + Bartels-Golub update
            - ``cgr`` : Schur complement + Givens rotation update

    simplex_options : dict
        Options specific to simplex solver. The dictionary consists of
        the following fields:

            - primal : bool
                Primal or dual simplex. Default is ``True`` (primal).
            - init_basis : { 'std', 'adv', 'bib' }
                Choice of initial basis.  Default is 'adv'.
                One of the following:

                    - ``std`` : standard initial basis of all slacks
                    - ``adv`` : advanced initial basis
                    - ``bib`` : Bixby's initial basis

            - steep : bool
                Use steepest edge technique or standard "textbook"
                pricing.  Default is ``True`` (steepest edge).

            - ratio : { 'relax', 'norelax', 'flip' }
                Ratio test strategy. Default is ``relax``.
                One of the following:

                    - ``relax`` : Harris' two-pass ratio test
                    - ``norelax`` : standard "textbook" ratio test
                    - ``flip`` : long-step ratio test

            - presolve : bool
                Use presolver (assumes ``scale=True`` and
                ``init_basis='adv'``. Default is ``True``.

            - exact : bool
                Use simplex method based on exact arithmetic.
                Default is ``False``.

            - xcheck : bool
                Check final basis using exact arithmetic.
                Default is ``False``.

    ip_options : dict
        Options specific to interior-pooint solver.
        The dictionary consists of the following fields:

            - ordering : { 'nord', 'qmd', 'amd', 'symamd' }
                Ordering algorithm used before Cholesky factorizaiton.
                Default is ``amd``. One of the following:

                    - ``nord`` : natural (original) ordering
                    - ``qmd`` : quotient minimum degree ordering
                    - ``amd`` : approximate minimum degree ordering
                    - ``symamd`` : approximate minimum degree ordering
                        algorithm for Cholesky factorization of symmetric matrices.

    mip_options : dict
        Options specific to MIP solver.
        The dictionary consists of the following fields:

            - nomip : bool
                consider all integer variables as continuous
                (allows solving MIP as pure LP). Default is ``False``.
            - branch : { 'first', 'last', 'mostf', 'drtom', 'pcost' }
                Branching rule. Default is ``drtom``.
                One of the following:

                    - ``first`` : branch on first integer variable
                    - ``last`` : branch on last integer variable
                    - ``mostf`` : branch on most fractional variable
                    - ``drtom`` : branch using heuristic by Driebeck and Tomlin
                    - ``pcost`` : branch using hybrid pseudocost heuristic (may be
                        useful for hard instances)

            - backtrack : { 'dfs', 'bfs', 'bestp', 'bestb' }
                Backtracking rule. Default is ``bestb``.
                One of the following:

                    - ``dfs`` : backtrack using depth first search
                    - ``bfs`` : backtrack using breadth first search
                    - ``bestp`` : backtrack using the best projection heuristic
                    - ``bestb`` : backtrack using node with best local bound

            - presolve : bool
                Use MIP presolver. Default is ``True``.

            - binarize : bool
                replace general integer variables by binary ones
                (assumes ``presolve=True``). Default is ``False``.

            - fpump : bool
                Apply feasibility pump heuristic. Default is ``False``.

            - proxy : int
                Apply proximity search heuristic (in seconds). Default is 60.

            - cuts : list of { 'gomory', 'mir', 'cover', 'clique', 'all' }
                Cuts to generate. Default is no cuts. List of the following:

                    - ``gomory`` : Gomory's mixed integer cuts
                    - ``mir`` : MIR (mixed integer rounding) cuts
                    - ``cover`` : mixed cover cuts
                    - ``clique`` : clique cuts
                    - ``all`` : generate all cuts above

            - tol : float
                Relative mip gap tolerance.

            - minisat : bool
                translate integer feasibility problem to CNF-SAT
                and solve it with MiniSat solver. Default is ``False``.

            - bound : float
                add inequality obj <= bound (minimization) or
                obj >= bound (maximization) to integer feasibility
                problem (assumes ``minisat=True``).
    '''

    # Make sure we can actually call the program
    if find_executable(exe) is None:
        raise SystemError('Could not find executable "%s"' % exe)
    args = [exe]

    # Expand the path to file so glpsol can find it:
    infile = str(pathlib.Path(infile).expanduser().resolve())
    args.append('--' + fmt)
    args.append(infile)

    # Seed is an int or '?'
    if seed is not None:
        args.append('--seed')
        args.append(seed)

    # Solver is 'simplex' or 'interior'
    if solver not in {'simplex', 'interior'}:
        warn('solver not one of "simplex" or "interior".  Using default.', OptimizeWarning)
    else:
        args.append('--' + solver)

    # Sense is 'min' or 'max'
    if sense not in {'min', 'max'}:
        warn('sense not one of "min" or "max". Using default.', OptimizeWarning)
    else:
        args.append('--' + sense)

    # Scale the problem (or not)
    if scale:
        args.append('--scale')
    else:
        args.append('--noscale')

    # Write sensitivity analysis to file
    if ranges is not None:
        args.append('--ranges')
        args.append(ranges)

    # Time and memory limits
    if tmlim is not None:
        args.append('--tmlim')
        args.append(str(tmlim))
    if memlim is not None:
        args.append('--memlim')
        args.append(str(memlim))

    # LP basis factorization strategy
    if basis_fac not in {'luf+ft', 'luf+cbg', 'luf+cgr', 'btf+cbg', 'btf+cgr'}:
        warn('basis_fac not a valid option, using default.', OptimizeWarning)
    else:
        for bf in basis_fac.split('+'):
            args.append('--' + bf)

    # Unpack simplex solver options
    if simplex_options is not None:
        primal = simplex_options.get('primal', None)
        if primal is not None:
            if primal:
                args.append('--primal')
            else:
                args.append('--dual')

        init_basis = simplex_options.get('init_basis', None)
        if init_basis is not None:
            if init_basis not in {'std', 'adv', 'bib'}:
                warn('init_basis not valid. Using default.', OptimizeWarning)
            else:
                args.append('--' + init_basis)

        steep = simplex_options.get('steep', None)
        if steep is not None:
            if steep:
                args.append('--steep')
            else:
                args.append('--nosteep')

        ratio = simplex_options.get('ratio', None)
        if ratio is not None:
            if ratio not in {'relax', 'norelax', 'flip'}:
                warn('ratio is not valid. Using default.', OptimizeWarning)
            else:
                args.append('--' + ratio)

        presolve = simplex_options.get('presolve', None)
        if presolve is not None:
            if presolve:
                args.append('--presol')
            else:
                args.append('--nopresol')

        exact = simplex_options.get('exact', None)
        if exact is not None and exact:
            args.append('--exact')

        xcheck = simplex_options.get('xcheck', None)
        if xcheck is not None and xcheck:
            args.append('--xcheck')

    # Unpack interior-point solver options
    if ip_options is not None:
        ordering = ip_options.get('ordering', None)
        if ordering not in {'nord', 'qmd', 'amd', 'symamd'}:
            warn('ordering is not valid. Using default.', OptimizeWarning)
        else:
            args.append('--' + ordering)

    # Unpack MIP solver options
    if mip_options is not None:
        nomip = mip_options.get('nomip', None)
        if nomip is not None and nomip:
            args.append('--nomip')

        branch = mip_options.get('branch', None)
        if branch is not None:
            if branch not in {'first', 'last', 'mostf', 'drtom', 'pcost'}:
                warn('branching rule is not valid. Using default.', OptimizeWarning)
            else:
                args.append('--' + branch)

        backtrack = mip_options.get('backtrack', None)
        if backtrack is not None:
            if backtrack not in {'dfs', 'bfs', 'bestp', 'bestb'}:
                warn('Bactracking rule not valid. Using default.', OptimizeWarning)
            else:
                args.append('--' + backtrack)

        presolve = mip_options.get('presolve', None)
        if presolve is not None:
            if presolve:
                args.append('--intopt')
            else:
                args.append('--nointopt')

        binarize = mip_options.get('binarize', None)
        if binarize is not None and binarize:
            args.append('--binarize')

        fpump = mip_options.get('fpump', None)
        if fpump is not None and fpump:
            args.append('--fpump')

        proxy = mip_options.get('proxy', None)
        if proxy is not None:
            args.append('--proxy')
            args.append(str(proxy))

        cuts = mip_options.get('cuts', None)
        if cuts is not None:
            cuts = set(list(cuts))
            diff = cuts - {'gomory', 'mir', 'cover', 'clique', 'all'}
            cuts = cuts - diff
            if diff:
                warn('%s are invalid options and will be ignored.' % diff, OptimizeWarning)
            if 'all' in cuts:
                args.append('--cuts')
            else:
                for c in cuts:
                    args.append('--' + c)

        tol = mip_options.get('tol', None)
        if tol is not None:
            args.append('--mipgap')
            args.append(str(tol))

        minisat = mip_options.get('minisat', None)
        if minisat is not None and minisat:
            args.append('--minisat')

        bound = mip_options.get('bound', None)
        if bound is not None:
            args.append('--objbound')
            args.append(str(bound))

    # Run the solver
    subprocess.run(args)
