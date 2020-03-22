/// Dense matrix implementation of the revised simplex method.
//
// Follows the algorithm described in Chapter 3.7 of [1]_.
// Uses BLAS/LAPACK calls for dense matrix factorization and linear algebra.
//
// #define EXPLICIT_INVERSE to compute B^-1 and use it for calculations.  The
// Sherman-Morrison equation is used to update B^-1 every iteration.
//
// If EXPLICIT_INVERSE is not defined, then the LU decomposition will be run
// run every [num] iterations.
//
// #define DEBUG for lots of debugging info to stdout each iteration.
//
// References
// ----------
// .. [1] Shu-Cherng, Fang, and Sarat Puthenpura. "Linear Optimization and
//        Extensions. Theory and Algorithms." (1993).
// .. [2] https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula

// Define these in setup.py
//#define EXPLICIT_INVERSE 1
//#define DEBUG 1

#include <algorithm>
#include <cblas.h>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "_revised_simplex.hpp"

LPResult revised_simplex(const std::size_t m, const std::size_t n,
                         const double *c, const double *A, const double *b,
                         const double *bfs, const idx_t *B_tilde0) {
#ifdef DEBUG
  std::cout << "==========STARTING REVISED_SIMPLEX==========" << std::endl;
#endif

  // Intial solution is the basic feasible solution (bfs)
  auto res = LPResult(m, n, bfs);

  // Populate basis indices
  std::vector<idx_t> B_tilde = initial_B_tilde(B_tilde0, m);
  std::vector<idx_t> N_tilde = initial_N_tilde(B_tilde, n);
  std::size_t mn = N_tilde.size(); // m - n

  // Put all basis costs in cB to begin with
  auto cB = std::make_unique<double[]>(m);
  take(c, B_tilde, cB);

  // It seem like we do need a copy of N to work with
  auto N = std::make_unique<double[]>(m * mn);
  auto cN = std::make_unique<double[]>(mn);
  take_cols(m, n, A, N_tilde, mn, N);
  take(c, N_tilde, cN);

  // Info for initial LU factorization
  auto LU = std::make_unique<double[]>(m * m);
  auto IPIV = std::make_unique<int[]>(m);
  int INFO;
  int m_int = (int)m;

  // Intermediate solutions:
  auto w = std::make_unique<double[]>(m);
  auto rN = std::make_unique<double[]>(mn);
  auto d = std::make_unique<double[]>(m);

#ifndef EXPLICIT_INVERSE

  // How often do we refactor? (Not used when EXPLICIT_INVERSE defined!
  const std::size_t refactor_every = 1; // ~30 when doing BG updates

  char NOTRANS[] = "N";
  char TRANS[] = "T";
  int NRHS = 1;
  int LDB = m;

#else
  // If we are explicitly inverting B, then we can rely on Sherman-Morrison
  // while iterating. Compute the inverse explcitly the first time (most likely
  // identity matrix, but do the full thing in case we get a curve ball)

  // We will need some memory to do some calculations.  The strategy will be
  // allocate enough for both matrix inversion and tmps needed for
  // Sherman-Morrison update
  int LWORK = m * m; // WORK size for matrix inversion

  // Temps for the Sherman-Morrison update.
  int actual_work_size =
      m > 4 ? m * m : m * 4; // make enough room for temps in WORK
  auto WORK = std::make_unique<double[]>(actual_work_size);
  auto u_ptr = WORK.get();
  auto ejp_ptr = WORK.get() + m;
  auto qq_ptr = WORK.get() + 2 * m;
  auto r_ptr = WORK.get() + 3 * m;

  // Do the actual inversion
  auto Binv = std::make_unique<double[]>(m * m);
  inverse(m, n, B_tilde, A, m_int, Binv, IPIV, WORK, LWORK, INFO);

#endif

  // Loop till you just can't loop anymore!
  while (true) {

#ifdef DEBUG
    std::cout << "**********ITER " << res.nit << "**********" << std::endl;
    std::cout << "B_tilde: [ ";
    for (auto const &el : B_tilde) {
      std::cout << el << ' ';
    }
    std::cout << ']' << std::endl;
    std::cout << "N_tilde: [ ";
    for (auto const &el : N_tilde) {
      std::cout << el << ' ';
    }
    std::cout << ']' << std::endl;
#endif

    // Don't do this step if we're doing explicit inverse matrix updates
#ifndef EXPLICIT_INVERSE

    // (Step 0: Factorizations)
    if (res.nit % refactor_every == 0) {

#ifdef DEBUG
      std::cout << "Doing LU factorization!" << std::endl;
      std::cout << "B matrix is:" << std::endl;
      for (idx_t ii = 0; ii < m; ++ii) {
        for (auto const &el : B_tilde) {
          std::cout << A[el + ii * n] << ' ';
        }
        std::cout << std::endl;
      }
#endif

      // NOTE: U: CblasUnit, L : CBlasNonUnit (Upper has ones, lower has no
      // ones)
      Status status = lu(m, n, B_tilde, A, m_int, LU, IPIV, INFO);
      if (status != StatusSuccess) {
        res.status = status;
        return res;
      }

    } else {

      // Instead of recomputing the factorization, do an update!
      std::cerr << "Not implemented yet!" << std::endl;
      res.status = StatusNotImplemented;
      return res;
    }
#endif

    // Step 1: Compute the "simplex multipliers"
    //     B.T @ w = cB => w = B.T^-1 @ cB
    //     U.T @ y = cB
    //     L.T @ w = y
#ifdef DEBUG
    std::cout << "cB: ";
    print_matrix(1, m, cB.get());
    std::cout << std::endl;
#endif
#ifndef EXPLICIT_INVERSE
    std::copy(cB.get(), cB.get() + m, w.get());
    dgetrs_(NOTRANS, &m_int, &NRHS, LU.get(), &m_int, IPIV.get(), w.get(), &LDB,
            &INFO);
#else
    // Use explicit Binv:
    cblas_dgemv(CblasRowMajor, CblasTrans, m, m, 1.0, Binv.get(), m, cB.get(),
                1, 0.0, w.get(), 1);
#endif
#ifdef DEBUG
    std::cout << "w: [ ";
    print_matrix(1, m, w.get());
    std::cout << ']' << std::endl;
#endif

    // Step 2: Compute the reduced costs
    //     rN = cN - w.T @ N

#ifdef DEBUG
    std::cout << "cN: [ ";
    print_matrix(1, mn, cN.get());
    std::cout << "]" << std::endl;
#endif
    std::copy(cN.get(), cN.get() + mn, rN.get());
#ifdef DEBUG
    std::cout << "N matrix is:" << std::endl;
    for (idx_t ii = 0; ii < m; ++ii) {
      for (idx_t jj = 0; jj < mn; ++jj) {
        std::cout << N.get()[jj + ii * mn] << ' ';
      }
      std::cout << std::endl;
    }
#endif
    cblas_dgemv(CblasColMajor, CblasNoTrans, mn, m, -1.0, N.get(), mn, w.get(),
                1, 1.0, rN.get(), 1);
#ifdef DEBUG
    std::cout << "rN: [ ";
    print_matrix(1, mn, rN.get());
    std::cout << ']' << std::endl;
#endif

    // Step 3: Check for optimality / Step 4: Enter the basis
    idx_t q = choose_leaving(N_tilde, rN, res);
    if (res.status == StatusQNotFound) {
#ifdef DEBUG
      std::cout << "Found optimal solution!" << std::endl;
#endif
      res.status = StatusSuccess;
      res.fun = cblas_ddot(n, c, 1, res.x.get(), 1);
      return res;
    }
    // TODO: better way to look up local index?
    idx_t q_local = std::distance(
        N_tilde.cbegin(), std::find(N_tilde.cbegin(), N_tilde.cend(), q));
#ifdef DEBUG
    std::cout << "q/q_local: " << q << "/" << q_local << std::endl;
    std::string truth = N_tilde[q] == q ? "True" : "False";
    std::cout << "N_tilde[q_local] == q: " << N_tilde[q] << " == " << q << ", "
              << truth << std::endl;
#endif

    // Step 5: Edge direction
    //    B d = -Aq
#ifndef EXPLICIT_INVERSE
    // L U d = -Aq
    for (idx_t ii = 0; ii < m; ++ii) {
      d.get()[ii] = -A[q + ii * n];
    }
    dgetrs_(TRANS, &m_int, &NRHS, LU.get(), &m_int, IPIV.get(), d.get(), &LDB,
            &INFO);
#else
    // Use explicit matrix inverse:
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m, m, -1.0, Binv.get(), m, &A[q],
                n, 0.0, d.get(), 1);
#endif

#ifdef DEBUG
    std::cout << "d: [";
    print_matrix(1, m, d.get());
    std::cout << "]" << std::endl;
#endif

    // Step 6: Check for unboundedness / Leave the basis and step-length
    res.status = StatusJPSearch;
    double alpha = std::numeric_limits<double>::max(); // should be a better way
                                                       // to do this...
    double potential_alpha;
    idx_t jp = 0;       // this value not used unless entering basis found
    idx_t jp_local = 0; // this value not used unless jp found
    idx_t d_idx = 0;
#ifdef DEBUG
    std::cout << "alphas: [ ";
#endif
    for (auto const &basis_idx : B_tilde) {
      if (d.get()[d_idx] < 0) { // TODO: use tol instead of 0
        potential_alpha = -res.x.get()[basis_idx] / d.get()[d_idx];
#ifdef DEBUG
        std::cout << potential_alpha << ' ';
#endif
        // Bland's rule: if tie, choose smallest index
        if (potential_alpha < alpha) {
          alpha = potential_alpha;
          jp = basis_idx;
          jp_local = d_idx;
          res.status = StatusSuccess; // let res know we found a jp
        }
      }
      ++d_idx;
    }
#ifdef DEBUG
    std::cout << "]" << std::endl;
#endif
    if (res.status == StatusJPSearch) {
      std::cerr << "Solution is unbounded!" << std::endl;
      res.status = StatusUnbounded;
      return res;
    }
#ifdef DEBUG
    std::cout << "alpha: " << alpha << std::endl;
    std::cout << "jp/jp_local: " << jp << "/" << jp_local << std::endl;
    truth = B_tilde[jp_local] == jp ? "True" : "False";
    std::cout << "B_tilde[jp_local] == jp: " << B_tilde[jp_local]
              << " == " << jp << ", " << truth << std::endl;
#endif

    // Step 8: Update
    res.x.get()[q] = alpha;
    d_idx = 0;
    for (auto const &basis_idx : B_tilde) {
      res.x.get()[basis_idx] += alpha * d.get()[d_idx];
      ++d_idx;
    }
#ifdef DEBUG
    std::cout << "x: [ ";
    print_matrix(1, n, res.x.get());
    std::cout << "]" << std::endl;
    std::cout << "Updating N:" << std::endl;
#endif
    add_basis_col(m, mn, N, cN, n, A, c, q_local, jp);
    cB.get()[jp_local] = c[q];
    B_tilde[jp_local] = q;
    N_tilde[q_local] = jp;

#ifdef EXPLICIT_INVERSE
    // This is a rank-1 update to B, can use Sherman-Morrison formula to update
    // See:
    //     https://www.maths.ed.ac.uk/hall/RealSimplex/25_01_07_talk1.pdf
    //     http://www-personal.umich.edu/~mepelman/teaching/IOE610/Handouts/610SimpexIIIF13.pdf
    // B = B + (Aq - Ajp) @ epj.T
    // where epj is an m-vector with one at its pth element and zero everywhere
    // else. [thing]_ptr refers to either memory allocated using
    // std::make_unique or in WORK (depending on if WORK was large enough to
    // hold it).
    sherman_morrison_update(m, n, A, Binv, q, jp, jp_local, u_ptr, ejp_ptr,
                            qq_ptr, r_ptr);
#endif

    // And that's an iteration
    ++res.nit;
  }

  // Should never get here...
  return res;
}
