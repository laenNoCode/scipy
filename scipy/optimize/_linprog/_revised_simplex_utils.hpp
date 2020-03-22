#ifndef __REVISED_SIMPLEX_UTILS_HPP
#define __REVISED_SIMPLEX_UTILS_HPP

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

typedef std::size_t idx_t;
enum Status {
  StatusSuccess,
  StatusUnbounded,
  StatusLUFailure,
  StatusDGETRF_illegal_value,
  StatusInverseFailure,
  StatusDGETRI_illegal_value,
  StatusNotImplemented,
  StatusQNotFound,
  StatusJPSearch
};
std::map<Status, std::string> StatusMessages = {
    {StatusSuccess, "Optimal solution was found."},
    {StatusUnbounded, "Solution is unbounded."},
    {StatusLUFailure, "LU factorization found that U is exactly singular."},
    {StatusDGETRF_illegal_value, "dgetrf_ argument had an illegal value."},
    {StatusInverseFailure, "dgetri_ foud that U is exactly singular."},
    {StatusDGETRI_illegal_value, "dgetri_ argument had an illegal value."},
    {StatusNotImplemented, "LU rank-one update not implemented."},
    {StatusQNotFound, "Leaving basis index q was not found."},
    {StatusJPSearch, "Searching for entering basis, jp."}};

extern "C" {
// LU decomoposition of a general matrix
void dgetrf_(int *M, int *N, double *A, int *lda, int *IPIV, int *INFO);

// Solve linear system using LU decomposition
void dgetrs_(char *TRANS, int *N, int *NRHS, double *A, int *LDA, int *IPIV,
             double *B, int *LDB, int *INFO);

// generate inverse of a matrix given its LU decomposition
void dgetri_(int *N, double *A, int *lda, int *IPIV, double *WORK, int *lwork,
             int *INFO);
}

template <class T>
void print_matrix(const std::size_t m, const std::size_t n, const T *M) {
  for (idx_t ii = 0; ii < m; ++ii) {
    for (idx_t jj = 0; jj < n; ++jj) {
      std::cout << M[jj + ii * n] << ' ';
    }
  }
}

class LPResult {
public:
  std::size_t m;
  std::size_t n;
  std::unique_ptr<double[]> x;
  double fun;
  std::size_t nit;
  Status status;

  /// Cython nullary constuctor
  LPResult() : m(0), n(0), x(NULL), fun(0), nit(0), status(StatusSuccess) {}

  LPResult(const std::size_t m, const std::size_t n, const double *bfs)
      : fun(0), nit(0) {
    this->m = m;
    this->n = n;
    this->x = std::make_unique<double[]>(this->n);
    std::copy(bfs, bfs + n, this->x.get());
  }

  /// Cython access to contents of primal solution variables
  const double *get_x() { return this->x.get(); }

  void print() {
    // Assumes first m variables are x, next n - m variables are slack
    std::cout << "LPResult:" << std::endl;
    std::cout << "\tstatus: " << this->status << std::endl;
    std::cout << "\tmessage: " << StatusMessages[this->status] << std::endl;
    std::cout << "\tnit: " << this->nit << std::endl;
    std::cout << "\tx: [ ";
    print_matrix(1, this->m, this->x.get());
    std::cout << ']' << std::endl;
    std::cout << "\ts: [ ";
    print_matrix(1, this->n - this->m, &this->x.get()[this->m]);
    std::cout << ']' << std::endl;
    std::cout << "\tfun: " << this->fun << std::endl;
  }
};

std::vector<idx_t> initial_B_tilde(const idx_t *B_tilde0,
                                   const std::size_t nnz) {
  auto B_tilde = std::vector<idx_t>();
  for (std::size_t ii = 0; ii < nnz; ++ii) {
    B_tilde.push_back(B_tilde0[ii]);
  }
#ifdef DEBUG
  std::cout << "B_tilde: [" << ' ';
  for (auto const &el : B_tilde) {
    std::cout << el << ' ';
  }
  std::cout << ']' << std::endl;
#endif
  return B_tilde;
}

std::vector<idx_t> initial_N_tilde(const std::vector<idx_t> &B_tilde,
                                   const std::size_t n) {
  auto N_tilde = std::vector<idx_t>();
  for (idx_t jj = 0; jj < n; ++jj) {
    // TODO: optimize
    if (std::find(B_tilde.cbegin(), B_tilde.cend(), jj) == B_tilde.cend()) {
      N_tilde.push_back(jj);
    }
  }
#ifdef DEBUG
  std::cout << "N_tilde: [" << ' ';
  for (auto const &el : N_tilde) {
    std::cout << el << ' ';
  }
  std::cout << ']' << std::endl;
#endif
  return N_tilde;
}

void add_basis_col(const std::size_t m, const std::size_t n,
                   std::unique_ptr<double[]> &B, std::unique_ptr<double[]> &cB,
                   const std::size_t k, const double *A, const double *c,
                   const idx_t to_col_idx, const idx_t from_col_idx) {
#ifdef DEBUG
  std::cout << "Old col (" << to_col_idx << "), new col (" << from_col_idx
            << "):" << std::endl;
#endif
  for (idx_t ii = 0; ii < m; ++ii) {
#ifdef DEBUG
    std::cout << "\t" << B[to_col_idx + ii * n] << " -> "
              << A[from_col_idx + ii * k] << std::endl;
#endif
    B[to_col_idx + ii * n] = A[from_col_idx + ii * k];
  }
#ifdef DEBUG
  std::cout << "\tcB: " << cB[to_col_idx] << " -> " << c[from_col_idx]
            << std::endl;
#endif
  cB[to_col_idx] = c[from_col_idx];
}

idx_t choose_leaving(const std::vector<idx_t> &N_tilde,
                     const std::unique_ptr<double[]> &rN, LPResult &res) {
  auto leaving_idx = std::map<double, std::set<idx_t>>();
  idx_t inner_idx = 0;
  for (auto const &outer_idx : N_tilde) {
    if (rN.get()[inner_idx] < 0) {
      leaving_idx[rN.get()[inner_idx]].emplace(outer_idx);
    }
    ++inner_idx;
  }
  // Bland's rule: choose lowest index to enter if there are
  //               multiple min values
  // We are relying on the fact that map and set are sorted!
  if (leaving_idx.size()) {
    return *leaving_idx.cbegin()->second.cbegin();
  }
#ifdef DEBUG
  std::cout << "Did not find leaving_idx!" << std::endl;
#endif
  // Set status letting caller know that leaving_idx was not found
  res.status = StatusQNotFound;
  return 0; // this value is never used, doesn't mean anything!
}

/// Take ``from`` at indices ``idx``  and pack them into ``to``.
void take(const double *from, const std::vector<idx_t> &idx,
          std::unique_ptr<double[]> &to) {
  idx_t to_idx = 0;
  for (auto const &from_idx : idx) {
    to.get()[to_idx] = from[from_idx];
    ++to_idx;
  }
}

/// Copy ``col`` in ``from`` into ``to``, both with ``m`` rows.
/**

``from`` : ``m`` by ``n`` matrix
``to`` : ``m`` by ``k`` matrix
``col`` : column to copy
  */
void copy_col(const std::size_t &m, const std::size_t &n, const double *from,
              const idx_t &from_col, const std::size_t &k,
              std::unique_ptr<double[]> &to, const idx_t &to_col) {
  for (idx_t ii = 0; ii < m; ++ii) {
    to.get()[to_col + ii * k] = from[from_col + ii * n];
  }
}

/// Copy ``cols`` in ``from`` into ``to``, both with ``m`` rows.
void take_cols(const std::size_t &m, const std::size_t &n, const double *from,
               const std::vector<idx_t> &cols, const std::size_t &k,
               std::unique_ptr<double[]> &to) {
  std::size_t to_col = 0;
  for (auto const &from_col : cols) {
    copy_col(m, n, from, from_col, k, to, to_col);
    ++to_col;
  }
}

/// Compute LU factorization of matrix B.
Status lu(const std::size_t &m, const std::size_t &n,
          const std::vector<idx_t> &idx, const double *A, int &m_int,
          std::unique_ptr<double[]> &LU, std::unique_ptr<int[]> &IPIV,
          int &INFO) {

  // Copy over Basis Columns
  idx_t col_ctr;
  for (idx_t ii = 0; ii < m; ++ii) {
    col_ctr = 0;
    for (auto const &basis_col : idx) {
      LU.get()[col_ctr + ii * m] = A[basis_col + ii * n];
      ++col_ctr;
    }
  }

  // Do factorization:
  dgetrf_(&m_int, &m_int, LU.get(), &m_int, IPIV.get(), &INFO);

  // Error handling:
  if (INFO > 0) {
    std::cerr << "U[" << INFO << ", " << INFO << "] is exactly zero!"
              << std::endl;
    return StatusLUFailure;
  } else if (INFO < 0) {
    std::cerr << "LU factorization: argument " << -1 * INFO
              << " of dgetrf_ had an illegal value!" << std::endl;
    return StatusDGETRF_illegal_value;
  }

  return StatusSuccess;
}

/// Compute inverse of matrix.
Status inverse(const std::size_t &m, const std::size_t &n,
               const std::vector<idx_t> &idx, const double *A, int &m_int,
               std::unique_ptr<double[]> &Binv, std::unique_ptr<int[]> &IPIV,
               std::unique_ptr<double[]> &WORK, int &LWORK, int &INFO) {

#ifdef DEBUG
  std::cout << "Computing explicit inverse!" << std::endl;
#endif

  // Get LU factorization
  Status status = lu(m, n, idx, A, m_int, Binv, IPIV, INFO);
  if (status != StatusSuccess) {
    return status;
  }

  // Now compute the inverse
  dgetri_(&m_int, Binv.get(), &m_int, IPIV.get(), WORK.get(), &LWORK, &INFO);

  // Error handling
  if (INFO > 0) {
    std::cerr << "U[" << INFO << ", " << INFO << "] is exactly zero!"
              << std::endl;
    return StatusInverseFailure;
  } else if (INFO < 0) {
    std::cerr << "DGETRI: argument " << -1 * INFO << " had an illegal value!"
              << std::endl;
    return StatusDGETRI_illegal_value;
  }

  return StatusSuccess;
}

/// Sherman-Morrison formula to do inverse update after rank-1 update
void sherman_morrison_update(const std::size_t &m, const std::size_t &n,
                             const double *A, std::unique_ptr<double[]> &Binv,
                             const idx_t &q, const idx_t &jp,
                             const idx_t &jp_local, double *u_ptr,
                             double *ejp_ptr, double *qq_ptr, double *r_ptr) {

#ifdef DEBUG
  std::cout << "Doing Sherman-Morrison update!" << std::endl;
#endif

  // TODO: Probably could be done more efficiently because epj is a unit
  // vector.

  // Compute Aq - Ajp
  for (idx_t ii = 0; ii < m; ++ii) {
    u_ptr[ii] = A[q + ii * n] - A[jp + ii * n];
  }

  // ejp should be all zeros except when being used
  ejp_ptr[jp_local] = 1;

  // Compute qq = B^-1 @ u
  cblas_dgemv(CblasRowMajor, CblasNoTrans, m, m, 1.0, Binv.get(), m, u_ptr, 1,
              0.0, qq_ptr, 1);

  // Compute r.T = v.T @ B^-1
  cblas_dgemv(CblasRowMajor, CblasTrans, m, m, 1.0, Binv.get(), m, ejp_ptr, 1,
              0.0, r_ptr, 1);

  // Do the update: B^-1 = B^-1 - (qq @ r.T)/den
  double den = 1 + cblas_ddot(m, ejp_ptr, 1, qq_ptr, 1);
  cblas_dger(CblasRowMajor, m, m, -1 / den, qq_ptr, 1, r_ptr, 1, Binv.get(), m);

  // Reset ejp to all zeros
  ejp_ptr[jp_local] = 0;
}

#endif
