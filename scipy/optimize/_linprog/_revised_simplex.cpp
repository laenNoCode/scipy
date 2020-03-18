#include <vector>
#include <utility>
#include <limits>
#include <memory>
#include <cblas.h>
#include <iostream>

#include "_revised_simplex_utils.hpp"

namespace linprog {

    template<class T>
    void revised_simplex(
        const std::size_t m,
        const std::size_t n,
        const T * c,
        const T * A,
        const T * b,
        const T & eps1,
        const T * bfs,
        T * solution) {

        // We know how big B_indices, V_indices will be, so we can
        // allocate now and swap around later; unique_ptrs should
        // have no overhead if you're not constantly
        // constructing/destructing
        auto V_size = abs(n - m);
        auto B_indices = std::make_unique<std::size_t[]>(m);
        auto V_indices = std::make_unique<std::size_t[]>(V_size);

        // Find current basis indices (where) and nonbasis indicies
        // (wheren't) from the basic feasible solution
        linprog::argwhere_and_wherenot(
                n, bfs, B_indices, V_indices);

        // Pre-initialize c_tilde
        auto c_tilde_V = std::make_unique<T[]>(V_size); // n - m zeros
        auto c_tilde_intermediate = std::make_unique<T[]>(m*V_size);
        std::size_t j; // index of min value of c_tilde
        T cj; // min value of c_tilde

        // Pre-initialize A[:, B_indices], A[:, V_indices], Binv, d,
        // cB, cV, w;  All are initialized to 0
        auto AB = std::make_unique<T[]>(m*m);
        auto AV = std::make_unique<T[]>(m*V_size);
        auto Binv = std::make_unique<T[]>(m*m);
        auto d = std::make_unique<T[]>(m);
        auto cB = std::make_unique<T[]>(m);
        auto w = std::make_unique<T[]>(m);

        // Pack in initial values
        for (std::size_t ii = 0; ii < m; ++ii) {
            for (std::size_t jj = 0; jj < m; ++jj) {
                AB[ii + jj*m] = A[ii + B_indices[jj]*m];

                // Initialize Binv with AB
                Binv[ii + jj*m] = A[ii + B_indices[jj]*m];
            }
            for (std::size_t jj = 0; jj < V_size; ++jj) {
                AV[ii + jj*m] = A[ii + V_indices[jj]*m];
            }
            cB[ii] = c[B_indices[ii]];
        }
        for (std::size_t ii = 0; ii < V_size; ++ii) {
            c_tilde_V[ii] = c[V_indices[ii]];
        }

        // Initialize outside loop so we don't do it every iteration:
        // not going to be a big deal, but we're looking for speed
        std::size_t k;
        std::size_t min_idx;
        T min_val;
        T val0;

        // Simplex method loops continuously until solution is found
        // or discovered to be impossible.
        std::size_t iters = 0;
        while(true) {
            ++iters;

            //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            // Step 1
            // compute B^-1

            // Binv    inverse of the basis (directly computed)

            // Binv = np.linalg.pinv(A[:, B_indices])

            // Binv always needs to start out with what AB has in it!
            // TODO: LU factorization
            linprog::inverse(m, Binv);

            //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            // Step 2
            // compute d = B^-1 * b

            // d    current solution vector
            // d = Binv @ b
            cblas_dgemv(
                CblasColMajor, CblasNoTrans, m, m, 1.0, Binv.get(), m,
                b, 1, 0.0, d.get(), 1);

            //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            // Step 3/Step 4/Step 5
            // compute c_tilde = c_V - c_B * B^-1 * V

            // c_tilde     modified cost vector

            // c_tilde[V_indices] = c[V_indices] - (
            //     c[B_indices] @ Binv @ A[:, V_indices])
            // c_tilde_V = cV - cB @ Binv @ AV
            // c_tilde_V needs to always have what cV has in it
            cblas_dgemm(
                    CblasColMajor, CblasNoTrans, CblasNoTrans, m,
                    V_size, m, 1.0, Binv.get(), m, AV.get(), m, 0.0,
                    c_tilde_intermediate.get(), m);
            cblas_dgemv(
                CblasColMajor, CblasTrans, m, V_size, -1.0,
                c_tilde_intermediate.get(), m, cB.get(), 1, 1.0,
                c_tilde_V.get(), 1);

            //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            // Step 6
            // compute j s.t. c_tilde[j] <= c_tilde[k] for all k in
            // V_indices
            // cj minimum cost value (negative) of non-basic columns
            // j column in A corresponding to minimum cost value

            // nonzero values can only be in V_indices, so only check
            // there
            // TODO: we're doing more work here than we need to; we
            //       could simply keep track of the min value and its
            //       location and update when B and V are updated
            linprog::argmin_and_element(V_size, c_tilde_V, j, cj);
            // Map local index in c_tilde_V to the correct index in
            // c_tilde:
            j = V_indices[j];

            //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            // Step 7
            // if cj >= 0 , then we're done -- return solution which
            // is optimal

            if (cj >= -eps1) {
                // zero out nonbasis solutions (caller may have had
                // something in here)
                for (std::size_t ii = 0; ii < V_size; ++ii) {
                    solution[V_indices[ii]] = 0;
                }
                // Assign the basis values
                for (std::size_t ii = 0; ii < m; ++ii) {
                    solution[B_indices[ii]] = d[ii];
                }
                // TODO: more informative exit
                // return OptimizeResult({
                //     'x': solution,
                //     'nit': iters,
                // })
                std::cout << "nit: " << iters << std::endl;
                return;
            }

            //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            // Step 8
            // compute w = B^-1 * a[j]

            // w: relative weight (vector) of cols entering the basis
            // w = Binv @ Aj
            cblas_dgemv(
                CblasColMajor, CblasNoTrans, m, m, 1.0, Binv.get(), m,
                A + j*m, 1, 0.0, w.get(), 1);


            //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            // Step 9
            // compute i s.t. w[i]>0 and d[i]/w[i] is a smallest
            // positive ratio
            // swap column j into basis and swap column i out of basis
            min_val = std::numeric_limits<T>::max();
            for (std::size_t ii = 0; ii < m; ++ii) {
                if (w[ii] > eps1) {
                    val0 = d[ii]/w[ii];
                    if (val0 < min_val) {
                        min_idx = ii;
                        min_val = val0;
                    }
                }
            }

            if (min_val == std::numeric_limits<T>::max()) {
                // raise ValueError("System is unbounded.");
                std::cout << "System is unbounded." << std::endl;
                return;
            }

            // k is outgoing (into V)
            // j in incoming (into B)
            // Here's what the python does:
            // k = B_indices[i];
            // B_indices[i] = j;
            // V_indices[j == V_indices] = k

            // We have a little more to do because CBLAS does things
            // inplace, so we need to repopulate some variables for
            // the next time around the horn....

            // Do swap and updates concurrently, taking advantage of
            // any looping that's happening
            k = B_indices[min_idx];
            B_indices[min_idx] = j;
            cB[min_idx] = c[j];
            for (std::size_t jj = 0; jj < V_size; ++jj) {
                if (j == V_indices[jj]) {
                    V_indices[j] = k;
                    for (std::size_t ii = 0; ii < m; ++ii) {
                        AB[ii + min_idx*m] = A[ii + j*m];
                        AV[ii + jj*m] = A[ii + k*m];

                        // Binv needs to have AV completely restored
                        // TODO: Can we just update Binv?
                        for (std::size_t kk = 0; kk < m; ++kk) {
                            Binv[ii + kk*m] = A[ii + B_indices[kk]*m];
                        }
                    }
                    break;
                }
            }
            // c_tilde_V needs cV restored
            for (std::size_t jj = 0; jj < V_size; ++jj) {
                c_tilde_V[jj] = c[V_indices[jj]];
            }

            //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            // Step 10
            // REPEAT

        }
    }
}

int main() {

    // double eps1 = 10e-5;
    // std::size_t m = 2;
    // std::size_t n = 5;
    // double c[] {-2, -3, -4, 0, 0};
    // // double A[] {
    // //     3, 2, 1, 1, 0,
    // //     2, 5, 3, 0, 1
    // // };
    // double A[] { // column major (b/c Fortran...)
    //     3, 2,
    //     2, 5,
    //     1, 3,
    //     1, 0,
    //     0, 1
    // };
    // double b[] {10, 15};
    // double bfs[] {0, 0, 0, 10, 15};
    // double solution[] {1, 2, 3, 4, 5};

    double eps1 = 10e-5;
    std::size_t m = 3;
    std::size_t n = 8;
    double c[] {0, 1, 1, 1, -2, 0, 0, 0};
    double A[] {
        3, 1, -3,
        1, 1, 0,
        0, 1, 2,
        0, 1, 1,
        -1, 0, 5,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    };
    double b[] {1, 2, 6};
    double bfs[] {0, 0, 0, 0, 0, 1, 2, 6};
    double solution[] = {0, 0, 0, 0, 0, 0, 0, 0};

    linprog::revised_simplex(m, n, c, A, b, eps1, bfs, solution);

    std::cout << "Solution:" << std::endl;
    for (std::size_t ii = 0; ii < n; ++ii) {
        std::cout << solution[ii] << " ";
    }
    std::cout << std::endl;

    return EXIT_SUCCESS;
}
