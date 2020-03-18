#include <vector>
#include <utility>
#include <limits>
#include <memory>
#include <cblas.h>
#include <iostream>

extern "C" {
    // LU decomoposition of a general matrix
    void dgetrf_(int * M, int * N, double * A, int * lda, int * IPIV, int * INFO);

    // generate inverse of a matrix given its LU decomposition
    void dgetri_(int * N, double * A, int * lda, int * IPIV, double * WORK, int * lwork, int * INFO);
}

// TODO: Better error handling!
template<class T>
void inverse_inplace(
        const std::size_t & m,
        std::unique_ptr<T[]> & A)
{

    // Possible overflow, unlikely though
    auto N = static_cast<int>(m);

    // Do LAPACK stuff
    int *IPIV = new int[N];
    int LWORK = N*N;
    double *WORK = new double[LWORK];
    int INFO;

    dgetrf_(& N, & N, A.get(), & N, IPIV, & INFO);

    dgetri_(& N, A.get(), & N, IPIV, WORK, & LWORK, & INFO);

    delete [] IPIV;
    delete [] WORK;
}


template<class T>
void set_to_zero_inplace(const std::size_t & n, T * arr) {
    for(std::size_t ii = 0; ii < n; ++ii) {
        arr[ii] = 0;
    }
}

template<class T>
void set_to_zero_at_indices_inplace(
        std::unique_ptr<T> & arr,
        const std::size_t valid_idx_size,
        const std::unique_ptr<T> & valid_idx) {

    for (std::size_t ii = 0; ii < valid_idx_size; ++ii) {
        arr[valid_idx[ii]] = 0;
    }
}

template<class T>
void argmin_and_element_inplace(
        const std::size_t & n,
        const std::unique_ptr<T[]> & arr,
        std::size_t & out_idx,
        T & out_val) {

    out_idx = 0;
    out_val = arr[out_idx];
    for (std::size_t ii = 1; ii < n; ++ii) {
        if (arr[ii] < out_val) {
            out_idx = ii;
            out_val = arr[out_idx];
        }
    }
}

template<class T>
void argmin_and_element_at_indices_inplace(
        const std::size_t & n,
        const std::unique_ptr<T[]> & arr,
        const std::size_t & valid_idx_size,
        const std::unique_ptr<std::size_t[]> & valid_idx,
        std::size_t & out_idx,
        T & val) {

    // Don't assume value already in ``val``;
    out_idx = valid_idx[0];
    val = arr[out_idx];
    for (std::size_t ii = 1; ii < valid_idx_size; ++ii) {
        if (arr[valid_idx[ii]] < val) {
            out_idx = valid_idx[ii];
            val = arr[out_idx];
        }
    }
}

template<class T>
void argwhere_and_wherenot_inplace(
        const std::size_t & n,
        const T * arr,
        std::unique_ptr<std::size_t[]> & idx,
        std::unique_ptr<std::size_t[]> & not_idx) {

    // trusting that idx/not_idx are large enough!
    std::size_t idx_place = 0;
    std::size_t not_idx_place = 0;

    for (std::size_t ii = 0; ii < n; ++ii) {
        if (arr[ii] > 0) {
            idx[idx_place] = ii;
            ++idx_place;
        }
        else {
            not_idx[not_idx_place] = ii;
            ++not_idx_place;
        }
    }
}

template<class T>
std::vector<std::size_t> argwhere(const std::size_t & n, const T * arr) {
    auto idx = std::vector<std::size_t>();
    for (std::size_t ii = 0; ii < n; ++ii) {
        if (arr[ii] > 0) {
            idx.push_back(ii);
        }
    }
    return idx;
}

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
    // allocate now and swap around later;
    // unique_ptrs should have no overhead if you're not constantly
    // constructing/destructing
    auto B_size = m;
    auto V_size = n - m;
    auto B_indices = std::make_unique<std::size_t[]>(B_size);
    auto V_indices = std::make_unique<std::size_t[]>(V_size);
    argwhere_and_wherenot_inplace(n, bfs, B_indices, V_indices);

    // Simplex method loops continuously until solution is found or
    // discovered to be impossible.

    // Pre-initialize c_tilde
    auto c_tilde_V = std::make_unique<T[]>(V_size); // n - m zeros
    auto c_tilde_intermediate = std::make_unique<T[]>(m*V_size);
    std::size_t j; // index of min value of c_tilde
    T cj; // min value of c_tilde

    // Pre-initialize A[:, B_indices], A[:, V_indices], Binv, d, cB, cV, Aj, w;
    // All are initialized to 0 (I believe...)
    auto AB = std::make_unique<T[]>(m*B_size);
    auto AV = std::make_unique<T[]>(m*V_size);
    auto Binv = std::make_unique<T[]>(m*m);
    auto d = std::make_unique<T[]>(m);
    auto cB = std::make_unique<T[]>(B_size);
    auto Aj = std::make_unique<T[]>(m); // single column of A -- TODO: may be able to get away without this?
    auto w = std::make_unique<T[]>(m);

    // Pack AB, AV and Binv
    for (std::size_t ii = 0; ii < m; ++ii) {
        for (std::size_t jj = 0; jj < B_size; ++jj) {
            AB[ii + jj*m] = A[ii + B_indices[jj]*m];
            Binv[ii + jj*m] = A[ii + B_indices[jj]*m];
        }
        for (std::size_t jj = 0; jj < V_size; ++jj) {
            AV[ii + jj*m] = A[ii + V_indices[jj]*m];
        }
    }

    // TODO: can we merge upper and bottom loops?

    // Pack cB, cV
    for (std::size_t ii = 0; ii < B_size; ++ii) {
        cB[ii] = c[B_indices[ii]];
    }
    for (std::size_t ii = 0; ii < V_size; ++ii) {
        c_tilde_V[ii] = c[V_indices[ii]];
    }

    // TODO: Might be able to delay this and only zero out indices
    //       not in the solution basis!
    // Set solution to zero initially
    set_to_zero_inplace(n, solution);

    // Initialize outside loop so we don't do it every iteration: used at end
    std::size_t min_idx;
    T min_val;
    T val0;

    // Start your engines!
    std::size_t iters = 0;
    while(true) {
        ++iters;

        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        // Step 1
        // compute B^-1

        // Binv    inverse of the basis (directly computed)

        // Binv = np.linalg.pinv(A[:, B_indices])

        // Binv always needs to start out with what AB has in it!
        // TODO: LU factorization
        inverse_inplace(m, Binv);

        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        // Step 2
        // compute d = B^-1 * b

        // d    current solution vector
        // d = Binv @ b
        cblas_dgemv(
            CblasColMajor, CblasNoTrans, m, m, 1.0, Binv.get(), m, b,
            1, 0.0, d.get(), 1);

        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        // Step 3/Step 4/Step 5
        // compute c_tilde = c_V - c_B * B^-1 * V

        // c_tilde     modified cost vector

        // c_tilde[V_indices] = c[V_indices] - c[B_indices] @ Binv @ A[:, V_indices]
        // c_tilde_V = cV - cB @ Binv @ AV
        // c_tilde_V needs to always have what cV has in it
        cblas_dgemm(
                CblasColMajor, CblasNoTrans, CblasNoTrans, m, V_size,
                m, 1.0, Binv.get(), m, AV.get(), m, 0.0,
                c_tilde_intermediate.get(), m);
        cblas_dgemv(
            CblasColMajor, CblasTrans, m, V_size, -1.0,
            c_tilde_intermediate.get(), m, cB.get(), 1, 1.0,
            c_tilde_V.get(), 1);

        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        // Step 6
        // compute j s.t. c_tilde[j] <= c_tilde[k] for all k in V_indices
        // cj minimum cost value (negative) of non-basic columns
        // j column in A corresponding to minimum cost value

        // j = np.argmin(c_tilde)
        // cj = c_tilde[j]
        // nonzero values can only be in V_indices, so only check there
        // argmin_and_element_at_indices_inplace(n, c_tilde, V_size, V_indices, j, cj);
        argmin_and_element_inplace(V_size, c_tilde_V, j, cj);
        // Map local index in c_tilde_V to the correct index in c_tilde:
        j = V_indices[j];

        // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        // Step 7
        // if cj >= 0 , then we're done -- return solution which is optimal

        if (cj >= -eps1) {
            // solution = np.zeros(n)
            for (std::size_t ii = 0; ii < B_size; ++ii) {
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

        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        // Step 8
        // compute w = B^-1 * a[j]

        // w    relative weight (vector) of column entering the basis

        // Populate Aj: TODO: might not need explicitly? Could use LDA arg in blas?
        for (std::size_t ii = 0; ii < m; ++ii) {
            Aj[ii] = A[ii + j*m];
        }
        // w = Binv @ Aj
        cblas_dgemv(
            CblasColMajor, CblasNoTrans, m, m, 1.0, Binv.get(), m,
            Aj.get(), 1, 0.0, w.get(), 1);

        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        // Step 9
        // compute i s.t. w[i]>0 and d[i]/w[i] is a smallest positive ratio
        // swap column j into basis and swap column i out of basis

        // mn		minimum of d[i]/w[i] when w[i] > 0
        // i		row corresponding to mn -- determines outgoing column
        // k		temporary storage variable

        // zz = np.argwhere(w > eps1)
        // ii = np.argmin(d[zz]/w[zz])
        // i = zz[ii];
        min_idx = -1;
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

        if (!min_idx) { // i == 0
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
        // inplace, so we need to repopulate some variables for the
        // next time around the horn....

        // NEED TO UPDATE ALL AV, AB and Binv together, and everything else!
        // Do swap and updates concurrently, taking advantage of any looping that's happening
        auto k = B_indices[min_idx];
        B_indices[min_idx] = j;
        cB[min_idx] = c[j];
        for (std::size_t jj = 0; jj < V_size; ++jj) {
            if (j == V_indices[jj]) {
                V_indices[jj] = k;
                c_tilde_V[jj] = c[k];
                for (std::size_t ii = 0; ii < m; ++ii) {
                    AB[ii + min_idx*m] = A[ii + B_indices[min_idx]*m];
                    Binv[ii + min_idx*m] = A[ii + B_indices[min_idx]*m];
                    AV[ii + jj*m] = A[ii + V_indices[jj]*m];
                }
                break;
            }
        }

        // //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        // // Step 10
        // // REPEAT

    }

}

int main() {

    double eps1 = 10e-5;
    std::size_t m = 2;
    std::size_t n = 5;
    double c[] {-2, -3, -4, 0, 0};
    // double A[] {
    //     3, 2, 1, 1, 0,
    //     2, 5, 3, 0, 1
    // };
    double A[] { // column major (b/c Fortran...)
        3, 2,
        2, 5,
        1, 3,
        1, 0,
        0, 1
    };
    double b[] {10, 15};
    double bfs[] {0, 0, 0, 10, 15};
    double solution[] {0, 0, 0, 0, 0};

    revised_simplex(m, n, c, A, b, eps1, bfs, solution);

    std::cout << "Solution:" << std::endl;
    for (std::size_t ii = 0; ii < n; ++ii) {
        std::cout << solution[ii] << " ";
    }
    std::cout << std::endl;

    return EXIT_SUCCESS;
}
