#include <memory>

extern "C" {
    // LU decomoposition of a general matrix
    void dgetrf_(int * M, int * N, double * A, int * lda, int * IPIV, int * INFO);

    // generate inverse of a matrix given its LU decomposition
    void dgetri_(int * N, double * A, int * lda, int * IPIV, double * WORK, int * lwork, int * INFO);
}

namespace linprog {

    // TODO: Better error handling!
    template<class T>
    void inverse(
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

        dgetrf_(&N, &N, A.get(), &N, IPIV, &INFO);

        dgetri_(&N, A.get(), &N, IPIV, WORK, &LWORK, &INFO);

        delete [] IPIV;
        delete [] WORK;
    }

    template<class T>
    void argmin_and_element(
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
    void argmin_and_element_at_indices(
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
    void argwhere_and_wherenot(
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

}
