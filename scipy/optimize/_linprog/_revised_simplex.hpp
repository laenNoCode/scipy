#ifndef __REVISED_SIMPLEX_HPP
#define __REVISED_SIMPLEX_HPP

#include <memory>

namespace linprog {

    template<class T>
    class RevisedSimplexResult {
    public:
        std::unique_ptr<T[]> x; // the solution
        std::size_t nit; // how many iterations
        T fun; // the function value

        // Nullary constructor for Cython to be happy
        RevisedSimplexResult() { }

        // This is the constructor you probably want to use
        RevisedSimplexResult(const std::size_t & n)
            : nit(0),
              fun(0) {
            this->x = std::make_unique<T[]>(n);
        }

        // Helper function for Cython to get at contents of unique_ptr
        T * get_x() {
            return this->x.get();
        }
    };

    template<class T>
    RevisedSimplexResult<T> revised_simplex(
        const std::size_t m,
        const std::size_t n,
        const T * c,
        const T * A,
        const T * b,
        const T & eps1,
        const T * bfs);
}

#endif
