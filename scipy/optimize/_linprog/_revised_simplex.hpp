#ifndef __REVISED_SIMPLEX_HPP
#define __REVISED_SIMPLEX_HPP

#include <iostream>
#include "_revised_simplex.hpp"

LPResult revised_simplex(const std::size_t m, const std::size_t n,
                         const double *c, const double *A, const double *b,
                         const double *bfs, const idx_t *B_tilde0);
#endif
