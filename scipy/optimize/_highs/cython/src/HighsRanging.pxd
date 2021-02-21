# distutils: language=c++
# cython: language_level=3

from libcpp.vector cimport vector

cdef extern from "HighsRanging.h":
    # From HiGHS/src/lp_data/HighsRanging.h

    cdef cppclass HighsRangingRecord:
        vector[double] value_
        vector[double] objective_
        vector[int] in_var_
        vector[int] ou_var_

    cdef cppclass HighsRanging:
          HighsRangingRecord col_cost_up
          HighsRangingRecord col_cost_dn
          HighsRangingRecord col_bound_up
          HighsRangingRecord col_bound_dn
          HighsRangingRecord row_bound_up
          HighsRangingRecord row_bound_dn
