# ===========================================================================
#      https://www.gnu.org/software/autoconf-archive/ax_prog_cc_mpi.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_CHECK_OMP
#
# DESCRIPTION
#
#   This macro checks if OMP support is wanted and available
#

AC_DEFUN([AX_CHECK_OMP], [
   AC_PREREQ(2.50)
   # Use build in OMP checking macro
   AC_OPENMP
   # check if the omp tools header is available
   AM_COND_IF(
      [WITH_OMP],
      [AC_LANG(C)
       AC_CHECK_HEADER([omp-tools.h],
          [AC_MSG_FAILURE([unable to find omp-tools.h header! Ensure that your compiler supports OpenMP 5.0, or disable OpenMP support])])])
])
