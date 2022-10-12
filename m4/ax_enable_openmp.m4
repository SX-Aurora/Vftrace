# ===========================================================================
#      https://www.gnu.org/software/autoconf-archive/ax_prog_cc_mpi.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_ENABLE_OPENMP
#
# DESCRIPTION
#
#   This macro checks if OPENMP support is wanted and available
#
# Flags to try:  -fopenmp (gcc, ncc)
#                -qopenmp (icc>=15)

AC_DEFUN([AX_ENABLE_OPENMP], [
   AC_PREREQ(2.50)
   AC_ARG_ENABLE([openmp],
      [AS_HELP_STRING([--enable-openmp], [enable OpenMP profiling layer (currently disabled). [default=no]])],
      #[AS_HELP_STRING([--enable-openmp], [enable OpenMP profiling layer. [default=no]])],
      [enable_openmp_present="yes"],
      [enable_openmp_present="no"])
   AC_MSG_CHECKING([whether OpenMP is enabled])
   # if the option is not given, resort to default (no)
   AS_IF([test "x$enable_openmp_present" = "xno"], [enable_openmp="no"])
# OMP is disabled for now.
# So far no OMP implementation exists with a usable ompt_callback system.
# A few issues were posted to llvm github, but not answered, nor fixed yet.
enable_openmp="no"
   AM_CONDITIONAL([ENABLE_OMP], [test "x$enable_openmp" = "xyes"])
   AC_MSG_RESULT([$enable_openmp])


   # check which compile flag is accepted by the C-compiler
   AM_COND_IF(
      [ENABLE_OMP],
      [AC_LANG(C)
       AX_CHECK_COMPILE_FLAG([-fopenmp],
          [AX_APPEND_FLAG([-fopenmp])],
          [AX_CHECK_COMPILE_FLAG([-qopenmp],
             [AX_APPEND_FLAG([-qopenmp])],
             [AC_MSG_ERROR([Could not determine OpenMP CFLAG!])
          ])
       ])
   ])

   # check which compile flag is accepted by the F-compiler
   AM_COND_IF(
      [ENABLE_OMP],
      [AC_LANG(Fortran)
       AX_CHECK_COMPILE_FLAG([-fopenmp],
          [AX_APPEND_FLAG([-fopenmp])],
          [AX_CHECK_COMPILE_FLAG([-qopenmp],
             [AX_APPEND_FLAG([-qopenmp])],
             [AC_MSG_ERROR([Could not determine OpenMP Fortran-FLAG!])
          ])
       ])
   ])

   # check if the omp headers are available
   AM_COND_IF(
      [ENABLE_OMP],
      [AC_LANG(C)
       AC_CHECK_HEADER([omp.h],
          [ ],
          [AC_MSG_FAILURE([unable to find omp.h header!])])])

   AM_COND_IF(
      [ENABLE_OMP],
      [AC_LANG(C)
       AC_CHECK_HEADER([omp-tools.h],
          [ ],
          [AC_MSG_FAILURE([unable to find omp-tools.h header! Ensure that your compiler supports OpenMP 5.0, or disable OpenMP support])])])

   # check for omp library functions
   AM_COND_IF(
      [ENABLE_OMP],
      [AC_LANG(C)
       AC_CHECK_FUNC([omp_get_thread_num],
          [],
          [AC_MSG_FAILURE([Unable to find OpenMP functions!])])])
])
