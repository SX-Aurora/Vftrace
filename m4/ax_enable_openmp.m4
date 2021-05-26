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
#   This macro activates OpenMP suport in Vftrace.
#   Currently, this only means that for each Thread ID > 0,
#   we exit the function hook. This has merely the purpose of
#   preventing race conditions.
#
#
AC_DEFUN([AX_ENABLE_OPENMP], [
   AC_PREREQ(2.50)
   AC_ARG_ENABLE(
      [openmp],
      [AS_HELP_STRING([--enable-openmp], [enable OpenMP bounce [default=no]])],
      [],
      [enable_openmp="no"])
   AM_CONDITIONAL([ENABLE_OPENMP], [test "$enable_openmp" = "yes"])
])
