# ===========================================================================
#      https://www.gnu.org/software/autoconf-archive/ax_prog_cc_mpi.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_ENABLE_FORTRAN
#
# DESCRIPTION
#
#   This macro check for enabled fortran support
#

AC_DEFUN([AX_ENABLE_FORTRAN], [
   AC_PREREQ(2.50)
   AC_ARG_ENABLE(
      [fortran],
      [AS_HELP_STRING([--enable-fortran], [enable fortran interfaces for vftrace [default=yes]])],
      [],
      [enable_fortran="yes"])
   AM_CONDITIONAL([ENABLE_FORTRAN], [test "$enable_fortran" = "yes"])
])
