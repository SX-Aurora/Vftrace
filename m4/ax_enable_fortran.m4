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
      [enable_fortran_present="yes"],
      [enable_fortran_present="no"])
   AC_MSG_CHECKING([whether Fortran is enabled])
   # if the option is not given, resort to default (yes)
   AS_IF([test "x$enable_fortran_present" = "xno"], [enable_fortran="yes"])
   AM_CONDITIONAL([ENABLE_FORTRAN], [test "x$enable_fortran" = "xyes"])
   AC_MSG_RESULT([$enable_fortran])
])
