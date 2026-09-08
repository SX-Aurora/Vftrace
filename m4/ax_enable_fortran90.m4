# ===========================================================================
#      https://www.gnu.org/software/autoconf-archive/ax_prog_cc_mpi.html
# ===========================================================================
#
# SYNOPSIS
#
#   AX_ENABLE_FORTRAN90
#
# DESCRIPTION
#
#   This macro checks if MPI is built with Fortran90 support
#

AC_DEFUN([AX_ENABLE_FORTRAN90], [
   AC_PREREQ(2.50)
   AC_ARG_ENABLE(
      [fortran90],
      [AS_HELP_STRING([--enable-fortran90], [enable Fortran90 interfaces for Vftrace [default=yes]])],
      [enable_fortran90_present="yes"],
      [enable_fortran90_present="no"])
   AC_MSG_CHECKING([whether Fortran90 is enabled])
   # if the option is not given, resort to default (yes)
   AS_IF([test "x$enable_fortran90_present" = "xno"], [enable_fortran90="yes"])
   AM_CONDITIONAL([ENABLE_FORTRAN90], [test "x$enable_fortran90" = "xyes"])
   AC_MSG_RESULT([$enable_fortran90])
])
