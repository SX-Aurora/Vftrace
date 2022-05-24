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
   AM_CONDITIONAL([ENABLE_FORTRAN], [test "$enable_fortran" = "yes"])
   AC_MSG_RESULT([$enable_fortran])

   AC_ARG_ENABLE(
      [fortran08],
      [AS_HELP_STRING([--enable-fortran08], [enable fortran08 interfaces for vftrace [default=yes]])],
      [enable_fortran08_present="yes"],
      [enable_fortran08_present="no"])
   AC_MSG_CHECKING([whether Fortran08 is enabled])
   # if the option is not given, resort to default (yes)
   AS_IF([test "x$enable_fortran08_present" = "xno"], [enable_fortran08="yes"])
   # disable f08 as well if f is disabled
   AM_COND_IF(
      [ENABLE_FORTRAN],
      [],
      [enable_fortran08="no"])
   AM_CONDITIONAL([ENABLE_FORTRAN08], [test "$enable_fortran08" = "yes"])
   AC_MSG_RESULT([$enable_fortran])
])
