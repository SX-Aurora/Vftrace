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

   AC_ARG_ENABLE(
      [fortran08],
      [AS_HELP_STRING([--enable-fortran08], [enable fortran08 interfaces for vftrace [default=yes]])],
      [],
      [enable_fortran08="yes"])
   # disable f08 as well if f is disabled
   AM_COND_IF(
      [ENABLE_FORTRAN],
      [AM_CONDITIONAL([ENABLE_FORTRAN08], [test "$enable_fortran08" = "yes"])],
      [AC_MSG_WARN([Disableing Fortran08 because Fortran is disabled])
       AM_CONDITIONAL([ENABLE_FORTRAN08], [test "$enable_fortran" = "yes"])]
   )
])
